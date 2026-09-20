"""DST training: trust-weighted double Q + conditional value diffusion + sequence policy."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from .features import collate, FEATURE_CONFIG
from .model import DST, ModelConfig


def labels(samples, horizon, device):
    out = torch.full((len(samples), horizon), -1, dtype=torch.long, device=device)
    for i, s in enumerate(samples):
        seq = s["expert"][:horizon]
        out[i, : len(seq)] = torch.tensor(seq, device=device)
    return out


def losses(model, target, samples, bc_samples, device, discount=0.99, bc_weight=3.0):
    b = collate([s["obs"] for s in samples], device)
    n = collate([s["next_obs"] for s in samples], device)
    h = model.encoder(b)
    conditions = model.conditions(h, b)
    idx = torch.arange(len(samples), device=device)
    actions = torch.tensor([s["action"] for s in samples], device=device)
    reward = torch.tensor([s["reward"] for s in samples], device=device)
    # Paper mode: both termination and truncation end bootstrapping. Raw flags stay separate in data.
    done = torch.tensor(
        [s["terminated"] or s["truncated"] for s in samples], device=device
    )
    with torch.no_grad():
        hn = target.encoder(n)
        cn = target.conditions(hn, n)
        nq1, nq2 = target.critic(cn)
        valid = n["candidate_valid"] & n["reachable"][:, 0, 1:]
        trust = target.trust(cn, valid)
        # Candidate actions drawn from the next-state actor; duplicates represent their sampling mass.
        ctx = target.actor.context(hn, n)
        logits, _ = target.actor.logits(
            ctx, n, torch.empty(len(samples), 0, device=device, dtype=torch.long)
        )
        selected = torch.distributions.Categorical(logits=logits).sample((5,)).T
        weights = trust.gather(1, selected)
        values = torch.minimum(nq1, nq2).gather(1, selected)
        future = (weights * values).sum(1) / weights.sum(1).clamp_min(1e-8)
        td = reward + discount * (~done) * future
    q1, q2 = model.critic(conditions)
    critic = F.mse_loss(q1[idx, actions], td) + F.mse_loss(q2[idx, actions], td)
    diffusion = model.diffusion.loss(conditions[idx, actions], td)
    # Freeze value targets in policy-gradient scoring, but let the shared encoder learn via actor/BC.
    with torch.no_grad():
        target_conditions = target.conditions(target.encoder(b), b)
        score = torch.minimum(q1, q2) * target.trust(
            target_conditions, b["candidate_valid"] & b["reachable"][:, 0, 1:]
        )
    sequence, logp, valid_steps = model.actor.rollout(h, b, sample=True)
    selected_score = score.gather(1, sequence.clamp_min(0)) * valid_steps
    total = selected_score.sum(1) / valid_steps.sum(1).clamp_min(1)
    # An independent rollout is an action-independent baseline conditional on observation.
    with torch.no_grad():
        baseline_seq, _, baseline_mask = model.actor.rollout(h.detach(), b, sample=True)
        baseline = (score.gather(1, baseline_seq.clamp_min(0)) * baseline_mask).sum(
            1
        ) / baseline_mask.sum(1).clamp_min(1)
    # Score function for the WHOLE sequence objective retains effects on all future sampled prefixes.
    actor = -((total - baseline).detach() * logp.sum(1)).mean()
    bc_batch = collate([s["obs"] for s in bc_samples], device)
    bc_h = model.encoder(bc_batch)
    bc = model.actor.bc(
        bc_h, bc_batch, labels(bc_samples, model.config.horizon, device)
    )
    loss = critic + diffusion + actor + bc_weight * bc
    logs = {
        k: float(v.detach())
        for k, v in dict(
            loss=loss,
            critic=critic,
            diffusion=diffusion,
            actor=actor,
            bc=bc,
            td_mean=td.mean(),
            q_mean=q1[idx, actions].mean(),
        ).items()
    }
    return loss, logs


@torch.no_grad()
def evaluate(model, data, device, batch_size=16, limit=None):
    model.eval()
    ce = 0.0
    correct = 0
    count = 0
    limit = len(data) if limit is None else min(len(data), limit)
    for start in range(0, limit, batch_size):
        samples = data[start : min(start + batch_size, limit)]
        b = collate([x["obs"] for x in samples], device)
        h = model.encoder(b)
        ce += float(
            model.actor.bc(h, b, labels(samples, model.config.horizon, device))
        ) * len(samples)
        seq, _, _ = model.actor.rollout(h, b)
        truth = torch.tensor([x["expert"][0] for x in samples], device=device)
        correct += int((seq[:, 0] == truth).sum())
        count += len(samples)
    model.train()
    return {
        "sequence_ce": ce / max(count, 1),
        "first_action_accuracy": correct / max(count, 1),
        "samples": count,
    }


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument(
        "--stop-after",
        type=int,
        help="Save a resumable checkpoint at this absolute step without changing the LR schedule",
    )
    p.add_argument("--resume")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.batch_size % a.micro_batch or a.steps < 1:
        raise ValueError(
            "Positive steps and batch-size divisible by micro-batch required"
        )
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        a.device = "cuda:" + os.environ["LOCAL_RANK"]
        torch.cuda.set_device(a.device)
        dist.init_process_group("nccl")
    if a.batch_size % (a.micro_batch * world):
        raise ValueError("Global batch must divide micro-batch * world size")
    torch.set_num_threads(a.threads)
    random.seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    out = Path(a.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=bool(a.resume))
    if world > 1:
        dist.barrier()
    data = torch.load(a.dataset, map_location="cpu", weights_only=False)
    if data["feature_config"] != FEATURE_CONFIG:
        raise ValueError("Feature configuration mismatch")
    cfg = ModelConfig(hidden=a.hidden)
    model = DST(cfg).to(a.device)
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=a.steps, eta_min=1e-5
    )
    first = 0
    best = float("inf")
    dataset_hash = hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest()
    training_config = dict(
        batch_size=a.batch_size,
        micro_batch=a.micro_batch,
        seed=a.seed,
        discount=0.99,
        bc_weight=3.0,
        target_tau=0.005,
        learning_rate=1e-4,
    )
    if a.resume:
        ck = torch.load(a.resume, map_location=a.device, weights_only=False)
        if ck["dataset_sha256"] != dataset_hash or ck["model_config"] != asdict(cfg):
            raise ValueError("Resume contract mismatch")
        if ck.get("training_config", training_config) != training_config:
            raise ValueError("Resume must preserve batch and optimizer settings")
        if ck["training_steps"] != a.steps:
            raise ValueError("Resume must preserve scheduler total steps")
        model.load_state_dict(ck["model"])
        target.load_state_dict(ck["target"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        first = ck["step"]
        best = ck["best_ce"]
        if ck.get("world_size", 1) != world:
            raise ValueError("Resume world size must match checkpoint")
        rng = ck["rank_rng"][rank]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all([x.cpu() for x in rng["cuda"]])
    manifest = dict(
        arguments=vars(a),
        model_config=asdict(cfg),
        features=FEATURE_CONFIG,
        dataset_sha256=dataset_hash,
        reward="-distance_m + 100 * true_finish",
        algorithm="sequence trusted-Q + BC; twin-Q TD; TD-target conditional diffusion",
        started_at=time.time(),
    )
    manifest["world_size"] = world
    manifest["training_config"] = training_config
    manifest["runtime"] = dict(
        python=sys.version.split()[0],
        torch=str(torch.__version__),
        numpy=np.__version__,
        cuda=torch.version.cuda,
        gpu_names=[
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
    )
    manifest["source_sha256"] = {
        str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    if rank == 0:
        (out / "config.json").write_text(json.dumps(manifest, indent=2))
    # Identical parameter initialization across ranks, independent rollout/noise streams afterward.
    if not a.resume:
        torch.manual_seed(a.seed + rank)
    started = time.monotonic()

    def checkpoint(step):
        rng = dict(
            python=random.getstate(),
            torch=torch.get_rng_state(),
            cuda=[x.cpu() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
        )
        all_rng = [None] * world
        if world > 1:
            dist.all_gather_object(all_rng, rng)
        else:
            all_rng = [rng]
        return dict(
            training_config=training_config,
            world_size=world,
            rank_rng=all_rng,
            step=step,
            training_steps=a.steps,
            model=model.state_dict(),
            target=target.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            model_config=asdict(cfg),
            feature_config=FEATURE_CONFIG,
            dataset_sha256=dataset_hash,
            best_ce=best,
            python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        )

    step = first
    with (
        (out / "metrics.jsonl").open("a") if rank == 0 else open(os.devnull, "w")
    ) as log:
        for step in range(first + 1, a.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            metrics = {}
            accum = a.batch_size // (a.micro_batch * world)
            for _ in range(accum):
                samples = random.choices(data["train"], k=a.micro_batch)
                bc_samples = random.choices(data["bc_train"], k=a.micro_batch)
                loss, values = losses(model, target, samples, bc_samples, a.device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(values)
                (loss / accum).backward()
                for k, v in values.items():
                    metrics[k] = metrics.get(k, 0.0) + v / accum
            if world > 1:
                # Sum each parameter's gradient once; shared encoder belongs to one optimizer.
                for parameter in model.parameters():
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(world)
                keys = sorted(metrics)
                averages = torch.tensor([metrics[k] for k in keys], device=a.device)
                dist.all_reduce(averages)
                averages /= world
                metrics = dict(zip(keys, averages.tolist()))
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for p0, p1 in zip(target.parameters(), model.parameters()):
                    p0.lerp_(p1, 0.005)
            metrics.update(
                step=step, grad_norm=float(norm), elapsed_s=time.monotonic() - started
            )
            if (
                step == 1
                or step % a.eval_every == 0
                or step == a.steps
                or step == a.stop_after
            ):
                val = (
                    evaluate(model, data["bc_validation"], a.device)
                    if rank == 0
                    else None
                )
                if world > 1:
                    box = [val]
                    dist.broadcast_object_list(box, src=0)
                    val = box[0]
                metrics["validation"] = val
                ce = val["sequence_ce"]
                improved = ce < best
                best = min(best, ce)
                ck = checkpoint(step)
                if rank == 0:
                    if improved:
                        atomic_save(ck, out / "best.pt")
                    atomic_save(ck, out / "last.pt")
                    print(json.dumps(metrics), flush=True)
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
            log.flush()
            if step == a.stop_after and step < a.steps:
                break
    if rank == 0 and step == a.steps:
        (out / "completion.json").write_text(
            json.dumps(
                dict(
                    completed=True,
                    steps=a.steps,
                    best_validation_ce=best,
                    elapsed_s=time.monotonic() - started,
                    dataset_sha256=dataset_hash,
                ),
                indent=2,
            )
        )

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
