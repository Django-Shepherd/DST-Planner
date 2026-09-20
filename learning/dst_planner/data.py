"""Prepare audited, map-disjoint datasets from complete EPIC expert runs.

A decision is a receding-horizon viewpoint plan, not a completed arrival.
Only plans linked to published trajectories are TD actions. Transitions span
successive applied plans, using trajectory activation times. The source
observation age is retained, and no target arrival is assumed.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from .features import FEATURE_CONFIG, observation


REWARD_DESCRIPTION = "-actual_distance_m + 100 * true_finish"
TRANSITION_DESCRIPTION = (
    "trajectory_activation_to_next_applied_plan_activation; "
    "source_observation_age_preserved"
)


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def expert_sequence(record, horizon=5):
    route = record["expert_route"][1:]
    if len(set(route)) != len(route):
        raise ValueError("Repeated expert candidate")
    reachable = observation(record).reachable
    prefix = []
    last = 0
    for candidate in route[:horizon]:
        if not 1 <= candidate < len(reachable):
            raise ValueError("Expert ID outside candidate list")
        # Only a legal prefix is sequence supervision.
        if not reachable[last, candidate]:
            break
        prefix.append(candidate - 1)
        last = candidate
    return prefix


def episode(path, horizon=5):
    """Convert one validated episode into executed transitions and expert labels."""
    path = Path(path)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from validate_episode import validate

    validation = validate(path, require_complete=True)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["baseline"] or not metadata.get("learning_features"):
        raise ValueError("Not a learning episode")
    summary = json.loads((path / "summary.json").read_text())
    coverage = summary.get("coverage")
    if coverage and coverage["observed_fraction"] < 0.95:
        raise ValueError(
            "Training excludes premature finish below 95 percent observed floor coverage"
        )
    if coverage and (
        coverage["near_obstacle_odometry_samples"]
        or coverage["outside_map_odometry_samples"]
    ):
        raise ValueError(
            "Training excludes episodes with clearance/outside-map violations"
        )

    decisions = rows(path / "decisions.jsonl")
    events = rows(path / "events.jsonl")
    trigger = next(event["stamp"] for event in events if event["event"] == "trigger")
    finish = next(event["stamp"] for event in events if event.get("state") == "FINISH")
    odometry = sorted(
        (event for event in events if event["event"] == "odometry"),
        key=lambda event: event["stamp"],
    )
    times = np.array([event["stamp"] for event in odometry])
    positions = np.array([event["position"] for event in odometry])
    distances = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    ]
    observations = [observation(decision) for decision in decisions]
    sequences = [
        expert_sequence(decision, horizon) if decision["candidates"] else []
        for decision in decisions
    ]
    applied = {}
    execution = [event for event in events if event["event"] == "execution"]
    trajectory_ids = {
        event["traj_id"] for event in events if event["event"] == "trajectory"
    }
    for event in execution:
        if event["kind"] != "trajectory_applied" or event["stamp"] < trigger:
            continue
        index = event["decision_id"]
        if index < 0:
            continue
        if index >= len(decisions) or event["traj_id"] not in trajectory_ids:
            raise ValueError("Unknown applied decision/trajectory")
        decision = decisions[index]
        action = decision["selected_candidate_id"]
        if action is None or not np.allclose(
            decision["candidates"][action]["position"], event["goal"], atol=1e-4
        ):
            raise ValueError("Applied goal does not match decision")
        applied.setdefault(index, event)
    ids = sorted(
        index for index in applied if trigger <= decisions[index]["stamp"] < finish
    )
    if not ids:
        raise ValueError("No executed post-trigger decisions")

    samples = []
    rejected = 0
    for offset, index in enumerate(ids):
        decision = decisions[index]
        start = max(applied[index]["start_time"], applied[index]["stamp"])
        terminal = offset == len(ids) - 1
        next_index = ids[offset + 1] if not terminal else None
        end = (
            max(applied[next_index]["start_time"], applied[next_index]["stamp"])
            if next_index is not None
            else finish
        )
        if end <= start:
            raise ValueError("Nonpositive transition duration")
        # Recovery changes control authority: do not label it as a policy action.
        if any(
            event["kind"] == "recovery_applied" and start <= event["stamp"] < end
            for event in execution
        ):
            rejected += 1
            continue
        meters = float(
            np.interp(end, times, distances) - np.interp(start, times, distances)
        )
        samples.append(
            dict(
                obs=observations[index],
                next_obs=(
                    observations[next_index]
                    if next_index is not None
                    else observations[index]
                ),
                action=decision["selected_candidate_id"],
                expert=sequences[index],
                reward=-meters + 100.0 * terminal,
                terminated=terminal,
                truncated=False,
                distance_m=meters,
                duration_s=end - start,
                decision_id=index,
                next_decision_id=next_index,
                apply_latency_s=start - decision["stamp"],
                activation_stamp=start,
                next_activation_stamp=end,
                map_sha256=metadata["map_sha256"],
                episode=str(path),
            )
        )
    bc = [
        dict(obs=obs, expert=sequence)
        for decision, obs, sequence in zip(decisions, observations, sequences)
        if sequence and trigger <= decision["stamp"] < finish
    ]
    audit = dict(
        episode=str(path),
        map_sha256=metadata["map_sha256"],
        validated=validation,
        decisions=len(decisions),
        sequence_samples=len(bc),
        transitions=len(samples),
        recovery_intervals_excluded=rejected,
        finish_rewards=sum(sample["terminated"] for sample in samples),
        transition_distance_m=sum(sample["distance_m"] for sample in samples),
        total_reward=sum(sample["reward"] for sample in samples),
        max_apply_latency_s=max(sample["apply_latency_s"] for sample in samples),
    )
    assert audit["finish_rewards"] <= 1
    assert (
        abs(
            audit["total_reward"]
            - (-audit["transition_distance_m"] + 100 * audit["finish_rewards"])
        )
        < 1e-5
    )
    return samples, bc, audit


def admit_episode(path):
    """Require expert control and a complete coverage audit before labeling a run."""
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("policy_mode", "epic") != "epic":
        raise ValueError("Only EPIC expert runs may enter this dataset")
    if not metadata.get("coverage_grid"):
        raise ValueError("Coverage audit is required for dataset admission")
    summary = json.loads((path / "summary.json").read_text())
    coverage = summary.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Completed coverage audit is required for dataset admission")
    fraction = coverage.get("observed_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0.95 <= fraction <= 1.0
    ):
        raise ValueError(
            "Training requires at least 95 percent observed floor coverage"
        )
    for key in ("near_obstacle_odometry_samples", "outside_map_odometry_samples"):
        if type(coverage.get(key)) is not int or coverage[key] != 0:
            raise ValueError(
                "Training excludes episodes with clearance/outside-map violations"
            )
    samples, bc, audit = episode(path)
    if not samples or not bc:
        raise ValueError("No TD/sequence training samples")
    scene = Path(metadata["map"]).parent / "scene.json"
    audit["map_kind"] = (
        json.loads(scene.read_text()).get("kind", "unknown")
        if scene.exists()
        else "unknown"
    )
    return samples, bc, audit


def admit_roots(roots):
    """Read each episode directory once, retaining all admission failures."""
    samples = []
    bc = []
    audits = []
    rejected = []
    seen = set()
    kinds = {}
    for root in roots:
        for metadata in sorted(Path(root).glob("*/metadata.json")):
            path = metadata.parent.resolve()
            if path in seen:
                continue
            seen.add(path)
            try:
                episode_samples, episode_bc, audit = admit_episode(path)
                map_hash = audit["map_sha256"]
                kind = audit["map_kind"]
                if map_hash in kinds and kinds[map_hash] != kind:
                    raise ValueError("Conflicting scene types for the same map")
                kinds[map_hash] = kind
                samples.extend(episode_samples)
                bc.extend(dict(sample, map_sha256=map_hash) for sample in episode_bc)
                audits.append(audit)
            except (ValueError, KeyError, TypeError, OSError, AssertionError) as exc:
                rejected.append(dict(episode=str(path), reason=str(exc)))
    return samples, bc, audits, rejected


def split_maps(audits, seed):
    """Reserve 20 percent of each scene type, keeping every map's starts together."""
    kinds = {audit["map_sha256"]: audit["map_kind"] for audit in audits}
    maps = sorted(kinds)
    rng = np.random.default_rng(seed)
    validation = set()
    for kind in sorted(set(kinds.values())):
        group = [map_hash for map_hash in maps if kinds[map_hash] == kind]
        rng.shuffle(group)
        if len(group) < 2:
            raise ValueError(
                "Need at least two maps per scene type for stratified splitting"
            )
        validation.update(group[: max(1, round(0.2 * len(group)))])
    return validation


def build_dataset(roots, output, min_maps=3, min_episodes=3, seed=20260917):
    """Write admission results first, then a dataset with map-disjoint splits."""
    if min_maps < 1 or min_episodes < 1:
        raise ValueError("Minimum map and episode counts must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    samples, bc, audits, rejected = admit_roots(roots)
    by_map = Counter(audit["map_sha256"] for audit in audits)
    report = dict(
        seed=seed,
        episodes=audits,
        rejected=rejected,
        episodes_per_map=dict(by_map),
        reward=REWARD_DESCRIPTION,
        features=FEATURE_CONFIG,
    )
    (output / "admission.json").write_text(json.dumps(report, indent=2))
    if len(audits) < min_episodes:
        raise ValueError(
            f"Only {len(audits)} accepted episodes; require {min_episodes}"
        )
    if len(by_map) < min_maps:
        raise ValueError(f"Only {len(by_map)} accepted maps; require {min_maps}")
    validation = split_maps(audits, seed)
    data = dict(
        train=[sample for sample in samples if sample["map_sha256"] not in validation],
        validation=[sample for sample in samples if sample["map_sha256"] in validation],
        bc_train=[sample for sample in bc if sample["map_sha256"] not in validation],
        bc_validation=[sample for sample in bc if sample["map_sha256"] in validation],
        feature_config=FEATURE_CONFIG,
    )
    counts = {key: len(value) for key, value in data.items() if isinstance(value, list)}
    if not all(counts.values()):
        raise ValueError("Empty dataset split")
    torch.save(data, output / "dataset.pt")
    report.update(
        training_maps=sorted(set(by_map) - validation),
        validation_maps=sorted(validation),
        counts=counts,
        transition_semantics=TRANSITION_DESCRIPTION,
        dataset_sha256=hashlib.sha256((output / "dataset.pt").read_bytes()).hexdigest(),
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Collection directories containing episode subdirectories",
    )
    parser.add_argument("--output", required=True, help="New dataset directory")
    parser.add_argument("--min-maps", type=int, default=3)
    parser.add_argument("--min-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.min_maps < 1 or args.min_episodes < 1:
        parser.error("Minimum map and episode counts must be positive")
    try:
        report = build_dataset(
            args.roots, args.output, args.min_maps, args.min_episodes, args.seed
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Dataset preparation failed: {exc}\n")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
