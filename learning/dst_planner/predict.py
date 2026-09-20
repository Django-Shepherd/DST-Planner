"""Actor-only checkpoint export and a bounded local Unix-socket inference service."""

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import signal
import time
import torch
from .features import observation, collate, FEATURE_CONFIG
from .model import ModelConfig, GraphEncoder, SequenceActor


class Predictor:
    def __init__(self, path, device="cpu"):
        self.device = device
        ck = torch.load(path, map_location=device, weights_only=True)
        if ck["feature_config"] != FEATURE_CONFIG:
            raise ValueError("Actor feature contract mismatch")
        self.config = ModelConfig(**ck["model_config"])
        self.encoder = GraphEncoder(self.config).to(device).eval()
        self.actor = SequenceActor(self.config).to(device).eval()
        self.encoder.load_state_dict(ck["encoder"])
        self.actor.load_state_dict(ck["actor"])

    @torch.inference_mode()
    def predict(self, record):
        started = time.monotonic()
        obs = observation(record)
        if not len(obs.candidates):
            return dict(
                request_id=record["decision_id"],
                graph_version=record["graph_version"],
                route=[],
                inference_ms=0.0,
            )
        batch = collate([obs], self.device)
        h = self.encoder(batch)
        seq, _, _ = self.actor.rollout(h, batch)
        indices = [int(x) for x in seq[0].tolist() if x >= 0]
        if len(set(indices)) != len(indices):
            raise ValueError("Repeated action")
        previous = 0
        for i in indices:
            if not obs.reachable[previous, i + 1]:
                raise ValueError("Invalid connection")
            previous = i + 1
        if not indices:
            raise ValueError("No legal predicted action")
        return dict(
            request_id=record["decision_id"],
            graph_version=record["graph_version"],
            route=indices,
            inference_ms=(time.monotonic() - started) * 1000,
        )


def export(source, destination):
    src = Path(source)
    out = Path(destination)
    if out.exists():
        raise FileExistsError(out)
    source_bytes = src.read_bytes()
    ck = torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=False)
    state = ck["model"]
    payload = dict(
        model_config=ck["model_config"],
        feature_config=ck["feature_config"],
        encoder={
            k[len("encoder.") :]: v
            for k, v in state.items()
            if k.startswith("encoder.")
        },
        actor={
            k[len("actor.") :]: v for k, v in state.items() if k.startswith("actor.")
        },
        training_step=ck["step"],
        dataset_sha256=ck["dataset_sha256"],
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return payload


def serve(predictor, address):
    path = Path(address)
    if path.exists():
        raise FileExistsError("Refusing to replace existing socket: " + address)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(address)
    os.chmod(address, 0o600)
    server.listen(8)
    print(json.dumps({"ready": True, "socket": address}), flush=True)

    def shutdown(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, shutdown)
    try:
        while True:
            conn, _ = server.accept()
            with conn:
                conn.settimeout(3.0)
                data = bytearray()
                try:
                    while b"\n" not in data:
                        block = conn.recv(65536)
                        if not block:
                            raise EOFError("Incomplete request")
                        data.extend(block)
                        if len(data) > 64 * 1024 * 1024:
                            raise ValueError("Request too large")
                    record = json.loads(data.split(b"\n", 1)[0])
                    result = predictor.predict(record)
                    result["ok"] = True
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
                print(
                    json.dumps(dict(event="inference", **result), allow_nan=False),
                    flush=True,
                )
                try:
                    conn.sendall(json.dumps(result, allow_nan=False).encode() + b"\n")
                except OSError:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        server.close()
        path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--output", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--actor", required=True)
    s.add_argument("--socket", required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    if a.command == "export":
        export(a.checkpoint, a.output)
    else:
        torch.set_num_threads(a.threads)
        serve(Predictor(a.actor, a.device), a.socket)


if __name__ == "__main__":
    main()
