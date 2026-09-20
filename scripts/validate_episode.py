#!/usr/bin/env python3
"""Validate raw EPIC episodes before accepting them into a dataset (no ROS needed)."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def vector(value, size):
    return (
        isinstance(value, list) and len(value) == size and all(finite(v) for v in value)
    )


def rows(path):
    with path.open() as stream:
        for n, line in enumerate(stream, 1):
            require(line.endswith("\n"), f"{path.name}:{n}: incomplete line")
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}:{n}: {exc}") from exc


def validate_decision(d, expected):
    require(d["schema_version"] == 1, "Unknown decision schema")
    require(d["decision_id"] == expected, f"Noncontiguous decision ID at {expected}")
    require(finite(d["stamp"]), "Invalid decision time")
    for k in ["position", "velocity", "planning_anchor"]:
        require(vector(d[k], 3), f"Invalid {k}")
    require(finite(d["yaw"]) and finite(d["capture_ms"]), "Invalid yaw/capture time")
    nodes = d["nodes"]
    candidates = d["candidates"]
    n = len(candidates)
    require([p["id"] for p in nodes] == list(range(len(nodes))), "Invalid node IDs")
    require(
        sum(p["is_current"] for p in nodes) == 1, "Expected exactly one current node"
    )
    require(0 <= d["current_node_id"] < len(nodes), "Invalid current node ID")
    require(nodes[d["current_node_id"]]["is_current"], "Current node mismatch")
    for node in nodes:
        require(vector(node["position"], 3), "Invalid graph position")
    for edge in d["edges"]:
        require(
            len(edge) == 3
            and all(type(i) is int and 0 <= i < len(nodes) for i in edge[:2]),
            "Invalid edge endpoint",
        )
        # EPIC sometimes has adjacency without cached weights; preserve null explicitly.
        require(edge[2] is None or finite(edge[2]), "Invalid edge weight")
    require([c["id"] for c in candidates] == list(range(n)), "Invalid candidate IDs")
    for c in candidates:
        require(
            type(c["node_id"]) is int and 0 <= c["node_id"] < len(nodes),
            "Invalid candidate node ID",
        )
        require(
            c["position"] == nodes[c["node_id"]]["position"],
            "Candidate position mismatch",
        )
        require(
            finite(c["yaw"]) and finite(c["odom_cost"]), "Invalid candidate yaw/cost"
        )
    result = d["result"]
    route = d["expert_route"]
    m = d["matrix_dimension"]
    require(
        result
        in [
            "lkh",
            "single_candidate",
            "no_candidates",
            "no_reachable_candidates",
            "policy",
        ],
        "Unknown decision result",
    )
    if result == "lkh":
        require(n >= 2 and m == n + 1, "Invalid LKH dimension")
        require(
            len(d["cost_matrix"]) == m * m and all(finite(x) for x in d["cost_matrix"]),
            "Invalid cost matrix",
        )
        require(
            route and route[0] == 0 and sorted(route) == list(range(n + 1)),
            "Invalid LKH route",
        )
    elif result == "policy":
        require(n >= 2 and 2 <= len(route) <= min(n + 1, 6), "Invalid policy horizon")
        require(
            route[0] == 0
            and len(set(route)) == len(route)
            and all(0 <= x <= n for x in route),
            "Invalid policy route",
        )
        reach = d["reachable"]
        dim = d["reachability_dimension"]
        require(
            dim == n + 1 and len(reach) == dim * dim, "Invalid reachability dimensions"
        )
        require(
            all(reach[u * dim + v] for u, v in zip(route, route[1:])),
            "Unreachable policy edge",
        )
    elif result == "single_candidate":
        require(n == 1 and route == [0, 1], "Invalid single-candidate route")
        require(m == 0 and d["cost_matrix"] == [], "Unexpected single-candidate matrix")
    else:
        require(
            n == 0 and route == [] and d["selected_candidate_id"] is None,
            "Invalid empty decision",
        )
        require(m == 0 and d["cost_matrix"] == [], "Unexpected empty-decision matrix")
    if n:
        require(
            type(d["selected_candidate_id"]) is int
            and 0 <= d["selected_candidate_id"] < n,
            "Invalid selected candidate",
        )
        require(d["selected_candidate_id"] == route[1] - 1, "Action/route mismatch")
    return n


def validate(directory, require_complete=False):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    require(meta["schema_version"] == 1, "Unknown episode schema")
    require(not summary["errors"], f"Episode errors: {summary['errors']}")
    require(
        not summary.get("surviving_owned_pids", []),
        "Owned ROS processes survived cleanup",
    )
    reason = summary["reason"]
    require(
        reason in ["epic_finish", "time_limit"], "Episode did not complete normally"
    )
    require(summary["terminated"] == (reason == "epic_finish"), "Wrong terminated flag")
    require(summary["truncated"] == (reason == "time_limit"), "Wrong truncated flag")
    require(
        not require_complete or reason == "epic_finish",
        "Truncated episode is not a completed exploration",
    )
    require(
        finite(summary["distance_m"]) and summary["distance_m"] > 0.1,
        "No actual flight",
    )
    counts = Counter()
    trigger_stamp = None
    last_event = None
    for event in rows(directory / "events.jsonl"):
        kind = event["event"]
        counts[kind] += 1
        last_event = event
        require(finite(event["wall_time"]), "Invalid telemetry time")
        if kind == "trigger":
            trigger_stamp = event["stamp"]
        if kind == "odometry":
            for key, length in [("position", 3), ("velocity", 3), ("orientation", 4)]:
                require(vector(event[key], length), "Invalid odometry " + key)
        if kind == "trajectory":
            require(finite(event["start_time"]), "Invalid trajectory time")
            durations = event["duration"]
            require(
                durations and all(finite(x) and x > 0 for x in durations),
                "Invalid trajectory durations",
            )
            for key in ["coef_x", "coef_y", "coef_z"]:
                require(
                    vector(event[key], len(durations) * (event["order"] + 1)),
                    "Invalid trajectory coefficients",
                )
    require(
        counts["trigger"] == 1 and counts["episode_end"] == 1,
        "Missing/duplicate episode boundary",
    )
    require(
        last_event["event"] == "episode_end" and last_event["reason"] == reason,
        "Missing final episode boundary",
    )
    require(
        counts["odometry"] == summary["odometry_count"] and counts["odometry"] > 0,
        "Odometry count mismatch",
    )
    require(
        counts["trajectory"] == summary["trajectory_count"]
        and counts["trajectory"] > 0,
        "Trajectory count mismatch",
    )
    report = {
        "valid": True,
        "reason": reason,
        "distance_m": summary["distance_m"],
        "telemetry_counts": dict(counts),
    }
    if meta["baseline"]:
        require(
            not (directory / "decisions.jsonl").exists(),
            "Baseline unexpectedly contains decisions",
        )
        return report
    status = json.loads((directory / "decisions.jsonl.status.json").read_text())
    require(
        status["schema_version"] == 1 and not status["failed"],
        "Recorder reported failure",
    )
    total = 0
    kinds = Counter()
    latencies = []
    post_trigger = 0
    edge_null = 0
    max_candidates = 0
    for d in rows(directory / "decisions.jsonl"):
        n = validate_decision(d, total)
        total += 1
        kinds[d["result"]] += 1
        max_candidates = max(max_candidates, n)
        latencies.append(d["capture_ms"])
        post_trigger += d["stamp"] >= trigger_stamp
        edge_null += sum(edge[2] is None for edge in d["edges"])
    require(
        total > 0 and total == status["enqueued"] == status["written"],
        "Missing/unflushed decisions",
    )
    require(post_trigger > 0, "No decisions after trigger")
    latencies.sort()
    report.update(
        decisions=total,
        decision_kinds=dict(kinds),
        post_trigger_decisions=post_trigger,
        max_candidates=max_candidates,
        null_edge_weights=edge_null,
        capture_ms_p50=latencies[len(latencies) // 2],
        capture_ms_p95=latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        capture_ms_max=latencies[-1],
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode")
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    try:
        report = validate(a.episode, a.require_complete)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
