# Example Experiment

The following results were obtained on the generated 40 m scenes using the collection, training and evaluation pipeline in the README. Raw datasets and pretrained checkpoints are not included in this repository.

## Training and checkpoint selection

- EPIC base: `d73c3150e57d669ac21bcad5c859f52c2827c9ba`.
- Tested setup: Ubuntu 20.04, ROS Noetic, Python 3.10.18, PyTorch 2.1.0+cu121, NumPy 1.26.4, and two RTX 2080 Ti GPUs.
- Forty admitted episodes from twenty forest/partition maps, split by point-cloud hash into sixteen training maps and four validation maps. The three closed-loop test maps were disjoint.
- Training: 5,249 executed transitions and 5,419 BC sequences. Validation: 1,129 transitions and 1,176 sequences.
- Completed 10,000 steps in 46.34 minutes, with global batch 256, per-GPU micro-batch 128 and seed 20260917.
- Full twin-Q TD, conditional value diffusion, trust weighting and sequence policy-gradient/BC objectives. Online export contains only the encoder and Actor.

**Validation selected the step-250 checkpoint.** Its sequence cross-entropy was 0.816362 and first-viewpoint label agreement was 59.44%. At step 10,000 these were 1.116779 and 32.91%, indicating later regression. Completing all steps does not establish convergence of the last checkpoint. The guide exports `best.pt` by default.

## Selected Actor on independent maps

Each scene had a 300-second budget. All three naturally completed under `strict` control, with zero inference errors, fallback events, LKH decisions, near-obstacle records or out-of-map samples.

| Scene | Observed reachable-floor coverage | Odometry distance (m) | Time after trigger (s) | Feature/model inference p95 (ms) |
|---|---:|---:|---:|---:|
| Dungeon | 98.697% | 82.52 | 40.63 | 11.74 |
| Forest | 99.981% | 296.29 | 133.13 | 13.86 |
| Partition | 99.912% | 166.38 | 73.17 | 14.53 |

Coverage measures reachable floor cells actually observed by LiDAR, not complete 3D volume coverage. Distances include brief pre-trigger stabilization; training rewards integrate only actual execution intervals. Inference latency excludes full C++ communication and planning time.

The step-10,000 Actor passed dungeon and partition but timed out in forest after 300 seconds, reaching 62.241% coverage. Another intermediate checkpoint produced near-obstacle records in forest and timed out in partition. These failures are not included in the selected Actor's passing results.

The 40 m scenes and forty admitted episodes do not reproduce the full paper setting of 500 trajectories, 100 m maps and Hotel/Cave benchmarks, or demonstrate statistical superiority to EPIC. ROS scheduling and CUDA sparse accumulation introduce nondeterminism; identical commands need not produce identical weights or completion times.
