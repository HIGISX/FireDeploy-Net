# FireDeploy-Net

Deep reinforcement learning for multi-period stationing and repositioning of mobile fire units.

This repository provides the model, training and deterministic inference code accompanying:

**Augmenting Permanent Fire Station Networks through Multi-Period Stationing and Repositioning of Mobile Units: A Deep Reinforcement Learning Approach**

FireDeploy-Net supplements an existing permanent fire-station network. It selects mobile-unit locations for morning peak, midday off-peak and evening peak under a cumulative repositioning-distance budget. The policy combines a spatial attention encoder, dual-service gated fusion, a period–repositioning branch and a dynamic feasibility mask.

## Scope

The release contains a portable, single-process implementation of the model and REINFORCE learning objective. The original parameter names are retained to support research checkpoints containing a `policy` state dictionary. CPU and single-GPU execution are supported; the environment performs FP64 service updates and exact directed matching on the CPU.

The paper's training run also used distributed and GPU-environment optimizations. Those machine-specific execution components are not part of this release. This implementation is not expected to reproduce that run's exact stochastic trajectory or reported runtime. Compatibility of parameter names alone does not establish numerical parity of a complete Beijing rollout.

This is a **code-only release**. Beijing input arrays, pretrained and fine-tuned weights, experiment logs, comparison algorithms and plotting scripts are not included. The synthetic example below tests the software and does not reproduce paper results. Complete Beijing reproduction requires the processed instance and the single-period pretrained policy; U9050 inference requires the selected fine-tuned checkpoint.

## Installation

Python 3.10 or later is required. The research environment used Python 3.10.20 and PyTorch 2.6.0. Install an appropriate PyTorch 2.6.0 build for your CUDA driver or CPU first, then install this package:

```bash
git clone https://github.com/HIGISX/FireDeploy-Net.git
cd FireDeploy-Net
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

For a CUDA-specific wheel, use the installation command supplied by [PyTorch](https://pytorch.org/get-started/previous-versions/). PyTorch 1.13 is not supported by this implementation because the model uses scaled dot-product attention and non-reentrant activation checkpointing.

The dependency ranges allow PyTorch 2.6–2.x. Release checks were run on CPU with Python 3.13.13 and PyTorch 2.9.1; this is separate from the paper's training environment. The checks cover synthetic training/inference, feasibility, checkpoint loading, and agreement of encoder outputs and action scores with the archived model on small inputs. A complete Beijing run was not repeated for this release.

## Quick check on synthetic data

```bash
python examples/make_synthetic.py --output data/synthetic
python -m firedeploy.train --data data/synthetic/instance.json --config configs/smoke.json --output runs/smoke --from-scratch --device cpu --microbatch 2 --threads 1
python -m firedeploy.infer --data data/synthetic/instance.json --checkpoint runs/smoke/best.pt --output runs/smoke/deployment.json --device cpu --threads 1
python -m unittest discover -s tests -v
```

The example creates 12 artificial candidates and 20 demand grids, with two units in each of three periods. Only two tiny policy updates are performed. Use a new output directory when repeating a check.

## Training

```bash
python -m firedeploy.train --data data/beijing/instance.json --config configs/beijing.json --pretrained checkpoints/single_period.pt --output runs/beijing --device cuda --microbatch 8
```

The baseline configuration specifies 200 epochs, 50 updates per epoch, an effective batch of 128 complete deployments, Adam with learning rate `1e-4`, gradient-norm clipping at 1 and deterministic evaluation every 50 updates. There is no entropy regularization. The leave-one-out baseline is computed across the full effective batch. Microbatching uses sampled-action replay to reduce memory use without changing the batch-level REINFORCE objective.

Training writes `latest.pt` (optimizer and RNG state), `best.pt` (highest evaluated deterministic objective), `final.pt`, `history.json` and `summary.json`. Existing output directories are protected from accidental replacement. Resume using the same input, configuration, visible device setup and execution options:

```bash
python -m firedeploy.train --data data/beijing/instance.json --config configs/beijing.json --resume runs/beijing/latest.pt --output runs/beijing --device cuda --microbatch 8
```

`--resume` accepts checkpoints produced by this release. Historical distributed checkpoints have a different optimizer/RNG layout; their policy weights can be used for inference, but they are not supported as exact continuation checkpoints here. Load only trusted checkpoint files.

## Deterministic inference

```bash
python -m firedeploy.infer --data data/beijing/instance.json --checkpoint checkpoints/U9050.pt --output runs/U9050.json --device cuda
```

Decoding selects the highest-probability legal candidate at every step. Output includes candidate indices and IDs, per-period service metrics, matched transitions, total distance and feasibility. Initialization, warm-up, decoding and independent verification times are reported separately. The default single warm-up is excluded from `decode_seconds` but included in `total_seconds`.

## Objective and constraints

Let `U` be the equal-period mean demand-weighted continuous service utility, and `Q` the equal-period mean area-weighted squared excess response time above 5 minutes. The baseline objective is

```text
F = 0.5 * (U - U0) / delta_U + 0.5 * (Q0 - Q) / delta_Q
```

Utility equals 1 up to 5 minutes, follows a normalized truncated Gaussian between 5 and 20 minutes, and equals 0 from 20 minutes onward. The squared-overtime penalty continues beyond 20 minutes. Normalization constants compare the fixed-station baseline with an all-candidate service reference.

The paper's baseline uses 100 units per period and a cumulative budget of 250.6030337546764 km across the two period transitions. Matching uses directed road distances and exact one-to-one minimum-cost assignment. The constructive feasibility mask maintains a feasible completion; it is sufficient, not necessary, and may exclude some feasible actions. Full feasibility is checked independently after decoding.

The current network assumes exactly three periods and the baseline equal-weight objective. The input interface also allows smaller instances for testing; this should not be interpreted as validated cross-scale generalization.

## Input format and repository layout

See [docs/data_format.md](docs/data_format.md) for array shapes, units, gain definitions and the instance manifest.

```text
firedeploy/     Model, environment, training and inference
configs/        Beijing training budget and a tiny software-check budget
examples/       Synthetic input generator
tests/          Reward, feasibility, gradient and checkpoint checks
docs/           Processed-input specification
licenses/       Third-party license notices
```

## Attribution

This repository is released under the [MIT License](LICENSE).

The graph-attention encoder is adapted from [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) by Wouter Kool and collaborators, with the original projection layout retained and LayerNorm / PyTorch SDPA used here. The original MIT copyright and permission notice is retained in `licenses/`.

For scientific use, cite the associated manuscript. Publication metadata will be added when available.
