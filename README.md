# FireDeploy-Net

**Deep reinforcement learning for multi-period stationing and repositioning of mobile fire units.**

Code accompanying the manuscript:

> **Augmenting Permanent Fire Station Networks through Multi-Period Stationing and Repositioning of Mobile Units: A Deep Reinforcement Learning Approach**

## Overview

FireDeploy-Net generates mobile fire-unit deployment plans that supplement an existing permanent fire-station network. It jointly considers morning peak, midday off-peak and evening peak, subject to the number of available units and a cumulative repositioning-distance budget.

The model integrates a spatial attention encoder, dual-service gated fusion, a period–repositioning branch and a dynamic feasibility mask. Training uses REINFORCE with a leave-one-out baseline; deterministic inference selects the highest-probability legal candidate at each step.

The baseline objective combines normalized continuous service-utility improvement and squared-overtime reduction:

$$
F = \frac{1}{2}\frac{U-U_0}{\Delta U}
  + \frac{1}{2}\frac{Q_0-Q}{\Delta Q}.
$$

Here, $U$ is the three-period mean demand-weighted continuous service utility, and $Q$ is the three-period mean area-weighted squared response overtime above 5 minutes. Normalization uses the permanent-station baseline and an all-candidate service reference. Repositioning distances are evaluated through exact one-to-one minimum-cost matching on a directed distance matrix.

## Repository structure

```text
FireDeploy-Net/
├── firedeploy/              Model, environment, training and inference
├── configs/
│   ├── beijing.json         Beijing training configuration
│   └── smoke.json           Synthetic example configuration
├── examples/               Synthetic instance generator
├── tests/                  Automated correctness checks
├── docs/data_format.md     Input and checkpoint specifications
├── licenses/               Third-party license notices
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

**Requirements:** Python 3.10–3.13 and PyTorch 2.6 or later within the 2.x series. CPU and single-GPU execution are supported.

```bash
git clone https://github.com/HIGISX/FireDeploy-Net.git
cd FireDeploy-Net
python -m venv .venv
```

Activate the environment using `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell. For GPU execution, first install a suitable PyTorch build following the [official installation instructions](https://pytorch.org/get-started/previous-versions/).

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick start

This example runs without the Beijing dataset or pretrained weights. It creates an artificial instance with 12 candidates, 20 demand grids and two units per period, then performs two training updates and deterministic inference.

```bash
python examples/make_synthetic.py --output data/synthetic
python -m firedeploy.train --data data/synthetic/instance.json --config configs/smoke.json --output runs/smoke --from-scratch --device cpu --microbatch 2 --threads 1
python -m firedeploy.infer --data data/synthetic/instance.json --checkpoint runs/smoke/best.pt --output runs/smoke/deployment.json --device cpu --threads 1
```

Use a new output directory when repeating the example. Synthetic outputs verify software functionality and do not reproduce the study results.

## Data and checkpoints

This is a **code-only release**: processed Beijing data and model weights are not included.

For the Beijing case, prepare an `instance.json` manifest and its referenced arrays according to [docs/data_format.md](docs/data_format.md). Fine-tuning additionally requires a single-period pretrained policy; inference requires a full FireDeploy-Net checkpoint. All input paths are resolved relative to the manifest.

The implementation assumes three periods and the equal-weight objective defined above. The study's baseline uses 100 mobile units per period and a cumulative repositioning budget of approximately 250.60 km. The full-precision budget and data definitions are provided in the input specification.

## Training

Fine-tune from a single-period pretrained policy:

```bash
python -m firedeploy.train --data data/beijing/instance.json --config configs/beijing.json --pretrained checkpoints/single_period.pt --output runs/beijing --device cuda --microbatch 8
```

The paths above assume that the required data and weights have been prepared locally. Use `--device cpu` for CPU execution.

The baseline training settings are:

| Setting | Value |
|---|---:|
| Epochs | 200 |
| Updates per epoch | 50 |
| Effective batch size | 128 complete deployments |
| Optimizer | Adam |
| Learning rate | $10^{-4}$ |
| Gradient-norm clipping | 1 |
| Deterministic evaluation interval | 50 updates |
| Entropy regularization | None |

Configurable settings are defined in [configs/beijing.json](configs/beijing.json). The `--microbatch` option controls memory use through sampled-action replay; the leave-one-out baseline is computed over the complete effective batch.

Training outputs include:

| File | Description |
|---|---|
| `latest.pt` | Model, optimizer, random-number states and progress for resuming |
| `best.pt` | Checkpoint with the highest evaluated deterministic objective |
| `final.pt` | Model at the end of training |
| `history.json` | Training records and periodic evaluations |
| `summary.json` | Final and best results, update count and elapsed time |

Resume with the same data, configuration and execution settings:

```bash
python -m firedeploy.train --data data/beijing/instance.json --config configs/beijing.json --resume runs/beijing/latest.pt --output runs/beijing --device cuda --microbatch 8
```

The resume entry point accepts checkpoints generated by this implementation. Original distributed-training checkpoints use a different optimizer and random-state format and are not supported for exact continuation.

## Inference

Generate and evaluate a deterministic deployment:

```bash
python -m firedeploy.infer --data data/beijing/instance.json --checkpoint checkpoints/best.pt --output runs/deployment.json --device cuda
```

The output JSON reports selected candidate IDs, per-period service metrics, matched repositioning pairs, cumulative distance and feasibility. Service metrics and matching costs are independently recomputed after decoding.

Initialization, warm-up, decoding and verification times are reported separately. By default, one warm-up deployment is excluded from `decode_seconds` and included in `total_seconds`. Existing output files are not overwritten.

## Validation and reproducibility

```bash
python -m unittest discover -s tests -v
```

Tests cover service-utility thresholds, reward consistency, directed-distance budget feasibility, policy updates and checkpoint loading. Release checks also exercised the training, resume and inference commands on synthetic data and verified parameter-name compatibility and matching forward outputs against the research model on small inputs.

| Environment | Python | PyTorch |
|---|---|---|
| Study training environment | 3.10.20 | 2.6.0 |
| Release validation environment (CPU) | 3.13.13 | 2.9.1 |

This repository provides a portable single-process implementation. Service updates and exact matching run on the CPU, while the policy runs on CPU or GPU. Distributed training and the optimized GPU environment used in the study are not included; identical training trajectories and reported runtimes are therefore not guaranteed. A full Beijing experiment was not rerun for release validation.

The feasibility mask supplies a sufficient certificate and may exclude some feasible actions. Support for other input sizes does not by itself establish cross-scale or cross-scenario generalization.

## Citation

If you use this code in research, please cite the accompanying manuscript named above. A formatted citation will be added when publication metadata becomes available.

## License and acknowledgments

This project is released under the [MIT License](LICENSE).

The graph-attention encoder is adapted from [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route), retaining its projection parameter layout and using LayerNorm and PyTorch scaled dot-product attention. The original copyright and MIT license notice are retained in [licenses/attention-learn-to-route-MIT.txt](licenses/attention-learn-to-route-MIT.txt).
