# FireDeploy-Net

Deep reinforcement learning for stationing and repositioning mobile fire units across three periods, supplementing permanent fire stations under a cumulative repositioning-distance budget.

Associated manuscript: *Augmenting Permanent Fire Station Networks through Multi-Period Stationing and Repositioning of Mobile Units: A Deep Reinforcement Learning Approach*.

The policy combines spatial attention, gated service features and period–repositioning context, and is trained with REINFORCE and a leave-one-out baseline. Its objective equally weights normalized demand-weighted service-utility improvement and area-weighted squared-overtime reduction. The feasibility mask is conservative and may exclude feasible actions; completed plans are independently checked using response times and directed minimum-cost matching.

## Installation

Python 3.10–3.13 and PyTorch 2.6–2.x are required. CPU and single-GPU execution are supported.

```bash
git clone https://github.com/HIGISX/FireDeploy-Net.git
cd FireDeploy-Net
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. For GPU execution, first install a compatible [PyTorch build](https://pytorch.org/get-started/previous-versions/), then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick start

This synthetic example needs no external data or weights. It uses 12 candidates, 20 demand grids and two units per period, with two training updates.

```bash
python examples/make_synthetic.py --output data/synthetic
python -m firedeploy.train --data data/synthetic/instance.json --config configs/smoke.json --output runs/smoke --from-scratch --device cpu --microbatch 2 --threads 1
python -m firedeploy.infer --data data/synthetic/instance.json --checkpoint runs/smoke/best.pt --output runs/smoke/deployment.json --device cpu --threads 1
```

Use a new output directory when repeating the example. This checks functionality, not paper results.

## Data and training

Processed Beijing data and pretrained or fine-tuned weights are **not included**. Prepare an `instance.json` manifest following [the data and checkpoint specification](docs/data_format.md), including aligned demand weights, response times, candidate coordinates, directed distances and service-gain matrices. Road-network preprocessing and demand construction are outside this repository.

Fine-tuning requires a single-period pretrained policy:

```bash
python -m firedeploy.train --data data/beijing/instance.json --config configs/beijing.json --pretrained checkpoints/single_period.pt --output runs/beijing --device cuda --microbatch 8
```

[The Beijing configuration](configs/beijing.json) specifies 200 epochs, 50 updates per epoch, an effective batch of 128, a learning rate of `1e-4` and evaluation every 50 updates. Training uses Adam, gradient clipping at 1 and no entropy regularization. `--microbatch` controls memory use; the leave-one-out baseline uses the full effective batch.

Training saves `latest.pt` for resuming, `best.pt` for the highest evaluated deterministic objective, `final.pt`, `history.json` and `summary.json`.

To resume, retain the same data, configuration and execution settings:

```bash
python -m firedeploy.train --data data/beijing/instance.json --config configs/beijing.json --resume runs/beijing/latest.pt --output runs/beijing --device cuda --microbatch 8
```

Resume supports checkpoints from this public implementation, not the original distributed-training optimizer and random-state format.

## Inference

```bash
python -m firedeploy.infer --data data/beijing/instance.json --checkpoint runs/beijing/best.pt --output runs/deployment.json --device cuda
```

Use `--device cpu` for CPU execution. The output includes selected candidate IDs, service metrics, matched repositioning pairs, total distance and feasibility. Loading, warm-up, decoding and verification are timed separately; the default warm-up is excluded from decoding time but included in total time.

## Tests and reproducibility

```bash
python -m unittest discover -s tests -v
```

Tests cover utility thresholds, reward consistency, directed-distance budget feasibility, policy updates and checkpoint loading.

This is a portable single-process implementation: service updates and matching run on the CPU. The study's distributed training and optimized GPU environment are not included. Release validation used synthetic data on Python 3.13.13 / PyTorch 2.9.1 (CPU); the study used Python 3.10.20 / PyTorch 2.6.0. A full Beijing experiment was not rerun for release validation. Identical training trajectories or reported runtimes are not guaranteed, and support for other input sizes does not establish generalization.

## Citation and license

Please cite the associated manuscript when using this code. Publication metadata will be added when available.

Released under the [MIT License](LICENSE). The attention encoder is adapted from [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route); its [copyright and MIT notice](licenses/attention-learn-to-route-MIT.txt) are retained.
