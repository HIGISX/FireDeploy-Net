# FireDeploy-Net

### Multi-period deployment · Mobile fire units · Deep reinforcement learning

Supplement permanent fire stations with mobile units that station and reposition across three periods under a cumulative distance budget.

[Quick start](#quick-start) · [Training](#data-and-training) · [Inference](#inference) · [Data specification](docs/data_format.md) · [Reproducibility](#tests-and-reproducibility)

---

| Encode | Deploy | Verify |
| :--- | :--- | :--- |
| Spatial attention and gated service features | Sequential decisions with period–repositioning context | Independent service evaluation and directed minimum-cost matching |

The policy uses **REINFORCE with a leave-one-out baseline**. Its objective equally weights normalized demand-weighted service-utility improvement and area-weighted squared-overtime reduction.

> [!NOTE]
> Processed Beijing data and pretrained or fine-tuned weights are not included. The synthetic example below runs without them.

## Installation

**Python 3.10–3.13** · **PyTorch ≥2.6, <3.0** · **CPU or single GPU**

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

**12 candidates · 20 demand grids · 2 units per period · 2 training updates**

```bash
python examples/make_synthetic.py --output data/synthetic
python -m firedeploy.train --data data/synthetic/instance.json --config configs/smoke.json --output runs/smoke --from-scratch --device cpu --microbatch 2 --threads 1
python -m firedeploy.infer --data data/synthetic/instance.json --checkpoint runs/smoke/best.pt --output runs/smoke/deployment.json --device cpu --threads 1
```

Use a new output directory when repeating the example. This checks functionality, not paper results.

## Data and training

Prepare an `instance.json` manifest following the [data and checkpoint specification](docs/data_format.md): aligned demand weights, response times, candidate coordinates, directed distances and service-gain matrices. Road-network preprocessing and demand construction are outside this repository.

**Fine-tune** from a single-period pretrained policy:

```bash
python -m firedeploy.train --data data/beijing/instance.json --config configs/beijing.json --pretrained checkpoints/single_period.pt --output runs/beijing --device cuda --microbatch 8
```

[Beijing configuration](configs/beijing.json): 200 epochs, 50 updates per epoch, effective batch 128, learning rate `1e-4`, and evaluation every 50 updates. Training uses Adam, gradient clipping at 1 and no entropy regularization. `--microbatch` controls memory use; the leave-one-out baseline uses the full effective batch.

**Outputs:** `latest.pt` for resuming, `best.pt` for the highest evaluated deterministic objective, `final.pt`, `history.json` and `summary.json`.

**Resume** with the same data, configuration and execution settings:

```bash
python -m firedeploy.train --data data/beijing/instance.json --config configs/beijing.json --resume runs/beijing/latest.pt --output runs/beijing --device cuda --microbatch 8
```

Resume supports checkpoints from this public implementation, not the original distributed-training optimizer and random-state format.

## Inference

```bash
python -m firedeploy.infer --data data/beijing/instance.json --checkpoint runs/beijing/best.pt --output runs/deployment.json --device cuda
```

Use `--device cpu` for CPU execution.

- **Deployment:** selected candidate IDs, service metrics, matched repositioning pairs, total distance and feasibility.
- **Timing:** loading, warm-up, decoding and verification are recorded separately. The default warm-up is excluded from decoding time but included in total time.

## Tests and reproducibility

```bash
python -m unittest discover -s tests -v
```

Tests cover utility thresholds, reward consistency, directed-distance budget feasibility, policy updates and checkpoint loading.

| Study environment | Release validation |
| :--- | :--- |
| Python 3.10.20 · PyTorch 2.6.0 | Python 3.13.13 · PyTorch 2.9.1 · CPU · synthetic data |

**Scope.** This portable single-process implementation runs service updates and matching on the CPU. The study's distributed training and optimized GPU environment are not included. A full Beijing experiment was not rerun for release validation; identical training trajectories or reported runtimes are not guaranteed. Support for other input sizes does not establish generalization.

**Feasibility.** The conservative mask may exclude feasible actions. Completed plans are independently checked using response times and directed minimum-cost matching.

---

## Citation and license

*Augmenting Permanent Fire Station Networks through Multi-Period Stationing and Repositioning of Mobile Units: A Deep Reinforcement Learning Approach*

Please cite this manuscript when using the code. Publication metadata will be added when available.

[MIT License](LICENSE) · Attention encoder adapted from [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route), with its [copyright and MIT notice](licenses/attention-learn-to-route-MIT.txt) retained.
