# Processed input format

All paths in `instance.json` are resolved relative to that JSON file. Input files are read-only. No network access, credentials, server paths or automatic downloads are needed by the runtime.

There are exactly three periods, ordered `morning`, `midday`, `evening`. Let `N` be the number of candidates and `M` the number of demand grids. Use a consistent row order throughout.

```json
{
  "units_per_period": 100,
  "budget_km": 250.6030337546764,
  "baseline": "baseline.npz",
  "normalization": "normalization.json",
  "candidates": "candidates.csv",
  "distance_km": "distance_km.npy",
  "responses": {
    "morning": "morning_response.npy",
    "midday": "midday_response.npy",
    "evening": "evening_response.npy"
  },
  "gains": {
    "morning": {"U": "morning_U_gain.npz", "Q": "morning_Q_gain.npz"},
    "midday": {"U": "midday_U_gain.npz", "Q": "midday_Q_gain.npz"},
    "evening": {"U": "evening_U_gain.npz", "Q": "evening_Q_gain.npz"}
  }
}
```

## Arrays and units

| Input | Content |
|---|---|
| `baseline.npz: candidate_id` | `(N,)` numeric candidate identifiers |
| `baseline.npz: grid_id` | `(M,)` numeric demand-grid identifiers |
| `baseline.npz: weights` | `(3,M)` nonnegative demand weights, each period sums to 1 |
| `baseline.npz: area_weights` | `(M,)` nonnegative area weights summing to 1 |
| `baseline.npz: baseline_response_min` | `(3,M)` fixed-station response times in minutes |
| `candidates.csv` | `candidate_id,x_m,y_m`; coordinates in a common projected CRS, metres |
| `distance_km.npy` | `(N,N)` FP64 **directed** relocation distances in kilometres; exact zero diagonal; positive infinity may mark unreachable pairs |
| `*_response.npy` | `(N,M)` mobile response times in minutes |
| `*_U_gain.npz`, `*_Q_gain.npz` | SciPy sparse matrices `(N,M)` with **unnormalized**, nonnegative weighted gains |

The paper used 5,580 candidates and 25,577 valid 300 m demand grids. Its demand weights combined population and AHP-weighted POIs equally. Morning/evening used nighttime population and midday used daytime population. Road and response-time preprocessing is outside the scope of this repository.

## Gain construction

For period `t`, grid `i`, candidate `j`, baseline time `r0[t,i]`, mobile time `r[t,j,i]`, utility `u(r)` and squared overtime `q(r)=max(r-5,0)^2`:

```text
gain_U[t,j,i] = demand_weight[t,i] * max(u(r[t,j,i]) - u(r0[t,i]), 0)
gain_Q[t,j,i] = area_weight[i] * max(q(r0[t,i]) - q(r[t,j,i]), 0)
```

The runtime divides each matrix by its objective-specific normalization range and by 3 for equal-period aggregation. Do not normalize the input gain matrices a second time.

`normalization.json` contains:

```text
U0       = mean U of the fixed-station baseline
Q0       = mean Q of the fixed-station baseline
delta_U  = mean U of the all-candidate reference minus U0
delta_Q  = Q0 minus mean Q of the all-candidate reference
```

Both ranges must be positive. The all-candidate reference takes the elementwise minimum response time across the fixed baseline and every candidate. The published implementation checks the decoded reward against metrics recomputed from response times. That check detects inconsistent gains or normalization for the evaluated plan; it is not a complete preprocessing audit.

## Checkpoints

Single-period initialization expects a trusted PyTorch dictionary with `policy` containing spatial and gated-service parameters, without the `adapt_*` branch. The branch is added with a zero-initialized output layer.

Inference expects `policy` containing the full FireDeploy-Net parameters. `update` is optional metadata. No checkpoint download is bundled. A random-initialized synthetic run tests functionality only and must not be presented as reproducing the paper's selected solution or performance.
