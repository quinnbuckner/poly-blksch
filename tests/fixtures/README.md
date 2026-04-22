# `tests/fixtures/`

Synthetic data generators and recorded fixtures used by tests in all three levels.

## Files

- `synthetic.py` — RN-consistent jump-diffusion path generator. Matches paper §6.2/§6.4 setup (N=6000 at 1 Hz, symmetric Gaussian jumps, scheduled jump window). Ground truth for Track A calibration tests.
- `live_replay_*.parquet` — recorded Polymarket WS sessions (pending — Track A to drop once `polyclient.py` can record)
- `recorded_clob_responses.json` — pinned examples of live CLOB REST responses for contract tests (pending — Track A/C)

## How to use

```python
from tests.fixtures.synthetic import (
    SyntheticConfig,
    generate_rn_consistent_path,
    inject_microstructure_noise,
    causal_forward_sum_variance,
    qlike,
)

path = generate_rn_consistent_path()        # ground truth
y, sigma_eta2 = inject_microstructure_noise(path.x)  # noisy observations
# Track A's pipeline consumes `y` and (ideally) recovers `path.sigma_b_arr`,
# `path.lambda_arr`, `path.jump_sizes`.
```

## Ground-truth guarantees

- `path.x` is a discretized Itô-Lévy process with drift computed to satisfy the martingale restriction (paper eq 3) under the given `sigma_b` and `lambda`
- `path.jumps` marks jump indices; `path.jump_sizes` gives signed magnitudes
- `path.sigma_b_arr` and `path.lambda_arr` include the scheduled-window boost
- RN consistency is validated by a Monte Carlo martingale check inside the generator (`_mu_rn`)

## Adding new fixtures

- Keep synthetic generators deterministic via `rng_seed`
- Recorded fixtures must be scrubbed of any wallet addresses / API keys
- Large fixtures (> 1 MB) go in a subfolder with a README explaining provenance
