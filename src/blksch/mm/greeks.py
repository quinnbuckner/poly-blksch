"""Greeks in logit units (paper §4.1).

For the vanilla event contract V = p = S(x), where S is the logistic:

    S(x) = 1 / (1 + e^-x)
    Δ_x  = S'(x)  = p(1-p)
    Γ_x  = S''(x) = p(1-p)(1-2p)

Belief-vega ν_b := ∂V/∂σ_b and correlation-vega ν_ρ := ∂V/∂ρ_ij carry
different functional forms depending on the derivative. For the vanilla
contract, ν_b = 0 (the payoff does not depend on σ_b at fixed x). For the
short-maturity p-variance swap:

    ν_b ∝ (p(1-p))^2 · σ_b     (paper eq 6, Jacobian S'(x)^2)

For x-variance swaps ν_b ∝ σ_b (level-stable). ν_ρ applies to cross-event
hedging products and is non-zero only for multi-token claims.
"""

from __future__ import annotations

import math

__all__ = [
    "sigmoid",
    "logit",
    "delta_x",
    "gamma_x",
    "s_prime",
    "s_double_prime",
    "vega_b_xvar",
    "vega_b_pvar",
    "vega_rho_pair",
    "clip_p",
]

_EPS_DEFAULT = 1.0e-5


def clip_p(p: float, eps: float = _EPS_DEFAULT) -> float:
    """Clip probability to [eps, 1-eps] to avoid log-odds explosion."""
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")
    if p < eps:
        return eps
    if p > 1.0 - eps:
        return 1.0 - eps
    return p


def sigmoid(x: float) -> float:
    """S(x) = 1/(1+e^-x), numerically stable for large |x|."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float, eps: float = _EPS_DEFAULT) -> float:
    """logit(p) = log(p/(1-p)). Clips to avoid ±∞."""
    q = clip_p(p, eps)
    return math.log(q / (1.0 - q))


def s_prime(x: float) -> float:
    """S'(x) = p(1-p)."""
    p = sigmoid(x)
    return p * (1.0 - p)


def s_double_prime(x: float) -> float:
    """S''(x) = p(1-p)(1-2p)."""
    p = sigmoid(x)
    return p * (1.0 - p) * (1.0 - 2.0 * p)


def delta_x(p: float) -> float:
    """Δ_x = p(1-p). Input is probability, not logit."""
    return p * (1.0 - p)


def gamma_x(p: float) -> float:
    """Γ_x = p(1-p)(1-2p). Input is probability, not logit."""
    return p * (1.0 - p) * (1.0 - 2.0 * p)


def vega_b_xvar(sigma_b: float) -> float:
    """Belief-vega for an x-variance swap: ν_b ∝ σ_b (level-stable)."""
    return sigma_b


def vega_b_pvar(p: float, sigma_b: float) -> float:
    """Short-maturity p-variance belief-vega: ν_b ∝ (p(1-p))^2 · σ_b.

    Arises from the S'(x)^2 Jacobian in paper eq (6).
    """
    s = p * (1.0 - p)
    return (s * s) * sigma_b


def vega_rho_pair(p_i: float, p_j: float, sigma_b_i: float, sigma_b_j: float) -> float:
    """Correlation-vega for a vanilla pair position hedging dp_i against dp_j.

    ∂V/∂ρ_ij for a pair variance/covariance swap scales with the product of
    Jacobians and belief-vols: S'_i · S'_j · σ_b^i · σ_b^j.
    """
    return p_i * (1.0 - p_i) * p_j * (1.0 - p_j) * sigma_b_i * sigma_b_j
