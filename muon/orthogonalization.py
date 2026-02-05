from math import inf, sqrt

import numpy as np
import torch
from torch import Tensor


def newton_schulz(
    grad: Tensor,
    coefficients: list[tuple[float, float, float]],
    eps: float = 1e-7,
    safety_factor: float = 1e-2,
) -> Tensor:
    assert grad.dim() == 2, f"Input must be a 2D tensor, but got {grad.dim()}D tensor."
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm() * (1 + safety_factor) + eps)  # ensure spectral norm <= 1
    for a, b, c in coefficients:
        gram_mat = ortho_grad @ ortho_grad.mT
        # addmm(A, B, C, beta, alpha=1) computes beta * A + alpha * (B @ C)
        # so, gram_update <- b * G^2 + c * G^4
        gram_update = torch.addmm(gram_mat, gram_mat, gram_mat, beta=b, alpha=c)
        # and G <- a * G + (b * G^2 + c * G^4) G
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad.to(grad.dtype)


def newton_schulz_coefficients(
    steps: int,
) -> list[tuple[float, float, float]]:
    # Newton-Schulz orthogonalization, adopted from PyTorch's implementation
    return [(3.4445, -4.7750, 2.0315) for _ in range(steps)]


# Polar-Express Sign Method, adopted from https://github.com/NoahAmsel/PolarExpress


def optimal_quintic(
    l: float,
    u: float,
) -> tuple[float, float, float]:
    assert 0 <= l <= u, f"Expected 0 <= l <= u, got l={l}, u={u}"
    if 1 - 5e-6 <= l / u:
        # Above this threshold, the equioscillating polynomials
        # is numerically equal to...
        return (15 / 8) / u, (-10 / 8) / (u**3), (3 / 8) / (u**5)
    # This initialization becomes exact as l -> u
    q = (3 * l + 1) / 4
    r = (l + 3) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt((-3 * b + np.array([-1, 1]) * sqrt(9 * b**2 - 20 * a * c)) / (10 * c))
    return float(a), float(b), float(c)


target_slope = 0


def obj(
    l: float,
) -> float:
    a, b, c = optimal_quintic(l, 1)
    total = a + b + c
    a /= total
    b /= total
    c /= total
    local_argmin = np.sqrt((-3 * b + sqrt(9 * b**2 - 20 * a * c)) / (10 * c))
    local_min = a * local_argmin + b * local_argmin**3 + c * local_argmin**5
    return local_min / local_argmin - target_slope


def polar_express_coefficients(
    l: float,
    num_iters: int,
    safety_factor_eps: float,
    cushion: float,
) -> list[tuple[float, float, float]]:
    u = 1
    assert 0 <= l <= u
    safety_factor = 1 + safety_factor_eps
    coefficients = []
    for iter in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion * u), u)
        if cushion * u > l:
            # Due to cushioning, this may be centered around 1 with
            # respect to 0.024*u, u. Recenter it around 1 with respect
            # to l, u, meaning find c so that 1 - c*p(l) = c*p(u) - 1:
            pl = a * l + b * l**3 + c * l**5
            pu = a * u + b * u**3 + c * u**5
            rescaler = 2 / (pl + pu)
            a *= rescaler
            b *= rescaler
            c *= rescaler
        # Optionally incorporate safety factor here:
        if iter < num_iters - 1:  # don't apply to last polynomial
            a /= safety_factor
            b /= safety_factor**3
            c /= safety_factor**5
        coefficients.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return coefficients
