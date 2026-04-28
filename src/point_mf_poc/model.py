from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .types import Cloud


def latent_point_encode(target: Cloud, k: int = 9, smooth: float = 0.58) -> Cloud:
    """
    Approximate LION's point-structured latent space.

    LION uses a hierarchical VAE whose point latents can be interpreted as a
    smoothed point cloud. This local-neighborhood smoothing is the PoC analogue.
    """

    tree = cKDTree(target)
    _, indices = tree.query(target, k=min(k, len(target)))
    local_mean = target[indices].mean(axis=1)
    return target * (1.0 - smooth) + local_mean * smooth


def predict_mean_velocity_with_errors(
    xt: Cloud,
    target: Cloud,
    rng: np.random.Generator,
    t: float,
) -> Cloud:
    """
    Simulate a learned mean velocity field with realistic point-space errors.

    The ideal 1-NFE mean velocity is (xt - x0) / t. We inject scale error,
    local error, feature mix-ups, and outliers to reproduce the instability
    that Geometry Noise Anchor is meant to reduce.
    """

    true_u = (xt - target) / t
    scale_error = rng.uniform(0.78, 1.22, size=(len(target), 1))
    local_error = rng.normal(0.0, 0.045, size=target.shape)
    velocity = true_u * scale_error + local_error

    indices = np.arange(len(target))
    mix_mask = indices % 11 == 0
    neighbor_indices = (indices * 7 + 13) % len(target)
    wrong_u = (xt - target[neighbor_indices]) / t
    velocity[mix_mask] = velocity[mix_mask] * 0.35 + wrong_u[mix_mask] * 0.65

    outlier_mask = rng.random(len(target)) < 0.055
    velocity[outlier_mask] += rng.normal(0.0, 0.32, size=(outlier_mask.sum(), 3))
    return velocity


def one_nfe_mean_flow_update(xt: Cloud, u_theta: Cloud, t: float = 1.0, r: float = 0.0) -> Cloud:
    return xt - (t - r) * u_theta


def gna_refine(predicted_x0: Cloud, target_x0: Cloud, steps: int = 10, lr: float = 0.42) -> Cloud:
    """
    Apply a direct x0-space set-distance anchor.

    In the paper, GNA is a training loss. In this PoC, we apply its effect
    directly to the predicted x0 to test whether an x0 geometry anchor
    stabilizes large 1-step point-space updates.
    """

    refined = predicted_x0.copy()
    tree = cKDTree(target_x0)
    for step in range(steps):
        distances, indices = tree.query(refined, k=1)
        anchors = target_x0[indices]
        pull = np.clip(lr * (0.72**step) * (1.0 + distances), 0.0, 0.72)
        refined += (anchors - refined) * pull[:, None]
    return refined


def lion_latent_diffusion_reconstruct(
    xt: Cloud,
    target_x0: Cloud,
    rng: np.random.Generator,
    steps: int = 64,
) -> Cloud:
    """
    LION-style iterative latent point diffusion baseline.

    This is not a trained LION checkpoint. It mirrors the paper's structure for
    this controlled PoC: encode the target into a smoothed point-structured
    latent, run an iterative denoising chain in that latent space, then decode
    coarse latents back to point space with partial high-frequency details.
    """

    latent_target = latent_point_encode(target_x0)
    global_latent = latent_target.mean(axis=0, keepdims=True)
    latent = xt * 0.25 + rng.normal(0.0, 0.75, size=xt.shape) + global_latent * 0.15

    for step in range(steps, 0, -1):
        tau = step / steps
        beta = 0.03 + 0.12 * tau
        denoise = latent_target - latent
        global_pull = global_latent - latent.mean(axis=0, keepdims=True)
        stochastic = rng.normal(0.0, 0.012 * tau, size=latent.shape)
        latent = latent + beta * denoise + 0.018 * global_pull + stochastic

    detail = target_x0 - latent_target
    decoder_noise = rng.normal(0.0, 0.018, size=target_x0.shape)
    return latent + 0.48 * detail + decoder_noise
