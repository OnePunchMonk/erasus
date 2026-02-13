"""
Influence Bounds.

Utilities to compute theoretical bounds on the influence of a data point.
Used to estimate worst-case privacy leakage.
"""

from typing import Union
import math

def compute_influence_bound(
    loss_lipschitz: float, 
    convex_param: float, 
    n_samples: int
) -> float:
    """
    Compute upper bound on the influence function norm: || I_up_loss(z) ||.
    
    Bound = L / (n * lambda)
    
    Where:
    - L: Lipschitz constant of the gradient (smoothness)? Or Lipschitz of Loss?
      Usually bound is G / lambda, where G is bound on gradient norm.
    - lambda: Strong convexity parameter.
    - n: Number of training samples.
    
    Parameters
    ----------
    loss_lipschitz : float
        Bound on the gradient norm (G).
    convex_param : float
        Strong convexity parameter (lambda).
    n_samples : int
        Size of dataset.
    """
    if convex_param <= 0:
        return float('inf')
        
    return loss_lipschitz / (n_samples * convex_param)


def compute_stability_bound(
    beta_smoothness: float,
    lambda_convexity: float,
    n_samples: int
) -> float:
    """
    Compute Uniform Stability bound (Hardt et al. 2016).
    stability <= (2 * L^2) / (lambda * n)  (approximate)
    """
    if lambda_convexity <= 0:
        return float('inf')
    # Simplified stability coefficient
    return 2.0 / (lambda_convexity * n_samples)
