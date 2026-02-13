"""
Privacy Accountant.

Tracks privacy budget (epsilon, delta) consumption over composition of mechanisms.
Supports Basic Composition and Advanced Composition theorems.
"""

import math
from typing import Tuple, List, Tuple

class PrivacyAccountant:
    """
    Tracks (epsilon, delta) usage for unlearning steps.
    """
    
    def __init__(self):
        self.steps = 0
        self.total_epsilon = 0.0
        self.total_delta = 0.0
        self.mechanism_history: List[Tuple[str, float, float]] = []
    
    def step(self, epsilon: float, delta: float = 0.0, mechanism: str = "gaussian"):
        """
        Record a privacy expenditure.
        """
        self.steps += 1
        self.mechanism_history.append((mechanism, epsilon, delta))
        
        # Naive tracking (Basic Composition)
        self.total_epsilon += epsilon
        self.total_delta += delta
        
    def get_budget(self, advanced_composition: bool = False, delta_prime: float = 1e-5) -> Tuple[float, float]:
        """
        Return the total privacy cost (epsilon_global, delta_global).
        
        Parameters
        ----------
        advanced_composition : bool
            If True, uses Dwork's Advanced Composition Theorem.
        delta_prime : float
            The slack parameter delta' used in Advanced Composition.
            
        Returns
        -------
        (epsilon_total, delta_total)
        """
        if not advanced_composition:
            # Basic Composition Theorem
            # epsilon_total = sum(epsilon_i)
            # delta_total = sum(delta_i)
            return self.total_epsilon, self.total_delta
        
        # Advanced Composition Theorem (Dwork et al. 2010)
        # For k mechanisms M1...Mk where Mi is (eps, delta)-DP:
        # The composition is (eps_total, delta_total + delta')-DP
        # eps_total = sum(eps_i * (e^eps_i - 1)) + sqrt(2 * sum(eps_i^2) * log(1/delta'))
        
        # We assume heterogeneous epsilons.
        # Reference: "The Algorithmic Foundations of Differential Privacy", Theorem 3.20 (approx)
        
        sum_eps_squared = sum(eps**2 for _, eps, _ in self.mechanism_history)
        sum_eps_exp = sum(eps * (math.exp(eps) - 1) for _, eps, _ in self.mechanism_history)
        sum_deltas = sum(delta for _, _, delta in self.mechanism_history)
        
        eps_term_1 = sum_eps_exp # k * eps * (e^eps - 1) approx k * eps^2? No, explicit sum.
        # Actually standard bound for homogeneous is:
        # eps' = k*eps*(e^eps - 1) + eps * sqrt(2k log(1/delta'))
        
        # For heterogeneous:
        # eps_total = sum(eps_i * (e^eps_i - 1)) + sqrt(2 * log(1/delta') * sum(eps_i^2))
        
        if sum_eps_squared == 0:
            return 0.0, 0.0

        eps_total = sum_eps_exp + math.sqrt(2 * math.log(1/delta_prime) * sum_eps_squared)
        delta_total = sum_deltas + delta_prime
        
        return eps_total, delta_total
