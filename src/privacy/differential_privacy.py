"""
Simple Differential Privacy Protection Mechanisms
"""

import hashlib
import numpy as np
from typing import Any, Dict, Optional
import math

class DifferentialPrivacy:
    """Simple differential privacy implementation for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, enable_anonymization: bool = True):
        """
        Initialize differential privacy manager.
        
        Args:
            epsilon: Privacy budget parameter (smaller = more private)
            delta: Probability of privacy breach (should be very small)
            enable_anonymization: Whether to enable data anonymization
        """
        self.epsilon = epsilon
        self.delta = delta
        self.enable_anonymization = enable_anonymization
        
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple data anonymization with differential privacy principles."""
        if not self.enable_anonymization:
            return data
        
        anonymized = data.copy()
        
        # Remove or hash sensitive identifiers
        if 'user_id' in anonymized:
            anonymized['user_id'] = self._hash_identifier(anonymized['user_id'])
        
        if 'student_id' in anonymized:
            anonymized['student_id'] = self._hash_identifier(anonymized['student_id'])
        
        # Remove metadata that could identify users
        sensitive_keys = ['ip_address', 'device_id', 'session_id', 'timestamp']
        for key in sensitive_keys:
            if key in anonymized:
                del anonymized[key]
        
        return anonymized
    
    def _hash_identifier(self, identifier: str) -> str:
        """Hash sensitive identifiers with salt."""
        salt = "federated_learning_privacy"
        return hashlib.sha256((identifier + salt).encode()).hexdigest()[:16]
    
    def add_gaussian_noise(self, parameters: np.ndarray, sensitivity: float = 1.0, 
                          clip_norm: Optional[float] = None) -> np.ndarray:
        """
        Add Gaussian noise for differential privacy.
        
        Args:
            parameters: Model parameters to add noise to
            sensitivity: L2 sensitivity of the function
            clip_norm: Gradient clipping norm (if None, no clipping)
        
        Returns:
            Parameters with differential privacy noise added
        """
        # Clip gradients if specified
        if clip_norm is not None:
            parameters = self._clip_gradients(parameters, clip_norm)
            sensitivity = clip_norm
        
        # Calculate noise scale for Gaussian mechanism
        noise_scale = self._calculate_gaussian_noise_scale(sensitivity)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_scale, parameters.shape)
        return parameters + noise
    
    def add_laplace_noise(self, parameters: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """
        Add Laplace noise for differential privacy.
        
        Args:
            parameters: Model parameters to add noise to  
            sensitivity: L1 sensitivity of the function
        
        Returns:
            Parameters with differential privacy noise added
        """
        # Calculate noise scale for Laplace mechanism
        noise_scale = sensitivity / self.epsilon
        
        # Add Laplace noise
        noise = np.random.laplace(0, noise_scale, parameters.shape)
        return parameters + noise
    
    def _calculate_gaussian_noise_scale(self, sensitivity: float) -> float:
        """Calculate noise scale for Gaussian mechanism."""
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("Delta must be between 0 and 1")
        
        # Gaussian noise scale for (epsilon, delta)-differential privacy
        noise_scale = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return noise_scale
    
    def _clip_gradients(self, gradients: np.ndarray, clip_norm: float) -> np.ndarray:
        """Clip gradients to bound sensitivity."""
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > clip_norm:
            return gradients * (clip_norm / grad_norm)
        return gradients
    
    def compose_privacy_budget(self, num_queries: int) -> tuple:
        """
        Calculate composed privacy budget for multiple queries.
        
        Args:
            num_queries: Number of queries/rounds
            
        Returns:
            Tuple of (total_epsilon, total_delta) after composition
        """
        # Simple composition (conservative bound)
        total_epsilon = self.epsilon * num_queries
        total_delta = self.delta * num_queries
        
        return total_epsilon, total_delta
    
    def get_privacy_spent(self, num_rounds: int) -> Dict[str, float]:
        """Get privacy budget spent after num_rounds."""
        total_eps, total_delta = self.compose_privacy_budget(num_rounds)
        return {
            'epsilon_spent': total_eps,
            'delta_spent': total_delta,
            'epsilon_remaining': max(0, 10.0 - total_eps),  # Assume total budget of 10
            'privacy_level': 'high' if total_eps < 1.0 else 'medium' if total_eps < 5.0 else 'low'
        }

