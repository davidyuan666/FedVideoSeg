"""
Simple Privacy Protection Mechanisms
"""

import hashlib
import numpy as np
from typing import Any, Dict

class SimplePrivacyManager:
    """Simple privacy protection without complex differential privacy."""
    
    def __init__(self, enable_anonymization: bool = True):
        self.enable_anonymization = enable_anonymization
        
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple data anonymization."""
        if not self.enable_anonymization:
            return data
        
        anonymized = data.copy()
        
        # Remove or hash sensitive identifiers
        if 'user_id' in anonymized:
            anonymized['user_id'] = self._hash_identifier(anonymized['user_id'])
        
        if 'student_id' in anonymized:
            anonymized['student_id'] = self._hash_identifier(anonymized['student_id'])
        
        # Remove metadata that could identify users
        sensitive_keys = ['ip_address', 'device_id', 'session_id']
        for key in sensitive_keys:
            if key in anonymized:
                del anonymized[key]
        
        return anonymized
    
    def _hash_identifier(self, identifier: str) -> str:
        """Hash sensitive identifiers."""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def add_simple_noise(self, parameters: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
        """Add simple Gaussian noise to parameters."""
        noise = np.random.normal(0, noise_scale, parameters.shape)
        return parameters + noise 