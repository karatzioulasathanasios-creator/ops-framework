"""
Core OPS framework functions.
"""

import numpy as np
from scipy import stats, linalg
from typing import Tuple, Dict, Optional

def calculate_projectivity_index(loadings: np.ndarray) -> float:
    """
    Calculate the projectivity index Π.
    
    Parameters
    ----------
    loadings : np.ndarray
        Eigenvector loadings from PCA/EOF analysis
        
    Returns
    -------
    float
        Projectivity index Π ∈ [0, 1]
    """
    n = len(loadings)
    n_positive = np.sum(loadings > 0)
    f = n_positive / n
    pi = 1 - abs(2 * f - 1)
    return pi

def projective_structure_detection(data: np.ndarray, 
                                   n_components: int = 1) -> Dict:
    """
    Detect projective structure in multi-channel time series.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (n_time, n_channels) time series data
    n_components : int
        Number of principal components to extract
        
    Returns
    -------
    Dict with results including:
        - loadings: PC loadings
        - scores: PC scores
        - explained_variance: explained variance ratio
        - pi: projectivity index
    """
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # PCA/SVD
    U, s, Vt = linalg.svd(data_centered, full_matrices=False)
    
    # Extract components
    loadings = Vt[:n_components].T  # Shape (n_channels, n_components)
    scores = U[:, :n_components] * s[:n_components]
    
    # Explained variance
    total_var = np.sum(s ** 2)
    explained_var = (s ** 2) / total_var
    
    # Calculate Π for each component
    pi_values = [calculate_projectivity_index(loadings[:, i]) 
                 for i in range(n_components)]
    
    return {
        'loadings': loadings,
        'scores': scores,
        'explained_variance': explained_var[:n_components],
        'projectivity_index': pi_values,
        'singular_values': s[:n_components]
    }

def rolling_window_analysis(data: np.ndarray, 
                           window_size: int,
                           step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform rolling window stability analysis.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data
    window_size : int
        Size of rolling window
    step : int
        Step between windows
        
    Returns
    -------
    Tuple of (pi_values, stability_scores)
    """
    n_time = data.shape[0]
    pi_values = []
    stability_scores = []
    
    for start in range(0, n_time - window_size + 1, step):
        window_data = data[start:start + window_size]
        result = projective_structure_detection(window_data)
        pi_values.append(result['projectivity_index'][0])
        
        # Simple stability score (can be enhanced)
        if len(pi_values) > 1:
            stability = 1 - np.std(pi_values[-5:]) if len(pi_values) >= 5 else 1.0
            stability_scores.append(stability)
        else:
            stability_scores.append(1.0)
    
    return np.array(pi_values), np.array(stability_scores)

def null_calibration(data: np.ndarray, 
                    n_surrogates: int = 1000,
                    method: str = 'phase_randomization') -> Dict:
    """
    Generate null distributions for significance testing.
    
    Parameters
    ----------
    data : np.ndarray
        Original time series
    n_surrogates : int
        Number of surrogate datasets
    method : str
        Surrogate method: 'phase_randomization', 'time_permutation'
        
    Returns
    -------
    Dict with null distribution statistics
    """
    n_time, n_channels = data.shape
    pi_null = []
    
    for _ in range(n_surrogates):
        if method == 'phase_randomization':
            # Fourier phase randomization
            surrogate = _phase_randomize(data)
        elif method == 'time_permutation':
            # Randomly permute time indices
            surrogate = _permute_time(data)
        else:
            surrogate = data.copy()
        
        result = projective_structure_detection(surrogate)
        pi_null.append(result['projectivity_index'][0])
    
    pi_null = np.array(pi_null)
    original_result = projective_structure_detection(data)
    pi_original = original_result['projectivity_index'][0]
    
    # Calculate p-value
    p_value = np.mean(pi_null >= pi_original)
    
    return {
        'null_distribution': pi_null,
        'original_pi': pi_original,
        'p_value': p_value,
        'mean_null': np.mean(pi_null),
        'std_null': np.std(pi_null),
        'z_score': (pi_original - np.mean(pi_null)) / np.std(pi_null) if np.std(pi_null) > 0 else 0
    }

def _phase_randomize(data: np.ndarray) -> np.ndarray:
    """Phase randomization surrogate."""
    n_time, n_channels = data.shape
    surrogate = np.zeros_like(data)
    
    for ch in range(n_channels):
        # FFT
        freq = np.fft.fft(data[:, ch])
        amplitudes = np.abs(freq)
        phases = np.angle(freq)
        
        # Randomize phases
        random_phases = np.random.uniform(0, 2*np.pi, len(phases))
        # Keep zero frequency phase unchanged
        random_phases[0] = phases[0]
        
        # Inverse FFT
        surrogate[:, ch] = np.fft.ifft(amplitudes * np.exp(1j * random_phases)).real
    
    return surrogate

def _permute_time(data: np.ndarray) -> np.ndarray:
    """Time permutation surrogate."""
    n_time, n_channels = data.shape
    surrogate = data.copy()
    
    # Permute time indices independently for each channel
    for ch in range(n_channels):
        np.random.shuffle(surrogate[:, ch])
    
    return surrogate