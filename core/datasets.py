from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def generate_linear_data(
    separable: bool = True,
    n_samples: int = 50,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Genera un dataset sintético bidimensional con dos clases etiquetadas {-1, +1}."""
    rng: np.random.Generator = np.random.default_rng(seed)

    if separable:
        centers: NDArray[np.float64] = np.array([
            [-5.0, -5.0],  # Centro clase -1
            [ 5.0,  5.0],  # Centro clase +1
        ])
        std_dev: float = 1.2
    else:
        centers: NDArray[np.float64] = np.array([
            [-1.0, -1.0],  # Centro clase -1
            [ 1.0,  1.0],  # Centro clase +1
        ])
        std_dev: float = 3.5
        
    n_per_class: int = n_samples // 2
    n_remainder: int = n_samples - 2 * n_per_class

    X_neg_class: NDArray[np.float64] = rng.normal(
        loc=centers[0], scale=std_dev, size=(n_per_class, 2)
    )
    X_pos_class: NDArray[np.float64] = rng.normal(
        loc=centers[1], scale=std_dev, size=(n_per_class + n_remainder, 2)
    )

    X: NDArray[np.float64] = np.vstack([X_neg_class, X_pos_class])
    y: NDArray[np.int8] = np.concatenate([
        -np.ones(n_per_class, dtype=np.int8),
         np.ones(n_per_class + n_remainder, dtype=np.int8),
    ])

    idx: NDArray[np.intp] = rng.permutation(n_samples)
    return X[idx], y[idx]

def generate_nonlinear_data(
    type_name: str = 'moons',
    n_samples: int = 200,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Genera distribuciones clásicas no separables linealmente ('moons' o 'circles')."""
    rng = np.random.default_rng(seed)
    
    n_out = n_samples // 2
    n_in = n_samples - n_out

    if type_name == 'moons':
        t_out = np.linspace(0, np.pi, n_out)
        t_in = np.linspace(0, np.pi, n_in)
        
        X_out = np.column_stack([np.cos(t_out), np.sin(t_out)])
        X_in = np.column_stack([1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5])
        
    elif type_name == 'circles':
        t_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
        t_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
        
    else:
        raise ValueError("Tipo no soportado. Elija 'moons' o 'circles'.")

    X = np.vstack([X_out, X_in])
    y = np.concatenate([np.ones(n_out, dtype=np.int8), -np.ones(n_in, dtype=np.int8)])
    
    X += rng.normal(scale=noise, size=X.shape)
    
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]

def generate_ring_data(
    n_samples: int = 300,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Genera un dataset sintético bidimensional: anillos concéntricos no linealmente separables."""
    rng = np.random.default_rng(seed)
    n_class = n_samples // 2

    # Clase -1: Círculo interior
    radius_in = rng.uniform(0, 2.5, n_class)
    theta_in = rng.uniform(0, 2 * np.pi, n_class)
    X_in = np.column_stack([radius_in * np.cos(theta_in), radius_in * np.sin(theta_in)])
    y_in = -np.ones(n_class, dtype=np.int8)

    # Clase +1: Anillo exterior
    radius_out = rng.uniform(4.0, 6.0, n_class)
    theta_out = rng.uniform(0, 2 * np.pi, n_class)
    X_out = np.column_stack([radius_out * np.cos(theta_out), radius_out * np.sin(theta_out)])
    y_out = np.ones(n_class, dtype=np.int8)

    X = np.vstack([X_in, X_out])
    X += rng.standard_normal(X.shape) * noise
    y = np.concatenate([y_in, y_out])

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]

def generate_multiclass_data(
    n_classes: int = 4,
    points_per_class: int = 25,
    seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Genera un dataset sintético bidimensional: 4 clases (clusters) en las esquinas."""
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []
    
    # Centroides para 4 clases
    centroids = [
        [2.5, 2.5],   # Clase 0
        [2.5, -2.5],  # Clase 1
        [-2.5, 2.5],  # Clase 2
        [-2.5, -2.5]  # Clase 3
    ]
    
    for i in range(n_classes):
        points = rng.standard_normal((points_per_class, 2)) * 0.7 + np.array(centroids[i])
        X_list.append(points)
        y_list.append(np.full(points_per_class, i, dtype=np.int8))
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Mezclar
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]
