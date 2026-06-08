"""
Support Vector Machine — Dual Kernelizado Avanzado
===================================================

Formulación Dual con Kernel Trick:
    min_{α}  (1/2) Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j) - Σ_i α_i
    s.t.     Σ_i α_i y_i = 0
             0 ≤ α_i ≤ C,  ∀ i

"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult


# =============================================================================
# 1. FUNCIONES KERNEL
# =============================================================================

class Kernels:
    """Implementaciones vectorizadas de funciones Kernel (Refactorizado de Soft-Margin)."""

    @staticmethod
    def linear() -> callable:
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            return X1 @ X2.T
        return calc

    @staticmethod
    def gaussian(gamma: float = 0.5) -> callable:
        """Kernel Gaussiano (Radial Basis Function)."""
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            sq_norm1 = np.sum(X1**2, axis=1, keepdims=True)
            sq_norm2 = np.sum(X2**2, axis=1)
            dist_sq = sq_norm1 + sq_norm2 - 2 * (X1 @ X2.T)
            dist_sq = np.maximum(dist_sq, 0.0)
            return np.exp(-gamma * dist_sq)
        return calc

    @staticmethod
    def polynomial(degree: float = 3, c: float = 1.0) -> callable:
        """Kernel Polinomial."""
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            return (X1 @ X2.T + c) ** degree
        return calc


# =============================================================================
# 2. GENERACIÓN DE DATOS (Anillos Concéntricos)
# =============================================================================

def generar_datos_anillos(
    n_muestras: int = 300,
    ruido: float = 0.2,
    semilla: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Genera un dataset sintético bidimensional: anillos concéntricos no linealmente separables."""
    rng = np.random.default_rng(semilla)
    n_clase = n_muestras // 2

    # Clase -1: Círculo interior
    radio_int = rng.uniform(0, 2.5, n_clase)
    theta_int = rng.uniform(0, 2 * np.pi, n_clase)
    X_int = np.column_stack([radio_int * np.cos(theta_int), radio_int * np.sin(theta_int)])
    y_int = -np.ones(n_clase, dtype=np.int8)

    # Clase +1: Anillo exterior
    radio_ext = rng.uniform(4.0, 6.0, n_clase)
    theta_ext = rng.uniform(0, 2 * np.pi, n_clase)
    X_ext = np.column_stack([radio_ext * np.cos(theta_ext), radio_ext * np.sin(theta_ext)])
    y_ext = np.ones(n_clase, dtype=np.int8)

    X = np.vstack([X_int, X_ext])
    X += rng.standard_normal(X.shape) * ruido
    y = np.concatenate([y_int, y_ext])

    idx = rng.permutation(n_muestras)
    return X[idx], y[idx]


# =============================================================================
# 3. MODELO DE OPTIMIZACIÓN — SVM DUAL KERNEL
# =============================================================================

class SVMDualKernel:
    """Implementación OOP del SVM Dual con Kernels No Lineales."""
    def __init__(self, C: float = 1.0, kernel: callable = None, tol: float = 1e-5):
        self.C = C
        self.kernel = kernel if kernel is not None else Kernels.linear()
        self.tol = tol
        self.alpha = None
        self.b = 0.0
        self.X_fit = None
        self.y_fit = None
        self.sv_mask = None
        self.is_fit = False

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8], maxiter: int = 1000) -> bool:
        """Entrena el hiperplano en espacio dual usando SLSQP."""
        n_samples = X.shape[0]
        y_float = y.astype(np.float64)

        # 1. Matriz de Gram K y Matriz Q
        K = self.kernel(X, X)
        Q = np.outer(y_float, y_float) * K

        # 2. Objetivo Primal adaptado a Dual
        def objetivo(alpha: NDArray[np.float64]) -> np.float64:
            return np.float64(0.5 * np.dot(alpha, Q @ alpha) - np.sum(alpha))

        def grad_objetivo(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
            return Q @ alpha - 1.0

        # 3. Restricción y Jacobiano
        def restriccion(alpha: NDArray[np.float64]) -> np.float64:
            return np.dot(alpha, y_float)

        def jac_restriccion(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
            return y_float

        cons = {'type': 'eq', 'fun': restriccion, 'jac': jac_restriccion}
        bounds = [(0.0, self.C) for _ in range(n_samples)]
        x0 = np.zeros(n_samples, dtype=np.float64)

        # 4. Solución
        resultado: OptimizeResult = minimize(
            fun=objetivo,
            x0=x0,
            method='SLSQP',
            jac=grad_objetivo,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-6, 'disp': False}
        )

        if not resultado.success:
            print(f"  [!] Advertencia SLSQP: {resultado.message}")

        self.alpha = resultado.x
        self.alpha[self.alpha < self.tol] = 0.0

        self.X_fit = X
        self.y_fit = y_float
        self.is_fit = True

        # 5. Identificación KKT en Dual (Evolucionado desde Hard Margin)
        self.sv_mask = self.alpha > self.tol

        # 6. Intercepto b (Vectores estrictamente en el margen)
        margin_sv = (self.alpha > self.tol) & (self.alpha < self.C - self.tol)
        if np.any(margin_sv):
            idx = np.where(margin_sv)[0]
            b_vals = y_float[idx] - np.sum((self.alpha * y_float)[:, np.newaxis] * K[:, idx], axis=0)
            self.b = float(np.mean(b_vals))
        elif np.any(self.sv_mask):
            idx = np.where(self.sv_mask)[0]
            b_vals = y_float[idx] - np.sum((self.alpha * y_float)[:, np.newaxis] * K[:, idx], axis=0)
            self.b = float(np.mean(b_vals))
        else:
            self.b = 0.0
            
        return resultado.success

    def decision_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Proyección no lineal f(x) = sum alpha_i y_i K(x, x_i) + b"""
        if not self.is_fit:
            raise ValueError("El modelo no ha sido entrenado.")
        K_pred = self.kernel(X, self.X_fit)
        return np.dot(K_pred, self.alpha * self.y_fit) + self.b

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int8]:
        return np.sign(self.decision_function(X)).astype(np.int8)


# =============================================================================
# 4. VISUALIZACIÓN CONSISTENTE (Mapeo Topológico)
# =============================================================================

def graficar_resultados_kernel(
    modelo: SVMDualKernel,
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    titulo: str,
    filename: str
) -> None:
    """Conserva la firma visual de las iteraciones previas (Hard/Soft Margin)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))

    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.decision_function(X_grid).reshape(xx.shape)

    # Regiones
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#F44336', '#2196F3'], alpha=0.15)
    
    # Fronteras y márgenes
    contour = ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                         linestyles=['--', '-', '--'],
                         colors=['#F44336', 'k', '#2196F3'],
                         linewidths=[1.5, 2.5, 1.5])
    ax.clabel(contour, inline=True, fontsize=10, fmt={-1: 'Margen -1', 0: 'Frontera 0', 1: 'Margen +1'})

    # Puntos originales - Mantiene firma de colores del autor
    colores: dict[int, str] = {1: '#2196F3', -1: '#F44336'}
    marcadores: dict[int, str] = {1: 'o', -1: 's'}
    labels: dict[int, str] = {1: 'Clase +1 (Anillo Ext)', -1: 'Clase -1 (Centro)'}

    for clase in [1, -1]:
        mask = (y == clase)
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colores[clase], marker=marcadores[clase],
                   edgecolors='k', linewidth=0.5, s=65, label=labels[clase], zorder=3)

    # Resaltar Vectores de Soporte con el estándar oro definido en N-1
    if modelo.is_fit:
        sv_idx = np.where(modelo.sv_mask)[0]
        if len(sv_idx) > 0:
            ax.scatter(X[sv_idx, 0], X[sv_idx, 1],
                       s=180, facecolors='none', edgecolors='#FFD600',
                       linewidth=2.5, zorder=4, label=f'Vectores de Soporte ({len(sv_idx)})')

    ax.set_title(titulo, fontsize=15, fontweight='bold')
    ax.set_xlabel('Característica $x_1$', fontsize=12)
    ax.set_ylabel('Característica $x_2$', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout()
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, 'assets', 'dual_kernel')
    os.makedirs(assets_dir, exist_ok=True)
    ruta_guardado = os.path.join(assets_dir, filename)
    plt.savefig(ruta_guardado, dpi=150, bbox_inches='tight')
    print(f"  [+] Gráfico exportado a: {filename}")
    plt.close()


# =============================================================================
# 5. EJECUCIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("  SVM DUAL KERNELIZADO — ANILLOS CONCÉNTRICOS")
    print("=" * 70)

    X, y = generar_datos_anillos(n_muestras=250, ruido=0.3)

    # -------------------------------------------------------------------------
    # Prueba 1: Kernel Gaussiano (RBF)
    # -------------------------------------------------------------------------
    print("\n[Escenario 1] Kernel Gaussiano (RBF)")
    gamma_rbf = 0.5
    C_param = 10.0
    svm_rbf = SVMDualKernel(C=C_param, kernel=Kernels.gaussian(gamma=gamma_rbf))
    svm_rbf.fit(X, y)

    print(f"  Vectores de soporte : {np.sum(svm_rbf.sv_mask)}")
    print(f"  Intercepto (b)      : {svm_rbf.b:.4f}")
    
    graficar_resultados_kernel(
        svm_rbf, X, y,
        f'SVM Dual - Kernel Gaussiano (RBF)\nC={C_param}, γ={gamma_rbf}',
        'reporte_visual_rbf.png'
    )

    # -------------------------------------------------------------------------
    # Prueba 2: Kernel Polinomial
    # -------------------------------------------------------------------------
    print("\n[Escenario 2] Kernel Polinomial")
    grado = 2
    constante = 1.0
    svm_poly = SVMDualKernel(C=C_param, kernel=Kernels.polynomial(degree=grado, c=constante))
    svm_poly.fit(X, y)

    print(f"  Vectores de soporte : {np.sum(svm_poly.sv_mask)}")
    print(f"  Intercepto (b)      : {svm_poly.b:.4f}")

    graficar_resultados_kernel(
        svm_poly, X, y,
        f'SVM Dual - Kernel Polinomial\nC={C_param}, grado={grado}, c={constante}',
        'reporte_visual_poly.png'
    )
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, 'assets', 'dual_kernel')
    print("\n" + "=" * 60)
    print(f"  [✓] Finalizado exitosamente. Revisa las imágenes generadas en:\n      {assets_dir}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
