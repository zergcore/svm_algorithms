"""
Módulo 3: Kernel Trick (Formulación Dual)
=========================================

Esta es una iteración similar a Soft-Margin pero con un enfoque estrictamente
dedicado a demostrar el poder del Kernel Gaussiano (RBF) y Polinomial
sobre distribuciones concéntricas y cruzadas.
"""

from __future__ import annotations
import os
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult

from core.datasets import generate_ring_data
from core.kernels import Kernels
from core.visualization import plot_kernel_results

class SVMDualKernel:
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

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8], maxiter: int = 1500) -> bool:
        n_samples = X.shape[0]
        y_float = y.astype(np.float64)
        
        K = self.kernel(X, X)
        Q = np.outer(y_float, y_float) * K
        
        def objective(alpha: NDArray[np.float64]) -> np.float64:
            return np.float64(0.5 * np.dot(alpha, Q @ alpha) - np.sum(alpha))
            
        def objective_gradient(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
            return Q @ alpha - 1.0
            
        def constraint(alpha: NDArray[np.float64]) -> np.float64:
            return np.dot(alpha, y_float)
            
        def constraint_jacobian(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
            return y_float
            
        cons = {'type': 'eq', 'fun': constraint, 'jac': constraint_jacobian}
        bounds = [(0, self.C) for _ in range(n_samples)]
        x0 = np.zeros(n_samples, dtype=np.float64)
        
        result: OptimizeResult = minimize(
            fun=objective, x0=x0, method='SLSQP', jac=objective_gradient,
            bounds=bounds, constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-8, 'disp': False}
        )
        
        if not result.success:
            print(f"  [!] Advertencia SLSQP: {result.message}")
            return False
            
        self.alpha = result.x
        self.alpha[self.alpha < self.tol] = 0.0
        
        self.X_fit = X
        self.y_fit = y_float
        self.sv_mask = self.alpha > 0
        self.is_fit = True
        
        margin_sv = (self.alpha > self.tol) & (self.alpha < self.C - self.tol)
        if np.any(margin_sv):
            idx = np.where(margin_sv)[0]
            b_vals = y_float[idx] - np.sum(
                (self.alpha * y_float)[:, np.newaxis] * K[:, idx], axis=0
            )
            self.b = float(np.mean(b_vals))
        elif np.any(self.sv_mask):
            idx = np.where(self.sv_mask)[0]
            b_vals = y_float[idx] - np.sum(
                (self.alpha * y_float)[:, np.newaxis] * K[:, idx], axis=0
            )
            self.b = float(np.mean(b_vals))
        else:
            self.b = 0.0
            
        return True

    def decision_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fit:
            raise ValueError("El modelo no está entrenado.")
        K_pred = self.kernel(X, self.X_fit)
        return np.dot(K_pred, self.alpha * self.y_fit) + self.b
        
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int8]:
        return np.sign(self.decision_function(X)).astype(np.int8)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, 'assets', 'dual_kernel')
    os.makedirs(assets_dir, exist_ok=True)
    
    print("=" * 60)
    print("  KRNEL TRICK — DATOS NO SEPARABLES (ANILLOS)")
    print("=" * 60)
    
    X, y = generate_ring_data(n_samples=200, noise=0.15)
    
    # ---------------------------------------------------------
    # 1. Kernel Gaussiano (RBF)
    # ---------------------------------------------------------
    C_param = 5.0
    gamma_param = 1.0
    print(f"\n[Escenario 1] Kernel Gaussiano (RBF) - C={C_param}, gamma={gamma_param}")
    
    svm_rbf = SVMDualKernel(C=C_param, kernel=Kernels.gaussian(gamma=gamma_param))
    svm_rbf.fit(X, y)
    
    print(f"  Vectores de soporte : {np.sum(svm_rbf.sv_mask)}")
    print(f"  Intercepto (b)      : {svm_rbf.b:.4f}")

    plot_kernel_results(
        svm_rbf, X, y,
        f'SVM Dual - Kernel Gaussiano (RBF)\nC={C_param}, $\gamma$={gamma_param}',
        assets_dir,
        'reporte_visual_rbf.png'
    )

    # ---------------------------------------------------------
    # 2. Kernel Polinomial
    # ---------------------------------------------------------
    degree_param = 2
    constant_param = 1.0
    C_param = 1.0
    print(f"\n[Escenario 2] Kernel Polinomial - grado={degree_param}, c={constant_param}")
    
    svm_poly = SVMDualKernel(C=C_param, kernel=Kernels.polynomial(degree=degree_param, c=constant_param))
    svm_poly.fit(X, y)
    
    print(f"  Vectores de soporte : {np.sum(svm_poly.sv_mask)}")
    print(f"  Intercepto (b)      : {svm_poly.b:.4f}")

    plot_kernel_results(
        svm_poly, X, y,
        f'SVM Dual - Kernel Polinomial\nC={C_param}, grado={degree_param}, c={constant_param}',
        assets_dir,
        'reporte_visual_poly.png'
    )
    
    print("\n" + "=" * 60)
    print(f"  [✓] Finalizado exitosamente. Revisa las imágenes generadas en:\n      {assets_dir}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
