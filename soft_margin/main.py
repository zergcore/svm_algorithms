"""
Support Vector Machine — Soft-Margin Dual Formulation & Kernel Trick
====================================================================

Formulación Dual:
    min_{α}  (1/2) Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j) - Σ_i α_i
    s.t.     Σ_i α_i y_i = 0
             0 ≤ α_i ≤ C,  ∀ i

Condiciones KKT y variables de holgura (ξ):
    - Si ξ_i = 0 ∀i, el problema es linealmente separable en el espacio de
      características inducido por el kernel (el margen es respetado).
    - Si existe algún ξ_i > 0, el punto viola el margen (está dentro del margen
      o mal clasificado). El modelo Soft-Margin permite estas violaciones
      penalizándolas mediante el hiperparámetro C.
"""

from __future__ import annotations
import os
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult

from core.datasets import generate_nonlinear_data
from core.kernels import Kernels
from core.visualization import plot_soft_results


# =============================================================================
# 1. MODELO DE OPTIMIZACIÓN — SOFT-MARGIN SVM (Dual)
# =============================================================================

class SVMDual:
    def __init__(self, C: float = 1.0, kernel: callable = None, tol: float = 1e-4):
        self.C = C
        self.kernel = kernel if kernel is not None else Kernels.linear()
        self.tol = tol
        self.alpha = None
        self.b = 0.0
        self.X_fit = None
        self.y_fit = None
        self.sv_mask = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8], maxiter: int = 1500) -> None:
        n_samples = X.shape[0]
        y_float = y.astype(np.float64)
        
        # 1. Matriz de Gram K
        K = self.kernel(X, X)
        
        # 2. Matriz Q = (y_i y_j K_ij)
        Q = np.outer(y_float, y_float) * K
        
        # Objetivo: f(alpha) = 0.5 * alpha^T Q alpha - sum(alpha)
        def objective(alpha: NDArray[np.float64]) -> np.float64:
            return np.float64(0.5 * np.dot(alpha, Q @ alpha) - np.sum(alpha))
            
        def objective_gradient(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
            return Q @ alpha - 1.0
            
        # Restricción: sum(alpha_i y_i) = 0
        def constraint(alpha: NDArray[np.float64]) -> np.float64:
            return np.dot(alpha, y_float)
            
        def constraint_jacobian(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
            return y_float
            
        cons = {'type': 'eq', 'fun': constraint, 'jac': constraint_jacobian}
        # Cota para Soft-Margin: 0 <= alpha_i <= C
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        x0 = np.zeros(n_samples, dtype=np.float64)
        
        result: OptimizeResult = minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',
            jac=objective_gradient,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-8, 'disp': False}
        )
        
        if not result.success:
            print(f"  [!] Advertencia SLSQP: {result.message}")
            
        self.alpha = result.x
        # Forzar exactamente a 0 los alpha menores que la tolerancia
        self.alpha[self.alpha < self.tol] = 0.0
        
        self.X_fit = X
        self.y_fit = y_float
        
        # Identificación de Vectores de Soporte (alpha > 0)
        self.sv_mask = self.alpha > 0
        
        # Calcular 'b' usando los Margin Support Vectors (0 < alpha < C)
        margin_sv = (self.alpha > self.tol) & (self.alpha < self.C - self.tol)
        
        if np.any(margin_sv):
            idx = np.where(margin_sv)[0]
            b_vals = y_float[idx] - np.sum(
                (self.alpha * y_float)[:, np.newaxis] * K[:, idx], axis=0
            )
            self.b = float(np.mean(b_vals))
        else:
            if np.any(self.sv_mask):
                idx = np.where(self.sv_mask)[0]
                b_vals = y_float[idx] - np.sum(
                    (self.alpha * y_float)[:, np.newaxis] * K[:, idx], axis=0
                )
                self.b = float(np.mean(b_vals))
            else:
                self.b = 0.0

    def decision_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calcula el margen funcional crudo: f(x) = sum alpha_i y_i K(x, x_i) + b"""
        K_pred = self.kernel(X, self.X_fit)
        return np.dot(K_pred, self.alpha * self.y_fit) + self.b
        
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int8]:
        return np.sign(self.decision_function(X)).astype(np.int8)
        
    def calculate_slack(self, X: NDArray[np.float64], y: NDArray[np.int8]) -> NDArray[np.float64]:
        """Calcula ξ_i = max(0, 1 - y_i f(x_i)). Si ξ_i > 0, el punto viola el margen."""
        df = self.decision_function(X)
        xi = np.maximum(0, 1 - y * df)
        return xi


# =============================================================================
# 2. EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, 'assets', 'soft_margin')
    os.makedirs(assets_dir, exist_ok=True)
    
    print("=" * 70)
    print("  SOFT-MARGIN SVM Y KERNEL TRICK — ALGORITMO DUAL")
    print("=" * 70)
    
    # Prueba 1: Gaussiano
    print("\n[Escenario 1] Dataset 'Moons' con Kernel Gaussiano (RBF)")
    X_moons, y_moons = generate_nonlinear_data(type_name='moons', n_samples=150, noise=0.1)
    
    svm_rbf = SVMDual(C=5.0, kernel=Kernels.gaussian(gamma=2.0))
    svm_rbf.fit(X_moons, y_moons)
    
    xi_rbf = svm_rbf.calculate_slack(X_moons, y_moons)
    is_linearly_separable_rbf = np.all(xi_rbf < 1e-4)
    print(f"  Vectores de soporte : {np.sum(svm_rbf.sv_mask)}")
    print(f"  Masa total de ξ     : {np.sum(xi_rbf):.4f}")
    print(f"  Separabilidad en ϕ  : {'Sí (ξ_i = 0 ∀i)' if is_linearly_separable_rbf else 'No (existen ξ_i > 0)'}")
    
    plot_soft_results(svm_rbf, X_moons, y_moons, "SVM Soft-Margin — Kernel Gaussiano (Moons)", assets_dir)
    
    # Prueba 2: Polinomial
    print("\n[Escenario 2] Dataset 'Circles' con Kernel Polinomial")
    X_circ, y_circ = generate_nonlinear_data(type_name='circles', n_samples=150, noise=0.1)
    
    svm_poly = SVMDual(C=1.0, kernel=Kernels.polynomial(degree=2, c=1.0))
    svm_poly.fit(X_circ, y_circ)
    
    xi_poly = svm_poly.calculate_slack(X_circ, y_circ)
    is_linearly_separable_poly = np.all(xi_poly < 1e-4)
    print(f"  Vectores de soporte : {np.sum(svm_poly.sv_mask)}")
    print(f"  Masa total de ξ     : {np.sum(xi_poly):.4f}")
    print(f"  Separabilidad en ϕ  : {'Sí (ξ_i = 0 ∀i)' if is_linearly_separable_poly else 'No (existen ξ_i > 0)'}")
    
    plot_soft_results(svm_poly, X_circ, y_circ, "SVM Soft-Margin — Kernel Polinomial (Circles)", assets_dir)

    # Prueba 3: Lineal
    print("\n[Escenario 3] Demostración: Fracaso del Kernel Lineal en 'Moons'")
    svm_lin = SVMDual(C=1.0, kernel=Kernels.linear())
    svm_lin.fit(X_moons, y_moons)
    
    xi_lin = svm_lin.calculate_slack(X_moons, y_moons)
    is_linearly_separable_lin = np.all(xi_lin < 1e-4)
    print(f"  Vectores de soporte : {np.sum(svm_lin.sv_mask)}")
    print(f"  Masa total de ξ     : {np.sum(xi_lin):.4f}")
    print(f"  Separabilidad en X  : {'Sí (ξ_i = 0 ∀i)' if is_linearly_separable_lin else 'No (existen ξ_i > 0)'}")
    print("  -> El modelo penaliza enormemente, pero el hiperplano es recto en 2D.")
    
    plot_soft_results(svm_lin, X_moons, y_moons, "SVM Soft-Margin — Kernel Lineal (Moons)", assets_dir)
    
    # Prueba 4: Sigmoide
    print("\n[Escenario 4] Dataset 'Moons' con Kernel Sigmoide")
    svm_sig = SVMDual(C=1.0, kernel=Kernels.sigmoid(gamma=0.1, r=0.0))
    svm_sig.fit(X_moons, y_moons)
    
    xi_sig = svm_sig.calculate_slack(X_moons, y_moons)
    is_linearly_separable_sig = np.all(xi_sig < 1e-4)
    print(f"  Vectores de soporte : {np.sum(svm_sig.sv_mask)}")
    print(f"  Masa total de ξ     : {np.sum(xi_sig):.4f}")
    print(f"  Separabilidad en ϕ  : {'Sí (ξ_i = 0 ∀i)' if is_linearly_separable_sig else 'No (existen ξ_i > 0)'}")
    
    plot_soft_results(svm_sig, X_moons, y_moons, "SVM Soft-Margin — Kernel Sigmoide (Moons)", assets_dir)
    
    print("\n" + "=" * 60)
    print(f"  [✓] Finalizado exitosamente. Revisa las imágenes generadas en:\n      {assets_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
