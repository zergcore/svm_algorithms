"""
Support Vector Machine — Hard-Margin Primal Formulation
========================================================

Formulación matemática:
    min_{w, b}  (1/2) ||w||²

    s.t.  y_i (wᵀ x_i + b) ≥ 1,  ∀ i = 1, ..., n

Solver: SLSQP (Sequential Least Squares Quadratic Programming)
    - Gradiente analítico del objetivo:  ∇_w f = w,  ∂f/∂b = 0
    - Jacobiano analítico de restricciones:  ∂g_i/∂w_j = y_i · x_{ij},  ∂g_i/∂b = y_i
"""

from __future__ import annotations

import os
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult

from core.datasets import generate_linear_data
from core.visualization import plot_results

# =============================================================================
# 1. MODELO DE OPTIMIZACIÓN — HARD-MARGIN SVM (Primal OOP)
# =============================================================================

class SVMHardMargin:
    """Implementación Orientada a Objetos del SVM Hard-Margin (Formulación Primal)."""
    def __init__(self, tol: float = 1e-8, maxiter: int = 500, tol_kkt: float = 1e-4):
        self.tol = tol
        self.maxiter = maxiter
        self.tol_kkt = tol_kkt
        self.w = None
        self.b = 0.0
        self.X_fit = None
        self.y_fit = None
        self.sv_mask = None
        self.is_fit = False
        self.optimization_result: OptimizeResult | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8]) -> bool:
        """Entrena el hiperplano usando SLSQP."""
        n_samples, n_features = X.shape
        y_float: NDArray[np.float64] = y.astype(np.float64)
        yX: NDArray[np.float64] = y_float[:, np.newaxis] * X

        def objective(params: NDArray[np.float64]) -> np.float64:
            w: NDArray[np.float64] = params[:n_features]
            return np.float64(0.5 * np.dot(w, w))

        def objective_gradient(params: NDArray[np.float64]) -> NDArray[np.float64]:
            grad: NDArray[np.float64] = np.zeros_like(params)
            grad[:n_features] = params[:n_features]
            return grad

        def constraint(params: NDArray[np.float64]) -> NDArray[np.float64]:
            w: NDArray[np.float64] = params[:n_features]
            b: np.float64 = params[-1]
            return y_float * (X @ w + b) - 1.0

        def constraint_jacobian(params: NDArray[np.float64]) -> NDArray[np.float64]:
            jac: NDArray[np.float64] = np.column_stack([yX, y_float])
            return jac

        cons: dict = {
            'type': 'ineq',
            'fun': constraint,
            'jac': constraint_jacobian,
        }

        x0: NDArray[np.float64] = np.zeros(n_features + 1, dtype=np.float64)
        bounds: list[tuple[float, float]] = [(-1e3, 1e3) for _ in range(n_features + 1)]

        self.optimization_result = minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',
            jac=objective_gradient,
            bounds=bounds,
            constraints=cons,
            options={'ftol': self.tol, 'maxiter': self.maxiter, 'disp': False},
        )

        if self.optimization_result.success:
            self.w = self.optimization_result.x[:n_features]
            self.b = float(self.optimization_result.x[-1])
            self.X_fit = X
            self.y_fit = y
            self.is_fit = True
            
            # Identificación de Vectores de Soporte (KKT)
            functional_margin = y * (X @ self.w + self.b)
            self.sv_mask = np.abs(functional_margin - 1.0) < self.tol_kkt
            return True
            
        return False

    def decision_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calcula el margen funcional crudo f(x) = wᵀx + b."""
        if not self.is_fit:
            raise ValueError("El modelo no ha sido entrenado.")
        return X @ self.w + self.b

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int8]:
        """Proyecta la predicción al espacio de etiquetas {-1, +1}."""
        return np.sign(self.decision_function(X)).astype(np.int8)


# =============================================================================
# 2. EJECUCIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, 'assets', 'hard_margin')
    os.makedirs(assets_dir, exist_ok=True)
    print("=" * 60)
    print("  PARTE 1: DATOS LINEALMENTE SEPARABLES")
    print("=" * 60)

    X_sep, y_sep = generate_linear_data(separable=True)
    model_sep = SVMHardMargin()
    success_sep = model_sep.fit(X_sep, y_sep)

    print(f"  Estado del solver : {model_sep.optimization_result.message}")
    print(f"  Convergió         : {success_sep}")

    if success_sep:
        w_sep = model_sep.w
        b_sep = model_sep.b
        margin_sep = 2.0 / np.linalg.norm(w_sep)
        n_sv = int(np.sum(model_sep.sv_mask))

        print(f"  w                 : [{w_sep[0]:.6f}, {w_sep[1]:.6f}]")
        print(f"  b                 : {b_sep:.6f}")
        print(f"  ||w||             : {np.linalg.norm(w_sep):.6f}")
        print(f"  Margen geométrico : {margin_sep:.6f}")
        print(f"  Vectores soporte  : {n_sv}")

    sep_path = os.path.join(assets_dir, "hard_margin_separables.png")
    plot_results(model_sep, X_sep, y_sep, "SVM Hard-Margin — Datos Linealmente Separables", sep_path)

    print("\n" + "=" * 60)
    print("  PARTE 2: DATOS NO SEPARABLES (SUPERPUESTOS)")
    print("=" * 60)

    X_nosep, y_nosep = generate_linear_data(separable=False)
    model_nosep = SVMHardMargin()
    success_nosep = model_nosep.fit(X_nosep, y_nosep)

    print(f"  Estado del solver : {model_nosep.optimization_result.message}")
    print(f"  Convergió         : {success_nosep}")

    if not success_nosep:
        print("  → El Hard-Margin SVM no puede resolver datos no separables.")
        print("    Las restricciones y_i(wᵀx_i + b) ≥ 1 son incompatibles.")

    nosep_path = os.path.join(assets_dir, "hard_margin_no_separables.png")
    plot_results(model_nosep, X_nosep, y_nosep, "SVM Hard-Margin — Datos No Separables (Superpuestos)", nosep_path)

    print("\n" + "=" * 60)
    print(f"  [✓] Finalizado exitosamente. Revisa las imágenes generadas en:\n      {assets_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()