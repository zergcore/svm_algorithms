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
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult


# =============================================================================
# 1. GENERACIÓN DE DATOS
# =============================================================================

def generar_datos(
    separables: bool = True,
    n_muestras: int = 50,
    semilla: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Genera un dataset sintético bidimensional con dos clases etiquetadas {-1, +1}."""
    rng: np.random.Generator = np.random.default_rng(semilla)

    if separables:
        centros: NDArray[np.float64] = np.array([
            [-5.0, -5.0],  # Centro clase -1
            [ 5.0,  5.0],  # Centro clase +1
        ])
        desv_std: float = 1.2
    else:
        centros: NDArray[np.float64] = np.array([
            [-1.0, -1.0],  # Centro clase -1
            [ 1.0,  1.0],  # Centro clase +1
        ])
        desv_std: float = 3.5
        
    n_por_clase: int = n_muestras // 2
    n_resto: int = n_muestras - 2 * n_por_clase

    X_clase_neg: NDArray[np.float64] = rng.normal(
        loc=centros[0], scale=desv_std, size=(n_por_clase, 2)
    )
    X_clase_pos: NDArray[np.float64] = rng.normal(
        loc=centros[1], scale=desv_std, size=(n_por_clase + n_resto, 2)
    )

    X: NDArray[np.float64] = np.vstack([X_clase_neg, X_clase_pos])
    y: NDArray[np.int8] = np.concatenate([
        -np.ones(n_por_clase, dtype=np.int8),
         np.ones(n_por_clase + n_resto, dtype=np.int8),
    ])

    idx: NDArray[np.intp] = rng.permutation(n_muestras)
    return X[idx], y[idx]


# =============================================================================
# 2. MODELO DE OPTIMIZACIÓN — HARD-MARGIN SVM (Primal OOP)
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
        self.resultado_optimizacion: OptimizeResult | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8]) -> bool:
        """Entrena el hiperplano usando SLSQP."""
        n_samples, n_features = X.shape
        y_float: NDArray[np.float64] = y.astype(np.float64)
        yX: NDArray[np.float64] = y_float[:, np.newaxis] * X

        def objetivo(params: NDArray[np.float64]) -> np.float64:
            w: NDArray[np.float64] = params[:n_features]
            return np.float64(0.5 * np.dot(w, w))

        def gradiente_objetivo(params: NDArray[np.float64]) -> NDArray[np.float64]:
            grad: NDArray[np.float64] = np.zeros_like(params)
            grad[:n_features] = params[:n_features]
            return grad

        def restriccion(params: NDArray[np.float64]) -> NDArray[np.float64]:
            w: NDArray[np.float64] = params[:n_features]
            b: np.float64 = params[-1]
            return y_float * (X @ w + b) - 1.0

        def jacobiano_restriccion(params: NDArray[np.float64]) -> NDArray[np.float64]:
            jac: NDArray[np.float64] = np.column_stack([yX, y_float])
            return jac

        cons: dict = {
            'type': 'ineq',
            'fun': restriccion,
            'jac': jacobiano_restriccion,
        }

        x0: NDArray[np.float64] = np.zeros(n_features + 1, dtype=np.float64)
        bounds: list[tuple[float, float]] = [(-1e3, 1e3) for _ in range(n_features + 1)]

        self.resultado_optimizacion = minimize(
            fun=objetivo,
            x0=x0,
            method='SLSQP',
            jac=gradiente_objetivo,
            bounds=bounds,
            constraints=cons,
            options={'ftol': self.tol, 'maxiter': self.maxiter, 'disp': False},
        )

        if self.resultado_optimizacion.success:
            self.w = self.resultado_optimizacion.x[:n_features]
            self.b = float(self.resultado_optimizacion.x[-1])
            self.X_fit = X
            self.y_fit = y
            self.is_fit = True
            
            # Identificación de Vectores de Soporte (KKT)
            margen_funcional = y * (X @ self.w + self.b)
            self.sv_mask = np.abs(margen_funcional - 1.0) < self.tol_kkt
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
# 3. VISUALIZACIÓN ESTRUCTURADA
# =============================================================================

def graficar_resultados(
    modelo: SVMHardMargin,
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    titulo: str,
    nombre_archivo: str | None = None,
) -> None:
    """Conserva el flujo de graficación base (Sello del desarrollador)."""
    fig, ax = plt.subplots(figsize=(9, 7))

    colores: dict[int, str] = {1: '#2196F3', -1: '#F44336'}
    marcadores: dict[int, str] = {1: 'o', -1: 's'}

    # Graficar dispersión de datos original
    for clase in [1, -1]:
        mascara: NDArray[np.bool_] = (y == clase)
        ax.scatter(
            X[mascara, 0], X[mascara, 1],
            c=colores[clase], marker=marcadores[clase],
            edgecolors='k', linewidth=0.5, s=60,
            label=f'Clase {clase:+d}', alpha=0.85, zorder=3,
        )

    if modelo.is_fit:
        w = modelo.w
        b = modelo.b

        # Destacar Vectores de Soporte
        sv_mask = modelo.sv_mask
        n_sv: int = int(np.sum(sv_mask))

        if n_sv > 0:
            ax.scatter(
                X[sv_mask, 0], X[sv_mask, 1],
                s=200, facecolors='none', edgecolors='#FFD600',
                linewidth=2.5, zorder=4, label=f'Vectores de Soporte ({n_sv})',
            )

        # Límites del hiperplano
        x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
        x_plot: NDArray[np.float64] = np.linspace(x_min, x_max, 300)

        if np.abs(w[1]) > 1e-12:
            y_sep: NDArray[np.float64] = -(w[0] * x_plot + b) / w[1]
            y_pos: NDArray[np.float64] = -(w[0] * x_plot + b - 1) / w[1]
            y_neg: NDArray[np.float64] = -(w[0] * x_plot + b + 1) / w[1]

            ax.plot(x_plot, y_sep, 'k-', linewidth=2.0, label='Hiperplano Separador', zorder=2)
            ax.plot(x_plot, y_pos, '--', color='#2196F3', linewidth=1.2, label='Margen +1', alpha=0.7)
            ax.plot(x_plot, y_neg, '--', color='#F44336', linewidth=1.2, label='Margen −1', alpha=0.7)

            ax.fill_between(x_plot, y_pos, y_neg, alpha=0.08, color='#9E9E9E', zorder=1)

            y_data_min, y_data_max = X[:, 1].min() - 2, X[:, 1].max() + 2
            ax.set_ylim(y_data_min, y_data_max)
        else:
            x_vert: float = -b / w[0]
            ax.axvline(x=x_vert, color='k', linewidth=2.0, label='Hiperplano Separador')

        margen_geometrico: float = 2.0 / np.linalg.norm(w)
        info_text: str = (
            f'w = [{w[0]:.4f}, {w[1]:.4f}]\n'
            f'b = {b:.4f}\n'
            f'Margen = {margen_geometrico:.4f}\n'
            f'||w|| = {np.linalg.norm(w):.4f}\n'
            f'Vectores de Soporte = {n_sv}'
        )
        ax.text(
            0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', fontfamily='monospace', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9), zorder=5,
        )
    else:
        mensaje_error = "Optimización Fallida"
        if modelo.resultado_optimizacion:
            mensaje_error += f"\n{modelo.resultado_optimizacion.message}"
        
        ax.text(
            0.5, 0.5, mensaje_error, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=14, color='white',
            bbox=dict(facecolor='darkred', alpha=0.85, boxstyle='round,pad=0.8'),
        )

    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel('Característica $x_1$', fontsize=11)
    ax.set_ylabel('Característica $x_2$', fontsize=11)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    fig.tight_layout()
    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"\n[+] Gráfico exportado a: {nombre_archivo}")
    plt.close()


# =============================================================================
# 4. EJECUCIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, 'assets', 'hard_margin')
    os.makedirs(assets_dir, exist_ok=True)
    print("=" * 60)
    print("  PARTE 1: DATOS LINEALMENTE SEPARABLES")
    print("=" * 60)

    X_sep, y_sep = generar_datos(separables=True)
    modelo_sep = SVMHardMargin()
    exito_sep = modelo_sep.fit(X_sep, y_sep)

    print(f"  Estado del solver : {modelo_sep.resultado_optimizacion.message}")
    print(f"  Convergió         : {exito_sep}")

    if exito_sep:
        w_sep = modelo_sep.w
        b_sep = modelo_sep.b
        margen_sep = 2.0 / np.linalg.norm(w_sep)
        n_sv = int(np.sum(modelo_sep.sv_mask))

        print(f"  w                 : [{w_sep[0]:.6f}, {w_sep[1]:.6f}]")
        print(f"  b                 : {b_sep:.6f}")
        print(f"  ||w||             : {np.linalg.norm(w_sep):.6f}")
        print(f"  Margen geométrico : {margen_sep:.6f}")
        print(f"  Vectores soporte  : {n_sv}")

    ruta_sep = os.path.join(assets_dir, "hard_margin_separables.png")
    graficar_resultados(modelo_sep, X_sep, y_sep, "SVM Hard-Margin — Datos Linealmente Separables", ruta_sep)

    print("\n" + "=" * 60)
    print("  PARTE 2: DATOS NO SEPARABLES (SUPERPUESTOS)")
    print("=" * 60)

    X_nosep, y_nosep = generar_datos(separables=False)
    modelo_nosep = SVMHardMargin()
    exito_nosep = modelo_nosep.fit(X_nosep, y_nosep)

    print(f"  Estado del solver : {modelo_nosep.resultado_optimizacion.message}")
    print(f"  Convergió         : {exito_nosep}")

    if not exito_nosep:
        print("  → El Hard-Margin SVM no puede resolver datos no separables.")
        print("    Las restricciones y_i(wᵀx_i + b) ≥ 1 son incompatibles.")

    ruta_nosep = os.path.join(assets_dir, "hard_margin_no_separables.png")
    graficar_resultados(modelo_nosep, X_nosep, y_nosep, "SVM Hard-Margin — Datos No Separables (Superpuestos)", ruta_nosep)

    print("\n" + "=" * 60)
    print(f"  [✓] Finalizado exitosamente. Revisa las imágenes generadas en:\n      {assets_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()