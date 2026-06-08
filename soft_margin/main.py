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

Kernels implementados:
    - Lineal: K(x, x') = xᵀ x'
    - Gaussiano (RBF): K(x, x') = exp(-γ ||x - x'||²)
    - Polinomial: K(x, x') = (xᵀ x' + r)ᵈ
    - Sigmoide: K(x, x') = tanh(γ xᵀ x' + r)

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
    """Implementaciones vectorizadas de funciones Kernel."""

    @staticmethod
    def linear() -> callable:
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            return X1 @ X2.T
        return calc

    @staticmethod
    def gaussian(gamma: float = 1.0) -> callable:
        """Kernel Gaussiano (Radial Basis Function)."""
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            # ||X1 - X2||^2 = ||X1||^2 + ||X2||^2 - 2 X1 @ X2.T
            sq_norm1 = np.sum(X1**2, axis=1, keepdims=True)
            sq_norm2 = np.sum(X2**2, axis=1)
            dist_sq = sq_norm1 + sq_norm2 - 2 * (X1 @ X2.T)
            # Clip en 0 por inestabilidad numérica
            dist_sq = np.clip(dist_sq, 0, None)
            return np.exp(-gamma * dist_sq)
        return calc

    @staticmethod
    def polynomial(degree: float = 3, r: float = 1.0) -> callable:
        """Kernel Polinomial (homogéneo si r=0, no homogéneo si r>0)."""
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            return (X1 @ X2.T + r) ** degree
        return calc

    @staticmethod
    def sigmoid(gamma: float = 1.0, r: float = 0.0) -> callable:
        """Kernel Sigmoide."""
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.tanh(gamma * (X1 @ X2.T) + r)
        return calc


# =============================================================================
# 2. GENERACIÓN DE DATOS (No linealmente separables)
# =============================================================================

def generar_datos_no_lineales(
    tipo: str = 'moons',
    n_muestras: int = 200,
    ruido: float = 0.1,
    semilla: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Genera distribuciones clásicas no separables linealmente ('moons' o 'circles')."""
    rng = np.random.default_rng(semilla)
    
    n_out = n_muestras // 2
    n_in = n_muestras - n_out

    if tipo == 'moons':
        t_out = np.linspace(0, np.pi, n_out)
        t_in = np.linspace(0, np.pi, n_in)
        
        X_out = np.column_stack([np.cos(t_out), np.sin(t_out)])
        X_in = np.column_stack([1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5])
        
    elif tipo == 'circles':
        t_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
        t_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
        
        X_out = np.column_stack([np.cos(t_out), np.sin(t_out)])
        X_in = np.column_stack([0.5 * np.cos(t_in), 0.5 * np.sin(t_in)])
        
    else:
        raise ValueError("Tipo no soportado. Elija 'moons' o 'circles'.")

    X = np.vstack([X_out, X_in])
    y = np.concatenate([np.ones(n_out, dtype=np.int8), -np.ones(n_in, dtype=np.int8)])
    
    X += rng.normal(scale=ruido, size=X.shape)
    
    idx = rng.permutation(n_muestras)
    return X[idx], y[idx]


# =============================================================================
# 3. MODELO DE OPTIMIZACIÓN — SOFT-MARGIN SVM (Dual)
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
        def objetivo(alpha: NDArray[np.float64]) -> np.float64:
            return np.float64(0.5 * np.dot(alpha, Q @ alpha) - np.sum(alpha))
            
        def grad_objetivo(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
            return Q @ alpha - 1.0
            
        # Restricción: sum(alpha_i y_i) = 0
        def restriccion(alpha: NDArray[np.float64]) -> np.float64:
            return np.dot(alpha, y_float)
            
        def jac_restriccion(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
            return y_float
            
        cons = {'type': 'eq', 'fun': restriccion, 'jac': jac_restriccion}
        # Cota para Soft-Margin: 0 <= alpha_i <= C
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        x0 = np.zeros(n_samples, dtype=np.float64)
        
        resultado: OptimizeResult = minimize(
            fun=objetivo,
            x0=x0,
            method='SLSQP',
            jac=grad_objetivo,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-8, 'disp': False}
        )
        
        if not resultado.success:
            print(f"  [!] Advertencia SLSQP: {resultado.message}")
            
        self.alpha = resultado.x
        # Forzar exactamente a 0 los alpha menores que la tolerancia
        self.alpha[self.alpha < self.tol] = 0.0
        
        self.X_fit = X
        self.y_fit = y_float
        
        # Identificación de Vectores de Soporte (alpha > 0)
        self.sv_mask = self.alpha > 0
        
        # Calcular 'b' usando los Margin Support Vectors (0 < alpha < C)
        # Ellos satisfacen y_i(f(x_i)) = 1 exactamente.
        margin_sv = (self.alpha > self.tol) & (self.alpha < self.C - self.tol)
        
        if np.any(margin_sv):
            idx = np.where(margin_sv)[0]
            # y_i - sum_j alpha_j y_j K(x_i, x_j)
            b_vals = y_float[idx] - np.sum(
                (self.alpha * y_float)[:, np.newaxis] * K[:, idx], axis=0
            )
            self.b = float(np.mean(b_vals))
        else:
            # Fallback en caso de no encontrar puntos estrictamente en el margen
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
        
    def calcular_holguras(self, X: NDArray[np.float64], y: NDArray[np.int8]) -> NDArray[np.float64]:
        """Calcula ξ_i = max(0, 1 - y_i f(x_i)). Si ξ_i > 0, el punto viola el margen."""
        df = self.decision_function(X)
        xi = np.maximum(0, 1 - y * df)
        return xi


# =============================================================================
# 4. VISUALIZACIÓN
# =============================================================================

def graficar_resultados_soft(modelo: SVMDual, X: NDArray[np.float64], y: NDArray[np.int8], titulo: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Crear malla 2D para mapear contornos
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.decision_function(X_grid)
    Z = Z.reshape(xx.shape)
    
    # Rellenar regiones de decisión (f > 0 azul, f < 0 rojo)
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#F44336', '#2196F3'], alpha=0.15)
    
    # Dibujar márgenes (-1, 1) y frontera de decisión (0)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1],
               linestyles=['--', '-', '--'],
               colors=['#F44336', 'k', '#2196F3'],
               linewidths=[1.5, 2.5, 1.5])
    
    # Graficar puntos originales
    colores = {1: '#2196F3', -1: '#F44336'}
    marcadores = {1: 'o', -1: 's'}
    
    for clase in [1, -1]:
        mask = (y == clase)
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colores[clase], marker=marcadores[clase],
                   edgecolors='k', linewidth=0.5, s=65, label=f'Clase {clase:+d}', zorder=3)
                   
    # Resaltar Vectores de Soporte (alpha > 0)
    sv_idx = np.where(modelo.sv_mask)[0]
    if len(sv_idx) > 0:
        ax.scatter(X[sv_idx, 0], X[sv_idx, 1],
                   s=180, facecolors='none', edgecolors='#FFD600',
                   linewidth=2.5, zorder=4, label=f'Vectores de Soporte ({len(sv_idx)})')
                   
    # Marcar violaciones evidentes (ξ > 0) - puntos dentro del margen o mal clasificados
    xi = modelo.calcular_holguras(X, y)
    violaciones = xi > 1e-4
    if np.any(violaciones):
        ax.scatter(X[violaciones, 0], X[violaciones, 1],
                   s=40, color='black', marker='x',
                   linewidth=1.5, zorder=5, label=f'Violación del Margen ξ>0 ({np.sum(violaciones)})')
                   
    ax.set_title(titulo, fontsize=15, fontweight='bold')
    ax.set_xlabel('Característica $x_1$', fontsize=12)
    ax.set_ylabel('Característica $x_2$', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    # Exportar figura
    nombre_archivo = titulo.replace(" ", "_").replace("—", "-").lower() + ".png"
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, 'assets', 'soft_margin')
    os.makedirs(assets_dir, exist_ok=True)
    ruta_guardado = os.path.join(assets_dir, nombre_archivo)
    plt.savefig(ruta_guardado, dpi=150, bbox_inches='tight')
    print(f"  [+] Gráfico exportado a: {nombre_archivo}")
    plt.close()


# =============================================================================
# 5. EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    print("=" * 70)
    print("  SOFT-MARGIN SVM Y KERNEL TRICK — ALGORITMO DUAL")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Prueba 1: Gaussiano (RBF) en Dataset 'Moons'
    # -------------------------------------------------------------------------
    print("\n[Escenario 1] Dataset 'Moons' con Kernel Gaussiano (RBF)")
    X_moons, y_moons = generar_datos_no_lineales(tipo='moons', n_muestras=150, ruido=0.1)
    
    svm_rbf = SVMDual(C=5.0, kernel=Kernels.gaussian(gamma=2.0))
    svm_rbf.fit(X_moons, y_moons)
    
    xi_rbf = svm_rbf.calcular_holguras(X_moons, y_moons)
    linealmente_separable_rbf = np.all(xi_rbf < 1e-4)
    print(f"  Vectores de soporte : {np.sum(svm_rbf.sv_mask)}")
    print(f"  Masa total de ξ     : {np.sum(xi_rbf):.4f}")
    print(f"  Separabilidad en ϕ  : {'Sí (ξ_i = 0 ∀i)' if linealmente_separable_rbf else 'No (existen ξ_i > 0)'}")
    
    graficar_resultados_soft(svm_rbf, X_moons, y_moons, "SVM Soft-Margin — Kernel Gaussiano (Moons)")
    
    # -------------------------------------------------------------------------
    # Prueba 2: Polinomial en Dataset 'Circles'
    # -------------------------------------------------------------------------
    print("\n[Escenario 2] Dataset 'Circles' con Kernel Polinomial")
    X_circ, y_circ = generar_datos_no_lineales(tipo='circles', n_muestras=150, ruido=0.1)
    
    svm_poly = SVMDual(C=1.0, kernel=Kernels.polynomial(degree=2, r=1.0))
    svm_poly.fit(X_circ, y_circ)
    
    xi_poly = svm_poly.calcular_holguras(X_circ, y_circ)
    linealmente_separable_poly = np.all(xi_poly < 1e-4)
    print(f"  Vectores de soporte : {np.sum(svm_poly.sv_mask)}")
    print(f"  Masa total de ξ     : {np.sum(xi_poly):.4f}")
    print(f"  Separabilidad en ϕ  : {'Sí (ξ_i = 0 ∀i)' if linealmente_separable_poly else 'No (existen ξ_i > 0)'}")
    
    graficar_resultados_soft(svm_poly, X_circ, y_circ, "SVM Soft-Margin — Kernel Polinomial (Circles)")

    # -------------------------------------------------------------------------
    # Prueba 3: Demostración Fracaso con Kernel Lineal en 'Moons'
    # -------------------------------------------------------------------------
    print("\n[Escenario 3] Demostración: Fracaso del Kernel Lineal en 'Moons'")
    svm_lin = SVMDual(C=1.0, kernel=Kernels.linear())
    svm_lin.fit(X_moons, y_moons)
    
    xi_lin = svm_lin.calcular_holguras(X_moons, y_moons)
    linealmente_separable_lin = np.all(xi_lin < 1e-4)
    print(f"  Vectores de soporte : {np.sum(svm_lin.sv_mask)}")
    print(f"  Masa total de ξ     : {np.sum(xi_lin):.4f}")
    print(f"  Separabilidad en X  : {'Sí (ξ_i = 0 ∀i)' if linealmente_separable_lin else 'No (existen ξ_i > 0)'}")
    print("  -> El modelo penaliza enormemente, pero el hiperplano es recto en 2D.")
    
    graficar_resultados_soft(svm_lin, X_moons, y_moons, "SVM Soft-Margin — Kernel Lineal (Moons)")
    
    # -------------------------------------------------------------------------
    # Prueba 4: Sigmoide en Dataset 'Moons'
    # -------------------------------------------------------------------------
    print("\n[Escenario 4] Dataset 'Moons' con Kernel Sigmoide")
    # Kernel sigmoide es más inestable con parámetros aleatorios, ajustamos gamma y r
    svm_sig = SVMDual(C=1.0, kernel=Kernels.sigmoid(gamma=0.1, r=0.0))
    svm_sig.fit(X_moons, y_moons)
    
    xi_sig = svm_sig.calcular_holguras(X_moons, y_moons)
    linealmente_separable_sig = np.all(xi_sig < 1e-4)
    print(f"  Vectores de soporte : {np.sum(svm_sig.sv_mask)}")
    print(f"  Masa total de ξ     : {np.sum(xi_sig):.4f}")
    print(f"  Separabilidad en ϕ  : {'Sí (ξ_i = 0 ∀i)' if linealmente_separable_sig else 'No (existen ξ_i > 0)'}")
    
    graficar_resultados_soft(svm_sig, X_moons, y_moons, "SVM Soft-Margin — Kernel Sigmoide (Moons)")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, 'assets', 'soft_margin')
    print("\n" + "=" * 60)
    print(f"  [✓] Finalizado exitosamente. Revisa las imágenes generadas en:\n      {assets_dir}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
