import os
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import sys

# =============================================================================
# 1. IMPORTAR MODELO BASE (ITERACIÓN ANTERIOR)
# =============================================================================

# Añadir la raíz del proyecto al sys.path para permitir importaciones absolutas
ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ruta_base not in sys.path:
    sys.path.insert(0, ruta_base)

from dual_kernel.main import Kernels, SVMDualKernel




# =============================================================================
# 3. GENERACIÓN DE DATOS (Multiclase)
# =============================================================================

def generar_datos_multiclase(
    n_clases: int = 4,
    puntos_por_clase: int = 25,
    semilla: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Genera un dataset sintético bidimensional: 4 clases (clusters) en las esquinas."""
    rng = np.random.default_rng(semilla)
    X_list = []
    y_list = []
    
    # Centroides para 4 clases
    centroides = [
        [2.5, 2.5],   # Clase 0
        [2.5, -2.5],  # Clase 1
        [-2.5, 2.5],  # Clase 2
        [-2.5, -2.5]  # Clase 3
    ]
    
    for i in range(n_clases):
        puntos = rng.standard_normal((puntos_por_clase, 2)) * 0.7 + np.array(centroides[i])
        X_list.append(puntos)
        y_list.append(np.full(puntos_por_clase, i, dtype=np.int8))
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Mezclar
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]


# =============================================================================
# 4. ESTRATEGIA OVA (One-vs-All)
# =============================================================================

class ClasificadorOVA:
    """Implementación de clasificación multiclase 'Uno contra Todos'."""
    def __init__(self, C: float = 1.0, kernel: callable = None):
        self.C = C
        self.kernel = kernel if kernel is not None else Kernels.gaussian()
        self.modelos = []
        self.clases = []

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8]) -> None:
        self.clases = np.unique(y)
        self.modelos = []
        
        for k in self.clases:
            y_temp = np.where(y == k, 1, -1).astype(np.int8)
            modelo = SVMDualKernel(C=self.C, kernel=self.kernel)
            modelo.fit(X, y_temp)
            self.modelos.append(modelo)

    def predecir_OVA(self, X_new: NDArray[np.float64]) -> NDArray[np.int8]:
        # Evaluamos f_k(x) para todos los k clasificadores
        distancias = np.column_stack([m.decision_function(X_new) for m in self.modelos])
        # y_hat = argmax f_k(x)
        indices = np.argmax(distancias, axis=1)
        return self.clases[indices]


# =============================================================================
# 5. ESTRATEGIA OVO (One-vs-One)
# =============================================================================

class ClasificadorOVO:
    """Implementación de clasificación multiclase 'Uno contra Uno'."""
    def __init__(self, C: float = 1.0, kernel: callable = None):
        self.C = C
        self.kernel = kernel if kernel is not None else Kernels.gaussian()
        self.modelos = []
        self.clases = []

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8]) -> None:
        self.clases = np.unique(y)
        self.modelos = []
        n_clases = len(self.clases)
        
        for i in range(n_clases):
            for j in range(i + 1, n_clases):
                clase_i = self.clases[i]
                clase_j = self.clases[j]
                
                # Filtrar matriz X y etiquetas y para contener SOLO clase i y clase j
                mask = (y == clase_i) | (y == clase_j)
                X_ij = X[mask]
                y_ij_original = y[mask]
                
                # y_temp: +1 para clase i, -1 para clase j
                y_temp = np.where(y_ij_original == clase_i, 1, -1).astype(np.int8)
                
                modelo = SVMDualKernel(C=self.C, kernel=self.kernel)
                modelo.fit(X_ij, y_temp)
                
                self.modelos.append((clase_i, clase_j, modelo))

    def predecir_OVO(self, X_new: NDArray[np.float64]) -> NDArray[np.int8]:
        n_muestras = X_new.shape[0]
        # Array de votación
        votos = np.zeros((n_muestras, len(self.clases)))
        
        for clase_i, clase_j, modelo in self.modelos:
            pred = modelo.decision_function(X_new)
            
            idx_i = np.where(self.clases == clase_i)[0][0]
            idx_j = np.where(self.clases == clase_j)[0][0]
            
            # Votos: si f(x) > 0 gana i, sino gana j
            votos_i = (pred > 0)
            votos_j = (pred <= 0)
            
            votos[:, idx_i] += votos_i
            votos[:, idx_j] += votos_j
            
        # y_hat = argmax votos
        indices = np.argmax(votos, axis=1)
        return self.clases[indices]


# =============================================================================
# 6. VISUALIZACIÓN Y BENCHMARKING
# =============================================================================

def graficar_regiones(ax, clasificador, X, y, titulo, method='OVA'):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    
    if method == 'OVA':
        Z = clasificador.predecir_OVA(X_grid).reshape(xx.shape)
    else:
        Z = clasificador.predecir_OVO(X_grid).reshape(xx.shape)
        
    colores_bg = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']
    colores_pt = ['#F44336', '#4CAF50', '#2196F3', '#FFC107']
    marcadores = ['o', 's', '^', 'D']
    labels = ['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3']

    # Contorno de fondo
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=colores_bg, alpha=0.5)
    
    # Fronteras
    ax.contour(xx, yy, Z, levels=[0.5, 1.5, 2.5], colors='k', linewidths=0.5)
    
    for i, clase in enumerate(np.unique(y)):
        mask = y == clase
        ax.scatter(X[mask, 0], X[mask, 1], c=colores_pt[i], marker=marcadores[i],
                   edgecolors='k', linewidth=0.5, s=60, label=labels[i], zorder=3)
        
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel('Característica $x_1$', fontsize=11)
    ax.set_ylabel('Característica $x_2$', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')


def main():
    print("=" * 70)
    print("  SVM MULTICLASE: OvA vs OvO (Kernel Gaussiano)")
    print("=" * 70)
    
    # 1. Datos
    X, y = generar_datos_multiclase(n_clases=4, puntos_por_clase=25, semilla=10)
    
    kernel_rbf = Kernels.gaussian(gamma=0.5)
    C_param = 5.0
    
    # 2. Benchmarking OVA
    print("[+] Entrenando Clasificador Uno-contra-Todos (OvA)...")
    clf_ova = ClasificadorOVA(C=C_param, kernel=kernel_rbf)
    t0_ova = time.perf_counter()
    clf_ova.fit(X, y)
    t1_ova = time.perf_counter()
    tiempo_ova = (t1_ova - t0_ova) * 1000 # a milisegundos
    
    # 3. Benchmarking OVO
    print("[+] Entrenando Clasificador Uno-contra-Uno (OvO)...")
    clf_ovo = ClasificadorOVO(C=C_param, kernel=kernel_rbf)
    t0_ovo = time.perf_counter()
    clf_ovo.fit(X, y)
    t1_ovo = time.perf_counter()
    tiempo_ovo = (t1_ovo - t0_ovo) * 1000 # a milisegundos
    
    # 4. Reporte
    print("\n" + "-" * 40)
    print("RESULTADOS DEL BENCHMARK")
    print("-" * 40)
    print(f" Tiempo Entrenamiento OvA (4 modelos 100x100) : {tiempo_ova:.2f} ms")
    print(f" Tiempo Entrenamiento OvO (6 modelos 50x50)   : {tiempo_ovo:.2f} ms")
    if tiempo_ovo < tiempo_ova:
        print(f" -> OvO fue {tiempo_ova / tiempo_ovo:.2f}x más rápido que OvA.")
    else:
        print(f" -> OvA fue {tiempo_ovo / tiempo_ova:.2f}x más rápido que OvO.")
        
    print("-" * 40)
    
    # 5. Visualización
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    graficar_regiones(axes[0], clf_ova, X, y, f'Uno contra Todos (OvA)\nTiempo: {tiempo_ova:.2f} ms', method='OVA')
    graficar_regiones(axes[1], clf_ovo, X, y, f'Uno contra Uno (OvO)\nTiempo: {tiempo_ovo:.2f} ms', method='OVO')
    
    fig.tight_layout()
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, 'assets', 'multiclass_classifier')
    os.makedirs(assets_dir, exist_ok=True)
    ruta_guardado = os.path.join(assets_dir, 'benchmark_multiclase.png')
    plt.savefig(ruta_guardado, dpi=150, bbox_inches='tight')
    print(f"\n[+] Gráfico comparativo exportado a: benchmark_multiclase.png")
    plt.close()

    print("\n" + "=" * 60)
    print(f"  [✓] Finalizado exitosamente. Revisa las imágenes generadas en:\n      {assets_dir}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
