"""
Módulo 4: Clasificación Multiclase (OvA y OvO)
==============================================

Implementación de estrategias "Uno contra Todos" (One-vs-All) y 
"Uno contra Uno" (One-vs-One) utilizando el SVMDualKernel como modelo base.
Se incluye un benchmark para comparar los tiempos de entrenamiento.
"""

from __future__ import annotations
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from core.datasets import generate_multiclass_data
from core.kernels import Kernels
from core.visualization import plot_regions
from dual_kernel.main import SVMDualKernel


# =============================================================================
# 1. ESTRATEGIA ONE-VS-ALL (Uno contra Todos)
# =============================================================================

class MulticlassOVA:
    """
    Entrena N clasificadores SVM, uno para cada clase contra el resto.
    La predicción se realiza seleccionando la clase con el mayor margen funcional (argmax).
    """
    def __init__(self, C: float = 1.0, kernel: callable = None):
        self.C = C
        self.kernel = kernel if kernel is not None else Kernels.linear()
        self.models: dict[int, SVMDualKernel] = {}
        self.classes: NDArray[np.int8] = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8]) -> None:
        self.classes = np.unique(y)
        for class_label in self.classes:
            # Reetiquetado binario: +1 para la clase actual, -1 para todas las demás
            y_binary = np.where(y == class_label, 1, -1).astype(np.int8)
            
            model = SVMDualKernel(C=self.C, kernel=self.kernel)
            model.fit(X, y_binary)
            self.models[class_label] = model

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int8]:
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # Matriz para almacenar los márgenes funcionales de cada clasificador
        margins = np.zeros((n_samples, n_classes))
        
        for i, class_label in enumerate(self.classes):
            margins[:, i] = self.models[class_label].decision_function(X)
            
        # El ganador es el que reporta mayor confianza (margen)
        pred_idx = np.argmax(margins, axis=1)
        return self.classes[pred_idx]


# =============================================================================
# 2. ESTRATEGIA ONE-VS-ONE (Uno contra Uno)
# =============================================================================

class MulticlassOVO:
    """
    Entrena N(N-1)/2 clasificadores SVM, evaluando cada par de clases.
    La predicción se realiza mediante votación por mayoría.
    """
    def __init__(self, C: float = 1.0, kernel: callable = None):
        self.C = C
        self.kernel = kernel if kernel is not None else Kernels.linear()
        self.models: dict[tuple[int, int], SVMDualKernel] = {}
        self.classes: NDArray[np.int8] = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8]) -> None:
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_pos = self.classes[i]
                class_neg = self.classes[j]
                
                # Filtrar el dataset para contener solo las dos clases actuales
                mask = (y == class_pos) | (y == class_neg)
                X_pair = X[mask]
                y_pair_original = y[mask]
                
                # Reetiquetado binario: +1 (clase_pos), -1 (clase_neg)
                y_binary = np.where(y_pair_original == class_pos, 1, -1).astype(np.int8)
                
                model = SVMDualKernel(C=self.C, kernel=self.kernel)
                model.fit(X_pair, y_binary)
                
                self.models[(class_pos, class_neg)] = model

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int8]:
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))
        
        class_to_idx = {class_label: idx for idx, class_label in enumerate(self.classes)}
        
        for (class_pos, class_neg), model in self.models.items():
            pred = model.predict(X)
            
            idx_pos = class_to_idx[class_pos]
            idx_neg = class_to_idx[class_neg]
            
            for k in range(n_samples):
                if pred[k] == 1:
                    votes[k, idx_pos] += 1
                else:
                    votes[k, idx_neg] += 1
                    
        pred_idx = np.argmax(votes, axis=1)
        return self.classes[pred_idx]


# =============================================================================
# 3. BENCHMARK Y EJECUCIÓN
# =============================================================================

class MulticlassComparator:
    def __init__(self, classifier_ova, classifier_ovo):
        self.ova = classifier_ova
        self.ovo = classifier_ovo
        
    def predict_ova(self, X):
        return self.ova.predict(X)
        
    def predict_ovo(self, X):
        return self.ovo.predict(X)

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(root_dir, 'assets', 'multiclass_classifier')
    os.makedirs(assets_dir, exist_ok=True)
    
    print("=" * 70)
    print("  SVM MULTICLASE: OvA vs OvO (Kernel Gaussiano)")
    print("=" * 70)
    
    # Dataset con 4 clases
    X, y = generate_multiclass_data(n_classes=4, points_per_class=50)
    
    kernel_rbf = Kernels.gaussian(gamma=0.5)
    
    # ---------------------------------------------------------
    # Entrenamiento OvA
    # ---------------------------------------------------------
    print("[+] Entrenando Clasificador Uno-contra-Todos (OvA)...")
    clf_ova = MulticlassOVA(C=2.0, kernel=kernel_rbf)
    t_start = time.perf_counter()
    clf_ova.fit(X, y)
    t_ova = (time.perf_counter() - t_start) * 1000
    
    # ---------------------------------------------------------
    # Entrenamiento OvO
    # ---------------------------------------------------------
    print("[+] Entrenando Clasificador Uno-contra-Uno (OvO)...")
    clf_ovo = MulticlassOVO(C=2.0, kernel=kernel_rbf)
    t_start = time.perf_counter()
    clf_ovo.fit(X, y)
    t_ovo = (time.perf_counter() - t_start) * 1000
    
    print("\n----------------------------------------")
    print("RESULTADOS DEL BENCHMARK")
    print("----------------------------------------")
    print(f" Tiempo Entrenamiento OvA (4 modelos 200x200) : {t_ova:.2f} ms")
    print(f" Tiempo Entrenamiento OvO (6 modelos 100x100) : {t_ovo:.2f} ms")
    if t_ovo < t_ova:
        print(f" -> OvO fue {t_ova/t_ovo:.2f}x más rápido que OvA.")
    else:
        print(f" -> OvA fue {t_ovo/t_ova:.2f}x más rápido que OvO.")
    print("----------------------------------------")

    # ---------------------------------------------------------
    # Visualización Comparativa
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    comparator = MulticlassComparator(clf_ova, clf_ovo)
    
    plot_regions(ax1, comparator, X, y, 'Fronteras de Decisión: One-vs-All (OvA)', method='OVA', elapsed_ms=t_ova)
    plot_regions(ax2, comparator, X, y, 'Fronteras de Decisión: One-vs-One (OvO)', method='OVO', elapsed_ms=t_ovo)
    
    fig.tight_layout()
    save_path = os.path.join(assets_dir, 'benchmark_multiclase.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[+] Gráfico comparativo exportado a: benchmark_multiclase.png")
    plt.close()

    print("\n" + "=" * 60)
    print(f"  [✓] Finalizado exitosamente. Revisa las imágenes generadas en:\n      {assets_dir}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
