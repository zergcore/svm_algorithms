from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Any

def plot_results(
    model: Any,
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    title: str,
    file_name: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    colors: dict[int, str] = {1: '#2196F3', -1: '#F44336'}
    markers: dict[int, str] = {1: 'o', -1: 's'}

    # Graficar dispersión de datos original
    for class_label in [1, -1]:
        mask: NDArray[np.bool_] = (y == class_label)
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=colors[class_label], marker=markers[class_label],
            edgecolors='k', linewidth=0.5, s=60,
            label=f'Clase {class_label:+d}', alpha=0.85, zorder=3,
        )

    if model.is_fit:
        w = model.w
        b = model.b

        # Destacar Vectores de Soporte
        sv_mask = model.sv_mask
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

        geometric_margin: float = 2.0 / np.linalg.norm(w)
        info_text: str = (
            f'w = [{w[0]:.4f}, {w[1]:.4f}]\n'
            f'b = {b:.4f}\n'
            f'Margen = {geometric_margin:.4f}\n'
            f'||w|| = {np.linalg.norm(w):.4f}\n'
            f'Vectores de Soporte = {n_sv}'
        )
        ax.text(
            0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', fontfamily='monospace', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9), zorder=5,
        )
    else:
        error_msg = "Optimización Fallida"
        if model.optimization_result:
            error_msg += f"\n{model.optimization_result.message}"
        
        ax.text(
            0.5, 0.5, error_msg, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=14, color='white',
            bbox=dict(facecolor='darkred', alpha=0.85, boxstyle='round,pad=0.8'),
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Característica $x_1$', fontsize=11)
    ax.set_ylabel('Característica $x_2$', fontsize=11)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    fig.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f"\n[+] Gráfico exportado a: {os.path.basename(file_name)}")
    plt.close()


def plot_soft_results(model: Any, X: NDArray[np.float64], y: NDArray[np.int8], title: str, assets_dir: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Crear malla 2D para mapear contornos
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(X_grid)
    Z = Z.reshape(xx.shape)
    
    # Rellenar regiones de decisión (f > 0 azul, f < 0 rojo)
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#F44336', '#2196F3'], alpha=0.15)
    
    # Dibujar márgenes (-1, 1) y frontera de decisión (0)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1],
               linestyles=['--', '-', '--'],
               colors=['#F44336', 'k', '#2196F3'],
               linewidths=[1.5, 2.5, 1.5])
    
    # Graficar puntos originales
    colors = {1: '#2196F3', -1: '#F44336'}
    markers = {1: 'o', -1: 's'}
    
    for class_label in [1, -1]:
        mask = (y == class_label)
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[class_label], marker=markers[class_label],
                   edgecolors='k', linewidth=0.5, s=65, label=f'Clase {class_label:+d}', zorder=3)
                   
    # Resaltar Vectores de Soporte (alpha > 0)
    sv_idx = np.where(model.sv_mask)[0]
    if len(sv_idx) > 0:
        ax.scatter(X[sv_idx, 0], X[sv_idx, 1],
                   s=180, facecolors='none', edgecolors='#FFD600',
                   linewidth=2.5, zorder=4, label=f'Vectores de Soporte ({len(sv_idx)})')
                   
    # Marcar violaciones evidentes (ξ > 0) - puntos dentro del margen o mal clasificados
    xi = model.calculate_slack(X, y)
    violations = xi > 1e-4
    if np.any(violations):
        ax.scatter(X[violations, 0], X[violations, 1],
                   s=40, color='black', marker='x',
                   linewidth=1.5, zorder=5, label=f'Violación del Margen ξ>0 ({np.sum(violations)})')
                   
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Característica $x_1$', fontsize=12)
    ax.set_ylabel('Característica $x_2$', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    # Exportar figura
    file_name = title.replace(" ", "_").replace("—", "-").lower() + ".png"
    save_path = os.path.join(assets_dir, file_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [+] Gráfico exportado a: {file_name}")
    plt.close()


def plot_kernel_results(
    model: Any,
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    title: str,
    assets_dir: str,
    filename: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))

    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(X_grid).reshape(xx.shape)

    # Regiones
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#F44336', '#2196F3'], alpha=0.15)
    
    # Fronteras y márgenes
    contour = ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                         linestyles=['--', '-', '--'],
                         colors=['#F44336', 'k', '#2196F3'],
                         linewidths=[1.5, 2.5, 1.5])
    ax.clabel(contour, inline=True, fontsize=10, fmt={-1: 'Margen -1', 0: 'Frontera 0', 1: 'Margen +1'})

    # Puntos originales - Mantiene firma de colores del autor
    colors: dict[int, str] = {1: '#2196F3', -1: '#F44336'}
    markers: dict[int, str] = {1: 'o', -1: 's'}
    labels: dict[int, str] = {1: 'Clase +1 (Anillo Ext)', -1: 'Clase -1 (Centro)'}

    for class_label in [1, -1]:
        mask = (y == class_label)
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[class_label], marker=markers[class_label],
                   edgecolors='k', linewidth=0.5, s=65, label=labels[class_label], zorder=3)

    # Resaltar Vectores de Soporte con el estándar oro definido en N-1
    if model.is_fit:
        sv_idx = np.where(model.sv_mask)[0]
        if len(sv_idx) > 0:
            ax.scatter(X[sv_idx, 0], X[sv_idx, 1],
                       s=180, facecolors='none', edgecolors='#FFD600',
                       linewidth=2.5, zorder=4, label=f'Vectores de Soporte ({len(sv_idx)})')

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Característica $x_1$', fontsize=12)
    ax.set_ylabel('Característica $x_2$', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout()
    save_path = os.path.join(assets_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [+] Gráfico exportado a: {filename}")
    plt.close()


def plot_regions(ax: Any, classifier: Any, X: NDArray[np.float64], y: NDArray[np.int8], title: str, method: str = 'OVA', elapsed_ms: float | None = None) -> None:
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    
    if method == 'OVA':
        Z = classifier.predict_ova(X_grid).reshape(xx.shape)
    else:
        Z = classifier.predict_ovo(X_grid).reshape(xx.shape)
        
    bg_colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']
    pt_colors = ['#F44336', '#4CAF50', '#2196F3', '#FFC107']
    markers = ['o', 's', '^', 'D']
    labels = ['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3']

    # Contorno de fondo
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=bg_colors, alpha=0.5)
    
    # Fronteras
    ax.contour(xx, yy, Z, levels=[0.5, 1.5, 2.5], colors='k', linewidths=0.5)
    
    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        ax.scatter(X[mask, 0], X[mask, 1], c=pt_colors[i], marker=markers[i],
                   edgecolors='k', linewidth=0.5, s=60, label=labels[i], zorder=3)
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Característica $x_1$', fontsize=11)
    ax.set_ylabel('Característica $x_2$', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Anotar tiempo de entrenamiento si fue proporcionado
    if elapsed_ms is not None:
        n_models = 4 if method == 'OVA' else 6
        time_text = f'Modelos: {n_models}\nTiempo: {elapsed_ms:.2f} ms'
        ax.text(
            0.02, 0.02, time_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9),
            zorder=5,
        )
