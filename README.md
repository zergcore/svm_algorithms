# Máquinas de Vectores de Soporte (SVM) desde Cero

Este repositorio contiene la implementación desde cero de diferentes variaciones del algoritmo de Máquinas de Vectores de Soporte (SVM), desarrolladas en Python.

## Archivos y Modelos

Cada directorio contiene la implementación y el archivo principal de una variación particular de SVM. A continuación, un resumen de qué trata cada archivo:

1. **Modelo Lineal (Hard Margin)** - `hard_margin/main.py`: 
   El algoritmo base original para datos perfectamente separables.

2. **Modelo Suavizado (Soft Margin)** - `soft_margin/main.py`: 
   El algoritmo que incorpora variables de holgura ($\epsilon$) para detectar cuándo los datos no son separables y tolerar errores en el margen.

3. **Modelo de Núcleo (Kernel Trick)** - `dual_kernel/main.py`: 
   La resolución del problema dual aplicando Kernels (como el Gaussiano) para clasificar datos no separables linealmente.

4. **Multiclasificación** - `multiclass_classifier/main.py`: 
   La aplicación de las estrategias "Uno contra Todos" (OvA) y "Uno contra Uno" (OvO) sobre el modelo de núcleo para clasificar conjuntos de 4 o más clases, comparando sus tiempos de ejecución.

---

## Requisitos y Configuración del Entorno

Para ejecutar cualquiera de estos modelos, debes tener **Python** instalado en tu sistema. Además, se debe utilizar un entorno virtual para instalar y aislar las dependencias del proyecto.

Sigue estos pasos en el directorio raíz del proyecto (`svm_algorithms`) para prepararlo:

### 1. Crear el Entorno Virtual
Abre una terminal y ejecuta el siguiente comando para crear un entorno virtual llamado `.venv`:
```bash
python -m venv .venv
```

### 2. Activar el Entorno Virtual
El comando de activación varía según tu sistema operativo:

- **En Windows:**
  ```cmd
  .venv\Scripts\activate
  ```

- **En Mac / Linux:**
  ```bash
  source .venv/bin/activate
  ```
*(Sabrás que el entorno se activó correctamente porque verás `(.venv)` antes del nombre de usuario en tu terminal).*

### 3. Instalar las Librerías
Con el entorno activo, instala las librerías necesarias (`numpy`, `scipy`, `matplotlib`) mediante:
```bash
pip install -r requirements.txt
```

---

## Cómo Ejecutar Cada Archivo

Una vez activado el entorno virtual e instaladas las librerías, asegúrate de estar posicionado en el directorio raíz (`svm_algorithms`). 

A continuación se muestran los comandos explícitos para correr cada modelo:

**Para correr el Modelo Lineal (Hard Margin):**
```bash
python -m hard_margin.main
```

**Para correr el Modelo Suavizado (Soft Margin):**
```bash
python -m soft_margin.main
```

**Para correr el Modelo de Núcleo (Kernel Trick):**
```bash
python -m dual_kernel.main
```

**Para correr las Estrategias de Multiclasificación:**
```bash
python -m multiclass_classifier.main
```

Al ejecutar los scripts, podrás ver los resultados en la consola (como los vectores de soporte encontrados o los tiempos del benchmark). Además, **todos los resultados visuales y gráficos generados se guardarán automáticamente en la carpeta `assets/`**, agrupados por algoritmo.
