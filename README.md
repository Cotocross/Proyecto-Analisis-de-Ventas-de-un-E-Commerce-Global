# 🛍️ Análisis y Predicción de Ventas en un E-commerce Brasileño 🇧🇷

**Autor:** Alejandro Javier Contreras Olate  
**Nivel:** Científico de Datos Intermedio

---

## � Objetivos del Proyecto

1. Analizar datos reales de un e-commerce brasileño (Olist) para extraer insights de negocio.
2. Construir y evaluar modelos de Machine Learning para predecir el precio de productos
3. Proveer un flujo reproducible y automatizado para análisis, entrenamiento, predicción y validación.

---


## 📊 Descripción General

Este proyecto aborda un caso real de ciencia de datos en e-commerce utilizando datos públicos de [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). El objetivo principal es analizar el comportamiento de ventas y construir un modelo predictivo robusto que permita estimar el precio de productos en un entorno de ventas online.

A través de un flujo de trabajo reproducible y automatizado, el proyecto integra:

- **Análisis exploratorio de datos (EDA):** Identificación de tendencias, patrones de compra, categorías más relevantes y factores que influyen en las ventas.
- **Preprocesamiento y feature engineering:** Limpieza, transformación y enriquecimiento de los datos para maximizar el valor predictivo.
- **Modelado y evaluación:** Entrenamiento de modelos de Machine Learning (Random Forest y Regresión Lineal), comparación de métricas y selección del mejor enfoque.
- **Predicción y aplicación práctica:** Generación de predicciones sobre nuevos pedidos, con potencial para casos de negocio como pricing dinámico, detección de anomalías y optimización logística.
- **Automatización y calidad:** Pruebas unitarias y CI/CD con GitHub Actions para asegurar la confiabilidad y mantenibilidad del código.

Este repositorio está diseñado para ser una referencia profesional y didáctica, ideal para portafolios de ciencia de datos aplicada a negocios reales.

---

## 🗂️ Estructura del Proyecto

```
📦 Proyecto - Análisis de Ventas
 ┣ 📁 .github/workflows/ci.yml   <- Workflow de CI/CD (tests automáticos)
 ┣ 📁 data/                      <- Archivos .csv del dataset original y de ejemplo
 ┣ 📁 models/                    <- Modelos entrenados (.pkl)
 ┣ 📁 notebooks/                 <- Notebooks para análisis exploratorio y prototipos
 ┣ 📁 src/                       <- Código fuente modular
 ┃ ┣ 📄 config.py               <- Configuración y rutas
 ┃ ┣ 📄 data_preprocessing.py   <- Preprocesamiento y feature engineering
 ┃ ┣ 📄 train.py                <- Entrenamiento y guardado del modelo
 ┃ ┗ 📄 predict.py              <- Predicción sobre nuevos datos
 ┣ 📁 tests/                     <- Pruebas unitarias (pytest)
 ┃ ┣ 📄 test_data_preprocessing.py
 ┃ ┗ 📄 test_predict.py
 ┣ 📄 .gitignore
 ┣ 📄 LICENSE
 ┣ 📄 README.md
 ┗ 📄 requirements.txt           <- Dependencias del proyecto
```

---

## � Flujo de Trabajo del Proyecto

1. **Exploración y EDA:**
	- Notebook principal: `notebooks/analisis_y_prediccion_de_ventas.ipynb`.
	- Visualización de tendencias, categorías, métodos de pago y reviews.
2. **Preprocesamiento:**
	- Limpieza, unión y transformación de datos con `src/data_preprocessing.py`.
3. **Entrenamiento:**
	- Entrena un modelo Random Forest (`src/train.py`) y guarda el modelo en `models/`.
4. **Predicción:**
	- Predice precios para nuevos pedidos con `src/predict.py`.
5. **Testing:**
	- Pruebas unitarias de preprocesamiento y predicción en `tests/`.
6. **CI/CD:**
	- GitHub Actions ejecuta automáticamente los tests en cada push/pull request.

---

## 🚀 Cómo Usar Este Proyecto

### 1. Configuración del Entorno

```bash
git clone https://github.com/Cotocross/Proyecto-Analisis-de-Ventas-de-un-E-Commerce-Global.git
cd NOMBRE_REPO
python -m venv ven
# En Windows
ven\Scripts\activate
# En Linux/macOS
source ven/bin/activate
pip install -r requirements.txt
```

### 2. Entrenamiento del Modelo

```bash
python src/train.py
```


El modelo entrenado se guardará en `models/modelo_ventas_rf.pkl` (no incluido en el repositorio por su gran tamaño).
Puedes descargar el modelo desde el siguiente enlace externo (MEGA):

[Descargar modelo_ventas_rf.pkl desde MEGA](https://mega.nz/file/XOAGRQRJ#LPzgSzgenRS-ZcP6n-TeHFNmEGx_VZ5qRgXSws_nK5Q)

### 3. Realizar Predicciones

```bash
python src/predict.py data/new_orders_example.csv

```
El script imprimirá las predicciones para cada registro del archivo de entrada.

#### 🧠 ¿Cómo funciona el proceso de predicción?

1. **Carga del modelo:**
  - El script busca el archivo `models/modelo_ventas_rf.pkl` (descargado desde MEGA) y lo carga en memoria. Si el archivo no existe, el script se detiene y muestra un error.

2. **Carga y preprocesamiento de los datos nuevos:**
  - El archivo CSV de entrada debe tener el formato de `olist_order_items_dataset.csv` (ver ejemplo en `data/new_orders_example.csv`).
  - El script carga automáticamente los datos originales necesarios (productos, clientes, reviews, sellers, geolocalización, etc.) para enriquecer y transformar los nuevos pedidos igual que en el entrenamiento.
  - Se realiza el mismo preprocesamiento y feature engineering que se usó para entrenar el modelo, asegurando coherencia en las predicciones.

3. **Generación de predicciones:**
  - El modelo predice el precio para cada registro nuevo usando exactamente las mismas variables (features) que en el entrenamiento.
  - Si hay valores nulos inesperados, el script los rellena automáticamente con la mediana de cada columna.

4. **Salida de resultados:**
  - El script imprime en pantalla un DataFrame con los IDs relevantes (`order_id`, `product_id`) y el precio predicho (`predicted_price`).
  - Puedes redirigir la salida a un archivo CSV si lo deseas.

**Recomendaciones:**
- Asegúrate de tener el modelo en la carpeta `models/` antes de ejecutar predicciones.
- Si modificas el formato de entrada, revisa que contenga todas las columnas necesarias.
- El preprocesamiento es automático, pero si falta algún archivo de datos original, el script lo reportará y se detendrá.

Este flujo garantiza que las predicciones sean consistentes, reproducibles y alineadas con el entrenamiento del modelo.

### 4. Ejecutar Pruebas Unitarias

```bash
pytest
```

Esto ejecuta los tests de preprocesamiento y predicción. Si todo está correcto, verás que los tests pasan.

**Nota:** Los archivos de modelo `.pkl` no están en el repositorio por su tamaño. Descárgalos desde el enlace externo y colócalos en la carpeta `models/` o `notebooks/` según corresponda.

---

## 🧪 Sobre los Tests y la Integración Continua

- Los tests en `tests/` validan que el preprocesamiento y la predicción funcionen correctamente y que los resultados sean coherentes.
- El workflow `.github/workflows/ci.yml` ejecuta automáticamente los tests con pytest en cada push o pull request usando GitHub Actions.
- Si algún test falla, el workflow lo reporta y no permite hacer merge hasta que todo pase correctamente.

---

## ⚙️ Integración Continua (CI) con GitHub Actions

El proyecto utiliza **GitHub Actions** para asegurar la calidad y confiabilidad del código mediante integración continua (CI). Esto significa que cada vez que realizas un push o abres un pull request en la rama `main` o `master`, se ejecuta automáticamente un flujo de trabajo que valida el proyecto.

### ¿Qué hace el workflow de CI?

El archivo `.github/workflows/ci.yml` define el siguiente proceso:

1. **Disparadores:**
   - Se ejecuta en cada push o pull request a las ramas `main` o `master`.
2. **Entorno:**
   - Usa una máquina virtual Ubuntu con Python 3.11.
3. **Pasos principales:**
   - **Checkout:** Descarga el código del repositorio.
   - **Setup Python:** Instala la versión de Python especificada.
   - **Instalación de dependencias:** Ejecuta `pip install -r requirements.txt` para instalar todas las librerías necesarias.
   - **Ejecución de tests:** Lanza `pytest` para correr todas las pruebas unitarias del proyecto.

### ¿Qué aporta esto al proyecto?

- **Calidad y confianza:** Garantiza que cualquier cambio en el código pase por pruebas automáticas antes de integrarse, evitando errores en producción.
- **Automatización:** No necesitas ejecutar los tests manualmente en cada cambio; GitHub Actions lo hace por ti en la nube.
- **Colaboración segura:** Si trabajas en equipo, asegura que nadie pueda fusionar código que rompa el proyecto.
- **Historial de builds:** Puedes revisar el historial de ejecuciones y detectar rápidamente cuándo y por qué falló algo.

### Personalización

Puedes modificar el archivo `.github/workflows/ci.yml` para agregar más pasos, como análisis de estilo de código, despliegue automático, o integración con otras herramientas.

---

## 📈 Ejemplo de Entrada y Salida para Predicción

**Entrada:** (formato CSV similar a `data/new_orders_example.csv`)

| order_id | product_id | seller_id | ... |
|----------|------------|-----------|-----|
| 1        | p1         | s1        | ... |

**Salida esperada:**

```bash
Predicciones:
Registro 1: $45.32
Registro 2: $23.10
...
```

---

## 🎯 Resultados Clave

- **Análisis Exploratorio:** Fuerte concentración de ventas en São Paulo y `cama_mesa_banho` como la categoría más vendida.
- **Modelado:** Un modelo **Random Forest** que predice el precio con un **R² ≈ 0.54**, superando a la Regresión Lineal (R² ≈ 0.19).
- **Aplicación de Negocio:** El modelo puede usarse para precios dinámicos, detección de anomalías y optimización de envíos.

---

## 🤖 Tecnologías y Librerías

- Python 3.11
- Pandas, NumPy
- Scikit-Learn
- Joblib
- Pytest (tests)
- GitHub Actions (CI/CD)

---

## ℹ️ Notas y Preguntas Frecuentes

- **¿Por qué no encuentra los archivos CSV?**
  - Asegúrate de ejecutar los scripts/notebooks desde la raíz del proyecto o ajusta las rutas a los datos según tu ubicación.
- **¿Cómo agregar nuevos tests?**
  - Crea un archivo nuevo en `tests/` siguiendo el formato de los existentes y usa pytest.
- **¿Dónde conseguir los datos originales?**
  - Descárgalos desde [Kaggle Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

---

✨ **¡Gracias por visitar mi proyecto!** Si te resulta útil, no olvides darle ⭐️ en GitHub.
