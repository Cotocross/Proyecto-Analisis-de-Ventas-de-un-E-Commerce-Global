
# 🛍️ Análisis y Predicción de Ventas en un E-commerce Brasileño 🇧🇷

**Autor:** Alejandro Javier Contreras Olate  
**Nivel:** Científico de Datos Intermedio

---

## � Objetivos del Proyecto

1. Analizar datos reales de un e-commerce brasileño (Olist) para extraer insights de negocio.
2. Construir y evaluar modelos de Machine Learning para predecir el precio de productos.
3. Proveer un flujo reproducible y automatizado para análisis, entrenamiento, predicción y validación.

---

## 📊 Descripción General

Este proyecto utiliza datos públicos de [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) para abordar un problema real de predicción de precios en e-commerce. Incluye:

- Análisis exploratorio de datos (EDA) en Jupyter Notebook.
- Scripts modulares en Python para preprocesamiento, entrenamiento y predicción.
- Pruebas automatizadas y CI/CD con GitHub Actions.

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
git clone https://github.com/TU_USUARIO/NOMBRE_REPO.git
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
El modelo entrenado se guardará en `models/modelo_ventas_rf.pkl`.

### 3. Realizar Predicciones

```bash
python src/predict.py data/new_orders_example.csv
```
El script imprimirá las predicciones para cada registro del archivo de entrada.

### 4. Ejecutar Pruebas Unitarias

```bash
pytest
```
Esto ejecuta los tests de preprocesamiento y predicción. Si todo está correcto, verás que los tests pasan.

---

## 🧪 Sobre los Tests y la Integración Continua

- Los tests en `tests/` validan que el preprocesamiento y la predicción funcionen correctamente y que los resultados sean coherentes.
- El workflow `.github/workflows/ci.yml` ejecuta automáticamente los tests con pytest en cada push o pull request usando GitHub Actions.
- Si algún test falla, el workflow lo reporta y no permite hacer merge hasta que todo pase correctamente.

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
