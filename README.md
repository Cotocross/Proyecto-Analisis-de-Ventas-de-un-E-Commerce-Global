# 🛍️ Análisis y Predicción de Ventas en un E-commerce Brasileño 🇧🇷

**Autor:** Alejandro Javier Contreras Olate  
**Nivel:** Científico de Datos Intermedio  

---

## 📊 Descripción del Proyecto

Este proyecto analiza datos de un e-commerce brasileño para extraer insights y construye un modelo de Machine Learning para predecir los precios de los productos.

Originalmente desarrollado en un Jupyter Notebook, el proyecto ha sido refactorizado a una estructura de scripts de Python para facilitar la reproducibilidad, el testing y la automatización. El notebook `notebooks/analisis_y_prediccion_de_ventas.ipynb` se conserva como el análisis exploratorio original.

Los datos provienen del dataset público de [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) disponible en Kaggle.

---

## 🗂️ Estructura del Proyecto

El proyecto ha sido organizado siguiendo las mejores prácticas para proyectos de Machine Learning, separando el código fuente, las pruebas, los datos y los modelos.

```
📦 Proyecto - Análisis de Ventas
 ┣ 📁 .github/workflows/ci.yml   <- Flujo de Integración Continua (CI)
 ┣ 📁 data/                      <- Archivos .csv del dataset
 ┣ 📁 models/                    <- Modelos entrenados (.pkl)
 ┣ 📁 notebooks/                 <- Notebooks para análisis exploratorio
 ┣ 📁 src/                       <- Código fuente del proyecto
 ┃ ┣ 📄 config.py               <- Variables de configuración y rutas
 ┃ ┣ 📄 data_preprocessing.py   <- Script de preprocesamiento de datos
 ┃ ┣ 📄 train.py                <- Script para entrenar el modelo
 ┃ ┗ 📄 predict.py              <- Script para realizar predicciones
 ┣ 📁 tests/                     <- Pruebas automatizadas
 ┃ ┣ 📄 test_data_preprocessing.py
 ┃ ┗ 📄 test_predict.py
 ┣ 📄 .gitignore
 ┣ 📄 LICENSE
 ┣ 📄 README.md
 ┗ 📄 requirements.txt           <- Dependencias del proyecto
```

---

## 🚀 Cómo Usar Este Proyecto

### 1. Configuración del Entorno

1️⃣ **Clona este repositorio:**
```bash
git clone https://github.com/TU_USUARIO/NOMBRE_REPO.git
cd NOMBRE_REPO
```

2️⃣ **Crea y activa un entorno virtual:**
```bash
python -m venv ven
# En Windows
ven\Scripts\activate
# En Linux/macOS
source ven/bin/activate
```

3️⃣ **Instala las dependencias:**
El archivo `requirements.txt` contiene las versiones exactas de las librerías para garantizar la reproducibilidad.
```bash
pip install -r requirements.txt
```

### 2. Entrenamiento del Modelo

Para entrenar el modelo de Random Forest, ejecuta el siguiente comando. El script procesará los datos de la carpeta `/data` y guardará el modelo final en `/models/modelo_ventas_rf.pkl`.

```bash
python src/train.py
```

### 3. Realizar Predicciones

Puedes usar el modelo entrenado para hacer predicciones sobre nuevos datos. El script espera un archivo CSV con un formato similar a `olist_order_items_dataset.csv`.

Se incluye un archivo de ejemplo en `data/new_orders_example.csv`.

```bash
# Ejemplo de uso con los datos de muestra
python src/predict.py data/new_orders_example.csv
```

### 4. Ejecutar Pruebas

El proyecto incluye una suite de pruebas automatizadas para verificar la funcionalidad del preprocesamiento y la predicción. Para ejecutarlas, usa `pytest`.

```bash
pytest
```
La configuración de Integración Continua en GitHub Actions también ejecuta estas pruebas automáticamente en cada `push` y `pull request`.

---

## 🎯 Resultados Clave

- **Análisis Exploratorio:** Fuerte concentración de ventas en São Paulo y `cama_mesa_banho` como la categoría más vendida.
- **Modelado:** Un modelo **Random Forest** que predice el precio con un **R² de 0.53**, superando a un modelo base de Regresión Lineal (R² de 0.19).
- **Aplicación de Negocio:** El modelo puede ser usado para estrategias de precios dinámicos, detección de anomalías y optimización de costos de envío.

---

## 🤖 Tecnologías y Librerías

- Python 3.11
- Pandas, NumPy
- Scikit-Learn
- Joblib
- Pytest (para pruebas automatizadas)
- GitHub Actions (para Integración Continua)

---

✨ **¡Gracias por visitar mi proyecto!** Si te resulta útil, no olvides darle ⭐️ en GitHub.
