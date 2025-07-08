
# 🛍️ Predicción de Ventas en E-commerce Brasileño 🇧🇷

**Autor:** Alejandro Javier Contreras Olate  
**Objetivo:** Analizar y predecir ventas usando técnicas de Machine Learning aplicadas a datos reales de un e-commerce brasileño.

---

## 📊 Descripción del Proyecto

📊 Descripción del Proyecto
Este repositorio contiene tres notebooks principales:


1️⃣ analisis_ventas.ipynb :	Exploración y análisis visual de categorías de productos, precios y costos de envío.

2️⃣ prediccion_ventas.ipynb :	Modelo de Regresión Lineal para estimar precios de productos basado en categorías, costos de envío e información del vendedor.

3️⃣ modelo_avanzado.ipynb : Entrenamiento de un modelo avanzado (Random Forest) con Pipeline y optimización de hiperparámetros usando GridSearchCV. Incluye exportación del modelo para producción.

Los datos provienen del dataset público de [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) disponible en Kaggle.

## 🤖 **Modelo entrenado**

Puedes descargar el archivo del modelo entrenado (`modelo_ventas_rf.pkl`) desde [MEGA](https://mega.nz/file/GLgFSYAK#xaeUEvgrHDQKX0oLVXxQW99E6bimnATZuNVv7MpsoJg).

PD: El archivo no se puede alojar en Github por el peso (1,14 gb).

---

## 🗂️ Estructura de Carpetas

```
📦 Proyecto - Análisis de Ventas de un E-Commerce Global
 ┣ 📁 data/
 ┃ ┣ 📄 olist_customers_dataset.csv
 ┃ ┣ 📄 olist_geolocation_dataset.csv
 ┃ ┣ 📄 olist_order_items_dataset.csv
 ┃ ┣ 📄 olist_order_payments_dataset.csv
 ┃ ┣ 📄 olist_order_reviews_dataset.csv
 ┃ ┣ 📄 olist_orders_dataset.csv
 ┃ ┣ 📄 olist_products_dataset.csv
 ┃ ┣ 📄 olist_sellers_dataset.csv
 ┃ ┣ 📄 product_category_name_translation.csv
 ┣ 📄 analisis_ventas.ipynb
 ┣ 📄 prediccion_ventas.ipynb
 ┣ 📄 modelo_avanzado.ipynb
 ┣ 📄 modelo_ventas_rf.pkl
 ┣ 📄 README.md
 ┣ 📄 requirements.txt
 ┣ 📄 .gitignore


---

## 🚀 Tecnologías y Librerías

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn (Pipeline, RandomForest, GridSearchCV)
- Joblib (para exportar el modelo entrenado)

---

## 📈 Pasos Principales

✅ Análisis Exploratorio
- Visualización de categorías principales.
- Análisis de precios y costos de envío.
- Limpieza de datos y codificación de categorías.

✅ Predicción de Ventas
- División de datos en entrenamiento y prueba.
- Modelo base: Regresión Lineal.
- Modelo mejorado: Random Forest.
- Pipeline + GridSearchCV para optimización.
- Exportación del modelo final.

---

## 📦 Cómo usar este proyecto

1️⃣ Clona este repositorio  
```bash
git clone https://github.com/Cotocross/Portafolio.git
cd Portafolio
```

2️⃣ Crea un entorno virtual  
```bash
python -m venv ven
source ven/bin/activate   # Linux/macOS
ven\Scripts\activate    # Windows
```

3️⃣ Instala dependencias  
```bash
pip install -r requirements.txt
```

4️⃣ Ejecuta los notebooks en VS Code o Jupyter Notebook.

---

✨ **¡Gracias por visitar mi proyecto!** Si te resulta útil, no olvides darle ⭐️ en GitHub.

---

