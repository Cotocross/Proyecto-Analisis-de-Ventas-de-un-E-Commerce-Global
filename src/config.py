from pathlib import Path

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorios de datos y modelos
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Rutas a los archivos de datos crudos
CUSTOMERS_PATH = DATA_DIR / "olist_customers_dataset.csv"
GEOLOCATION_PATH = DATA_DIR / "olist_geolocation_dataset.csv"
ORDER_ITEMS_PATH = DATA_DIR / "olist_order_items_dataset.csv"
ORDER_PAYMENTS_PATH = DATA_DIR / "olist_order_payments_dataset.csv"
ORDER_REVIEWS_PATH = DATA_DIR / "olist_order_reviews_dataset.csv"
ORDERS_PATH = DATA_DIR / "olist_orders_dataset.csv"
PRODUCTS_PATH = DATA_DIR / "olist_products_dataset.csv"
SELLERS_PATH = DATA_DIR / "olist_sellers_dataset.csv"
PRODUCT_CATEGORY_TRANSLATION_PATH = DATA_DIR / "product_category_name_translation.csv"

# Rutas a los archivos procesados
GEOLOCATION_CLEANED_PATH = DATA_DIR / "geolocation_cleaned.csv"

# Ruta al archivo de ejemplo para predicciones
NEW_ORDERS_EXAMPLE_PATH = DATA_DIR / "new_orders_example.csv"

# Ruta al modelo final
MODEL_PATH = MODELS_DIR / "modelo_ventas_rf.pkl"

# Columnas para el modelo (extraídas del notebook)
FEATURES = [
    "month",
    "category_encoded",
    "freight_value",
    "seller_order_count",
    "review_score",
    "delivery_time_days",
    "distance_km",
]

TARGET = "price"
