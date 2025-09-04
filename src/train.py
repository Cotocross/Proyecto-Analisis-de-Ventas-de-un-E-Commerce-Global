import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Importar módulos locales
import config
from data_preprocessing import load_and_preprocess_data
from preprocess_geolocation import generate_cleaned_geolocation

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_training():
    """
    Orquesta el pipeline completo de entrenamiento del modelo.
    """
    logging.info("Inicio del pipeline de entrenamiento.")

    # 1. Preprocesar geolocalización si es necesario
    if not config.GEOLOCATION_CLEANED_PATH.exists():
        logging.info("Archivo de geolocalización procesado no encontrado. Generando...")
        generate_cleaned_geolocation(
            input_path=str(config.GEOLOCATION_PATH),
            output_path=str(config.GEOLOCATION_CLEANED_PATH),
        )
    else:
        logging.info("Usando archivo de geolocalización procesado existente.")

    # 2. Cargar todos los datos
    logging.info("Cargando datasets...")
    try:
        orders = pd.read_csv(config.ORDERS_PATH)
        order_items = pd.read_csv(config.ORDER_ITEMS_PATH)
        products = pd.read_csv(config.PRODUCTS_PATH)
        customers = pd.read_csv(config.CUSTOMERS_PATH)
        reviews = pd.read_csv(config.ORDER_REVIEWS_PATH)
        sellers = pd.read_csv(config.SELLERS_PATH)
        geolocation_cleaned = pd.read_csv(config.GEOLOCATION_CLEANED_PATH)
    except FileNotFoundError as e:
        logging.error(
            f"Error al cargar los datos: {e}. "
            "Asegúrate de que los archivos CSV estén en la carpeta 'data'."
        )
        return

    # 3. Preprocesamiento y Feature Engineering
    logging.info("Ejecutando preprocesamiento de datos y feature engineering...")
    df = load_and_preprocess_data(
        order_items, products, reviews, orders, customers, sellers, geolocation_cleaned
    )

    # NOTA: La traducción de categorías no está implementada en el notebook original.
    # Si se quisiera añadir, se haría aquí.

    # 4. Selección de variables y división de datos
    logging.info("Seleccionando features y dividiendo los datos...")

    # Usamos las features definidas en el notebook
    features = [
        "month",
        "category_encoded",
        "freight_value",
        "seller_order_count",
        "review_score",
        "delivery_time_days",
        "distance_km",
    ]
    target = "price"

    X = df[features].copy()
    y = df[target].copy()

    # Manejar NaNs que puedan haber quedado
    X.dropna(inplace=True)
    y = y[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logging.info(
        f"Datos de entrenamiento: {X_train.shape}, Datos de prueba: {X_test.shape}"
    )

    # 5. Definir y entrenar el pipeline con GridSearchCV
    logging.info("Definiendo el pipeline y la grilla de parámetros...")
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("rf", RandomForestRegressor(random_state=42))]
    )

    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [10, 20],
        "rf__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1
    )

    logging.info("Iniciando GridSearchCV... (Esto puede tardar)")
    grid_search.fit(X_train, y_train)

    # 6. Evaluación del modelo final
    logging.info(f"Mejores parámetros encontrados: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Modelo final - R² Score en conjunto de prueba: {r2:.4f}")

    # 7. Guardar el modelo
    logging.info(f"Guardando el modelo en: {config.MODEL_PATH}")
    joblib.dump(best_model, config.MODEL_PATH)
    logging.info("Modelo guardado exitosamente.")

    logging.info("Pipeline de entrenamiento completado.")


if __name__ == "__main__":
    run_training()
