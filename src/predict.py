import argparse
import logging
import sys

import joblib
import pandas as pd

# Añadir el directorio src al path para importar módulos locales
sys.path.append("src")

import config
from data_preprocessing import load_and_preprocess_data
from preprocess_geolocation import generate_cleaned_geolocation

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)



"""
Script de predicción para el modelo de ventas de un e-commerce brasileño.

Permite cargar un modelo entrenado y realizar predicciones sobre nuevos datos de pedidos.

Uso desde terminal:
    python src/predict.py data/new_orders_example.csv

El script imprime un DataFrame con los IDs y el precio predicho para cada registro.
"""

def load_model(model_path):
    """
    Carga un modelo entrenado desde un archivo .pkl.

    Args:
        model_path (str): Ruta al archivo del modelo entrenado (formato joblib/pkl).

    Returns:
        Modelo de machine learning cargado.

    Raises:
        SystemExit: Si el archivo no existe.
    """
    try:
        model = joblib.load(model_path)
        logging.info(f"Modelo cargado exitosamente desde {model_path}")
        return model
    except FileNotFoundError:
        logging.error(
            f"No se encontró el archivo del modelo en {model_path}. "
            "Asegúrate de que el modelo ha sido entrenado y guardado."
        )
        sys.exit(1)

def make_predictions(model, new_data_df):
    """
    Realiza predicciones sobre nuevos datos de pedidos.

    Args:
        model: Modelo de machine learning entrenado (RandomForestRegressor).
        new_data_df (pd.DataFrame): DataFrame con los nuevos datos de pedidos
            (debe tener formato similar a 'olist_order_items_dataset.csv').

    Returns:
        np.ndarray: Array con los valores predichos para el precio.

    Ejemplo de uso:
        >>> model = load_model('models/modelo_ventas_rf.pkl')
        >>> new_data = pd.read_csv('data/new_orders_example.csv')
        >>> preds = make_predictions(model, new_data)
    """
    logging.info("Cargando datos adicionales para el preprocesamiento...")
    try:
        orders = pd.read_csv(config.ORDERS_PATH)
        products = pd.read_csv(config.PRODUCTS_PATH)
        customers = pd.read_csv(config.CUSTOMERS_PATH)
        reviews = pd.read_csv(config.ORDER_REVIEWS_PATH)
        sellers = pd.read_csv(config.SELLERS_PATH)

        if not config.GEOLOCATION_CLEANED_PATH.exists():
            logging.warning(
                "Archivo de geolocalización procesado no encontrado. "
                "Generándolo ahora..."
            )
            generate_cleaned_geolocation(
                input_path=str(config.GEOLOCATION_PATH),
                output_path=str(config.GEOLOCATION_CLEANED_PATH),
            )
        geolocation_cleaned = pd.read_csv(config.GEOLOCATION_CLEANED_PATH)

    except FileNotFoundError as e:
        logging.error(f"Error al cargar los datasets necesarios: {e}")
        sys.exit(1)

    logging.info("Preprocesando los nuevos datos...")
    # La función de preprocesamiento espera todos los dataframes
    # Pasamos los nuevos datos como 'order_items'
    processed_df = load_and_preprocess_data(
        new_data_df, products, reviews, orders, customers, sellers, geolocation_cleaned
    )

    # Usamos todas las filas procesadas del input, no solo nuevos 'order_id'.
    # Esto permite hacer predicciones sobre datos existentes para evaluación.
    prediction_df = processed_df

    if prediction_df.empty:
        logging.warning("El preprocesamiento no generó datos para la predicción.")
        return None

    # Seleccionar las mismas features que en el entrenamiento
    X_new = prediction_df[config.FEATURES].copy()

    # Manejar posibles valores nulos que no se manejaron en el preprocesamiento
    if X_new.isnull().sum().any():
        logging.warning(
            "Se encontraron valores nulos en las features. "
            "Se rellenarán con la media/mediana."
        )
        for col in X_new.columns:
            if X_new[col].isnull().any():
                # Usamos la mediana para robustez ante outliers
                median_val = X_new[col].median()
                X_new[col].fillna(median_val, inplace=True)

    logging.info(f"Realizando predicciones sobre {len(X_new)} nuevos registros.")

    predictions = model.predict(X_new)

    return predictions

def main():
    """
    Función principal para ejecutar el script de predicción desde la terminal.

    Uso:
        python src/predict.py data/new_orders_example.csv

    El script imprime un DataFrame con los IDs y el precio predicho para cada registro.
    """
    parser = argparse.ArgumentParser(
        description="Script para hacer predicciones de ventas de productos nuevos."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help=(
            "Ruta al archivo CSV con los nuevos datos (debe tener el formato de "
            "'olist_order_items_dataset.csv')."
        ),
    )
    args = parser.parse_args()

    # Cargar el modelo entrenado
    model = load_model(config.MODEL_PATH)

    # Cargar los nuevos datos
    try:
        new_data = pd.read_csv(args.input_file)
        logging.info(f"Nuevos datos cargados desde {args.input_file}")
    except FileNotFoundError:
        logging.error(f"No se pudo encontrar el archivo de entrada: {args.input_file}")
        sys.exit(1)

    # Realizar predicciones
    predictions = make_predictions(model, new_data)

    if predictions is not None:
        logging.info("Predicciones generadas:")
        # Crear un DataFrame para mostrar las predicciones
        # junto con alguna información de identificación
        new_data["predicted_price"] = predictions
        print(new_data[["order_id", "product_id", "predicted_price"]])


if __name__ == "__main__":
    main()
