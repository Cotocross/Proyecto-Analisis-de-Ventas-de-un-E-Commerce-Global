
"""
Preprocesamiento de datos de geolocalización para el modelo de ventas e-commerce.

Este script calcula las coordenadas promedio por prefijo de código postal y guarda el resultado listo para unir con los datos principales.
"""

import logging
import pandas as pd

# Configurar logging para el módulo
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)



def generate_cleaned_geolocation(input_path: str, output_path: str):
    """
    Procesa el archivo de geolocalización de Olist y genera un CSV con coordenadas promedio por prefijo postal.

    Args:
        input_path (str): Ruta al archivo CSV de geolocalización original.
        output_path (str): Ruta donde se guardará el archivo procesado.

    Returns:
        None. El archivo limpio se guarda en disco.
    """
    try:
        logging.info(f"Cargando archivo de geolocalización desde {input_path}...")
        geolocation = pd.read_csv(input_path)

        # Agrupar por prefijo postal y calcular medias
        logging.info("Calculando coordenadas promedio por código postal...")
        geolocation_cleaned = (
            geolocation.groupby("geolocation_zip_code_prefix")
            .agg(
                geolocation_lat=("geolocation_lat", "mean"),
                geolocation_lng=("geolocation_lng", "mean"),
            )
            .reset_index()
        )

        logging.info(f"Guardando datos limpios en {output_path}...")
        geolocation_cleaned.to_csv(output_path, index=False)

        logging.info("Preprocesamiento de geolocalización completado exitosamente.")
    except FileNotFoundError:
        logging.error(f"Error: El archivo de entrada no fue encontrado en {input_path}")
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
    # Rutas para ejecución directa del script
    # Esto asume que el script se ejecuta desde el directorio raíz del proyecto
    INPUT_FILE = "data/olist_geolocation_dataset.csv"
    OUTPUT_FILE = "data/geolocation_cleaned.csv"

    generate_cleaned_geolocation(INPUT_FILE, OUTPUT_FILE)
