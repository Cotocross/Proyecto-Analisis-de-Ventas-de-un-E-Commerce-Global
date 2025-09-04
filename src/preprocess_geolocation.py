import logging

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_cleaned_geolocation(input_path: str, output_path: str):
    """
    Carga los datos de geolocalización, calcula las coordenadas promedio
    por código postal y guarda el resultado en un nuevo archivo CSV.

    Args:
        input_path (str): Ruta al archivo CSV de geolocalización de Olist.
        output_path (str): Ruta donde se guardará el archivo CSV procesado.
    """
    try:
        logging.info(f"Cargando archivo de geolocalización desde {input_path}...")
        geolocation = pd.read_csv(input_path)

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
