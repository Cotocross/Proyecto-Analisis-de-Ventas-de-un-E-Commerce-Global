import os
import sys

import numpy as np
import pandas as pd
import pytest

# Añadir el directorio src al path para poder importar los módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Ahora las importaciones de src deberían funcionar
import config
from predict import load_model, make_predictions

# Cargar el modelo real una sola vez para todas las pruebas de este módulo.
# Usamos un try-except para dar un mensaje más claro si el modelo no existe.
try:
    MODEL = load_model(config.MODEL_PATH)
except SystemExit as e:
    MODEL = None
    print(
        f"Error al cargar el modelo para las pruebas: {e}. "
        "Asegúrate de que el modelo esté entrenado."
    )


@pytest.fixture
def new_orders_sample():
    """Fixture para cargar los datos de ejemplo de nuevas órdenes."""
    return pd.read_csv(config.NEW_ORDERS_EXAMPLE_PATH)


@pytest.mark.skipif(
    MODEL is None,
    reason="El modelo no pudo ser cargado, se omite la prueba de predicción.",
)
def test_make_predictions_returns_valid_output(new_orders_sample):
    """
    Prueba que make_predictions devuelve un array de numpy con el número
    correcto de predicciones.
    """
    # Realizar predicciones
    predictions = make_predictions(MODEL, new_orders_sample)

    # Verificar que el resultado no es None
    assert predictions is not None, "La función de predicción devolvió None."

    # Verificar que el resultado es un array de numpy
    assert isinstance(
        predictions, np.ndarray
    ), "El resultado de la predicción no es un array de numpy."

    # Verificar que el número de predicciones es igual al número de filas de entrada
    assert len(predictions) == len(
        new_orders_sample
    ), "El número de predicciones no coincide con el número de registros de entrada."

    # Verificar que las predicciones no están vacías
    assert predictions.size > 0, "El array de predicciones está vacío."

    # Verificar que los valores son numéricos (float o int)
    assert all(
        isinstance(p, (np.floating, np.integer)) for p in predictions
    ), "No todas las predicciones son numéricas."
