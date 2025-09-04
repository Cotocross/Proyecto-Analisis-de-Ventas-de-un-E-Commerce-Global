import os
import sys

import numpy as np
import pandas as pd

# Añadir el directorio raíz al path para poder importar desde 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_and_preprocess_data


def test_load_and_preprocess_data():
    # --- 1. Setup: Crear dataframes de prueba ---
    order_items_data = {
        "order_id": ["1", "2"],
        "product_id": ["p1", "p1"],
        "seller_id": ["s1", "s2"],
        "shipping_limit_date": ["2024-01-15 12:00:00", "2024-02-01 12:00:00"],
    }
    order_items = pd.DataFrame(order_items_data)

    products_data = {"product_id": ["p1"], "product_category_name": ["catA"]}
    products = pd.DataFrame(products_data)

    reviews_data = {"order_id": ["1", "2"], "review_score": [5.0, np.nan]}
    reviews = pd.DataFrame(reviews_data)

    orders_data = {
        "order_id": ["1", "2"],
        "customer_id": ["c1", "c2"],
        "order_purchase_timestamp": ["2024-01-10 10:00:00", "2024-01-25 10:00:00"],
        "order_delivered_customer_date": ["2024-01-15 18:00:00", "2024-02-01 18:00:00"],
    }
    orders = pd.DataFrame(orders_data)

    customers_data = {
        "customer_id": ["c1", "c2"],
        "customer_zip_code_prefix": [10001, 20002],
    }
    customers = pd.DataFrame(customers_data)

    sellers_data = {"seller_id": ["s1", "s2"], "seller_zip_code_prefix": [30003, 40004]}
    sellers = pd.DataFrame(sellers_data)

    geo_data = {
        "geolocation_zip_code_prefix": [10001, 20002, 30003, 40004],
        "geolocation_lat": [-23.55, -22.90, -23.56, -22.91],
        "geolocation_lng": [-46.63, -43.20, -46.64, -43.21],
    }
    geolocation_cleaned = pd.DataFrame(geo_data)

    # --- 2. Execution: Llamar a la función ---
    processed_df = load_and_preprocess_data(
        order_items, products, reviews, orders, customers, sellers, geolocation_cleaned
    )

    # --- 3. Assertions: Verificar los resultados ---
    assert isinstance(processed_df, pd.DataFrame)
    assert processed_df.shape[0] == 2

    expected_cols = [
        "month",
        "category_encoded",
        "seller_order_count",
        "delivery_time_days",
        "distance_km",
    ]
    for col in expected_cols:
        assert col in processed_df.columns
        assert not processed_df[col].isnull().any(), (
            f"La columna '{col}' tiene valores nulos"
        )

    assert processed_df["month"].iloc[0] == 1
    assert processed_df["delivery_time_days"].iloc[0] == 5
    assert processed_df["delivery_time_days"].iloc[1] == 7

    assert processed_df["distance_km"].iloc[0] > 0
