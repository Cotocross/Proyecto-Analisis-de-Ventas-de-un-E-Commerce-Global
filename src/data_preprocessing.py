
"""
Funciones de preprocesamiento y feature engineering para el modelo de ventas e-commerce.

Incluye:
- Cálculo de distancia Haversine entre cliente y vendedor.
- Unión y transformación de los datasets de Olist.
- Generación de variables para el modelado.
"""

import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia Haversine entre dos puntos geográficos.

    Args:
        lat1, lon1: Coordenadas del primer punto (cliente).
        lat2, lon2: Coordenadas del segundo punto (vendedor).

    Returns:
        float o array: Distancia en kilómetros.
    """
    R = 6371  # Radio de la Tierra en kilómetros
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def load_and_preprocess_data(
    order_items, products, reviews, orders, customers, sellers, geolocation_cleaned
):
    """
    Une y transforma los datasets de Olist para el modelado de ventas.

    Realiza:
    - Unión de tablas principales (items, productos, reviews, órdenes, clientes, vendedores).
    - Unión de coordenadas promedio para cliente y vendedor.
    - Feature engineering: mes, categoría codificada, popularidad vendedor, review_score, tiempo de entrega, distancia.

    Args:
        order_items (pd.DataFrame): Items de pedidos.
        products (pd.DataFrame): Productos.
        reviews (pd.DataFrame): Reviews de pedidos.
        orders (pd.DataFrame): Órdenes.
        customers (pd.DataFrame): Clientes.
        sellers (pd.DataFrame): Vendedores.
        geolocation_cleaned (pd.DataFrame): Coordenadas promedio por código postal.

    Returns:
        pd.DataFrame: DataFrame listo para entrenamiento/modelado.
    """
    # --- Unir datasets principales ---
    df = pd.merge(order_items, products, on="product_id")
    df = pd.merge(df, reviews[["order_id", "review_score"]], on="order_id", how="left")
    df = pd.merge(df, orders, on="order_id", how="left")

    # --- Unir datos de cliente y vendedor ---
    df = pd.merge(df, customers, on="customer_id", how="left")
    df = pd.merge(df, sellers, on="seller_id", how="left")

    # --- Unir con datos de geolocalización ---
    # Para el cliente
    df = pd.merge(
        df,
        geolocation_cleaned,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )
    df.rename(
        columns={"geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng"},
        inplace=True,
    )
    df.drop("geolocation_zip_code_prefix", axis=1, inplace=True)

    # Para el vendedor
    df = pd.merge(
        df,
        geolocation_cleaned,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )
    df.rename(
        columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"},
        inplace=True,
    )
    df.drop("geolocation_zip_code_prefix", axis=1, inplace=True)

    # --- Feature Engineering ---

    # 1. Extraer el mes de la fecha de envío
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"])
    df["month"] = df["shipping_limit_date"].dt.month

    # 2. Codificar la categoría del producto
    if "product_category_name" in df.columns:
        df["product_category_name"] = df["product_category_name"].fillna("unknown")
        df["category_encoded"] = (
            df["product_category_name"].astype("category").cat.codes
        )
    else:
        df["category_encoded"] = 0

    # 3. Popularidad del vendedor (número de ventas)
    if "seller_id" in df.columns:
        seller_order_counts = df["seller_id"].value_counts().to_dict()
        df["seller_order_count"] = df["seller_id"].map(seller_order_counts).fillna(0)
    else:
        df["seller_order_count"] = 0

    # 4. Imputar valores nulos en 'review_score' con la media
    if "review_score" in df.columns:
        df["review_score"] = df["review_score"].fillna(df["review_score"].mean())

    # 5. Calcular días de entrega
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"] = pd.to_datetime(
        df["order_delivered_customer_date"]
    )
    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["delivery_time_days"] = df["delivery_time_days"].fillna(
        df["delivery_time_days"].median()
    )

    # 6. Calcular distancia entre vendedor y cliente
    df["distance_km"] = haversine_distance(
        df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"]
    )
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())

    return df
