from fastapi import FastAPI
from fastapi.responses import Response
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = FastAPI()

# Cargar el modelo KNN y los datos originales
data_knn = joblib.load("knn_model.pkl")
knn_model = data_knn["model"]
X_train_knn = data_knn["X_train"]
y_train_knn = data_knn["y_train"]
scaler = data_knn["scaler"]

@app.get("/")
def read_root():
    return {"message": "API con modelo KNN"}

@app.get("/knn_plot")
def plot_knn(x_query: float = 5.0):
    """Genera y devuelve un gráfico mostrando los 5 vecinos más cercanos"""
    # Escalar el dato de entrada
    x_scaled = scaler.transform(np.array([[x_query]]))
    
    # Obtener los 5 vecinos más cercanos
    distances, indices = knn_model.kneighbors(x_scaled)
    nearest_X = X_train_knn[indices[0]]
    nearest_y = y_train_knn[indices[0]]

    # Crear gráfico de dispersión
    plt.figure(figsize=(6, 4))
    plt.scatter(X_train_knn, y_train_knn, label="Datos reales", alpha=0.6, color="blue")
    plt.scatter(nearest_X, nearest_y, color="red", label="Vecinos más cercanos", marker="o", s=100)
    plt.scatter(x_query, knn_model.predict(x_scaled), color="green", marker="X", s=150, label="Predicción")
    
    plt.xlabel("X")
    plt.ylabel("Valor Predicho")
    plt.title(f"Distribución de Datos - KNN (X={x_query})")
    plt.legend()
    plt.grid()

    # Guardar la imagen en memoria
    img_io = io.BytesIO()
    plt.savefig(img_io, format="png")
    img_io.seek(0)

    return Response(content=img_io.getvalue(), media_type="image/png")

@app.get("/knn_predict")
def knn_predict(x: float):
    """Realiza una predicción y asigna un cluster basado en los 5 vecinos más cercanos"""
    x_scaled = scaler.transform(np.array([[x]]))  # Escalar el dato de entrada
    prediction = knn_model.predict(x_scaled)

    # Obtener los 5 vecinos más cercanos
    distances, indices = knn_model.kneighbors(x_scaled)
    nearest_y = y_train_knn[indices[0]]

    # Crear cluster basado en la media de los 5 vecinos
    cluster = round(np.mean(nearest_y))  # Aproximar a un cluster

    return {
        "prediction": float(prediction[0][0]),
        "assigned_cluster": cluster
    }

@app.get("/knn_metrics")
def knn_metrics():
    """Calcula y devuelve métricas del modelo KNN"""
    # Escalar datos de entrenamiento
    X_train_scaled = scaler.transform(X_train_knn)

    # Predicciones del modelo
    y_pred_knn = knn_model.predict(X_train_scaled)

    # Calcular métricas
    rmse = mean_squared_error(y_train_knn, y_pred_knn) ** 0.5
    mae = mean_absolute_error(y_train_knn, y_pred_knn)
    r2 = r2_score(y_train_knn, y_pred_knn)

    return {
        "R² Score": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4)
    }
