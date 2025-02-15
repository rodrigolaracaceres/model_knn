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
def plot_knn():
    """Genera y devuelve un gráfico de dispersión de los datos"""
    plt.figure(figsize=(6, 4))
    plt.scatter(X_train_knn, y_train_knn, label="Datos reales", alpha=0.6, color="blue")
    plt.xlabel("X")
    plt.ylabel("Valor Predicho")
    plt.title("Distribución de Datos - Modelo KNN")
    plt.legend()
    plt.grid()

    # Guardar la imagen en memoria
    img_io = io.BytesIO()
    plt.savefig(img_io, format="png")
    img_io.seek(0)

    return Response(content=img_io.getvalue(), media_type="image/png")

@app.get("/knn_predict")
def knn_predict(x: float):
    """Realiza una predicción con el modelo KNN"""
    x_scaled = scaler.transform(np.array([[x]]))  # Escalar el dato de entrada
    prediction = knn_model.predict(x_scaled)
    return {"prediction": float(prediction[0][0])}

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
