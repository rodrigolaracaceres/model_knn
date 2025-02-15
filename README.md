# 🚀 API de Regresión y Agrupamiento con KNN en FastAPI

Este proyecto implementa un modelo **K-Nearest Neighbors (KNN) Regresor** usando **FastAPI** y lo despliega en **Render**.  
Permite hacer predicciones, visualizar la distribución de los datos y asignar clusters.

## 🔹 Tecnologías Utilizadas
- **Python 3.9+**
- **FastAPI** para la API
- **Uvicorn** como servidor ASGI
- **Scikit-learn** para el modelo KNN
- **Matplotlib** para visualización
- **Joblib** para guardar y cargar modelos
- **Render** para el despliegue en la nube

---

## 🔹 Instalación Local
Para correr la API en tu máquina:

1️⃣ **Clonar el repositorio**  
```bash
git clone https://github.com/rodrigolaracaceres/model_knn.git
cd tu-repo

2️⃣ Crear entorno virtual (opcional pero recomendado)

python -m venv venv
source venv/bin/activate  # En macOS/Linux
venv\Scripts\activate     # En Windows

3️⃣ Instalar dependencias

pip install -r requirements.txt

4️⃣ Entrenar y guardar el modelo KNN

python train_knn.py

5️⃣ Ejecutar la API en local

uvicorn main:app --host 0.0.0.0 --port 8080
La API estará disponible en http://127.0.0.1:8080.

🔹 Despliegue en Render
Para desplegar la API en Render, sigue estos pasos:

1️⃣ Subir el proyecto a GitHub
2️⃣ Ir a Render y crear un nuevo servicio web
3️⃣ Conectar el repositorio de GitHub
4️⃣ Configurar el Start Command con:

uvicorn main:app --host 0.0.0.0 --port 8080
5️⃣ Seleccionar la versión de Python 3.9+
6️⃣ Hacer Deploy y Render generará la URL de tu API

🔹 Uso de la API
📊 Ver gráfico de clusters y vecinos cercanos

GET https://https://model-knn.onrender.com/knn_plot?x_query=5.0
🔹 Respuesta:
📌 Un gráfico con los datos de entrenamiento, los 5 vecinos más cercanos y la predicción.

🔮 Hacer una predicción con KNN

GET https://https://model-knn.onrender.com/knn_predict?x=5.0
🔹 Respuesta esperada:

{
    "prediction": 11.02,
    "assigned_cluster": 1
}
📌 Devuelve la predicción y el cluster asignado.

📈 Obtener métricas del modelo KNN

GET https://https://model-knn.onrender.com/knn_metrics
🔹 Respuesta esperada:

{
    "R² Score": 0.89,
    "RMSE": 1.45,
    "MAE": 1.20
}
📌 Muestra la calidad del modelo KNN en los datos de entrenamiento.

🔎 Ver cuántos clusters tiene el dataset

GET https://https://model-knn.onrender.com/knn_clusters
🔹 Respuesta esperada:

{
    "optimal_clusters": 3,
    "inertia_values": [1250.2, 940.5, 710.1, 600.3, 500.9]
}
📌 Determina cuántos clusters naturales hay en los datos.

📜 Estructura del Proyecto

/tu-repo
│── main.py               # Código de la API FastAPI
│── train_knn.py          # Entrena y guarda el modelo KNN
│── requirements.txt      # Dependencias del proyecto
│── README.md             # Documentación del proyecto