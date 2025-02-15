# KNN model

## Crear un archivo requirements.txt con las librerías necesarias para la api

## El nombre del archivo de la api debe llamarse "main.py" para que render lo tome automaticamente

## Iniciar sesión en render con el login de github

## En render declarar como start command:
uvicorn main:app --host 0.0.0.0 --port 8080

## Pruebas
https://https:/model-knn.onrender.com/knn_plot?x_query=5.0

https://https://model-knn.onrender.com/knn_predict?x=5.0

https://https://model-knn.onrender.com/knn_metrics

https://https://model-knn.onrender.com/knn_clusters

## Estructura
/tu-repo
│── main.py               # Código de la API FastAPI
│── train_knn.py          # Entrena y guarda el modelo KNN
│── requirements.txt      # Dependencias del proyecto
│── README.md             # Documentación del proyecto

🎯 Autor
📌 Rodrigo Alberto Lara Cáceres
🔗 [LinkedIn](https://www.linkedin.com/in/rodrigo-lara-caceres/)
📧 rodlarca@gmail.com