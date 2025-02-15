import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Simulación de datos
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Variable predictora
y = 2.5 * X + np.random.randn(100, 1) * 2  # Variable objetivo

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo KNN
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Guardar modelo, datos de entrenamiento y scaler
joblib.dump({"model": knn_model, "X_train": X_train, "y_train": y_train, "scaler": scaler}, "knn_model.pkl")

print("Modelo KNN entrenado y guardado exitosamente.")
