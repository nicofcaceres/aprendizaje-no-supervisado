import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv("datos_transporte_sintetico.csv")

# Crear variable "total_pasajeros"
df["total_pasajeros"] = df["pasajeros_suben"] + df["pasajeros_bajan"]

# Seleccionar las variables para clustering (sin la estación)
variables = df[["hora_pico", "clima_lluvioso", "tipo_dia", "evento_especial", "total_pasajeros"]]

# Escalar los datos
scaler = StandardScaler()
variables_scaled = scaler.fit_transform(variables)

# Determinar el número óptimo de clusters (opcional, usando codo)
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(variables_scaled)
    inertia.append(kmeans.inertia_)

# Graficar método del codo
plt.figure(figsize=(8, 4))
plt.plot(range(1, 10), inertia, marker='o')
plt.title("Método del Codo para elegir K")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Inercia")
plt.grid(True)
plt.show()

# Entrenar KMeans con el número de clusters elegido (ejemplo: 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(variables_scaled)

# Ver resumen por cluster
print(df.groupby("cluster")[["hora_pico", "total_pasajeros", "clima_lluvioso", "evento_especial"]].mean())

# Visualización en 2D con seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="total_pasajeros",
    y="hora_pico",
    hue="cluster",
    palette="Set2"
)
plt.title("Agrupamiento K-Means por Total de Pasajeros y Hora Pico")
plt.xlabel("Total de Pasajeros")
plt.ylabel("¿Hora Pico?")
plt.grid(True)
plt.show()
