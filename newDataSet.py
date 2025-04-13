import pandas as pd
import numpy as np
import random

# Parámetros
np.random.seed(42)
num_registros = 500

# Posibles estaciones
estaciones = [f"EST-{i:03d}" for i in range(1, 21)]

# Generar datos sintéticos
data = {
    "estacion_id": np.random.choice(estaciones, num_registros),
    "hora_pico": np.random.choice([0, 1], num_registros, p=[0.6, 0.4]),
    "pasajeros_suben": np.random.poisson(lam=40, size=num_registros),
    "pasajeros_bajan": np.random.poisson(lam=30, size=num_registros),
    "clima_lluvioso": np.random.choice([0, 1], num_registros, p=[0.8, 0.2]),
    "tipo_dia": np.random.choice([0, 1, 2], num_registros, p=[0.7, 0.2, 0.1]),
    "evento_especial": np.random.choice([0, 1], num_registros, p=[0.9, 0.1])
}

df = pd.DataFrame(data)

# Exportar a CSV
df.to_csv("datos_transporte_sintetico.csv", index=False)

print("Dataset generado y guardado como 'datos_transporte_sintetico.csv'")
