# Actividad 4 - Métodos de aprendizaje no supervisado

## Facultad De Ingeniería
**Jhon Fredy Rada Loaiza**  
**Nicolas Fabian Caceres**  

### Corporación Universitaria Iberoamericana
**Ingeniería de Software**
**JORGE ISAAC CASTAÑEDA VALBUENA**

## Introducción

En esta actividad se desarrolló un modelo de **aprendizaje no supervisado** utilizando el algoritmo **K-Means**, con el objetivo de encontrar patrones o agrupamientos dentro de un conjunto de datos sintéticos relacionados con el transporte masivo. Esta técnica es útil cuando no se tienen etiquetas previamente definidas y se desea descubrir estructuras ocultas en los datos, como comportamientos de estaciones, horarios de mayor afluencia o condiciones externas (clima, eventos).

Para ello, se implementaron dos scripts principales:

- Un **generador de datos sintéticos** que simula escenarios reales del sistema de transporte.
- Un **modelo K-Means** que realiza el análisis y agrupamiento de los datos.

### ¿Qué es K-Means?
K-Means es un algoritmo de aprendizaje no supervisado que se utiliza para agrupar (clusterizar) datos en distintos grupos (o clusters) sin que estos tengan etiquetas previas.

#### ¿Cómo funciona?
Se elige un número K, que representa la cantidad de grupos que se desea encontrar.

El algoritmo selecciona aleatoriamente K puntos llamados centroides (uno por cada grupo).

Luego:

- Cada punto del dataset se asigna al centroide más cercano.

- Se recalculan los centroides como el promedio de todos los puntos asignados a cada grupo.

- Se repite el proceso hasta que los centroides dejan de moverse significativamente (convergencia).

#### ¿Para qué sirve?
- Para descubrir patrones ocultos en los datos.

- Identificar segmentos de usuarios (por ejemplo, tipos de pasajeros en un sistema de transporte).

- Reducir la dimensionalidad para análisis visuales.

- Clasificar información cuando no hay etiquetas disponibles.

---
# Descripción de los Datos

## Archivo: datos_transporte_sintetico.csv

Este conjunto de datos contiene información sintética relacionada con el transporte urbano. Cada fila representa una observación en una determinada condición ambiental, de tiempo y de eventos especiales. A continuación, se describen las variables incluidas:

### Variables del Dataset

| Nombre de la Variable | Tipo de Dato | Descripción |
|------------------------|--------------|-------------|
| **hora_pico**          | Binaria (0 o 1) | Indica si la observación se realizó en hora pico (1) o no (0). |
| **total_pasajeros**    | Numérica (entero) | Número total de pasajeros observados en una unidad de transporte en ese instante. |
| **clima_lluvioso**     | Binaria (0 o 1) | Indica si había lluvia al momento de la observación (1 = sí, 0 = no). |
| **evento_especial**    | Binaria (0 o 1) | Indica si se estaba realizando un evento especial que pudiera afectar el tráfico (1 = sí, 0 = no). |

### Descripción General

Este dataset fue diseñado para realizar **análisis de agrupamiento** (clustering) utilizando métodos de aprendizaje no supervisado como **K-Means**. Las variables seleccionadas permiten evaluar el comportamiento de los usuarios del transporte en función de condiciones ambientales y temporales.

### Posibles Usos

- Identificación de patrones de demanda de pasajeros.
- Clasificación de situaciones comunes y atípicas en el transporte.
- Segmentación de escenarios según condiciones ambientales y de evento.

---

## Librerías utilizadas

### En el generador del dataset (`new_data_set.py`)

- **`pandas`**: para crear y guardar estructuras de datos tipo DataFrame.
- **`numpy`**: para generar datos aleatorios con control sobre la distribución de los valores.
- **`random`**: para seleccionar valores aleatorios en variables categóricas simuladas.

### En el script de modelado (`kmeans_model.py`)

- **`pandas`**: para manipulación del dataset.
- **`scikit-learn (sklearn)`**:
  - `StandardScaler`: para escalar las variables y que tengan igual peso.
  - `KMeans`: implementación del algoritmo K-Means.
- **`matplotlib.pyplot` y `seaborn`**: para visualización de resultados, como el método del codo y el gráfico de clusters.

---

## Explicación del script (`kmeans_model.py`)

### 1. Carga del dataset

```python
df = pd.read_csv("datos_transporte_sintetico.csv")
```
Se lee el archivo CSV que contiene los datos simulados del sistema de transporte.
### 2. Creación de variable objetivo
```python
df["total_pasajeros"] = df["pasajeros_suben"] + df["pasajeros_bajan"]
```
Se crea una nueva columna total_pasajeros que combina la cantidad de personas que suben y bajan en cada registro.
### 3. Selección de variables
```python
variables = df[["hora_pico", "clima_lluvioso", "tipo_dia", "evento_especial", "total_pasajeros"]]
```
Se seleccionan solo las variables numéricas o convertidas a numéricas que se usarán para el agrupamiento.

### 4. Escalado de variables
```python
scaler = StandardScaler()
variables_scaled = scaler.fit_transform(variables)
```
El escalado permite que todas las variables influyan por igual en el análisis, evitando que los valores más grandes dominen.

### 5. Selección de número de clusters (método del codo)
```python
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(variables_scaled)
```
Este ciclo permite visualizar la inercia (distancia total interna del cluster) para distintos valores de K y elegir el óptimo.

### 6. Entrenamiento de K-Means
```python
kmeans = KMeans(n_clusters=3)
df["cluster"] = kmeans.fit_predict(variables_scaled)
```
Se entrena el modelo con el número de clusters deseado (en este caso, 3) y se asigna a cada fila un número de cluster.

### 7. Análisis y visualización
```python
sns.scatterplot(x="total_pasajeros", y="hora_pico", hue="cluster", data=df)
```
Se grafica la relación entre total de pasajeros y hora pico, coloreando los puntos según el grupo asignado por el modelo.

### Conclusión
Este ejercicio permitió aplicar técnicas de aprendizaje no supervisado para analizar datos de transporte masivo, facilitando la identificación de patrones en el comportamiento de los usuarios bajo diferentes condiciones. El uso de K-Means ofrece una visión exploratoria útil para segmentar estaciones o situaciones y apoyar la toma de decisiones en políticas de movilidad urbana.

[Video Presentacion](https://laiberocol-my.sharepoint.com/:v:/g/personal/ncacere1_estudiante_ibero_edu_co/EYkRQ2K_v0JLkdKxOpjbi-wB0zJZez2SL1JEd5y2PI2fmw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=rdGEbv)