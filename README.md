# Guía de uso — Data Mining Lab

---

## 1. Carga de datos

**Qué hace:** Punto de entrada de la app. Todo el flujo parte de aquí.

### Componentes

| Componente | Descripción |
|---|---|
| **Subir CSV** | Carga un archivo local. Permite ajustar separador (`,` `;` `TAB` `|`) y decimal (`.` `,`) |
| **Datasets de muestra** | Carga archivos CSV de la carpeta `data/` sin necesidad de subir nada |
| **Métricas rápidas** | Filas, columnas, cuántas son numéricas, total de nulos |
| **Vista previa** | Primeras 10 filas del dataset |
| **Tipos de columnas** | Tabla con tipo de dato, cantidad de nulos y valores únicos por columna |


---

## 2. EDA — Análisis Exploratorio

**Qué hace:** Descripción estadística completa del dataset antes de modelar. Ayuda a entender la estructura, calidad y relaciones en los datos.

### Componentes y cómo interpretarlos

#### Estadísticas descriptivas
Tabla con `count`, `mean`, `std`, `min`, percentiles y `max` para cada columna. Detecte:
- **Columnas con `count` menor al total** → tienen nulos
- **`std` muy alto vs `mean`** → distribución dispersa o con outliers
- **`min`/`max` extremos** → posibles errores de captura

#### Valores nulos *(solo aparece si hay nulos)*
Barras rojas mostrando cuántos nulos tiene cada columna. Una barra alta indica que esa variable puede no ser confiable para modelar.

#### Matriz de correlación
Heatmap azul-verde. Cada celda muestra la correlación de Pearson entre dos variables numéricas (-1 a 1):
- **~1 (verde oscuro):** correlación positiva fuerte → cuando una sube, la otra también
- **~−1 (azul oscuro):** correlación negativa fuerte → se mueven en sentidos opuestos
- **~0 (neutro):** sin relación lineal

> **Precaución:** correlación alta entre variables predictoras (multicolinealidad) puede afectar modelos como Regresión Lineal. Para árboles (Random Forest, XGBoost) no es un problema mayor.

#### Distribuciones
Histogramas por variable. Observe:
- **Distribución normal (campana):** estadísticas como media y std son representativas
- **Sesgada a la derecha/izquierda:** considere transformación logarítmica antes de modelar
- **Bimodal (dos picos):** puede haber dos subpoblaciones mezcladas

#### Boxplots — detección de outliers
Cada caja muestra Q1, mediana, Q3 e IQR. Los **puntos fuera de los bigotes** son outliers estadísticos:
- Muchos outliers → el modelo podría sobreajustarse a ellos o ignorarlos según el algoritmo
- Random Forest y GBM son más robustos ante outliers que Regresión Lineal o KNN

#### Scatter matrix (pares de variables)
Grilla de dispersión entre todas las variables seleccionadas. Si puede colorear por variable categórica, verá si hay separación visual entre clases → señal de que la clasificación será factible. Una separación clara entre colores es buena noticia para cualquier clasificador.

### Cómo elegir columnas para visualizar
- **Distribuciones/Boxplots:** priorice variables que usará como features en los modelos
- **Scatter matrix:** incluya el target junto con las features más correlacionadas; si el dataset es grande, limítese a 4-5 columnas para no saturar la vista

---

## 3. Clasificación

**Qué hace:** Entrena y evalúa modelos de aprendizaje supervisado. Cubre clasificación y regresión, con validación cruzada, curvas ROC y análisis de balanceo.

### Configuración inicial (obligatoria)

| Parámetro | Qué hace | Cómo elegirlo |
|---|---|---|
| **Variable objetivo** | La columna a predecir (y) | Categórica para clasificación, numérica para regresión |
| **Test size (%)** | Porcentaje de datos reservado para evaluar | 20-25% es estándar; si el dataset es pequeño (<500 filas) use 20% |
| **Semilla aleatoria** | Fija el split para reproducibilidad | 42 es convención; cámbielo para verificar estabilidad del modelo |

Presione **⬡ Preparar datos** antes de continuar. Esto codifica variables categóricas, separa train/test y guarda el estado para las tabs.

---

### Tab 1 — Modelo Individual

#### Validación cruzada

| Parámetro | Descripción | Recomendación |
|---|---|---|
| **Folds (k)** | En cuántas partes se divide el train para validar | 5 es el estándar; con datasets pequeños use 3; con datasets grandes puede subir a 10 |
| **Método CV** | `stratified`: mantiene proporción de clases en cada fold (recomendado para clasificación desbalanceada). `kfold`: split aleatorio puro | Use `stratified` siempre en clasificación; `kfold` si las clases están muy balanceadas o si hay clases con 1 sola muestra |
| **Métrica primaria** | Con qué criterio optimizar | `accuracy` cuando las clases están balanceadas; `f1_weighted` cuando hay desbalance (más representativo) |
| **Balanceo** | Técnica aplicada en cada fold durante el entrenamiento | `none` para empezar; cambie según los resultados del Tab 3 |

**Resultado — gráfico de métricas con barras de error:**
Las barras verticales son el **desvío estándar (±σ)** entre folds:
- Barras cortas → modelo estable (se comporta parecido en todos los folds)
- Barras largas → modelo inestable o dataset muy pequeño

#### Curva ROC — AUC

La curva ROC grafica la **Tasa de Verdaderos Positivos** (sensibilidad) vs la **Tasa de Falsos Positivos** en todos los umbrales de clasificación.

| AUC | Interpretación |
|---|---|
| 1.0 | Clasificación perfecta |
| 0.9 – 1.0 | Excelente |
| 0.7 – 0.9 | Bueno |
| 0.5 – 0.7 | Moderado |
| ~0.5 | Equivale a adivinar al azar |

- La **línea punteada diagonal** es el clasificador aleatorio (referencia)
- Cuanto más hacia **arriba-izquierda** esté la curva, mejor
- En multiclase se muestra una curva por clase + la **macro-average** (promedio ponderado igual entre clases)

---

### Tab 2 — Benchmark de Modelos

**Qué hace:** Entrena los 5 modelos disponibles con los mismos parámetros y los compara en una tabla y gráficos.

| Modelo | Fortalezas | Limitaciones |
|---|---|---|
| **Random Forest** | Robusto, maneja nulos/outliers, poca configuración | Más lento en datasets grandes |
| **Decision Tree** | Muy interpretable, rápido | Se sobreajusta fácilmente sin poda |
| **KNN** | Simple, no asume distribución | Lento con muchas features; sensible a escala |
| **XGBoost (GBM)** | Alta precisión, maneja relaciones no lineales | Puede sobreajustarse si no se regula |
| **AdaBoost** | Bueno en datasets limpios y balanceados | Muy sensible a outliers y ruido |

**Gráfico de radar:** compara las 4 métricas simultáneamente para todos los modelos. El modelo que dibuja el polígono más grande en todas las dimensiones es el mejor general.

**Tabla + barras AUC:** ranking final por AUC macro en el split train/test fijo. Complementa la CV.

#### Parámetros a elegir
- Use el mismo `k` y método CV que en el modelo individual para comparar de forma justa
- Cambie el **balanceo** en este tab para ver si impacta el ranking de modelos

---

### Tab 3 — Impacto del Balanceo

**Cuándo es relevante:** Cuando las clases están desbalanceadas (por ejemplo: 90% clase A, 10% clase B). Un modelo que siempre predice "A" tendría 90% de accuracy pero sería inútil.

| Técnica | Cómo funciona | Cuándo usarla |
|---|---|---|
| **none** | Sin modificación | Referencia base |
| **oversample** | Duplica aleatoriamente muestras de la clase minoritaria | Dataset pequeño; simple y efectivo |
| **undersample** | Reduce la clase mayoritaria al tamaño de la minoritaria | Cuando hay muchos datos de la clase mayoritaria |
| **smote** | Genera muestras sintéticas interpolando vecinos entre ejemplos minoritarios | Cuando oversample no basta; necesita mínimo ~5 muestras por clase |
| **class_weight** | Penaliza más los errores en la clase minoritaria durante el entrenamiento | Cuando no quiere alterar el dataset; solo funciona con modelos que lo soporten (RF, GBM) |

**Gráfico principal:** barras con error estándar por técnica. La barra verde es la mejor.

**Gráfico Δ vs baseline:** muestra cuánto ganó (verde) o perdió (rojo) cada técnica respecto a no balancear. Si todas las barras son negativas o neutras, el dataset ya estaba balanceado.

---

### Regresión *(cuando el target es numérico)*

**Qué hace:** Benchmark automático de todos los modelos de regresión disponibles.

| Métrica | Interpretación |
|---|---|
| **RMSE** (Root Mean Square Error) | Error en las mismas unidades que el target; penaliza errores grandes. Menor = mejor |
| **MAE** (Mean Absolute Error) | Error promedio absoluto; más robusto ante outliers que RMSE. Menor = mejor |
| **ER** (Error Relativo) | RMSE como porcentaje de la media del target. Útil para comparar entre datasets |

**Scatter RMSE vs MAE:** los modelos arriba-izquierda son los mejores (RMSE bajo, MAE bajo). Si un modelo tiene RMSE alto pero MAE bajo, tiene pocos errores grandes (outliers en los residuos).

---

## 4. Series de Tiempo

**Qué hace:** Analiza y pronostica series temporales usando modelos clásicos de suavizamiento y ARIMA.

### Configuración

| Parámetro | Descripción | Cómo elegirlo |
|---|---|---|
| **Columna de fechas** | Índice temporal de la serie | La app detecta automáticamente columnas con nombres tipo `fecha`, `date`, `time` |
| **Columna de valores** | La variable a pronosticar | Debe ser numérica continua |
| **Test size (%)** | Porcentaje reservado para evaluar el pronóstico | 20% es estándar; con series cortas (<50 obs) use 10-15% |

---

### Visualización de la serie

**Gráfico temporal con media móvil:**
- La **línea verde** es la serie original
- La **línea naranja punteada** es la media móvil (ventana = 10% de la serie): suaviza el ruido para ver la tendencia
- El **range slider** inferior permite hacer zoom en períodos específicos

---

### Descomposición estacional

Separa la serie en 4 componentes:

| Componente | Color | Interpretación |
|---|---|---|
| **Original** | Verde | La serie completa sin modificar |
| **Tendencia** | Azul | Dirección de largo plazo (sube, baja, estable) |
| **Estacionalidad** | Naranja | Patrón que se repite periódicamente (semanal, mensual, etc.) |
| **Residuo** | Rojo | Lo que no explican tendencia ni estacionalidad; debe verse como "ruido blanco" |

> Si el **residuo** tiene patrones claros (picos repetitivos, tendencia), el modelo aditivo no captura bien la serie y convendría probar el modelo multiplicativo o transformar los datos.

---

### ACF y PACF

Herramientas clave para configurar ARIMA manualmente:

| Gráfico | Qué mide | Cómo leerlo |
|---|---|---|
| **ACF** (Autocorrelación) | Correlación de la serie consigo misma en distintos desfases | Barras que caen lentamente → serie no estacionaria (diferenciación necesaria). Barras que cortan abruptamente en lag *q* → orden MA del ARIMA |
| **PACF** (Autocorrelación Parcial) | Correlación directa en cada lag eliminando efectos intermedios | Primera barra significativa y corte abrupto en lag *p* → orden AR del ARIMA |

Las **líneas punteadas grises** son los intervalos de confianza (±1.96/√n). Las barras que las superan son estadísticamente significativas.

---

### Tab — Modelo Individual

Los 4 modelos disponibles:

| Modelo | Descripción | Cuándo usarlo |
|---|---|---|
| **Holt-Winters** | Suavizamiento exponencial triple con parámetros fijos | Series con tendencia y estacionalidad claras; rápido |
| **Holt-Winters calibrado** | Busca los mejores parámetros α, β, γ probando combinaciones | Cuando la versión simple no ajusta bien; más lento |
| **ARIMA** | Modelo autoregresivo con diferenciación y media móvil, orden (1,1,1) | Series moderadamente complejas; buen punto de partida |
| **ARIMA calibrado** | Prueba múltiples combinaciones (p,d,q) y elige la de menor MSE | Cuando se necesita el mejor ajuste posible; el más lento |

**Métricas de salida:**

| Métrica | Interpretación |
|---|---|
| **RMSE** | Error en unidades del target; penaliza errores grandes |
| **MAE** | Error promedio absoluto; más intuitivo |
| **CORR** | Correlación entre predicción y valores reales (0-1); mide si captura la forma de la curva |
| **RE** | Error relativo; permite comparar entre series de distinta escala |

**Gráfico de pronóstico:**
- **Gris:** datos de entrenamiento
- **Azul:** valores reales del período de prueba
- **Verde punteado:** predicción del modelo
- **Banda verde translúcida:** intervalo de incertidumbre ±1σ (desvío estándar de los residuos)

> Un modelo puede tener buen RMSE pero mala CORR si predice el nivel correcto pero no la forma de la curva. Busque ambos valores altos.

**Gráficos de residuos:**
- **Residuos en el tiempo:** deben verse aleatorios alrededor de cero. Patrones sistemáticos (tendencia, ciclos) indican que el modelo no capturó toda la estructura.
- **Distribución de residuos:** idealmente una campana centrada en cero. Cola larga = errores grandes sistemáticos.

---

### Tab — Comparar todos los modelos

**Qué hace:** Ejecuta todos los modelos seleccionados con el mismo split y los pone en una tabla comparativa.

**Tabla comparativa:** ordenada por RMSE ascendente. El primer modelo es el recomendado para ese dataset.

**Gráfico RMSE vs RE:** los modelos en la esquina inferior-izquierda son los mejores en ambas métricas. Si un modelo tiene RMSE bajo pero RE alto, el target tiene una escala pequeña que amplifica el error relativo.

**Scatter RMSE vs CORR:** el cuadrante ideal es **abajo-izquierda** (RMSE bajo, CORR alta). Los modelos en ese cuadrante predicen bien tanto el nivel como la forma de la serie.

**Pronósticos superpuestos:** gráfico donde se ven todos los modelos sobre el período de test al mismo tiempo. Permite ver visualmente cuál sigue mejor la línea azul (real).

---
