# Guía de uso — Data Mining Lab

---

## 1. Carga de datos

![Carga de datos](screenshots/CargaDatos.png)

Punto de entrada de la app. Permite cargar un CSV local (con separador y decimal configurables) o seleccionar un dataset de muestra desde `data/`. Muestra métricas rápidas, vista previa y tipos de columnas.

---

## 2. EDA — Análisis Exploratorio

![EDA — Análisis Exploratorio](screenshots/EDA.png)

Descripción estadística completa del dataset. Incluye:

- **Estadísticas descriptivas** — count, mean, std, min, percentiles, max
- **Valores nulos** _(solo si los hay)_ — barras por columna
- **Matriz de correlación** — heatmap de Pearson; valores cercanos a ±1 indican relación fuerte
- **Distribuciones** — histogramas por variable (detecte sesgo o bimodalidad)
- **Boxplots** — puntos fuera de los bigotes son outliers estadísticos
- **Scatter matrix** — dispersión entre pares de variables; coloreable por categoría

---

## 3. Clasificación

![Clasificación](screenshots/Classification.png)

Entrena y evalúa modelos supervisados. Configure variable objetivo, test size (%) y semilla, luego presione **⬡ Preparar datos**.

### Tab 1 — Modelo Individual

Validación cruzada configurable (folds, método, métrica, balanceo). Las barras de error muestran estabilidad entre folds. Incluye curva ROC/AUC:

| AUC       | Interpretación  |
| --------- | --------------- |
| 0.9 – 1.0 | Excelente       |
| 0.7 – 0.9 | Bueno           |
| 0.5 – 0.7 | Moderado        |
| ~0.5      | Equivale a azar |

### Tab 2 — Benchmark de Modelos

Compara Random Forest, Decision Tree, KNN, XGBoost y AdaBoost con los mismos parámetros. Gráfico de radar + ranking por AUC.

### Tab 3 — Impacto del Balanceo

Útil cuando las clases están desbalanceadas. Técnicas disponibles: `none`, `oversample`, `undersample`, `smote`, `class_weight`. El gráfico Δ vs baseline muestra la ganancia/pérdida de cada técnica.

### Regresión _(target numérico)_

Benchmark automático con métricas RMSE, MAE y Error Relativo (%).

---

## 4. Series de Tiempo

![Series de Tiempo](screenshots/TimeSeries.png)

Analiza y pronostica series temporales. Configure columna de fechas, columna de valores y test size.

Incluye visualización con media móvil, descomposición estacional (tendencia, estacionalidad, residuo) y gráficos ACF/PACF para configurar ARIMA.

### Modelos disponibles

| Modelo                     | Descripción                                      |
| -------------------------- | ------------------------------------------------ |
| **Holt-Winters**           | Suavizamiento exponencial triple; rápido         |
| **Holt-Winters calibrado** | Optimiza parámetros α, β, γ automáticamente      |
| **ARIMA (1,1,1)**          | Buen punto de partida para series moderadas      |
| **ARIMA calibrado**        | Prueba múltiples (p,d,q) y elige el de menor MSE |

**Métricas:** RMSE, MAE, Correlación (CORR) y Error Relativo (RE). El tab **Comparar todos** ejecuta todos los modelos y los pone en tabla comparativa con gráficos de dispersión y pronósticos superpuestos.

---

## 5. Datasets — Versiones Limpias

Se incluyen versiones **limpias** de los datasets, sin **data leakage** (variables que revelan prácticamente la respuesta).

### Cambios realizados

#### **BankChurn_clean.csv**

- **Problemas removidos:** Variable `IsActiveMember` tenía correlación directa con `Exited` (leakage)
- **Variables removidas:** `IsActiveMember`, `CustomerId` (ID)
- **Features restantes:** CreditScore, Age, Tenure, EstimatedSalary
- **Target:** `Exited` (0 = Cliente retiene, 1 = Cliente se va)
- **Accuracy realista esperado:** 60-75% (sin leakage)

#### **kepler_mission_clean.csv**

- **Problemas removidos:** Variable `koi_score` era prácticamente sinónimo del resultado (mean=0.96 para CONFIRMED, mean=0.01 para FALSE POSITIVE)
- **Variables removidas:** `koi_score`, `koi_fpflag_*` (4 banderas de falso positivo = outputs, no predictores), identificadores (`rowid`, `kepid`, etc.), variables intermedias
- **Features restantes:** Características astronómicas reales (período orbital, impacto, profundidad, temperatura, etc.) — ~39 variables
- **Target:** `koi_disposition` (CONFIRMED, FALSE POSITIVE, CANDIDATE)
- **Accuracy realista esperado:** 60-75% (sin leakage)

#### **actividad_sismica_clean.csv**

- **Problemas removidos:** Timestamps (`time`, `updated`) son leakage temporal; `id` es identificador único; `place` es casi único por registro
- **Variables removidas:** `time`, `updated`, `id`, `place`, `locationSource`, `magSource` (redundantes)
- **Features restantes:** Características geofísicas (latitud, longitud, profundidad, magnitud, gap, RMS, etc.) — 16 variables
- **Target:** `type` (earthquake, quarry blast, explosion, etc.)
- **Nota important:** Dataset desequilibrado (98% earthquake vs 2% otros)
- **Accuracy realista esperado:** 70-85% (problema inherentemente desbalanceado)

#### **Walmart_Sales_clean.csv**

- **Problemas removidos:** `Holiday_Flag` estaba desequilibrado 93-7% (modelo podría predecir siempre clase mayoritaria)
- **Cambio de target:** En lugar de `Holiday_Flag`, se creó nuevo target `High_Sales` basado en mediana de ventas
- **Variables removidas:** `Date`, `Store` (IDs), `Weekly_Sales` (usado para derivar target), `Holiday_Flag`
- **Features restantes:** Temperature, Fuel_Price, CPI, Unemployment
- **Target:** `High_Sales` (1 = Weekly_Sales > mediana, 0 = else)
- **Distribución:** **Balanceado 50-50** (3217 vs 3218)
- **Accuracy realista esperado:** 55-70% (problema desafiante)

### Cómo usar en Streamlit

1. Abre la app: `streamlit run mineria_app/streamlit_app.py`
2. Ve a **📂 Carga de datos** → **📦 Datasets de muestra**
3. Selecciona el dataset limpio (ej: `BankChurn_clean.csv`)
4. Configurá separador (`,` o `;`) y decimal (`.` o `,`)
5. Presiona **Cargar dataset**
6. En **Clasificación**, selecciona el **target correcto** (ver tabla arriba)

### Datasets originales vs Limpios

| Dataset       | Original                | Limpio                        | Cambio                                 |
| ------------- | ----------------------- | ----------------------------- | -------------------------------------- |
| **BankChurn** | `BankChurn.csv`         | `BankChurn_clean.csv`         | -2 cols (IsActiveMember, CustomerId)   |
| **Kepler**    | `kepler_mission.csv`    | `kepler_mission_clean.csv`    | -11 cols (koi_score, fpflags, IDs)     |
| **Sísmica**   | `actividad_sismica.csv` | `actividad_sismica_clean.csv` | -6 cols (timestamps, IDs, redundantes) |
| **Walmart**   | `Walmart_Sales.csv`     | `Walmart_Sales_clean.csv`     | +1 col (High_Sales target), -3 cols    |

**Recomendación:** Usa las versiones `*_clean.csv` para análisis educativos sin sorpresas de data leakage.
