"""
Paquete mineria avanzada- Jorge Chacon, Stacy Quesada
Clases: EDA, Supervisado (hereda EDA), NoSupervisado (hereda EDA), WebScraping
"""
import warnings
import pandas as pd
import numpy as np
import math
from functools import wraps
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil, pi
from scipy.cluster.hierarchy import dendrogram, ward, single, complete, average, fcluster
from scipy import signal
from scipy.stats import boxcox
from numpy import corrcoef
import statistics
from abc import ABCMeta, abstractmethod

# Imports opcionales
try:
    import umap as um
except ImportError:
    um = None
try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
try:
    from statsmodels.tsa.api import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
try:
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
try:
    from sklearn_extra.cluster import KMedoids
except (ImportError, ValueError):
    KMedoids = KMeans
try:
    from prince import PCA as PCA_Prince
except ImportError:
    PCA_Prince = None
try:
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer, Continuous, Categorical
    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False

pd.options.display.max_rows = 10
warnings.filterwarnings('ignore')

# ============================================================================
# UTILIDADES
# ============================================================================


class ErrorHandler:
    @staticmethod
    def handle_error(msg="Error", raise_exception=False):
        print(f"ERROR: {msg}")
        if raise_exception:
            raise Exception(msg)

    @staticmethod
    def validate_dataframe(df, min_rows=1, min_cols=1):
        if not isinstance(df, pd.DataFrame) or df.empty or df.shape[0] < min_rows or df.shape[1] < min_cols:
            ErrorHandler.handle_error(
                "DataFrame inválido", raise_exception=True)
        return True


def error_handler_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\nError en {func.__name__}: {str(e)}")
            return None
    return wrapper


class Utilidades:
    @staticmethod
    def cargar_datos(path, sep=",", decimal=".", index_col=None):
        try:
            df = pd.read_csv(path, sep=sep, decimal=decimal,
                             index_col=index_col)
            print(f"Datos cargados: {df.shape}")
            return df
        except Exception as e:
            ErrorHandler.handle_error(
                f"Error al cargar: {str(e)}", raise_exception=True)

    @staticmethod
    def centroide(num_cluster, datos, clusters):
        return pd.DataFrame(datos[clusters == num_cluster].mean()).T

# ============================================================================
# CLASE EDA
# ============================================================================


class EDA:
    def __init__(self, path=None, df=None, sep=",", decimal=".", index_col=None):
        if path:
            self.df = Utilidades.cargar_datos(path, sep, decimal, index_col)
        elif df is not None:
            ErrorHandler.validate_dataframe(df)
            self.df = df.copy()
        else:
            ErrorHandler.handle_error(
                "Debe proporcionar 'path' o 'df'", raise_exception=True)
        self.df_original = self.df.copy()

    def analisis_numerico(self):
        self.df = self.df.select_dtypes(include=["number"])
        print(f"Análisis numérico: {self.df.shape[1]} columnas")
        return self

    def analisis_completo(self):
        self.df = pd.get_dummies(self.df)
        print(f"Análisis completo: {self.df.shape[1]} columnas")
        return self

    def resumen_estadistico(self):
        print("\n" + "="*70)
        print("RESUMEN ESTADISTICO")
        print("="*70)
        print(f"\nDimensiones: {self.df.shape}")
        print(f"\nPrimeras filas:\n{self.df.head()}")
        print(f"\nDescripción:\n{self.df.describe()}")
        print(f"\nNulos:\n{self.df.isnull().sum()}")
        return self

    def _plot(self, kind, title, figsize=(12, 8), **kwargs):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        if kind == 'box':
            self.df.boxplot(ax=ax)
        elif kind in ['density', 'hist']:
            self.df.plot(kind=kind, ax=ax, **kwargs)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        return self

    def grafico_boxplot(self, figsize=(15, 8)):
        return self._plot('box', 'Boxplot - Outliers', figsize)

    def grafico_densidad(self, figsize=(12, 8)):
        return self._plot('density', 'Función de Densidad', figsize)

    def grafico_histograma(self, figsize=(10, 6)):
        return self._plot('hist', 'Histograma', figsize, alpha=0.7)

    def matriz_correlacion(self, figsize=(12, 8), mostrar_valores=True):
        corr = self.df.corr(numeric_only=True)
        print(f"\nMatriz de Correlación:\n{corr}")
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=sns.diverging_palette(220, 10, as_cmap=True).reversed(),
                    square=True, annot=mostrar_valores, fmt='.2f', ax=ax)
        plt.title("Mapa de Calor - Correlaciones")
        plt.tight_layout()
        plt.show()
        return self

    def analisis_completo_visual(self):
        print("\nEjecutando análisis visual completo...")
        self.grafico_boxplot().grafico_densidad().grafico_histograma().matriz_correlacion()
        print("Análisis completado")
        return self

    def reset(self):
        self.df = self.df_original.copy()
        print("DataFrame restaurado")
        return self

# ============================================================================
# CLASE SUPERVISADO
# ============================================================================


class Supervisado(EDA):
    def __init__(self, df, target_col='target'):
        super().__init__(df=df)
        self.target_col = target_col
        self.X_train = self.X_test = self.y_train = self.y_test = self.y = None
        if target_col not in df.columns:
            ErrorHandler.handle_error(
                f"Columna '{target_col}' no encontrada", raise_exception=True)
        print(
            f"Supervisado inicializado - Target: {target_col}, Shape: {self.df.shape}")

    def preparar_datos(self, test_size=0.25, random_state=42):
        """Prepara datos para clasificación o regresión sin escalado (se hace en pipeline)"""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        print(
            f"Datos preparados - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self

    @staticmethod
    def _calcular_metricas_clasificacion(y_test, y_pred, y_unique):
        MC = confusion_matrix(y_test, y_pred)
        precision = np.sum(MC.diagonal()) / np.sum(MC)
        return {
            "Matriz de Confusión": MC,
            "Precisión Global": precision,
            "Error Global": 1 - precision,
            "Precisión por categoría": pd.DataFrame(MC.diagonal()/np.sum(MC, axis=1)).T
        }

    def _entrenar_clasificador(self, modelo, nombre, scale=True, **params):
        print(f"\n{nombre}")
        estimator = modelo(
            **params, random_state=42) if 'random_state' in modelo.__init__.__code__.co_varnames else modelo(**params)
        pipeline = make_pipeline(
            StandardScaler(), estimator) if scale else estimator
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        metricas = self._calcular_metricas_clasificacion(
            self.y_test, y_pred, list(np.unique(self.y)))
        for k, v in metricas.items():
            print(f"{k}:\n{v}")
        return pipeline, metricas

    def clasificacion_knn(self, n_neighbors=3, algorithm='auto', scale=True):
        return self._entrenar_clasificador(KNeighborsClassifier, "KNN", scale=scale, n_neighbors=n_neighbors, algorithm=algorithm)

    def clasificacion_decision_tree(self, min_samples_split=2, max_depth=None, scale=False):
        return self._entrenar_clasificador(DecisionTreeClassifier, "Decision Tree", scale=scale,
                                           min_samples_split=min_samples_split, max_depth=max_depth)

    def clasificacion_random_forest(self, n_estimators=100, min_samples_split=2, max_depth=None, scale=False):
        return self._entrenar_clasificador(RandomForestClassifier, "Random Forest", scale=scale,
                                           n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)

    def clasificacion_xgboost(self, n_estimators=100, min_samples_split=2, max_depth=3, scale=False):
        return self._entrenar_clasificador(GradientBoostingClassifier, "XGBoost", scale=scale,
                                           n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)

    def clasificacion_adaboost(self, n_estimators=50, estimator=None, scale=False):
        if estimator is None:
            estimator = DecisionTreeClassifier(max_depth=1)
        return self._entrenar_clasificador(AdaBoostClassifier, "AdaBoost", scale=scale, estimator=estimator, n_estimators=n_estimators)

    def benchmark_clasificacion(self):
        print("\n" + "="*70 + "\nBENCHMARK DE CLASIFICACION\n" + "="*70)
        modelos = {
            'KNN': lambda: self.clasificacion_knn(n_neighbors=3),
            'Decision Tree': lambda: self.clasificacion_decision_tree(max_depth=5),
            'Random Forest': lambda: self.clasificacion_random_forest(n_estimators=100),
            'XGBoost': lambda: self.clasificacion_xgboost(n_estimators=100),
            'AdaBoost': lambda: self.clasificacion_adaboost(n_estimators=50)
        }
        resultados = [{'Modelo': nombre, **{k: v for k, v in func()[1].items() if k != 'Matriz de Confusión'}}
                      for nombre, func in modelos.items()]
        df_res = pd.DataFrame([{k: v for k, v in r.items() if k in ['Modelo', 'Precisión Global', 'Error Global']}
                              for r in resultados]).sort_values('Precisión Global', ascending=False)
        print(f"\nResultados:\n{df_res.to_string(index=False)}")
        return df_res

    @staticmethod
    def _calcular_errores_regresion(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        MSE = np.sum(np.square(y_true - y_pred)) / len(y_true)
        MAE = np.sum(np.abs(y_true - y_pred)) / len(y_true)
        return pd.DataFrame({
            'Métrica': ['RMSE', 'MAE', 'ER'],
            'Valor': [math.sqrt(MSE), MAE, np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))]
        })

    def _entrenar_regresor(self, modelo, nombre, scale=True, **params):
        print(f"\n{nombre}")
        estimator = modelo(
            **params, random_state=42) if 'random_state' in modelo.__init__.__code__.co_varnames else modelo(**params)
        pipeline = make_pipeline(
            StandardScaler(), estimator) if scale else estimator
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        errores = self._calcular_errores_regresion(self.y_test, y_pred)
        print(errores)
        return pipeline, errores

    def regresion_lineal(self, scale=True):
        return self._entrenar_regresor(LinearRegression, "Regresión Lineal", scale=scale)

    def regresion_lasso(self, alpha=0.1, scale=True):
        return self._entrenar_regresor(Lasso, f"Lasso (alpha={alpha})", scale=scale, alpha=alpha)

    def regresion_ridge(self, alpha=1.0, scale=True):
        return self._entrenar_regresor(Ridge, f"Ridge (alpha={alpha})", scale=scale, alpha=alpha)

    def regresion_svm(self, kernel='rbf', C=100, epsilon=0.1, scale=True):
        return self._entrenar_regresor(SVR, f"SVM Regresión (kernel={kernel})", scale=scale, kernel=kernel, C=C, epsilon=epsilon)

    def regresion_decision_tree(self, max_depth=3, scale=False):
        return self._entrenar_regresor(DecisionTreeRegressor, f"Decision Tree (max_depth={max_depth})", scale=scale, max_depth=max_depth)

    def regresion_random_forest(self, n_estimators=100, max_depth=None, scale=False):
        return self._entrenar_regresor(RandomForestRegressor, f"Random Forest (n={n_estimators})", scale=scale,
                                       n_estimators=n_estimators, max_depth=max_depth)

    def regresion_xgboost(self, n_estimators=100, max_depth=4, scale=False):
        return self._entrenar_regresor(GradientBoostingRegressor, f"XGBoost (n={n_estimators})", scale=scale,
                                       n_estimators=n_estimators, max_depth=max_depth)

    def benchmark_regresion(self):
        print("\n" + "="*70 + "\nBENCHMARK DE REGRESION\n" + "="*70)
        modelos = {
            'Lineal': self.regresion_lineal, 'Lasso': self.regresion_lasso, 'Ridge': self.regresion_ridge,
            'SVM (RBF)': lambda: self.regresion_svm(kernel='rbf'), 'Decision Tree': self.regresion_decision_tree,
            'Random Forest': self.regresion_random_forest, 'XGBoost': self.regresion_xgboost
        }
        resultados = []
        for nombre, func in modelos.items():
            _, errores = func()
            resultados.append({
                'Modelo': nombre,
                'RMSE': errores.loc[errores['Métrica'] == 'RMSE', 'Valor'].values[0],
                'MAE': errores.loc[errores['Métrica'] == 'MAE', 'Valor'].values[0],
                'ER': errores.loc[errores['Métrica'] == 'ER', 'Valor'].values[0]
            })
        df_res = pd.DataFrame(resultados).sort_values('RMSE')
        print(f"\nResultados:\n{df_res.to_string(index=False)}")
        return df_res

    def optimizar_con_ga(self, tipo='clasificacion', modelo='random_forest', pop_size=8, generations=8):
        if not GENETIC_AVAILABLE:
            print(
                "ADVERTENCIA: sklearn-genetic-opt no instalado\nInstala: pip install sklearn-genetic-opt")
            return None, None
        print("\n" + "="*70 + "\nOPTIMIZACION CON ALGORITMOS GENETICOS\n" + "="*70)

        if modelo == 'random_forest':
            estimator = RandomForestClassifier(
                random_state=42, n_jobs=-1) if tipo == 'clasificacion' else RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {'n_estimators': Integer(50, 200), 'max_depth': Integer(3, 20),
                          'min_samples_split': Integer(2, 10), 'min_samples_leaf': Integer(1, 5)}
        else:
            estimator = GradientBoostingClassifier(
                random_state=42) if tipo == 'clasificacion' else GradientBoostingRegressor(random_state=42)
            param_grid = {'n_estimators': Integer(50, 200), 'learning_rate': Continuous(0.01, 0.3),
                          'max_depth': Integer(3, 10), 'min_samples_split': Integer(2, 10)}

        ga_search = GASearchCV(estimator=estimator, cv=3,
                               scoring='accuracy' if tipo == 'clasificacion' else 'neg_mean_squared_error',
                               population_size=pop_size, generations=generations, n_jobs=-1, verbose=False, param_grid=param_grid)

        print(f"Ejecutando GA para {modelo} ({tipo})...")
        ga_search.fit(self.X_train, self.y_train)
        print(
            f"\nOptimización completada!\n   Mejor score (CV): {ga_search.best_score_:.4f}\n   Mejores parámetros:")
        for param, valor in ga_search.best_params_.items():
            print(f"      - {param}: {valor}")

        y_pred = ga_search.best_estimator_.predict(self.X_test)
        if tipo == 'clasificacion':
            print(
                f"   Accuracy en Test: {accuracy_score(self.y_test, y_pred):.4f}")
        else:
            print(
                f"\n   Errores en Test:\n{self._calcular_errores_regresion(self.y_test, y_pred).to_string(index=False)}")
        return ga_search.best_estimator_, ga_search

    def validacion_cruzada(self, modelo, n_folds=10, scale=True, scoring='accuracy', **params):
        """
        Validación cruzada con K-Fold para cualquier modelo

        Parámetros:
        -----------
        modelo : clase del modelo (ej: DecisionTreeClassifier, RandomForestRegressor)
        n_folds : int, número de folds (default=10)
        scale : bool, si se aplica escalado (default=True)
        scoring : str, métrica de evaluación (default='accuracy' para clasificación)
                  Para regresión usa: 'neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'
        **params : parámetros adicionales del modelo

        Retorna:
        --------
        dict con resultados de cada fold y promedio
        """
        print(f"\n{'='*70}")
        print(f"VALIDACION CRUZADA - {modelo.__name__}")
        print(f"{'='*70}")
        print(f"Folds: {n_folds}")
        print(f"Métrica: {scoring}")

        # Preparar datos completos (X, y)
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Crear modelo
        estimator = modelo(
            **params, random_state=42) if 'random_state' in modelo.__init__.__code__.co_varnames else modelo(**params)

        # Pipeline con o sin escalado
        pipeline = make_pipeline(
            StandardScaler(), estimator) if scale else estimator

        # K-Fold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Ejecutar validación cruzada
        resultados = cross_val_score(pipeline, X, y, cv=kfold, scoring=scoring)

        # Mostrar resultados
        print(f"\nResultados por fold:")
        for i, score in enumerate(resultados, 1):
            print(f"   Fold {i:2d}: {score:.4f}")

        print(f"\n{'='*70}")
        print(f"Promedio: {resultados.mean():.4f}")
        print(f"Desviación Estándar: {resultados.std():.4f}")
        print(f"Min: {resultados.min():.4f} | Max: {resultados.max():.4f}")
        print(f"{'='*70}")

        return {
            'resultados': resultados,
            'promedio': resultados.mean(),
            'std': resultados.std(),
            'min': resultados.min(),
            'max': resultados.max()
        }

    def validacion_cruzada_completa(self, modelo, n_folds=10, scale=True, **params):
        """
        Validación cruzada con múltiples métricas

        Parámetros:
        -----------
        modelo : clase del modelo
        n_folds : int, número de folds (default=10)
        scale : bool, si se aplica escalado (default=True)
        **params : parámetros adicionales del modelo

        Retorna:
        --------
        DataFrame con todas las métricas
        """
        print(f"\n{'='*70}")
        print(f"VALIDACION CRUZADA COMPLETA - {modelo.__name__}")
        print(f"{'='*70}")

        # Preparar datos
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Crear modelo
        estimator = modelo(
            **params, random_state=42) if 'random_state' in modelo.__init__.__code__.co_varnames else modelo(**params)
        pipeline = make_pipeline(
            StandardScaler(), estimator) if scale else estimator

        # K-Fold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Determinar si es clasificación o regresión
        es_clasificacion = y.dtype == 'object' or len(np.unique(y)) < 20

        if es_clasificacion:
            scoring_metrics = ['accuracy', 'precision_weighted',
                               'recall_weighted', 'f1_weighted']
        else:
            scoring_metrics = ['neg_mean_squared_error',
                               'neg_mean_absolute_error', 'r2']

        # Ejecutar validación cruzada con múltiples métricas
        resultados = cross_validate(pipeline, X, y, cv=kfold, scoring=scoring_metrics,
                                    return_train_score=True)

        # Crear DataFrame de resultados
        df_resultados = pd.DataFrame()
        for metric in scoring_metrics:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'

            # Para MSE y MAE, convertir a positivo
            multiplicador = -1 if 'neg_' in metric else 1

            df_resultados = pd.concat([df_resultados, pd.DataFrame({
                'Métrica': [metric.replace('neg_', '').replace('_weighted', '')],
                'Train (promedio)': [resultados[train_key].mean() * multiplicador],
                'Test (promedio)': [resultados[test_key].mean() * multiplicador],
                'Test (std)': [resultados[test_key].std() * multiplicador]
            })], ignore_index=True)

        print(f"\nResultados:")
        print(df_resultados.to_string(index=False))
        print(f"{'='*70}")

        return df_resultados

    # Funciones para Series de Tiempo
    def arima_model(self, order=(1, 1, 1)):
        """
        Modelo ARIMA para series de tiempo
        Nota: Utiliza la clase SeriesTiempo para análisis completo
        """
        if not STATSMODELS_AVAILABLE:
            print("ADVERTENCIA: statsmodels no disponible")
            print("Instala: pip install statsmodels")
            return None

        print(f"\nModelo ARIMA{order}")
        print("Para análisis completo usa: SeriesTiempo(ts).arima(...)")
        print("Ejemplo de uso directo con SARIMAX:")
        print(f"  from statsmodels.tsa.statespace.sarimax import SARIMAX")
        print(
            f"  modelo = SARIMAX(serie, order={order}, seasonal_order=(0,0,0,0))")
        print(f"  resultado = modelo.fit()")
        print(f"  prediccion = resultado.forecast(steps=10)")
        return None

    def sarima_model(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Modelo SARIMA para series de tiempo con componente estacional
        Nota: Utiliza la clase SeriesTiempo para análisis completo
        """
        if not STATSMODELS_AVAILABLE:
            print("ADVERTENCIA: statsmodels no disponible")
            print("Instala: pip install statsmodels")
            return None

        print(f"\nModelo SARIMA{order}x{seasonal_order}")
        print("Para análisis completo usa: SeriesTiempo(ts)")
        print("Ejemplo de uso directo con SARIMAX:")
        print(f"  from statsmodels.tsa.statespace.sarimax import SARIMAX")
        print(
            f"  modelo = SARIMAX(serie, order={order}, seasonal_order={seasonal_order})")
        print(f"  resultado = modelo.fit()")
        print(f"  prediccion = resultado.forecast(steps=10)")
        return None

    def prophet_model(self, seasonality_mode='additive', changepoint_prior_scale=0.05):
        """
        Modelo Prophet de Facebook para pronósticos
        Requiere: pip install prophet (fbprophet)
        """
        print("\nModelo Prophet")
        print("Requiere: pip install prophet")
        print("Documentación: https://facebook.github.io/prophet/")
        print("\nEjemplo de uso:")
        print("  from prophet import Prophet")
        print("  # df debe tener columnas 'ds' (fechas) y 'y' (valores)")
        print(
            f"  modelo = Prophet(seasonality_mode='{seasonality_mode}', changepoint_prior_scale={changepoint_prior_scale})")
        print("  modelo.fit(df)")
        print("  future = modelo.make_future_dataframe(periods=365)")
        print("  forecast = modelo.predict(future)")
        print("  modelo.plot(forecast)")
        return None

    def exponential_smoothing(self, seasonal='add', seasonal_periods=12):
        """
        Suavizado Exponencial (Holt-Winters)
        Usa la clase SeriesTiempo para implementación completa
        """
        if not STATSMODELS_AVAILABLE:
            print("ADVERTENCIA: statsmodels no disponible")
            print("Instala: pip install statsmodels")
            return None

        print("\nSuavizado Exponencial (Holt-Winters)")
        print("Para uso directo con SeriesTiempo:")
        print("  st = SeriesTiempo(ts=tu_serie)")
        print("  modelo = st.holt_winters(trend='add', seasonal='add')")
        print("  prediccion = modelo.forecast(steps=10)")
        print("\nO con calibración automática:")
        print("  train, test = st.train_test_split(test_size=0.2)")
        print("  st_train = SeriesTiempo(ts=train)")
        print(
            "  modelo_calibrado = st_train.holt_winters_calibrado(test, paso=0.1)")
        print("  prediccion = modelo_calibrado.forecast(steps=10)")
        return None

# ============================================================================
# CLASE NO SUPERVISADO
# ============================================================================


class NoSupervisado(EDA):
    def __init__(self, df):
        super().__init__(df=df)
        self.df_scaled = None
        print(f"NoSupervisado inicializado - Shape: {self.df.shape}")

    def escalar_datos(self):
        """Escala datos manualmente (opcional - los pipelines lo hacen automático)"""
        self.df_scaled = pd.DataFrame(StandardScaler().fit_transform(self.df),
                                      columns=self.df.columns, index=self.df.index)
        print(f"Datos escalados: {self.df_scaled.shape}")
        return self

    def pca(self, n_componentes=2, plot=True):
        if PCA_Prince is None:
            print("ADVERTENCIA: Librería 'prince' no disponible, usando sklearn PCA")
            return self.pca_sklearn(n_componentes, plot)
        print(f"\nPCA con {n_componentes} componentes")
        datos = self.df_scaled if self.df_scaled is not None else self.df
        modelo = PCA_Prince(n_components=n_componentes).fit(datos)
        coordenadas = modelo.row_coordinates(datos)
        var_explicada = modelo.percentage_of_variance_
        print(f"\nVarianza explicada por componente:")
        for i, var in enumerate(var_explicada):
            print(f"   PC{i+1}: {var:.2f}%")
        print(f"   Total: {sum(var_explicada):.2f}%")
        if plot and n_componentes >= 2:
            self._plot_pca(coordenadas, var_explicada,
                           modelo.column_correlations)
        return modelo, coordenadas, var_explicada

    def pca_sklearn(self, n_componentes=2, plot=True, scale=True):
        print(f"\nPCA (sklearn) con {n_componentes} componentes")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            pipeline = make_pipeline(
                StandardScaler(), PCA(n_components=n_componentes))
            componentes = pipeline.fit_transform(datos)
            pca = pipeline.named_steps['pca']
        else:
            pca = PCA(n_components=n_componentes)
            componentes = pca.fit_transform(datos)
            pipeline = pca

        var_explicada = pca.explained_variance_ratio_ * 100
        print(
            f"\nVarianza explicada: {[f'{v:.2f}%' for v in var_explicada]}, Total: {sum(var_explicada):.2f}%")
        if plot and n_componentes >= 2:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6)
            ax.set_xlabel(f'PC1 ({var_explicada[0]:.2f}%)')
            ax.set_ylabel(f'PC2 ({var_explicada[1]:.2f}%)')
            ax.set_title('PCA - Plano Principal')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        return pipeline, componentes, var_explicada

    def _plot_pca(self, coordenadas, var_explicada, correlaciones):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
        x, y = coordenadas[0].values, coordenadas[1].values
        axes[0].scatter(x, y, alpha=0.6)
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel(f'PC1 ({var_explicada[0]:.2f}%)')
        axes[0].set_ylabel(f'PC2 ({var_explicada[1]:.2f}%)')
        axes[0].set_title('Plano Principal')
        axes[0].grid(True, alpha=0.3)

        cor = correlaciones.iloc[:, [0, 1]].values
        circle = plt.Circle((0, 0), 1, color='steelblue', fill=False)
        axes[1].add_patch(circle)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        for i in range(cor.shape[0]):
            axes[1].arrow(0, 0, cor[i, 0]*0.95, cor[i, 1]*0.95,
                          color='steelblue', alpha=0.6, head_width=0.05)
            axes[1].text(cor[i, 0]*1.1, cor[i, 1]*1.1,
                         correlaciones.index[i], fontsize=9, ha='center')
        axes[1].set_xlim(-1.2, 1.2)
        axes[1].set_ylim(-1.2, 1.2)
        axes[1].set_xlabel(f'PC1 ({var_explicada[0]:.2f}%)')
        axes[1].set_ylabel(f'PC2 ({var_explicada[1]:.2f}%)')
        axes[1].set_title('Círculo de Correlación')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _ejecutar_clustering(self, modelo, nombre, n_clusters, plot, scale=True, **kwargs):
        print(f"\n{nombre} con {n_clusters} clusters")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            pipeline = make_pipeline(StandardScaler(), modelo)
            clusters = pipeline.fit_predict(datos)
            centros = pipeline.named_steps[list(pipeline.named_steps.keys())[
                1]].cluster_centers_
            # Escalamos datos para visualización
            datos_escalados = pipeline.named_steps['standardscaler'].transform(
                datos)
        else:
            pipeline = modelo
            clusters = modelo.fit_predict(datos)
            centros = modelo.cluster_centers_
            datos_escalados = datos

        print(f"\nDistribución:")
        for i in range(n_clusters):
            count = np.sum(clusters == i)
            print(f"   Cluster {i}: {count} ({count/len(clusters)*100:.1f}%)")
        if plot:
            self._plot_clusters(datos_escalados, clusters, centros, nombre)
        return pipeline, clusters, centros

    def kmeans(self, n_clusters=3, max_iter=500, n_init=150, plot=True, scale=True):
        return self._ejecutar_clustering(KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=42),
                                         "K-Means", n_clusters, plot, scale=scale)

    def kmedoids(self, n_clusters=3, max_iter=500, plot=True, scale=True):
        return self._ejecutar_clustering(KMedoids(n_clusters=n_clusters, max_iter=max_iter, metric='cityblock', random_state=42),
                                         "K-Medoids", n_clusters, plot, scale=scale)

    def _plot_clusters(self, datos, clusters, centros, titulo):
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(datos)
        centros_pca = pca.transform(centros)
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        colores = ['red', 'green', 'blue', 'orange',
                   'purple', 'brown', 'pink', 'gray']
        for i in range(len(centros)):
            mask = clusters == i
            ax.scatter(componentes[mask, 0], componentes[mask, 1], c=colores[i % len(colores)],
                       label=f'Cluster {i}', alpha=0.6, s=50)
        ax.scatter(centros_pca[:, 0], centros_pca[:, 1], c='black', marker='X', s=200,
                   label='Centroides', edgecolors='white', linewidths=2)
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_title(f'{titulo} - Visualización con PCA')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def hac(self, n_clusters=3, metodo='ward', plot=True, scale=True):
        print(f"\nClustering Jerárquico ({metodo}) con {n_clusters} clusters")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        # Escalar si es necesario
        if scale and self.df_scaled is None:
            scaler = StandardScaler()
            datos_para_hac = scaler.fit_transform(datos)
        else:
            datos_para_hac = datos

        Z = {'ward': ward, 'average': average, 'single': single,
             'complete': complete}[metodo](datos_para_hac)
        clusters = fcluster(Z, n_clusters, criterion='maxclust') - 1
        print(f"\nDistribución:")
        for i in range(n_clusters):
            count = np.sum(clusters == i)
            print(f"   Cluster {i}: {count} ({count/len(clusters)*100:.1f}%)")
        if plot:
            fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
            dendrogram(Z, labels=datos.index.tolist(), ax=ax)
            ax.set_title(f'Dendrograma - Método {metodo.capitalize()}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
        return Z, clusters

    def tsne(self, n_componentes=2, perplexity=30, plot=True, scale=True):
        if TSNE is None:
            print("ADVERTENCIA: t-SNE no disponible")
            return None, None
        print(f"\nt-SNE con {n_componentes} componentes")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            pipeline = make_pipeline(StandardScaler(), TSNE(
                n_components=n_componentes, perplexity=perplexity, random_state=42))
            componentes = pipeline.fit_transform(datos)
        else:
            componentes = TSNE(n_components=n_componentes,
                               perplexity=perplexity, random_state=42).fit_transform(datos)
            pipeline = None

        if plot and n_componentes >= 2:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6)
            ax.set_xlabel('Componente t-SNE 1')
            ax.set_ylabel('Componente t-SNE 2')
            ax.set_title('t-SNE - Reducción Dimensional')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        print("t-SNE completado")
        return pipeline, componentes

    def umap_reduction(self, n_componentes=2, n_neighbors=15, plot=True, scale=True):
        if um is None:
            print("ADVERTENCIA: UMAP no disponible. Instala: pip install umap-learn")
            return None, None
        print(f"\nUMAP con {n_componentes} componentes")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            pipeline = make_pipeline(StandardScaler(), um.UMAP(
                n_components=n_componentes, n_neighbors=n_neighbors, random_state=42))
            componentes = pipeline.fit_transform(datos)
        else:
            modelo_umap = um.UMAP(n_components=n_componentes,
                                  n_neighbors=n_neighbors, random_state=42)
            componentes = modelo_umap.fit_transform(datos)
            pipeline = modelo_umap

        if plot and n_componentes >= 2:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6)
            ax.set_xlabel('Componente UMAP 1')
            ax.set_ylabel('Componente UMAP 2')
            ax.set_title('UMAP - Reducción Dimensional')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        print("UMAP completado")
        return pipeline, componentes

    @staticmethod
    def bar_plot(centros, labels, scale=False, figsize=(15, 8)):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        centros_plot = np.copy(centros)
        if scale:
            for col in range(centros_plot.shape[1]):
                max_val = np.max(np.abs(centros_plot[:, col]))
                if max_val > 0:
                    centros_plot[:, col] = centros_plot[:, col] / max_val
        n_clusters, x, width = centros_plot.shape[0], np.arange(
            len(labels)), 0.8 / centros_plot.shape[0]
        for i in range(n_clusters):
            ax.bar(x + width * i - (width * (n_clusters - 1) / 2), centros_plot[i],
                   width, label=f'Cluster {i}', alpha=0.8)
        ax.set_xlabel('Variables')
        ax.set_ylabel('Valor')
        ax.set_title('Comparación de Centroides por Cluster')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def radar_plot(centros, labels):
        centros_norm = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if max(n) != min(n) else (n/n * 50)
                                for n in centros.T])
        angulos = [n / float(len(labels)) * 2 *
                   pi for n in range(len(labels))] + [0]
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150,
                               subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], labels)
        plt.yticks([25, 50, 75, 100], ["25%", "50%",
                   "75%", "100%"], color="grey", size=8)
        plt.ylim(0, 100)
        colores = ['blue', 'red', 'green', 'orange', 'purple']
        for i in range(centros_norm.shape[1]):
            valores = centros_norm[:, i].tolist(
            ) + [centros_norm[:, i].tolist()[0]]
            ax.plot(angulos, valores, linewidth=2,
                    label=f'Cluster {i}', color=colores[i % len(colores)])
            ax.fill(angulos, valores, alpha=0.25,
                    color=colores[i % len(colores)])
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Radar Plot - Comparación de Clusters', y=1.08)
        plt.tight_layout()
        plt.show()

# ============================================================================
# CLASES DE SERIES DE TIEMPO
# ============================================================================


class BasePrediccion(metaclass=ABCMeta):
    @abstractmethod
    def forecast(self):
        pass


class Prediccion(BasePrediccion):
    def __init__(self, modelo):
        self.__modelo = modelo

    @property
    def modelo(self):
        return self.__modelo

    @modelo.setter
    def modelo(self, modelo):
        if isinstance(modelo, Modelo):
            self.__modelo = modelo
        else:
            warnings.warn('El objeto debe ser una instancia de Modelo.')


class BaseModelo(metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        pass


class Modelo(BaseModelo):
    def __init__(self, ts):
        self.__ts = ts
        self._coef = None

    @property
    def ts(self):
        return self.__ts

    @ts.setter
    def ts(self, ts):
        if isinstance(ts, pd.core.series.Series):
            if ts.index.freqstr is not None:
                self.__ts = ts
            else:
                warnings.warn(
                    'ERROR: No se indica la frecuencia de la serie de tiempo.')
        else:
            warnings.warn(
                'ERROR: El parámetro ts no es una instancia de serie de tiempo.')

    @property
    def coef(self):
        return self._coef


# Modelos Básicos de Predicción
class meanfPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = [self.modelo.coef for _ in range(steps)]
        start = self.modelo.ts.index[-1]
        freq = self.modelo.ts.index.freqstr
        fechas = pd.date_range(start=start, periods=steps+1, freq=freq)
        fechas = fechas.delete(0)
        return pd.Series(res, index=fechas)


class naivePrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = [self.modelo.coef for _ in range(steps)]
        start = self.modelo.ts.index[-1]
        freq = self.modelo.ts.index.freqstr
        fechas = pd.date_range(start=start, periods=steps+1, freq=freq)
        fechas = fechas.delete(0)
        return pd.Series(res, index=fechas)


class snaivePrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = []
        pos = 0
        for _ in range(steps):
            if pos >= len(self.modelo.coef):
                pos = 0
            res.append(self.modelo.coef[pos])
            pos += 1
        start = self.modelo.ts.index[-1]
        freq = self.modelo.ts.index.freqstr
        fechas = pd.date_range(start=start, periods=steps+1, freq=freq)
        fechas = fechas.delete(0)
        return pd.Series(res, index=fechas)


class driftPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = [self.modelo.ts[-1] + self.modelo.coef * i for i in range(steps)]
        start = self.modelo.ts.index[-1]
        freq = self.modelo.ts.index.freqstr
        fechas = pd.date_range(start=start, periods=steps+1, freq=freq)
        fechas = fechas.delete(0)
        return pd.Series(res, index=fechas)


class meanf(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self):
        self._coef = statistics.mean(self.ts)
        return meanfPrediccion(self)


class naive(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self):
        self._coef = self.ts[-1]
        return naivePrediccion(self)


class snaive(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self, h=1):
        self._coef = self.ts.values[-h:]
        return snaivePrediccion(self)


class drift(Modelo):
    def __init__(self, ts):
        super().__init__(ts)

    def fit(self):
        self._coef = (self.ts[-1] - self.ts[0]) / len(self.ts)
        return driftPrediccion(self)


# Holt-Winters
class HW_Prediccion(Prediccion):
    def __init__(self, modelo, alpha, beta, gamma):
        super().__init__(modelo)
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

    @property
    def alpha(self):
        return self.__alpha

    @property
    def beta(self):
        return self.__beta

    @property
    def gamma(self):
        return self.__gamma

    def forecast(self, steps=1):
        return self.modelo.forecast(steps)


class HW_calibrado(Modelo):
    def __init__(self, ts, test, trend='add', seasonal='add'):
        super().__init__(ts)
        self.__test = test
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels no disponible. Instala: pip install statsmodels")
        self.__modelo = ExponentialSmoothing(
            ts, trend=trend, seasonal=seasonal)

    @property
    def test(self):
        return self.__test

    @test.setter
    def test(self, test):
        if isinstance(test, pd.core.series.Series):
            if test.index.freqstr is not None:
                self.__test = test
            else:
                warnings.warn(
                    'ERROR: No se indica la frecuencia de la serie de tiempo.')
        else:
            warnings.warn(
                'ERROR: El parámetro ts no es una instancia de serie de tiempo.')

    def fit(self, paso=0.1):
        error = float("inf")
        n = np.append(np.arange(0, 1, paso), 1)
        for alpha in n:
            for beta in n:
                for gamma in n:
                    model_fit = self.__modelo.fit(
                        smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
                    pred = model_fit.forecast(len(self.test))
                    mse = sum((pred - self.test)**2)
                    if mse < error:
                        res_alpha = alpha
                        res_beta = beta
                        res_gamma = gamma
                        error = mse
                        res = model_fit
        return HW_Prediccion(res, res_alpha, res_beta, res_gamma)


# LSTM para Series de Tiempo
class LSTM_TSPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)
        if not KERAS_AVAILABLE:
            raise ImportError(
                "Keras/TensorFlow no disponible. Instala: pip install tensorflow")
        self.__scaler = MinMaxScaler(feature_range=(0, 1))
        self.__X = self.__scaler.fit_transform(self.modelo.ts.to_frame())

    def __split_sequence(self, sequence, n_steps):
        X, y = [], []
        for i in range(n_steps, len(sequence)):
            X.append(self.__X[i-n_steps:i, 0])
            y.append(self.__X[i, 0])
        return np.array(X), np.array(y)

    def forecast(self, steps=1):
        res = []
        p = self.modelo.p
        for i in range(steps):
            y_pred = [self.__X[-p:].tolist()]
            X, y = self.__split_sequence(self.__X, p)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            self.modelo.m.fit(X, y, epochs=10, batch_size=1, verbose=0)
            pred = self.modelo.m.predict(y_pred, verbose=0)
            res.append(
                self.__scaler.inverse_transform(pred).tolist()[0][0])
            self.__X = np.append(self.__X, pred.tolist(), axis=0)

        start = self.modelo.ts.index[-1]
        freq = self.modelo.ts.index.freqstr
        fechas = pd.date_range(start=start, periods=steps+1, freq=freq)
        fechas = fechas.delete(0)
        return pd.Series(res, index=fechas)


class LSTM_TS(Modelo):
    def __init__(self, ts, p=1, lstm_units=50, dense_units=1, optimizer='rmsprop', loss='mse'):
        super().__init__(ts)
        if not KERAS_AVAILABLE:
            raise ImportError(
                "Keras/TensorFlow no disponible. Instala: pip install tensorflow")
        self.__p = p
        self.__m = Sequential()
        self.__m.add(LSTM(units=lstm_units, input_shape=(p, 1)))
        self.__m.add(Dense(units=dense_units))
        self.__m.compile(optimizer=optimizer, loss=loss)

    @property
    def m(self):
        return self.__m

    @property
    def p(self):
        return self.__p

    def fit(self):
        return LSTM_TSPrediccion(self)


# Clase de Errores para Series de Tiempo
class ts_error:
    def __init__(self, preds, real, nombres=None):
        self.__preds = preds if isinstance(
            preds, list) else [preds]
        self.__real = real
        self.__nombres = nombres

    @property
    def preds(self):
        return self.__preds

    @preds.setter
    def preds(self, preds):
        if isinstance(preds, (pd.core.series.Series, np.ndarray)):
            self.__preds = [preds]
        elif isinstance(preds, list):
            self.__preds = preds
        else:
            warnings.warn(
                'ERROR: El parámetro preds debe ser una serie de tiempo o una lista de series de tiempo.')

    @property
    def real(self):
        return self.__real

    @real.setter
    def real(self, real):
        self.__real = real

    @property
    def nombres(self):
        return self.__nombres

    @nombres.setter
    def nombres(self, nombres):
        if isinstance(nombres, str):
            nombres = [nombres]
        if len(nombres) == len(self.__preds):
            self.__nombres = nombres
        else:
            warnings.warn(
                'ERROR: Los nombres no calzan con la cantidad de métodos.')

    def RSS(self):
        return [sum((pred - self.real)**2) for pred in self.preds]

    def MSE(self):
        return [rss / len(self.real) for rss in self.RSS()]

    def RMSE(self):
        return [math.sqrt(mse) for mse in self.MSE()]

    def RE(self):
        return [sum(abs(self.real - pred)) / sum(abs(self.real)) for pred in self.preds]

    def CORR(self):
        res = []
        for pred in self.preds:
            corr = corrcoef(self.real, pred)[0, 1]
            res.append(0 if math.isnan(corr) else corr)
        return res

    def df_errores(self):
        res = pd.DataFrame({'MSE': self.MSE(), 'RMSE': self.RMSE(),
                           'RE': self.RE(), 'CORR': self.CORR()})
        if self.nombres is not None:
            res.index = self.nombres
        return res

    def __escalar(self):
        res = self.df_errores()
        for nombre in res.columns.values:
            res[nombre] = res[nombre] - min(res[nombre])
            max_val = max(res[nombre])
            if max_val > 0:
                res[nombre] = res[nombre] / max_val * 100
        return res

    def plot_errores(self):
        plt.figure(figsize=(8, 8))
        df = self.__escalar()
        if len(df) == 1:
            df.loc[0] = 100

        N = len(df.columns.values)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], df.columns.values)
        ax.set_rlabel_position(0)
        plt.yticks([0, 25, 50, 75, 100], ["0%", "25%",
                   "50%", "75%", "100%"], color="grey", size=10)
        plt.ylim(-10, 110)

        for i in df.index.values:
            p = df.loc[i].values.tolist()
            p = p + p[:1]
            ax.plot(angles, p, linewidth=1, linestyle='solid', label=i)
            ax.fill(angles, p, alpha=0.1)

        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def plotly_errores(self):
        if not PLOTLY_AVAILABLE:
            print("Plotly no disponible. Usando matplotlib...")
            return self.plot_errores()

        df = self.__escalar()
        etqs = df.columns.values.tolist()
        etqs = etqs + etqs[:1]
        if len(df) == 1:
            df.loc[0] = 100

        fig = go.Figure()
        for i in df.index.values:
            p = df.loc[i].values.tolist()
            p = p + p[:1]
            fig.add_trace(go.Scatterpolar(
                r=p, theta=etqs, fill='toself', name=i
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[-10, 110])
            ))
        return fig


# Clase Periodograma
class Periodograma:
    def __init__(self, ts):
        self.__ts = ts
        self.__freq, self.__spec = signal.periodogram(ts)

    @property
    def ts(self):
        return self.__ts

    @property
    def freq(self):
        return self.__freq

    @property
    def spec(self):
        return self.__spec

    def mejor_freq(self, best=3):
        res = np.argsort(-self.spec)
        res = res[res != 0][0:best]
        return self.freq[res]

    def mejor_periodos(self, best=3):
        return 1 / self.mejor_freq(best)

    def plot_periodograma(self, best=3):
        res = self.mejor_freq(best)
        plt.figure(figsize=(10, 6))
        plt.plot(self.freq, self.spec, color="darkgray")
        for i in range(best):
            plt.axvline(x=res[i], label=f"Mejor {i + 1}",
                        ls='--', c=np.random.rand(3,))
        plt.xlabel('Frecuencia')
        plt.ylabel('Densidad Espectral')
        plt.title('Periodograma')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plotly_periodograma(self, best=3):
        if not PLOTLY_AVAILABLE:
            print("Plotly no disponible. Usando matplotlib...")
            return self.plot_periodograma(best)

        res = self.mejor_freq(best)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=self.freq, y=self.spec,
                       mode='lines+markers', line_color='darkgray')
        )
        for i in range(best):
            v = np.random.rand(3)
            color = f"rgb({v[0]}, {v[1]}, {v[2]})"
            fig.add_vline(x=res[i], line_width=2, line_dash="dash",
                          annotation_text=f"Mejor {i + 1}",
                          line_color=color)
        fig.update_layout(
            title='Periodograma',
            xaxis_title='Frecuencia',
            yaxis_title='Densidad Espectral'
        )
        return fig


# Clase Principal de Series de Tiempo
class SeriesTiempo:

    def __init__(self, ts=None, path=None, date_col='fecha', value_col=None, freq='D'):

        if ts is not None:
            if not isinstance(ts, pd.Series):
                raise ValueError("ts debe ser una pandas Series")
            self.ts = ts
        elif path is not None:
            df = pd.read_csv(path)
            if value_col is None:
                value_col = [
                    c for c in df.columns if c != date_col][0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            self.ts = df[value_col]
            self.ts.index.freq = freq
        else:
            raise ValueError("Debe proporcionar 'ts' o 'path'")

        print(f"Serie de tiempo cargada: {len(self.ts)} observaciones")
        if self.ts.index.freqstr:
            print(f"Frecuencia: {self.ts.index.freqstr}")

    def info(self):
        print("\n" + "="*70)
        print("INFORMACION DE SERIE DE TIEMPO")
        print("="*70)
        print(f"Observaciones: {len(self.ts)}")
        print(f"Frecuencia: {self.ts.index.freqstr}")
        print(f"Rango: {self.ts.index[0]} a {self.ts.index[-1]}")
        print(f"\nEstadísticas:\n{self.ts.describe()}")
        return self

    def plot(self, title='Serie de Tiempo', figsize=(12, 6)):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        self.ts.plot(ax=ax, marker='o', markersize=3)
        ax.set_title(title)
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valor')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return self

    def plotly_plot(self, title='Serie de Tiempo'):
        if not PLOTLY_AVAILABLE:
            print("Plotly no disponible. Usando matplotlib...")
            return self.plot(title)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=self.ts.index, y=self.ts.values,
                       mode='lines+markers', name='Serie')
        )
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(title=title, xaxis_title='Fecha',
                          yaxis_title='Valor')
        fig.show()
        return self

    def meanf(self):
        modelo = meanf(self.ts)
        return modelo.fit()

    def naive(self):
        modelo = naive(self.ts)
        return modelo.fit()

    def snaive(self, h=1):
        modelo = snaive(self.ts)
        return modelo.fit(h)

    def drift(self):
        modelo = drift(self.ts)
        return modelo.fit()

    def holt_winters(self, trend='add', seasonal='add'):
        if not STATSMODELS_AVAILABLE:
            print("statsmodels no disponible. Instala: pip install statsmodels")
            return None
        modelo = ExponentialSmoothing(
            self.ts, trend=trend, seasonal=seasonal)
        modelo_fit = modelo.fit()
        print(f"Holt-Winters ajustado (trend={trend}, seasonal={seasonal})")
        return modelo_fit

    def holt_winters_calibrado(self, test, paso=0.1, trend='add', seasonal='add'):
        modelo = HW_calibrado(self.ts, test, trend, seasonal)
        resultado = modelo.fit(paso)
        print(
            f"HW Calibrado - alpha: {resultado.alpha:.3f}, beta: {resultado.beta:.3f}, gamma: {resultado.gamma:.3f}")
        return resultado

    def lstm(self, p=1, lstm_units=50, dense_units=1, optimizer='rmsprop', loss='mse'):
        modelo = LSTM_TS(self.ts, p, lstm_units,
                         dense_units, optimizer, loss)
        return modelo.fit()

    def periodograma(self, best=3, plot=True):
        periodo = Periodograma(self.ts)
        print("\nAnálisis de Periodicidad:")
        print(f"Mejores frecuencias: {periodo.mejor_freq(best)}")
        print(f"Mejores períodos: {periodo.mejor_periodos(best)}")
        if plot:
            periodo.plot_periodograma(best)
        return periodo

    @staticmethod
    def calcular_errores(predicciones, valores_reales, nombres=None):

        errores = ts_error(predicciones, valores_reales, nombres)
        df_errores = errores.df_errores()
        print("\nMétricas de Error:")
        print(df_errores)
        return errores

    def train_test_split(self, test_size=0.2):
        n_test = int(len(self.ts) * test_size)
        train = self.ts[:-n_test]
        test = self.ts[-n_test:]
        print(
            f"Train: {len(train)} observaciones, Test: {len(test)} observaciones")
        return train, test


# ============================================================================
# CLASE WEB SCRAPING
# ============================================================================


class WebScraping:
    def __init__(self, headers=None):
        self.session = None
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        print("WebScraping inicializado")

    def iniciar_sesion(self):
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update(self.headers)
            print("Sesión iniciada")
            return self
        except ImportError:
            print("ERROR: requests no instalado. Instala: pip install requests")
            return None

    def obtener_html(self, url, timeout=10):
        try:
            import requests
            response = (self.session or requests).get(
                url, headers=self.headers if not self.session else {}, timeout=timeout)
            response.raise_for_status()
            print(f"HTML obtenido de: {url}")
            return response.text
        except ImportError:
            print("ERROR: requests no instalado")
            return None
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def parsear_html(self, html):
        try:
            from bs4 import BeautifulSoup
            print("HTML parseado")
            return BeautifulSoup(html, 'html.parser')
        except ImportError:
            print(
                "ERROR: beautifulsoup4 no instalado. Instala: pip install beautifulsoup4")
            return None

    def scrape_tabla_simple(self, url, selector='table', indice_tabla=0):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        tablas = soup.find_all(selector)
        if not tablas or indice_tabla >= len(tablas):
            print(f"No se encontraron tablas válidas")
            return None

        tabla = tablas[indice_tabla]
        headers = [th.get_text(strip=True) for th in tabla.find(
            'thead').find_all('th')] if tabla.find('thead') else []
        tbody = tabla.find('tbody') or tabla
        rows = [[td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                for tr in tbody.find_all('tr') if tr.find_all(['td', 'th'])]
        df = pd.DataFrame(rows, columns=headers if headers else None)
        print(f"Tabla extraída: {df.shape}")
        return df

    def scrape_texto(self, url, selector):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        textos = [elem.get_text(strip=True) for elem in soup.select(selector)]
        print(f"Extraídos {len(textos)} textos")
        return textos

    def scrape_enlaces(self, url, filtro=None):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        enlaces = [link['href'] for link in soup.find_all(
            'a', href=True) if filtro is None or filtro in link['href']]
        print(f"Extraídos {len(enlaces)} enlaces")
        return enlaces

    def scrape_imagenes(self, url, atributo_src='src'):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        imagenes = [img.get(atributo_src)
                    for img in soup.find_all('img') if img.get(atributo_src)]
        print(f"Extraídas {len(imagenes)} imágenes")
        return imagenes

    def scrape_multiples_paginas(self, urls, funcion_scraping, **kwargs):
        import time
        print(f"Scraping de {len(urls)} páginas...")
        resultados = []
        for i, url in enumerate(urls, 1):
            print(f"Procesando {i}/{len(urls)}: {url}")
            resultados.append(funcion_scraping(url, **kwargs))
            if i < len(urls):
                time.sleep(1)
        print("Scraping completado")
        return resultados

    def extraer_metadata(self, url):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        metadata = {
            'titulo': soup.find('title').get_text(strip=True) if soup.find('title') else None,
            'descripcion': soup.find('meta', attrs={'name': 'description'}).get('content') if soup.find('meta', attrs={'name': 'description'}) else None,
            'keywords': soup.find('meta', attrs={'name': 'keywords'}).get('content') if soup.find('meta', attrs={'name': 'keywords'}) else None,
            'autor': soup.find('meta', attrs={'name': 'author'}).get('content') if soup.find('meta', attrs={'name': 'author'}) else None
        }
        print("Metadata extraída:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")
        return metadata

    def descargar_archivo(self, url, nombre_archivo=None, directorio='descargas'):
        try:
            import requests
            import os
            if not os.path.exists(directorio):
                os.makedirs(directorio)
            nombre_archivo = nombre_archivo or url.split('/')[-1]
            ruta = os.path.join(directorio, nombre_archivo)
            response = requests.get(url, headers=self.headers, stream=True)
            response.raise_for_status()
            with open(ruta, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Archivo descargado: {ruta}")
            return ruta
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def cerrar_sesion(self):
        if self.session:
            self.session.close()
            print("Sesión cerrada")
        self.session = None
