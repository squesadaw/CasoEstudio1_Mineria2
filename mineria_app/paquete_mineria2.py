"""
Paquete mineria avanzada- Jorge Chacon, Stacy Quesada
Clases: EDA, Supervisado (hereda EDA), NoSupervisado (hereda EDA), WebScraping
"""
import warnings
import inspect
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
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                          TimeSeriesSplit, cross_val_score, cross_validate)
from sklearn.metrics import (confusion_matrix, accuracy_score, get_scorer,
                               precision_score, recall_score, f1_score,
                               mean_squared_error, mean_absolute_error, r2_score)
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # backend no-interactivo; compatible con Streamlit
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


def _tiene_parametro(clase, nombre_param):
    """Verifica de forma segura si un modelo sklearn acepta un parámetro dado.
    
    FIX: reemplaza el uso frágil de __code__.co_varnames que fallaba con
    clases que usan herencia o *args/**kwargs.
    """
    try:
        sig = inspect.signature(clase.__init__)
        return nombre_param in sig.parameters
    except (ValueError, TypeError):
        return False


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

    # ------------------------------------------------------------------
    # FIX (Streamlit): todos los métodos de graficación retornan fig.
    # El parámetro show=True mantiene compatibilidad con uso standalone.
    # ------------------------------------------------------------------

    def _plot(self, kind, title, figsize=(12, 8), show=True, **kwargs):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        if kind == 'box':
            self.df.boxplot(ax=ax)
        elif kind in ['density', 'hist']:
            self.df.plot(kind=kind, ax=ax, **kwargs)
        plt.title(title)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def grafico_boxplot(self, figsize=(15, 8), show=True):
        return self._plot('box', 'Boxplot - Outliers', figsize, show=show)

    def grafico_densidad(self, figsize=(12, 8), show=True):
        return self._plot('density', 'Función de Densidad', figsize, show=show)

    def grafico_histograma(self, figsize=(10, 6), show=True):
        return self._plot('hist', 'Histograma', figsize, show=show, alpha=0.7)

    def matriz_correlacion(self, figsize=(12, 8), mostrar_valores=True, show=True):
        corr = self.df.corr(numeric_only=True)
        print(f"\nMatriz de Correlación:\n{corr}")
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=sns.diverging_palette(220, 10, as_cmap=True).reversed(),
                    square=True, annot=mostrar_valores, fmt='.2f', ax=ax)
        plt.title("Mapa de Calor - Correlaciones")
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def analisis_completo_visual(self, show=True):
        print("\nEjecutando análisis visual completo...")
        figs = {
            'boxplot':     self.grafico_boxplot(show=show),
            'densidad':    self.grafico_densidad(show=show),
            'histograma':  self.grafico_histograma(show=show),
            'correlacion': self.matriz_correlacion(show=show),
        }
        print("Análisis completado")
        return figs

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

    # ------------------------------------------------------------------
    # Codificación interna: garantiza que df_encoded siempre exista.
    # FIX: evita data leakage en CV por falta de df_encoded.
    # ------------------------------------------------------------------
    def _asegurar_df_encoded(self):
        if not hasattr(self, 'df_encoded'):
            df_encoded = self.df.copy()
            for col in df_encoded.columns:
                if df_encoded[col].dtype == 'object':
                    df_encoded[col] = df_encoded[col].astype('category').cat.codes
            self.df_encoded = df_encoded

    def preparar_datos(self, test_size=0.25, random_state=42):
        """Prepara datos para clasificación o regresión sin escalado (se hace en pipeline)."""
        df_encoded = self.df.copy()
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                df_encoded[col] = df_encoded[col].astype('category').cat.codes
        self.df_encoded = df_encoded

        X = df_encoded.drop(columns=[self.target_col])
        y = df_encoded[self.target_col]
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

    def _balance_data(self, X, y, method=None, random_state=42):
        """Balancea clases en el conjunto de entrenamiento.

        Métodos soportados:
          - None / 'none': no cambia
          - 'oversample': sobre-muestreo aleatorio de la clase minoritaria
          - 'undersample': sub-muestreo aleatorio de la clase mayoritaria
          - 'smote': generación de muestras sintéticas (solo features numéricas)
        """
        if method is None or method == 'none':
            return X, y

        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy()

        counts = y.value_counts()
        if len(counts) <= 1:
            return X, y

        if method == 'oversample':
            max_count = counts.max()
            resampled_idx = []
            for cls, n in counts.items():
                cls_idx = y[y == cls].index.to_numpy()
                if n < max_count:
                    extra = np.random.choice(cls_idx, size=(max_count - n), replace=True)
                    idx = np.concatenate([cls_idx, extra])
                else:
                    idx = cls_idx
                resampled_idx.append(idx)
            idx = np.concatenate(resampled_idx)
            np.random.shuffle(idx)
            idx = pd.Index(idx)
            return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)

        if method == 'undersample':
            min_count = counts.min()
            resampled_idx = []
            for cls in counts.index:
                cls_idx = y[y == cls].index.to_numpy()
                idx = np.random.choice(cls_idx, size=min_count, replace=False)
                resampled_idx.append(idx)
            idx = np.concatenate(resampled_idx)
            np.random.shuffle(idx)
            idx = pd.Index(idx)
            return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)

        if method == 'smote':
            try:
                from sklearn.neighbors import NearestNeighbors
            except ImportError:
                print("SMOTE requiere scikit-learn; no está disponible")
                return X, y

            X_num = X.select_dtypes(include=[np.number]).copy()
            if X_num.shape[1] == 0:
                # FIX: advertencia explícita en lugar de fallo silencioso
                print("ADVERTENCIA: SMOTE requiere columnas numéricas. Se devuelven datos originales.")
                return X, y

            # Imputar NaN con la mediana antes de aplicar NearestNeighbors
            if X_num.isna().any().any():
                X_num = X_num.fillna(X_num.median())

            max_count = counts.max()
            resampled_X = [X_num]
            resampled_y = [y]

            for cls, n in counts.items():
                if n == max_count:
                    continue
                cls_idx = y[y == cls].index
                X_cls = X_num.loc[cls_idx].to_numpy()
                if len(X_cls) < 2:
                    continue
                neigh = NearestNeighbors(n_neighbors=min(5, len(X_cls)), metric='euclidean')
                neigh.fit(X_cls)
                n_samples = max_count - n
                synthetic = []
                for _ in range(n_samples):
                    idx = np.random.randint(0, len(X_cls))
                    nn = neigh.kneighbors([X_cls[idx]], return_distance=False)[0]
                    nn = nn[nn != idx]
                    if len(nn) == 0:
                        neighbor = X_cls[idx]
                    else:
                        neighbor = X_cls[np.random.choice(nn)]
                    diff = neighbor - X_cls[idx]
                    gap = np.random.rand()
                    synthetic.append(X_cls[idx] + gap * diff)
                if synthetic:
                    synthetic = pd.DataFrame(synthetic, columns=X_num.columns)
                    resampled_X.append(synthetic)
                    resampled_y.append(pd.Series([cls] * len(synthetic)))

            X_bal = pd.concat(resampled_X, ignore_index=True)
            y_bal = pd.concat(resampled_y, ignore_index=True)
            return X_bal, y_bal

        print(f"Método de balanceo desconocido: {method}")
        return X, y

    def _entrenar_clasificador(self, modelo, nombre, scale=True, balance_method=None,
                                class_weight=None, random_state=42, **params):
        print(f"\n{nombre}")
        X_train, y_train = self.X_train.copy(), self.y_train.copy()
        if balance_method in ['oversample', 'undersample', 'smote']:
            X_train, y_train = self._balance_data(X_train, y_train, method=balance_method,
                                                   random_state=random_state)

        # FIX: usa inspect en lugar de __code__.co_varnames
        if class_weight is not None and _tiene_parametro(modelo, 'class_weight'):
            params['class_weight'] = class_weight
        if _tiene_parametro(modelo, 'random_state') and 'random_state' not in params:
            params['random_state'] = 42

        estimator = modelo(**params)
        pipeline = (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
                    if scale else make_pipeline(SimpleImputer(strategy='median'), estimator))
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(self.X_test)
        metricas = self._calcular_metricas_clasificacion(
            self.y_test, y_pred, list(np.unique(self.y_test)))
        for k, v in metricas.items():
            print(f"{k}:\n{v}")
        return pipeline, metricas

    def clasificacion_knn(self, n_neighbors=3, algorithm='auto', scale=True):
        return self._entrenar_clasificador(KNeighborsClassifier, "KNN", scale=scale,
                                           n_neighbors=n_neighbors, algorithm=algorithm)

    def clasificacion_decision_tree(self, min_samples_split=2, max_depth=None, scale=False):
        return self._entrenar_clasificador(DecisionTreeClassifier, "Decision Tree", scale=scale,
                                           min_samples_split=min_samples_split, max_depth=max_depth)

    def clasificacion_random_forest(self, n_estimators=100, min_samples_split=2, max_depth=None, scale=False):
        return self._entrenar_clasificador(RandomForestClassifier, "Random Forest", scale=scale,
                                           n_estimators=n_estimators, min_samples_split=min_samples_split,
                                           max_depth=max_depth)

    def clasificacion_xgboost(self, n_estimators=100, min_samples_split=2, max_depth=3, scale=False):
        return self._entrenar_clasificador(GradientBoostingClassifier, "XGBoost", scale=scale,
                                           n_estimators=n_estimators, min_samples_split=min_samples_split,
                                           max_depth=max_depth)

    def clasificacion_adaboost(self, n_estimators=50, estimator=None, scale=False):
        if estimator is None:
            estimator = DecisionTreeClassifier(max_depth=1)
        return self._entrenar_clasificador(AdaBoostClassifier, "AdaBoost", scale=scale,
                                           estimator=estimator, n_estimators=n_estimators)

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
        # FIX: usa inspect en lugar de __code__.co_varnames
        if _tiene_parametro(modelo, 'random_state') and 'random_state' not in params:
            params['random_state'] = 42
        estimator = modelo(**params)
        pipeline = (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
                    if scale else make_pipeline(SimpleImputer(strategy='median'), estimator))
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
        return self._entrenar_regresor(SVR, f"SVM Regresión (kernel={kernel})", scale=scale,
                                       kernel=kernel, C=C, epsilon=epsilon)

    def regresion_decision_tree(self, max_depth=3, scale=False):
        return self._entrenar_regresor(DecisionTreeRegressor, f"Decision Tree (max_depth={max_depth})",
                                       scale=scale, max_depth=max_depth)

    def regresion_random_forest(self, n_estimators=100, max_depth=None, scale=False):
        return self._entrenar_regresor(RandomForestRegressor, f"Random Forest (n={n_estimators})",
                                       scale=scale, n_estimators=n_estimators, max_depth=max_depth)

    def regresion_xgboost(self, n_estimators=100, max_depth=4, scale=False):
        return self._entrenar_regresor(GradientBoostingRegressor, f"XGBoost (n={n_estimators})",
                                       scale=scale, n_estimators=n_estimators, max_depth=max_depth)

    def benchmark_regresion(self):
        print("\n" + "="*70 + "\nBENCHMARK DE REGRESION\n" + "="*70)
        modelos = {
            'Lineal':        self.regresion_lineal,
            'Lasso':         self.regresion_lasso,
            'Ridge':         self.regresion_ridge,
            'SVM (RBF)':     lambda: self.regresion_svm(kernel='rbf'),
            'Decision Tree': self.regresion_decision_tree,
            'Random Forest': self.regresion_random_forest,
            'XGBoost':       self.regresion_xgboost,
        }
        resultados = []
        for nombre, func in modelos.items():
            _, errores = func()
            resultados.append({
                'Modelo': nombre,
                'RMSE': errores.loc[errores['Métrica'] == 'RMSE', 'Valor'].values[0],
                'MAE':  errores.loc[errores['Métrica'] == 'MAE',  'Valor'].values[0],
                'ER':   errores.loc[errores['Métrica'] == 'ER',   'Valor'].values[0],
            })
        df_res = pd.DataFrame(resultados).sort_values('RMSE')
        print(f"\nResultados:\n{df_res.to_string(index=False)}")
        return df_res

    def benchmark_clasificacion(self, cv_method='stratified', balance_method=None, n_folds=5):
        print("\n" + "="*70 + "\nBENCHMARK DE CLASIFICACION\n" + "="*70)
        modelos = {
            'KNN':          (KNeighborsClassifier, {'n_neighbors': 5}),
            'Decision Tree':(DecisionTreeClassifier,  {'random_state': 42}),
            'Random Forest':(RandomForestClassifier,  {'n_estimators': 100, 'random_state': 42}),
            'XGBoost':      (GradientBoostingClassifier, {'n_estimators': 100, 'random_state': 42}),
            'AdaBoost':     (AdaBoostClassifier,      {'n_estimators': 50,  'random_state': 42}),
        }
        resultados = []
        for nombre, (modelo, params) in modelos.items():
            df_metricas = self.validacion_cruzada_completa(
                modelo, n_folds=n_folds, cv_method=cv_method,
                balance_method=balance_method, **params
            )
            resultados.append({
                'Modelo':    nombre,
                'Accuracy':  df_metricas.loc[df_metricas['Métrica'] == 'accuracy',           'Test (promedio)'].values[0],
                'Precision': df_metricas.loc[df_metricas['Métrica'] == 'precision_weighted', 'Test (promedio)'].values[0],
                'Recall':    df_metricas.loc[df_metricas['Métrica'] == 'recall_weighted',    'Test (promedio)'].values[0],
                'F1':        df_metricas.loc[df_metricas['Métrica'] == 'f1_weighted',        'Test (promedio)'].values[0],
            })
        df_res = pd.DataFrame(resultados).sort_values('F1', ascending=False)
        print(f"\nResultados:\n{df_res.to_string(index=False)}")
        return df_res

    def benchmark_balanceo(self, modelo=None, n_folds=5, scoring='accuracy',
                           cv_method='stratified', scale=True, **params):
        """Benchmark de balanceo de clases usando distintos métodos.

        Compara el desempeño de un mismo modelo con:
        - Sin balanceo, Oversampling, Undersampling, SMOTE, class_weight='balanced'
        """
        if modelo is None:
            modelo = RandomForestClassifier

        metodos = ['none', 'oversample', 'undersample', 'smote', 'class_weight']
        resultados = []

        for metodo in metodos:
            print(f"\n{'='*70}\nBalanceo: {metodo}\n{'='*70}")

            if metodo == 'class_weight':
                if not _tiene_parametro(modelo, 'class_weight'):
                    print(f"  ADVERTENCIA: {modelo.__name__} no acepta class_weight. Omitido.")
                    resultados.append({'Balance': metodo, 'Promedio': float('nan'), 'Std': float('nan')})
                    continue
                res = self.validacion_cruzada(modelo, n_folds=n_folds, scale=scale,
                                              scoring=scoring, cv_method=cv_method,
                                              class_weight='balanced', **params)
            else:
                res = self.validacion_cruzada(modelo, n_folds=n_folds, scale=scale,
                                              scoring=scoring, cv_method=cv_method,
                                              balance_method=None if metodo == 'none' else metodo,
                                              **params)

            resultados.append({
                'Balance':  metodo,
                'Promedio': res['promedio'],
                'Std':      res['std'],
            })

        df_res = pd.DataFrame(resultados).sort_values('Promedio', ascending=False)
        print(f"\nBenchmark de balanceo:\n{df_res.to_string(index=False)}")
        return df_res

    def validacion_cruzada(self, modelo, n_folds=10, scale=True, scoring='accuracy',
                            cv_method='kfold', balance_method=None, balance_params=None,
                            class_weight=None, **params):
        """Validación cruzada con diferentes métodos de particionado.

        Parámetros
        ----------
        modelo        : clase del modelo (ej: DecisionTreeClassifier)
        n_folds       : número de folds (default=10)
        scale         : aplicar StandardScaler en pipeline (default=True)
        scoring       : métrica de evaluación (default='accuracy')
        cv_method     : 'kfold' | 'stratified' | 'timeseries'
        balance_method: 'oversample' | 'undersample' | 'smote' | None
        balance_params: dict con kwargs adicionales para _balance_data
        class_weight  : valor para el parámetro class_weight del modelo (ej: 'balanced')
        **params      : kwargs adicionales del modelo

        Retorna
        -------
        dict con resultados por fold y estadísticas globales
        """
        print(f"\n{'='*70}")
        print(f"VALIDACION CRUZADA - {modelo.__name__}")
        print(f"{'='*70}")
        print(f"Folds: {n_folds} | Métrica: {scoring} | CV method: {cv_method}")
        if balance_method:
            print(f"Balanceo: {balance_method}")

        self._asegurar_df_encoded()
        df = self.df_encoded
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        if _tiene_parametro(modelo, 'random_state') and 'random_state' not in params:
            params['random_state'] = 42
        if class_weight is not None and _tiene_parametro(modelo, 'class_weight'):
            params['class_weight'] = class_weight

        estimator = modelo(**params)
        pipeline = (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
                    if scale else make_pipeline(SimpleImputer(strategy='median'), estimator))

        if cv_method == 'stratified':
            min_class = int(y.value_counts().min())
            if min_class < 2:
                print(f"ADVERTENCIA: alguna clase tiene solo {min_class} muestra(s). "
                      f"Se cambia a KFold para evitar error en StratifiedKFold.")
                cv_method = 'kfold'
            elif n_folds > min_class:
                print(f"ADVERTENCIA: n_folds={n_folds} excede el mínimo de muestras por clase ({min_class}). "
                      f"Se ajusta a {min_class}.")
                n_folds = min_class

        if cv_method == 'kfold':
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif cv_method == 'stratified':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif cv_method == 'timeseries':
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            raise ValueError("cv_method debe ser 'kfold', 'stratified' o 'timeseries'.")

        scorer = get_scorer(scoring)
        resultados = []
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

            if balance_method:
                X_tr, y_tr = self._balance_data(
                    X_tr, y_tr, method=balance_method, **(balance_params or {}))

            pipeline.fit(X_tr, y_tr)
            score = scorer(pipeline, X_te, y_te)
            resultados.append(score)
            print(f"   Fold {i:2d}: {score:.4f}")

        resultados = np.array(resultados)
        print(f"\n{'='*70}")
        print(f"Promedio: {resultados.mean():.4f} | Std: {resultados.std():.4f} | "
              f"Min: {resultados.min():.4f} | Max: {resultados.max():.4f}")
        print(f"{'='*70}")

        return {
            'resultados': resultados,
            'promedio':   resultados.mean(),
            'std':        resultados.std(),
            'min':        resultados.min(),
            'max':        resultados.max(),
        }

    def validacion_cruzada_completa(self, modelo, n_folds=10, scale=True, cv_method='kfold',
                                     balance_method=None, balance_params=None,
                                     scoring=None, **params):
        """Validación cruzada con múltiples métricas (clasificación y regresión).

        Retorna
        -------
        DataFrame con columnas: Métrica, Test (promedio), Test (std)
        """
        print(f"\n{'='*70}")
        print(f"VALIDACION CRUZADA COMPLETA - {modelo.__name__}")
        print(f"{'='*70}")
        print(f"CV method: {cv_method}")
        if balance_method:
            print(f"Balanceo: {balance_method}")

        self._asegurar_df_encoded()
        df = self.df_encoded
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        if _tiene_parametro(modelo, 'random_state') and 'random_state' not in params:
            params['random_state'] = 42

        estimator = modelo(**params)
        pipeline = (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
                    if scale else make_pipeline(SimpleImputer(strategy='median'), estimator))

        if cv_method == 'stratified':
            min_class = int(y.value_counts().min())
            if min_class < 2:
                print(f"ADVERTENCIA: alguna clase tiene solo {min_class} muestra(s). "
                      f"Se cambia a KFold para evitar error en StratifiedKFold.")
                cv_method = 'kfold'
            elif n_folds > min_class:
                print(f"ADVERTENCIA: n_folds={n_folds} excede el mínimo de muestras por clase ({min_class}). "
                      f"Se ajusta a {min_class}.")
                n_folds = min_class

        if cv_method == 'kfold':
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif cv_method == 'stratified':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif cv_method == 'timeseries':
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            raise ValueError("cv_method debe ser 'kfold', 'stratified' o 'timeseries'.")

        es_clasificacion = y.dtype == 'object' or len(np.unique(y)) < 20

        if es_clasificacion:
            metric_funcs = {
                'accuracy':           accuracy_score,
                'precision_weighted': lambda yt, yp: precision_score(yt, yp, average='weighted', zero_division=0),
                'recall_weighted':    lambda yt, yp: recall_score(yt, yp, average='weighted', zero_division=0),
                'f1_weighted':        lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0),
            }
        else:
            metric_funcs = {
                'mse': lambda yt, yp: mean_squared_error(yt, yp),
                'mae': lambda yt, yp: mean_absolute_error(yt, yp),
                'r2':  lambda yt, yp: r2_score(yt, yp),
            }

        scores = {m: [] for m in metric_funcs}
        for train_idx, test_idx in cv.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

            if balance_method:
                X_tr, y_tr = self._balance_data(
                    X_tr, y_tr, method=balance_method, **(balance_params or {}))

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_te)

            for name, func in metric_funcs.items():
                scores[name].append(func(y_te, y_pred))

        df_resultados = pd.DataFrame([{
            'Métrica':        name,
            'Test (promedio)': np.mean(vals),
            'Test (std)':      np.std(vals),
        } for name, vals in scores.items()])

        print(f"\nResultados:\n{df_resultados.to_string(index=False)}")
        print(f"{'='*70}")
        return df_resultados

    def optimizar_con_ga(self, tipo='clasificacion', modelo='random_forest',
                         pop_size=8, generations=8):
        if not GENETIC_AVAILABLE:
            print("ADVERTENCIA: sklearn-genetic-opt no instalado\nInstala: pip install sklearn-genetic-opt")
            return None, None
        print("\n" + "="*70 + "\nOPTIMIZACION CON ALGORITMOS GENETICOS\n" + "="*70)

        if modelo == 'random_forest':
            estimator = (RandomForestClassifier(random_state=42, n_jobs=-1) if tipo == 'clasificacion'
                         else RandomForestRegressor(random_state=42, n_jobs=-1))
            param_grid = {
                'n_estimators':     Integer(50, 200),
                'max_depth':        Integer(3, 20),
                'min_samples_split':Integer(2, 10),
                'min_samples_leaf': Integer(1, 5),
            }
        else:
            estimator = (GradientBoostingClassifier(random_state=42) if tipo == 'clasificacion'
                         else GradientBoostingRegressor(random_state=42))
            param_grid = {
                'n_estimators':     Integer(50, 200),
                'learning_rate':    Continuous(0.01, 0.3),
                'max_depth':        Integer(3, 10),
                'min_samples_split':Integer(2, 10),
            }

        ga_search = GASearchCV(
            estimator=estimator, cv=3,
            scoring='accuracy' if tipo == 'clasificacion' else 'neg_mean_squared_error',
            population_size=pop_size, generations=generations,
            n_jobs=-1, verbose=False, param_grid=param_grid)

        print(f"Ejecutando GA para {modelo} ({tipo})...")
        ga_search.fit(self.X_train, self.y_train)
        print(f"\nOptimización completada!\n   Mejor score (CV): {ga_search.best_score_:.4f}\n   Mejores parámetros:")
        for param, valor in ga_search.best_params_.items():
            print(f"      - {param}: {valor}")

        y_pred = ga_search.best_estimator_.predict(self.X_test)
        if tipo == 'clasificacion':
            print(f"   Accuracy en Test: {accuracy_score(self.y_test, y_pred):.4f}")
        else:
            print(f"\n   Errores en Test:\n{self._calcular_errores_regresion(self.y_test, y_pred).to_string(index=False)}")
        return ga_search.best_estimator_, ga_search


    def arima_model(self, order=(1, 1, 1)):
        if not STATSMODELS_AVAILABLE:
            print("ADVERTENCIA: statsmodels no disponible\nInstala: pip install statsmodels")
            return None
        print(f"\nModelo ARIMA{order}")
        print("Para análisis completo usa: SeriesTiempo(ts).arima(...)")
        return None

    def sarima_model(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        if not STATSMODELS_AVAILABLE:
            print("ADVERTENCIA: statsmodels no disponible\nInstala: pip install statsmodels")
            return None
        print(f"\nModelo SARIMA{order}x{seasonal_order}")
        print("Para análisis completo usa: SeriesTiempo(ts)")
        return None

    def prophet_model(self, seasonality_mode='additive', changepoint_prior_scale=0.05):
        print("\nModelo Prophet — requiere: pip install prophet")
        return None

    def exponential_smoothing(self, seasonal='add', seasonal_periods=12):
        if not STATSMODELS_AVAILABLE:
            print("ADVERTENCIA: statsmodels no disponible\nInstala: pip install statsmodels")
            return None
        print("\nSuavizado Exponencial (Holt-Winters)")
        print("Usa SeriesTiempo(ts=tu_serie).holt_winters(...)")
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
        self.df_scaled = pd.DataFrame(StandardScaler().fit_transform(self.df),
                                      columns=self.df.columns, index=self.df.index)
        print(f"Datos escalados: {self.df_scaled.shape}")
        return self

    def pca(self, n_componentes=2, plot=True, show=True):
        if PCA_Prince is None:
            print("ADVERTENCIA: Librería 'prince' no disponible, usando sklearn PCA")
            return self.pca_sklearn(n_componentes, plot, show=show)
        print(f"\nPCA con {n_componentes} componentes")
        datos = self.df_scaled if self.df_scaled is not None else self.df
        modelo = PCA_Prince(n_components=n_componentes).fit(datos)
        coordenadas = modelo.row_coordinates(datos)
        var_explicada = modelo.percentage_of_variance_
        print(f"\nVarianza explicada por componente:")
        for i, var in enumerate(var_explicada):
            print(f"   PC{i+1}: {var:.2f}%")
        print(f"   Total: {sum(var_explicada):.2f}%")
        fig = None
        if plot and n_componentes >= 2:
            fig = self._plot_pca(coordenadas, var_explicada,
                                 modelo.column_correlations, show=show)
        return modelo, coordenadas, var_explicada, fig

    def pca_sklearn(self, n_componentes=2, plot=True, scale=True, show=True):
        print(f"\nPCA (sklearn) con {n_componentes} componentes")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            pipeline = make_pipeline(StandardScaler(), PCA(n_components=n_componentes))
            componentes = pipeline.fit_transform(datos)
            pca = pipeline.named_steps['pca']
        else:
            pca = PCA(n_components=n_componentes)
            componentes = pca.fit_transform(datos)
            pipeline = pca

        var_explicada = pca.explained_variance_ratio_ * 100
        print(f"\nVarianza explicada: {[f'{v:.2f}%' for v in var_explicada]}, Total: {sum(var_explicada):.2f}%")

        fig = None
        if plot and n_componentes >= 2:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6)
            ax.set_xlabel(f'PC1 ({var_explicada[0]:.2f}%)')
            ax.set_ylabel(f'PC2 ({var_explicada[1]:.2f}%)')
            ax.set_title('PCA - Plano Principal')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if show:
                plt.show()
        return pipeline, componentes, var_explicada, fig

    def _plot_pca(self, coordenadas, var_explicada, correlaciones, show=True):
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
        if show:
            plt.show()
        return fig

    def _ejecutar_clustering(self, modelo, nombre, n_clusters, plot, scale=True, show=True, **kwargs):
        print(f"\n{nombre} con {n_clusters} clusters")
        datos = self.df_scaled if self.df_scaled is not None else self.df

        if scale and self.df_scaled is None:
            pipeline = make_pipeline(StandardScaler(), modelo)
            clusters = pipeline.fit_predict(datos)
            centros = pipeline.named_steps[list(pipeline.named_steps.keys())[1]].cluster_centers_
            datos_escalados = pipeline.named_steps['standardscaler'].transform(datos)
        else:
            pipeline = modelo
            clusters = modelo.fit_predict(datos)
            centros = modelo.cluster_centers_
            datos_escalados = datos

        print(f"\nDistribución:")
        for i in range(n_clusters):
            count = np.sum(clusters == i)
            print(f"   Cluster {i}: {count} ({count/len(clusters)*100:.1f}%)")

        fig = None
        if plot:
            fig = self._plot_clusters(datos_escalados, clusters, centros, nombre, show=show)
        return pipeline, clusters, centros, fig

    def kmeans(self, n_clusters=3, max_iter=500, n_init=150, plot=True, scale=True, show=True):
        return self._ejecutar_clustering(
            KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=42),
            "K-Means", n_clusters, plot, scale=scale, show=show)

    def kmedoids(self, n_clusters=3, max_iter=500, plot=True, scale=True, show=True):
        return self._ejecutar_clustering(
            KMedoids(n_clusters=n_clusters, max_iter=max_iter, metric='cityblock', random_state=42),
            "K-Medoids", n_clusters, plot, scale=scale, show=show)

    def _plot_clusters(self, datos, clusters, centros, titulo, show=True):
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(datos)
        centros_pca = pca.transform(centros)
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        colores = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
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
        if show:
            plt.show()
        return fig

    def hac(self, n_clusters=3, metodo='ward', plot=True, scale=True, show=True):
        print(f"\nClustering Jerárquico ({metodo}) con {n_clusters} clusters")
        datos = self.df_scaled if self.df_scaled is not None else self.df

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

        fig = None
        if plot:
            fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
            dendrogram(Z, labels=datos.index.tolist(), ax=ax)
            ax.set_title(f'Dendrograma - Método {metodo.capitalize()}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            if show:
                plt.show()
        return Z, clusters, fig

    def tsne(self, n_componentes=2, perplexity=30, plot=True, scale=True, show=True):
        if TSNE is None:
            print("ADVERTENCIA: t-SNE no disponible")
            return None, None, None
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

        fig = None
        if plot and n_componentes >= 2:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6)
            ax.set_xlabel('Componente t-SNE 1')
            ax.set_ylabel('Componente t-SNE 2')
            ax.set_title('t-SNE - Reducción Dimensional')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if show:
                plt.show()
        print("t-SNE completado")
        return pipeline, componentes, fig

    def umap_reduction(self, n_componentes=2, n_neighbors=15, plot=True, scale=True, show=True):
        if um is None:
            print("ADVERTENCIA: UMAP no disponible. Instala: pip install umap-learn")
            return None, None, None
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

        fig = None
        if plot and n_componentes >= 2:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6)
            ax.set_xlabel('Componente UMAP 1')
            ax.set_ylabel('Componente UMAP 2')
            ax.set_title('UMAP - Reducción Dimensional')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if show:
                plt.show()
        print("UMAP completado")
        return pipeline, componentes, fig

    @staticmethod
    def bar_plot(centros, labels, scale=False, figsize=(15, 8), show=True):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        centros_plot = np.copy(centros)
        if scale:
            for col in range(centros_plot.shape[1]):
                max_val = np.max(np.abs(centros_plot[:, col]))
                if max_val > 0:
                    centros_plot[:, col] = centros_plot[:, col] / max_val
        n_clusters = centros_plot.shape[0]
        x = np.arange(len(labels))
        width = 0.8 / n_clusters
        for i in range(n_clusters):
            ax.bar(x + width * i - (width * (n_clusters - 1) / 2),
                   centros_plot[i], width, label=f'Cluster {i}', alpha=0.8)
        ax.set_xlabel('Variables')
        ax.set_ylabel('Valor')
        ax.set_title('Comparación de Centroides por Cluster')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    @staticmethod
    def radar_plot(centros, labels, show=True):
        centros_norm = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if max(n) != min(n) else (n/n * 50)
                                 for n in centros.T])
        angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))] + [0]
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150, subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angulos[:-1], labels)
        plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=8)
        plt.ylim(0, 100)
        colores = ['blue', 'red', 'green', 'orange', 'purple']
        for i in range(centros_norm.shape[1]):
            valores = centros_norm[:, i].tolist() + [centros_norm[:, i].tolist()[0]]
            ax.plot(angulos, valores, linewidth=2, label=f'Cluster {i}', color=colores[i % len(colores)])
            ax.fill(angulos, valores, alpha=0.25, color=colores[i % len(colores)])
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Radar Plot - Comparación de Clusters', y=1.08)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

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
            if isinstance(ts.index, pd.DatetimeIndex) and ts.index.freqstr is not None:
                self.__ts = ts
            else:
                warnings.warn('ERROR: La serie debe tener un DatetimeIndex con frecuencia especificada.')
        else:
            warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')

    @property
    def coef(self):
        return self._coef


def _safe_freq(ts):
    """Retorna la frecuencia de la serie o 'D' como fallback seguro.
    
    FIX: evita AttributeError en forecast cuando la frecuencia no pudo inferirse.
    """
    freq = getattr(ts.index, 'freq', None)
    if freq is not None:
        return freq
    freqstr = getattr(ts.index, 'freqstr', None)
    if freqstr:
        return freqstr
    # Fallback: inferir de los datos
    inferred = pd.infer_freq(ts.index)
    return inferred if inferred else 'D'


# Modelos Básicos de Predicción
class meanfPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = [self.modelo.coef for _ in range(steps)]
        freq = _safe_freq(self.modelo.ts)
        fechas = pd.date_range(start=self.modelo.ts.index[-1], periods=steps + 1, freq=freq)
        return pd.Series(res, index=fechas[1:])


class naivePrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = [self.modelo.coef for _ in range(steps)]
        freq = _safe_freq(self.modelo.ts)
        fechas = pd.date_range(start=self.modelo.ts.index[-1], periods=steps + 1, freq=freq)
        return pd.Series(res, index=fechas[1:])


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
        freq = _safe_freq(self.modelo.ts)
        fechas = pd.date_range(start=self.modelo.ts.index[-1], periods=steps + 1, freq=freq)
        return pd.Series(res, index=fechas[1:])


class driftPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)

    def forecast(self, steps=1):
        res = [self.modelo.ts[-1] + self.modelo.coef * i for i in range(steps)]
        freq = _safe_freq(self.modelo.ts)
        fechas = pd.date_range(start=self.modelo.ts.index[-1], periods=steps + 1, freq=freq)
        return pd.Series(res, index=fechas[1:])


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
    def __init__(self, ts, test, trend=None, seasonal=None):
        super().__init__(ts)
        self.__test = test
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels no disponible. Instala: pip install statsmodels")
        if seasonal is not None:
            self.__modelo = ExponentialSmoothing(ts, trend=trend, seasonal=seasonal)
        else:
            self.__modelo = ExponentialSmoothing(ts, trend=trend)

    @property
    def test(self):
        return self.__test

    @test.setter
    def test(self, test):
        if isinstance(test, pd.core.series.Series):
            if test.index.freqstr is not None:
                self.__test = test
            else:
                warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
        else:
            warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')

    def fit(self, paso=0.1):
        if self.__modelo.trend is None and self.__modelo.seasonal is None:
            model_fit = self.__modelo.fit()
            alpha = getattr(model_fit.params, 'smoothing_level', None)
            beta  = getattr(model_fit.params, 'smoothing_trend', None)
            gamma = getattr(model_fit.params, 'smoothing_seasonal', None)
            return HW_Prediccion(model_fit, alpha, beta, gamma)

        error = float("inf")
        best_model = None
        best_params = {'alpha': None, 'beta': None, 'gamma': None}
        n = np.append(np.arange(0, 1, paso), 1)
        has_trend    = self.__modelo.trend    is not None
        has_seasonal = self.__modelo.seasonal is not None

        for alpha in n:
            for beta in (n if has_trend else [None]):
                for gamma in (n if has_seasonal else [None]):
                    fit_kwargs = {'smoothing_level': alpha}
                    if beta  is not None: fit_kwargs['smoothing_trend']    = beta
                    if gamma is not None: fit_kwargs['smoothing_seasonal'] = gamma
                    try:
                        model_fit = self.__modelo.fit(**fit_kwargs)
                        pred = np.array(model_fit.forecast(len(self.test)))
                        mse = np.mean((pred - self.test.values) ** 2)
                        if mse < error:
                            error = mse
                            best_model = model_fit
                            best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
                    except Exception:
                        continue

        if best_model is None:
            model_fit = self.__modelo.fit()
            alpha = getattr(model_fit.params, 'smoothing_level', None)
            beta  = getattr(model_fit.params, 'smoothing_trend', None)
            gamma = getattr(model_fit.params, 'smoothing_seasonal', None)
            return HW_Prediccion(model_fit, alpha, beta, gamma)

        return HW_Prediccion(best_model, best_params['alpha'], best_params['beta'], best_params['gamma'])


# LSTM para Series de Tiempo
class LSTM_TSPrediccion(Prediccion):
    def __init__(self, modelo):
        super().__init__(modelo)
        if not KERAS_AVAILABLE:
            raise ImportError("Keras/TensorFlow no disponible. Instala: pip install tensorflow")
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
            res.append(self.__scaler.inverse_transform(pred).tolist()[0][0])
            self.__X = np.append(self.__X, pred.tolist(), axis=0)

        freq = _safe_freq(self.modelo.ts)
        fechas = pd.date_range(start=self.modelo.ts.index[-1], periods=steps + 1, freq=freq)
        return pd.Series(res, index=fechas[1:])


class LSTM_TS(Modelo):
    def __init__(self, ts, p=1, lstm_units=50, dense_units=1, optimizer='rmsprop', loss='mse'):
        super().__init__(ts)
        if not KERAS_AVAILABLE:
            raise ImportError("Keras/TensorFlow no disponible. Instala: pip install tensorflow")
        try:
            from keras.models import Sequential
            from keras.layers import LSTM, Dense
        except ImportError:
            raise ImportError("Keras no disponible")
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
        self.__preds  = preds if isinstance(preds, list) else [preds]
        self.__real   = real
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
            warnings.warn('ERROR: preds debe ser una serie o lista de series.')

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
            warnings.warn('ERROR: Los nombres no calzan con la cantidad de métodos.')

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

    def plot_errores(self, show=True):
        fig = plt.figure(figsize=(8, 8))
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
        plt.yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"],
                   color="grey", size=10)
        plt.ylim(-10, 110)
        for i in df.index.values:
            p = df.loc[i].values.tolist() + df.loc[i].values.tolist()[:1]
            ax.plot(angles, p, linewidth=1, linestyle='solid', label=i)
            ax.fill(angles, p, alpha=0.1)
        plt.legend(loc='best')
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plotly_errores(self):
        if not PLOTLY_AVAILABLE:
            print("Plotly no disponible. Usando matplotlib...")
            return self.plot_errores()

        df = self.__escalar()
        etqs = df.columns.values.tolist() + df.columns.values.tolist()[:1]
        if len(df) == 1:
            df.loc[0] = 100

        fig = go.Figure()
        for i in df.index.values:
            p = df.loc[i].values.tolist() + df.loc[i].values.tolist()[:1]
            fig.add_trace(go.Scatterpolar(r=p, theta=etqs, fill='toself', name=i))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-10, 110])))
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

    def plot_periodograma(self, best=3, show=True):
        res = self.mejor_freq(best)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.freq, self.spec, color="darkgray")
        for i in range(best):
            ax.axvline(x=res[i], label=f"Mejor {i + 1}", ls='--', c=np.random.rand(3,))
        ax.set_xlabel('Frecuencia')
        ax.set_ylabel('Densidad Espectral')
        ax.set_title('Periodograma')
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plotly_periodograma(self, best=3):
        if not PLOTLY_AVAILABLE:
            print("Plotly no disponible. Usando matplotlib...")
            return self.plot_periodograma(best)

        res = self.mejor_freq(best)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.freq, y=self.spec, mode='lines+markers', line_color='darkgray'))
        for i in range(best):
            v = np.random.rand(3)
            color = f"rgb({v[0]}, {v[1]}, {v[2]})"
            fig.add_vline(x=res[i], line_width=2, line_dash="dash",
                          annotation_text=f"Mejor {i + 1}", line_color=color)
        fig.update_layout(title='Periodograma', xaxis_title='Frecuencia', yaxis_title='Densidad Espectral')
        return fig


# Clase Principal de Series de Tiempo
class SeriesTiempo:

    def __init__(self, ts=None, path=None, date_col='fecha', value_col=None, freq='D'):

        if ts is not None:
            if not isinstance(ts, pd.Series):
                raise ValueError("ts debe ser una pandas Series")
            if not isinstance(ts.index, pd.DatetimeIndex):
                raise ValueError(
                    "El índice de la serie debe ser DatetimeIndex. "
                    "Use pd.to_datetime() para convertir.")
            if ts.index.isna().any():
                raise ValueError(
                    "El índice contiene valores NaT. Verifique que las fechas sean válidas.")
            if not pd.api.types.is_numeric_dtype(ts):
                try:
                    ts = pd.to_numeric(ts, errors='coerce')
                    print("Los valores se convirtieron a numérico")
                except Exception:
                    raise ValueError("Los valores de la serie deben ser numéricos")
            if ts.isna().any():
                print(f"Advertencia: {ts.isna().sum()} valores no numéricos → NaN. Se rellenarán.")
                ts = ts.ffill().bfill()
            self.ts = ts

            # Inferir frecuencia si no está definida
            if self.ts.index.freq is None:
                try:
                    inferred_freq = pd.infer_freq(self.ts.index)
                    if inferred_freq:
                        self.ts.index.freq = inferred_freq
                    else:
                        try:
                            time_diffs = self.ts.index.to_series().diff().dropna()
                            if len(time_diffs) > 0:
                                most_common_diff = time_diffs.value_counts().index[0]
                                if   most_common_diff.days == 1:                                  self.ts.index.freq = 'D'
                                elif most_common_diff.days == 7:                                  self.ts.index.freq = 'W'
                                elif 28 <= most_common_diff.days <= 31:                           self.ts.index.freq = 'ME'
                        except Exception:
                            pass
                except Exception:
                    pass

            # Segunda pasada de NaN tras conversión (por si bfill no alcanzó)
            if self.ts.isna().any():
                print(f"Advertencia: {self.ts.isna().sum()} NaN tras conversión. Se rellenarán.")
                # FIX: mismo fix de ffill/bfill
                self.ts = self.ts.ffill().bfill()

        elif path is not None:
            df = pd.read_csv(path)
            if value_col is None:
                value_col = [c for c in df.columns if c != date_col][0]
            try:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
            except ValueError as e:
                raise ValueError(f"Error al convertir fechas en {path}: {e}")
            df = df.set_index(date_col)
            self.ts = df[value_col]
            self.ts.index.freq = freq
        else:
            raise ValueError("Debe proporcionar 'ts' o 'path'")

        print(f"Serie de tiempo cargada: {len(self.ts)} observaciones")
        freq_str = getattr(self.ts.index, 'freqstr', None)
        if freq_str:
            print(f"Frecuencia: {freq_str}")

    def info(self):
        print("\n" + "="*70)
        print("INFORMACION DE SERIE DE TIEMPO")
        print("="*70)
        print(f"Observaciones: {len(self.ts)}")
        freq_str = getattr(self.ts.index, 'freqstr', None)
        print(f"Frecuencia: {freq_str if freq_str else 'No especificada'}")
        print(f"Rango: {self.ts.index[0]} a {self.ts.index[-1]}")
        print(f"\nEstadísticas:\n{self.ts.describe()}")
        return self

    def plot(self, title='Serie de Tiempo', figsize=(12, 6), show=True):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        self.ts.plot(ax=ax, marker='o', markersize=3)
        ax.set_title(title)
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valor')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plotly_plot(self, title='Serie de Tiempo'):
        if not PLOTLY_AVAILABLE:
            print("Plotly no disponible. Usando matplotlib...")
            return self.plot(title)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.ts.index, y=self.ts.values,
                                 mode='lines+markers', name='Serie'))
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(title=title, xaxis_title='Fecha', yaxis_title='Valor')
        fig.show()
        return self

    def meanf(self):
        return meanf(self.ts).fit()

    def naive(self):
        return naive(self.ts).fit()

    def snaive(self, h=1):
        return snaive(self.ts).fit(h)

    def drift(self):
        return drift(self.ts).fit()

    def holt_winters(self, trend=None, seasonal=None, seasonal_periods=None):
        if not STATSMODELS_AVAILABLE:
            print("statsmodels no disponible. Instala: pip install statsmodels")
            return None
        modelo = ExponentialSmoothing(self.ts, trend=trend, seasonal=seasonal,
                                      seasonal_periods=seasonal_periods)
        modelo_fit = modelo.fit()
        print(f"Holt-Winters ajustado (trend={trend}, seasonal={seasonal})")
        return modelo_fit

    def holt_winters_calibrado(self, test, paso=0.1, trend=None, seasonal=None):
        modelo = HW_calibrado(self.ts, test, trend, seasonal)
        resultado = modelo.fit(paso)
        # FIX: alpha/beta/gamma pueden ser None → format seguro
        fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
        print(f"HW Calibrado - alpha: {fmt(resultado.alpha)}, "
              f"beta: {fmt(resultado.beta)}, gamma: {fmt(resultado.gamma)}")
        return resultado

    def lstm(self, p=1, lstm_units=50, dense_units=1, optimizer='rmsprop', loss='mse'):
        modelo = LSTM_TS(self.ts, p, lstm_units, dense_units, optimizer, loss)
        return modelo.fit()

    def periodograma(self, best=3, plot=True, show=True):
        periodo = Periodograma(self.ts)
        print("\nAnálisis de Periodicidad:")
        print(f"Mejores frecuencias: {periodo.mejor_freq(best)}")
        print(f"Mejores períodos:    {periodo.mejor_periodos(best)}")
        if plot:
            periodo.plot_periodograma(best, show=show)
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
        test  = self.ts[-n_test:]
        if hasattr(self.ts.index, 'freq') and self.ts.index.freq is not None:
            train.index.freq = self.ts.index.freq
            test.index.freq  = self.ts.index.freq
        print(f"Train: {len(train)} observaciones, Test: {len(test)} observaciones")
        return train, test

    def arima(self, order=(0, 1, 1), seasonal_order=(0, 0, 0, 0), trend='n'):
        if not STATSMODELS_AVAILABLE:
            print("statsmodels no disponible. Instala: pip install statsmodels")
            return None
        modelo = SARIMAX(self.ts, order=order, seasonal_order=seasonal_order,
                         trend=trend, enforce_stationarity=False, enforce_invertibility=False)
        resultado = modelo.fit(disp=False)
        print(f"ARIMA ajustado (order={order}, seasonal_order={seasonal_order})")
        return resultado

    def arima_calibrado(self, test, p_values=(0, 1, 2), d_values=(0, 1), q_values=(0, 1, 2),
                        seasonal_order=(0, 0, 0, 0)):
        if not STATSMODELS_AVAILABLE:
            print("statsmodels no disponible. Instala: pip install statsmodels")
            return None, None

        best_score  = float("inf")
        best_order  = None
        best_result = None
        n_steps = len(test)

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        modelo = SARIMAX(self.ts, order=(p, d, q),
                                         seasonal_order=seasonal_order,
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)
                        resultado = modelo.fit(disp=False, maxiter=100, method='lbfgs')
                        pred = np.array(resultado.forecast(steps=n_steps))
                        mse = np.mean((test.values - pred) ** 2)
                        if mse < best_score:
                            best_score  = mse
                            best_order  = (p, d, q)
                            best_result = resultado
                    except Exception:
                        continue

        if best_result is None:
            print("No se pudo ajustar ningún modelo ARIMA con los parámetros dados")
            return None, None

        print(f"ARIMA calibrado (order={best_order}) - MSE en test: {best_score:.4f}")
        return best_result, best_order

    # FIX: benchmark unificado — LSTM desactivado por defecto (requiere TensorFlow)
    def benchmark(self, test_size=0.2, incluir_lstm=False,
                  lstm_kwargs=None, hw_kwargs=None, hw_cal_kwargs=None,
                  arima_order=(1, 1, 1), arima_cal_params=None):
        """Ejecuta un benchmark entre los modelos de series de tiempo disponibles.

        Parámetros
        ----------
        test_size     : fracción del dataset para test (default=0.2)
        incluir_lstm  : incluir Red Neuronal LSTM (requiere TensorFlow, default=False)
        lstm_kwargs   : kwargs para SeriesTiempo.lstm()
        hw_kwargs     : kwargs para holt_winters()
        hw_cal_kwargs : kwargs para holt_winters_calibrado()
        arima_order   : orden (p,d,q) para ARIMA fijo (default=(1,1,1))
        arima_cal_params : dict con p_values/d_values/q_values para ARIMA calibrado

        Retorna
        -------
        DataFrame con métricas MSE, RMSE, RE, CORR por modelo
        """
        return self.benchmark_personalizado(
            test_size=test_size,
            incluir_hw=True, incluir_hw_cal=True,
            incluir_arima=True, incluir_arima_cal=True,
            incluir_lstm=incluir_lstm,
            lstm_kwargs=lstm_kwargs, hw_kwargs=hw_kwargs,
            hw_cal_kwargs=hw_cal_kwargs,
            arima_order=arima_order, arima_cal_params=arima_cal_params,
        )

    def benchmark_personalizado(self, test_size=0.2,
                                 incluir_hw=True, incluir_hw_cal=True,
                                 incluir_arima=True, incluir_arima_cal=True,
                                 incluir_lstm=False,
                                 lstm_kwargs=None, hw_kwargs=None, hw_cal_kwargs=None,
                                 arima_order=(1, 1, 1), arima_cal_params=None):
        """Benchmark personalizado con selección de modelos a incluir."""
        train, test = self.train_test_split(test_size)
        nombres      = []
        predicciones = []

        def _agregar(nombre, pred):
            """Valida y agrega predicción a la lista."""
            if pred is not None and len(pred) == len(test) and not pd.isna(pred).any():
                predicciones.append(pd.Series(pred.values, index=test.index))
                nombres.append(nombre)
            else:
                n = len(pred) if pred is not None else 'None'
                print(f"Advertencia {nombre}: predicción inválida (len={n})")

        # Holt-Winters
        if incluir_hw:
            try:
                hw_kwargs = hw_kwargs or {}
                m = SeriesTiempo(ts=train).holt_winters(**hw_kwargs)
                if m is not None:
                    _agregar("Holt-Winters", m.forecast(len(test)))
            except Exception as e:
                print(f"Advertencia Holt-Winters: {e}")

        # Holt-Winters calibrado
        if incluir_hw_cal:
            try:
                hw_cal_kwargs = hw_cal_kwargs or {}
                m = SeriesTiempo(ts=train).holt_winters_calibrado(test, **hw_cal_kwargs)
                if m is not None:
                    _agregar("Holt-Winters Calibrado", m.forecast(len(test)))
            except Exception as e:
                print(f"Advertencia Holt-Winters calibrado: {e}")

        # ARIMA
        if incluir_arima:
            try:
                m = SeriesTiempo(ts=train).arima(order=arima_order)
                if m is not None:
                    _agregar("ARIMA", m.forecast(steps=len(test)))
            except Exception as e:
                print(f"Advertencia ARIMA: {e}")

        # ARIMA calibrado
        if incluir_arima_cal:
            try:
                arima_cal_params = arima_cal_params or {'p_values': (0, 1), 'd_values': (0, 1), 'q_values': (0, 1)}
                m, _ = SeriesTiempo(ts=train).arima_calibrado(test, **arima_cal_params)
                if m is not None:
                    _agregar("ARIMA Calibrado", m.forecast(steps=len(test)))
            except Exception as e:
                print(f"Advertencia ARIMA calibrado: {e}")

        # LSTM (opcional)
        if incluir_lstm:
            try:
                lstm_kwargs = lstm_kwargs or {'lstm_units': 20, 'dense_units': 1}
                m = SeriesTiempo(ts=train).lstm(**lstm_kwargs)
                if m is not None:
                    _agregar("Red Neuronal", m.forecast(len(test)))
            except Exception as e:
                print(f"Advertencia LSTM: {e}")

        if not predicciones:
            print("No se pudieron generar predicciones para el benchmark.")
            return None

        errores = SeriesTiempo.calcular_errores(predicciones, test, nombres=nombres)
        df_resultados = errores.df_errores().reset_index().rename(columns={'index': 'Modelo'})
        return df_resultados


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
            print("ERROR: beautifulsoup4 no instalado. Instala: pip install beautifulsoup4")
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
            print("No se encontraron tablas válidas")
            return None
        tabla   = tablas[indice_tabla]
        headers = [th.get_text(strip=True) for th in tabla.find('thead').find_all('th')] if tabla.find('thead') else []
        tbody   = tabla.find('tbody') or tabla
        rows    = [[td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
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
        enlaces = [link['href'] for link in soup.find_all('a', href=True)
                   if filtro is None or filtro in link['href']]
        print(f"Extraídos {len(enlaces)} enlaces")
        return enlaces

    def scrape_imagenes(self, url, atributo_src='src'):
        html = self.obtener_html(url)
        if not html:
            return None
        soup = self.parsear_html(html)
        if not soup:
            return None
        imagenes = [img.get(atributo_src) for img in soup.find_all('img') if img.get(atributo_src)]
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
            'titulo':      soup.find('title').get_text(strip=True) if soup.find('title') else None,
            'descripcion': soup.find('meta', attrs={'name': 'description'}).get('content') if soup.find('meta', attrs={'name': 'description'}) else None,
            'keywords':    soup.find('meta', attrs={'name': 'keywords'}).get('content') if soup.find('meta', attrs={'name': 'keywords'}) else None,
            'autor':       soup.find('meta', attrs={'name': 'author'}).get('content') if soup.find('meta', attrs={'name': 'author'}) else None,
        }
        print("Metadata extraída:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")
        return metadata

    def descargar_archivo(self, url, nombre_archivo=None, directorio='descargas'):
        try:
            import requests, os
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