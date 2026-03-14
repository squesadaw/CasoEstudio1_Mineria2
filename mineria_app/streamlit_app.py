"""
Aplicación de Minería de Datos Avanzada
Autores: Jorge Chacón, Stacy Quesada
"""

# FIX: Forzar importación y reload del módulo para evitar caché de Streamlit
import matplotlib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import parse as dateutil_parse
from paquete_mineria2 import EDA, Supervisado, SeriesTiempo
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import traceback
import importlib
import sys
import paquete_mineria2
importlib.reload(paquete_mineria2)

matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Data Mining Lab",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14;
    color: #e8eaf0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1f2433;
}

/* ── hero header ── */
.hero {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1f2433;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.6rem, 3vw, 2.4rem);
    font-weight: 700;
    color: #e8eaf0;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero .sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 0.4rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.accent { color: #4ade80; }

/* ── section titles ── */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4ade80;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1f2433;
}

/* ── metric cards ── */
.metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    background: #13161e;
    border: 1px solid #1f2433;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    flex: 1;
    min-width: 140px;
}
.metric-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #4ade80;
}
.metric-card .value.neutral { color: #60a5fa; }
.metric-card .value.warn { color: #f59e0b; }

/* ── info badge ── */
.badge {
    display: inline-block;
    background: #1a2235;
    border: 1px solid #2d3a55;
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #60a5fa;
    margin: 0.2rem;
}

/* ── table styling ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1f2433;
    border-radius: 8px;
    overflow: hidden;
}

/* ── buttons ── */
.stButton > button {
    background: #4ade80 !important;
    color: #0d0f14 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── selectbox / slider ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stRadio"] label {
    font-size: 0.78rem !important;
    color: #9ca3af !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* ── sidebar nav ── */
.nav-item {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #6b7280;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1f2433;
    letter-spacing: 0.05em;
}
.nav-active { color: #4ade80 !important; }

/* ── expander ── */
[data-testid="stExpander"] {
    border: 1px solid #1f2433 !important;
    border-radius: 8px !important;
    background: #13161e !important;
}

/* ── spinner ── */
[data-testid="stSpinner"] { color: #4ade80 !important; }

/* ── success / warning / error ── */
.stSuccess { background: #0d2818 !important; border-color: #4ade80 !important; }
.stWarning { background: #1f1500 !important; }

/* ── divider ── */
hr { border-color: #1f2433 !important; }

/* ── plot background ── */
.stPlotlyChart { border: 1px solid #1f2433; border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#13161e",
    plot_bgcolor="#13161e",
    font=dict(family="DM Sans", color="#e8eaf0"),
    colorway=["#4ade80", "#60a5fa", "#f59e0b",
              "#f87171", "#c084fc", "#34d399"],
)
PLOTLY_THEME = dict(template="plotly_dark", **PLOTLY_LAYOUT)

# Ruta a la carpeta de datasets predefinidos (relativa al script)
DATA_PATH = "/Users/stacyquesada/Documents/ULEAD/Mineria de Datos Avanzada/CasoEstudio1_Mineria2/data"

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════


def parse_dates_column(col):
    ser = pd.to_datetime(col, errors='coerce', utc=True)
    if ser.notna().all():
        return ser
    parsed = []
    for x in col:
        if pd.isna(x):
            parsed.append(pd.NaT)
            continue
        try:
            dt = dateutil_parse(str(x), dayfirst=True)
            parsed.append(pd.Timestamp(dt).tz_localize('UTC'))
        except Exception:
            parsed.append(pd.NaT)
    return pd.Series(parsed, index=col.index)


def plotly_fig(**updates):
    """Parámetros para update_layout() — incluye paper_bgcolor, font, etc."""
    cfg = dict(**PLOTLY_THEME)
    cfg.update(updates)
    return cfg


def plotly_px(**updates):
    """Parámetros compatibles con px.*() — solo template es válido."""
    return {"template": "plotly_dark", **updates}


def metric_html(label, value, style=""):
    return f"""
    <div class='metric-card'>
        <div class='label'>{label}</div>
        <div class='value {style}'>{value}</div>
    </div>"""


def section(title):
    st.markdown(
        f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def compute_auc_multiclass(modelo_pipeline, X_test, y_test):
    """Calcula AUC (macro-average) para binario o multiclase."""
    classes = sorted(y_test.unique())
    n_classes = len(classes)
    try:
        if hasattr(modelo_pipeline, "predict_proba"):
            y_prob = modelo_pipeline.predict_proba(X_test)
        else:
            return None, None, None

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=classes[1])
            roc_auc = auc(fpr, tpr)
            return {"macro": roc_auc}, {"macro": (fpr, tpr)}, classes
        else:
            y_bin = label_binarize(y_test, classes=classes)
            fpr_dict, tpr_dict, auc_dict = {}, {}, {}
            for i, cls in enumerate(classes):
                fpr_dict[cls], tpr_dict[cls], _ = roc_curve(
                    y_bin[:, i], y_prob[:, i])
                auc_dict[cls] = auc(fpr_dict[cls], tpr_dict[cls])
            # macro
            all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in classes]))
            mean_tpr = np.zeros_like(all_fpr)
            for c in classes:
                mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
            mean_tpr /= n_classes
            auc_dict["macro"] = auc(all_fpr, mean_tpr)
            fpr_dict["macro"], tpr_dict["macro"] = all_fpr, mean_tpr
            return auc_dict, {k: (fpr_dict[k], tpr_dict[k]) for k in auc_dict}, classes
    except Exception:
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 1rem;'>
        <div style='font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;color:#e8eaf0;'>
            ⬡ Data Mining Lab
        </div>
        <div style='font-size:0.7rem;color:#4b5563;margin-top:0.3rem;letter-spacing:0.1em;text-transform:uppercase;'>
            Minería Avanzada
        </div>
    </div>
    <hr style='margin:0 0 1rem;'>
    """, unsafe_allow_html=True)

    pagina = st.radio(
        "Navegación",
        ["📂  Carga de datos", "🔍  EDA", "🤖  Clasificación", "📈  Series de Tiempo"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.68rem;color:#4b5563;font-family:Space Mono,monospace;line-height:1.6;'>
        AUTORES<br>
        <span style='color:#9ca3af;'>Jorge Chacón</span><br>
        <span style='color:#9ca3af;'>Stacy Quesada</span><br><br>
        ULEAD · Minería de Datos<br>Avanzada · 2026
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if "df" in st.session_state:
        df_info = st.session_state["df"]
        st.markdown(f"""
        <div style='font-size:0.72rem;color:#6b7280;font-family:Space Mono,monospace;'>
            DATASET ACTIVO<br>
            <span style='color:#4ade80;'>{df_info.shape[0]:,}</span> filas ×
            <span style='color:#60a5fa;'>{df_info.shape[1]}</span> cols
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

if pagina == "📂  Carga de datos":

    st.markdown("""
    <div class='hero'>
        <h1>⬡ Data Mining <span class='accent'>Lab</span></h1>
        <div class='sub'>Minería de Datos Avanzada</div>
    </div>
    """, unsafe_allow_html=True)

    section("Fuente de datos")

    tab1, tab2 = st.tabs(["⬆  Subir CSV", "📦  Datasets de muestra"])

    with tab1:
        archivo = st.file_uploader("Suba un archivo CSV", type=["csv"])
        col_sep, col_dec = st.columns(2)
        with col_sep:
            sep = st.selectbox("Separador", [",", ";", "\\t", "|"], index=0)
        with col_dec:
            dec = st.selectbox("Decimal", [".", ","], index=0)

        if archivo is not None:
            try:
                sep_real = "\t" if sep == "\\t" else sep
                try:
                    df = pd.read_csv(archivo, sep=sep_real, decimal=dec,
                                     engine='python', encoding='utf-8')
                except UnicodeDecodeError:
                    archivo.seek(0)
                    df = pd.read_csv(archivo, sep=sep_real, decimal=dec,
                                     engine='python', encoding='latin1')
                # df = normalizar_tipos_dataframe(df)  # FIX: función removida
                st.session_state["df"] = df
                st.session_state.pop("modelo_sup", None)
                st.session_state.pop("train", None)
                st.session_state.pop("test", None)
                st.success(
                    f"✓ Dataset cargado — {df.shape[0]:,} filas × {df.shape[1]} columnas")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

    with tab2:
        if not os.path.exists(DATA_PATH):
            st.warning(f"No se encontró la carpeta `data/` en `{DATA_PATH}`. "
                       "Créela junto a `app.py` y coloque sus archivos CSV dentro.")
        else:
            csv_files = sorted([f for f in os.listdir(
                DATA_PATH) if f.lower().endswith(".csv")])
            if not csv_files:
                st.warning(
                    "La carpeta `data/` existe pero no contiene archivos CSV.")
            else:
                col_ds, col_ds_sep, col_ds_dec = st.columns(3)
                with col_ds:
                    ds_name = st.selectbox(
                        "Seleccione un dataset", csv_files, key="ds_predefined")
                with col_ds_sep:
                    ds_sep = st.selectbox(
                        "Separador", [",", ";", "\\t", "|"], key="ds_sep")
                with col_ds_dec:
                    ds_dec = st.selectbox("Decimal", [".", ","], key="ds_dec")

                if st.button("Cargar dataset", key="btn_load_predefined"):
                    sep_real = "\t" if ds_sep == "\\t" else ds_sep
                    fpath = os.path.join(DATA_PATH, ds_name)
                    try:
                        try:
                            df = pd.read_csv(fpath, sep=sep_real, decimal=ds_dec,
                                             engine='python', encoding='utf-8')
                        except UnicodeDecodeError:
                            df = pd.read_csv(fpath, sep=sep_real, decimal=ds_dec,
                                             engine='python', encoding='latin1')
                        # df = normalizar_tipos_dataframe(df)  # FIX: función removida
                        st.session_state["df"] = df
                        st.session_state.pop("modelo_sup", None)
                        st.session_state.pop("train", None)
                        st.session_state.pop("test", None)
                        st.success(
                            f"✓ {ds_name} cargado — {df.shape[0]:,} filas × {df.shape[1]} columnas")
                    except Exception as e:
                        st.error(f"Error al leer {ds_name}: {e}")

    # Vista previa
    if "df" in st.session_state:
        df = st.session_state["df"]
        section("Vista previa")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_html(
                "Filas", f"{df.shape[0]:,}"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_html(
                "Columnas", df.shape[1], "neutral"), unsafe_allow_html=True)
        with m3:
            n_num = df.select_dtypes(include="number").shape[1]
            st.markdown(metric_html("Numéricas", n_num, "neutral"),
                        unsafe_allow_html=True)
        with m4:
            n_null = int(df.isnull().sum().sum())
            st.markdown(metric_html("Nulos", n_null,
                        "warn" if n_null > 0 else ""), unsafe_allow_html=True)

        st.dataframe(df.head(10), use_container_width=True)

        # Tipos de columnas
        section("Tipos de columnas")
        dtypes_df = pd.DataFrame({
            "Columna": df.dtypes.index,
            "Tipo": df.dtypes.values.astype(str),
            "Nulos": df.isnull().sum().values,
            "Únicos": df.nunique().values,
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: EDA
# ══════════════════════════════════════════════════════════════════════════════

elif pagina == "🔍  EDA":

    st.markdown("<div class='hero'><h1>Análisis <span class='accent'>Exploratorio</span></h1><div class='sub'>Distribuciones · correlaciones · outliers</div></div>", unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.warning("Primero cargue un dataset en la sección **Carga de datos**.")
        st.stop()

    df = st.session_state["df"]
    df_num = df.select_dtypes(include="number")

    # ── Estadísticas ──────────────────────────────────────────────────────────
    section("Estadísticas descriptivas")
    st.dataframe(df.describe(include="all").T.round(4),
                 use_container_width=True)

    # ── Nulos ─────────────────────────────────────────────────────────────────
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        section("Valores nulos")
        fig_null = px.bar(
            x=nulls.index, y=nulls.values,
            labels={"x": "Columna", "y": "Nulos"},
            title="Cantidad de valores nulos por columna",
            **plotly_px(),
        )
        fig_null.update_traces(marker_color="#f87171")
        fig_null.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_null, use_container_width=True)

    # ── Correlación ───────────────────────────────────────────────────────────
    if df_num.shape[1] >= 2:
        section("Matriz de correlación")
        corr = df_num.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, "#1e3a5f"], [0.5, "#13161e"], [1, "#064e3b"]],
            zmin=-1, zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_corr.update_layout(
            title="Correlación entre variables numéricas", **plotly_fig(), height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Distribuciones ────────────────────────────────────────────────────────
    if not df_num.empty:
        section("Distribuciones")
        cols_dist = st.multiselect(
            "Seleccione columnas a visualizar",
            df_num.columns.tolist(),
            default=df_num.columns.tolist()[:min(4, len(df_num.columns))],
        )
        if cols_dist:
            ncols = 2
            nrows = -(-len(cols_dist) // ncols)
            fig_dist = make_subplots(rows=nrows, cols=ncols,
                                     subplot_titles=cols_dist)
            colors = ["#4ade80", "#60a5fa", "#f59e0b",
                      "#f87171", "#c084fc", "#34d399"]
            for i, col in enumerate(cols_dist):
                r, c = divmod(i, ncols)
                fig_dist.add_trace(
                    go.Histogram(x=df_num[col], name=col,
                                 marker_color=colors[i % len(colors)],
                                 opacity=0.8, showlegend=False),
                    row=r + 1, col=c + 1,
                )
            fig_dist.update_layout(height=300 * nrows, **plotly_fig(),
                                   title="Distribución de variables numéricas")
            st.plotly_chart(fig_dist, use_container_width=True)

    # ── Boxplots ──────────────────────────────────────────────────────────────
    if not df_num.empty:
        section("Boxplots — detección de outliers")
        cols_box = st.multiselect(
            "Seleccione columnas",
            df_num.columns.tolist(),
            default=df_num.columns.tolist()[:min(5, len(df_num.columns))],
            key="box_cols",
        )
        if cols_box:
            fig_box = go.Figure()
            for i, col in enumerate(cols_box):
                fig_box.add_trace(go.Box(
                    y=df_num[col], name=col,
                    marker_color=colors[i % len(colors)],
                    line_color=colors[i % len(colors)],
                ))
            fig_box.update_layout(title="Boxplots", **plotly_fig(), height=420)
            st.plotly_chart(fig_box, use_container_width=True)

    # ── Scatter matrix ────────────────────────────────────────────────────────
    if df_num.shape[1] >= 2:
        section("Scatter matrix (pares de variables)")
        cols_scatter = df_num.columns.tolist()[:min(5, len(df_num.columns))]
        color_col = None
        cat_cols = df.select_dtypes(
            include=["object", "category"]).columns.tolist()
        if cat_cols:
            color_col = st.selectbox("Color por variable categórica (opcional)",
                                     ["—"] + cat_cols)
            if color_col == "—":
                color_col = None
        fig_scatter = px.scatter_matrix(
            df[cols_scatter + ([color_col] if color_col else [])],
            dimensions=cols_scatter,
            color=color_col,
            title="Scatter Matrix",
            **plotly_px(),
        )
        fig_scatter.update_traces(
            diagonal_visible=False, marker_size=3, opacity=0.7)
        fig_scatter.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

elif pagina == "🤖  Clasificación":

    st.markdown("<div class='hero'><h1>Modelos <span class='accent'>Supervisados</span></h1><div class='sub'>Clasificación · validación cruzada · AUC · balanceo</div></div>", unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.warning("Primero cargue un dataset en la sección **Carga de datos**.")
        st.stop()

    df = st.session_state["df"]

    # ── Configuración ─────────────────────────────────────────────────────────
    section("Configuración del problema")

    col_tgt, col_tipo = st.columns(2)
    with col_tgt:
        target = st.selectbox(
            "Variable objetivo (target)", df.columns.tolist())
    with col_tipo:
        tipo = st.radio("Tipo de problema", [
                        "Clasificación", "Regresión"], horizontal=True)

    col_ts, col_rs = st.columns(2)
    with col_ts:
        test_size = st.slider(
            "Tamaño del conjunto de prueba (%)", 10, 40, 25) / 100
    with col_rs:
        random_state = st.slider("Semilla aleatoria", 0, 100, 42)

    # FIX: Validar tipo de problema vs tipo de variable
    n_unique = df[target].nunique()
    es_continuo = n_unique >= 20

    if tipo == "Clasificación" and es_continuo:
        st.warning(
            f"⚠️ **Advertencia**: '{target}' tiene {n_unique} valores únicos (parece **continua**).\n\n"
            f"Para **Clasificación** necesitás una variable con <20 clases distint as.\n\n"
            f"**Recomendación**: Cambiá a **Regresión** para esta variable."
        )
    elif tipo == "Regresión" and not es_continuo:
        st.info(
            f"ℹ️  '{target}' tiene solo {n_unique} valores únicos (parece **discreta**).\n\n"
            f"Es más apropiado para **Clasificación**, pero podés usar Regresión si lo preferís."
        )

    if st.button("⬡  Preparar datos"):
        # Validación estricta: no permitir Clasificación con variable continua
        if tipo == "Clasificación" and es_continuo:
            st.error(
                f"❌ No es posible usar Clasificación con '{target}' (variable continua).\n\n"
                f"Por favor, seleccioná **Regresión**."
            )
            st.stop()

        with st.spinner("Preparando datos…"):
            try:
                modelo_sup = Supervisado(df=df, target_col=target)
                modelo_sup.preparar_datos(
                    test_size=test_size, random_state=random_state)
                st.session_state["modelo_sup"] = modelo_sup
                st.session_state["target_col"] = target
                st.session_state["tipo_prob"] = tipo
                st.success(
                    f"✓ Datos preparados — Train: {modelo_sup.X_train.shape[0]:,} · Test: {modelo_sup.X_test.shape[0]:,}")
            except Exception as e:
                st.error(f"Error al preparar los datos: {e}")
                st.stop()

    if "modelo_sup" not in st.session_state:
        st.info("Configure y prepare los datos para continuar.")
        st.stop()

    modelo_sup = st.session_state["modelo_sup"]
    tipo = st.session_state.get("tipo_prob", tipo)

    # ══════════════════════════════════════════════════════════════════════════
    # CLASIFICACIÓN
    # ══════════════════════════════════════════════════════════════════════════
    if tipo == "Clasificación":

        ESTIMATOR_MAP = {
            "Random Forest":  RandomForestClassifier,
            "Decision Tree":  DecisionTreeClassifier,
            "KNN":            KNeighborsClassifier,
            "XGBoost (GBM)":  GradientBoostingClassifier,
            "AdaBoost":       AdaBoostClassifier,
        }

        # ─────────────────────────────────────────────────────────────────────
        # TAB 1: Modelo individual + AUC
        # TAB 2: Benchmark de modelos
        # TAB 3: Benchmark de balanceo
        # ─────────────────────────────────────────────────────────────────────
        tab_ind, tab_bench, tab_balance = st.tabs([
            "🎯  Modelo Individual",
            "🏆  Benchmark de Modelos",
            "⚖  Impacto del Balanceo",
        ])

        # ── Tab 1: Modelo individual ──────────────────────────────────────────
        with tab_ind:
            section("Parámetros de validación cruzada")
            ci1, ci2, ci3 = st.columns(3)
            with ci1:
                modelo_sel = st.selectbox("Modelo", list(
                    ESTIMATOR_MAP.keys()), key="ind_model")
            with ci2:
                n_folds_ind = st.slider("Folds (k)", 2, 15, 5, key="ind_folds")
                cv_method_ind = st.selectbox("Método CV",
                                             ["stratified", "kfold"], key="ind_cv")
            with ci3:
                scoring_ind = st.selectbox("Métrica primaria",
                                           ["accuracy", "f1_weighted",
                                               "precision_weighted", "recall_weighted"],
                                           key="ind_scoring")
                balance_ind = st.selectbox("Balanceo",
                                           ["none", "oversample", "undersample",
                                               "smote", "class_weight"],
                                           key="ind_balance")

            if st.button("▶  Ejecutar validación cruzada", key="run_ind"):
                estimator = ESTIMATOR_MAP[modelo_sel]
                with st.spinner(f"Ejecutando CV con {modelo_sel}…"):
                    try:
                        df_metrics = modelo_sup.validacion_cruzada_completa(
                            modelo=estimator,
                            n_folds=n_folds_ind,
                            scale=True,
                            cv_method=cv_method_ind,
                            balance_method=None if balance_ind == "none" else balance_ind,
                        )
                        section("Resultados — métricas CV")
                        st.dataframe(df_metrics.round(
                            4), use_container_width=True, hide_index=True)

                        fig_m = px.bar(
                            df_metrics, x="Métrica", y="Test (promedio)",
                            error_y="Test (std)",
                            title=f"Métricas de CV — {modelo_sel}",
                            color="Métrica",
                            **plotly_px(),
                        )
                        fig_m.update_layout(
                            **PLOTLY_LAYOUT, showlegend=False, yaxis_range=[0, 1])
                        st.plotly_chart(fig_m, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {e}")

            st.markdown("---")
            section("Curva ROC — AUC")
            st.caption(
                "Entrena el modelo sobre el split train/test y calcula la curva ROC.")

            if st.button("▶  Calcular AUC y curva ROC", key="run_auc"):
                estimator_cls = ESTIMATOR_MAP[modelo_sel]
                with st.spinner("Entrenando y calculando AUC…"):
                    try:
                        try:
                            clf = estimator_cls(random_state=42)
                        except TypeError:
                            clf = estimator_cls()
                        pipe = make_pipeline(SimpleImputer(
                            strategy='median'), StandardScaler(), clf)

                        # FIX: Aplicar el MISMO balanceo que se seleccionó en CV
                        # para que AUC sea consistente con las métricas de CV
                        X_train_auc, y_train_auc = modelo_sup.X_train, modelo_sup.y_train
                        if balance_ind != "none":
                            X_train_auc, y_train_auc = modelo_sup._balance_data(
                                modelo_sup.X_train, modelo_sup.y_train,
                                method=balance_ind, random_state=42
                            )

                        pipe.fit(X_train_auc, y_train_auc)

                        auc_dict, roc_dict, classes = compute_auc_multiclass(
                            pipe, modelo_sup.X_test, modelo_sup.y_test)

                        if auc_dict is None:
                            st.warning(
                                "El modelo no soporta predict_proba, AUC no disponible.")
                        else:
                            macro_auc = auc_dict.get(
                                "macro", list(auc_dict.values())[0])
                            st.markdown(
                                f"<div class='metric-grid'>"
                                + metric_html("AUC Macro", f"{macro_auc:.4f}", "neutral" if macro_auc >= 0.7 else "warn")
                                + metric_html("Modelo", modelo_sel)
                                + metric_html("Clases", len(classes))
                                + "</div>",
                                unsafe_allow_html=True,
                            )

                            fig_roc = go.Figure()
                            fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                                              line=dict(dash="dash", color="#4b5563", width=1))
                            palette = ["#4ade80", "#60a5fa",
                                       "#f59e0b", "#f87171", "#c084fc"]
                            if len(classes) == 2:
                                fpr, tpr = roc_dict["macro"]
                                fig_roc.add_trace(go.Scatter(
                                    x=fpr, y=tpr, mode="lines",
                                    name=f"AUC = {macro_auc:.4f}",
                                    line=dict(color="#4ade80", width=2.5),
                                ))
                            else:
                                for i, cls in enumerate([k for k in roc_dict if k != "macro"]):
                                    fpr, tpr = roc_dict[cls]
                                    fig_roc.add_trace(go.Scatter(
                                        x=fpr, y=tpr, mode="lines",
                                        name=f"Clase {cls} (AUC={auc_dict[cls]:.3f})",
                                        line=dict(
                                            color=palette[i % len(palette)], width=1.8),
                                    ))
                                # macro
                                fpr_m, tpr_m = roc_dict["macro"]
                                fig_roc.add_trace(go.Scatter(
                                    x=fpr_m, y=tpr_m, mode="lines",
                                    name=f"Macro (AUC={macro_auc:.3f})",
                                    line=dict(color="#fff",
                                              width=2.5, dash="dot"),
                                ))
                            fig_roc.update_layout(
                                title=f"Curva ROC — {modelo_sel}",
                                xaxis_title="Tasa de Falsos Positivos",
                                yaxis_title="Tasa de Verdaderos Positivos",
                                **plotly_fig(), height=450,
                            )
                            st.plotly_chart(fig_roc, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al calcular AUC: {e}")
                        with st.expander("Detalle del error"):
                            st.code(traceback.format_exc())

        # ── Tab 2: Benchmark de modelos ───────────────────────────────────────
        with tab_bench:
            section("Comparar todos los modelos")
            cb1, cb2, cb3 = st.columns(3)
            with cb1:
                n_folds_bench = st.slider(
                    "Folds (k)", 2, 15, 5, key="bench_folds")
            with cb2:
                cv_bench = st.selectbox(
                    "Método CV", ["stratified", "kfold"], key="bench_cv")
            with cb3:
                balance_bench = st.selectbox("Balanceo",
                                             ["none", "oversample",
                                                 "undersample", "smote"],
                                             key="bench_balance")

            if st.button("▶  Ejecutar Benchmark de Modelos", key="run_bench"):
                with st.spinner("Evaluando todos los modelos…"):
                    try:
                        df_bench = modelo_sup.benchmark_clasificacion(
                            cv_method=cv_bench,
                            balance_method=None if balance_bench == "none" else balance_bench,
                            n_folds=n_folds_bench,
                        )
                        section("Tabla de resultados")
                        st.dataframe(df_bench.round(
                            4), use_container_width=True, hide_index=True)

                        # Mejor modelo resaltado
                        best = df_bench.iloc[0]
                        st.markdown(
                            f"<div class='metric-grid'>"
                            + metric_html("🏆 Mejor modelo", best["Modelo"])
                            + metric_html("F1 Score",
                                          f"{best['F1']:.4f}", "neutral")
                            + metric_html("Accuracy",
                                          f"{best['Accuracy']:.4f}", "neutral")
                            + metric_html("Precision",
                                          f"{best['Precision']:.4f}")
                            + "</div>",
                            unsafe_allow_html=True,
                        )

                        # Gráfico radar de métricas
                        metrics_cols = ["Accuracy",
                                        "Precision", "Recall", "F1"]
                        fig_radar = go.Figure()
                        palette = ["#4ade80", "#60a5fa",
                                   "#f59e0b", "#f87171", "#c084fc"]
                        for i, row in df_bench.iterrows():
                            vals = [row[m] for m in metrics_cols] + \
                                [row[metrics_cols[0]]]
                            fig_radar.add_trace(go.Scatterpolar(
                                r=vals,
                                theta=metrics_cols + [metrics_cols[0]],
                                fill="toself",
                                name=row["Modelo"],
                                line_color=palette[list(
                                    df_bench.index).index(i) % len(palette)],
                                opacity=0.75,
                            ))
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 1],
                                                color="#4b5563", gridcolor="#1f2433"),
                                angularaxis=dict(color="#9ca3af"),
                                bgcolor="#13161e",
                            ),
                            title="Comparación de Métricas por Modelo",
                            **plotly_fig(), height=480,
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                        # Barras agrupadas
                        df_melt = df_bench.melt(id_vars="Modelo",
                                                value_vars=metrics_cols,
                                                var_name="Métrica", value_name="Valor")
                        fig_bar = px.bar(
                            df_melt, x="Modelo", y="Valor", color="Métrica",
                            barmode="group",
                            title="Métricas por Modelo",
                            **plotly_px(),
                        )
                        fig_bar.update_layout(
                            **PLOTLY_LAYOUT, yaxis_range=[0, 1])
                        st.plotly_chart(fig_bar, use_container_width=True)

                        # AUC por modelo
                        section("AUC por modelo (train/test split)")
                        auc_rows = []
                        for nombre, estimator_cls in ESTIMATOR_MAP.items():
                            try:
                                try:
                                    clf = estimator_cls(random_state=42)
                                except TypeError:
                                    clf = estimator_cls()
                                pipe = make_pipeline(SimpleImputer(
                                    strategy='median'), StandardScaler(), clf)
                                pipe.fit(modelo_sup.X_train,
                                         modelo_sup.y_train)
                                auc_dict_m, _, _ = compute_auc_multiclass(
                                    pipe, modelo_sup.X_test, modelo_sup.y_test)
                                if auc_dict_m:
                                    auc_rows.append(
                                        {"Modelo": nombre, "AUC Macro": round(auc_dict_m["macro"], 4)})
                            except Exception:
                                pass
                        if auc_rows:
                            df_auc = pd.DataFrame(auc_rows).sort_values(
                                "AUC Macro", ascending=False)
                            st.dataframe(
                                df_auc, use_container_width=True, hide_index=True)
                            fig_auc = px.bar(
                                df_auc, x="Modelo", y="AUC Macro",
                                title="AUC Macro por Modelo",
                                color="AUC Macro",
                                color_continuous_scale=["#1f2433", "#4ade80"],
                                **plotly_px(),
                            )
                            fig_auc.update_layout(
                                **PLOTLY_LAYOUT, yaxis_range=[0, 1])
                            st.plotly_chart(fig_auc, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error en benchmark: {e}")
                        with st.expander("Detalle del error"):
                            st.code(traceback.format_exc())

        # ── Tab 3: Impacto del balanceo ────────────────────────────────────────
        with tab_balance:
            section("Comparar técnicas de balanceo")
            cb_b1, cb_b2, cb_b3 = st.columns(3)
            with cb_b1:
                modelo_bal = st.selectbox("Modelo base", list(
                    ESTIMATOR_MAP.keys()), key="bal_model")
            with cb_b2:
                n_folds_bal = st.slider("Folds (k)", 2, 15, 5, key="bal_folds")
                cv_bal = st.selectbox(
                    "Método CV", ["stratified", "kfold"], key="bal_cv")
            with cb_b3:
                scoring_bal = st.selectbox("Métrica",
                                           ["accuracy", "f1_weighted",
                                               "precision_weighted", "recall_weighted"],
                                           key="bal_scoring")

            st.markdown("""
            <div style='background:#1a2235;border:1px solid #2d3a55;border-radius:8px;
                        padding:1rem;margin:1rem 0;font-size:0.82rem;color:#9ca3af;'>
                <strong style='color:#60a5fa;'>¿Qué hace cada técnica?</strong><br>
                <span class='badge'>none</span> Sin modificación del dataset.<br>
                <span class='badge'>oversample</span> Duplica aleatoriamente muestras de la clase minoritaria.<br>
                <span class='badge'>undersample</span> Reduce la clase mayoritaria al tamaño de la minoritaria.<br>
                <span class='badge'>smote</span> Genera muestras sintéticas interpolando vecinos cercanos.<br>
                <span class='badge'>class_weight</span> Penaliza el error en la clase minoritaria durante el entrenamiento.
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶  Analizar impacto del balanceo", key="run_balance"):
                estimator_bal = ESTIMATOR_MAP[modelo_bal]
                with st.spinner("Evaluando todas las técnicas de balanceo…"):
                    try:
                        df_bal = modelo_sup.benchmark_balanceo(
                            modelo=estimator_bal,
                            n_folds=n_folds_bal,
                            scoring=scoring_bal,
                            cv_method=cv_bal,
                            scale=True,
                        )
                        section("Resultados por técnica de balanceo")
                        st.dataframe(df_bal.round(
                            4), use_container_width=True, hide_index=True)

                        # Métricas rápidas
                        best_bal = df_bal.dropna(subset=["Promedio"]).iloc[0]
                        st.markdown(
                            f"<div class='metric-grid'>"
                            + metric_html("Mejor técnica", best_bal["Balance"])
                            + metric_html(f"{scoring_bal}",
                                          f"{best_bal['Promedio']:.4f}", "neutral")
                            + metric_html("Std", f"{best_bal['Std']:.4f}")
                            + "</div>",
                            unsafe_allow_html=True,
                        )

                        # Gráfico principal
                        df_bal_clean = df_bal.dropna(subset=["Promedio"])
                        fig_bal = go.Figure()
                        fig_bal.add_trace(go.Bar(
                            x=df_bal_clean["Balance"],
                            y=df_bal_clean["Promedio"],
                            error_y=dict(type="data", array=df_bal_clean["Std"].tolist(),
                                         color="#4b5563"),
                            marker_color=["#4ade80" if r == best_bal["Balance"] else "#1f2433"
                                          for r in df_bal_clean["Balance"]],
                            marker_line_color="#4ade80",
                            marker_line_width=1,
                            text=df_bal_clean["Promedio"].round(4),
                            textposition="outside",
                        ))
                        fig_bal.update_layout(
                            title=f"Impacto del balanceo sobre {scoring_bal} — {modelo_bal}",
                            xaxis_title="Técnica de balanceo",
                            yaxis_title=scoring_bal,
                            **plotly_fig(), height=420,
                        )
                        st.plotly_chart(fig_bal, use_container_width=True)

                        # Antes vs Después
                        baseline = df_bal_clean[df_bal_clean["Balance"]
                                                == "none"]["Promedio"].values
                        if len(baseline) > 0:
                            baseline_val = baseline[0]
                            df_delta = df_bal_clean[df_bal_clean["Balance"] != "none"].copy(
                            )
                            df_delta["Δ vs baseline"] = df_delta["Promedio"] - \
                                baseline_val
                            section("Ganancia vs sin balanceo")
                            fig_delta = px.bar(
                                df_delta, x="Balance", y="Δ vs baseline",
                                title="Ganancia (Δ) respecto a sin balanceo",
                                color="Δ vs baseline",
                                color_continuous_scale=[
                                    "#f87171", "#13161e", "#4ade80"],
                                **plotly_px(),
                            )
                            fig_delta.add_hline(y=0, line_dash="dash",
                                                line_color="#4b5563", line_width=1)
                            fig_delta.update_layout(**PLOTLY_LAYOUT)
                            st.plotly_chart(
                                fig_delta, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error en análisis de balanceo: {e}")
                        with st.expander("Detalle del error"):
                            st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════════════
    # REGRESIÓN
    # ══════════════════════════════════════════════════════════════════════════
    elif tipo == "Regresión":

        section("Benchmark de Regresión")

        if st.button("▶  Ejecutar Benchmark de Regresión"):
            with st.spinner("Evaluando modelos de regresión…"):
                try:
                    df_reg = modelo_sup.benchmark_regresion()
                    st.dataframe(df_reg.round(
                        4), use_container_width=True, hide_index=True)

                    best_reg = df_reg.iloc[0]
                    st.markdown(
                        f"<div class='metric-grid'>"
                        + metric_html("🏆 Mejor modelo", best_reg["Modelo"])
                        + metric_html("RMSE",
                                      f"{best_reg['RMSE']:.4f}", "neutral")
                        + metric_html("MAE",
                                      f"{best_reg['MAE']:.4f}", "neutral")
                        + metric_html("ER", f"{best_reg['ER']:.4f}")
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                    # RMSE comparativo
                    fig_reg = px.bar(
                        df_reg, x="Modelo", y="RMSE",
                        title="RMSE por Modelo de Regresión",
                        color="RMSE",
                        color_continuous_scale=["#4ade80", "#f87171"],
                        **plotly_px(),
                    )
                    fig_reg.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig_reg, use_container_width=True)

                    # MAE comparativo
                    fig_mae = px.scatter(
                        df_reg, x="RMSE", y="MAE", text="Modelo",
                        title="RMSE vs MAE por Modelo",
                        **plotly_px(),
                    )
                    fig_mae.update_traces(textposition="top center", marker_size=10,
                                          marker_color="#4ade80")
                    fig_mae.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig_mae, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
                    with st.expander("Detalle"):
                        st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: SERIES DE TIEMPO
# ══════════════════════════════════════════════════════════════════════════════

elif pagina == "📈  Series de Tiempo":

    st.markdown("<div class='hero'><h1>Series de <span class='accent'>Tiempo</span></h1><div class='sub'>Pronósticos · benchmarking · comparación de modelos</div></div>", unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.warning("Primero cargue un dataset en la sección **Carga de datos**.")
        st.stop()

    df = st.session_state["df"]

    # ── Selección de columnas ─────────────────────────────────────────────────
    section("Configuración de la serie")

    possible_date_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in ("fecha", "date", "time", "periodo", "mes", "año", "year"))
        or str(df[c].dtype).startswith("datetime")
    ]
    date_options = ["— Sin columna de fecha —"] + possible_date_cols + \
                   [c for c in df.columns if c not in possible_date_cols]

    ct1, ct2 = st.columns(2)
    with ct1:
        col_fecha = st.selectbox("Columna de fechas", date_options,
                                 index=1 if possible_date_cols else 0)
    with ct2:
        cols_val = [c for c in df.columns if c != col_fecha]
        col_valor = st.selectbox("Columna de valores", cols_val)

    if col_fecha == "— Sin columna de fecha —":
        st.error("Seleccione una columna de fechas válida.")
        st.stop()

    # ── Procesamiento ─────────────────────────────────────────────────────────
    try:
        df_ts = df.copy()
        df_ts[col_valor] = pd.to_numeric(
            df_ts[col_valor].astype(str).str.replace(",", "."), errors="coerce")
        df_ts[col_fecha] = parse_dates_column(df_ts[col_fecha])
        invalid_dates = df_ts[df_ts[col_fecha].isna()]
        if len(invalid_dates) > 0:
            st.error(
                f"Se encontraron {len(invalid_dates)} fechas inválidas. Revise la columna.")
            st.stop()
        df_ts[col_fecha] = df_ts[col_fecha].dt.tz_convert(None)
        df_ts = df_ts.dropna(subset=[col_valor])
        df_ts = df_ts.sort_values(col_fecha).set_index(col_fecha)
        serie = df_ts[col_valor].dropna()

        if len(serie) < 10:
            st.error("La serie necesita al menos 10 observaciones.")
            st.stop()

        ts_obj = SeriesTiempo(ts=serie)

    except Exception as e:
        st.error(f"Error al procesar la serie: {e}")
        st.stop()

    # ── Info de la serie ──────────────────────────────────────────────────────
    section("Información de la serie")
    ci1, ci2, ci3, ci4 = st.columns(4)
    with ci1:
        st.markdown(metric_html("Observaciones",
                    f"{len(serie):,}"), unsafe_allow_html=True)
    with ci2:
        st.markdown(metric_html(
            "Media", f"{serie.mean():.2f}", "neutral"), unsafe_allow_html=True)
    with ci3:
        st.markdown(metric_html(
            "Mín", f"{serie.min():.2f}"), unsafe_allow_html=True)
    with ci4:
        st.markdown(metric_html(
            "Máx", f"{serie.max():.2f}"), unsafe_allow_html=True)

    # ── Visualización principal ───────────────────────────────────────────────
    section("Comportamiento temporal")

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=serie.index, y=serie.values,
        mode="lines", name="Serie",
        line=dict(color="#4ade80", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(74,222,128,0.07)",
    ))
    # Media móvil
    if len(serie) >= 12:
        window = max(3, len(serie) // 10)
        ma = serie.rolling(window=window, center=True).mean()
        fig_ts.add_trace(go.Scatter(
            x=ma.index, y=ma.values,
            mode="lines", name=f"Media móvil ({window})",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
        ))
    fig_ts.update_layout(
        title="Serie temporal completa",
        xaxis_title="Fecha", yaxis_title="Valor",
        **plotly_fig(), height=380,
    )
    fig_ts.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Estadísticas adicionales ──────────────────────────────────────────────
    with st.expander("Estadísticas descriptivas completas"):
        st.dataframe(serie.describe().to_frame().T.round(4),
                     use_container_width=True)

    # ── Descomposición estacional ─────────────────────────────────────────────
    section("Análisis de tendencia y estacionalidad")

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        freq_map = {"D": 7, "W": 52, "ME": 12, "M": 12, "MS": 12,
                    "QE": 4, "Q": 4, "A": 1, "YE": 1}
        freq_str = getattr(serie.index, "freqstr", None) or ""
        period = freq_map.get(freq_str, min(12, len(serie) // 3))

        if len(serie) >= 2 * period:
            decomp = seasonal_decompose(
                serie, model="additive", period=period, extrapolate_trend="freq")
            fig_dec = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    subplot_titles=[
                                        "Original", "Tendencia", "Estacionalidad", "Residuo"],
                                    vertical_spacing=0.06)
            pairs = [
                (serie, "#4ade80"),
                (decomp.trend, "#60a5fa"),
                (decomp.seasonal, "#f59e0b"),
                (decomp.resid, "#f87171"),
            ]
            for i, (s, color) in enumerate(pairs, 1):
                fig_dec.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                             line=dict(color=color, width=1.5),
                                             showlegend=False), row=i, col=1)
            fig_dec.update_layout(height=600, title="Descomposición estacional (aditiva)",
                                  **plotly_fig())
            st.plotly_chart(fig_dec, use_container_width=True)
        else:
            st.info(
                f"Se necesitan al menos {2*period} observaciones para la descomposición (actual: {len(serie)}).")
    except Exception as e:
        st.info(f"Descomposición no disponible: {e}")

    # ── Autocorrelación ───────────────────────────────────────────────────────
    section("Autocorrelación (ACF / PACF)")
    try:
        from statsmodels.tsa.stattools import acf, pacf
        max_lags = min(40, len(serie) // 3)
        acf_vals = acf(serie.dropna(),  nlags=max_lags, fft=True)
        pacf_vals = pacf(serie.dropna(), nlags=max_lags)
        ci = 1.96 / np.sqrt(len(serie))
        lags_x = list(range(len(acf_vals)))

        fig_acf = make_subplots(rows=1, cols=2,
                                subplot_titles=["ACF", "PACF"])
        for vals, col_idx, color in [(acf_vals, 1, "#60a5fa"), (pacf_vals, 2, "#f59e0b")]:
            for lag in range(1, len(vals)):
                fig_acf.add_trace(go.Scatter(
                    x=[lag, lag], y=[0, vals[lag]],
                    mode="lines", line=dict(color=color, width=2),
                    showlegend=False,
                ), row=1, col=col_idx)
            fig_acf.add_trace(go.Scatter(
                x=lags_x, y=[ci] * len(lags_x), mode="lines",
                line=dict(color="#4b5563", dash="dash", width=1),
                showlegend=False,
            ), row=1, col=col_idx)
            fig_acf.add_trace(go.Scatter(
                x=lags_x, y=[-ci] * len(lags_x), mode="lines",
                line=dict(color="#4b5563", dash="dash", width=1),
                showlegend=False,
            ), row=1, col=col_idx)
        fig_acf.update_layout(height=340, **plotly_fig(),
                              title="Función de Autocorrelación")
        st.plotly_chart(fig_acf, use_container_width=True)
    except Exception as e:
        st.info(f"ACF/PACF no disponible: {e}")

    # ── Modelos ───────────────────────────────────────────────────────────────
    st.markdown("---")

    tab_ind_ts, tab_bench_ts = st.tabs(
        ["🎯  Modelo individual", "🏆  Comparar todos los modelos"])

    # ── Configuración de split ─────────────────────────────────────────────
    test_pct_ts = st.slider(
        "Tamaño del conjunto de prueba (%)", 10, 40, 20, key="ts_test") / 100
    n_test = max(2, int(len(serie) * test_pct_ts))
    train_ts = serie.iloc[:-n_test]
    test_ts = serie.iloc[-n_test:]

    # Preservar frecuencia
    if serie.index.freq is not None:
        try:
            train_ts = train_ts.copy()
            test_ts = test_ts.copy()
            train_ts.index.freq = serie.index.freq
            test_ts.index.freq = serie.index.freq
        except Exception:
            pass

    st.markdown(f"<span class='badge'>Train: {len(train_ts)} obs</span>"
                f"<span class='badge'>Test: {len(test_ts)} obs</span>", unsafe_allow_html=True)

    # ── Tab: Modelo individual ────────────────────────────────────────────────
    with tab_ind_ts:
        section("Seleccionar y ejecutar modelo")

        modelo_ts_sel = st.selectbox("Modelo", [
            "Holt-Winters",
            "Holt-Winters calibrado",
            "ARIMA",
            "ARIMA calibrado",
        ], key="ts_ind_model")

        if st.button("▶  Ejecutar modelo", key="run_ts_ind"):
            pred = None
            nombre_pred = modelo_ts_sel

            with st.spinner(f"Ajustando {modelo_ts_sel}…"):
                try:
                    ts_train_obj = SeriesTiempo(ts=train_ts)

                    if modelo_ts_sel == "Holt-Winters":
                        m = ts_train_obj.holt_winters()
                        if m:
                            pred = pd.Series(
                                np.array(m.forecast(len(test_ts))), index=test_ts.index)

                    elif modelo_ts_sel == "Holt-Winters calibrado":
                        m = ts_train_obj.holt_winters_calibrado(test_ts)
                        if m:
                            pred = pd.Series(
                                np.array(m.forecast(len(test_ts))), index=test_ts.index)

                    elif modelo_ts_sel == "ARIMA":
                        # FIX: Auto-detecta d + walk-forward para evitar colapso a la media
                        m = ts_train_obj.arima(
                            order=None,           # auto-detecta d con ADF
                            test=test_ts,         # habilita walk-forward
                            walk_forward=True,
                        )
                        if m:
                            pred = pd.Series(
                                np.array(m.forecast(steps=len(test_ts))), index=test_ts.index)

                    elif modelo_ts_sel == "ARIMA calibrado":
                        # Auto-detecta d + walk-forward para forecast robusto
                        m, best_order = ts_train_obj.arima_calibrado(
                            test_ts,
                            walk_forward=True,  # FIX CLAVE: evita colapso
                        )
                        if m:
                            pred = pd.Series(
                                np.array(m.forecast(steps=len(test_ts))), index=test_ts.index)
                            nombre_pred = f"ARIMA {best_order}"

                    if pred is not None and not pred.isna().all():
                        # Métricas
                        errores_obj = SeriesTiempo.calcular_errores(
                            pred, test_ts, nombres=[nombre_pred])
                        df_err = errores_obj.df_errores()

                        st.markdown(
                            f"<div class='metric-grid'>"
                            + metric_html("RMSE", f"{df_err['RMSE'].values[0]:.4f}", "neutral")
                            + metric_html("MAE", f"{df_err['MSE'].apply(np.sqrt).values[0]:.4f}")
                            + metric_html("CORR", f"{df_err['CORR'].values[0]:.4f}", "neutral")
                            + metric_html("RE",
                                          f"{df_err['RE'].values[0]:.4f}")
                            + "</div>",
                            unsafe_allow_html=True,
                        )

                        # Gráfico train / test / predicción
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(
                            x=train_ts.index, y=train_ts.values,
                            name="Train", mode="lines",
                            line=dict(color="#4b5563", width=1.5),
                        ))
                        fig_pred.add_trace(go.Scatter(
                            x=test_ts.index, y=test_ts.values,
                            name="Test (real)", mode="lines",
                            line=dict(color="#60a5fa", width=2),
                        ))
                        fig_pred.add_trace(go.Scatter(
                            x=pred.index, y=pred.values,
                            name=f"Predicción ({nombre_pred})", mode="lines",
                            line=dict(color="#4ade80", width=2, dash="dot"),
                        ))
                        # Banda de error (± std residuos)
                        resid_std = (test_ts - pred).std()
                        fig_pred.add_trace(go.Scatter(
                            x=np.concatenate([pred.index, pred.index[::-1]]),
                            y=np.concatenate(
                                [pred.values + resid_std, (pred.values - resid_std)[::-1]]),
                            fill="toself", fillcolor="rgba(74,222,128,0.08)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="Intervalo ±σ", showlegend=True,
                        ))
                        fig_pred.update_layout(
                            title=f"Pronóstico — {nombre_pred}",
                            xaxis_title="Fecha", yaxis_title="Valor",
                            **plotly_fig(), height=430,
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

                        section("Residuos del modelo")
                        residuos = test_ts - pred
                        fig_resid = make_subplots(rows=1, cols=2,
                                                  subplot_titles=["Residuos en el tiempo", "Distribución de residuos"])
                        fig_resid.add_trace(go.Scatter(
                            x=residuos.index, y=residuos.values, mode="lines+markers",
                            line=dict(color="#f59e0b", width=1.5), showlegend=False,
                        ), row=1, col=1)
                        fig_resid.add_hline(
                            y=0, line_dash="dash", line_color="#4b5563", row=1, col=1)
                        fig_resid.add_trace(go.Histogram(
                            x=residuos.values, marker_color="#f59e0b", opacity=0.8,
                            showlegend=False,
                        ), row=1, col=2)
                        fig_resid.update_layout(height=320, **plotly_fig())
                        st.plotly_chart(fig_resid, use_container_width=True)

                    else:
                        st.error("El modelo no generó predicciones válidas.")

                except Exception as e:
                    st.error(f"Error: {e}")
                    with st.expander("Detalle"):
                        st.code(traceback.format_exc())

    # ── Tab: Comparar todos ───────────────────────────────────────────────────
    with tab_bench_ts:
        section("Seleccionar modelos a comparar")

        cb_col1, cb_col2 = st.columns(2)
        with cb_col1:
            inc_hw = st.checkbox("Holt-Winters",
                                 value=True,  key="inc_hw")
            inc_hw_cal = st.checkbox(
                "Holt-Winters calibrado", value=True,  key="inc_hwc")
        with cb_col2:
            inc_arima = st.checkbox(
                "ARIMA",           value=True,  key="inc_arima")
            inc_arima_cal = st.checkbox(
                "ARIMA calibrado", value=True,  key="inc_arima_c")

        if st.button("▶  Ejecutar comparación de modelos", key="run_ts_bench"):
            if not any([inc_hw, inc_hw_cal, inc_arima, inc_arima_cal]):
                st.warning("Seleccione al menos un modelo.")
            else:
                with st.spinner("Comparando modelos… puede tardar unos segundos"):
                    try:
                        ts_train_obj2 = SeriesTiempo(ts=train_ts)
                        # FIX: No pasar len(test_ts)/len(train_ts) porque causa un segundo split
                        # agresivo dentro de benchmark_personalizado. Usar test_size=0.2 fijo
                        # para mantener suficientes datos de entrenamiento para los modelos.
                        df_resultados = ts_train_obj2.benchmark_personalizado(
                            test_size=0.2,  # 20% del train_ts, no calcular del test_ts
                            incluir_hw=inc_hw,
                            incluir_hw_cal=inc_hw_cal,
                            incluir_arima=inc_arima,
                            incluir_arima_cal=inc_arima_cal,
                            incluir_lstm=False,
                        )

                        if df_resultados is not None:
                            section("Tabla comparativa de errores")
                            st.dataframe(df_resultados.round(
                                4), use_container_width=True, hide_index=True)

                            # Mejor modelo
                            best_ts = df_resultados.sort_values("RMSE").iloc[0]
                            st.markdown(
                                f"<div class='metric-grid'>"
                                + metric_html("🏆 Mejor modelo",
                                              best_ts["Modelo"])
                                + metric_html("RMSE", f"{best_ts['RMSE']:.4f}", "neutral")
                                + metric_html("MAE", f"{best_ts['MSE']:.4f}")
                                + metric_html("CORR", f"{best_ts['CORR']:.4f}", "neutral")
                                + "</div>",
                                unsafe_allow_html=True,
                            )

                            # Gráfico de métricas comparativo
                            fig_comp = px.bar(
                                df_resultados, x="Modelo", y=["RMSE", "RE"],
                                barmode="group",
                                title="Comparación de Errores por Modelo",
                                **plotly_px(),
                            )
                            fig_comp.update_layout(**PLOTLY_LAYOUT)
                            st.plotly_chart(fig_comp, use_container_width=True)

                            # Scatter RMSE vs CORR
                            fig_scatter_ts = px.scatter(
                                df_resultados, x="RMSE", y="CORR", text="Modelo",
                                title="RMSE vs Correlación (arriba-izquierda = mejor)",
                                **plotly_px(),
                            )
                            fig_scatter_ts.update_traces(
                                textposition="top center", marker_size=12,
                                marker_color="#4ade80",
                            )
                            fig_scatter_ts.update_layout(**PLOTLY_LAYOUT)
                            st.plotly_chart(
                                fig_scatter_ts, use_container_width=True)

                            # Pronósticos superpuestos en el periodo de test
                            section("Pronósticos vs valores reales")
                            ts_train_obj3 = SeriesTiempo(ts=train_ts)
                            fig_all_pred = go.Figure()
                            fig_all_pred.add_trace(go.Scatter(
                                x=train_ts.index[-min(60, len(train_ts)):],
                                y=train_ts.values[-min(60, len(train_ts)):],
                                name="Train (últimos)", mode="lines",
                                line=dict(color="#4b5563", width=1.5),
                            ))
                            fig_all_pred.add_trace(go.Scatter(
                                x=test_ts.index, y=test_ts.values,
                                name="Test (real)", mode="lines",
                                line=dict(color="#60a5fa", width=2.5),
                            ))
                            palette_ts = ["#4ade80", "#f59e0b",
                                          "#f87171", "#c084fc"]
                            model_runners = []
                            if inc_hw:
                                model_runners.append(
                                    ("Holt-Winters", lambda t: t.holt_winters()))
                            if inc_hw_cal:
                                model_runners.append(
                                    ("HW Calibrado", lambda t: t.holt_winters_calibrado(test_ts)))
                            if inc_arima:
                                model_runners.append(
                                    ("ARIMA", lambda t: t.arima()))
                            if inc_arima_cal:
                                model_runners.append(
                                    ("ARIMA Calibrado", lambda t: t.arima_calibrado(test_ts)[0]))

                            for i, (nom, runner) in enumerate(model_runners):
                                try:
                                    ts_r = SeriesTiempo(ts=train_ts)
                                    m = runner(ts_r)
                                    if m is None:
                                        continue
                                    p = m.forecast(steps=len(test_ts))
                                    p = pd.Series(p.values if hasattr(p, "values") else p,
                                                  index=test_ts.index)
                                    fig_all_pred.add_trace(go.Scatter(
                                        x=p.index, y=p.values,
                                        name=nom, mode="lines",
                                        line=dict(color=palette_ts[i % len(palette_ts)],
                                                  width=1.8, dash="dot"),
                                    ))
                                except Exception:
                                    pass

                            fig_all_pred.update_layout(
                                title="Pronósticos superpuestos en periodo de test",
                                xaxis_title="Fecha", yaxis_title="Valor",
                                **plotly_fig(), height=450,
                            )
                            st.plotly_chart(
                                fig_all_pred, use_container_width=True)

                        else:
                            st.error(
                                "No se generaron predicciones válidas. Intente con modelos individuales.")

                    except Exception as e:
                        st.error(f"Error en comparación: {e}")
                        with st.expander("Detalle"):
                            st.code(traceback.format_exc())
