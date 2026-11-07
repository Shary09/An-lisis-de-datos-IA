"""
Interfaz Streamlit – Lab 01: Métodos no supervisados
EDA · ACP · K-Means · HAC (usando el paquete MVC del curso)

Ejecutar:
    streamlit run app_lab01.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Ruta al paquete del curso (ajusta si es necesario)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataFrame import DataFrame as DFCurso
from model import DataFrameModel
from Controller import DataFrameController
from clustering import Clustering
from acp import ACP
from analisis_supervisados import analizar_modelo, analizar_global, introduccion_datos
from guardar_resultados import guardar_comparacion

# ---------------------------------------------------------------------------
# Configuración de página y estilo
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Exploración y análisis de datos",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta visual propia: azul petróleo, turquesa apagado y dorado suave
PRIMARY = "#287D82"       # teal apagado
PRIMARY_DARK = "#183F56"  # azul petróleo
ACCENT = "#C49A5A"        # dorado suave
BG_SIDEBAR = "#17364A"    # azul profundo
BG_MAIN = "#F3F6F8"       # gris azulado claro
CARD = "#FFFFFF"
TEXT = "#243746"

st.markdown(
    f"""
    <style>
    :root {{
      --navy: #183F56;
      --teal: #287D82;
      --teal-soft: #E4F2F2;
      --gold: #C49A5A;
      --canvas: #F3F6F8;
      --ink: #243746;
      --muted: #5F7280;
      --line: #D8E4EA;
    }}
    .stApp {{ background: var(--canvas); color: var(--ink); }}
    [data-testid="stHeader"] {{ background: rgba(243,246,248,.96); }}
    [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #183F56 0%, #123247 100%);
      border-right: 1px solid #244C61;
    }}
    [data-testid="stSidebar"] * {{ color: #EDF5F8; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
      color: #FFFFFF !important; font-size: 1.02rem; letter-spacing: .01em;
    }}
    [data-testid="stSidebar"] hr {{ border-color: rgba(220,239,245,.22); }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{ color: #E8F1F5; }}
    h1, h2, h3, h4 {{ color: var(--navy) !important; font-weight: 700 !important; }}
    h1 {{ font-size: 1.9rem !important; }}
    h2 {{ font-size: 1.45rem !important; }}
    h3 {{ font-size: 1.12rem !important; }}
    p, li, label, .stMarkdown {{ color: var(--ink); }}
    [data-testid="stCaptionContainer"] {{ color: var(--muted); }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1600px; }}
    .app-header {{
      background: linear-gradient(115deg, #FFFFFF 0%, #EAF3F7 100%);
      border: 1px solid #D6E5EC; border-left: 5px solid var(--teal);
      color: var(--navy); padding: 1.05rem 1.35rem; border-radius: 12px;
      margin-bottom: 1.15rem; box-shadow: 0 3px 12px rgba(24,63,86,.045);
    }}
    .app-header h1 {{ color: var(--navy) !important; margin: 0; font-size: 1.55rem !important; }}
    .app-header p {{ color: #526F80; margin: .25rem 0 0 0; font-size: .94rem; }}
    .section-kicker {{ color: #526F80; font-size: .92rem; margin-top: -.35rem; margin-bottom: 1rem; }}
    .stButton > button, [data-testid="stDownloadButton"] button {{
      background: var(--teal) !important; color: #FFFFFF !important;
      border: 1px solid var(--teal) !important; border-radius: 8px !important;
      font-weight: 600 !important; min-height: 2.45rem; transition: .15s ease;
    }}
    .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {{
      background: var(--navy) !important; border-color: var(--navy) !important;
    }}
    .stButton > button:focus, [data-testid="stDownloadButton"] button:focus {{ box-shadow: 0 0 0 .2rem #B8DCDD !important; }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
      border: 1px solid var(--line); border-radius: 10px; overflow: hidden;
      background: #FFFFFF;
    }}
    [data-testid="stMetric"] {{
      background: #FFFFFF; border: 1px solid var(--line); border-radius: 11px;
      padding: 13px 16px; box-shadow: 0 2px 8px rgba(24,63,86,.035);
    }}
    [data-testid="stMetricLabel"] {{ color: #5F7280 !important; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ color: var(--teal) !important; }}
    .stTextInput input, .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div, .stNumberInput input,
    .stTextArea textarea {{ border-radius: 8px !important; border-color: #C9D9E2 !important; background: #FFFFFF !important; color: #1F2D3A !important; caret-color: #1F2D3A !important; }}
    .stTextArea textarea::placeholder {{ color: #687A86 !important; opacity: 1 !important; }}
    [data-testid="stTextArea"] label p {{ color: var(--navy) !important; font-weight: 600 !important; }}
    [data-testid="stFileUploader"] {{ background: #FFFFFF; border: 1px dashed #7CAEB2; border-radius: 10px; padding: .4rem; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--line); }}
    .stTabs [data-baseweb="tab"] {{ color: var(--navy); border-radius: 8px 8px 0 0; padding: .6rem .9rem; }}
    .stTabs [aria-selected="true"] {{ color: var(--teal) !important; border-bottom-color: var(--teal) !important; }}
    [data-testid="stExpander"] {{ background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px; }}
    [data-testid="stAlert"] {{ border-radius: 9px; }}
    .data-panel {{ background: #FFFFFF; border: 1px solid #D7E5EC; border-radius: 12px; padding: 1rem 1rem .8rem 1rem; margin: .25rem 0 1rem 0; }}
    .guide-panel {{ background: linear-gradient(120deg, #F0F8F8 0%, #FFFFFF 100%); border: 1px solid #CDE3E5; border-radius: 12px; padding: 1.1rem 1.25rem; margin-top: 1.15rem; }}
    .guide-panel h3 {{ margin: 0 0 .2rem 0; color: var(--navy) !important; }}
    .guide-panel p {{ color: #496273; font-size: .9rem; margin: .2rem 0 .5rem 0; }}
    .guide-step {{ display: flex; align-items: flex-start; gap: .65rem; margin: .65rem 0; }}
    .guide-num {{ flex: 0 0 28px; height: 28px; width: 28px; display:flex; align-items:center; justify-content:center; border-radius:50%; background:var(--teal); color:#fff; font-weight:700; }}
    .guide-step strong {{ color: var(--navy); }}
    .guide-step span {{ color: #496273; font-size: .88rem; }}
    @media (max-width: 800px) {{ .block-container {{ padding-left: 1rem; padding-right: 1rem; }} .app-header {{ padding: .9rem; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_prep" not in st.session_state:
    st.session_state.df_prep = None
if "df_curso" not in st.session_state:
    st.session_state.df_curso = None
if "filename" not in st.session_state:
    st.session_state.filename = None

# Abrir directamente con un conjunto de ejemplo para que la interfaz principal
# se vea al iniciar, sin exigir una carga manual en la primera ejecución.
if st.session_state.df_raw is None:
    sample_path = ROOT / "titanic4_2026.csv"
    if sample_path.exists():
        try:
            sample_df = pd.read_csv(sample_path)
            st.session_state.df_raw = sample_df.copy()
            st.session_state.df_prep = sample_df.copy()
            st.session_state.df_curso = DFCurso(sample_df)
            st.session_state.filename = sample_path.name
        except Exception:
            pass


def get_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=["float64", "int64", "float32", "int32"]).copy()


def preparar_datos(df: pd.DataFrame, drop_cols, eliminar_dup, imputar, escalado):
    out = df.copy()
    if drop_cols:
        out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    if eliminar_dup:
        out = out.drop_duplicates()
    num = get_numeric(out)
    if imputar and not num.empty:
        out[num.columns] = num.fillna(num.mean())
    if escalado == "Estandarizar" and not num.empty:
        num2 = get_numeric(out)
        out[num2.columns] = (num2 - num2.mean()) / num2.std(ddof=0).replace(0, 1)
    elif escalado == "Normalizar" and not num.empty:
        num2 = get_numeric(out)
        rng = (num2.max() - num2.min()).replace(0, 1)
        out[num2.columns] = (num2 - num2.min()) / rng
    return out


# ===========================================================================
# SIDEBAR – Fuente de datos
# ===========================================================================
with st.sidebar:
    st.markdown("## Cargar datos")
    origen = st.radio("Origen", ["Subir CSV", "CSV local"], index=0)

    sep = st.text_input("Separador", value=",")
    sep_dec = st.text_input("Separador decimal", value=".")
    encoding = st.selectbox("Codificación", ["utf-8", "latin-1", "cp1252"], index=0)
    usar_indice = st.checkbox("Usar primera columna como índice", value=False)

    df_loaded = None
    fname = None

    if origen == "Subir CSV":
        up = st.file_uploader("Selecciona un archivo CSV", type=["csv"], key="csv_upload")
        if st.button("Cargar CSV", type="primary", use_container_width=True, key="load_uploaded_csv"):
            if up is None:
                st.warning("Primero selecciona un archivo CSV.")
            else:
                try:
                    up.seek(0)
                    df_loaded = pd.read_csv(
                        up,
                        sep=sep if sep else ",",
                        decimal=sep_dec if sep_dec else ".",
                        encoding=encoding,
                        index_col=0 if usar_indice else None,
                    )
                    if df_loaded.empty:
                        st.warning("El archivo se leyó, pero no contiene filas de datos.")
                    else:
                        st.session_state.df_raw = df_loaded.copy()
                        st.session_state.df_prep = df_loaded.copy()
                        st.session_state.df_curso = DFCurso(df_loaded.copy())
                        st.session_state.filename = up.name
                        # Evita mostrar resultados de análisis calculados con el CSV anterior.
                        for key in ["acp_obj", "acp_coords", "acp_var", "acp_corr", "acp_ejes", "acp_df_var",
                                    "km_labels", "km_centers", "km_cols", "km_k", "km_metricas",
                                    "hac_Z", "hac_labels", "hac_k", "hac_linkage", "hac_metricas", "hac_centers", "hac_cols",
                                    "tsne_coords", "tsne_kl", "tsne_perp", "umap_coords", "umap_nn", "umap_md"]:
                            st.session_state.pop(key, None)
                        st.success(f"CSV cargado correctamente: {up.name} ({df_loaded.shape[0]} filas, {df_loaded.shape[1]} columnas).")
                        st.rerun()
                except Exception as e:
                    st.error(f"No se pudo leer el CSV. Revisa el separador, el decimal y la codificación. Detalle: {e}")
    else:
        ruta = st.text_input("Ruta del CSV", value="titanic4_2026.csv")
        if st.button("Cargar archivo local", type="primary", use_container_width=True):
            try:
                df_loaded = pd.read_csv(
                    ruta,
                    sep=sep if sep else ",",
                    decimal=sep_dec if sep_dec else ".",
                    encoding=encoding,
                    index_col=0 if usar_indice else None,
                )
                if df_loaded.empty:
                    st.warning("El archivo se leyó, pero no contiene filas de datos.")
                else:
                    st.session_state.df_raw = df_loaded.copy()
                    st.session_state.df_prep = df_loaded.copy()
                    st.session_state.df_curso = DFCurso(df_loaded.copy())
                    st.session_state.filename = Path(ruta).name
                    st.success(f"CSV cargado correctamente: {Path(ruta).name} ({df_loaded.shape[0]} filas, {df_loaded.shape[1]} columnas).")
                    st.rerun()
            except Exception as e:
                st.error(f"No se pudo leer el CSV. Detalle: {e}")

    st.markdown("---")
    st.markdown("## Navegación")
    pagina = st.radio(
        "Sección",
        [
            "Datos",
            "Estadísticas básicas",
            "ACP",
            "K-Means",
            "HAC",
            "t-SNE",
            "UMAP",
            "Clasificación supervisada",
            "Regresión supervisada",
            "Guía de utilización",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Análisis exploratorio y métodos no supervisados")

# ===========================================================================
# HEADER
# ===========================================================================
st.markdown(
    f"""
    <div class="app-header">
        <h1>Exploración y análisis de datos</h1>
        <p>Consulta, prepara y analiza conjuntos de datos desde un solo espacio de trabajo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = st.session_state.df_prep
df_raw = st.session_state.df_raw

# ===========================================================================
# PÁGINA: GUÍA DE UTILIZACIÓN
# ===========================================================================
if pagina == "Guía de utilización":
    st.subheader("Guía de utilización")
    st.write("Sigue estos pasos para cargar, preparar y analizar un conjunto de datos.")
    with st.container(border=True):
        st.markdown("#### 1. Cargar los datos")
        st.markdown("En **Fuente de datos**, selecciona **Subir CSV** para elegir un archivo de tu computadora o **CSV local** para leer uno que esté en la carpeta del proyecto. Revisa el separador, el separador decimal y la codificación.")
        st.info("Puedes comenzar con el archivo de ejemplo `titanic4_2026.csv` incluido en el proyecto.")
    with st.container(border=True):
        st.markdown("#### 2. Explorar la tabla")
        st.markdown("En **Datos** consulta el número de filas y columnas. Ajusta cuántas filas deseas visualizar y abre **Tipos de datos y nulos** para revisar tipos, valores faltantes y cantidad de valores únicos.")
    with st.container(border=True):
        st.markdown("#### 3. Preparar y limpiar")
        st.markdown("Selecciona columnas que quieras excluir y, si corresponde, activa la eliminación de duplicados o la imputación de valores nulos. Elige **Estandarizar** o **Normalizar** solo si el análisis lo requiere y pulsa **Aplicar preparación**. Usa **Restaurar original** para volver al archivo cargado.")
        st.warning("La preparación modifica la versión de trabajo de la sesión, no el archivo CSV original.")
    with st.container(border=True):
        st.markdown("#### 4. Analizar")
        st.markdown("- **Estadísticas básicas:** resumen numérico, distribuciones, correlaciones y boxplots.\n- **ACP:** resume variables numéricas en componentes principales.\n- **K-Means:** agrupa observaciones en una cantidad definida de grupos.\n- **HAC:** explora agrupamientos jerárquicos.\n- **t-SNE y UMAP:** proyectan datos de varias dimensiones en un espacio de menor dimensión para explorar patrones.")
    with st.container(border=True):
        st.markdown("#### 5. Interpretar los resultados")
        st.markdown("Revisa las etiquetas de ejes, escalas, variables seleccionadas y parámetros antes de sacar conclusiones. Los grupos o patrones visuales son indicios que deben interpretarse con el contexto de los datos.")
    with st.container(border=True):
        st.markdown("#### Recomendaciones")
        st.markdown("- Guarda una copia del CSV antes de hacer cambios importantes.\n- Para ACP y clustering, revisa si las variables numéricas necesitan estandarización.\n- Si un análisis falla, comprueba que haya suficientes filas y variables numéricas, y que no queden valores nulos.\n- Carga datos de prueba sin información personal o confidencial.")
    st.success("¡Listo! Ya puedes explorar y analizar tus datos.")

elif df is None:
    st.info("👈 Carga un archivo CSV desde la barra lateral para comenzar, o abre la Guía de utilización.")
    st.stop()

# ===========================================================================
# PÁGINA: DATOS
# ===========================================================================
if pagina == "Datos":
    st.subheader("Datos")
    st.markdown('<div class="section-kicker">Vista general del conjunto de datos cargado en la aplicación.</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Filas", f"{df.shape[0]:,}")
    col_b.metric("Columnas", df.shape[1])
    col_c.metric("Archivo", st.session_state.filename or "Archivo cargado")

    with st.container(border=True):
        st.markdown("### Vista de la tabla")
        top_left, top_right = st.columns([1.6, 1])
        with top_left:
            st.caption("Explora los registros. Puedes ordenar las columnas desde sus encabezados.")
        with top_right:
            n_show = st.slider("Filas a mostrar", 5, max(5, min(200, len(df))), min(15, max(5, len(df))))
        st.dataframe(df.head(n_show), use_container_width=True, height=370, hide_index=False)
        csv_export = df.to_csv(index=False).encode("utf-8-sig")
        dl_col, info_col = st.columns([1, 3])
        with dl_col:
            st.download_button("Descargar CSV", data=csv_export, file_name="datos_exportados.csv", mime="text/csv", use_container_width=True)
        with info_col:
            st.caption("La descarga contiene el conjunto de datos de trabajo que se muestra en esta sección.")

    with st.expander("Preparación y limpieza de datos", expanded=False):
        st.write("Aplica cambios a una copia de trabajo. El CSV original no se modifica.")
        st.caption("Sin aplicar: se conserva el conjunto cargado, incluidos sus nulos, duplicados y escalas originales. Al aplicar: se crea una copia de trabajo y se ejecutan únicamente las operaciones seleccionadas. La imputación sustituye valores faltantes; eliminar duplicados quita filas repetidas; estandarizar centra y escala variables numéricas; normalizar las lleva normalmente al rango 0–1. El archivo original no se sobrescribe.")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            drop_cols = st.multiselect(
                "Columnas a eliminar",
                options=list(df_raw.columns) if df_raw is not None else list(df.columns),
            )
        with c2:
            eliminar_dup = st.checkbox("Eliminar duplicados", value=True)
            imputar = st.checkbox("Imputar valores nulos", value=True)
        with c3:
            escalado = st.selectbox(
                "Escalado",
                ["Ninguno", "Estandarizar", "Normalizar"],
                index=0,
            )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Aplicar preparación", use_container_width=True):
                base = st.session_state.df_raw
                prep = preparar_datos(base, drop_cols, eliminar_dup, imputar, escalado)
                st.session_state.df_prep = prep
                st.session_state.df_curso = DFCurso(prep)
                st.success(f"Conjunto preparado: {prep.shape[0]} filas × {prep.shape[1]} columnas")
                before = base
                after = prep
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Filas", f"{len(before)} → {len(after)}")
                m2.metric("Columnas", f"{before.shape[1]} → {after.shape[1]}")
                m3.metric("Valores nulos", f"{int(before.isna().sum().sum())} → {int(after.isna().sum().sum())}")
                m4.metric("Duplicados", f"{int(before.duplicated().sum())} → {int(after.duplicated().sum())}")
                st.rerun()
        with b2:
            if st.button("Restaurar original", use_container_width=True):
                st.session_state.df_prep = st.session_state.df_raw.copy()
                st.session_state.df_curso = DFCurso(st.session_state.df_raw)
                st.rerun()

    with st.expander("Tipos de datos, valores nulos y valores únicos"):
        info = pd.DataFrame({
            "Tipo de dato": df.dtypes.astype(str),
            "Valores nulos": df.isnull().sum(),
            "Valores únicos": df.nunique(),
        })
        st.dataframe(info, use_container_width=True)

    st.markdown(
        '''<div class="guide-panel">
          <h3>Guía de utilización</h3>
          <p>Pasos rápidos para navegar y trabajar con tus datos.</p>
          <div class="guide-step"><div class="guide-num">1</div><div><strong>Cargar datos</strong><br><span>Usa el panel lateral para subir un archivo CSV o cargar uno local.</span></div></div>
          <div class="guide-step"><div class="guide-num">2</div><div><strong>Explorar la tabla</strong><br><span>Revisa los registros, ordena columnas y descarga una copia en CSV.</span></div></div>
          <div class="guide-step"><div class="guide-num">3</div><div><strong>Ordenar y filtrar</strong><br><span>Utiliza las herramientas de la tabla para inspeccionar los valores y ajustar las filas visibles.</span></div></div>
          <div class="guide-step"><div class="guide-num">4</div><div><strong>Preparar los datos</strong><br><span>Abre “Preparación y limpieza” para tratar nulos, duplicados, columnas y escalas.</span></div></div>
          <div class="guide-step"><div class="guide-num">5</div><div><strong>Continuar con el análisis</strong><br><span>Elige Estadísticas básicas, ACP, K-Means, HAC, t-SNE o UMAP desde el menú lateral.</span></div></div>
        </div>''',
        unsafe_allow_html=True,
    )

# ===========================================================================
# PÁGINA: ESTADÍSTICAS BÁSICAS
# ===========================================================================
elif pagina == "Estadísticas básicas":
    st.subheader("Estadísticas básicas")

    tab_res, tab_dist, tab_corr, tab_box = st.tabs(
        ["Resumen numérico", "Distribución", "Correlación", "Boxplots"]
    )

    num = get_numeric(df)

    with tab_res:
        if num.empty:
            st.warning("No hay columnas numéricas.")
        else:
            # Tarjetas de estadísticas con la paleta de la aplicación
            cols_per_row = 4
            variables = list(num.columns)
            for i in range(0, len(variables), cols_per_row):
                row_vars = variables[i : i + cols_per_row]
                cols = st.columns(len(row_vars))
                for col, var in zip(cols, row_vars):
                    s = num[var]
                    with col:
                        st.markdown(
                            f"""
                            <div class="stat-card">
                                <h4>{var}</h4>
                                <p><b>Mediana:</b> {s.median():.3f}</p>
                                <p><b>Promedio:</b> {s.mean():.3f}</p>
                                <p><b>Desv. est.:</b> {s.std():.3f}</p>
                                <p><b>Mín / Máx:</b> {s.min():.3f} / {s.max():.3f}</p>
                                <p><b>Q1 / Q3:</b> {s.quantile(0.25):.3f} / {s.quantile(0.75):.3f}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.markdown("#### Estadísticas descriptivas")
            st.caption("Resumen estadístico de las variables numéricas: cantidad de valores, promedio, desviación estándar, mínimo, cuartiles y máximo.")
            st.dataframe(num.describe().T.rename_axis("Variable").reset_index().style.format({c: "{:.3f}" for c in num.columns}), use_container_width=True)

        # Categóricas
        cat = df.select_dtypes(include=["object", "category", "bool"])
        if not cat.empty:
            st.markdown("#### Variables categóricas")
            for c in cat.columns:
                vc = df[c].value_counts().head(15)
                st.markdown(f"**{c}**")
                st.bar_chart(vc)

    with tab_dist:
        if num.empty:
            st.warning("No hay columnas numéricas.")
        else:
            vars_sel = st.multiselect(
                "Variables a graficar",
                options=list(num.columns),
                default=list(num.columns)[: min(4, len(num.columns))],
            )
            if vars_sel:
                n = len(vars_sel)
                ncols = min(2, n)
                nrows = (n + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
                axes = np.array(axes).reshape(-1) if n > 1 else [axes]
                for ax, v in zip(axes, vars_sel):
                    ax.hist(num[v].dropna(), bins=25, color=PRIMARY, edgecolor="white", alpha=0.85)
                    ax.set_title(v)
                    ax.grid(True, alpha=0.3)
                for j in range(len(vars_sel), len(axes)):
                    axes[j].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    with tab_corr:
        if num.shape[1] < 2:
            st.warning("Se necesitan al menos 2 variables numéricas.")
        else:
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(
                num.corr(),
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                center=0,
                ax=ax,
                square=True,
            )
            ax.set_title("Matriz de correlación")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab_box:
        if num.empty:
            st.warning("No hay columnas numéricas.")
        else:
            vars_box = st.multiselect(
                "Variables",
                options=list(num.columns),
                default=list(num.columns)[: min(6, len(num.columns))],
                key="box_vars",
            )
            if vars_box:
                fig, ax = plt.subplots(figsize=(max(8, len(vars_box) * 1.2), 5))
                num[vars_box].boxplot(ax=ax, color=dict(boxes=PRIMARY, whiskers=PRIMARY_DARK))
                ax.set_title("Boxplots")
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ===========================================================================
# PÁGINA: ACP
# ===========================================================================
elif pagina == "ACP":
    st.subheader("🔵 Análisis de Componentes Principales (ACP)")

    num = get_numeric(df)
    if num.shape[1] < 2:
        st.error("Se necesitan al menos 2 variables numéricas para ACP.")
        st.stop()

    # Preparar datos estandarizados para ACP
    num_imp = num.fillna(num.mean())
    num_std = (num_imp - num_imp.mean()) / num_imp.std(ddof=0).replace(0, 1)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_comp = st.slider("Número de componentes", 2, min(10, num_std.shape[1]), 5)
    with c2:
        eje_x = st.selectbox("Eje X (componente)", list(range(n_comp)), index=0, format_func=lambda i: f"PC{i+1}")
    with c3:
        eje_y = st.selectbox("Eje Y (componente)", list(range(n_comp)), index=min(1, n_comp - 1), format_func=lambda i: f"PC{i+1}")

    usar_prince = st.checkbox("Usar prince (ACP completo con círculo de correlación)", value=True)

    if st.button("Ejecutar ACP", type="primary"):
        with st.spinner("Calculando ACP..."):
            if usar_prince:
                try:
                    acp = ACP(num_std, n_componentes=n_comp)
                    st.session_state["acp_obj"] = acp
                    st.session_state["acp_coords"] = acp.coordenadas_ind
                    st.session_state["acp_var"] = acp.var_explicada
                    st.session_state["acp_corr"] = acp.correlacion_var
                    st.session_state["acp_ejes"] = [eje_x, eje_y]
                    st.success("ACP (prince) calculado.")
                except Exception as e:
                    st.error(f"Error con prince: {e}. Se usará sklearn.")
                    usar_prince = False

            if not usar_prince:
                coords, modelo, df_var = Clustering.pca(num_std, n_components=n_comp)
                st.session_state["acp_obj"] = None
                st.session_state["acp_coords"] = coords
                st.session_state["acp_var"] = df_var["Varianza explicada (%)"].values
                st.session_state["acp_corr"] = None
                st.session_state["acp_ejes"] = [eje_x, eje_y]
                st.session_state["acp_df_var"] = df_var
                st.success("ACP (sklearn) calculado.")

    if "acp_coords" in st.session_state:
        ejes = st.session_state.get("acp_ejes", [0, 1])
        var_exp = st.session_state["acp_var"]
        coords = st.session_state["acp_coords"]

        # Varianza
        st.markdown("#### Varianza explicada")
        if "acp_df_var" in st.session_state:
            st.dataframe(st.session_state["acp_df_var"], use_container_width=True)
        else:
            df_v = pd.DataFrame({
                "Componente": [f"PC{i+1}" for i in range(len(var_exp))],
                "Varianza (%)": np.round(var_exp, 2),
                "Acumulada (%)": np.round(np.cumsum(var_exp), 2),
            })
            st.dataframe(df_v, use_container_width=True)

        color_options = ["Sin agrupación por variable"] + list(df.columns)
        color_variable = st.selectbox("Colorear individuos por variable (opcional)", color_options, key="pca_color_variable")
        tabs = st.tabs(["Individuos", "Círculo de correlación", "Sobreposición", "Resultados numéricos"])

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(8, 6))
            x = coords.iloc[:, ejes[0]] if hasattr(coords, "iloc") else coords[:, ejes[0]]
            y = coords.iloc[:, ejes[1]] if hasattr(coords, "iloc") else coords[:, ejes[1]]
            # Distinct, color-rich palette helps separate observations visually while preserving the existing layout.
            if color_variable != "Sin agrupación por variable" and color_variable in df.columns and df[color_variable].nunique(dropna=True) <= 20:
                categories = df[color_variable].astype("string").fillna("Missing")
                codes, labels = pd.factorize(categories, sort=True)
                cmap = plt.get_cmap("tab20", max(len(labels), 1))
                scatter = ax.scatter(x, y, c=codes, cmap=cmap, vmin=-0.5, vmax=max(len(labels)-0.5, 0.5), alpha=0.82, s=38, edgecolors="white", linewidths=0.3)
                handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=cmap(i), label=str(label), markersize=7) for i, label in enumerate(labels)]
                ax.legend(handles=handles, title=color_variable, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
            elif color_variable != "Sin agrupación por variable" and color_variable in df.columns and pd.api.types.is_numeric_dtype(df[color_variable]):
                scatter = ax.scatter(x, y, c=df[color_variable].fillna(df[color_variable].median()), cmap="viridis", alpha=0.82, s=38, edgecolors="white", linewidths=0.3)
                fig.colorbar(scatter, ax=ax, label=color_variable)
            else:
                ax.scatter(x, y, color=PRIMARY, alpha=0.78, s=38, edgecolors="white", linewidths=0.3)
            ax.axhline(0, color="gray", ls="--", lw=0.8)
            ax.axvline(0, color="gray", ls="--", lw=0.8)
            vx = round(float(var_exp[ejes[0]]), 2)
            vy = round(float(var_exp[ejes[1]]), 2)
            ax.set_xlabel(f"PC{ejes[0]+1} ({vx}%)")
            ax.set_ylabel(f"PC{ejes[1]+1} ({vy}%)")
            ax.set_title("Plano principal – Individuos")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with tabs[1]:
            acp_obj = st.session_state.get("acp_obj")
            if acp_obj is not None:
                fig, ax = plt.subplots(figsize=(7, 7))
                # Usar método de la clase dibujando sobre el axis actual
                plt.sca(ax)
                acp_obj.plot_circulo(ejes=ejes, titulo="Círculo de correlación")
                st.pyplot(fig)
                plt.close()
            else:
                st.info("El círculo de correlación requiere la librería prince. Activa la opción y vuelve a ejecutar.")

        with tabs[2]:
            st.caption("Los puntos representan individuos y las flechas representan variables; interpreta ambos elementos con la leyenda y los nombres de las variables.")
            acp_obj = st.session_state.get("acp_obj")
            if acp_obj is not None:
                fig, ax = plt.subplots(figsize=(8, 7))
                plt.sca(ax)
                acp_obj.plot_sobreposicion(ejes=ejes, ind_labels=False, titulo="Sobreposición Plano–Círculo")
                st.pyplot(fig)
                plt.close()
            else:
                st.info("La sobreposición requiere prince.")

        with tabs[3]:
            st.markdown("**Coordenadas de individuos (primeras filas)**")
            st.dataframe(coords.head(20), use_container_width=True)
            if st.session_state.get("acp_corr") is not None:
                st.markdown("**Correlación variables–componentes**")
                st.dataframe(st.session_state["acp_corr"], use_container_width=True)

# ===========================================================================
# PÁGINA: K-MEANS
# ===========================================================================
elif pagina == "K-Means":
    st.subheader("K-Means")

    num = get_numeric(df)
    if num.empty:
        st.error("No hay variables numéricas.")
        st.stop()

    num_imp = num.fillna(num.mean())
    estandarizar = st.checkbox("Estandarizar antes de clusterizar", value=True)
    if estandarizar:
        datos_km = (num_imp - num_imp.mean()) / num_imp.std(ddof=0).replace(0, 1)
    else:
        datos_km = num_imp

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        k = st.slider("Número de clústeres (k)", 2, 10, 3)
    with c2:
        init = st.selectbox(
            "Inicialización",
            ["k-means++", "random"],
            help="k-means++ elige centroides iniciales favoreciendo puntos alejados de los ya elegidos y suele ser una buena opción predeterminada. random selecciona los centroides iniciales al azar; puede producir resultados distintos. Usa una semilla fija para reproducibilidad.",
        )
    with c3:
        n_init = st.number_input("n_init", 1, 50, 10)
    with c4:
        max_iter = st.number_input("max_iter", 50, 1000, 300)

    col_run, col_elbow = st.columns(2)
    with col_run:
        run_km = st.button("Ejecutar K-Means", type="primary", use_container_width=True)
    with col_elbow:
        run_elbow = st.button("Método del codo / Silhouette", use_container_width=True)

    if run_elbow:
        with st.spinner("Calculando codo y silhouette..."):
            # No usamos Clustering.kmeans_elbow porque hace plt.show(); lo replicamos
            resultados = []
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            for kk in range(2, 11):
                km = KMeans(n_clusters=kk, n_init=10, random_state=42)
                etiq = km.fit_predict(datos_km)
                sil = silhouette_score(datos_km, etiq)
                resultados.append({"k": kk, "inercia": km.inertia_, "silhouette": sil})
            df_res = pd.DataFrame(resultados)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(df_res["k"], df_res["inercia"], marker="o", color=PRIMARY)
            axes[0].set_xlabel("k")
            axes[0].set_ylabel("Inercia")
            axes[0].set_title("Método del codo")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(df_res["k"], df_res["silhouette"], marker="o", color=ACCENT)
            axes[1].set_xlabel("k")
            axes[1].set_ylabel("Silhouette")
            axes[1].set_title("Silhouette vs k")
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.dataframe(df_res.style.format({"inercia": "{:.1f}", "silhouette": "{:.4f}"}), use_container_width=True)

    if run_km:
        with st.spinner("Ejecutando K-Means..."):
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

            km = KMeans(
                n_clusters=k,
                init=init,
                n_init=int(n_init),
                max_iter=int(max_iter),
                random_state=42,
            )
            etiquetas = km.fit_predict(datos_km)
            st.session_state["km_labels"] = etiquetas
            st.session_state["km_centers"] = km.cluster_centers_
            st.session_state["km_cols"] = list(datos_km.columns)
            st.session_state["km_k"] = k

            metricas = {
                "silhouette": float(silhouette_score(datos_km, etiquetas)),
                "calinski_harabasz": float(calinski_harabasz_score(datos_km, etiquetas)),
                "davies_bouldin": float(davies_bouldin_score(datos_km, etiquetas)),
                "inercia": float(km.inertia_),
            }
            st.session_state["km_metricas"] = metricas

    if "km_labels" in st.session_state:
        metricas = st.session_state["km_metricas"]
        etiquetas = st.session_state["km_labels"]
        centers = st.session_state["km_centers"]
        cols = st.session_state["km_cols"]
        k = st.session_state["km_k"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Silhouette", f"{metricas['silhouette']:.4f}")
        m2.metric("Calinski-Harabasz", f"{metricas['calinski_harabasz']:.1f}")
        m3.metric("Davies-Bouldin", f"{metricas['davies_bouldin']:.4f}")
        m4.metric("Inercia", f"{metricas['inercia']:.1f}")

        # Inercia inter / intra estilo captura
        # Inercia total ≈ suma de distancias al centro global; aquí usamos inercia KMeans como intra
        st.markdown("#### Descomposición de inercia (aprox.)")
        centro_global = datos_km.mean().values
        inercia_total = float(np.sum((datos_km.values - centro_global) ** 2))
        inercia_intra = metricas["inercia"]
        inercia_inter = max(inercia_total - inercia_intra, 0)
        pct_inter = 100 * inercia_inter / inercia_total if inercia_total else 0
        pct_intra = 100 * inercia_intra / inercia_total if inercia_total else 0

        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.barh([0], [100], color=PRIMARY, height=0.5, label="Inercia total")
        ax.barh([1], [pct_inter], color=ACCENT, height=0.5, label=f"Inter-clase {pct_inter:.1f}%")
        ax.barh([1], [pct_intra], left=[pct_inter], color="#64748B", height=0.5, label=f"Intra-clase {pct_intra:.1f}%")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Total", "Partición"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("%")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Inercia total ≈ {inercia_total:.1f}  |  Inter {inercia_inter:.1f}  |  Intra {inercia_intra:.1f}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        tabs_km = st.tabs(["Perfiles (barras)", "Radar", "Asignación", "Centroides"])

        with tabs_km[0]:
            # Bar plot de centroides (estilo Clustering.bar_plot)
            fig, axes = plt.subplots(1, k, figsize=(4 * k, 5), sharey=True)
            if k == 1:
                axes = [axes]
            colores = sns.color_palette("viridis", n_colors=len(cols))
            for i, ax in enumerate(axes):
                vals = centers[i]
                ax.barh(range(len(cols)), vals, color=colores)
                ax.set_title(f"Cluster {i}")
                if i == 0:
                    ax.set_yticks(range(len(cols)))
                    ax.set_yticklabels(cols)
                ax.axvline(0, color="gray", lw=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tabs_km[1]:
            # Radar
            from math import pi

            centros_norm = np.array([
                ((n - n.min()) / (n.max() - n.min()) * 100) if n.max() != n.min() else np.full_like(n, 50)
                for n in centers.T
            ]).T  # shape (k, n_features) wait - we need per cluster
            # Recalcular por fila (cluster)
            centros_pct = []
            for c in centers:
                if c.max() != c.min():
                    centros_pct.append((c - c.min()) / (c.max() - c.min()) * 100)
                else:
                    centros_pct.append(np.full_like(c, 50.0))
            centros_pct = np.array(centros_pct)

            angulos = [n / float(len(cols)) * 2 * pi for n in range(len(cols))]
            angulos += angulos[:1]

            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angulos[:-1], cols, size=8)
            ax.set_ylim(0, 100)
            palette = sns.color_palette("husl", k)
            for i in range(k):
                vals = centros_pct[i].tolist() + [centros_pct[i][0]]
                ax.plot(angulos, vals, linewidth=1.8, label=f"Cluster {i}", color=palette[i])
                ax.fill(angulos, vals, alpha=0.15, color=palette[i])
            ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
            ax.set_title("Perfil radar de clústeres")
            st.pyplot(fig)
            plt.close()

        with tabs_km[2]:
            df_asig = df.copy()
            df_asig["cluster"] = etiquetas
            st.dataframe(df_asig.head(50), use_container_width=True)
            st.bar_chart(pd.Series(etiquetas).value_counts().sort_index())

        with tabs_km[3]:
            df_cent = pd.DataFrame(centers, columns=cols)
            df_cent.index = [f"Cluster {i}" for i in range(k)]
            st.dataframe(df_cent.style.format("{:.3f}"), use_container_width=True)

# ===========================================================================
# PÁGINA: HAC
# ===========================================================================
elif pagina == "HAC":
    st.subheader("🟣 Agrupamiento Jerárquico (HAC)")

    num = get_numeric(df)
    if num.empty:
        st.error("No hay variables numéricas.")
        st.stop()

    num_imp = num.fillna(num.mean())
    estandarizar = st.checkbox("Estandarizar", value=True, key="hac_std")
    if estandarizar:
        datos_hac = (num_imp - num_imp.mean()) / num_imp.std(ddof=0).replace(0, 1)
    else:
        datos_hac = num_imp

    c1, c2, c3 = st.columns(3)
    with c1:
        k_hac = st.slider("Número de clústeres", 2, 10, 3, key="hac_k")
    with c2:
        linkage_m = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
    with c3:
        mostrar_dendro = st.checkbox("Mostrar dendrograma", value=True)

    if st.button("Ejecutar HAC", type="primary"):
        with st.spinner("Calculando HAC..."):
            from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

            metric = "euclidean"
            Z = linkage(datos_hac, method=linkage_m, metric=metric)
            etiquetas = fcluster(Z, k_hac, criterion="maxclust") - 1

            st.session_state["hac_Z"] = Z
            st.session_state["hac_labels"] = etiquetas
            st.session_state["hac_linkage"] = linkage_m

            metricas = {
                "silhouette": float(silhouette_score(datos_hac, etiquetas)),
                "calinski_harabasz": float(calinski_harabasz_score(datos_hac, etiquetas)),
                "davies_bouldin": float(davies_bouldin_score(datos_hac, etiquetas)),
            }
            st.session_state["hac_metricas"] = metricas

            # Centroides
            centros = []
            for c in range(k_hac):
                centros.append(datos_hac.values[etiquetas == c].mean(axis=0))
            st.session_state["hac_centers"] = np.array(centros)
            st.session_state["hac_cols"] = list(datos_hac.columns)

    # Evita errores si quedó un resultado HAC antiguo incompleto en la sesión.
    if "hac_labels" in st.session_state and "hac_metricas" not in st.session_state:
        for clave in ("hac_Z", "hac_labels", "hac_k", "hac_linkage", "hac_metricas", "hac_centers", "hac_cols"):
            st.session_state.pop(clave, None)

    if "hac_labels" in st.session_state and "hac_metricas" in st.session_state:
        metricas = st.session_state["hac_metricas"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Silhouette", f"{metricas['silhouette']:.4f}")
        m2.metric("Calinski-Harabasz", f"{metricas['calinski_harabasz']:.1f}")
        m3.metric("Davies-Bouldin", f"{metricas['davies_bouldin']:.4f}")

        if mostrar_dendro and "hac_Z" in st.session_state:
            from scipy.cluster.hierarchy import dendrogram

            fig, ax = plt.subplots(figsize=(12, 5))
            dendrogram(
                st.session_state["hac_Z"],
                truncate_mode="lastp",
                p=30,
                leaf_rotation=90,
                leaf_font_size=8,
                show_contracted=True,
                ax=ax,
            )
            ax.set_title(f"Dendrograma HAC ({st.session_state['hac_linkage']})")
            ax.set_xlabel("Índice de muestra o (tamaño del clúster)")
            ax.set_ylabel("Distancia")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Perfiles
        centers = st.session_state["hac_centers"]
        cols = st.session_state["hac_cols"]
        k = st.session_state["hac_k"]
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 5), sharey=True)
        if k == 1:
            axes = [axes]
        colores = sns.color_palette("viridis", n_colors=len(cols))
        for i, ax in enumerate(axes):
            ax.barh(range(len(cols)), centers[i], color=colores)
            ax.set_title(f"Cluster {i}")
            if i == 0:
                ax.set_yticks(range(len(cols)))
                ax.set_yticklabels(cols)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        df_asig = df.copy()
        df_asig["cluster"] = st.session_state["hac_labels"]
        st.markdown("#### Asignación de clústeres")
        st.dataframe(df_asig.head(40), use_container_width=True)

# ===========================================================================
# PÁGINA: t-SNE
# ===========================================================================
elif pagina == "t-SNE":
    st.subheader("🟢 t-SNE")

    num = get_numeric(df)
    if num.empty:
        st.error("No hay variables numéricas.")
        st.stop()

    num_imp = num.fillna(num.mean())
    estandarizar = st.checkbox("Estandarizar antes de t-SNE", value=True, key="tsne_std")
    if estandarizar:
        datos_tsne = (num_imp - num_imp.mean()) / num_imp.std(ddof=0).replace(0, 1)
    else:
        datos_tsne = num_imp

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        perplexity = st.slider("Perplexity", 5, min(50, max(5, len(datos_tsne) // 3)), 30)
    with c2:
        n_iter = st.selectbox("Iteraciones", [500, 1000, 1500, 2000], index=1)
    with c3:
        init_tsne = st.selectbox("Init", ["pca", "random"], index=0)
    with c4:
        # color opcional por columna categórica o cluster previo
        color_opts = ["(ninguno)"] + [c for c in df.columns if df[c].nunique() <= 20]
        color_by_name = st.selectbox("Colorear por", color_opts)

    run_std = st.button("Ejecutar t-SNE", type="primary")
    run_var = st.button("Variaciones de perplexity")

    color_by = None
    if color_by_name != "(ninguno)":
        color_by = df[color_by_name].values

    if run_var:
        with st.spinner("Probando varias perplexities..."):
            from sklearn.manifold import TSNE

            configs = [
                {"perplexity": 5, "nombre": "perp=5"},
                {"perplexity": 15, "nombre": "perp=15"},
                {"perplexity": 30, "nombre": "perp=30"},
                {"perplexity": min(50, max(5, len(datos_tsne) // 3)), "nombre": "perp=50 (o máx)"},
            ]
            figs = []
            rows = []
            for cfg in configs:
                nombre = cfg["nombre"]
                perp = min(cfg["perplexity"], max(5, (len(datos_tsne) - 1) // 3))
                modelo = TSNE(
                    n_components=2,
                    perplexity=perp,
                    max_iter=1000,
                    init="pca",
                    random_state=42,
                    learning_rate="auto",
                )
                coords = modelo.fit_transform(datos_tsne)
                kl = getattr(modelo, "kl_divergence_", None)
                rows.append({"config": nombre, "perplexity": perp, "kl_divergence": round(kl, 4) if kl is not None else None})

                fig, ax = plt.subplots(figsize=(5, 4))
                if color_by is not None:
                    sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_by, cmap="tab10", s=18, alpha=0.75)
                    plt.colorbar(sc, ax=ax, fraction=0.046)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1], c=PRIMARY, s=18, alpha=0.65)
                ax.set_title(f"t-SNE – {nombre}")
                ax.set_xlabel("TSNE1")
                ax.set_ylabel("TSNE2")
                ax.grid(True, alpha=0.3)
                figs.append(fig)

            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            cols_f = st.columns(2)
            for i, fig in enumerate(figs):
                with cols_f[i % 2]:
                    st.pyplot(fig)
                plt.close(fig)

    if run_std:
        with st.spinner("Calculando t-SNE..."):
            from sklearn.manifold import TSNE

            n_samples = len(datos_tsne)
            max_perp = max(5, (n_samples - 1) // 3)
            perp = min(perplexity, max_perp)

            modelo = TSNE(
                n_components=2,
                perplexity=perp,
                max_iter=int(n_iter),
                init=init_tsne,
                random_state=42,
                learning_rate="auto",
            )
            coords = modelo.fit_transform(datos_tsne)
            st.session_state["tsne_coords"] = coords
            st.session_state["tsne_kl"] = getattr(modelo, "kl_divergence_", None)
            st.session_state["tsne_perp"] = perp
            st.success(f"t-SNE listo (perplexity={perp})")

    if "tsne_coords" in st.session_state:
        coords = st.session_state["tsne_coords"]
        kl = st.session_state.get("tsne_kl")
        perp = st.session_state.get("tsne_perp")
        m1, m2 = st.columns(2)
        m1.metric("Perplexity usada", perp)
        m2.metric("KL divergence", f"{kl:.4f}" if kl is not None else "—")

        fig, ax = plt.subplots(figsize=(8, 6))
        if color_by is not None:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_by, cmap="tab10", s=22, alpha=0.75)
            plt.colorbar(sc, ax=ax, label=color_by_name)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], c=PRIMARY, s=22, alpha=0.65)
        ax.set_xlabel("TSNE1")
        ax.set_ylabel("TSNE2")
        ax.set_title(f"t-SNE (perplexity={perp})")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        df_c = pd.DataFrame(coords, columns=["TSNE1", "TSNE2"])
        st.dataframe(df_c.head(20), use_container_width=True)

# ===========================================================================
# PÁGINA: UMAP
# ===========================================================================
elif pagina == "UMAP":
    st.subheader("🟠 UMAP")

    num = get_numeric(df)
    if num.empty:
        st.error("No hay variables numéricas.")
        st.stop()

    try:
        import umap
        umap_ok = True
    except ImportError:
        umap_ok = False
        st.error("umap-learn no está instalado. Ejecuta: `pip install umap-learn`")
        st.stop()

    num_imp = num.fillna(num.mean())
    estandarizar = st.checkbox("Estandarizar antes de UMAP", value=True, key="umap_std")
    if estandarizar:
        datos_umap = (num_imp - num_imp.mean()) / num_imp.std(ddof=0).replace(0, 1)
    else:
        datos_umap = num_imp

    c1, c2, c3 = st.columns(3)
    with c1:
        n_neighbors = st.slider("n_neighbors", 2, 50, 15)
    with c2:
        min_dist = st.slider("min_dist", 0.0, 0.99, 0.1, step=0.05)
    with c3:
        color_opts = ["(ninguno)"] + [c for c in df.columns if df[c].nunique() <= 20]
        color_by_name = st.selectbox("Colorear por", color_opts, key="umap_color")

    color_by = None
    if color_by_name != "(ninguno)":
        color_by = df[color_by_name].values

    run_umap = st.button("Ejecutar UMAP", type="primary")
    run_umap_var = st.button("Variaciones n_neighbors / min_dist")

    if run_umap_var:
        with st.spinner("Probando configuraciones UMAP..."):
            configs = [
                {"n_neighbors": 5, "min_dist": 0.1, "nombre": "n_neighbors=5"},
                {"n_neighbors": 15, "min_dist": 0.1, "nombre": "estándar (15, 0.1)"},
                {"n_neighbors": 30, "min_dist": 0.1, "nombre": "n_neighbors=30"},
                {"n_neighbors": 15, "min_dist": 0.0, "nombre": "min_dist=0.0"},
                {"n_neighbors": 15, "min_dist": 0.5, "nombre": "min_dist=0.5"},
            ]
            rows = []
            figs = []
            for cfg in configs:
                nombre = cfg["nombre"]
                modelo = umap.UMAP(
                    n_components=2,
                    n_neighbors=cfg["n_neighbors"],
                    min_dist=cfg["min_dist"],
                    random_state=42,
                )
                coords = modelo.fit_transform(datos_umap)
                rows.append({
                    "config": nombre,
                    "n_neighbors": cfg["n_neighbors"],
                    "min_dist": cfg["min_dist"],
                })
                fig, ax = plt.subplots(figsize=(5, 4))
                if color_by is not None:
                    sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_by, cmap="tab10", s=18, alpha=0.75)
                    plt.colorbar(sc, ax=ax, fraction=0.046)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1], c=ACCENT, s=18, alpha=0.65)
                ax.set_title(f"UMAP – {nombre}")
                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                ax.grid(True, alpha=0.3)
                figs.append(fig)

            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            cols_f = st.columns(2)
            for i, fig in enumerate(figs):
                with cols_f[i % 2]:
                    st.pyplot(fig)
                plt.close(fig)

    if run_umap:
        with st.spinner("Calculando UMAP..."):
            modelo = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
            )
            coords = modelo.fit_transform(datos_umap)
            st.session_state["umap_coords"] = coords
            st.session_state["umap_nn"] = n_neighbors
            st.session_state["umap_md"] = min_dist
            st.success("UMAP listo")

    if "umap_coords" in st.session_state:
        coords = st.session_state["umap_coords"]
        m1, m2 = st.columns(2)
        m1.metric("n_neighbors", st.session_state.get("umap_nn"))
        m2.metric("min_dist", st.session_state.get("umap_md"))

        fig, ax = plt.subplots(figsize=(8, 6))
        if color_by is not None:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_by, cmap="tab10", s=22, alpha=0.75)
            plt.colorbar(sc, ax=ax, label=color_by_name)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], c=ACCENT, s=22, alpha=0.65)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(
            f"UMAP (n_neighbors={st.session_state.get('umap_nn')}, "
            f"min_dist={st.session_state.get('umap_md')})"
        )
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        df_c = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
        st.dataframe(df_c.head(20), use_container_width=True)

# ==========================================================
# LABORATORIO 3 - REGRESIÓN SUPERVISADA
# ==========================================================
elif pagina == "Regresión supervisada":
    st.subheader("Regresión supervisada - Laboratorio 3")
    st.write("Benchmarking reproducible de ocho algoritmos de regresión. La variable objetivo debe ser numérica.")
    target_options = list(df_raw.columns)
    numeric_targets = [c for c in target_options if pd.api.types.is_numeric_dtype(df_raw[c])]
    if not numeric_targets:
        st.warning("El dataset no contiene variables numéricas que puedan utilizarse como objetivo de regresión.")
    else:
        target = st.selectbox("Variable objetivo numérica", numeric_targets, key="reg_target")
        test_size = st.slider("Proporción para prueba", 0.1, 0.4, 0.2, 0.05, key="reg_test_size")
        st.caption("Las imputaciones, codificaciones y escalado se ajustan dentro del pipeline usando solo el conjunto de entrenamiento. Se evalúa en un conjunto de prueba separado.")
        if st.button("Ejecutar benchmarking de regresión", type="primary", key="run_regression"):
            try:
                from framework_ia.models.supervised.regression_benchmark import benchmark_regression
                with st.spinner("Entrenando y evaluando los modelos de regresión..."):
                    benchmark = benchmark_regression(df_raw, target, test_size=test_size, random_state=42)
                    result_table = benchmark["results"]
                    st.session_state["regression_results"] = result_table
                    st.session_state["regression_target"] = target
                    st.session_state.pop("scientific_report_regression", None)
            except Exception as exc:
                st.error(f"No se pudo completar el benchmarking: {exc}")
        if "regression_results" in st.session_state and st.session_state.get("regression_target") == target:
            results = st.session_state["regression_results"]
            st.markdown("### Comparación de modelos")
            st.dataframe(results, use_container_width=True, hide_index=True)
            valid = results.dropna(subset=["RMSE"])
            if not valid.empty:
                best = valid.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Mejor RMSE de prueba", f"{best['RMSE']:.4f}")
                c2.metric("Modelo", str(best["model"]))
                c3.metric("R²", f"{best['R2']:.4f}")
                st.caption("Estas métricas corresponden a prueba. Para seleccionar definitivamente se requiere validación cruzada; el agente lo indicará explícitamente.")

# ==========================================================
# LABORATORIO 2 - CLASIFICACIÓN SUPERVISADA
# ==========================================================


elif pagina == "Clasificación supervisada":
    st.subheader("Clasificación supervisada - Laboratorio 2")
    st.write("Evalúa un algoritmo o compara los cinco modelos usando los mismos datos.")

    objetivo = st.selectbox(
        "Variable objetivo", options=list(df_raw.columns),
        index=list(df_raw.columns).index("survived") if "survived" in df_raw.columns else 0,
        key="sup_objetivo",
    )
    modo = st.radio(
        "Tipo de análisis",
        ["Análisis individual", "Comparación global (5 algoritmos)"],
        horizontal=True, key="sup_modo",
    )
    algoritmos = ["KNN", "Arbol de Decision", "Random Forest", "XGBoost", "AdaBoost"]
    algoritmo = None
    if modo == "Análisis individual":
        algoritmo = st.selectbox("Algoritmo de clasificación", algoritmos, key="sup_algoritmo")

    with st.expander("Descripción de los datos y variable objetivo", expanded=False):
        st.write(introduccion_datos(df_raw, objetivo))

    st.info(
        "En ambos análisis se utiliza 80 % de entrenamiento y 20 % de prueba, "
        "semilla 42 y búsqueda de hiperparámetros con validación cruzada estratificada de 5 particiones. "
        "La comparación global puede tardar varios minutos."
    )

    # La clave identifica el conjunto y la configuración para evitar resultados desactualizados.
    import hashlib
    huella_datos = hashlib.sha256(pd.util.hash_pandas_object(df_raw, index=True).values.tobytes()).hexdigest()
    configuracion = (huella_datos, objetivo, modo, algoritmo)
    if st.session_state.get("sup_configuracion") != configuracion:
        st.session_state.pop("sup_resultado", None)
        st.session_state.pop("sup_global", None)
        st.session_state.pop("sup_ruta_guardado", None)
        st.session_state.pop("sup_error_guardado", None)

    etiqueta_boton = (
        "Entrenar y comparar configuraciones" if modo == "Análisis individual"
        else "Ejecutar comparación de los cinco algoritmos"
    )
    if st.button(etiqueta_boton, type="primary", key="sup_entrenar"):
        try:
            with st.spinner("Entrenando modelos y realizando validación cruzada..."):
                modelo_mvc = DataFrameModel(dataframe=df_raw.copy())
                controlador_mvc = DataFrameController(modelo_mvc, view=None)
                if modo == "Análisis individual":
                    resultado_ejecucion = controlador_mvc.ejecutar(
                        "Clasificación supervisada", objetivo=objetivo,
                        algoritmo=algoritmo, semilla=42,
                    )
                    st.session_state["sup_resultado"] = resultado_ejecucion
                else:
                    resultados_ejecucion = controlador_mvc.ejecutar(
                        "Comparación global supervisada", objetivo=objetivo, semilla=42,
                    )
                    st.session_state["sup_global"] = resultados_ejecucion
                    # Guardar una sola vez por clic, con una carpeta nueva por ejecución.
                    try:
                        ruta_guardado = guardar_comparacion(
                            resultados_ejecucion,
                            df_raw,
                            objetivo,
                            st.session_state.get("filename") or "dataset.csv",
                            ROOT / "resultados_lab02",
                        )
                        st.session_state["sup_ruta_guardado"] = str(ruta_guardado)
                        st.session_state.pop("sup_error_guardado", None)
                    except Exception as error_guardado:
                        st.session_state.pop("sup_ruta_guardado", None)
                        st.session_state["sup_error_guardado"] = str(error_guardado)
                st.session_state["sup_configuracion"] = configuracion
            st.success("Experimento completado.")
        except Exception as e:
            st.session_state.pop("sup_resultado", None)
            st.session_state.pop("sup_global", None)
            st.error(f"Error al entrenar: {e}")

    if st.session_state.get("sup_configuracion") == configuracion:
        if modo == "Análisis individual" and "sup_resultado" in st.session_state:
            resultado = st.session_state["sup_resultado"]
            metricas = resultado["metricas"]
            st.markdown("### Resultados del modelo")
            st.write("Algoritmo:", resultado["algoritmo"])
            st.write("Mejores parámetros:", resultado["mejores_parametros"])
            if "es_baseline" in resultado:
                st.caption(f"Configuración estándar: F1 macro CV = {resultado['es_baseline']['f1_macro_cv']:.4f}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metricas['accuracy']:.4f}")
            c2.metric("Precision macro", f"{metricas['precision_macro']:.4f}")
            c3.metric("Recall macro", f"{metricas['recall_macro']:.4f}")
            c4.metric("F1 macro prueba", f"{metricas['f1_macro']:.4f}")
            st.metric("F1 macro validación cruzada", f"{resultado['f1_macro_cv']:.4f}")
            st.markdown("### Comparación de configuraciones")
            st.dataframe(resultado["variaciones"], use_container_width=True)
            st.markdown("### Matriz de confusión")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(metricas["matriz_confusion"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=resultado["clases"], yticklabels=resultado["clases"], ax=ax)
            ax.set_xlabel("Clase predicha")
            ax.set_ylabel("Clase real")
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("### Interpretación de resultados")
            texto_analisis = ("DESCRIPCIÓN DE LOS DATOS\n\n" + introduccion_datos(df_raw, objetivo)
                              + "\n\nANÁLISIS DEL MODELO\n\n" + analizar_modelo(resultado))
            st.markdown(texto_analisis.replace("\n\n", "\n\n"))
            st.download_button("Descargar análisis individual (.txt)", texto_analisis.encode("utf-8"),
                               "analisis_modelo.txt", "text/plain", key="sup_descarga_narrativa_ind")

        if modo == "Comparación global (5 algoritmos)" and "sup_global" in st.session_state:
            resultados = st.session_state["sup_global"]
            filas = []
            for nombre, res in resultados.items():
                m = res["metricas"]
                filas.append({
                    "Algoritmo": nombre,
                    "F1 macro CV": res["f1_macro_cv"],
                    "Accuracy": m["accuracy"],
                    "Precision macro": m["precision_macro"],
                    "Recall macro": m["recall_macro"],
                    "F1 macro prueba": m["f1_macro"],
                })
            tabla = pd.DataFrame(filas).sort_values("F1 macro CV", ascending=False).reset_index(drop=True)
            ganador = tabla.iloc[0]["Algoritmo"]
            if st.session_state.get("sup_ruta_guardado"):
                st.success("Resultados guardados automáticamente en: " + st.session_state["sup_ruta_guardado"])
                st.caption("Cada ejecución crea una carpeta distinta dentro de resultados_lab02 para no sobrescribir experimentos anteriores.")
            elif st.session_state.get("sup_error_guardado"):
                st.warning("Los modelos se ejecutaron, pero no se pudieron guardar los archivos: "
                           + st.session_state["sup_error_guardado"])
            st.markdown("### Comparación general de los cinco algoritmos")
            st.caption("CV = promedio de validación cruzada sobre entrenamiento. Las otras métricas se calculan en prueba.")
            st.dataframe(tabla.style.format({col: "{:.4f}" for col in tabla.columns if col != "Algoritmo"}),
                         use_container_width=True, hide_index=True)
            st.download_button("Descargar comparación CSV", tabla.to_csv(index=False).encode("utf-8-sig"),
                               "comparacion_modelos.csv", "text/csv", key="sup_descargar")
            st.success(f"Mejor modelo según F1 macro de validación cruzada: {ganador} "
                       f"({resultados[ganador]['f1_macro_cv']:.4f}).")
            st.caption("La selección se basa en la validación cruzada, no en el conjunto de prueba. "
                       "La diferencia entre modelos también debe analizarse antes de concluir cuál conviene utilizar.")

            st.markdown("### Comparación gráfica")
            metrica_grafica = st.selectbox("Métrica que deseas visualizar", [
                "F1 macro CV", "Accuracy", "Precision macro", "Recall macro", "F1 macro prueba"
            ], key="sup_metrica_global")
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(tabla["Algoritmo"], tabla[metrica_grafica])
            ax.set_ylim(0, 1)
            ax.set_ylabel(metrica_grafica)
            ax.tick_params(axis="x", labelrotation=20)
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Configuraciones seleccionadas y matrices de confusión")
            modelo_detalle = st.selectbox("Modelo para inspeccionar", tabla["Algoritmo"].tolist(),
                                         key="sup_detalle_global")
            detalle = resultados[modelo_detalle]
            st.write("Mejores parámetros:", detalle["mejores_parametros"])
            if "es_baseline" in detalle:
                st.caption(f"Configuración estándar: F1 macro CV = {detalle['es_baseline']['f1_macro_cv']:.4f}")
            st.write("Entrenamiento:", detalle["n_entrenamiento"],
                     "| Prueba:", detalle["n_prueba"])
            st.dataframe(detalle["variaciones"], use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(detalle["metricas"]["matriz_confusion"], annot=True, fmt="d",
                        cmap="Blues", xticklabels=detalle["clases"],
                        yticklabels=detalle["clases"], ax=ax)
            ax.set_xlabel("Clase predicha")
            ax.set_ylabel("Clase real")
            ax.set_title(modelo_detalle)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Análisis e interpretación general")
            texto_global = analizar_global(resultados, df_raw, objetivo)
            st.info("Si el texto se quiere copiar al informe, puedes revisarlo aquí y también descargarlo en formato .txt.")
            st.text_area("Análisis para revisar y adaptar al informe", value=texto_global,
                         height=460, key="sup_narrativa_global")
            st.download_button("Descargar análisis completo (.txt)", texto_global.encode("utf-8"),
                               "analisis_comparacion_global.txt", "text/plain", key="sup_descarga_narrativa_global")
            st.caption("El texto interpreta los resultados observados; revisa los términos de las columnas y ajusta la redacción al contexto de tu dataset antes de incluirlo en Overleaf.")


# ============================================================================
# ONE OPENAI DATA-SCIENTIST AGENT PER ANALYTICAL PAGE
# ============================================================================
from framework_ia.agents.section_agents import SECTION_AGENTS
from framework_ia.services.section_evidence_service import build_evidence
from framework_ia.services.openai_interpretation_service import OpenAIInterpretationService
from framework_ia.services.scientific_pdf_service import create_pdf
import hashlib
import json
import os

if pagina in SECTION_AGENTS:
    st.markdown('---')
    st.markdown('### Interpretación técnico-científica del apartado')
    st.caption('Este agente analiza los resultados de ESTA sección: tablas, métricas y evidencia estadística '
               'que sustenta los gráficos. No entrena ni ejecuta modelos. Una solicitud a OpenAI tiene costo.')
    evidence = build_evidence(pagina, df_raw if pagina in ('Datos','Estadísticas básicas') else df, st.session_state)
    if evidence is None:
        st.info('Primero ejecuta el análisis de esta sección para obtener métricas y gráficos.')
    else:
        evidence_hash = hashlib.sha256(json.dumps(evidence,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
        report_key = 'scientific_section_report_' + pagina
        current = st.session_state.get(report_key)
        if current and current.get('hash') != evidence_hash:
            del st.session_state[report_key]
            current = None
            st.info('Los resultados cambiaron. Genera una interpretación nueva para este análisis.')
        if st.button('Generar análisis con OpenAI',key='generate_analysis_'+pagina,type='primary'):
            key=os.getenv('OPENAI_API_KEY')
            model=os.getenv('OPENAI_MODEL')
            try:
                key=key or st.secrets.get('OPENAI_API_KEY')
                model=model or st.secrets.get('OPENAI_MODEL')
            except (FileNotFoundError,KeyError,AttributeError):
                pass
            if not key:
                st.error('No se encontró OPENAI_API_KEY. Configúrala en .streamlit/secrets.toml para activar la interpretación.')
            else:
                with st.spinner('El científico de datos está interpretando los resultados...'):
                    agent=SECTION_AGENTS[pagina](service=OpenAIInterpretationService(api_key=key,model=model))
                    response=agent.interpret(evidence)
                    st.session_state[report_key]={'hash':evidence_hash,'response':response,'evidence':evidence}
                    current=st.session_state[report_key]
        if current:
            response=current['response']
            if response.get('api_warning'):
                st.error(response['api_warning'])
                st.caption('La API no generó un informe. Los siguientes puntos son reglas locales, NO IA generativa.')
            if response.get('scientific_report'):
                interpretation=response['scientific_report']
                st.markdown(interpretation)
                interpretation_mode='openai'
            else:
                interpretation='\n'.join(response.get('findings',[])+response.get('limitations',[]))
                st.info(interpretation)
                interpretation_mode='deterministic'
            st.download_button('Descargar informe de este apartado (PDF)',
                data=create_pdf(pagina,current['evidence'],interpretation,interpretation_mode),
                file_name='analisis_'+pagina.lower().replace(' ','_').replace('-','_')+'.pdf',
                mime='application/pdf',key='download_pdf_'+pagina)
            st.caption('El PDF contiene la interpretación y las evidencias numéricas utilizadas. '
                       'El agente no recibe el CSV completo ni imágenes sin procesar.')
