import traceback
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from pathlib import Path

try:
    df = cargar_datos()
except FileNotFoundError:
    st.error(
        "⚠️ No se encontró `data/processed/dinagua_limpio.csv`. "
        "Ejecutá primero el notebook `practice.ipynb` para generarlo."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Error inesperado al cargar los datos: {e}")
    st.code(traceback.format_exc())
    st.stop()
# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Gestión de Recursos Hídricos - DINAGUA",
    page_icon="💧",
    layout="wide"
)
sns.set_theme(style="whitegrid", context="notebook")

# =========================================================
# CARGA DE DATOS
# =========================================================
BASE_DIR = Path(__file__).parent

@st.cache_data
def cargar_datos():
    df = pd.read_csv(
        BASE_DIR / "data" / "processed" / "dinagua_limpio.csv",
        encoding="utf-8"
    )
    for col in ["Fecha de Inscripción", "Fecha de Resolución"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["Fecha de Solicitud"] = pd.to_datetime(df["Fecha de Solicitud"], errors="coerce")
    if "Dias_Tramite" not in df.columns:
        df["Dias_Tramite"] = (
            df["Fecha de Inscripción"] - df["Fecha de Resolución"]
        ).dt.days
    df = df[df["Dias_Tramite"] >= 0]
    df["Anio_Solicitud"] = df["Fecha de Solicitud"].dt.year
    return df

try:
    df = cargar_datos()
except FileNotFoundError:
    st.error(
        "⚠️ No se encontró `data/processed/dinagua_limpio.csv`. "
        "Ejecutá primero el notebook `practice.ipynb` para generarlo."
    )
    st.stop()

# =========================================================
# SIDEBAR — FILTROS
# =========================================================
st.sidebar.markdown("## 🔧 Filtros")
st.sidebar.markdown("---")

p95 = int(df["Dias_Tramite"].quantile(0.95))
rango_dias = st.sidebar.slider(
    "Rango de días del trámite",
    min_value=int(df["Dias_Tramite"].min()),
    max_value=p95,
    value=(0, p95),
    help="Filtra registros según la demora del trámite (hasta el percentil 95)"
)

usos_disponibles = sorted(df["Uso"].dropna().unique().tolist())
usos_sel = st.sidebar.multiselect(
    "Tipo de Uso", options=usos_disponibles, default=usos_disponibles
)

regionales_disponibles = sorted(df["Regional"].dropna().unique().tolist())
regionales_sel = st.sidebar.multiselect(
    "Regional", options=regionales_disponibles, default=regionales_disponibles
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "📁 Fuente: [DINAGUA](https://www.gub.uy/ministerio-ambiente/) "
    "— Ministerio de Ambiente, Uruguay"
)

# Aplicar filtros
df_f = df[
    (df["Dias_Tramite"] >= rango_dias[0]) &
    (df["Dias_Tramite"] <= rango_dias[1]) &
    (df["Uso"].isin(usos_sel)) &
    (df["Regional"].isin(regionales_sel))
].copy()

# =========================================================
# ENCABEZADO
# =========================================================
st.title("💧 Gestión de Recursos Hídricos — DINAGUA")
st.markdown(
    "Análisis interactivo de solicitudes de derechos de uso de agua "
    "registradas por la Dirección Nacional de Aguas del Uruguay."
)
st.markdown("---")

# =========================================================
# SECCIÓN 1 — PANEL INICIAL (MÉTRICAS)
# =========================================================
# Cálculo sobre df completo (no filtrado) para las métricas globales
anio_max = int(df["Anio_Solicitud"][df["Anio_Solicitud"] < 2026].max())
por_anio = df[df["Anio_Solicitud"] < 2026].groupby("Anio_Solicitud").size()
promedio_anual = round(por_anio.mean(), 0)
ultimo_anio_count = int(por_anio.get(anio_max, 0))

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Registros filtrados",
    f"{len(df_f):,}"
)
col2.metric(
    "Mediana de demora",
    f"{df_f['Dias_Tramite'].median():.0f} días"
)
col3.metric(
    f"Derechos otorgados en {anio_max}",
    f"{ultimo_anio_count:,}"
)
col4.metric(
    "Promedio anual histórico",
    f"{promedio_anual:,.0f}"
)
col5.metric(
    "Departamentos",
    f"{df_f['Departamento'].nunique()}"
)

st.markdown("---")

# =========================================================
# SECCIÓN 2 — ANÁLISIS DESCRIPTIVO
# =========================================================
st.header("📋 Resumen Descriptivo")

tab_num, tab_cat = st.tabs(["Variables numéricas", "Variables categóricas"])

with tab_num:
    st.markdown("Estadísticas del dataset **después de aplicar los filtros**:")
    cols_desc = ["Dias_Tramite", "Caudal", "Volumen", "profundidad"]
    cols_desc = [c for c in cols_desc if c in df_f.columns]

    desc = df_f[cols_desc].agg(
        ["mean", "median", "std", "min",
         lambda x: x.quantile(0.25),
         lambda x: x.quantile(0.75),
         "max"]
    ).T
    desc.columns = ["Media", "Mediana", "Desv. Std", "Mínimo",
                    "Q1 (25%)", "Q3 (75%)", "Máximo"]
    desc["Rango"] = desc["Máximo"] - desc["Mínimo"]
    desc = desc[["Mediana", "Media", "Desv. Std", "Mínimo",
                 "Q1 (25%)", "Q3 (75%)", "Máximo", "Rango"]]
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

with tab_cat:
    st.markdown("Distribución de las variables categóricas principales:")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Tipo de Uso**")
        uso_cnt = df_f["Uso"].value_counts().reset_index()
        uso_cnt.columns = ["Uso", "Cantidad"]
        uso_cnt["% del total"] = (uso_cnt["Cantidad"] / uso_cnt["Cantidad"].sum() * 100).round(1)
        st.dataframe(uso_cnt, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Destino**")
        dest_cnt = df_f["Destino"].value_counts().reset_index()
        dest_cnt.columns = ["Destino", "Cantidad"]
        dest_cnt["% del total"] = (dest_cnt["Cantidad"] / dest_cnt["Cantidad"].sum() * 100).round(1)
        st.dataframe(dest_cnt, use_container_width=True, hide_index=True)

    with c3:
        st.markdown("**Departamento**")
        depto_cnt = df_f["Departamento"].value_counts().reset_index()
        depto_cnt.columns = ["Departamento", "Cantidad"]
        depto_cnt["% del total"] = (depto_cnt["Cantidad"] / depto_cnt["Cantidad"].sum() * 100).round(1)
        st.dataframe(depto_cnt, use_container_width=True, hide_index=True)

st.markdown("---")

# =========================================================
# SECCIÓN 3 — DISTRIBUCIÓN DEL TARGET
# =========================================================
st.header("📊 Distribución de la Demora del Trámite")
st.markdown(
    "Histograma de `Dias_Tramite` — días transcurridos entre la "
    "Resolución y la Inscripción del derecho de uso."
)

fig1, ax1 = plt.subplots(figsize=(12, 4))
sns.histplot(
    df_f["Dias_Tramite"], bins=50, color="#1D9E75",
    edgecolor="white", kde=True,
    line_kws={"color": "#085041", "linewidth": 2}, ax=ax1
)
ax1.axvline(df_f["Dias_Tramite"].median(), color="coral",
            linestyle="--", linewidth=1.8,
            label=f"Mediana: {df_f['Dias_Tramite'].median():.0f} días")
ax1.axvline(df_f["Dias_Tramite"].mean(), color="navy",
            linestyle="--", linewidth=1.8,
            label=f"Media: {df_f['Dias_Tramite'].mean():.0f} días")
ax1.set_title("Distribución de Dias_Tramite", fontsize=14, fontweight="bold")
ax1.set_xlabel("Días (Resolución → Inscripción)", fontsize=11)
ax1.set_ylabel("Frecuencia", fontsize=11)
ax1.legend(fontsize=10)
sns.despine()
plt.tight_layout()
st.pyplot(fig1)
st.markdown("---")

# =========================================================
# SECCIÓN 5 — PANEL DE DISTRIBUCIONES CATEGÓRICAS
# =========================================================
st.header("📈 Distribución por Categorías")

fig3, axis3 = plt.subplots(2, 2, figsize=(16, 10))

orden_reg = df_f["Regional"].value_counts().index
sns.countplot(data=df_f, y="Regional", hue="Regional", order=orden_reg,
              palette="tab10", legend=False, ax=axis3[0, 0])
axis3[0, 0].set_title("Solicitudes por Regional", fontweight="bold")
axis3[0, 0].set_xlabel("Cantidad")
axis3[0, 0].set_ylabel("")

orden_uso = df_f["Uso"].value_counts().index
sns.countplot(data=df_f, y="Uso", hue="Uso", order=orden_uso,
              palette="tab10", legend=False, ax=axis3[0, 1])
axis3[0, 1].set_title("Solicitudes por Tipo de Uso", fontweight="bold")
axis3[0, 1].set_xlabel("Cantidad")
axis3[0, 1].set_ylabel("")

limite = df_f["Dias_Tramite"].quantile(0.95)
df_box = df_f[df_f["Dias_Tramite"] <= limite]
orden_box = (df_box.groupby("Regional")["Dias_Tramite"]
             .median().sort_values(ascending=False).index)
sns.boxplot(data=df_box, x="Dias_Tramite", y="Regional", order=orden_box,
            palette="tab10", ax=axis3[1, 0])
axis3[1, 0].set_title("Demora por Regional (hasta p95)", fontweight="bold")
axis3[1, 0].set_xlabel("Días")
axis3[1, 0].set_ylabel("")

orden_uso_box = (df_box.groupby("Uso")["Dias_Tramite"]
                 .median().sort_values(ascending=False).index)
sns.boxplot(data=df_box, x="Dias_Tramite", y="Uso", order=orden_uso_box,
            palette="tab10", ax=axis3[1, 1])
axis3[1, 1].set_title("Demora por Tipo de Uso (hasta p95)", fontweight="bold")
axis3[1, 1].set_xlabel("Días")
axis3[1, 1].set_ylabel("")

sns.despine()
plt.tight_layout()
st.pyplot(fig3)
st.markdown("---")
# =========================================================
# SECCIÓN 5 — HEATMAP TEMPORAL: SOLICITUDES POR DEPARTAMENTO
# =========================================================
st.header("🗓️ Solicitudes por Departamento y Año")
st.markdown(
    "Intensidad de solicitudes de derechos de uso por departamento "
    "a lo largo del tiempo. Colores más oscuros indican mayor actividad."
)

# Preparar pivot
df_heat = df_f[df_f["Anio_Solicitud"].between(2010, 2025)].copy()

pivot = (
    df_heat.groupby(["Departamento", "Anio_Solicitud"])
    .size()
    .unstack(fill_value=0)
)
pivot.columns = pivot.columns.astype(int)

# Ordenar departamentos por total descendente
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

# Selector: valores absolutos o normalizados por fila
modo_heat = st.radio(
    "Mostrar valores:",
    options=["Absolutos (cantidad)", "Normalizados por departamento (%)"],
    horizontal=True,
    key="radio_heat"
)

if modo_heat == "Normalizados por departamento (%)":
    datos_heat = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fmt        = ".0f"
    cbar_label = "% del total del departamento"
    cmap_heat  = "Blues"
else:
    datos_heat = pivot
    fmt        = "d"
    cbar_label = "Cantidad de solicitudes"
    cmap_heat  = "YlOrRd"

# Gráfico
fig_heat, ax_heat = plt.subplots(figsize=(14, 7))

sns.heatmap(
    datos_heat,
    annot=True,
    fmt=fmt,
    cmap=cmap_heat,
    linewidths=0.4,
    linecolor="black",
    cbar_kws={"label": cbar_label, "shrink": 0.8},
    ax=ax_heat
)

ax_heat.set_title(
    "Solicitudes de derechos de uso por departamento y año",
    fontsize=14, fontweight="bold", pad=16
)
ax_heat.set_xlabel("Año", fontsize=11)
ax_heat.set_ylabel("Departamento", fontsize=11)
ax_heat.tick_params(axis="x", rotation=45, labelsize=10)
ax_heat.tick_params(axis="y", rotation=0,  labelsize=10)

plt.tight_layout()
st.pyplot(fig_heat)

# Observaciones automáticas debajo del gráfico
col_max = datos_heat.max(axis=0)
anio_pico = int(col_max.idxmax())
depto_top = datos_heat.sum(axis=1).idxmax()
depto_crecimiento = (datos_heat[2025] - datos_heat[2010]).idxmax()

st.markdown(
    f"**Observaciones:** el año con mayor actividad global fue **{anio_pico}**. "
    f"El departamento con más solicitudes históricas es **{depto_top}**. "
    f"El mayor crecimiento entre 2010 y 2025 se registró en "
    f"**{depto_crecimiento}**."
)

st.markdown("---")
# =========================================================
# SECCIÓN 6 — MAPA GEOGRÁFICO
# =========================================================
st.header("🗺️ Distribución Geográfica de las Solicitudes")

modo_mapa = st.radio(
    "Colorear puntos por:",
    options=["Tipo de Uso", "Volumen"],
    horizontal=True
)

df_mapa = df_f[["Latitud", "Longitud", "Uso", "Volumen"]].copy()
df_mapa = df_mapa.rename(columns={"Latitud": "lat", "Longitud": "lon"})
df_mapa["lat"] = pd.to_numeric(df_mapa["lat"], errors="coerce")
df_mapa["lon"] = pd.to_numeric(df_mapa["lon"], errors="coerce")
df_mapa = df_mapa[
    (df_mapa["lat"].between(-35.5, -30.0)) &
    (df_mapa["lon"].between(-59.5, -53.0))
].dropna(subset=["lat", "lon"])

if df_mapa.empty:
    st.warning("⚠️ No hay puntos georreferenciados para los filtros seleccionados.")
else:
    if modo_mapa == "Tipo de Uso":
        # Paleta tab10 — un color por uso
        usos_unicos = sorted(df_mapa["Uso"].dropna().unique())
        tab10 = plt.cm.get_cmap("tab10", len(usos_unicos))
        color_map = {uso: tab10(i) for i, uso in enumerate(usos_unicos)}

        def uso_a_rgb(uso):
            r, g, b, _ = color_map.get(uso, (0.5, 0.5, 0.5, 1))
            return [int(r*255), int(g*255), int(b*255), 180]

        df_mapa["color"] = df_mapa["Uso"].apply(uso_a_rgb)

        # Leyenda
        st.markdown("**Referencia de colores:**")
        cols_leyenda = st.columns(len(usos_unicos))
        for i, uso in enumerate(usos_unicos):
            r, g, b, _ = color_map[uso]
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(r*255), int(g*255), int(b*255)
            )
            cols_leyenda[i].markdown(
                f"<span style='background:{hex_color};"
                f"padding:2px 10px;border-radius:4px;"
                f"color:white;font-size:12px'>{uso}</span>",
                unsafe_allow_html=True
            )
        st.markdown("")
        st.caption(f"📍 {len(df_mapa):,} registros ubicados en el territorio nacional")

        import pydeck as pdk
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_mapa,
            get_position=["lon", "lat"],
            get_color="color",
            get_radius=3000,
            pickable=True,
        )
        view = pdk.ViewState(latitude=-32.5, longitude=-56.0, zoom=6)
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "Uso: {Uso}"}
        ))

    else:
        # Paleta coolwarm — tamaño y color según volumen
        df_mapa["Volumen"] = pd.to_numeric(df_mapa["Volumen"], errors="coerce")
        df_mapa_vol = df_mapa.dropna(subset=["Volumen"]).copy()

        # Normalizar volumen al percentil 95 para escala de color y tamaño
        vol_p95 = df_mapa_vol["Volumen"].quantile(0.95)
        df_mapa_vol["vol_norm"] = (
            df_mapa_vol["Volumen"].clip(upper=vol_p95) / vol_p95
        )

        # Color coolwarm: azul (bajo) → rojo (alto)
        cmap = plt.cm.get_cmap("coolwarm")
        def vol_a_rgb(norm):
            r, g, b, _ = cmap(norm)
            return [int(r*255), int(g*255), int(b*255), 180]

        df_mapa_vol["color"] = df_mapa_vol["vol_norm"].apply(vol_a_rgb)
        # Radio proporcional al volumen (mín 1500m, máx 8000m)
        df_mapa_vol["radio"] = (
            1500 + df_mapa_vol["vol_norm"] * 6500
        ).astype(int)

        st.markdown(
            "🔵 **Azul** = volumen bajo &nbsp;&nbsp; 🔴 **Rojo** = volumen alto &nbsp;&nbsp; "
            "⬤ **Tamaño** proporcional al volumen (hasta p95)"
        )
        st.caption(
            f"📍 {len(df_mapa_vol):,} registros con volumen georreferenciados"
        )

        import pydeck as pdk
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_mapa_vol,
            get_position=["lon", "lat"],
            get_color="color",
            get_radius="radio",
            pickable=True,
        )
        view = pdk.ViewState(latitude=-32.5, longitude=-56.0, zoom=6)
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "Volumen: {Volumen} m³\nUso: {Uso}"}
        ))
