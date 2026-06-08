# -*- coding: utf-8 -*-
import traceback
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# =========================================================
# CONFIGURACION DE LA PAGINA
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
    df["Fecha de Solicitud"] = pd.to_datetime(
        df["Fecha de Solicitud"], errors="coerce"
    )
    # Calcular los tres indicadores temporales
    df["Demora_Tecnica"] = (
        df["Fecha de Resolución"] - df["Fecha de Solicitud"]
    ).dt.days
    df["Demora_Registral"] = (
        df["Fecha de Inscripción"] - df["Fecha de Resolución"]
    ).dt.days
    df["Demora_Total"] = (
        df["Fecha de Inscripción"] - df["Fecha de Solicitud"]
    ).dt.days
    # Solo tramites validos
    df = df[
        (df["Demora_Tecnica"] >= 0) &
        (df["Demora_Registral"] >= 0) &
        (df["Demora_Total"] >= 0)
    ]
    df["Anio_Solicitud"] = df["Fecha de Solicitud"].dt.year
    return df

try:
    df = cargar_datos()
except FileNotFoundError:
    st.error(
        "No se encontro data/processed/dinagua_limpio.csv. "
        "Ejecuta primero el notebook practice.ipynb para generarlo."
    )
    st.stop()
except Exception as e:
    st.error(f"Error inesperado al cargar los datos: {e}")
    st.code(traceback.format_exc())
    st.stop()

# =========================================================
# SIDEBAR — FILTROS
# =========================================================
st.sidebar.markdown("## Filtros")
st.sidebar.markdown("---")

p95 = int(df["Demora_Tecnica"].quantile(0.95))
rango_dias = st.sidebar.slider(
    "Rango de demora tecnica (dias)",
    min_value=int(df["Demora_Tecnica"].min()),
    max_value=p95,
    value=(0, p95),
    help="Filtra registros segun la demora tecnica (hasta el percentil 95)"
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
    "Fuente: [DINAGUA](https://www.gub.uy/ministerio-ambiente/) "
    "— Ministerio de Ambiente, Uruguay"
)

# Aplicar filtros
df_f = df[
    (df["Demora_Tecnica"] >= rango_dias[0]) &
    (df["Demora_Tecnica"] <= rango_dias[1]) &
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
# SECCION 1 — PANEL INICIAL (METRICAS)
# =========================================================
anio_max = int(df["Anio_Solicitud"][df["Anio_Solicitud"] < 2026].max())
por_anio = df[df["Anio_Solicitud"] < 2026].groupby("Anio_Solicitud").size()
promedio_anual = round(por_anio.mean(), 0)
ultimo_anio_count = int(por_anio.get(anio_max, 0))

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Registros filtrados", f"{len(df_f):,}")
col2.metric("Mediana demora técnica", f"{df_f['Demora_Tecnica'].median():.0f} dias")
col3.metric(f"Derechos otorgados en {anio_max}", f"{ultimo_anio_count:,}")
col4.metric("Promedio anual histórico", f"{promedio_anual:,.0f}")
col5.metric("Departamentos", f"{df_f['Departamento'].nunique()}")

st.markdown("---")

# =========================================================
# SECCION 2 — RESUMEN DESCRIPTIVO
# =========================================================
st.header("📋 Resumen Descriptivo")

tab_num, tab_cat = st.tabs(["Variables numericas", "Variables categoricas"])

with tab_num:
    st.markdown("Estadisticas del dataset **despues de aplicar los filtros**:")
    cols_desc = ["Demora_Tecnica", "Demora_Registral", "Demora_Total",
                 "Caudal", "Volumen", "profundidad"]
    cols_desc = [c for c in cols_desc if c in df_f.columns]
    desc = df_f[cols_desc].agg(
        ["mean", "median", "std", "min",
         lambda x: x.quantile(0.25),
         lambda x: x.quantile(0.75),
         "max"]
    ).T
    desc.columns = ["Media", "Mediana", "Desv. Std", "Minimo",
                    "Q1 (25%)", "Q3 (75%)", "Maximo"]
    desc["Rango"] = desc["Maximo"] - desc["Minimo"]
    desc = desc[["Mediana", "Media", "Desv. Std", "Minimo",
                 "Q1 (25%)", "Q3 (75%)", "Maximo", "Rango"]]
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

with tab_cat:
    st.markdown("Distribucion de las variables categoricas principales:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Tipo de Uso**")
        uso_cnt = df_f["Uso"].value_counts().reset_index()
        uso_cnt.columns = ["Uso", "Cantidad"]
        uso_cnt["% del total"] = (
            uso_cnt["Cantidad"] / uso_cnt["Cantidad"].sum() * 100
        ).round(1)
        st.dataframe(uso_cnt, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Destino**")
        dest_cnt = df_f["Destino"].value_counts().reset_index()
        dest_cnt.columns = ["Destino", "Cantidad"]
        dest_cnt["% del total"] = (
            dest_cnt["Cantidad"] / dest_cnt["Cantidad"].sum() * 100
        ).round(1)
        st.dataframe(dest_cnt, use_container_width=True, hide_index=True)
    with c3:
        st.markdown("**Departamento**")
        depto_cnt = df_f["Departamento"].value_counts().reset_index()
        depto_cnt.columns = ["Departamento", "Cantidad"]
        depto_cnt["% del total"] = (
            depto_cnt["Cantidad"] / depto_cnt["Cantidad"].sum() * 100
        ).round(1)
        st.dataframe(depto_cnt, use_container_width=True, hide_index=True)

st.markdown("---")

# =========================================================
# SECCION 3 — DISTRIBUCION DE INDICADORES TEMPORALES
# =========================================================
st.header("📊 Distribución de los Indicadores Temporales")
st.markdown(
    "Distribución de las etapas del proceso administrativo "
    "de solicitudes de derechos de uso de agua."
)

indicadores = [
    ("Demora_Tecnica",   "Demora Tecnica (Solicitud → Resolución)",   "#1D9E75"),
    ("Demora_Registral", "Demora Registral (Resolucion → Inscripción)", "#5A7DD8"),
    ("Demora_Total",     "Demora Total (Solicitud → Inscripción)",     "#D85A30"),
]

fig_dist, axes_dist = plt.subplots(3, 1, figsize=(12, 10))

for i, (col, titulo, color) in enumerate(indicadores):
    limite = df_f[col].quantile(0.95)
    datos  = df_f[df_f[col] <= limite][col].dropna()

    sns.histplot(datos, bins=40, color=color, edgecolor="white",
                 kde=False, ax=axes_dist[i])
    axes_dist[i].axvline(
        datos.median(), color="navy", linestyle="--", linewidth=1.5,
        label=f"Mediana: {datos.median():.0f} dias"
    )
    axes_dist[i].axvline(
        datos.mean(), color="red", linestyle="--", linewidth=1.5,
        label=f"Media: {datos.mean():.0f} dias"
    )
    axes_dist[i].set_title(titulo, fontsize=12, fontweight="bold")
    axes_dist[i].set_xlabel("Dias", fontsize=10)
    axes_dist[i].set_ylabel("Frecuencia", fontsize=10)
    axes_dist[i].legend(fontsize=9)
    axes_dist[i].grid(True, linestyle="--", linewidth=0.7,
                      color="gray", alpha=0.5)
    axes_dist[i].set_axisbelow(True)

sns.despine()
plt.tight_layout()
st.pyplot(fig_dist)
st.markdown("---")

# =========================================================
# SECCION 4 — DISTRIBUCION POR CATEGORIAS
# =========================================================
st.header("📈 Distribucion por Categorias")

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

limite = df_f["Demora_Tecnica"].quantile(0.95)
df_box = df_f[df_f["Demora_Tecnica"] <= limite]
orden_box = (df_box.groupby("Regional")["Demora_Tecnica"]
             .median().sort_values(ascending=False).index)
sns.boxplot(data=df_box, x="Demora_Tecnica", y="Regional", order=orden_box,
            palette="tab10", ax=axis3[1, 0])
axis3[1, 0].set_title("Demora Tecnica por Regional (hasta p95)",
                      fontweight="bold")
axis3[1, 0].set_xlabel("Dias")
axis3[1, 0].set_ylabel("")

orden_uso_box = (df_box.groupby("Uso")["Demora_Tecnica"]
                 .median().sort_values(ascending=False).index)
sns.boxplot(data=df_box, x="Demora_Tecnica", y="Uso", order=orden_uso_box,
            palette="tab10", ax=axis3[1, 1])
axis3[1, 1].set_title("Demora Tecnica por Tipo de Uso (hasta p95)",
                      fontweight="bold")
axis3[1, 1].set_xlabel("Dias")
axis3[1, 1].set_ylabel("")

sns.despine()
plt.tight_layout()
st.pyplot(fig3)
st.markdown("---")

# =========================================================
# =========================================================
# SECCION 5 — SERIES TEMPORALES
# =========================================================
st.header("📅 Evolución Temporal")

df_temp = df_f[df_f["Anio_Solicitud"].between(2000, 2025)].copy()

tab_vol, tab_sol = st.tabs(["Volumen autorizado", "Cantidad de solicitudes"])

with tab_vol:
    modo_vol = st.radio(
        "Desglosar por:",
        options=["Total", "Uso", "Departamento"],
        horizontal=True, key="radio_vol"
    )
    fig_v, ax_v = plt.subplots(figsize=(14, 5))

    if modo_vol == "Total":
        vol_anual = df_temp.groupby("Anio_Solicitud")["Volumen"].sum().reset_index()
        vol_anual["Volumen_M"] = vol_anual["Volumen"] / 1e6
        ax_v.fill_between(vol_anual["Anio_Solicitud"], vol_anual["Volumen_M"],
                          alpha=0.25, color="#1D9E75")
        ax_v.plot(vol_anual["Anio_Solicitud"], vol_anual["Volumen_M"],
                  color="#1D9E75", linewidth=2.5, marker="o", markersize=5)
        ax_v.set_ylabel("Volumen autorizado (millones m³)", fontsize=11)
        ax_v.set_title("Volumen total autorizado por año", fontsize=14,
                       fontweight="bold")
    elif modo_vol == "Uso":
        vol_uso = (df_temp.groupby(["Anio_Solicitud", "Uso"])["Volumen"]
                   .sum().unstack(fill_value=0) / 1e6)
        for col in vol_uso.columns:
            ax_v.plot(vol_uso.index, vol_uso[col], linewidth=2,
                      marker="o", markersize=4, label=col)
        ax_v.set_ylabel("Volumen (millones m³)", fontsize=11)
        ax_v.set_title("Volumen autorizado por año y Tipo de Uso",
                       fontsize=14, fontweight="bold")
        ax_v.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        top_depto = df_temp["Departamento"].value_counts().head(5).index
        df_dep = df_temp[df_temp["Departamento"].isin(top_depto)]
        vol_dep = (df_dep.groupby(["Anio_Solicitud", "Departamento"])["Volumen"]
                   .sum().unstack(fill_value=0) / 1e6)
        for col in vol_dep.columns:
            ax_v.plot(vol_dep.index, vol_dep[col], linewidth=2,
                      marker="o", markersize=4, label=col)
        ax_v.set_ylabel("Volumen (millones m³)", fontsize=11)
        ax_v.set_title("Volumen autorizado por año y Departamento (top 5)",
                       fontsize=14, fontweight="bold")
        ax_v.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")

    ax_v.set_xlabel("Año", fontsize=11)
    ax_v.set_xticks(range(2000, 2026, 5))
    ax_v.tick_params(axis="x", rotation=45)
    ax_v.grid(axis="y", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_v)

with tab_sol:
    modo_sol = st.radio(
        "Desglosar por:",
        options=["Total", "Uso", "Departamento"],
        horizontal=True, key="radio_sol"
    )
    fig_s, ax_s = plt.subplots(figsize=(14, 5))

    if modo_sol == "Total":
        sol_anual = df_temp.groupby("Anio_Solicitud").size().reset_index(
            name="Cantidad")
        ax_s.fill_between(sol_anual["Anio_Solicitud"], sol_anual["Cantidad"],
                          alpha=0.25, color="#5A7DD8")
        ax_s.plot(sol_anual["Anio_Solicitud"], sol_anual["Cantidad"],
                  color="#5A7DD8", linewidth=2.5, marker="o", markersize=5)
        ax_s.set_ylabel("Cantidad de solicitudes", fontsize=11)
        ax_s.set_title("Cantidad de solicitudes por año", fontsize=14,
                       fontweight="bold")
    elif modo_sol == "Uso":
        sol_uso = (df_temp.groupby(["Anio_Solicitud", "Uso"])
                   .size().unstack(fill_value=0))
        for col in sol_uso.columns:
            ax_s.plot(sol_uso.index, sol_uso[col], linewidth=2,
                      marker="o", markersize=4, label=col)
        ax_s.set_ylabel("Cantidad de solicitudes", fontsize=11)
        ax_s.set_title("Solicitudes por año y Tipo de Uso",
                       fontsize=14, fontweight="bold")
        ax_s.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        top_depto = df_temp["Departamento"].value_counts().head(5).index
        df_dep2 = df_temp[df_temp["Departamento"].isin(top_depto)]
        sol_dep = (df_dep2.groupby(["Anio_Solicitud", "Departamento"])
                   .size().unstack(fill_value=0))
        for col in sol_dep.columns:
            ax_s.plot(sol_dep.index, sol_dep[col], linewidth=2,
                      marker="o", markersize=4, label=col)
        ax_s.set_ylabel("Cantidad de solicitudes", fontsize=11)
        ax_s.set_title("Solicitudes por año y Departamento (top 5)",
                       fontsize=14, fontweight="bold")
        ax_s.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")

    ax_s.set_xlabel("Año", fontsize=11)
    ax_s.set_xticks(range(2000, 2026, 5))
    ax_s.tick_params(axis="x", rotation=45)
    ax_s.grid(axis="y", linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_s)

st.markdown("---")

# =========================================================
# =========================================================
# SECCION 6 — HEATMAP TEMPORAL
# =========================================================
st.header("🗓️ Solicitudes por Departamento y Año")
st.markdown(
    "Intensidad de solicitudes por departamento a lo largo del tiempo. "
    "Colores más oscuros indican mayor actividad."
)

df_heat = df_f[df_f["Anio_Solicitud"].between(2010, 2025)].copy()
pivot = (
    df_heat.groupby(["Departamento", "Anio_Solicitud"])
    .size().unstack(fill_value=0)
)
pivot.columns = pivot.columns.astype(int)
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

modo_heat = st.radio(
    "Mostrar valores:",
    options=["Absolutos (cantidad)", "Normalizados por departamento (%)"],
    horizontal=True, key="radio_heat"
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

fig_heat, ax_heat = plt.subplots(figsize=(14, 7))
sns.heatmap(datos_heat, annot=True, fmt=fmt, cmap=cmap_heat,
            linewidths=0.4, linecolor="black",
            cbar_kws={"label": cbar_label, "shrink": 0.8}, ax=ax_heat)
ax_heat.set_title("Solicitudes por departamento y año",
                  fontsize=14, fontweight="bold", pad=16)
ax_heat.set_xlabel("Año", fontsize=11)
ax_heat.set_ylabel("Departamento", fontsize=11)
ax_heat.tick_params(axis="x", rotation=45, labelsize=10)
ax_heat.tick_params(axis="y", rotation=0, labelsize=10)
plt.tight_layout()
st.pyplot(fig_heat)

col_max   = datos_heat.max(axis=0)
anio_pico = int(col_max.idxmax())
depto_top = datos_heat.sum(axis=1).idxmax()
depto_crecimiento = (datos_heat[2025] - datos_heat[2010]).idxmax()
st.markdown(
    f"**Observaciones:** el año con mayor actividad global fue **{anio_pico}**. "
    f"El departamento con más solicitudes históricas es **{depto_top}**. "
    f"El mayor crecimiento entre 2010 y 2025 se registró en **{depto_crecimiento}**."
)
st.markdown("---")

# =========================================================
# SECCION 7 — MAPA GEOGRAFICO
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

MAPA_ESTILO = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

if df_mapa.empty:
    st.warning("No hay puntos georreferenciados para los filtros seleccionados.")
else:
    import pydeck as pdk

    if modo_mapa == "Tipo de Uso":
        usos_unicos = sorted(df_mapa["Uso"].dropna().unique())
        tab10 = plt.colormaps["tab10"].resampled(len(usos_unicos))
        color_map = {}
        for i, uso in enumerate(usos_unicos):
            r, g, b, _ = tab10(i)
            color_map[uso] = [int(r*255), int(g*255), int(b*255), 200]

        df_mapa["color"] = df_mapa["Uso"].apply(
            lambda u: color_map.get(u, [150, 150, 150, 180])
        )

        leyenda_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;'>"
        for i, uso in enumerate(usos_unicos):
            r, g, b, _ = tab10(i)
            hex_c = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
            leyenda_html += (
                f"<span style='background:{hex_c};color:white;"
                f"padding:3px 10px;border-radius:12px;font-size:12px;'>{uso}</span>"
            )
        leyenda_html += "</div>"
        st.markdown(leyenda_html, unsafe_allow_html=True)
        st.caption(f"📍 {len(df_mapa):,} registros georreferenciados")

        layer = pdk.Layer(
            "ScatterplotLayer", data=df_mapa,
            get_position=["lon", "lat"], get_color="color",
            get_radius=3000, pickable=True, auto_highlight=True,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=-32.5, longitude=-56.0, zoom=6),
            map_style=MAPA_ESTILO,
            tooltip={"text": "Uso: {Uso}"}
        ))

    else:
        df_mapa_vol = df_mapa.copy()
        df_mapa_vol["Volumen"] = pd.to_numeric(
            df_mapa_vol["Volumen"], errors="coerce")
        df_mapa_vol = df_mapa_vol.dropna(subset=["Volumen"])

        vol_p95 = df_mapa_vol["Volumen"].quantile(0.95)
        df_mapa_vol["vol_norm"] = (
            df_mapa_vol["Volumen"].clip(upper=vol_p95) / vol_p95
        )
        cmap_vol = plt.colormaps["coolwarm"]
        df_mapa_vol["color"] = df_mapa_vol["vol_norm"].apply(
            lambda n: [int(c*255) for c in cmap_vol(n)[:3]] + [200]
        )
        df_mapa_vol["radio"] = (
            1500 + df_mapa_vol["vol_norm"] * 6500).astype(int)

        vol_min = int(df_mapa_vol["Volumen"].min())
        vol_med = int(df_mapa_vol["Volumen"].median())
        vol_max = int(vol_p95)
        r_b, g_b, b_b, _ = cmap_vol(0.0)
        r_m, g_m, b_m, _ = cmap_vol(0.5)
        r_a, g_a, b_a, _ = cmap_vol(1.0)
        hex_b = "#{:02x}{:02x}{:02x}".format(
            int(r_b*255), int(g_b*255), int(b_b*255))
        hex_m = "#{:02x}{:02x}{:02x}".format(
            int(r_m*255), int(g_m*255), int(b_m*255))
        hex_a = "#{:02x}{:02x}{:02x}".format(
            int(r_a*255), int(g_a*255), int(b_a*255))

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;"
            f"margin-bottom:12px;font-size:12px;'>"
            f"<span>Volumen:</span>"
            f"<span style='background:{hex_b};color:white;padding:3px 10px;"
            f"border-radius:12px;'>Bajo (&lt;{vol_min:,} m³)</span>"
            f"<span style='background:{hex_m};color:white;padding:3px 10px;"
            f"border-radius:12px;'>Medio (~{vol_med:,} m³)</span>"
            f"<span style='background:{hex_a};color:white;padding:3px 10px;"
            f"border-radius:12px;'>Alto (&gt;{vol_max:,} m³)</span>"
            f"<span style='color:#aaa;'>Tamaño proporcional al volumen</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.caption(
            f"📍 {len(df_mapa_vol):,} registros con volumen georreferenciados")

        layer = pdk.Layer(
            "ScatterplotLayer", data=df_mapa_vol,
            get_position=["lon", "lat"], get_color="color",
            get_radius="radio", pickable=True, auto_highlight=True,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=-32.5, longitude=-56.0, zoom=6),
            map_style=MAPA_ESTILO,
            tooltip={"text": "Volumen: {Volumen} m³\nUso: {Uso}"}
        ))