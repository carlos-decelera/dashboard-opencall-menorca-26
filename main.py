import streamlit as st
import pandas as pd
import httpx
import asyncio
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pyairtable import Api

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ==============================================================================

st.set_page_config(page_title="Attio Deal Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Contenedores nativos de Streamlit con estilo de tarjeta */
    [data-testid="stVerticalBlock"] > div:has(div.stPlotlyChart) {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #efefef;
        margin-bottom: 20px;
    }
    /* Fondos de la aplicación */
    .stApp { background-color: #f8f9fa; }
    [data-testid="stAppViewContainer"] { background-color: #f9f9fb; }
    
    /* Estilos del Header */
    .outer-container { display: flex; justify-content: center; width: 100%; }
    .container { display: flex; align-items: center; }
    .logo-img { width: 80px; height: 80px; margin-right: 20px; }
    .title-text { font-size: 2.5em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# NUEVAS CONSTANTES AIRTABLE (Añadir en la sección de constantes)
# ==============================================================================
AIRTABLE_API_KEY = st.secrets["AIRTABLE_API_KEY"]
AIRTABLE_BASE_ID = "appi6rzeSzAiwou4K"
AIRTABLE_TABLE_NAME = "tblUudGTbW2y9mzaO"

# ==============================================================================
# NUEVA FUNCIÓN DE EXTRACCIÓN AIRTABLE
# ==============================================================================

@st.cache_data(ttl=600)
def get_airtable_df():
    # Inicialización
    api = Api(AIRTABLE_API_KEY)
    table = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)
    
    # 1. Extraemos solo el diccionario de campos de cada registro en una sola línea (list comprehension)
    # table.all() ya gestiona la paginación de 100 en 100 de forma eficiente
    raw_data = [r['fields'] for r in table.all(view="Applicants DEC MENORCA 2025")]
    
    # 2. Convertimos a DataFrame de golpe
    df_air = pd.DataFrame(raw_data)
    
    # 3. Limpieza de columnas (si no existen las crea como NaN para evitar errores)
    cols_interes = {
        "Created": "fecha_raw", 
        "PH1_reference_$startups": "reference"
    }
    # Renombramos y nos quedamos solo con lo que necesitamos
    df_air = df_air.rename(columns=cols_interes)[list(cols_interes.values())]
    
    # 4. Transformación vectorizada de fechas con Pandas
    # El formato 5/9/2025 1:38am es flexible, 'dayfirst' ayuda si es formato europeo
    df_air["fecha_dt"] = pd.to_datetime(df_air["fecha_raw"], errors='coerce')
    
    return df_air

with st.spinner("Conectando con Airtable..."):
    df = get_airtable_df()

def map_airtable_categories(df_air):
    if df_air.empty:
        return df_air
        
    # 1. Limpieza de strings (quitamos saltos de línea y espacios extra)
    df_air["reference_clean"] = df_air["reference"].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

    # 2. Diccionario de mapeo basado en tus valores reales
    mapeo_air = {
        # --- MARKETING ---
        'Your Newsletter': 'Marketing',
        'Online Magazine/blog/newsletter/press': 'Marketing',
        'Instagram': 'Marketing',
        'Google': 'Marketing',
        'Twitter': 'Marketing',
        'LinkedIn': 'Marketing', # Si no es outreach directo, suele caer en marketing social
        
        # --- REFERRAL ---
        'Referral': 'Referral',
        
        # --- OUTREACH ---
        'Decelera team reached through email': 'Outreach',
        'Decelera Team reached through Linkedin': 'Outreach',
        'Startup Event / Network Event (i.e. SXSW)': 'Outreach',
        
        # --- OTROS / PLATAFORMAS ---
        'Gust': 'Otros',
        'YouNoodle': 'Otros'
    }

    # 3. Aplicar mapeo
    df_air["categoria_reference"] = df_air["reference_clean"].map(mapeo_air).fillna("Otros")
    
    return df_air

# ==============================================================================
# CONSTANTES Y CONFIGURACIÓN DE API ATTIO
# ==============================================================================

ATTIO_API_KEY = st.secrets["ATTIO_API_KEY"]
DEALS_ID = "dbcd94bf-ec33-4f00-a7c8-74f57a559869"
DEAL_FLOW_ID = "54265eb6-d53d-465d-ad35-4e823e135629"
BASE_URL = "https://api.attio.com/v2"
HEADERS = {
    "Authorization": f"Bearer {ATTIO_API_KEY}",
    "Content-Type": "application/json"
}

# ==============================================================================
# FUNCIONES DE EXTRACCIÓN Y TRANSFORMACIÓN (LÓGICA CORE)
# ==============================================================================

def extract_value(attr_list):
    """Extrae el dato real de los valores de atributo de Attio."""
    if not attr_list: return None
    extracted = []
    for item in attr_list:
        attr_type = item.get("attribute_type", "")
        val = None
        if attr_type == "status": val = item.get("status", {}).get("title")
        elif attr_type == "select": val = item.get("option", {}).get("title")
        elif attr_type == "domain": val = item.get("domain")
        elif attr_type == "location":
            val = ", ".join(filter(None, [
                item.get("line_1"), item.get("locality"),
                item.get("region"), item.get("postcode"),
                item.get("country_code"),
            ]))
        elif attr_type == "personal-name": val = item.get("full_name")
        elif attr_type == "email-address": val = item.get("email_address")
        elif attr_type == "phone-number": val = item.get("phone_number")
        elif attr_type == "record-reference": val = item.get("target_record_id")
        elif attr_type == "actor-reference": val = item.get("referenced_actor_id")
        elif attr_type == "interaction": val = item.get("interacted_at")
        elif attr_type == "currency": val = item.get("currency_value")
        elif attr_type in ("text", "number", "date", "timestamp", "checkbox", "rating"):
            val = item.get("value")
        else: val = item.get("value")
        if val is not None: extracted.append(str(val))
    return extracted[0] if len(extracted) == 1 else extracted or None

async def fetch_data(client, url, payload=None):
    """Maneja la paginación de la API de Attio."""
    all_data = []
    limit, offset = 100, 0
    while True:
        current_payload = payload.copy() if payload else {}
        current_payload.update({"limit": limit, "offset": offset})
        response = await client.post(url, headers=HEADERS, json=current_payload)
        response.raise_for_status()
        data = response.json().get("data", [])
        all_data.extend(data)
        if len(data) < limit: break
        offset += limit
    return all_data

def transform_attio_to_df(attio_data):
    """Convierte la respuesta JSON de Attio en un DataFrame de Pandas."""
    rows = []
    for record in attio_data:
        record_id = record.get("id", {}).get("record_id") or record.get("parent_record_id")
        row = {"record_id": record_id, "created_at": record.get("created_at")}
        values_source = record.get("entry_values", {}) or record.get("values", {})
        for attr_name, attr_list in values_source.items():
            row[attr_name] = extract_value(attr_list)
        rows.append(row)
    return pd.DataFrame(rows)

# ==============================================================================
# GESTIÓN DE DATOS ASÍNCRONA Y CACHÉ
# ==============================================================================

@st.cache_data(ttl=600)
def get_combined_dataframe():
    async def run_parallel_fetches():
        async with httpx.AsyncClient() as client:
            records_task = fetch_data(client, f"{BASE_URL}/objects/{DEALS_ID}/records/query", 
                                    payload={"$or": [{"stage": "Menorca 2026"}, {"stage": "Leads Menorca 2026"}]})
            entries_task = fetch_data(client, f"{BASE_URL}/lists/{DEAL_FLOW_ID}/entries/query")
            return await asyncio.gather(records_task, entries_task)

    raw_records, raw_entries = asyncio.run(run_parallel_fetches())
    df_rec = transform_attio_to_df(raw_records)
    df_ent = transform_attio_to_df(raw_entries)
    
    if df_rec.empty or df_ent.empty: return pd.DataFrame()
    return pd.merge(df_rec, df_ent, on="record_id")

# ==============================================================================
# HEADER Y CONTROLES DE INTERFAZ
# ==============================================================================

col_title, col_btn = st.columns([0.85, 0.15])
with col_btn:
    if st.button("🔄 Refrescar"):
        get_combined_dataframe.clear()
        st.rerun()

st.markdown("""
<div class="outer-container"><div class="container">
    <img class="logo-img" src="https://images.squarespace-cdn.com/content/v1/67811e8fe702fd5553c65249/c5500619-9712-4b9b-83ee-a697212735ae/Disen%CC%83o+sin+ti%CC%81tulo+%2840%29.png">
    <h1 class="title-text">Deal Flow - Funnel<br>Menorca 2026</h1>
</div></div>
""", unsafe_allow_html=True)

st.markdown("")

with st.spinner("Sincronizando con Attio..."):
    df = get_combined_dataframe()

# ==============================================================================
# SECCIÓN: KPIs Y FILTROS TEMPORALES
# ==============================================================================

# KPIs rápidos
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Deals", len(df))
col2.metric("Not Qualified", f"{len(df[df['status'] == 'Not qualified'])} ({round(len(df[df['status'] == 'Not qualified'])/len(df)*100, 2)} %)")
col3.metric("Initial screening", f"{len(df[df['status'] == 'Initial screening'])} ({round(len(df[df['status']=='Initial screening'])/len(df)*100, 2)} %)")
col4.metric("First interaction", f"{len(df[df['status']=='First interaction'])} ({round(len(df[df['status']=='First interaction'])/len(df)*100, 2)} %)")
col5.metric("Deep dive", f"{len(df[df['status']=='Deep dive'])} ({round(len(df[df['status']=='Deep dive'])/len(df)*100, 2)} %)")

# Filtros de periodo
col_filtro1, col_filtro2, col_filtro3, col_espacio = st.columns([0.2, 0.2, 0.2, 0.4])
if "periodo" not in st.session_state: st.session_state.periodo = "Todo"

with col_filtro1:
    if st.button("🌐 Visión Global", use_container_width=True):
        st.session_state.periodo = "Todo"; st.rerun()
with col_filtro2:
    if st.button("📅 Visión Semanal", use_container_width=True):
        st.session_state.periodo = "Semana"; st.rerun()
with col_filtro3:
    if st.button("📅 Semana Anterior", use_container_width=True):
        st.session_state.periodo = "Semana Anterior"; st.rerun()

# ==============================================================================
# LÓGICA DE FILTRADO PARA AMBOS (Dentro del flujo principal)
# ==============================================================================
with st.spinner("Sincronizando con Airtable..."):
    df_air = get_airtable_df()

# 1. Extraemos el número de semana actual de 2026
hoy_2026 = pd.Timestamp.now().normalize()
semana_actual_2026 = hoy_2026.isocalendar()[1]
semana_anterior_2026 = semana_actual_2026 - 1

# Aseguramos que ambas columnas sean datetime (Naive)
df["fecha_dt"] = pd.to_datetime(df["created_at_y"]).dt.tz_localize(None)
df_air["fecha_dt"] = pd.to_datetime(df_air["fecha_raw"], dayfirst=True, errors='coerce').dt.tz_localize(None)

# 2. Aplicamos el filtro según el botón seleccionado
if st.session_state.periodo == "Semana":
    # Filtramos semana actual (ej. Semana 10 de 2026 vs Semana 10 de 2025)
    df = df[
        (df["fecha_dt"].dt.isocalendar().week == semana_actual_2026) & 
        (df["fecha_dt"].dt.year == 2026)
    ].copy()
    
    df_air = df_air[
        (df_air["fecha_dt"].dt.isocalendar().week == semana_actual_2026) & 
        (df_air["fecha_dt"].dt.year == 2025)
    ].copy()

elif st.session_state.periodo == "Semana Anterior":
    # Filtramos semana anterior (ej. Semana 9 de 2026 vs Semana 9 de 2025)
    df = df[
        (df["fecha_dt"].dt.isocalendar().week == semana_anterior_2026) & 
        (df["fecha_dt"].dt.year == 2026)
    ].copy()
    
    df_air = df_air[
        (df_air["fecha_dt"].dt.isocalendar().week == semana_anterior_2026) & 
        (df_air["fecha_dt"].dt.year == 2025)
    ].copy()

# Si es "Todo", no filtramos nada y se muestran los históricos completos.

# ==============================================================================
# SECCIÓN: MÉTRICAS GENERALES POR OWNER
# ==============================================================================

if df.empty:
    st.warning("No se encontraron datos para los filtros aplicados.")
else:
    member_map = {
        '7f0c4189-764d-453a-8d6b-e416adf7583b': 'Raquel Polgrabia',
        '7f35b25b-4398-4f28-bcf3-1bf59c2b04d4': 'Alejandro Perez',
        '8bd199e1-4aac-485c-b70f-a9b7679286d1': 'Diego Navarro',
        '648bf97f-8d29-4965-ab20-6b4cc63f37ee': 'Carlota L',
        'c8d13743-d7e8-4e9e-b967-3d8e6ac3750e': 'Lorenzo Hurtado de Saracho',
    }
    st.title("General Metrics")

    df["fecha_iso"] = pd.to_datetime(df["created_at_y"], errors='coerce').dt.strftime('%Y-%m-%d')
    target = "2026-02-16"
    df_filtrado = df[df["fecha_iso"] != target].copy() if st.session_state.periodo == "Semana" else df.copy()

    counts = df_filtrado[df_filtrado["status"] != "Not qualified"]["owner"].astype(str).value_counts()
    cols = st.columns(len(member_map))
    for i, (user_id, name) in enumerate(member_map.items()):
        with cols[i]:
            total = counts.get(str(user_id), 0)
            st.metric(label=name, value=int(total))

# ==============================================================================
# SECCIÓN: EVOLUCIÓN TEMPORAL Y DISTRIBUCIÓN POR FUENTE
# ==============================================================================

    if not df["created_at_y"].empty and "reference_3" in df.columns:
        status_map = {"Leads Menorca 2026": "Deal Flow", "Menorca 2026": "Open Call"}
        mapeo_reference = {
            "Mail from Decelera Team": "Marketing", "Social media (LinkedIn, X, Instagram...)": "Marketing",
            "Press": "Marketing", "Google": "Marketing", "Decelera Newsletter": "Marketing",
            "Referral": "Referral", "Investor": "Referral", "Portfolio": "Referral", "Alumni": "Referral", "EM": "Referral",
            "Event": "Outreach", "Contacted by LinkedIn": "Outreach", "Outbound": "Outreach",
            "Inbound": "Marketing", "Decelera Team": "Outreach", "Other": "Otros"
        }

        df["categoria_reference"] = df["reference_3"].map(mapeo_reference).fillna("No Especificado")
        df["stage_limpio"] = df["stage"].astype(str).str.strip()
        df["stage_bonito"] = df["stage_limpio"].map(status_map)
        df["fecha"] = pd.to_datetime(df["created_at_y"], errors="coerce").dt.date

        # --- GRÁFICA DE EVOLUCIÓN (LINE CHART) ---
        def get_traces_for_status(status=None):
            df_f = df[df["stage_bonito"] == status].copy() if status else df.copy()
            df_total = df_f.groupby("fecha").size().reset_index(name="aplicaciones")
            fecha_target = pd.to_datetime("2026-02-16").date()
            if status is None or status == "Deal Flow":
                mask = df_total["fecha"] == fecha_target
                if mask.any(): df_total.loc[mask, "aplicaciones"] = (df_total.loc[mask, "aplicaciones"] - 268).clip(lower=0)
            df_cat = df_f.groupby(["fecha", "categoria_reference"]).size().reset_index(name="count")
            if status is None or status == "Deal Flow":
                mask_cat = (df_cat["fecha"] == fecha_target) & (df_cat["categoria_reference"] == "Otros")
                if mask_cat.any(): df_cat.loc[mask_cat, "count"] = (df_cat.loc[mask_cat, "count"] - 268).clip(lower=0)
            return df_total.sort_values("fecha"), df_cat.sort_values("fecha")

        df_total_all, df_cat_all = get_traces_for_status()
        fig = px.line(df_cat_all, x="fecha", y="count", color="categoria_reference", markers=True, template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe)
        fig.add_trace(go.Scatter(x=df_total_all["fecha"], y=df_total_all["aplicaciones"], name="Total", line=dict(color="#1FD0EF", width=4, dash="dot"), mode="lines+markers"))

        line_buttons = [dict(method="update", label="Todos", args=[{"x": [df_cat_all[df_cat_all["categoria_reference"]==c]["fecha"] for c in df_cat_all["categoria_reference"].unique()] + [df_total_all["fecha"]], "y": [df_cat_all[df_cat_all["categoria_reference"]==c]["count"] for c in df_cat_all["categoria_reference"].unique()] + [df_total_all["aplicaciones"]]}, {"title.text": f'📈 Menorca 2026: Evolución Temporal (Total: {df_total_all["aplicaciones"].sum()})'}])]
        for status in ["Deal Flow", "Open Call"]:
            df_t, df_c = get_traces_for_status(status)
            new_x, new_y = [], []
            for trace in fig.data:
                if trace.name == "Total":
                    new_x.append(df_t["fecha"]); new_y.append(df_t["aplicaciones"])
                else:
                    filtered_cat = df_c[df_c["categoria_reference"] == trace.name]
                    new_x.append(filtered_cat["fecha"]); new_y.append(filtered_cat["count"])
            line_buttons.append(dict(method="update", label=status, args=[{"x": new_x, "y": new_y}, {"title.text": f'📈 Menorca 2026: Evolución Temporal (Total: {df_t["aplicaciones"].sum()})'}]))

        fig.update_layout(updatemenus=[dict(buttons=line_buttons, direction="down", showactive=True, x=1.0, xanchor="right", y=1.2, yanchor="top", bgcolor="white", bordercolor="#bec8d9")], title=f'📈 Menorca 2026: Evolución Temporal (Total: {df_total_all["aplicaciones"].sum()})', hovermode='x unified', xaxis=dict(type='date', tickformat='%d %b'), yaxis=dict(rangemode="tozero"), margin=dict(t=100), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        for trace in fig.data: 
            if trace.name != "Total": trace.visible = "legendonly"

        cols_row1 = st.columns(2)
        with cols_row1[0]: st.plotly_chart(fig, use_container_width=True)

    with cols_row1[1]:
        if not df_air.empty:
            df_air["fecha"] = df_air["fecha_dt"].dt.date
            
            df_air = map_airtable_categories(df_air)
            
            df_total_air = df_air.groupby("fecha").size().reset_index(name="aplicaciones")
            df_cat_air = df_air.groupby(["fecha", "categoria_reference"]).size().reset_index(name="count")
            
            fig_air = px.line(df_cat_air, x="fecha", y="count", color="categoria_reference", 
                            markers=True, template="plotly_white", 
                            color_discrete_sequence=px.colors.qualitative.Safe)
            
            fig_air.add_trace(go.Scatter(x=df_total_air["fecha"], y=df_total_air["aplicaciones"], 
                                        name="Total", line=dict(color="#1FD0EF", width=4, dash="dot"), 
                                        mode="lines+markers"))
            
            fig_air.update_layout(
                title=f'📈 Menorca 2025: Evolución Temporal (Total: {len(df_air)})',
                hovermode='x unified',
                xaxis=dict(type='date', tickformat='%d %b'),
                yaxis=dict(rangemode="tozero"),
                margin=dict(t=100),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            # Ocultar categorías por defecto como en la original
            for trace in fig_air.data: 
                if trace.name != "Total": trace.visible = "legendonly"
                
            st.plotly_chart(fig_air, use_container_width=True)

# ==============================================================================
# SECCIÓN: COMPARATIVA DE FUENTES (ATTIO 2026 VS AIRTABLE 2025)
# ==============================================================================

st.title("🎯 Comparativa de Distribución por Fuente")

color_map = {
    "Marketing": "#1FD0EF",
    "Referral": "#FFB950",
    "Outreach": "#B9C1D4",
    "Otros": "#F2F8FA",
    "No Especificado": "#FAF3DC"
}

# Creamos dos columnas iguales
cols_pie = st.columns(2)

# --- 1. PIE CHART ATTIO (IZQUIERDA) ---
with cols_pie[0]:
    if not df.empty:
        df_counts_all = df["categoria_reference"].value_counts()
        fig_pie = px.pie(
            names=df_counts_all.index, 
            values=df_counts_all.values, 
            title=f'<b>Attio 2026</b> ({st.session_state.periodo})', 
            hole=0.4,
            color=df_counts_all.index,
            color_discrete_map=color_map
        )
        
        # Botones de filtro por Stage (Solo para el de la izquierda)
        pie_buttons = [dict(method="restyle", label="Todos", args=[{"values": [df_counts_all.values], "labels": [df_counts_all.index]}])]
        if "stage_bonito" in df.columns:
            for status in df["stage_bonito"].dropna().unique().tolist():
                df_temp = df[df["stage_bonito"] == status]["categoria_reference"].value_counts()
                pie_buttons.append(dict(method="restyle", label=status, args=[{"values": [df_temp.values], "labels": [df_temp.index]}]))

        fig_pie.update_layout(
            updatemenus=[dict(buttons=pie_buttons, direction="down", showactive=True, x=1.0, xanchor="left", y=1.2, yanchor="top", bgcolor="white", bordercolor="#bec8d9")],
            margin=dict(t=100, b=50, l=40, r=150), 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
        )
        fig_pie.update_traces(
            textinfo='percent+value', 
            textposition='auto', 
            marker=dict(line=dict(color="#000000", width=1)), 
            textfont=dict(color="black", size=14)
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No hay datos de Attio para esta selección.")

# --- 2. PIE CHART AIRTABLE (DERECHA) ---
with cols_pie[1]:
    if not df_air.empty:
        # Aseguramos que las categorías estén mapeadas para Airtable
        df_air = map_airtable_categories(df_air)
        df_counts_air = df_air["categoria_reference"].value_counts()
        
        fig_pie_air = px.pie(
            names=df_counts_air.index, 
            values=df_counts_air.values, 
            title=f'<b>Airtable 2025</b> ({st.session_state.periodo})', 
            hole=0.4,
            color=df_counts_air.index,
            color_discrete_map=color_map
        )
        
        # En el de Airtable no ponemos el menú desplegable para mantenerlo como referencia fija
        fig_pie_air.update_layout(
            margin=dict(t=100, b=50, l=40, r=150), # Exactamente el mismo margen
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1) # Misma posición de leyenda
        )
        fig_pie_air.update_traces(
            textinfo='percent+value', 
            textposition='auto', 
            marker=dict(line=dict(color="#000000", width=1)), 
            textfont=dict(color="black", size=14)
        )
        st.plotly_chart(fig_pie_air, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No hay datos de Airtable para esta selección.")

# ==============================================================================
# SECCIÓN: UBICACIÓN Y SCORING (BARS & DISTPLOT)
# ==============================================================================

col1_loc, col2_gf = st.columns(2)
campo_const = "constitution_company"
if campo_const in df.columns:
    df_const = df.copy(); df_const[campo_const] = df_const[campo_const].fillna("Sin especificar")
    def get_labels_with_pct(counts_series):
        total = counts_series.sum()
        return [f"{v} ({(v/total)*100:.1f}%)" if total > 0 else f"{v}" for v in counts_series]
    df_all_loc = df_const.groupby(campo_const).size().reset_index(name="Cantidad").sort_values("Cantidad", ascending=False)
    text_all_loc = get_labels_with_pct(df_all_loc["Cantidad"])
    fig_const = px.bar(df_all_loc, x=campo_const, y="Cantidad", title="Constitution location", color="Cantidad", color_continuous_scale="Blues", text=text_all_loc, template="plotly_white")
    const_buttons = [dict(method="restyle", label="Todos", args=[{"x": [df_all_loc[campo_const].tolist()], "y": [df_all_loc["Cantidad"].tolist()], "text": [text_all_loc], "marker.color": [df_all_loc["Cantidad"].tolist()]}])]
    for status in df_const["stage_bonito"].dropna().unique().tolist():
        df_filtered = df_const[df_const["stage_bonito"] == status]
        df_temp = df_filtered.groupby(campo_const).size().reset_index(name="Cantidad").sort_values("Cantidad", ascending=False)
        text_temp = get_labels_with_pct(df_temp["Cantidad"])
        const_buttons.append(dict(method="restyle", label=status, args=[{"x": [df_temp[campo_const].tolist()], "y": [df_temp["Cantidad"].tolist()], "text": [text_temp], "marker.color": [df_temp["Cantidad"].tolist()]}]))
    fig_const.update_layout(updatemenus=[dict(buttons=const_buttons, direction="down", showactive=True, x=1.0, xanchor="right", y=1.25, yanchor="top", bgcolor="white", bordercolor="#bec8d9")], margin=dict(t=120, b=50, l=40, r=40), xaxis={'categoryorder':'total descending'}, coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_const.update_traces(textposition='outside', cliponaxis=False, textfont=dict(size=12, color="black"))
    with col1_loc: st.plotly_chart(fig_const, use_container_width=True, config={"displayModeBar": False})

# --- DISTPLOT Y GREEN FLAGS ---
df_score = df[df["form_score"].notna()].copy()
df_score["form_score"] = pd.to_numeric(df_score["form_score"], errors="coerce")
df_score = df_score[df_score["form_score"] > 0]

st.title(f"📈 Form Scoring de las aplicaciones: {len(df_score)} aplicaciones")
c_score1, c_score2 = st.columns(2)

if len(df_score) <= 1 or df_score["form_score"].nunique() <= 1:
    with c_score1: 
        st.warning("No hay suficientes datos variados para generar la curva de densidad.")
        if not df_score.empty: 
            st.plotly_chart(px.histogram(df_score, x="form_score", nbins=20, title="Distribución Simple"), use_container_width=True)
else:
    # 1. Preparar los datos: Total + Desglose por categoría
    hist_data = [df_score["form_score"].tolist()]
    group_labels = ['Total']
    
    # Añadimos los datos de cada categoría que tenga al menos 2 valores (necesario para la curva KDE)
    for cat in df_score["categoria_reference"].unique():
        if pd.isna(cat): continue
        cat_values = df_score[df_score["categoria_reference"] == cat]["form_score"].tolist()
        if len(cat_values) > 1: # Necesitamos al menos 2 puntos para una curva
            hist_data.append(cat_values)
            group_labels.append(cat)

    # 2. Definir colores (usando tu mapa de colores anterior para consistencia)
    # El primer color es el 'Total' (#1FD0EF), los demás siguen el mapa
    colores_dist = ['#1FD0EF'] + [color_map.get(label, "#bdc3c7") for label in group_labels[1:]]

    # 3. Crear el distplot
    fig_dist = ff.create_distplot(
        hist_data, 
        group_labels, 
        show_hist=False, 
        show_rug=False, 
        colors=colores_dist
    )

    # 4. Configurar visibilidad inicial: Solo 'Total' visible, las demás en la leyenda
    for trace in fig_dist.data:
        if trace.name != 'Total':
            trace.visible = 'legendonly'
        else:
            trace.line = dict(width=4) # Hacer la línea total más gruesa

    # 5. Anotaciones y líneas de umbral (referentes al Total)
    total_s = len(df_score)
    n_bajo = len(df_score[df_score["form_score"] < 30])
    n_medio = len(df_score[(df_score["form_score"] >= 30) & (df_score["form_score"] < 65)])
    n_alto = len(df_score[df_score["form_score"] >= 65])
    
    fig_dist.add_vline(x=30, line_dash="dash", line_color="#ef4444", line_width=2)
    fig_dist.add_vline(x=65, line_dash="dash", line_color="#22c55e", line_width=2)
    
    for s in [{"x": 15, "n": n_bajo, "lbl": "Bajo"}, {"x": 47, "n": n_medio, "lbl": "Medio"}, {"x": 82, "n": n_alto, "lbl": "Alto"}]:
        fig_dist.add_annotation(
            x=s["x"], y=0.95, yref="paper", 
            text=f"<b>{s['lbl']}</b><br>{s['n']} deals<br>{(s['n']/total_s)*100:.1f}%", 
            showarrow=False, font=dict(size=12)
        )

    fig_dist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=20, r=20), 
        xaxis=dict(title="Puntuación (Score)", range=[0, 100], dtick=10), 
        yaxis=dict(title="Densidad", showticklabels=False)
    )

    with c_score1: 
        st.plotly_chart(fig_dist, use_container_width=True, key="dist_plot_combined")

# Green Flags
campo_green_flags = 'green_flags_form'
if campo_green_flags in df.columns:
    df_green = df[df[campo_green_flags].str.contains("🟢", na=False)].copy()
    if not df_green.empty:
        all_green = []
        for entry in df_green[campo_green_flags]: all_green.extend(list(set([g.strip() for g in str(entry).split('\n') if "🟢" in g])))
        df_gf_counts = pd.Series(all_green).value_counts().reset_index(); df_gf_counts.columns = ['Green Flag', 'Cantidad']; df_gf_counts['Porcentaje'] = (df_gf_counts['Cantidad'] / len(df_green)) * 100
        fig_gf = px.bar(df_gf_counts, x='Green Flag', y='Cantidad', title=f'🟢 Green Flags: {len(df_green)} compañías', color='Cantidad', color_continuous_scale='Greens', custom_data=[df_gf_counts['Porcentaje']])
        fig_gf.update_traces(texttemplate='%{y}<br>(%{customdata[0]:.1f}%)', textposition='outside', textfont=dict(color='black', size=12), cliponaxis=False)
        fig_gf.update_layout(yaxis=dict(range=[0, df_gf_counts['Cantidad'].max() * 1.3]), xaxis=dict(tickangle=45, automargin=True), margin=dict(t=80, b=120), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
        with c_score2: st.plotly_chart(fig_gf, use_container_width=True)

# ==============================================================================
# SECCIÓN: FUNNEL DE STARTUPS Y OBJETIVOS (CON FILTRO DE STAGE)
# ==============================================================================

st.title("📊 Funnel de Startups por Reference con Objetivos")

# 1. Definimos los mapeos y objetivos
OBJ_CAT = {
    "Marketing": {"Initial screening": 660, "Deep dive": 40, "Pre-committee": 2}, 
    "Referral": {"Initial screening": 150, "Deep dive": 50, "Pre-committee": 5}, 
    "Outreach": {"Initial screening": 325, "Deep dive": 50, "Pre-committee": 5}, 
    "Otros": {"Initial screening": 0, "Deep dive": 0, "Pre-committee": 0}
}
ORDEN_ESTADOS = ["Initial screening", "Deep dive", "Pre-committee"]
colores_fun = {"Marketing": "#1FD0EF", "Referral": "#FFB950", "Outreach": "#22c55e", "Otros": "#bdc3c7"}

# 2. Selector de Stage para el Funnel (Opcional: puedes usar st.session_state si quieres vincularlo a los botones de arriba)
# Si quieres que sea independiente, usamos un radio o selectbox:
filtro_funnel = st.radio("Seleccionar flujo:", ["Todos", "Deal Flow", "Open Call"], horizontal=True, key="funnel_filter_radio")

# 3. Filtrado de datos según la selección
df_funnel_input = df.copy()
if filtro_funnel != "Todos":
    # Usamos el mapeo que definiste antes: "Leads Menorca 2026" -> "Deal Flow", "Menorca 2026" -> "Open Call"
    status_map_inv = {"Deal Flow": "Leads Menorca 2026", "Open Call": "Menorca 2026"}
    df_funnel_input = df_funnel_input[df_funnel_input["stage"] == status_map_inv[filtro_funnel]]

st.markdown(f"Mostrando funnel para: **{filtro_funnel}**", unsafe_allow_html=True)

# 4. LÓGICA DE PROCESAMIENTO DE DATOS (REHECHA PARA PRECISIÓN)
funnel_list = []
# Nos aseguramos de tener las categorías
cats_presentes = [c for c in df_funnel_input["categoria_reference"].unique() if pd.notna(c) and c != "Otros"]

for cat in cats_presentes:
    # EL UNIVERSO TOTAL DE ESTA CATEGORÍA (Lo que sale en el Pie Chart)
    df_cat_universo = df_funnel_input[df_funnel_input["categoria_reference"] == cat]
    total_pie_chart = len(df_cat_universo)
    
    # Pre-committee
    mask_pre = ((df_cat_universo["status"] == "Pre-committee") | (df_cat_universo["reason"].isin(["Pre-comitee", "Pre-committee"])))
    val_pre = len(df_cat_universo[mask_pre])
    
    # Deep dive (acumulado)
    mask_in_play = ((df_cat_universo["status"] == "Deep dive") | (df_cat_universo["reason"] == "Signals (In play)"))
    val_in_play = len(df_cat_universo[mask_in_play]) + val_pre
    
    # Initial screening (acumulado)
    mask_qual = ((df_cat_universo["status"] == "Initial screening") | (df_cat_universo["reason"] == "Signals (Qualified)") | (df_cat_universo["status"] == "First interaction"))
    val_qual = len(df_cat_universo[mask_qual]) + val_in_play
    
    # CÁLCULO DE PORCENTAJES REALES
    # % de calificación: (Calificados / Total en Pie Chart) -> Ej: 30 / 145 = 20.7%
    pct_calif = (val_qual / total_pie_chart * 100) if total_pie_chart > 0 else 0
    
    # % de avance interno
    pct_in_play = (val_in_play / val_qual * 100) if val_qual > 0 else 0
    pct_pre = (val_pre / val_in_play * 100) if val_in_play > 0 else 0
    
    obj_dict = OBJ_CAT.get(cat, {"Initial screening": 0, "Deep dive": 0, "Pre-committee": 0})
    
    funnel_list.append({"Fuente": cat, "Etapa": "Initial screening", "Actual": val_qual, "Objetivo": obj_dict["Initial screening"], "Pct": pct_calif})
    funnel_list.append({"Fuente": cat, "Etapa": "Deep dive", "Actual": val_in_play, "Objetivo": obj_dict["Deep dive"], "Pct": pct_in_play})
    funnel_list.append({"Fuente": cat, "Etapa": "Pre-committee", "Actual": val_pre, "Objetivo": obj_dict["Pre-committee"], "Pct": pct_pre})

df_final_funnel = pd.DataFrame(funnel_list)

# 5. DIBUJAR COLUMNAS
if df_final_funnel.empty:
    st.info("No hay datos para este flujo.")
else:
    totales_col = {etapa: df_final_funnel[df_final_funnel["Etapa"] == etapa]["Actual"].sum() for etapa in ORDEN_ESTADOS}
    cols_funnel = st.columns(3)

    for i, etapa in enumerate(ORDEN_ESTADOS):
        with cols_funnel[i]:
            df_etapa = df_final_funnel[df_final_funnel["Etapa"] == etapa].reset_index(drop=True)
            t_actual = totales_col[etapa]
            
            # Título de columna
            if i == 0:
                texto_titulo = f"<b>{etapa}</b><br><span style='font-size:14px;'>Total Calificadas: {t_actual}</span>"
            else:
                t_previo = totales_col[ORDEN_ESTADOS[i-1]]
                conv = (t_actual / t_previo * 100) if t_previo > 0 else 0
                texto_titulo = f"<b>{etapa}</b><br><span style='font-size:14px;'>Total: {t_actual} ({conv:.1f}% vs ant.)</span>"

            # Etiquetas de las barras: usamos el "Pct" que calculamos en el bucle
            labels = [f"{val}<br><span style='font-size:15px;'>({pct:.1f}%)</span>" for val, pct in zip(df_etapa["Actual"], df_etapa["Pct"])]
            
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Bar(
                x=df_etapa["Fuente"], 
                y=df_etapa["Actual"], 
                marker_color=[colores_fun.get(f, "#bdc3c7") for f in df_etapa["Fuente"]], 
                text=labels, 
                textposition='outside',
                cliponaxis=False
            ))
            
            # Línea de objetivo
            fig_ind.add_trace(go.Scatter(
                x=df_etapa["Fuente"], 
                y=df_etapa["Objetivo"], 
                mode='markers', 
                marker=dict(symbol="line-ew", size=40, line=dict(width=2, color="#555555"))
            ))
            
            fig_ind.update_layout(
                title=texto_titulo, 
                showlegend=False, height=400, margin=dict(l=20, r=20, t=85, b=40),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0, max(df_etapa["Actual"].max(), df_etapa["Objetivo"].max()) * 1.3])
            )
            st.plotly_chart(fig_ind, use_container_width=True, key=f"funnel_final_{i}")

# ==============================================================================
# SECCIÓN: DESGLOSE DE NOT QUALIFIED (RESTURACIÓN DE ESTILO ORIGINAL)
# ==============================================================================

st.title("🚫 Desglose de los 'Not Qualified'")
cols_nq = st.columns(2)
df_not_qual = df[df['status'].str.contains("Not qualified", case=False, na=False) | df["status"].str.contains("Killed", case=False, na=False)].copy()
total_nq = len(df_not_qual)

if total_nq > 0:
    # --- 1. Motivos Not Qualified (Reason) ---
    df_rs = df_not_qual.groupby("reason").size().reset_index(name="conteo")
    df_rs["porcentaje"] = (df_rs["conteo"] / total_nq) * 100
    
    fig_rs = px.bar(df_rs, x="reason", y="conteo", title="Motivos de 'Not Qualified'", 
                    color="conteo", color_continuous_scale="Reds", 
                    custom_data=[df_rs["porcentaje"]])
    
    fig_rs.update_traces(
        texttemplate='%{y}<br>(%{customdata[0]:.1f}%)', 
        textposition='outside', 
        textfont=dict(color='black', size=12), 
        cliponaxis=False # Restaurado
    )
    
    fig_rs.update_layout(
        yaxis=dict(range=[0, df_rs['conteo'].max() * 1.2]), # Espacio para el texto
        xaxis=dict(tickangle=45, automargin=True), 
        margin=dict(t=80, b=120), 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        coloraxis_showscale=False
    )
    
    with cols_nq[0]: 
        st.plotly_chart(fig_rs, use_container_width=True, key=f"nq_reason_final_{total_nq}")

    # --- 2. Red Flags de Tesis ---
    all_rf = []
    for entry in df_not_qual['red_flags_form_7'].dropna(): 
        all_rf.extend(list(set([r.strip() for r in str(entry).split('\n') if "🛑" in r])))
    
    if all_rf:
        df_rf = pd.Series(all_rf).value_counts().reset_index()
        df_rf.columns = ['Motivo', 'Cantidad']
        df_rf['Porcentaje'] = (df_rf['Cantidad'] / total_nq) * 100
        
        fig_rf = px.bar(df_rf, x='Motivo', y='Cantidad', title='🚫 % Red Flags de Tesis', 
                        color='Cantidad', color_continuous_scale='Reds', 
                        custom_data=[df_rf['Porcentaje']])
        
        fig_rf.update_traces(
            texttemplate='%{y}<br>(%{customdata[0]:.1f}%)', # Restaurado
            textposition='outside', 
            textfont=dict(color='black', size=12), 
            cliponaxis=False # Restaurado
        )
        
        fig_rf.update_layout(
            yaxis=dict(range=[0, df_rf['Cantidad'].max() * 1.2]), 
            xaxis=dict(tickangle=45, automargin=True), 
            margin=dict(t=80, b=120), 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            coloraxis_showscale=False
        )
        
        with cols_nq[1]: 
            st.plotly_chart(fig_rf, use_container_width=True, key=f"nq_rf_final_{len(all_rf)}")
    else:
        with cols_nq[1]:
            st.info("No hay Red Flags registradas.")