import streamlit as st
import pandas as pd
import httpx
import asyncio
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Attio Deal Dashboard", layout="wide")

# CSS para los contenedores
st.markdown("""
    <style>
    /* Buscamos el contenedor nativo de Streamlit y le damos tu estilo */
    [data-testid="stVerticalBlock"] > div:has(div.stPlotlyChart) {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #efefef;
        margin-bottom: 20px;
    }
    /* Ponemos el fondo de la web gris claro para que resalten las tarjetas */
    .stApp {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

#fondo
st.markdown('<style>[data-testid="stAppViewContainer"]{background-color: #f9f9fb;}</style>', unsafe_allow_html=True)

# --- CONSTANTES ---
ATTIO_API_KEY = st.secrets["ATTIO_API_KEY"]
DEALS_ID = "dbcd94bf-ec33-4f00-a7c8-74f57a559869"
DEAL_FLOW_ID = "54265eb6-d53d-465d-ad35-4e823e135629"
BASE_URL = "https://api.attio.com/v2"
HEADERS = {
    "Authorization": f"Bearer {ATTIO_API_KEY}",
    "Content-Type": "application/json"
}

# --- FUNCIONES DE EXTRACCIÓN (Tus funciones optimizadas) ---
def extract_value(attr_list):
    """Extrae el dato real de los valores de atributo de Attio."""
    if not attr_list:
        return None

    extracted = []
    for item in attr_list:
        attr_type = item.get("attribute_type", "")
        val = None

        if attr_type == "status":
            val = item.get("status", {}).get("title")
        elif attr_type == "select":
            val = item.get("option", {}).get("title")
        elif attr_type == "domain":
            val = item.get("domain")
        elif attr_type == "location":
            val = ", ".join(filter(None, [
                item.get("line_1"), item.get("locality"),
                item.get("region"), item.get("postcode"),
                item.get("country_code"),
            ]))
        elif attr_type == "personal-name":
            val = item.get("full_name")
        elif attr_type == "email-address":
            val = item.get("email_address")
        elif attr_type == "phone-number":
            val = item.get("phone_number")
        elif attr_type == "record-reference":
            val = item.get("target_record_id")
        elif attr_type == "actor-reference":
            val = item.get("referenced_actor_id")
        elif attr_type == "interaction":
            val = item.get("interacted_at")
        elif attr_type == "currency":
            val = item.get("currency_value")
        elif attr_type in ("text", "number", "date", "timestamp", "checkbox", "rating"):
            val = item.get("value")
        else:
            val = item.get("value")

        if val is not None:
            extracted.append(str(val))

    return extracted[0] if len(extracted) == 1 else extracted or None

async def fetch_data(client, url, payload=None):
    """Función genérica para manejar la paginación de forma eficiente."""
    all_data = []
    limit = 100 # Aumentamos a 100 para menos saltos de red
    offset = 0
    
    while True:
        current_payload = payload.copy() if payload else {}
        current_payload.update({"limit": limit, "offset": offset})
        
        response = await client.post(url, headers=HEADERS, json=current_payload)
        response.raise_for_status()
        data = response.json().get("data", [])
        all_data.extend(data)
        
        if len(data) < limit:
            break
        offset += limit
    return all_data

def transform_attio_to_df(attio_data):
    rows = []
    for record in attio_data:
        record_id = (
            record.get("id", {}).get("record_id") or 
            record.get("parent_record_id")
        )
        row = {
            "record_id": record_id,
            "created_at": record.get("created_at")
        }
        values_source = record.get("entry_values", {}) or record.get("values", {})
        for attr_name, attr_list in values_source.items():
            row[attr_name] = extract_value(attr_list)
        rows.append(row)
    return pd.DataFrame(rows)

# --- NÚCLEO DEL DASHBOARD (Caché y Paralelismo) ---
@st.cache_data(ttl=600) # El dashboard será instantáneo durante 10 min
def get_combined_dataframe():
    async def run_parallel_fetches():
        async with httpx.AsyncClient() as client:
            # Lanzamos ambas peticiones al mismo tiempo
            records_task = fetch_data(client, f"{BASE_URL}/objects/{DEALS_ID}/records/query", 
                                    payload={
                                        "$or": [
                                            {"stage": "Menorca 2026"},
                                            {"stage": "Leads Menorca 2026"}
                                        ]
                                    })
            entries_task = fetch_data(client, f"{BASE_URL}/lists/{DEAL_FLOW_ID}/entries/query")
            
            return await asyncio.gather(records_task, entries_task)

    # Ejecutar el loop asíncrono
    raw_records, raw_entries = asyncio.run(run_parallel_fetches())
    
    df_rec = transform_attio_to_df(raw_records)
    df_ent = transform_attio_to_df(raw_entries)
    
    # Merge final
    if df_rec.empty or df_ent.empty:
        return pd.DataFrame()
        
    return pd.merge(df_rec, df_ent, on="record_id")

# --- INTERFAZ STREAMLIT ---
st.markdown("""
<style>
.outer-container {
    display: flex;
    justify-content: center;
    width: 100%;
}
.container {
    display: flex;
    align-items: center;
}
.logo-img {
    width: 80px;
    height: 80px;
    margin-right: 20px;
}
.title-text {
    font-size: 2.5em;
    font-weight: bold;
}
</style>
<div class="outer-container">
<div class="container">
    <img class="logo-img" src="https://images.squarespace-cdn.com/content/v1/67811e8fe702fd5553c65249/c5500619-9712-4b9b-83ee-a697212735ae/Disen%CC%83o+sin+ti%CC%81tulo+%2840%29.png">
    <h1 class="title-text">Deal Flow - Funnel<br>Menorca 2026</h1>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

with st.spinner("Sincronizando con Attio..."):
    df = get_combined_dataframe()

if df.empty:
    st.warning("No se encontraron datos para los filtros aplicados.")
else:
    # KPIs rápidos
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Deals", len(df))
    col2.metric("Not Qualified", f"{len(df[df["status"] == "Not qualified"])} ({round(len(df[df["status"] == "Not qualified"])/len(df)*100, 2)} %)")
    col3.metric("Qualified", f"{len(df[df["status"] == "Qualified"])} ({round(len(df[df["status"]=="Qualified"])/len(df)*100, 2)} %)")
    col4.metric("In Play", f"{len(df[df["status"]=="In play"])} ({round(len(df[df["status"]=="In play"])/len(df)*100, 2)} %)")
    
    if not df["created_at_y"].empty and "created_at_y" in df.columns and not df["reference_3"].empty and "reference_3" in df.columns:
        cols = st.columns(2)

        

        # Vamos a poner por cada fuente
        mapeo_reference = {
            "Mail from Decelera Team": "Marketing",
            "Social media (LinkedIn, X, Instagram...)": "Marketing",
            "Press": "Marketing",
            "Google": "Marketing",
            "Decelera Newsletter": "Marketing",
            "Referral": "Referral",
            "Investor": "Referral",
            "Portfolio": "Referral",
            "Alumni": "Referral",
            "EM": "Referral",
            "Event": "Outreach",
            "Contacted by LinkedIn": "Outreach",
            "Outbound": "Outreach",
            "Inbound": "Marketing",
            "Decelera Team": "Outreach"
        }

        df["categoria_reference"] = df["reference_3"].map(mapeo_reference).fillna("Otros")

        # VAMOS A HACER UNA GRAFICA DE APLICACIONES POR DIA ==========================
        df_apps = df[df["stage"] == "Menorca 2026"]
        df_apps["fecha"] = pd.to_datetime(df["created_at_y"], errors="coerce").dt.date

        #agrupamos por fecha y contamos
        df_counts_date = df_apps.groupby("fecha").size().reset_index(name="aplicaciones")
        df_counts_date = df_counts_date.sort_values("fecha")

        df_apps["categoria_reference"] = df_apps["reference_3"].map(mapeo_reference).fillna("Otros")
        df_categoria_date = df_apps.groupby(["fecha", "categoria_reference"]).size().reset_index(name="reference_per_day")

        df_categoria_date = df_categoria_date.sort_values("fecha")
        
        #creamos la grafica con plotly
        fig = px.line(
            df_categoria_date,
            x="fecha",
            y="reference_per_day",
            title="Applications received per day",
            color="categoria_reference",
            markers=True,
            labels={"fecha": "Date", "reference_per_day": "Number of Applications"},
            template="plotly_white"
        )

        fig.add_trace(
            go.Scatter(
                x=df_counts_date["fecha"],
                y=df_counts_date["aplicaciones"],
                name="Total",
                line=dict(color="#1FD0EF", width=4, dash="dot"),
                mode="lines+markers"
            )
        )

        fig.update_layout(
            title='📈 Evolución de Aplicaciones: Reference vs Total',
            hovermode='x unified',
            xaxis=dict(
                type='date',
                dtick=86400000.0,
                tickmode='linear',
                tickformat='%d %b'
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', # Fondo del lienzo transparente
            plot_bgcolor='rgba(0,0,0,0)'   # Fondo de la gráfica transparente
        )

        fig.for_each_trace(lambda trace: 
            # 2. Si el nombre no es "Total", la ocultamos en la leyenda por defecto
            trace.update(visible="legendonly") if trace.name != "Total" else ()
        )

        with cols[0]:
            st.plotly_chart(fig, use_container_width=True)

        # --- PIE CHART DE CATEGORIAS ---
        df_pie_reference = df["categoria_reference"].value_counts().reset_index()
        df_pie_reference.columns = ["Referencia", "Total"]

        fig_pie = px.pie(
            df_pie_reference,
            values="Total",
            names="Referencia",
            title='🎯 Distribución Total por Reference',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig_pie.update_traces(
            textinfo='percent+value',
            textposition='auto',
            marker=dict(line=dict(color="#000000", width=1)),
            textfont=dict(
                color="black",
                size=14
            )
        )

        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig_pie.update_layout(
            legend=dict(
                orientation="v",      # Vertical
                yanchor="middle",     # Centrada verticalmente respecto al donut
                y=0.5,                
                xanchor="left",       # El anclaje empieza a la izquierda de la leyenda
                x=1.15                # <--- "El pelín lejos": 1.0 es el borde, 1.15 da el espacio
            ),
            # Es VITAL aumentar el margen derecho (r) para que la leyenda "quepa" en la tarjeta
            margin=dict(t=50, b=100, l=40, r=150) 
        )

        with cols[1]:
            st.plotly_chart(fig_pie, use_container_width=True, config={"responsive": True, "displayModeBar": False})

        st.title("📊Funnel de Aplicaciones por Reference con Objetivos")
        st.markdown("Las referencias se han agrupado en función de los objetivos definidos:<br> - Marketing: Mail from Decelera Team, Social media, Press, Google, Decelera Newsletter<br> - Referral: Referral<br> - Outreach: Event, Contacted via LinkedIn", unsafe_allow_html=True)

        # --- VAMOS CON LAS CATEGORIAS, CONVERSION Y OBJETIVOS
        OBJETIVOS_POR_CATEGORIA = {
            "Marketing": {"Qualified": 660, "In Play": 40, "Pre-committee": 2},
            "Referral": {"Qualified": 150, "In Play": 50, "Pre-committee": 5},
            "Outreach": {"Qualified": 325, "In Play": 50, "Pre-committee": 5},
            "Otros": {"Qualified": 0, "In Play": 0, "Pre-committee": 0}
        }

        ORDEN_ESTADOS = ["Qualified", "In Play", "Pre-committee"]

        df_status = df.groupby(["categoria_reference", "status"]).size().reset_index(name="conteo")

        funnel_list = []
        for cat in df["categoria_reference"].unique():
            temp = df_status[df_status["categoria_reference"] == cat]

            #Valores reales acumulados hacia atras
            val_pre = temp[temp["status"] == "Pre-committee"]["conteo"].sum()
            val_in_play = temp[temp["status"] == "In play"]["conteo"].sum() + val_pre
            val_qual = temp[temp["status"] == "Qualified"]["conteo"].sum() + val_in_play

            #Obtenemos los objetivos
            obj_dict = OBJETIVOS_POR_CATEGORIA.get(cat)

            for etapa, valor in zip(ORDEN_ESTADOS, [val_qual, val_in_play, val_pre]):
                funnel_list.append({
                    "Fuente": cat,
                    "Etapa": etapa,
                    "Actual": valor,
                    "Objetivo": obj_dict.get(etapa, 0)
                })
        
        df_final_funnel = pd.DataFrame(funnel_list)
        df_final_funnel = df_final_funnel[df_final_funnel["Fuente"] != "Otros"]

        # VAMOS CON LA GRAFICA DE BARRAS

        col_qual, col_play, col_pre = st.columns(3)

        # 2. Definimos los colores corporativos
        colores = {"Marketing": "#1FD0EF", "Referral": "#FFB950", "Outreach": "#FAF3DC"}

        # 3. Iteramos por cada estado para crear su gráfica individual
        for col, etapa in zip([col_qual, col_play, col_pre], ORDEN_ESTADOS):
            with col:
                # Filtramos los datos solo para esta etapa
                df_etapa = df_final_funnel[df_final_funnel["Etapa"] == etapa]
                
                fig_individual = go.Figure()

                # Añadimos la barra de datos reales
                fig_individual.add_trace(go.Bar(
                    x=df_etapa["Fuente"],
                    y=df_etapa["Actual"],
                    marker_color=[colores.get(f, "#bdc3c7") for f in df_etapa["Fuente"]],
                    text=df_etapa["Actual"],
                    textposition='outside',
                    name="Actual",
                    textfont=dict(color='black')
                ))

                # Añadimos las metas como marcadores (al ser barras simples, ahora sí se alinean solas)
                fig_individual.add_trace(go.Scatter(
                    x=df_etapa["Fuente"],
                    y=df_etapa["Objetivo"],
                    mode='markers',
                    marker=dict(
                        symbol="line-ew", 
                        size=40, 
                        line=dict(
                            width=2,         # Más fina (antes era 4)
                            color="#555555"  # Gris oscuro en vez de negro puro
                        )
                    ),
                    hoverinfo="text",
                    text=[f"Meta: {obj}" for obj in df_etapa["Objetivo"]]
                ))

                # Ajustes de diseño para que quepan bien en columnas pequeñas
                fig_individual.update_layout(
                    title=f"<b>{etapa}</b>",
                    showlegend=False,
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    # --- Configuración de Ejes ---
                    xaxis=dict(
                        tickfont=dict(color='black'), # Eje X en Negro
                        linecolor='#d0d0d0'
                    ),
                    yaxis=dict(
                        tickfont=dict(color='black'), # Eje Y también en Negro
                        gridcolor='#f0f0f0'            # Cuadrícula muy suave
                    )
                )

                # Envolvemos cada una en tu contenedor curvado
                st.plotly_chart(fig_individual, use_container_width=True)

    
        # VAMOS CON LOS NOT QUALIFIED
        st.title("🚫 Desglose de los 'Not Qualified'")

        # 1. Filtramos las empresas "Not Qualified"
        df_not_qual = df[df['status'].str.contains("Not qualified", case=False, na=False)].copy()
        total_empresas_not_qual = len(df_not_qual) # Este es nuestro nuevo denominador

        # 2. Extraemos los motivos asegurándonos de no contar dos veces el mismo motivo por empresa
        all_reasons = []
        for entry in df_not_qual['red_flags_form_7'].dropna():
            # Usamos set() para que si una empresa tiene escrito dos veces lo mismo, solo cuente una vez
            reasons = list(set([r.strip() for r in str(entry).split('\n') if r.strip()]))
            all_reasons.extend(reasons)

        # 3. Creamos el conteo
        df_reasons = pd.Series(all_reasons).value_counts().reset_index()
        df_reasons.columns = ['Motivo', 'Cantidad']

        # 4. Calculamos el porcentaje sobre el TOTAL DE EMPRESAS
        df_reasons['Porcentaje'] = (df_reasons['Cantidad'] / total_empresas_not_qual) * 100

        # 5. Gráfica con las etiquetas corregidas
        fig_redflags = px.bar(
            df_reasons,
            x='Motivo',
            y='Cantidad',
            title='🚫 % de Empresas por cada Red Flag',
            color='Cantidad',
            color_continuous_scale='Reds',
            # Pasamos el porcentaje calculado correctamente
            custom_data=[df_reasons['Porcentaje']]
        )

        fig_redflags.update_traces(
            # %{y} es el número de empresas, %{customdata[0]} es el % sobre el total de empresas
            texttemplate='%{y}<br>(%{customdata[0]:.1f}%)',
            textposition='outside',
            textfont=dict(color='black', size=12),
            cliponaxis=False
        )

        fig_redflags.update_layout(
            yaxis=dict(range=[0, df_reasons['Cantidad'].max() * 1.2]), # Espacio para el texto
            xaxis=dict(tickangle=45, automargin=True),
            margin=dict(t=80, b=120),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )

        st.plotly_chart(fig_redflags, use_container_width=True)
