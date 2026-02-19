import streamlit as st
import pandas as pd
import httpx
import asyncio
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

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

col_title, col_btn = st.columns([0.85, 0.15])

with col_btn:
    if st.button("🔄 Refrescar"):
        get_combined_dataframe.clear()
        st.rerun()

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
            "Decelera Team": "Outreach",
            "Other": "Otros"
        }

        df["categoria_reference"] = df["reference_3"].map(mapeo_reference).dropna()

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
        status_map = {
            "Leads Menorca 2026": "Deal Flow",
            "Menorca 2026": "Open Call"
        }

        # 1. Crear la columna con nombres limpios
        df["stage_bonito"] = df["stage"].map(status_map)
        # Sacamos la lista de la columna 'stage_bonito'
        status_list = df["stage_bonito"].dropna().unique().tolist()

        # 2. Datos iniciales (Todos)
        df_counts_all = df["categoria_reference"].value_counts()

        fig_pie = px.pie(
            names=df_counts_all.index,
            values=df_counts_all.values,
            title='🎯 Distribución por Reference',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        # --- LÓGICA DE BOTONES INTERNOS ---
        buttons = []

        # Botón "Todos"
        buttons.append(dict(
            method="restyle",
            label="Todos",
            args=[{"values": [df_counts_all.values], "labels": [df_counts_all.index]}]
        ))

        # Botones por Status
        for status in status_list:
            # FILTRAMOS por la columna 'stage_bonito' para que coincida con el texto del botón
            df_temp = df[df["stage_bonito"] == status]["categoria_reference"].value_counts()
            
            buttons.append(dict(
                method="restyle",
                label=status,
                args=[{"values": [df_temp.values], "labels": [df_temp.index]}]
            ))

        # 3. Configurar Layout y Menú
        fig_pie.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.0,      # 0.0 es la izquierda del gráfico
                    xanchor="left",
                    y=1.2,      # Un poco más arriba para que no tape el título
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="#bec8d9"
                )
            ],
            margin=dict(t=100, b=50, l=40, r=150) # Aumentamos margen superior (t) para el botón
        )

        # --- Estética ---
        fig_pie.update_traces(
            textinfo='percent+value',
            textposition='auto',
            marker=dict(line=dict(color="#000000", width=1)),
            textfont=dict(color="black", size=14)
        )

        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1
            )
        )

        with cols[1]:
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        # Vamos a hacer unas barras para ver los paises de los que vienen
        col1, col2 = st.columns(2)
        campo_const = "constitution_company"

        if campo_const in  df.columns:
            df_const = df.copy()
            df_const[campo_const] = df_const[campo_const].fillna("Sin especificar")


            # Agrupamos y contamos
            df_const_counts = df_const.groupby(campo_const).size().reset_index(name="Cantidad")
            df_const_counts = df_const_counts.sort_values("Cantidad", ascending=False)

            # Creamos las gráficas
            fig_const = px.bar(
                df_const_counts,
                x=campo_const,
                y="Cantidad",
                color="Cantidad",
                color_continuous_scale="Blues",
                text="Cantidad",
                template="plotly_white",
                labels={
                    campo_const: "Location",
                    "Cantidad": "Number of companies"
                }
            )

            # 3. Estética
            fig_const.update_traces(textposition='outside', cliponaxis=False)
            fig_const.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                xaxis={'categoryorder':'total descending'}
            )

            with col1:
                st.plotly_chart(fig_const, use_container_width=True)
        else:
            st.error(f"No se encontró la columna '{campo_const}' en los datos de Attio.")

        # --- GRÁFICA DE DISTRIBUCIÓN DE FORM SCORE ---
        # --- GRÁFICA DE DISTRIBUCIÓN CONTINUA (KDE) ---
        
        # 1. Limpieza y preparación (igual que antes)
        df_score = df[df["form_score"].notna()].copy()
        df_score["form_score"] = pd.to_numeric(df_score["form_score"], errors="coerce")
        df_score = df_score[df_score["form_score"] > 0]

        st.title(f"📈 Form Scoring de las aplicaciones: {len(df_score)} aplicaciones")
        col1, col2 = st.columns(2)

        if df_score.empty:
            st.warning("No hay suficientes datos para generar la curva de distribución.")
        else:
            # Datos para la curva
            hist_data = [df_score["form_score"]]
            group_labels = ['Form Score']
            
            # 2. Crear distplot con curva de distribución
            # show_hist=False si solo quieres la línea, show_curve=True es la clave
            fig_dist = ff.create_distplot(
                hist_data, group_labels, 
                show_hist=False, # He dejado el histograma de fondo muy suave para dar contexto
                show_rug=False, 
                colors=['#1FD0EF']
            )

            # 3. Cálculos de segmentos para las etiquetas
            total = len(df_score)
            n_bajo = len(df_score[df_score["form_score"] < 30])
            n_medio = len(df_score[(df_score["form_score"] >= 30) & (df_score["form_score"] < 65)])
            n_alto = len(df_score[df_score["form_score"] >= 65])

            # 4. Añadir líneas verticales
            fig_dist.add_vline(x=30, line_dash="dash", line_color="#ef4444", line_width=2)
            fig_dist.add_vline(x=65, line_dash="dash", line_color="#22c55e", line_width=2)

            # 5. Etiquetas de segmentos
            segmentos = [
                {"x": 15, "n": n_bajo, "lbl": "Bajo"},
                {"x": 47, "n": n_medio, "lbl": "Medio"},
                {"x": 82, "n": n_alto, "lbl": "Alto"}
            ]

            for s in segmentos:
                pct = (s["n"] / total) * 100
                fig_dist.add_annotation(
                    x=s["x"], y=0.85, yref="paper",
                    text=f"<b>{s['lbl']}</b><br>{s['n']} deals<br>{pct:.1f}%",
                    showarrow=False, font=dict(size=13)
                )

            # 6. Estética final para que encaje con tu dashboard
            fig_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(t=40, b=40, l=20, r=20),
                xaxis=dict(title="Puntuación", range=[0, 100], dtick=10),
                yaxis=dict(title="Densidad de probabilidad")
            )
            with col1:
                st.plotly_chart(fig_dist, use_container_width=True)

            # VAMOS A DARLE A LAS GREEN FLAGS

            # --- SECCIÓN GREEN FLAGS ---            
            # 1. Filtramos compañías que tengan el campo de green flags (ajusta el nombre del campo)
            # Asumo que el campo se llama 'green_flags_form_7', cámbialo si es necesario
            campo_green_flags = 'green_flags_form' 
            
            if campo_green_flags in df.columns:
                df_green = df[df[campo_green_flags].str.contains("🟢", na=False)].copy()
                total_companias_con_flags = len(df_green)
                
                if total_companias_con_flags > 0:
                    all_green_reasons = []
                    for entry in df_green[campo_green_flags]:
                        # Extraemos líneas que contienen el círculo verde, limpiamos y evitamos duplicados por empresa
                        flags = list(set([g.strip() for g in str(entry).split('\n') if (g.strip() and "🟢" in g)]))
                        all_green_reasons.extend(flags)

                    # 2. Conteo y preparación de DataFrame
                    df_gf_counts = pd.Series(all_green_reasons).value_counts().reset_index()
                    df_gf_counts.columns = ['Green Flag', 'Cantidad']
                    
                    # 3. Calcular % sobre el total de compañías que tienen alguna flag
                    df_gf_counts['Porcentaje'] = (df_gf_counts['Cantidad'] / total_companias_con_flags) * 100

                    # 4. Visualización
                    fig_greenflags = px.bar(
                        df_gf_counts,
                        x='Green Flag',
                        y='Cantidad',
                        title=f'🟢 Prevalencia de Green Flags: {total_companias_con_flags} compañías',
                        color='Cantidad',
                        color_continuous_scale='Greens',
                        custom_data=[df_gf_counts['Porcentaje']]
                    )

                    fig_greenflags.update_traces(
                        texttemplate='%{y}<br>(%{customdata[0]:.1f}%)',
                        textposition='outside',
                        textfont=dict(color='black', size=12),
                        cliponaxis=False
                    )

                    fig_greenflags.update_layout(
                        yaxis=dict(range=[0, df_gf_counts['Cantidad'].max() * 1.3]), # Espacio para etiquetas
                        xaxis=dict(tickangle=45, automargin=True),
                        margin=dict(t=80, b=120),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        coloraxis_showscale=False
                    )

                    with col2:
                        st.plotly_chart(fig_greenflags, use_container_width=True)
                else:
                    st.info("No se han encontrado Green Flags registradas en las compañías.")
            else:
                st.error(f"El campo '{campo_green_flags}' no se encuentra en el DataFrame. Verifica el nombre del atributo en Attio.")

        st.title("📊 Funnel de Startups por Reference con Objetivos")
        st.markdown("Las referencias se han agrupado en función de los objetivos definidos:<br> - Marketing: Mail from Decelera Team, Social media, Press, Google, Decelera Newsletter<br> - Referral: Referral<br> - Outreach: Event, Contacted via LinkedIn", unsafe_allow_html=True)

        # --- CONFIGURACIÓN ---
        OBJETIVOS_POR_CATEGORIA = {
            "Marketing": {"Qualified": 660, "In Play": 40, "Pre-committee": 2},
            "Referral": {"Qualified": 150, "In Play": 50, "Pre-committee": 5},
            "Outreach": {"Qualified": 325, "In Play": 50, "Pre-committee": 5},
            "Otros": {"Qualified": 0, "In Play": 0, "Pre-committee": 0}
        }

        ORDEN_ESTADOS = ["Qualified", "In Play", "Pre-committee"]
        colores = {"Marketing": "#1FD0EF", "Referral": "#FFB950", "Outreach": "#FAF3DC"}

        # --- PROCESAMIENTO DE DATOS (CORREGIDO) ---

        # 1. Agrupamos incluyendo 'reason' para poder filtrar por ella después
        # 1. Trabajamos sobre el dataframe original (df) para evitar perder datos en la agrupación
        funnel_list = []
        categorias_validas = [c for c in df["categoria_reference"].unique() if pd.notna(c) and c != "Otros"]

        for cat in categorias_validas:
            # Filtramos el DF por la categoría actual
            df_cat = df[df["categoria_reference"] == cat]

            # --- Conteo de Pre-committee (Los 4 que mencionas) ---
            # Buscamos en ambas columnas: status o reason
            mask_pre = (
                (df_cat["status"] == "Pre-committee") | 
                (df_cat["reason"] == "Pre-comitee") | 
                (df_cat["reason"] == "Pre-committee") # Por si acaso se corrige el typo
            )
            val_pre = len(df_cat[mask_pre])

            # --- Conteo de In play (Acumulado) ---
            mask_in_play = (
                (df_cat["status"] == "In play") | 
                (df_cat["reason"] == "Signals (In play)")
            )
            val_in_play = len(df_cat[mask_in_play]) + val_pre

            # --- Conteo de Qualified (Acumulado) ---
            # Nota: Aquí sumamos los que están en Qualified puro + los acumulados
            mask_qual = (
                (df_cat["status"] == "Qualified") | 
                (df_cat["reason"] == "Signals (Qualified)") |
                (df_cat["reason"].isna() & (df_cat["status"] == "Qualified"))
            )
            val_qual = len(df_cat[mask_qual]) + val_in_play

            pct_in_play = round(val_in_play / val_qual * 100, 2) if val_qual > 0 else 0
            pct_pre = round(val_pre / val_in_play * 100, 2) if val_in_play > 0 else 0

            # Obtenemos los objetivos
            obj_dict = OBJETIVOS_POR_CATEGORIA.get(cat, {"Qualified": 0, "In Play": 0, "Pre-committee": 0})

            # Guardamos cada etapa en la lista para el gráfico
            funnel_list.append({"Fuente": cat, "Etapa": "Qualified", "Actual": val_qual, "Objetivo": obj_dict.get("Qualified", 0), "Pct": 100})
            funnel_list.append({"Fuente": cat, "Etapa": "In Play", "Actual": val_in_play, "Objetivo": obj_dict.get("In Play", 0), "Pct": pct_in_play})
            funnel_list.append({"Fuente": cat, "Etapa": "Pre-committee", "Actual": val_pre, "Objetivo": obj_dict.get("Pre-committee", 0), "Pct": pct_pre})

        df_final_funnel = pd.DataFrame(funnel_list)

        # --- VISUALIZACIÓN ---

        col_qual, col_play, col_pre = st.columns(3)

        for col, etapa in zip([col_qual, col_play, col_pre], ORDEN_ESTADOS):
            with col:
                df_etapa = df_final_funnel[df_final_funnel["Etapa"] == etapa].reset_index(drop=True)

                #creamos etiquetas personalizadas
                if etapa == "Qualified":
                    labels = [f"{val}" for val in df_etapa["Actual"]]
                else:
                    labels = [f"{val}<br><span style='font-size:15px;'>({pct:.1f}%)</span>"
                            for val, pct in zip(df_etapa["Actual"], df_etapa["Pct"])]
                
                fig_individual = go.Figure()

                # Barra de Actual
                fig_individual.add_trace(go.Bar(
                    x=df_etapa["Fuente"],
                    y=df_etapa["Actual"],
                    marker_color=[colores.get(f, "#bdc3c7") for f in df_etapa["Fuente"]],
                    text=labels,
                    textposition='outside',
                    cliponaxis=False,
                    name="Actual",
                    textfont=dict(color='black')
                ))

                # Marcador de Objetivo
                fig_individual.add_trace(go.Scatter(
                    x=df_etapa["Fuente"],
                    y=df_etapa["Objetivo"],
                    mode='markers',
                    marker=dict(
                        symbol="line-ew", 
                        size=40, 
                        line=dict(width=2, color="#555555")
                    ),
                    hoverinfo="text",
                    text=[f"Meta: {obj}" for obj in df_etapa["Objetivo"]]
                ))

                fig_individual.update_layout(
                    title=f"<b>{etapa}</b>",
                    showlegend=False,
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(tickfont=dict(color='black'), linecolor='#d0d0d0'),
                    yaxis=dict(tickfont=dict(color='black'), gridcolor='#f0f0f0')
                )

                st.plotly_chart(fig_individual, use_container_width=True)

    
        # VAMOS CON LOS NOT QUALIFIED
        st.title("🚫 Desglose de los 'Not Qualified'")
        col1, col2 = st.columns(2)

        # 1. Filtramos las empresas "Not Qualified"
        df_not_qual = df[df['status'].str.contains("Not qualified", case=False, na=False)].copy()
        total_empresas_not_qual = len(df_not_qual) # Este es nuestro nuevo denominador

        # 2. Extraemos los motivos asegurándonos de no contar dos veces el mismo motivo por empresa
        all_reasons = []
        for entry in df_not_qual['red_flags_form_7'].dropna():
            # Usamos set() para que si una empresa tiene escrito dos veces lo mismo, solo cuente una vez
            reasons = list(set([r.strip() for r in str(entry).split('\n') if (r.strip() and "🛑" in r)]))
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
            title='🚫 % de Empresas por cada Red Flag de Tesis',
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

        with col2:
            st.plotly_chart(fig_redflags, use_container_width=True)

        df_reasons_not_qual = df_not_qual.groupby("reason").size().reset_index()
        df_reasons_not_qual.columns = ["reason", "conteo"]

        df_reasons_not_qual["porcentaje"] = (df_reasons_not_qual["conteo"] / total_empresas_not_qual) * 100

        fig_not_qual = px.bar(
            df_reasons_not_qual,
            x="reason",
            y="conteo",
            title="Motivos de 'Not Qualified'",
            color="conteo",
            color_continuous_scale="Reds",
            custom_data=[df_reasons_not_qual["porcentaje"]]
        )

        fig_not_qual.update_traces(
            # %{y} es el número de empresas, %{customdata[0]} es el % sobre el total de empresas
            texttemplate='%{y}<br>(%{customdata[0]:.1f}%)',
            textposition='outside',
            textfont=dict(color='black', size=12),
            cliponaxis=False
        )

        fig_not_qual.update_layout(
            yaxis=dict(range=[0, df_reasons_not_qual['conteo'].max() * 1.2]), # Espacio para el texto
            xaxis=dict(tickangle=45, automargin=True),
            margin=dict(t=80, b=120),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )

        with col1:
            st.plotly_chart(fig_not_qual, use_container_width=True)