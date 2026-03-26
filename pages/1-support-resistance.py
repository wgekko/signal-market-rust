import streamlit as st
import datetime
import pandas as pd
from utils.utils1 import load_data_stooq, get_key_levels, plot_key_levels_chart

st.set_page_config(page_title="Key Levels Detector", page_icon=":material/analytics:",layout="wide")

st.header("Detector de Soportes y Resistencias")
st.markdown("Basado en la frecuencia de pivots y agrupación por histogramas.")

# --- SIDEBAR ---
with st.sidebar:
    st.header(":material/settings_alert: Configuración")
    ticker = st.text_input("Ticker", value="NVDA")
    
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Inicio", datetime.date.today() - datetime.timedelta(days=365))
    with col2:
        end = st.date_input("Fin", datetime.date.today())
        
    sensitivity = st.slider("Sensibilidad (Bin Width %)", 0.1, 1.0, 0.3, step=0.1)
    btn = st.button("Calcular Niveles", width='stretch') #use_container_width=True

# --- PROCESAMIENTO ---
if btn:
    with st.spinner("Descargando datos de Stooq..."):
        df = load_data_stooq(ticker, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        
    if not df.empty:
        # Convertir sensibilidad de % a valor nominal
        bw = df['Close'].mean() * (sensitivity / 100)
        
        # Obtener niveles
        levels = get_key_levels(df, bin_width=bw)
        
        # Mostrar Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio Actual", f"${df['Close'].iloc[-1]:.2f}")
        c2.metric("Niveles Detectados", len(levels))
        c3.metric("Rango Analizado", f"{len(df)} velas")
        
        # Gráfico
        fig = plot_advanced_chart = plot_key_levels_chart(df, ticker, levels)
        st.plotly_chart(fig, width='stretch' ) #use_container_width=True
        
        # Tabla de niveles
        with st.expander("Ver lista de precios clave"):
            st.write(pd.DataFrame(levels, columns=["Precio"]).sort_values("Precio", ascending=False))
    else:
        st.error("No se pudo obtener información para ese Ticker.")