import streamlit as st
from utils.utils import load_data_stooq, apply_rust_analysis
import sys
import os

# Esto le dice a Python que busque archivos en la carpeta donde está main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from utils.utils import load_data_stooq, apply_rust_analysis

# Configuración inicial de la página
st.set_page_config(page_title="App triang Ascendentes", page_icon=":material/analytics:",  layout="wide")


st.header(":material/finance: Análisis de de detección de triángulos ascendentes")

ticker_input = st.text_input("Ingrese Ticker (ej: TSLA, AAPL, AMD)", "AMD")

# main.py (fragmento final)
if st.button("Analizar"):
    df = load_data_stooq(ticker_input, "2023-01-01", "2026-03-24")
    
    if not df.empty:
        # 1. Procesar con Rust
        df = apply_rust_analysis(df)
        
        st.success(f"ANALIZADOS {len(df)} DIAS DE  {ticker_input.upper()}")
        
        # 2. Mostrar métricas rápidas
        col1, col2 = st.columns(2)
        last_slope_h = df['slope_high'].iloc[-1]
        last_slope_l = df['slope_low'].iloc[-1]
        col1.metric("Pendiente Superior", f"{last_slope_h:.4f}")
        col2.metric("Pendiente Inferior", f"{last_slope_l:.4f}")
        
        # 3. Mostrar el nuevo gráfico avanzado
        from utils.utils import plot_advanced_chart
        fig = plot_advanced_chart(df, ticker_input)
        st.plotly_chart(fig, width='stretch' )#use_container_width=True
        
        # 4. Tabla de datos
        st.subheader("Últimos Registros")
        st.dataframe(df.tail(10))
    else:
        st.error("No se encontraron datos en Stooq para este ticker.")