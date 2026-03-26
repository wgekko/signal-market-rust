import streamlit as st
import datetime
import sys
import os

# Asegurar que el path sea el correcto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils import load_data_stooq, apply_rust_analysis, plot_advanced_chart




# Configuración de la página
st.set_page_config(page_title="support-resistance-flag", page_icon=":material/analytics:", layout="wide")

# Asegurar que el path sea el correcto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.header(":material/finance: Análisis de detección soportes, resistencia y flag")

# --- BARRA LATERAL PARA PARÁMETROS ---
with st.sidebar:
    st.subheader(":material/settings_alert: Configuración")
    
    ticker_input = st.text_input("Ticker de la acción", value="AMD").strip()
    
    today = datetime.date.today()
    one_year_ago = today - datetime.timedelta(days=365)
    
    start_date = st.date_input("Fecha de inicio", value=one_year_ago)
    end_date = st.date_input("Fecha de fin", value=today)
    
    lookback_val = st.slider("Ventana de análisis (Lookback)", 10, 100, 40)
    
    # En Streamlit moderno, use_container_width reemplazó a width='stretch'
    btn_analizar = st.button("Ejecutar Análisis", width='stretch')

# --- CUERPO PRINCIPAL ---
if btn_analizar:
    if not ticker_input:
        st.error("Por favor, ingresa un código de acción.")
    elif start_date > end_date:
        st.error("La fecha de inicio no puede ser posterior a la de fin.")
    else:
        with st.spinner(f"Obteniendo datos de {ticker_input}.US desde Stooq..."):
            # Pasamos las fechas seleccionadas a la función de carga
            df = load_data_stooq(
                ticker_input, 
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d')
            )
            
            if not df.empty:
                # 1. Procesamiento con el motor de Rust
                df = apply_rust_analysis(df, lookback=lookback_val)
                
                # 2. Resumen de métricas
                st.subheader(f"Resultados para {ticker_input.upper()}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Precio Actual", f"${df['Close'].iloc[-1]:.2f}")
                m2.metric("Inicio Periodo", start_date.strftime('%d/%m/%Y'))
                m3.metric("Fin Periodo", end_date.strftime('%d/%m/%Y'))
                
                # 3. Gráfico avanzado (Plotly)
                fig = plot_advanced_chart(df, ticker_input)
                st.plotly_chart(fig, width='stretch')
                
                # 4. Tabla de datos crudos
                with st.expander("Ver tabla de datos detallada"):
                    st.dataframe(df.sort_index(ascending=False), width='stretch' ) #use_container_width=True
            else:
                st.error(f"No se encontraron datos para '{ticker_input}'. Asegúrate de que el código sea correcto.")