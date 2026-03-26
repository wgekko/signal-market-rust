import streamlit as st
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import signal_market_rust
import sys
import os

st.set_page_config(page_title="Monte Carlo Simulation", page_icon=":material/chart_data:", layout="wide")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.subheader(":material/chart_data: Simulación de Monte Carlo")
st.markdown("""
Esta herramienta genera miles de trayectorias aleatorias basadas en la **volatilidad histórica** y el **retorno esperado** para visualizar el rango de precios probable en el futuro.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.subheader(":material/settings_alert: Parámetros de Simulación")
    ticker = st.text_input("Ticker", "TSLA").upper()
    ticker_query = f"{ticker}.US" if not ticker.endswith(".US") else ticker
    
    simulations = st.select_slider("Número de simulaciones", options=[100, 1000, 5000, 10000], value=1000)
    days_to_sim = st.slider("Días a proyectar", 5, 252, 30)
    
    btn_run = st.button("Lanzar Simulaciones", use_container_width=True)

if btn_run:
    try:
        with st.spinner("Analizando volatilidad histórica..."):
            # Obtener 1 año de datos para calcular mu y sigma
            start = pd.Timestamp.today() - pd.DateOffset(years=1)
            df = web.DataReader(ticker_query, "stooq", start=start).sort_index()
            
            # Calcular retornos logarítmicos diarios
            log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            mu = log_returns.mean()
            sigma = log_returns.std()
            last_price = df['Close'].iloc[-1]

        # 1. LLAMADA A RUST (Ultra rápido)
        with st.spinner(f"Rust generando {simulations} trayectorias..."):
            paths = signal_market_rust.monte_carlo_simulation(
                last_price, mu, sigma, days_to_sim, simulations
            )
            paths_array = np.array(paths)

        # 2. PROCESAMIENTO DE RESULTADOS
        # Calculamos percentiles para las bandas de probabilidad
        final_prices = paths_array[:, -1]
        p10 = np.percentile(final_prices, 10)
        p50 = np.percentile(final_prices, 50) # Mediana
        p90 = np.percentile(final_prices, 90)

        # 3. VISUALIZACIÓN
        fig = go.Figure()

        # Dibujamos una muestra de 100 trayectorias para no saturar el navegador
        future_index = [df.index[-1] + pd.Timedelta(days=i) for i in range(days_to_sim + 1)]
        
        for i in range(min(100, simulations)):
            fig.add_trace(go.Scatter(
                x=future_index, y=paths_array[i], 
                mode='lines', line=dict(width=1), 
                opacity=0.1, showlegend=False, hoverinfo='skip'
            ))

        # Dibujar la mediana y los percentiles clave
        fig.add_trace(go.Scatter(x=future_index, y=paths_array.mean(axis=0), name="Promedio Esperado", line=dict(color='yellow', width=3)))
        
        fig.update_layout(
            title=f"Simulación de {simulations} escenarios para {ticker}",
            xaxis_title="Fecha Proyectada",
            yaxis_title="Precio USD",
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig, width='stretch' ) #use_container_width=True

        # 4. MÉTRICAS DE RIESGO
        st.subheader(":material/area_chart: Análisis de Probabilidad a Final de Periodo")
        c1, c2, c3 = st.columns(3)
        c1.metric("Escenario Optimista (P90)", f"${p90:.2f}")
        c2.metric("Escenario Mediano (P50)", f"${p50:.2f}")
        c3.metric("Escenario Pesimista (P10)", f"${p10:.2f}")
        
        st.info(f"Según la simulación, hay un 80% de probabilidad de que {ticker} termine entre ${p10:.2f} y ${p90:.2f} en {days_to_sim} días.")

    except Exception as e:
        st.error(f"Error en la simulación: {e}")