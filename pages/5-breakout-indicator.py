import streamlit as st
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import signal_market_rust
import sys
import os

st.set_page_config(page_title="Breakout Indicator", page_icon=":material/chart_data:", layout="wide")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.title(":material/chart_data: Detector de Rupturas de Canal (Breakout)")
st.markdown("Esta página identifica canales de consolidación y detecta el momento exacto de la ruptura (Breakout).")

# --- SIDEBAR ---
with st.sidebar:
    st.subheader(":material/settings_alert: Parámetros del Canal")
    ticker = st.text_input("Ticker", "NVDA").upper()
    ticker_query = f"{ticker}.US" if not ticker.endswith(".US") else ticker
    
    backcandles = st.slider("Velas de formación de canal", 10, 50, 20)
    btn_calc = st.button("Escanear Rupturas", width='stretch' ) #use_container_width=True

if btn_calc:
    try:
        with st.spinner("Obteniendo datos..."):
            df = web.DataReader(ticker_query, "stooq").sort_index()
            # Limitar a los últimos 300 días para claridad visual
            df = df.tail(300)

        # 1. LLAMADA A RUST: Procesamiento de canales
        with st.spinner("Rust calculando regresiones de canal..."):
            h = df['High'].tolist()
            l = df['Low'].tolist()
            c = df['Close'].tolist()
            
            sl, il, sh, ih = signal_market_rust.detect_channel_breakout(h, l, c, backcandles)
            
            df['slope_low'] = sl
            df['interc_low'] = il
            df['slope_high'] = sh
            df['interc_high'] = ih

        # 2. Identificar puntos de ruptura
        # Una ruptura ocurre cuando el Close cruza la proyección de la regresión
        df['breakout'] = 0
        for i in range(backcandles, len(df)):
            # Proyección de la línea de tendencia al punto actual (x = backcandles)
            current_resistence = df['slope_high'].iloc[i] * backcandles + df['interc_high'].iloc[i]
            current_support = df['slope_low'].iloc[i] * backcandles + df['interc_low'].iloc[i]
            
            if df['Close'].iloc[i] > current_resistence:
                df.at[df.index[i], 'breakout'] = 1 # Ruptura Alcista
            elif df['Close'].iloc[i] < current_support:
                df.at[df.index[i], 'breakout'] = 2 # Ruptura Bajista

        # 3. Gráfico Interactiva
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"
        )])

        # Añadir flechas de ruptura
        bullish = df[df['breakout'] == 1]
        bearish = df[df['breakout'] == 2]

        fig.add_trace(go.Scatter(
            x=bullish.index, y=bullish['Low'] * 0.98, mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='lime'), name='Breakout Alcista'
        ))

        fig.add_trace(go.Scatter(
            x=bearish.index, y=bearish['High'] * 1.02, mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='red'), name='Breakout Bajista'
        ))

        # Dibujar el último canal detectado
        last_idx = len(df) - 1
        x_range = np.array(range(last_idx - backcandles, last_idx + 1))
        x_dates = df.index[x_range]
        
        # Líneas del canal actual
        rel_x = np.array(range(len(x_range)))
        y_high = df['slope_high'].iloc[last_idx] * rel_x + df['interc_high'].iloc[last_idx]
        y_low = df['slope_low'].iloc[last_idx] * rel_x + df['interc_low'].iloc[last_idx]

        fig.add_trace(go.Scatter(x=x_dates, y=y_high, mode='lines', line=dict(color='yellow', dash='dot'), name='Canal Sup.'))
        fig.add_trace(go.Scatter(x=x_dates, y=y_low, mode='lines', line=dict(color='orange', dash='dot'), name='Canal Inf.'))

        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width='stretch' ) #use_container_width=True

        # 4. Estadísticas
        col1, col2 = st.columns(2)
        col1.metric("Rupturas Alcistas", len(bullish))
        col2.metric("Rupturas Bajistas", len(bearish))

    except Exception as e:
        st.error(f"Error detectando rupturas: {e}")