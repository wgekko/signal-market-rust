import streamlit as st
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import signal_market_rust
import sys
import os

st.set_page_config(page_title="Asset Correlation", page_icon=":material/line_axis:", layout="wide")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.title(":material/line_axis: Análisis de Correlación y Desacople")
st.markdown("Detecta anomalías estadísticas cuando dos activos que suelen moverse juntos pierden su sincronía.")

# --- SIDEBAR ---
with st.sidebar:
    st.subheader(":material/settings_alert: Configuración")
    base_ticker = st.text_input("Activo Principal", "SPY.US").upper()
    comp_tickers = st.text_input("Comparar contra (separados por coma)", "QQQ.US, GLD.US, TLT.US").upper().split(",")
    
    window = st.slider("Ventana de Correlación (Corta)", 10, 60, 30)
    wide_window = st.slider("Ventana de Media (Larga)", 50, 200, 100)
    std_factor = st.slider("Factor de Desviación (Señal)", 1.0, 4.0, 2.0)
    
    btn_analyze = st.button("Calcular Correlaciones", width='stretch' ) #use_container_width=True

if btn_analyze:
    try:
        all_tickers = [base_ticker] + [t.strip() for t in comp_tickers]
        
        with st.spinner("Descargando múltiples activos..."):
            data_frames = {}
            for t in all_tickers:
                df_temp = web.DataReader(t, "stooq").sort_index()
                data_frames[t] = df_temp['Close']
            
            master_df = pd.concat(data_frames, axis=1).dropna()

        # 1. MATRIZ DE CORRELACIÓN (Mejora visual)
        st.subheader(":material/grid_4x4: Matriz de Correlación Actual")
        corr_matrix = master_df.corr()
        fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_heat, width='stretch' ) #use_container_width=True

        # 2. ANÁLISIS DINÁMICO (Cálculo con Rust)
        target_comp = [t.strip() for t in comp_tickers][0] # Analizamos el primero de la lista en detalle
        st.subheader(f"Análisis Detallado: {base_ticker} vs {target_comp}")

        with st.spinner("Calculando señales con motor Rust..."):
            x_vals = master_df[base_ticker].tolist()
            y_vals = master_df[target_comp].tolist()
            
            # Llamada a Rust
            rolling_corr = signal_market_rust.rolling_correlation(x_vals, y_vals, window)
            
            df_signals = pd.DataFrame(index=master_df.index)
            df_signals['corr'] = rolling_corr
            df_signals['mean'] = df_signals['corr'].rolling(wide_window).mean()
            df_signals['std'] = df_signals['corr'].rolling(wide_window).std()
            
            # Generar señales (Lógica del notebook mejorada)
            df_signals['upper'] = df_signals['mean'] + (df_signals['std'] * std_factor)
            df_signals['lower'] = df_signals['mean'] - (df_signals['std'] * std_factor)
            
            df_signals['signal'] = 0
            df_signals.loc[df_signals['corr'] < df_signals['lower'], 'signal'] = 1 # Desacople negativo
            df_signals.loc[df_signals['corr'] > df_signals['upper'], 'signal'] = -1 # Exceso de correlación

        # 3. GRÁFICOS INTERACTIVOS
        # Gráfico de Correlación
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(x=df_signals.index, y=df_signals['corr'], name="Corr. Móvil", line=dict(color='#00d1ff')))
        fig_corr.add_trace(go.Scatter(x=df_signals.index, y=df_signals['mean'], name="Promedio", line=dict(color='white', dash='dash')))
        
        # Sombreado de bandas
        fig_corr.add_trace(go.Scatter(x=df_signals.index, y=df_signals['upper'], line=dict(width=0), showlegend=False))
        fig_corr.add_trace(go.Scatter(x=df_signals.index, y=df_signals['lower'], fill='tonexty', fillcolor='rgba(255,255,255,0.1)', name="Banda de Confianza", line=dict(width=0)))

        # Marcar señales de desacople
        signals = df_signals[df_signals['signal'] != 0]
        fig_corr.add_trace(go.Scatter(x=signals.index, y=signals['corr'], mode='markers', marker=dict(color='yellow', size=10), name="Alerta de Desacople"))

        fig_corr.update_layout(title="Correlación Dinámica y Bandas de Bollinger de Correlación", template="plotly_dark", height=500)
        st.plotly_chart(fig_corr, width='stretch' ) #use_container_width=True

        # 4. EXPLICACIÓN PARA EL USUARIO
        st.info(f"""
        **Interpretación:** - Cuando la línea azul sale de la banda sombreada (puntos amarillos), los activos han perdido su relación histórica. 
        - Si la correlación cae bruscamente, podría indicar que un activo está reaccionando a una noticia que el otro aún no procesa.
        """)

    except Exception as e:
        st.error(f"Error en el análisis de correlación: {e}")