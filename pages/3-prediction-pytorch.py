import streamlit as st
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import signal_market_rust
import sys
import os

# Configuración de página
st.set_page_config(page_title="PyTorch Prediction", page_icon="🔥", layout="wide")

# Asegurar acceso a módulos superiores
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- DEFINICIÓN DEL MODELO PYTORCH ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

st.header(":material/finance_mode: Predicción con PyTorch + Rust")
st.markdown("Implementación de una red **LSTM profunda** usando tensores de PyTorch y procesamiento de secuencias en Rust.")

# --- SIDEBAR ---
with st.sidebar:
    st.header(":materail/settings_b_roll: Parámetros PyTorch")
    ticker = st.text_input("Ticker", "AAPL").upper()
    ticker_query = f"{ticker}.US" if not ticker.endswith(".US") else ticker
    
    epochs = st.slider("Epochs (Pasadas)", 10, 100, 50)
    lr = st.select_slider("Learning Rate", options=[0.01, 0.005, 0.001], value=0.001)
    time_step = st.slider("Ventana (Lookback)", 30, 100, 60)
    
    btn_train = st.button("Entrenar Modelo PyTorch", use_container_width=True)

if btn_train:
    try:
        # 1. Carga de datos
        with st.spinner("Cargando datos históricos..."):
            df = web.DataReader(ticker_query, "stooq").sort_index()
            data = df['Close'].values.reshape(-1, 1)
            
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = scaler.fit_transform(data).flatten().tolist()

        # 2. PROCESAMIENTO CON RUST ( create_sequences )
        with st.spinner("Rust está procesando las secuencias..."):
            X_raw, y_raw = signal_market_rust.create_sequences(scaled_data, time_step)
            
            # Convertir a Tensores de PyTorch
            X_tensor = torch.FloatTensor(np.array(X_raw)).unsqueeze(-1) # [B, T, F]
            y_tensor = torch.FloatTensor(np.array(y_raw)).unsqueeze(-1)

        # 3. ENTRENAMIENTO
        with st.spinner(f"Entrenando en CPU... (Epochs: {epochs})"):
            model = LSTMModel()
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            model.train()
            progress_bar = st.progress(0)
            for i in range(epochs):
                optimizer.zero_grad()
                y_pred = model(X_tensor)
                single_loss = loss_function(y_pred, y_tensor)
                single_loss.backward()
                optimizer.step()
                progress_bar.progress((i + 1) / epochs)
            
            st.success(f"Modelo entrenado. Loss final: {single_loss.item():.6f}")

        # 4. PREDICCIÓN (30 DÍAS)
        model.eval()
        preds = []
        current_batch = torch.FloatTensor(scaled_data[-time_step:]).view(1, time_step, 1)

        with torch.no_grad():
            for _ in range(30):
                pred = model(current_batch)
                preds.append(pred.item())
                # Actualizar ventana: quitar primero, añadir predicción al final
                new_pred_tensor = pred.view(1, 1, 1)
                current_batch = torch.cat((current_batch[:, 1:, :], new_pred_tensor), dim=1)

        # Invertir normalización
        actual_predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

        # 5. GRÁFICO
        st.subheader(f"Resultado de la Proyección: {ticker}")
        
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq="B")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], name="Real", line=dict(color="#00ffcc")))
        fig.add_trace(go.Scatter(x=future_dates, y=actual_predictions.flatten(), name="PyTorch Pred", line=dict(color="#ff007f", dash='dot')))
        
        fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")
        st.plotly_chart(fig, width='stretch' ) #use_container_width=True

        st.divider()
        col_t1, col_t2 = st.columns([1, 2])
        
        with col_t1:
            st.subheader(":material/list_alt: Valores Proyectados")
            st.info("Precios calculados mediante la iteración de la ventana de tiempo (Lookback) sobre la red LSTM.")

        with col_t2:
            # Crear DataFrame de resultados
            df_pytorch = pd.DataFrame({
                "Fecha": future_dates.strftime('%Y-%m-%d'),
                "Predicción (USD)": actual_predictions.flatten()
            })
            
            # Mostrar tabla interactiva
            st.dataframe(
                df_pytorch.style.highlight_max(axis=0, color='#ff007f33')
                .highlight_min(axis=0, color='#00ffcc33')
                .format({"Predicción (USD)": "${:.2f}"}),
                width='stretch'  
            ) #use_container_width=True

    except Exception as e:
        st.error(f"Error en el flujo: {e}")