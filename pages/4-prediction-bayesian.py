import streamlit as st
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import signal_market_rust
import sys
import os

st.set_page_config(page_title="Bayesian Analysis", page_icon=":material/table_chart_view:", layout="wide")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- MODELO BAYESIANO (BNN) ---
class BayesianLSTM(PyroModule):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = PyroModule[nn.Linear](hidden_dim, output_dim)
        
        # Definimos los priors (creencias previas) para los pesos de la capa final
        self.fc.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, hidden_dim]).to_event(2))
        self.fc.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

    def forward(self, x, y=None):
        lstm_out, _ = self.lstm(x)
        mu = self.fc(lstm_out[:, -1, :]).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(1., 1.))
        
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu

st.header(":material/table_chart_view: Redes Neuronales Bayesianas (BNN)")
st.markdown("Este modelo utiliza **Inferencia Variacional** para calcular no solo la tendencia, sino el **margen de riesgo** de la predicción.")

# --- SIDEBAR ---
with st.sidebar:
    st.header(":material/settings_alert: Parámetros Bayesianos")
    ticker = st.text_input("Ticker", "NVDA").upper()
    ticker_query = f"{ticker}.US" if not ticker.endswith(".US") else ticker
    
    iterations = st.slider("Iteraciones de Entrenamiento (SVI)", 100, 1000, 500)
    time_step = st.slider("Ventana de Observación", 10, 60, 30)
    
    btn_run = st.button("Ejecutar Inferencia", width='stretch') #use_container_width=True

if btn_run:
    try:
        # 1. Datos
        with st.spinner("Descargando desde Stooq..."):
            df = web.DataReader(ticker_query, "stooq").sort_index()
            data = df['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data).flatten().tolist()

        # 2. RUST: Creación de secuencias
        with st.spinner("Rust preparando tensores..."):
            X_raw, y_raw = signal_market_rust.create_sequences(scaled_data, time_step)
            x_tensor = torch.tensor(X_raw, dtype=torch.float).unsqueeze(-1)
            y_tensor = torch.tensor(y_raw, dtype=torch.float)

        # 3. Entrenamiento Bayesiano (Inferencia Variacional)
        with st.spinner("Realizando Inferencia Variacional (SVI)..."):
            pyro.clear_param_store()
            model = BayesianLSTM()
            guide = AutoDiagonalNormal(model)
            optimizer = pyro.optim.Adam({"lr": 0.01})
            svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

            for i in range(iterations):
                loss = svi.step(x_tensor, y_tensor)
                if i % 100 == 0:
                    st.write(f"Iteración {i} - Loss: {loss:.4f}")

        # 4. Predicción con Incertidumbre
        with st.spinner("Calculando bandas de confianza..."):
            predictive = Predictive(model, guide=guide, num_samples=100)
            
            # Preparar último bloque para predecir futuro
            last_window = torch.tensor(scaled_data[-time_step:], dtype=torch.float).view(1, time_step, 1)
            samples = predictive(last_window)
            
            # Extraer media y desviación estándar de las muestras
            future_preds = samples['obs'].detach().numpy()
            mean_pred = future_preds.mean(axis=0)
            std_pred = future_preds.std(axis=0)

            # Invertir escala
            final_mean = scaler.inverse_transform(mean_pred.reshape(-1, 1))
            upper_bound = scaler.inverse_transform((mean_pred + (1.96 * std_pred)).reshape(-1, 1))
            lower_bound = scaler.inverse_transform((mean_pred - (1.96 * std_pred)).reshape(-1, 1))

        # 5. GRÁFICO DE INCERTIDUMBRE
        st.subheader(f"Proyección Bayesiana: {ticker}")
        
        fig = go.Figure()
        
        # Histórico
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], name="Precio Real", line=dict(color="white")))
        
        # Bandas de Incertidumbre (95% confianza)
        future_date = df.index[-1] + pd.Timedelta(days=1)
        
        fig.add_trace(go.Scatter(
            x=[future_date], y=[upper_bound[0][0]],
            mode='markers', name='Límite Superior', marker=dict(color='rgba(255, 0, 0, 0)')
        ))
        fig.add_trace(go.Scatter(
            x=[future_date], y=[lower_bound[0][0]],
            fill='tonexty', fillcolor='rgba(255, 255, 255, 0.2)',
            mode='none', name='Rango de Incertidumbre'
        ))
        
        # Punto de predicción media
        fig.add_trace(go.Scatter(
            x=[future_date], y=[final_mean[0][0]], 
            mode='markers+text', name='Predicción Media',
            text=[f"${final_mean[0][0]:.2f}"], textposition="top center",
            marker=dict(color='yellow', size=12, symbol='diamond')
        ))

        fig.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig, width='stretch') #use_container_width=True
        
        st.info("Las bandas sombreadas representan el área donde el modelo espera que el precio se mueva con un 95% de probabilidad.")

    except Exception as e:
        st.error(f"Error en el modelo bayesiano: {e}")