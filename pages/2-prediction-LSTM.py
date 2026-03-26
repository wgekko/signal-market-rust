import streamlit as st
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import signal_market_rust
import sys
import os

# Asegurar que encuentre las utilidades y el módulo de Rust si están en la carpeta superior
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Predicción LSTM-tensorflow", page_icon=":material/graph_2:", layout="wide")

st.header("Predicción de Precios con Redes Neuronales (LSTM)- Tensorflow")
st.markdown("Modelo impulsado por TensorFlow y optimizado con **Rust** para el procesamiento de tensores.")

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("Parámetros del Modelo")
    ticker = st.text_input("Ticker (ej: AAPL, AMD)", "AMD").upper()
    
    if not ticker.endswith(".US"):
        ticker_query = f"{ticker}.US"
    else:
        ticker_query = ticker
        
    years = st.slider("Años de historial", 1, 10, 5)
    epochs_val = st.slider("Epochs de entrenamiento", 1, 20, 5)
    time_step = st.slider("Ventana de tiempo (Días)", 30, 100, 60)
    
    btn_predict = st.button("Entrenar y Predecir", width='stretch' ) #use_container_width=True

# --- EJECUCIÓN ---
if btn_predict:
    try:
        with st.spinner(f"Descargando {years} años de datos para {ticker}..."):
            start_date = pd.Timestamp.today() - pd.DateOffset(years=years)
            df = web.DataReader(ticker_query, "stooq", start=start_date).sort_index()

        if df.empty:
            st.error(":material/error: No se encontraron datos.")
        else:
            st.success(":material/done_all: Datos descargados correctamente.")
            
            # --- PREPARACIÓN (Optimizada con Rust) ---
            with st.spinner("Preparando tensores con motor Rust..."):
                data_close = df.filter(['Close']).values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data_close)
                
                # Convertimos a lista plana de f64 para Rust
                flat_data = scaled_data.flatten().tolist()
                
                # ¡MAGIA DE RUST AQUÍ! Reemplaza el lento bucle for de Python
                x_train_rust, y_train_rust = signal_market_rust.create_sequences(flat_data, time_step)
                
                # Convertir de vuelta a arrays de numpy para TensorFlow
                x_train = np.array(x_train_rust)
                y_train = np.array(y_train_rust)
                
                # Reshape requerido por LSTM [muestras, pasos de tiempo, características]
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # --- ENTRENAMIENTO LSTM ---
            with st.spinner(f"Entrenando red neuronal LSTM ({epochs_val} epochs)..."):
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=epochs_val, batch_size=32, verbose=0)
            
            # --- PREDICCIÓN FUTURA ---
            with st.spinner("Proyectando siguientes 30 días..."):
                last_window = scaled_data[-time_step:]
                future_input = last_window.reshape(1, -1)
                
                predictions = []
                steps = 30
                
                for _ in range(steps):
                    pred = model.predict(future_input.reshape(1, time_step, 1), verbose=0)
                    predictions.append(pred[0][0])
                    future_input = np.append(future_input[:, 1:], pred).reshape(1, -1)
                
                # Invertir la escala para tener precios reales
                predictions_real = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            # --- VISUALIZACIÓN INTERACTIVA (Plotly) ---
            st.subheader(":material/bar_chart_4_bars: Proyección del Modelo")
            
            # Fechas futuras
            future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="B")
            
            fig = go.Figure()
            # Datos históricos (últimos 150 días para ver de cerca)
            fig.add_trace(go.Scatter(
                x=df.index[-150:], y=df['Close'].iloc[-150:], 
                mode='lines', name='Histórico', line=dict(color='white')
            ))
            # Predicción
            fig.add_trace(go.Scatter(
                x=future_dates, y=predictions_real.flatten(), 
                mode='lines', name='Predicción LSTM', line=dict(color='#ff4b4b', dash='dash')
            ))
            
            fig.update_layout(
                title=f"Predicción a 30 días: {ticker}",
                yaxis_title="Precio USD",
                template="plotly_dark",
                height=600,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig,  width='stretch') #use_container_width=True

            st.divider()
            st.subheader(":material/table_rows: Detalle de Proyección (Próximos 30 Días)")
            
            # Crear DataFrame para la tabla
            df_proyeccion = pd.DataFrame({
                "Día": range(1, steps + 1),
                "Fecha Est.:calendar:": future_dates.strftime('%d/%m/%Y'),
                "Precio Predicho (USD):moneybag:": predictions_real.flatten()
            })

            # Mostrar tabla con formato
            st.dataframe(
                df_proyeccion.style.format({"Precio Predicho (USD):moneybag:": "{:.2f}"})
                .background_gradient(cmap='YlOrRd', subset=["Precio Predicho (USD):moneybag:"]),
                width='stretch',
                hide_index=True
            ) #use_container_width=True

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")