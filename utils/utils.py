import pandas_datareader.data as web
import pandas as pd
import signal_market_rust 
import plotly.graph_objects as go


def load_data_stooq(symbol: str, start: str, end: str):
    # Requisito: Autocompletar .US para mercado americano
    ticker = symbol.upper()
    if not ticker.endswith(".US"):
        ticker = f"{ticker}.US"
    
    try:
        # Descarga sin límites de consulta
        df = web.DataReader(ticker, 'stooq', start, end)
        return df.sort_index(ascending=True)
    except Exception as e:
        print(f"Error en Stooq: {e}")
        return pd.DataFrame()

def apply_rust_analysis(df, lookback=40):
    if df.empty: return df
    
    # Llamada al motor de Rust
    s_high, s_low = signal_market_rust.fast_process_signals(
        df['High'].tolist(), 
        df['Low'].tolist(), 
        lookback
    )
    
    df['slope_high'] = s_high
    df['slope_low'] = s_low
    
    # Lógica de detección: Pendiente superior negativa y inferior positiva (convergencia)
    # Ajustamos umbrales para que no sea tan estricto
    df['is_flag'] = (df['slope_high'] < -0.001) & (df['slope_low'] > 0.001)
    
    return df

def plot_advanced_chart(df, ticker):
    # Crear el gráfico de velas
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Precio"
    )])

    # Filtrar donde hay banderas detectadas para dibujar las líneas
    flags = df[df['is_flag']].tail(5) # Mostramos las últimas 5 banderas detectadas
    
    for idx, row in flags.iterrows():
        # Obtenemos la posición en el tiempo
        end_date = idx
        # Encontramos la posición entera para el lookback (aprox 40 días atrás)
        start_idx = df.index.get_loc(idx) - 40
        if start_idx < 0: continue
        start_date = df.index[start_idx]
        
        # Proyectar línea superior (Resistencia)
        # Usamos el High actual como punto final y retrocedemos con la pendiente
        high_val = row['High']
        slope_h = row['slope_high']
        
        fig.add_trace(go.Scatter(
            x=[start_date, end_date],
            y=[high_val - (slope_h * 40), high_val],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name="Resistencia Flag"
        ))
        
        # Proyectar línea inferior (Soporte)
        low_val = row['Low']
        slope_l = row['slope_low']
        
        fig.add_trace(go.Scatter(
            x=[start_date, end_date],
            y=[low_val - (slope_l * 40), low_val],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name="Soporte Flag"
        ))

    fig.update_layout(
        title=f"Análisis Técnico: {ticker}.US",
        yaxis_title="Precio USD",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=700
    )
    
    return fig