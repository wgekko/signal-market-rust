import pandas_datareader.data as web
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def load_data_stooq(symbol: str, start: str, end: str):
    ticker = symbol.upper()
    if not ticker.endswith(".US"):
        ticker = f"{ticker}.US"
    try:
        df = web.DataReader(ticker, 'stooq', start, end)
        return df.sort_index(ascending=True)
    except Exception as e:
        print(f"Error en Stooq: {e}")
        return pd.DataFrame()

def get_key_levels(df, bin_width=None):
    if df.empty: return []
    
    # 1. Identificar Pivots (Máximos y Mínimos locales)
    # Usamos una ventana simple de 2 velas a cada lado
    highs = df[
        (df['High'] > df['High'].shift(1)) & 
        (df['High'] > df['High'].shift(2)) & 
        (df['High'] > df['High'].shift(-1)) & 
        (df['High'] > df['High'].shift(-2))
    ]['High']
    
    lows = df[
        (df['Low'] < df['Low'].shift(1)) & 
        (df['Low'] < df['Low'].shift(2)) & 
        (df['Low'] < df['Low'].shift(-1)) & 
        (df['Low'] < df['Low'].shift(-2))
    ]['Low']

    # 2. Lógica del Notebook: Histogramas y Bins
    # Si no se define bin_width, usamos el 0.3% del precio promedio
    if bin_width is None:
        bin_width = df['Close'].mean() * 0.003

    all_pivots = pd.concat([highs, lows])
    
    # Agrupamos los precios en "bins" (contenedores)
    min_p = all_pivots.min()
    max_p = all_pivots.max()
    bins = np.arange(min_p, max_p + bin_width, bin_width)
    
    # Contamos cuántos pivots caen en cada rango de precio
    counts, bin_edges = np.histogram(all_pivots, bins=bins)
    
    # 3. Filtrar niveles significativos
    # Consideramos nivel clave si tiene más de 2 toques (frecuencia > 2)
    key_levels = []
    for i in range(len(counts)):
        if counts[i] >= 2:
            level = bin_edges[i] + (bin_width / 2)
            key_levels.append(level)
            
    return key_levels

def plot_key_levels_chart(df, ticker, levels):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Precio"
    )])

    # Dibujar los niveles detectados
    for lvl in levels:
        fig.add_hline(
            y=lvl,
            line_dash="dash",
            line_color="rgba(173, 216, 230, 0.6)", # Azul claro traslúcido
            annotation_text=f"Lvl: {lvl:.2f}",
            annotation_position="bottom right"
        )

    fig.update_layout(
        title=f"Detección de Niveles Clave (Histograma): {ticker}",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=700
    )
    return fig