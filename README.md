# signal-market-rust
Dashboard Cuantitativo completo de 6 niveles! 
Tienes todo lo necesario para analizar el mercado desde perspectivas técnicas, 
estadísticas y de inteligencia artificial. 

para instalar desde linux el archivo requiremets.txt
pip install -r requirements.txt --break-system-packages

para generar la carpeta target necesaria para que se despliegue la app
deben escribir el comando 
con entorno de desarrollo 
maturin develop --release
sin entorno de desarrollo
maturin build --release
para configurar las fuentes, colores de fondos y demas 
deben crear una carpeta  .streamlit y colocar dentro el archivo que esta en la carpeta adjunta de la carpeta streamlit

para ejecutar la app deben correr desde la terminal dentro de la carpeta de la app el comando 
streamlit run main.py

-----------------------------------------------------------------------------------------------------------------------------------------------
Donde el Código se encuentra con los Mercados: Mi nueva "Quant Station" (Python + Rust)
¿Qué pasa cuando combinas la agilidad de Python con la potencia bruta de Rust para el análisis financiero? El resultado es una herramienta de grado institucional que acabo de desarrollar.

He construido una plataforma de análisis cuantitativo integrada en Streamlit que no solo observa el pasado, sino que modela el futuro con una arquitectura híbrida optimizada.

🛠️ El Stack Tecnológico:
Core de Cálculo: Desarrollado en Rust (vía Maturin) para procesar simulaciones y secuencias de datos a velocidad nativa.

Inteligencia Artificial: Implementaciones de LSTM (TensorFlow), PyTorch y Bayesian Neural Networks (Pyro) para capturar no solo la tendencia, sino la incertidumbre del mercado.

Visualización: Dashboards interactivos con Plotly para análisis técnico avanzado.

Lo que hace especial a esta App:
Predicción Bayesiana: No solo obtenemos un precio objetivo, sino un rango de confianza (probabilidad) para gestionar el riesgo.

Simulaciones de Monte Carlo: 10,000+ escenarios generados en milisegundos gracias a la eficiencia de Rust.

Detección de Rupturas y Niveles: Algoritmos de regresión lineal para identificar canales de consolidación y "breakouts" en tiempo real.

Análisis de Correlación: Matrices dinámicas para detectar desacoples entre activos y arbitraje estadístico.

Este proyecto nació de la necesidad de tener herramientas que no se "congelen" al procesar grandes volúmenes de datos y que ofrezcan una visión matemática profunda más allá de los indicadores clásicos.

#Python #Rust #FinTech #DataScience #QuantitativeAnalysis #MachineLearning #Trading


video demo 

