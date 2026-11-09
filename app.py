import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Accidentalidad - Medellín", page_icon="🚦", layout="centered")

# 2. Carga de artefactos en caché
@st.cache_resource
def load_models():
    scaler = joblib.load('minmax_scaler.joblib')
    ohe = joblib.load('onehot_encoder.joblib')
    modelo = joblib.load('modelo_logistico.joblib')
    le = joblib.load('label_encoder.joblib')
    return scaler, ohe, modelo, le

scaler, ohe, modelo, le = load_models()

# 3. Interfaz de Usuario
st.title("🚦 Predictor de Gravedad en Incidentes Viales (Medellín)")
st.markdown("""
Esta aplicación predice si un incidente vial resultará en **Solo daños (Leve)** o si habrá **Heridos/Muertos (Grave)**.
""")

st.header("Ingrese los datos del incidente")

col1, col2 = st.columns(2)

with col1:
    hora = st.slider("Hora del día (0-23)", min_value=0, max_value=23, value=12)
    mes = st.slider("Mes del año (1-12)", min_value=1, max_value=12, value=6)
    dia_semana = st.selectbox("Día de la semana", ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])

with col2:
    clase_accidente = st.selectbox("Clase de Accidente", ['Choque', 'Atropello', 'Caida Ocupante', 'Volcamiento', 'Otro'])
    diseno = st.selectbox("Diseño de la vía", ['Tramo de via', 'Interseccion', 'Glorieta', 'Lote o Predio', 'Paso Peatonal'])
    comuna = st.selectbox("Comuna", ['La Candelaria', 'Laureles', 'El Poblado', 'Belen', 'Robledo', 'Manrique', 'Aranjuez', 'Doce de Octubre', 'Buenos Aires', 'Villa Hermosa', 'San Javier', 'America', 'Castilla', 'Popular', 'Santa Cruz', 'Guayabal'])

# 4. Botón de Predicción
if st.button("Predecir Severidad", type="primary"):
    datos_num = pd.DataFrame({'HORA': [hora], 'MES': [mes]})
    datos_cat = pd.DataFrame({'CLASE_ACCIDENTE': [clase_accidente], 'DISEÑO': [diseno], 'COMUNA': [comuna], 'DIA_SEMANA': [dia_semana]})
    
    num_scaled = scaler.transform(datos_num)
    cat_encoded = ohe.transform(datos_cat)
    X_input = np.hstack((num_scaled, cat_encoded))
    
    prediccion_num = modelo.predict(X_input)
    prediccion_texto = le.inverse_transform(prediccion_num)[0]
    
    st.divider()
    if "Grave" in prediccion_texto:
        st.error(f"🚨 **ALERTA ALTA:** El modelo predice un accidente **{prediccion_texto}**. Se recomienda despacho prioritario de ambulancia.")
    else:
        st.success(f"✅ **ALERTA BAJA:** El modelo predice un accidente **{prediccion_texto}**. Asignar unidad de tránsito estándar.")
