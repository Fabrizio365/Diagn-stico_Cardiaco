import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------------ CARGA MODELO YA ENTRENADO ------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
    X = df.drop(["target", "patientid"], axis=1)
    y = df["target"]
    return X, y

@st.cache_resource
def cargar_modelos():
    modelo = joblib.load("modelo_hard_voting.pkl")
    scaler = joblib.load("scaler.pkl")
    return modelo, scaler

X, y = cargar_datos()
modelo, scaler = cargar_modelos()

# --------------------------- INTERFAZ STREAMLIT ---------------------------
st.set_page_config(page_title="Predicción Cardiovascular", layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Open Sans', sans-serif;
            background: linear-gradient(to bottom right, #8B0000, #FF6347);
            color: white;
        }
        .fade-transition {
            animation: fadeForm 1s ease-in-out;
        }
        @keyframes fadeForm {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in {
            animation: fadeIn 2s ease-in;
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .form-container {
            animation: rgbGlow 5s infinite alternate;
            border-radius: 15px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
        }
        @keyframes rgbGlow {
            0% { box-shadow: 0 0 15px red; }
            33% { box-shadow: 0 0 15px green; }
            66% { box-shadow: 0 0 15px blue; }
            100% { box-shadow: 0 0 15px red; }
        }
        .rgb-text {
            animation: rgbText 5s infinite alternate;
        }
        @keyframes rgbText {
            0% { text-shadow: 0 0 5px red; color: red; }
            33% { text-shadow: 0 0 5px green; color: green; }
            66% { text-shadow: 0 0 5px blue; color: blue; }
            100% { text-shadow: 0 0 5px red; color: red; }
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

if "formulario" not in st.session_state:
    st.session_state.formulario = False

if not st.session_state.formulario:
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("""
            <div class='fade-in' style='padding: 2rem; border-radius: 1rem; box-shadow: 0 0 10px rgba(0,0,0,0.2);'>
            <h2 class='rgb-text'>🪀 Bienvenido a la Evaluación de Riesgo Cardiovascular</h2>
            <p style='font-size:17px; line-height:1.6;'>
            <strong>¿Qué es esto?</strong><br>
            Esta herramienta usa inteligencia artificial para ayudarte a saber si tienes un riesgo alto o bajo de enfermedad al corazón.
            </p>
            <p style='font-size:17px; line-height:1.6;'>
            <strong>¿Cómo funciona?</strong><br>
            Combina tres métodos inteligentes: <b>Árbol de Decisión</b>, <b>K-Nearest Neighbors</b> y <b>Gradient Boosting</b>, y juntos toman una decisión final como equipo.
            </p>
            <p style='font-size:17px; line-height:1.6;'>
            <strong>¿Por qué importa?</strong><br>
            Porque conocer tu salud cardiovascular te ayuda a tomar decisiones antes de que sea tarde. Esto <b>no reemplaza a un médico</b>, pero sí puede ayudarte a saber si necesitas uno pronto.
            </p>
            <blockquote style='font-size:16px;'>💖 Prevenir es vivir. Cuida tu corazón desde hoy.</blockquote>
            </div>
        """, unsafe_allow_html=True)
    with colB:
        st.image("imagen_logo.png", width=300)

    st.markdown("<div class='start-button'>", unsafe_allow_html=True)
    if st.button("🧪 Iniciar Evaluación", key="iniciar"):
        st.session_state.formulario = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Sidebar con glosario de variables
with st.sidebar:
    st.image("imagen_logo.png")
    with st.expander("ℹ️ Glosario de variables del formulario", expanded=False):
        st.markdown("""
        <ul style='font-size: 15px; line-height: 1.6;'>
            <li><b>Edad:</b> Años que tiene el paciente.</li>
            <li><b>Género:</b> <b>0</b> para mujer, <b>1</b> para hombre.</li>
            <li><b>Tipo de dolor en el pecho:</b> 
                <ul>
                    <li><b>0:</b> Angina típica (relacionada al esfuerzo)</li>
                    <li><b>1:</b> Angina atípica</li>
                    <li><b>2:</b> Dolor no anginoso</li>
                    <li><b>3:</b> Asintomático (sin dolor)</li>
                </ul>
            </li>
            <li><b>Presión arterial en reposo:</b> Presión sistólica (valor alto) mientras el paciente está en reposo (mmHg).</li>
            <li><b>Nivel de colesterol sérico:</b> Cantidad total de colesterol en sangre (mg/dL).</li>
            <li><b>¿Azúcar en ayunas &gt; 120?</b> <b>1 = Sí</b>, <b>0 = No</b>.</li>
            <li><b>Electrocardiograma en reposo:</b>
                <ul>
                    <li><b>0:</b> Normal</li>
                    <li><b>1:</b> Anomalía ST-T (posible isquemia)</li>
                    <li><b>2:</b> Hipertrofia ventricular izquierda</li>
                </ul>
            </li>
            <li><b>Frecuencia cardiaca máxima:</b> Mayor número de latidos durante esfuerzo.</li>
            <li><b>¿Angina inducida por ejercicio?:</b> Dolor en el pecho al hacer ejercicio.</li>
            <li><b>Oldpeak:</b> Descenso del ST en ECG. Valores altos = posible isquemia.</li>
            <li><b>Pendiente del ST:</b> Forma de la curva ST en ECG:
                <ul>
                    <li><b>0:</b> Descendente (riesgo alto)</li>
                    <li><b>1:</b> Plana (riesgo medio)</li>
                    <li><b>2:</b> Ascendente (normal)</li>
                </ul>
            </li>
            <li><b>Número de vasos mayores:</b> Cantidad observada con contraste (de 0 a 3).</li>
        </ul>
        """, unsafe_allow_html=True)

# Formulario principal
with st.container():
    st.markdown("<div class='fade-transition form-container'>", unsafe_allow_html=True)
    st.title("📜 Formulario de Evaluación")

col1, col2, col3 = st.columns(3)
with col1:
    edad = st.number_input("Edad", min_value=1, max_value=120, step=1)
    genero = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    dolor_pecho = st.selectbox("Tipo de dolor en el pecho", [0, 1, 2, 3])
    presion = st.number_input("Presión arterial en reposo", min_value=80, max_value=200)

with col2:
    colesterol = st.number_input("Nivel de colesterol sérico", min_value=100, max_value=600)
    azucar = st.radio("¿Azúcar en ayunas > 120?", [0, 1], format_func=lambda x: f"{'✅ Sí' if x == 1 else '❌ No'}")
    electro = st.selectbox("Electrocardiograma en reposo", [0, 1, 2])
    frecuencia = st.number_input("Frecuencia cardiaca máxima", min_value=60, max_value=250)

with col3:
    angina = st.radio("¿Angina inducida por ejercicio?", [0, 1], format_func=lambda x: f"{'✅ Sí' if x == 1 else '❌ No'}")
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)
    pendiente = st.selectbox("Pendiente del ST", [0, 1, 2])
    vasos = st.selectbox("Número de vasos mayores", [0, 1, 2, 3])

st.markdown("</div>", unsafe_allow_html=True)

# Predicción
if st.button("🔍 Predecir estado de salud"):
    campos = [edad, genero, dolor_pecho, presion, colesterol, azucar, electro, frecuencia, angina, oldpeak, pendiente, vasos]
    if all(v is not None for v in campos):
        entrada = np.array([campos])
        try:
            entrada_esc = scaler.transform(entrada)
            resultado = modelo.predict(entrada_esc)

            if resultado[0] == 1:
                st.markdown("<div class='fade-transition'>", unsafe_allow_html=True)
                st.error("⚠️ Riesgo cardiovascular detectado")
                st.image("riesgo.png", width=150)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='fade-transition'>", unsafe_allow_html=True)
                st.success("✅ Estado cardiovascular saludable")
                st.image("saludable.png", width=150)
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Ocurrió un error al procesar los datos: {e}")
    else:
        st.warning("⚠️ Por favor completa todos los campos antes de predecir.")
