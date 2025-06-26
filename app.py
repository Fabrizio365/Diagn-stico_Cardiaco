import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------ ENTRENAMIENTO DIRECTO ------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
    X = df.drop(["target", "patientid"], axis=1)
    y = df["target"]
    return X, y

X, y = cargar_datos()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# Modelos base
dt = DecisionTreeClassifier(max_depth=20, criterion='gini', min_samples_leaf=5, splitter='random')
knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)

modelo = VotingClassifier(estimators=[('dt', dt), ('knn', knn), ('gb', gb)], voting='hard')
modelo.fit(X_train, y_train)

# --------------------------- INTERFAZ STREAMLIT ---------------------------
st.set_page_config(page_title="Predicción Cardiovascular", layout="wide")

color_fondo_bienvenida = "linear-gradient(to bottom right, #8B0000, #FF6347);"
color_fondo_formulario = "linear-gradient(to right, #003366, #004080);"

if "formulario" not in st.session_state:
    st.session_state.formulario = False

if not st.session_state.formulario:
    st.markdown(f"""
        <style>
        html, body, [class*="css"]  {{
            font-family: 'Open Sans', sans-serif;
            background: {color_fondo_bienvenida};
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <style>
        html, body, [class*="css"]  {{
            font-family: 'Open Sans', sans-serif;
            background: {color_fondo_formulario};
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
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
            border-radius: 15px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

if not st.session_state.formulario:
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("""
            <div class='fade-in' style='padding: 2rem; border-radius: 1rem; box-shadow: 0 0 10px rgba(0,0,0,0.2);'>
            <h2 style='color: white;'>🫀 Bienvenido a la Evaluación de Riesgo Cardiovascular</h2>
            <p style='font-size:18px;'>
            Este sistema predictivo usa <strong>inteligencia artificial</strong> para evaluar el riesgo de enfermedades cardíacas en función de tus datos clínicos.
            </p>
            <p><strong>¿Cómo funciona?</strong><br>
            Se combinan tres modelos: <b>Árbol de Decisión</b>, <b>K-Nearest Neighbors</b> y <b>Gradient Boosting</b> usando <em>Ensemble Learning</em> (votación mayoritaria). Esto nos permite tomar decisiones más precisas y seguras al integrar la lógica de varios algoritmos.</p>
            <p style='font-size:17px;'>
            <strong>¿Por qué importa?</strong><br>
            Las enfermedades del corazón son la causa #1 de muerte en el mundo 🌎. Anticipar el riesgo permite tomar acciones preventivas antes de que aparezcan síntomas.
            </p>
            <blockquote style='font-size:15px;'>❗ Nota: Este sistema no reemplaza una consulta médica. Es una herramienta educativa y preventiva.</blockquote>
            </div>
        """, unsafe_allow_html=True)
    with colB:
        st.image("imagen_logo.png", width=300)

    if st.button("🧪 Iniciar Evaluación", key="iniciar"):
        st.session_state.formulario = True
        st.rerun()
    st.stop()

# SIDEBAR GLOSARIO
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
            <li><b>Presión arterial en reposo:</b> Valor de presión cuando el paciente está en descanso (mmHg).</li>
            <li><b>Colesterol sérico:</b> Cantidad de colesterol total en sangre (mg/dL).</li>
            <li><b>¿Azúcar en ayunas > 120?:</b> Nivel alto de glucosa antes del desayuno.</li>
            <li><b>Electrocardiograma en reposo:</b> Evaluación de la actividad eléctrica del corazón sin esfuerzo.</li>
            <li><b>Frecuencia cardíaca máxima:</b> Máximo de pulsaciones por minuto en una prueba de esfuerzo.</li>
            <li><b>¿Angina inducida?:</b> Dolor en el pecho causado por el ejercicio.</li>
            <li><b>Oldpeak:</b> Descenso del segmento ST observado durante ejercicio (riesgo de isquemia).</li>
            <li><b>Pendiente del ST:</b> Forma que toma la curva ST en el ECG, indica el pronóstico.</li>
            <li><b>Número de vasos mayores:</b> Vasos sanguíneos principales detectados en angiografía (0 a 3).</li>
        </ul>
        """, unsafe_allow_html=True)

# FORMULARIO
with st.container():
    st.markdown("<div class='fade-transition form-container'>", unsafe_allow_html=True)
    st.title("🧾 Formulario de Evaluación")

col1, col2, col3 = st.columns(3)
with col1:
    edad = st.number_input("Edad", min_value=1, max_value=120, step=1)
    genero = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    dolor_pecho = st.selectbox("Tipo de dolor en el pecho", [0, 1, 2, 3])
    presion = st.number_input("Presión arterial en reposo", min_value=80, max_value=200)

with col2:
    colesterol = st.number_input("Colesterol sérico", min_value=100, max_value=600)
    azucar = st.radio("¿Azúcar en ayunas > 120?", [0, 1], format_func=lambda x: f"{'✅ Sí' if x == 1 else '❌ No'}")
    electro = st.selectbox("Electrocardiograma en reposo", [0, 1, 2])
    frecuencia = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=250)

with col3:
    angina = st.radio("¿Angina inducida por ejercicio?", [0, 1], format_func=lambda x: f"{'✅ Sí' if x == 1 else '❌ No'}")
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)
    pendiente = st.selectbox("Pendiente del ST", [0, 1, 2])
    vasos = st.selectbox("Número de vasos mayores", [0, 1, 2, 3])

st.markdown("</div>", unsafe_allow_html=True)

# PREDICCIÓN
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

