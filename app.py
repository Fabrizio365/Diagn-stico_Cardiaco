import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------ ENTRENAMIENTO DIRECTo ------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

X, y = cargar_datos()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# Modelos base
dt = DecisionTreeClassifier(max_depth=20, criterion='gini', min_samples_leaf=5, splitter='random')
knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
rf = RandomForestClassifier(n_estimators=25, criterion='gini', max_features='sqrt')

modelo = VotingClassifier(estimators=[('dt', dt), ('knn', knn), ('rf', rf)], voting='hard')
modelo.fit(X_train, y_train)

# --------------------------- INTERFAZ STREAMLIT ---------------------------
st.set_page_config(page_title="Predicción Cardiovascular", layout="wide")

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
    </style>
""", unsafe_allow_html=True)

if "formulario" not in st.session_state:
    st.session_state.formulario = False

if not st.session_state.formulario:
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("""
            <div class='fade-in' style='background: linear-gradient(to right, #ff4d4d, #ff9999); padding: 2rem; border-radius: 1rem; box-shadow: 0 0 10px rgba(0,0,0,0.2); color: white;'>
            <h2 style='text-shadow: 2px 2px 4px #000000;'>🫀 Bienvenido a la Evaluación de Riesgo Cardiovascular</h2>
            <p><strong>Nuestro Aporte:</strong><br>
            Se empleará una combinación de tres de los algoritmos más efectivos (<b>Árbol de Decisión</b>, <b>K-Nearest Neighbors</b> y <b>Random Forest</b>) junto con una técnica de ensamblaje (<em>Ensemble</em>) para optimizar sus fortalezas y mitigar los posibles sesgos o variaciones inherentes a cada modelo individual.</p>
            <p>La técnica de ensamblaje seleccionada será la <b>Votación Mayoritaria (Hard Voting)</b>, que ofrece la ventaja de reducir el riesgo de sobreajuste asociado a un único modelo, particularmente relevante en el caso del Árbol de Decisión, que podría alcanzar una precisión del 100%, lo que indicaría un posible sobreajuste.</p>
            <blockquote>Las enfermedades cardíacas son la principal causa de muerte a nivel mundial. Nuestro objetivo es acercar la ciencia y la inteligencia artificial a la salud preventiva.</blockquote>
            </div>
        """, unsafe_allow_html=True)
    with colB:
        st.image("imagen_logo.png", width=300)

    st.markdown("""
    <style>
        .start-button {
            animation: fadeIn 1s ease-in-out;
        }
    </style>
    <div class="start-button">
    """, unsafe_allow_html=True)
    if st.button("🧪 Iniciar Evaluación", key="iniciar"):
        st.session_state.formulario = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Sidebar con info
with st.sidebar:
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    st.image("imagen_logo.png")
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("ℹ️ Información de las variables", expanded=False):
        st.markdown("Edad, Género, Tipo de dolor en el pecho, Presión, Colesterol, Azúcar en ayunas, ECG, Frecuencia máxima, Angina, Oldpeak, Pendiente del ST, Vasos, Tipo de talasemia")

# Formulario
with st.container():
    st.markdown("<div class='fade-transition'>", unsafe_allow_html=True)
    st.title("🧾 Formulario de Evaluación")

col1, col2, col3 = st.columns(3)
with col1:
    edad = st.number_input("Edad", min_value=1, max_value=120, step=1)
    genero = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    dolor_pecho = st.selectbox("Tipo de dolor en el pecho", [0,1,2,3])
    presion = st.number_input("Presión arterial en reposo", min_value=80, max_value=200)

with col2:
    colesterol = st.number_input("Nivel de colesterol sérico", min_value=100, max_value=600)
    azucar = st.selectbox("¿Azúcar en ayunas > 120?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    electro = st.selectbox("Electrocardiograma en reposo", [0,1,2])
    frecuencia = st.number_input("Frecuencia cardiaca máxima", min_value=60, max_value=250)

with col3:
    angina = st.selectbox("¿Angina inducida por ejercicio?", [0,1], format_func=lambda x: "No" if x == 0 else "Sí")
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)
    pendiente = st.selectbox("Pendiente del ST", [0,1,2])
    vasos = st.selectbox("Número de vasos mayores", [0,1,2,3])
    thal = st.selectbox("Tipo de talasemia (Thal)", [0, 1, 2, 3])

st.markdown("</div>", unsafe_allow_html=True)

# Predicción
if st.button("🔍 Predecir estado de salud"):
    campos = [edad, genero, dolor_pecho, presion, colesterol, azucar, electro, frecuencia, angina, oldpeak, pendiente, vasos, thal]
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

