import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ------------------------ FUNCIONES DE DATOS ------------------------
@st.cache_data
def cargar_datos_default():
    """Carga el dataset por defecto"""
    try:
        df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
        X = df.drop(["target", "patientid"], axis=1)
        y = df["target"]
        return X, y, df
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def procesar_csv_subido(uploaded_file):
    """Procesa el archivo CSV subido por el usuario"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Verificar que exista la columna target
        if 'target' not in df.columns:
            st.error("El archivo CSV debe contener una columna llamada 'target'")
            return None, None, None
            
        # Remover columnas no numéricas o identificadores
        cols_to_drop = ['target']
        if 'patientid' in df.columns:
            cols_to_drop.append('patientid')
            
        X = df.drop(cols_to_drop, axis=1)
        y = df['target']
        
        # Verificar que todas las columnas sean numéricas
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            st.warning(f"Se encontraron columnas no numéricas que serán omitidas: {non_numeric_cols}")
            X = X.select_dtypes(include=[np.number])
            
        return X, y, df
    except Exception as e:
        st.error(f"Error al procesar el archivo CSV: {str(e)}")
        return None, None, None

@st.cache_data
def entrenar_modelo(X, y):
    """Entrena el modelo de Hard Voting con los tres algoritmos especificados"""
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir los modelos base
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3,
        random_state=42
    )
    
    random_forest = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    svm_model = SVC(
        kernel='rbf', 
        C=1.0, 
        gamma='scale',
        probability=True,  # Necesario para soft voting si se requiere
        random_state=42
    )
    
    # Crear el ensemble con Hard Voting
    voting_classifier = VotingClassifier(
        estimators=[
            ('gradient_boosting', gradient_boosting),
            ('random_forest', random_forest),
            ('svm', svm_model)
        ],
        voting='hard'
    )
    
    # Entrenar el modelo
    voting_classifier.fit(X_train_scaled, y_train)
    
    # Evaluar el modelo
    y_pred = voting_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Obtener métricas individuales
    individual_scores = {}
    for name, model in [('Gradient Boosting', gradient_boosting), 
                       ('Random Forest', random_forest), 
                       ('SVM', svm_model)]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        individual_scores[name] = accuracy_score(y_test, pred)
    
    return voting_classifier, scaler, accuracy, individual_scores, y_test, y_pred

# --------------------------- INTERFAZ STREAMLIT ---------------------------
st.set_page_config(page_title="Predicción Cardiovascular - Hard Voting", layout="wide")

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
        .metric-container {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;700&display=swap" rel="rel">
""", unsafe_allow_html=True)

# Inicializar estado
if "modelo_entrenado" not in st.session_state:
    st.session_state.modelo_entrenado = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "formulario" not in st.session_state:
    st.session_state.formulario = False

# PANTALLA DE INICIO
if not st.session_state.formulario:
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("""
            <div class='fade-in' style='padding: 2rem; border-radius: 1rem; box-shadow: 0 0 10px rgba(0,0,0,0.2);'>
            <h2 class='rgb-text'>🫀 Sistema de Predicción Cardiovascular con Hard Voting</h2>
            <p style='font-size:18px;'>
            Este sistema utiliza <strong>Hard Voting</strong> combinando tres algoritmos de machine learning para evaluar el riesgo cardiovascular.
            </p>
            <p><strong>Algoritmos utilizados:</strong><br>
            📊 <b>Gradient Boosting</b>: Optimiza errores secuencialmente<br>
            🌲 <b>Random Forest</b>: Combina múltiples árboles de decisión<br>
            🎯 <b>Support Vector Machine (SVM)</b>: Encuentra el hiperplano óptimo<br>
            🗳️ <b>Hard Voting</b>: Decisión por mayoría de votos</p>
            
            <p style='font-size:17px;'>
            <strong>🔄 Nuevas funcionalidades:</strong><br>
            • Entrena con tu propio dataset CSV<br>
            • Métricas detalladas de rendimiento<br>
            • Comparación entre algoritmos individuales
            </p>
            <blockquote style='font-size:15px;'>❗ Herramienta educativa. No reemplaza consulta médica profesional.</blockquote>
            </div>
        """, unsafe_allow_html=True)
    
    with colB:
        try:
            st.image("imagen_logo.png", width=300)
        except:
            st.markdown("🏥 **Logo del Sistema**")

    # SECCIÓN DE CARGA DE DATOS
    st.markdown("---")
    st.subheader("📁 Configuración de Datos")
    
    option = st.radio(
        "Selecciona la fuente de datos:",
        ["Usar dataset por defecto", "Subir mi propio archivo CSV"]
    )
    
    if option == "Subir mi propio archivo CSV":
        uploaded_file = st.file_uploader(
            "Sube tu archivo CSV", 
            type=['csv'],
            help="El archivo debe contener una columna 'target' con valores 0 (saludable) y 1 (riesgo cardiovascular)"
        )
        
        if uploaded_file is not None:
            with st.expander("📋 Requisitos del archivo CSV"):
                st.markdown("""
                - **Columna obligatoria**: `target` (0 = saludable, 1 = riesgo)
                - **Columnas opcionales a omitir**: `patientid` o similares
                - **Todas las demás columnas deben ser numéricas**
                - **Formato**: CSV separado por comas
                
                **Ejemplo de estructura:**
                ```
                edad,genero,presion,colesterol,target
                45,1,120,200,0
                60,0,140,250,1
                ```
                """)
            
            # Procesar archivo subido
            X, y, df = procesar_csv_subido(uploaded_file)
            
            if X is not None and y is not None:
                st.success("✅ Archivo CSV cargado correctamente")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total de muestras", len(df))
                with col2:
                    st.metric("📈 Características", X.shape[1])
                with col3:
                    risk_percentage = (y.sum() / len(y)) * 100
                    st.metric("⚠️ Casos de riesgo", f"{risk_percentage:.1f}%")
                
                # Vista previa de los datos
                with st.expander("👁️ Vista previa de los datos"):
                    st.dataframe(df.head())
                
                # Entrenar modelo
                if st.button("🚀 Entrenar Modelo con Datos Subidos"):
                    with st.spinner("Entrenando modelo..."):
                        try:
                            modelo, scaler, accuracy, individual_scores, y_test, y_pred = entrenar_modelo(X, y)
                            
                            st.session_state.modelo_entrenado = modelo
                            st.session_state.scaler = scaler
                            
                            # Mostrar resultados del entrenamiento
                            st.success(f"✅ Modelo entrenado exitosamente - Precisión: {accuracy:.2%}")
                            
                            # Métricas individuales
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"""
                                <div class='metric-container'>
                                <h4>🚀 Gradient Boosting</h4>
                                <h2>{individual_scores['Gradient Boosting']:.2%}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"""
                                <div class='metric-container'>
                                <h4>🌲 Random Forest</h4>
                                <h2>{individual_scores['Random Forest']:.2%}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            with col3:
                                st.markdown(f"""
                                <div class='metric-container'>
                                <h4>🎯 SVM</h4>
                                <h2>{individual_scores['SVM']:.2%}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Matriz de confusión
                            cm = confusion_matrix(y_test, y_pred)
                            
                            # Crear matriz de confusión con Streamlit
                            st.markdown("### 📊 Matriz de Confusión - Hard Voting")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"""
                                <div class='metric-container'>
                                <h4>✅ Verdaderos Negativos</h4>
                                <h2>{cm[0,0]}</h2>
                                <small>Predicciones correctas: Saludable</small>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div class='metric-container'>
                                <h4>❌ Falsos Positivos</h4>
                                <h2>{cm[0,1]}</h2>
                                <small>Predicciones incorrectas: Riesgo</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class='metric-container'>
                                <h4>❌ Falsos Negativos</h4>
                                <h2>{cm[1,0]}</h2>
                                <small>Predicciones incorrectas: Saludable</small>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div class='metric-container'>
                                <h4>✅ Verdaderos Positivos</h4>
                                <h2>{cm[1,1]}</h2>
                                <small>Predicciones correctas: Riesgo</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error durante el entrenamiento: {str(e)}")
        else:
            st.info("👆 Sube un archivo CSV para continuar")
    
    else:
        # Usar dataset por defecto
        X, y, df = cargar_datos_default()
        
        if X is not None and y is not None:
            st.success("✅ Dataset por defecto cargado")
            
            if st.button("🚀 Entrenar Modelo con Dataset Por Defecto"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        modelo, scaler, accuracy, individual_scores, y_test, y_pred = entrenar_modelo(X, y)
                        
                        st.session_state.modelo_entrenado = modelo
                        st.session_state.scaler = scaler
                        
                        st.success(f"✅ Modelo entrenado - Precisión: {accuracy:.2%}")
                        
                        # Mostrar métricas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🚀 Gradient Boosting", f"{individual_scores['Gradient Boosting']:.2%}")
                        with col2:
                            st.metric("🌲 Random Forest", f"{individual_scores['Random Forest']:.2%}")
                        with col3:
                            st.metric("🎯 SVM", f"{individual_scores['SVM']:.2%}")
                            
                    except Exception as e:
                        st.error(f"Error durante el entrenamiento: {str(e)}")
        else:
            st.warning("⚠️ No se pudo cargar el dataset por defecto. Por favor, sube tu propio archivo CSV.")
    
    # Botón para continuar al formulario
    if st.session_state.modelo_entrenado is not None:
        if st.button("🧪 Continuar a Evaluación Individual", key="iniciar"):
            st.session_state.formulario = True
            st.rerun()
    else:
        st.info("👆 Primero entrena un modelo para continuar")
    
    st.stop()

# SIDEBAR GLOSARIO
with st.sidebar:
    try:
        st.image("imagen_logo.png")
    except:
        st.markdown("🏥 **Sistema de Predicción**")
        
    st.markdown("### 🤖 Hard Voting Ensemble")
    st.markdown("""
    **Algoritmos utilizados:**
    - 🚀 **Gradient Boosting**
    - 🌲 **Random Forest** 
    - 🎯 **SVM**
    
    **Hard Voting:** Cada algoritmo vota por una clase, gana la mayoría.
    """)
    
    with st.expander("ℹ️ Glosario de variables", expanded=False):
        st.markdown("""
        <ul style='font-size: 15px; line-height: 1.6;'>
            <li><b>Edad:</b> Años del paciente</li>
            <li><b>Género:</b> 0=Mujer, 1=Hombre</li>
            <li><b>Tipo de dolor pecho:</b>
                <ul>
                    <li>0: Angina típica</li>
                    <li>1: Angina atípica</li>
                    <li>2: Dolor no anginoso</li>
                    <li>3: Asintomático</li>
                </ul>
            </li>
            <li><b>Presión arterial:</b> mmHg en reposo</li>
            <li><b>Colesterol:</b> mg/dL en sangre</li>
            <li><b>Azúcar en ayunas:</b> >120 mg/dL</li>
            <li><b>ECG reposo:</b> 0=Normal, 1=Anormalidad ST, 2=Hipertrofia</li>
            <li><b>Frecuencia máxima:</b> Pulsaciones/minuto</li>
            <li><b>Angina inducida:</b> Por ejercicio</li>
            <li><b>Oldpeak:</b> Depresión ST ejercicio vs reposo</li>
            <li><b>Pendiente ST:</b> 0=Ascendente, 1=Plana, 2=Descendente</li>
            <li><b>Vasos mayores:</b> 0-3 coloreados por fluoroscopia</li>
        </ul>
        """, unsafe_allow_html=True)

# FORMULARIO DE PREDICCIÓN
if st.session_state.modelo_entrenado is not None:
    with st.container():
        st.markdown("<div class='fade-transition form-container'>", unsafe_allow_html=True)
        st.title("🧾 Formulario de Evaluación Individual")
        st.markdown("*Completa todos los campos para obtener una predicción*")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.number_input("Edad", min_value=1, max_value=120, step=1, value=50)
        genero = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
        dolor_pecho = st.selectbox("Tipo de dolor en el pecho", [0, 1, 2, 3])
        presion = st.number_input("Presión arterial en reposo (mmHg)", min_value=80, max_value=200, value=120)

    with col2:
        colesterol = st.number_input("Colesterol sérico (mg/dL)", min_value=100, max_value=600, value=200)
        azucar = st.radio("¿Azúcar en ayunas > 120?", [0, 1], format_func=lambda x: f"{'✅ Sí' if x == 1 else '❌ No'}")
        electro = st.selectbox("Electrocardiograma en reposo", [0, 1, 2])
        frecuencia = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=250, value=150)

    with col3:
        angina = st.radio("¿Angina inducida por ejercicio?", [0, 1], format_func=lambda x: f"{'✅ Sí' if x == 1 else '❌ No'}")
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
        pendiente = st.selectbox("Pendiente del ST", [0, 1, 2])
        vasos = st.selectbox("Número de vasos mayores", [0, 1, 2, 3])

    st.markdown("</div>", unsafe_allow_html=True)

    # PREDICCIÓN
    if st.button("🔍 Realizar Predicción con Hard Voting"):
        campos = [edad, genero, dolor_pecho, presion, colesterol, azucar, electro, frecuencia, angina, oldpeak, pendiente, vasos]
        
        if all(v is not None for v in campos):
            entrada = np.array([campos])
            
            try:
                # Escalar la entrada
                entrada_escalada = st.session_state.scaler.transform(entrada)
                
                # Realizar predicción
                resultado = st.session_state.modelo_entrenado.predict(entrada_escalada)[0]
                
                # Obtener predicciones individuales
                predicciones_individuales = {}
                for name, estimator in st.session_state.modelo_entrenado.named_estimators_.items():
                    pred = estimator.predict(entrada_escalada)[0]
                    predicciones_individuales[name] = pred
                
                st.markdown("<div class='fade-transition'>", unsafe_allow_html=True)
                
                # Mostrar resultado principal
                if resultado == 1:
                    st.error("⚠️ **RIESGO CARDIOVASCULAR DETECTADO**")
                    try:
                        st.image("riesgo.png", width=150)
                    except:
                        st.markdown("🚨 **Imagen de riesgo**")
                else:
                    st.success("✅ **ESTADO CARDIOVASCULAR SALUDABLE**")
                    try:
                        st.image("saludable.png", width=150)
                    except:
                        st.markdown("💚 **Imagen saludable**")
                
                # Mostrar desglose de votación
                st.markdown("### 🗳️ Desglose de Votación (Hard Voting)")
                
                col1, col2, col3 = st.columns(3)
                
                gb_vote = predicciones_individuales['gradient_boosting']
                rf_vote = predicciones_individuales['random_forest']  
                svm_vote = predicciones_individuales['svm']
                
                with col1:
                    vote_color = "🔴" if gb_vote == 1 else "🟢"
                    st.markdown(f"""
                    <div class='metric-container'>
                    <h4>🚀 Gradient Boosting</h4>
                    <h2>{vote_color} {"Riesgo" if gb_vote == 1 else "Saludable"}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    vote_color = "🔴" if rf_vote == 1 else "🟢"
                    st.markdown(f"""
                    <div class='metric-container'>
                    <h4>🌲 Random Forest</h4>
                    <h2>{vote_color} {"Riesgo" if rf_vote == 1 else "Saludable"}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    vote_color = "🔴" if svm_vote == 1 else "🟢"
                    st.markdown(f"""
                    <div class='metric-container'>
                    <h4>🎯 SVM</h4>
                    <h2>{vote_color} {"Riesgo" if svm_vote == 1 else "Saludable"}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Resumen de votos
                votos_riesgo = sum([gb_vote, rf_vote, svm_vote])
                st.markdown(f"""
                ### 📊 Resumen de Votación:
                - **Votos por "Riesgo":** {votos_riesgo}/3
                - **Votos por "Saludable":** {3-votos_riesgo}/3
                - **Decisión Final:** {"🔴 RIESGO" if resultado == 1 else "🟢 SALUDABLE"} (por mayoría)
                """)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error al realizar la predicción: {str(e)}")
        else:
            st.warning("⚠️ Por favor completa todos los campos antes de predecir.")

    # Botón para volver al inicio
    if st.button("🏠 Volver al Inicio"):
        st.session_state.formulario = False
        st.rerun()

else:
    st.error("⚠️ No hay modelo entrenado. Por favor vuelve al inicio y entrena un modelo primero.")
    if st.button("🏠 Volver al Inicio"):
        st.session_state.formulario = False
        st.rerun()
