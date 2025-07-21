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
        probability=True,
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

# --------------------------- CONFIGURACIÓN STREAMLIT ---------------------------
st.set_page_config(
    page_title="🫀 CardioPredict AI: Machine Learning para la Salud del Corazón", 
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PERSONALIZADO MEJORADO
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Header principal con gradiente animado */
        .hero-header {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .hero-subtitle {
            font-size: 1.4rem;
            font-weight: 300;
            opacity: 0.95;
            margin-bottom: 2rem;
        }
        
        /* Cards de algoritmos */
        .algorithm-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .algorithm-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            background: rgba(255, 255, 255, 0.2);
        }
        
        .algorithm-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .algorithm-card:hover::before {
            left: 100%;
        }
        
        /* Métricas mejoradas */
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
        }
        
        .metric-card:hover {
            transform: scale(1.05);
            background: rgba(255, 255, 255, 0.2);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0.5rem 0;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
            font-weight: 500;
        }
        
        /* Formulario estilizado */
        .form-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        /* Botones mejorados */
        .stButton > button {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        /* Sidebar personalizada */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
        }
        
        /* Alertas personalizadas */
        .success-alert {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            color: #2d3748;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            font-weight: 500;
            box-shadow: 0 4px 20px rgba(132, 250, 176, 0.3);
        }
        
        .error-alert {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            color: #2d3748;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            font-weight: 500;
            box-shadow: 0 4px 20px rgba(255, 154, 158, 0.3);
        }
        
        /* Animaciones */
        .fade-in {
            animation: fadeInUp 0.8s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        /* Resultado de predicción */
        .prediction-result {
            text-align: center;
            padding: 2rem;
            border-radius: 20px;
            margin: 2rem 0;
            font-size: 1.5rem;
            font-weight: 600;
            animation: bounceIn 0.8s ease-out;
        }
        
        @keyframes bounceIn {
            0% { transform: scale(0.3); opacity: 0; }
            50% { transform: scale(1.05); }
            70% { transform: scale(0.9); }
            100% { transform: scale(1); opacity: 1; }
        }
        
        .healthy-result {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            color: #2d3748;
            box-shadow: 0 10px 30px rgba(132, 250, 176, 0.4);
        }
        
        .risk-result {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            color: #2d3748;
            box-shadow: 0 10px 30px rgba(255, 154, 158, 0.4);
        }
        
        /* Voting breakdown */
        .vote-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .vote-card:hover {
            transform: translateY(-3px);
            background: rgba(255, 255, 255, 0.25);
        }
        
        .vote-emoji {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .hero-title { font-size: 2.5rem; }
            .hero-subtitle { font-size: 1.2rem; }
            .metric-value { font-size: 2rem; }
        }
    </style>
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
    # HEADER PRINCIPAL
    st.markdown("""
        <div class="hero-header fade-in">
            <div class="hero-title">🫀 CardioPredict AI</div>
            <div class="hero-subtitle">Machine Learning para la Salud del Corazón</div>
            <p style="font-size: 1.1rem; opacity: 0.9; max-width: 800px; margin: 0 auto;">
                Tecnología de vanguardia que combina <strong>Gradient Boosting</strong>, <strong>Random Forest</strong> 
                y <strong>SVM</strong> mediante <em>Hard Voting</em> para detectar riesgos cardiovasculares con precisión médica.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # ALGORITMOS Y CARACTERÍSTICAS
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="fade-in">
                <h2 style="margin-bottom: 2rem;">🚀 Tecnología de Ensemble Learning</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Cards de algoritmos
        algorithms = [
            {
                "icon": "🌲",
                "name": "Random Forest",
                "description": "Combina múltiples árboles de decisión para reducir el overfitting y mejorar la generalización",
                "benefits": "• Robusto ante datos ruidosos<br>• Maneja características categóricas<br>• Proporciona importancia de variables"
            },
            {
                "icon": "🚀", 
                "name": "Gradient Boosting",
                "description": "Optimiza errores secuencialmente, construyendo modelos que corrigen las predicciones anteriores",
                "benefits": "• Alta precisión predictiva<br>• Excelente para datos estructurados<br>• Manejo automático de valores faltantes"
            },
            {
                "icon": "🎯",
                "name": "Support Vector Machine",
                "description": "Encuentra el hiperplano óptimo que mejor separa las clases en espacios de alta dimensión",
                "benefits": "• Efectivo en espacios de alta dimensión<br>• Memoria eficiente<br>• Versátil con diferentes kernels"
            }
        ]
        
        for i, algo in enumerate(algorithms):
            st.markdown(f"""
                <div class="algorithm-card fade-in" style="animation-delay: {i*0.2}s;">
                    <h3>{algo['icon']} {algo['name']}</h3>
                    <p style="margin-bottom: 1rem; opacity: 0.9;">{algo['description']}</p>
                    <div style="font-size: 0.9rem; opacity: 0.8;">{algo['benefits']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Usar imagen o emoji como fallback
        try:
            st.image("imagen_logo.png", width=350)
        except:
            # Crear un logo visual simple con componentes nativos
            st.markdown("### 🏥")
            st.markdown("#### CardioPredict AI")
            st.markdown("**Machine Learning para la Salud del Corazón**")
    
    # SECCIÓN DE HARD VOTING - Usando componentes nativos
    st.markdown("---")
    st.markdown("## 🗳️ ¿Cómo Funciona el Hard Voting?")
    
    # Crear tres columnas para los algoritmos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌲")
        st.markdown("**Random Forest**")
        st.markdown("Voto: Riesgo/Saludable")
    
    with col2:
        st.markdown("### 🚀") 
        st.markdown("**Gradient Boosting**")
        st.markdown("Voto: Riesgo/Saludable")
    
    with col3:
        st.markdown("### 🎯")
        st.markdown("**SVM**")
        st.markdown("Voto: Riesgo/Saludable")
    
    st.markdown("---")
    
    # Explicación de la decisión por mayoría
    st.markdown("### ⚖️ Decisión por Mayoría")
    st.markdown("""
    La predicción final se basa en el voto de la mayoría de los tres algoritmos, 
    asegurando mayor confiabilidad y reduciendo errores individuales.
    """)

    # SECCIÓN DE CARGA DE DATOS
    st.markdown("---")
    st.markdown("""
        <div class="fade-in">
            <h2 style="text-align: center; margin-bottom: 2rem;">📊 Configuración de Datos</h2>
        </div>
    """, unsafe_allow_html=True)
    
    option = st.radio(
        "**Selecciona la fuente de datos:**",
        ["🗂️ Usar dataset cardiovascular por defecto", "📁 Subir mi propio archivo CSV"],
        help="El dataset por defecto contiene datos clínicos validados. También puedes usar tus propios datos."
    )
    
    if option == "📁 Subir mi propio archivo CSV":
        uploaded_file = st.file_uploader(
            "**Sube tu archivo CSV**", 
            type=['csv'],
            help="El archivo debe contener una columna 'target' con valores 0 (saludable) y 1 (riesgo cardiovascular)"
        )
        
        if uploaded_file is not None:
                with st.expander("📋 **Requisitos del archivo CSV**", expanded=True):
                    st.markdown("#### 📝 Estructura Requerida:")
                    
                    st.markdown("**Columna obligatoria:** `target` (0 = saludable, 1 = riesgo)")
                    st.markdown("**Columnas opcionales a omitir:** `patientid` o identificadores similares")
                    st.markdown("**Todas las demás columnas deben ser numéricas**")
                    st.markdown("**Formato:** CSV separado por comas")
                    
                    st.markdown("#### 💡 Ejemplo de estructura:")
                    st.code("""
edad,genero,presion,colesterol,target
45,1,120,200,0
60,0,140,250,1
52,1,130,180,0
                    """, language="csv")
            
            # Procesar archivo subido
            X, y, df = procesar_csv_subido(uploaded_file)
            
            if X is not None and y is not None:
                st.markdown("""
                    <div class="success-alert">
                        ✅ <strong>Archivo CSV cargado correctamente</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                # Métricas del dataset
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{len(df)}</div>
                            <div class="metric-label">📊 Total Muestras</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{X.shape[1]}</div>
                            <div class="metric-label">📈 Características</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    risk_percentage = (y.sum() / len(y)) * 100
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{risk_percentage:.1f}%</div>
                            <div class="metric-label">⚠️ Casos Riesgo</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col4:
                    healthy_percentage = ((len(y) - y.sum()) / len(y)) * 100
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{healthy_percentage:.1f}%</div>
                            <div class="metric-label">✅ Casos Saludables</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Vista previa de los datos
                with st.expander("👁️ **Vista previa de los datos**"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Entrenar modelo
                if st.button("🚀 **Entrenar Modelo con Datos Subidos**", key="train_uploaded"):
                    with st.spinner("🔄 Entrenando modelo con Hard Voting..."):
                        try:
                            modelo, scaler, accuracy, individual_scores, y_test, y_pred = entrenar_modelo(X, y)
                            
                            st.session_state.modelo_entrenado = modelo
                            st.session_state.scaler = scaler
                            
                            # Resultado del entrenamiento
                            st.markdown(f"""
                                <div class="success-alert">
                                    🎉 <strong>Modelo entrenado exitosamente</strong><br>
                                    <span style="font-size: 1.2rem;">Precisión del Ensemble: {accuracy:.2%}</span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Métricas individuales
                            st.markdown("### 🏆 Rendimiento Individual vs Ensemble")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown(f"""
                                    <div class="vote-card">
                                        <span class="vote-emoji">🚀</span>
                                        <h4>Gradient Boosting</h4>
                                        <div class="metric-value">{individual_scores['Gradient Boosting']:.2%}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                            with col2:
                                st.markdown(f"""
                                    <div class="vote-card">
                                        <span class="vote-emoji">🌲</span>
                                        <h4>Random Forest</h4>
                                        <div class="metric-value">{individual_scores['Random Forest']:.2%}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                            with col3:
                                st.markdown(f"""
                                    <div class="vote-card">
                                        <span class="vote-emoji">🎯</span>
                                        <h4>SVM</h4>
                                        <div class="metric-value">{individual_scores['SVM']:.2%}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                            with col4:
                                st.markdown(f"""
                                    <div class="vote-card" style="background: rgba(255,215,0,0.2);">
                                        <span class="vote-emoji">🏆</span>
                                        <h4>Hard Voting</h4>
                                        <div class="metric-value" style="color: #FFD700;">{accuracy:.2%}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Matriz de confusión
                            cm = confusion_matrix(y_test, y_pred)
                            
                            st.markdown("### 📊 Matriz de Confusión - Hard Voting")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"""
                                    <div class="vote-card">
                                        <span class="vote-emoji">✅</span>
                                        <h4>Verdaderos Negativos</h4>
                                        <div class="metric-value">{cm[0,0]}</div>
                                        <small>Predicciones correctas: Saludable</small>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div class="vote-card">
                                        <span class="vote-emoji">❌</span>
                                        <h4>Falsos Positivos</h4>
                                        <div class="metric-value">{cm[0,1]}</div>
                                        <small>Predicciones incorrectas: Riesgo</small>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                    <div class="vote-card">
                                        <span class="vote-emoji">❌</span>
                                        <h4>Falsos Negativos</h4>
                                        <div class="metric-value">{cm[1,0]}</div>
                                        <small>Predicciones incorrectas: Saludable</small>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div class="vote-card">
                                        <span class="vote-emoji">✅</span>
                                        <h4>Verdaderos Positivos</h4>
                                        <div class="metric-value">{cm[1,1]}</div>
                                        <small>Predicciones correctas: Riesgo</small>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.markdown(f"""
                                <div class="error-alert">
                                    ❌ <strong>Error durante el entrenamiento:</strong> {str(e)}
                                </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("👆 **Sube un archivo CSV para continuar con el entrenamiento personalizado**")
    
    else:
        # Usar dataset por defecto
        X, y, df = cargar_datos_default()
        
        if X is not None and y is not None:
            st.markdown("""
                <div class="success-alert">
                    ✅ <strong>Dataset cardiovascular por defecto cargado correctamente</strong>
                </div>
            """, unsafe_allow_html=True)
            
            # Métricas del dataset por defecto
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(df)}</div>
                        <div class="metric-label">📊 Total Muestras</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{X.shape[1]}</div>
                        <div class="metric-label">📈 Características</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col3:
                risk_percentage = (y.sum() / len(y)) * 100
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{risk_percentage:.1f}%</div>
                        <div class="metric-label">⚠️ Casos Riesgo</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col4:
                healthy_percentage = ((len(y) - y.sum()) / len(y)) * 100
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{healthy_percentage:.1f}%</div>
                        <div class="metric-label">✅ Casos Saludables</div>
                    </div>
                """, unsafe_allow_html=True)
            
            if st.button("🚀 **Entrenar Modelo con Dataset Por Defecto**", key="train_default"):
                with st.spinner("🔄 Entrenando modelo con Hard Voting..."):
                    try:
                        modelo, scaler, accuracy, individual_scores, y_test, y_pred = entrenar_modelo(X, y)
                        
                        st.session_state.modelo_entrenado = modelo
                        st.session_state.scaler = scaler
                        
                        st.markdown(f"""
                            <div class="success-alert">
                                🎉 <strong>Modelo entrenado exitosamente</strong><br>
                                <span style="font-size: 1.2rem;">Precisión del Ensemble: {accuracy:.2%}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Métricas de rendimiento
                        st.markdown("### 🏆 Rendimiento Individual vs Ensemble")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                                <div class="vote-card">
                                    <span class="vote-emoji">🚀</span>
                                    <h4>Gradient Boosting</h4>
                                    <div class="metric-value">{individual_scores['Gradient Boosting']:.2%}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown(f"""
                                <div class="vote-card">
                                    <span class="vote-emoji">🌲</span>
                                    <h4>Random Forest</h4>
                                    <div class="metric-value">{individual_scores['Random Forest']:.2%}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        with col3:
                            st.markdown(f"""
                                <div class="vote-card">
                                    <span class="vote-emoji">🎯</span>
                                    <h4>SVM</h4>
                                    <div class="metric-value">{individual_scores['SVM']:.2%}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        with col4:
                            st.markdown(f"""
                                <div class="vote-card" style="background: rgba(255,215,0,0.2);">
                                    <span class="vote-emoji">🏆</span>
                                    <h4>Hard Voting</h4>
                                    <div class="metric-value" style="color: #FFD700;">{accuracy:.2%}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.markdown(f"""
                            <div class="error-alert">
                                ❌ <strong>Error durante el entrenamiento:</strong> {str(e)}
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="error-alert">
                    ⚠️ <strong>No se pudo cargar el dataset por defecto.</strong><br>
                    Por favor, sube tu propio archivo CSV para continuar.
                </div>
            """, unsafe_allow_html=True)
    
    # Botón para continuar al formulario
    if st.session_state.modelo_entrenado is not None:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🧪 **Continuar a Evaluación Individual**", key="iniciar", use_container_width=True):
                st.session_state.formulario = True
                st.rerun()
    else:
        st.markdown("""
            <div style="text-align: center; padding: 2rem; margin: 2rem 0;">
                <h3>👆 Primero entrena un modelo para continuar</h3>
                <p style="opacity: 0.8;">Selecciona una fuente de datos y haz clic en "Entrenar Modelo"</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# SIDEBAR MEJORADA
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem; margin-bottom: 2rem; 
                    background: rgba(255,255,255,0.1); border-radius: 15px;">
            <h2 style="margin: 0;">🫀 CardioPredict AI</h2>
            <p style="margin: 0.5rem 0; opacity: 0.9; font-size: 0.9rem;">Machine Learning para la Salud del Corazón</p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        st.image("imagen_logo.png", use_container_width=True)
    except:
        st.markdown("""
            <div style="text-align: center; padding: 2rem; 
                        background: rgba(255,255,255,0.1); border-radius: 15px;">
                <div style="font-size: 4rem;">🏥</div>
                <p><strong>Logo del Sistema</strong></p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 Hard Voting Ensemble")
    
    st.markdown("**🎯 Algoritmos Utilizados:**")
    st.markdown("🚀 **Gradient Boosting** - Optimización secuencial de errores")
    st.markdown("🌲 **Random Forest** - Consenso de múltiples árboles de decisión")  
    st.markdown("🎯 **SVM** - Separación óptima mediante hiperplanos")
    
    st.markdown("**🗳️ Hard Voting:**")
    st.markdown("Cada algoritmo emite un voto binario. La decisión final se toma por mayoría simple, asegurando robustez y reduciendo errores individuales.")
    
    st.markdown("**Ejemplo de votación:**")
    st.markdown("🚀 Voto: Riesgo | 🌲 Voto: Saludable | 🎯 Voto: Riesgo")
    st.markdown("**Resultado:** 2 votos por 'Riesgo' → **Predicción: Riesgo**")
    
    with st.expander("ℹ️ **Glosario de Variables**", expanded=False):
        # Usar components nativos de Streamlit en lugar de HTML
        st.markdown("#### 📋 Variables del Formulario:")
        
        st.markdown("**👤 Edad:** Años del paciente (1-120)")
        
        st.markdown("**⚧ Género:**")
        st.markdown("• 0 = Femenino  \n• 1 = Masculino")
        
        st.markdown("**💔 Tipo de dolor en el pecho:**")
        st.markdown("• 0 = Angina típica (relacionada al esfuerzo)  \n• 1 = Angina atípica  \n• 2 = Dolor no anginoso  \n• 3 = Asintomático (sin dolor)")
        
        st.markdown("**🩺 Presión arterial:** Presión sistólica en reposo (mmHg)")
        
        st.markdown("**🧪 Colesterol:** Colesterol sérico total (mg/dL)")
        
        st.markdown("**🍯 Azúcar en ayunas:** Glucosa en sangre mayor a 120 mg/dL")
        
        st.markdown("**📈 ECG en reposo:**")
        st.markdown("• 0 = Normal  \n• 1 = Anormalidad ST-T  \n• 2 = Hipertrofia ventricular izquierda")
        
        st.markdown("**💓 Frecuencia máxima:** Pulsaciones máximas alcanzadas (bpm)")
        
        st.markdown("**⚡ Angina inducida:** Dolor torácico causado por ejercicio")
        
        st.markdown("**📉 Oldpeak:** Depresión del segmento ST (ejercicio vs reposo)")
        
        st.markdown("**📊 Pendiente del ST:**")
        st.markdown("• 0 = Ascendente (mejor pronóstico)  \n• 1 = Plana (pronóstico intermedio)  \n• 2 = Descendente (peor pronóstico)")
        
        st.markdown("**🔍 Vasos mayores:** Número de vasos principales coloreados por fluoroscopia (0-3)")
        
        st.info("💡 **Nota:** Todos estos valores son utilizados por los algoritmos de Machine Learning para generar una predicción basada en patrones aprendidos de datos médicos validados.")

# FORMULARIO DE PREDICCIÓN MEJORADO
if st.session_state.modelo_entrenado is not None:
    
    st.markdown("""
        <div class="hero-header fade-in">
            <div class="hero-title" style="font-size: 2.5rem;">🧾 Evaluación Individual</div>
            <div class="hero-subtitle">Completa todos los campos para obtener una predicción personalizada</div>
        </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="form-container fade-in">', unsafe_allow_html=True)
        
        # Formulario organizado en pestañas
        tab1, tab2, tab3 = st.tabs(["👤 **Datos Personales**", "🩺 **Datos Clínicos**", "📊 **Datos de Laboratorio**"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                edad = st.number_input("**Edad**", min_value=1, max_value=120, step=1, value=50, 
                                     help="Edad del paciente en años")
                genero = st.selectbox("**Género**", [0, 1], 
                                    format_func=lambda x: "👩 Femenino" if x == 0 else "👨 Masculino")
            with col2:
                dolor_pecho = st.selectbox("**Tipo de dolor en el pecho**", [0, 1, 2, 3],
                                         format_func=lambda x: {
                                             0: "💔 Angina típica",
                                             1: "💜 Angina atípica", 
                                             2: "💙 Dolor no anginoso",
                                             3: "💚 Asintomático"
                                         }[x])
                angina = st.radio("**¿Angina inducida por ejercicio?**", [0, 1], 
                                format_func=lambda x: f"{'✅ Sí' if x == 1 else '❌ No'}")
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                presion = st.number_input("**Presión arterial en reposo (mmHg)**", 
                                        min_value=80, max_value=200, value=120,
                                        help="Presión sistólica en reposo")
                frecuencia = st.number_input("**Frecuencia cardíaca máxima**", 
                                           min_value=60, max_value=250, value=150,
                                           help="Máxima frecuencia alcanzada durante ejercicio")
            with col2:
                electro = st.selectbox("**Electrocardiograma en reposo**", [0, 1, 2],
                                     format_func=lambda x: {
                                         0: "📈 Normal",
                                         1: "⚠️ Anormalidad ST-T",
                                         2: "🔴 Hipertrofia ventricular"
                                     }[x])
                pendiente = st.selectbox("**Pendiente del segmento ST**", [0, 1, 2],
                                       format_func=lambda x: {
                                           0: "📈 Ascendente",
                                           1: "➡️ Plana",
                                           2: "📉 Descendente"
                                       }[x])
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                colesterol = st.number_input("**Colesterol sérico (mg/dL)**", 
                                           min_value=100, max_value=600, value=200,
                                           help="Nivel de colesterol total en sangre")
                azucar = st.radio("**¿Azúcar en ayunas > 120 mg/dL?**", [0, 1], 
                                format_func=lambda x: f"{'🍯 Sí (>120)' if x == 1 else '✅ No (≤120)'}")
            with col2:
                oldpeak = st.number_input("**Oldpeak (Depresión ST)**", 
                                        min_value=0.0, max_value=10.0, step=0.1, value=0.0,
                                        help="Depresión del segmento ST inducida por ejercicio")
                vasos = st.selectbox("**Número de vasos mayores**", [0, 1, 2, 3],
                                   help="Vasos principales coloreados por fluoroscopia")

        st.markdown('</div>', unsafe_allow_html=True)

    # SECCIÓN DE PREDICCIÓN
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔬 **Realizar Diagnóstico con IA**", key="predict", use_container_width=True):
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
                    
                    # Mostrar resultado principal con animación
                    if resultado == 1:
                        st.markdown("""
                            <div class="prediction-result risk-result">
                                <div style="font-size: 4rem;">⚠️</div>
                                <h2>RIESGO CARDIOVASCULAR DETECTADO</h2>
                                <p style="font-size: 1.2rem; margin-top: 1rem;">
                                    Se recomienda consulta médica inmediata para evaluación detallada
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        try:
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col2:
                                st.image("riesgo.png", width=200)
                        except:
                            pass
                            
                    else:
                        st.markdown("""
                            <div class="prediction-result healthy-result">
                                <div style="font-size: 4rem;">✅</div>
                                <h2>ESTADO CARDIOVASCULAR SALUDABLE</h2>
                                <p style="font-size: 1.2rem; margin-top: 1rem;">
                                    Mantén tus hábitos saludables y realiza chequeos preventivos regulares
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        try:
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col2:
                                st.image("saludable.png", width=200)
                        except:
                            pass
                    
                    # Desglose detallado de votación
                    st.markdown("---")
                    st.markdown("""
                        <div class="fade-in">
                            <h2 style="text-align: center; margin-bottom: 2rem;">🗳️ Análisis Detallado de Votación</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    gb_vote = predicciones_individuales['gradient_boosting']
                    rf_vote = predicciones_individuales['random_forest']  
                    svm_vote = predicciones_individuales['svm']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        vote_color = "🔴" if gb_vote == 1 else "🟢"
                        vote_text = "Riesgo" if gb_vote == 1 else "Saludable"
                        confidence_style = "background: rgba(255,99,99,0.2);" if gb_vote == 1 else "background: rgba(99,255,132,0.2);"
                        
                        st.markdown(f"""
                            <div class="vote-card" style="{confidence_style}">
                                <span class="vote-emoji">🚀</span>
                                <h3>Gradient Boosting</h3>
                                <div style="font-size: 3rem; margin: 1rem 0;">{vote_color}</div>
                                <h4>{vote_text}</h4>
                                <p style="font-size: 0.9rem; opacity: 0.8;">
                                    Optimización secuencial de errores
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        vote_color = "🔴" if rf_vote == 1 else "🟢"
                        vote_text = "Riesgo" if rf_vote == 1 else "Saludable"
                        confidence_style = "background: rgba(255,99,99,0.2);" if rf_vote == 1 else "background: rgba(99,255,132,0.2);"
                        
                        st.markdown(f"""
                            <div class="vote-card" style="{confidence_style}">
                                <span class="vote-emoji">🌲</span>
                                <h3>Random Forest</h3>
                                <div style="font-size: 3rem; margin: 1rem 0;">{vote_color}</div>
                                <h4>{vote_text}</h4>
                                <p style="font-size: 0.9rem; opacity: 0.8;">
                                    Consenso de múltiples árboles
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with col3:
                        vote_color = "🔴" if svm_vote == 1 else "🟢"
                        vote_text = "Riesgo" if svm_vote == 1 else "Saludable"
                        confidence_style = "background: rgba(255,99,99,0.2);" if svm_vote == 1 else "background: rgba(99,255,132,0.2);"
                        
                        st.markdown(f"""
                            <div class="vote-card" style="{confidence_style}">
                                <span class="vote-emoji">🎯</span>
                                <h3>Support Vector Machine</h3>
                                <div style="font-size: 3rem; margin: 1rem 0;">{vote_color}</div>
                                <h4>{vote_text}</h4>
                                <p style="font-size: 0.9rem; opacity: 0.8;">
                                    Separación óptima de clases
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Resumen final de votación
                    votos_riesgo = sum([gb_vote, rf_vote, svm_vote])
                    votos_saludable = 3 - votos_riesgo
                    
                    st.markdown(f"""
                        <div class="form-container" style="text-align: center; margin-top: 2rem;">
                            <h3>📊 Resumen Final de Votación</h3>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 2rem 0;">
                                <div>
                                    <div style="font-size: 3rem;">🔴</div>
                                    <h4>Votos por "Riesgo"</h4>
                                    <div class="metric-value">{votos_riesgo}/3</div>
                                </div>
                                <div>
                                    <div style="font-size: 3rem;">🟢</div>
                                    <h4>Votos por "Saludable"</h4>
                                    <div class="metric-value">{votos_saludable}/3</div>
                                </div>
                            </div>
                            <div style="padding: 1.5rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 2rem;">
                                <h3>🏆 Decisión Final por Mayoría</h3>
                                <div style="font-size: 2rem; margin: 1rem 0;">
                                    {"🔴 RIESGO CARDIOVASCULAR" if resultado == 1 else "🟢 ESTADO SALUDABLE"}
                                </div>
                                <p style="opacity: 0.9;">
                                    {"⚠️ Se recomienda consulta médica profesional para evaluación detallada" if resultado == 1 
                                     else "✅ Mantén tus hábitos saludables y realiza chequeos preventivos regulares"}
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f"""
                        <div class="error-alert">
                            ❌ <strong>Error al realizar la predicción:</strong> {str(e)}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="error-alert">
                        ⚠️ <strong>Por favor completa todos los campos antes de realizar el diagnóstico</strong>
                    </div>
                """, unsafe_allow_html=True)

    # Botones de navegación
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🏠 **Volver al Inicio**", use_container_width=True):
            st.session_state.formulario = False
            st.rerun()
    
    with col2:
        if st.button("🔄 **Limpiar Formulario**", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("📊 **Nueva Predicción**", use_container_width=True):
            st.rerun()

else:
    st.markdown("""
        <div class="error-alert">
            ⚠️ <strong>No hay modelo entrenado disponible</strong><br>
            Por favor vuelve al inicio y entrena un modelo primero.
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🏠 **Volver al Inicio**", use_container_width=True):
            st.session_state.formulario = False
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 2rem; opacity: 0.8;">
        <p>🫀 <strong>CardioPredict AI</strong> - Machine Learning para la Salud del Corazón</p>
        <p style="font-size: 0.9rem;">
            Desarrollado con ❤️ usando Machine Learning | 
            Hard Voting: Gradient Boosting + Random Forest + SVM
        </p>
        <p style="font-size: 0.8rem; opacity: 0.7;">
            ⚠️ Esta herramienta es solo para fines educativos y no reemplaza el diagnóstico médico profesional
        </p>
    </div>
""", unsafe_allow_html=True)
