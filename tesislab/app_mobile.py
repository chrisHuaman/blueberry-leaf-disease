import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import supervision as sv

# Configurar página
st.set_page_config(
    page_title="Detector de Hojas",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTitle {
        text-align: center;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🍃 Detector de Hojas de Arándano")
st.markdown("---")

# Cliente de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="1lZZgNOL20KSm5TMIKox"
)

# Interface de usuario
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📸 Selecciona o toma una foto", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convertir imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (600, 600))
    
    try:
        with st.spinner('Analizando imagen...'):
            result = CLIENT.infer(image, model_id="drberry-gen-2-hpfhd/1")
            
            # Procesar resultados
            detections = sv.Detections.from_inference(result)
            bounding_box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
            labels = [item["class"] for item in result["predictions"]]
            
            # Anotar imagen
            annotated_image = image.copy()
            annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            
            # Mostrar resultados
            with col1:
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Imagen Analizada")
            
            with col2:
                st.markdown("### 📊 Resultados del Análisis")
                for pred in result["predictions"]:
                    confidence = float(pred["confidence"]) * 100
                    st.markdown(f"""
                        <div style='padding:10px; border-radius:5px; margin:5px 0; background-color:#f0f2f6'>
                            <h4 style='color:#2c3e50; margin:0'>Tipo: {pred["class"]}</h4>
                            <p style='margin:5px 0'>Confianza: {confidence:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                if len(result["predictions"]) > 0:
                    st.success("✅ Análisis completado exitosamente")
                else:
                    st.warning("⚠️ No se detectaron hojas en la imagen")
        
    except Exception as e:
        st.error(f"❌ Error en la detección: {str(e)}")
else:
    with col2:
        st.info("👈 Por favor, selecciona una imagen para comenzar el análisis")