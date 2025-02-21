import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import supervision as sv

# Configurar página
st.set_page_config(
    page_title="Detector de Hojas",
    layout="centered",  # Cambiado a centered para vista móvil
    initial_sidebar_state="collapsed"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        max-width: 375px !important;  /* Ancho típico de móvil */
        padding: 0rem 0.5rem;
        margin: 0 auto;
        background-color: #ffffff;
    }
    .stTitle {
        text-align: center;
        color: #2F5233;
        font-size: 1.5rem !important;
    }
    .block-container {
        max-width: 375px !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    .detection-box {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        background-color: #F7F9F7;
        color: #2F5233;
        border: 1px solid #7dba4c;
    }
    div.stButton > button {
        width: 100%;
        background-color: #7dba4c;
        color: white;
        border-radius: 20px;
    }
    .uploadedFile {
        width: 100% !important;
    }
    .css-1kyxreq {
        justify-content: center !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for tracking analysis state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

st.title("🍃 Detector de Hojas de Arándano")
st.markdown("---")

# Cliente de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="1lZZgNOL20KSm5TMIKox"
)

# Show upload section only if analysis hasn't been done
if not st.session_state.analyzed:
    col1 = st.container()
    with col1:
        uploaded_file = st.file_uploader("📸 Selecciona o toma una foto", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            # Show preview of uploaded image
            st.image(uploaded_file, caption="Vista previa", use_container_width=True)
            # Add analyze button
            if st.button("Analizar Imagen"):
                st.session_state.analyzed = True
                st.session_state.image = uploaded_file
                st.rerun()

# Show results if analysis has been done
if st.session_state.analyzed and hasattr(st.session_state, 'image'):
    try:
        # Get image from session state
        file_bytes = np.asarray(bytearray(st.session_state.image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (600, 600))

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
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 
                    caption="Imagen Analizada",
                    use_container_width=True)
            
            st.markdown("### 📊 Resultados del Análisis")
            
            # Agrupar predicciones por tipo
            predictions_by_type = {}
            for pred in result["predictions"]:
                pred_type = pred["class"]
                if pred_type not in predictions_by_type:
                    predictions_by_type[pred_type] = []
                predictions_by_type[pred_type].append(float(pred["confidence"]) * 100)
            
            # Mostrar resultados agrupados
            for pred_type, confidences in predictions_by_type.items():
                st.markdown(f"""
                    <div class='detection-box'>
                        <h4 style='margin:0'>Tipo: {pred_type}</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                # Detalles expandibles
                with st.expander("Ver detalles"):
                    for i, conf in enumerate(confidences, 1):
                        st.markdown(f"""
                            <div style='padding:5px; margin:2px 0; color:#2c3e50;'>
                                <p style='margin:0'>Detección {i}: {conf:.1f}% de confianza</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            if len(result["predictions"]) > 0:
                st.success("✅ Análisis completado exitosamente")
            else:
                st.warning("⚠️ No se detectaron hojas en la imagen")

    except Exception as e:
        st.error(f"❌ Error en la detección: {str(e)}")
        if st.button("Intentar nuevamente"):
            st.session_state.analyzed = False
            st.rerun()
else:
    st.info("👈 Por favor, selecciona una imagen para comenzar el análisis")