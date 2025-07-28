import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

@st.cache_resource
def load_model():
    return YOLO("best.pt") 

model = load_model()

st.set_page_config(page_title="Bone Loss Detection", layout="centered")
st.title("Bone Loss Detection App")
st.markdown("Upload a dental X-ray image to detect signs of **bone loss** using a custom YOLOv8 model.")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    temp_path = "uploaded_temp.jpg"
    img.save(temp_path)

    st.info(" Running detection...")
    results = model.predict(source=temp_path, save=False, conf=0.25)

    result_img = results[0].plot()  
    st.image(result_img, caption="Detection Result", use_column_width=True)

    labels_detected = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        labels_detected.append(f"{label} ({conf:.2f})")

    if labels_detected:
        st.success(" Bone loss detected!")
        st.write("### Detected Regions:")
        for item in labels_detected:
            st.write(f"• {item}")
    else:
        st.success(" No bone loss detected in this image.")
