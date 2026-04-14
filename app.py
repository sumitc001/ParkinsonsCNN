import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib

st.set_page_config(
    page_title="Parkinson's Detection",
    page_icon="🧠",
    layout="wide"
)

data = joblib.load("parkinsons_pipeline.pkl")
pca = data["pca"]
voting_model = data["model"]

device = torch.device("cpu")

cnn_model = models.efficientnet_b0(pretrained=True)
cnn_model.classifier = nn.Identity()
cnn_model.eval()
cnn_model.to(device)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['Normal', 'Parkinson']

st.title("🧠 Parkinson's Disease Detection")
st.markdown("Upload a brain MRI scan to predict Parkinson’s disease using AI.")


uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 Prediction Result")

        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            features = cnn_model(img).numpy()
            features_pca = pca.transform(features)

            pred = voting_model.predict(features_pca)[0]
            prob = voting_model.predict_proba(features_pca)[0]

        predicted_class = class_names[pred]
        confidence = float(np.max(prob))

        if predicted_class == "Parkinson":
            st.error(f"⚠️ {predicted_class} Detected")
        else:
            st.success(f"✅ {predicted_class}")

        st.markdown(f"### Confidence: **{confidence:.2f}**")

        st.progress(confidence)

        st.markdown("### 📊 Class Probabilities")
        prob_dict = {
            "Normal": float(prob[0]),
            "Parkinson": float(prob[1])
        }
        st.bar_chart(prob_dict)

st.markdown("---")
st.markdown("Built using Deep Learning + Ensemble ML")