import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="VisionShop — Clasificador de Imágenes",
    page_icon="🔍",
    layout="wide"
)

# Definir la misma arquitectura CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

clases = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

emojis = {
    'airplane': '✈️', 'automobile': '🚗', 'bird': '🐦',
    'cat': '🐱', 'deer': '🦌', 'dog': '🐶',
    'frog': '🐸', 'horse': '🐴', 'ship': '🚢', 'truck': '🚚'
}

@st.cache_resource
def cargar_modelo():
    modelo = CNN()
    ruta = os.path.join(os.path.dirname(__file__), "models", "cnn_scratch.pth")
    modelo.load_state_dict(torch.load(ruta, map_location="cpu"))
    modelo.eval()
    return modelo

# UI
st.title("🔍 VisionShop — Clasificador de Imágenes")
st.markdown("Sube una imagen y la red neuronal la clasificará en una de 10 categorías usando una CNN entrenada con CIFAR-10.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Imagen")
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    
    if archivo:
        imagen = Image.open(archivo).convert("RGB")
        st.image(imagen, caption="Imagen subida", use_container_width=True)

with col2:
    st.subheader("🧠 Predicción")
    
    if archivo:
        modelo = cargar_modelo()
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        tensor = transform(imagen).unsqueeze(0)
        
        with torch.no_grad():
            salida = modelo(tensor)
            probabilidades = torch.softmax(salida, dim=1)[0]
            prediccion_idx = probabilidades.argmax().item()
            prediccion = clases[prediccion_idx]
            confianza = probabilidades[prediccion_idx].item() * 100
        
        st.markdown(f"### {emojis[prediccion]} {prediccion.upper()}")
        st.metric("Confianza", f"{confianza:.1f}%")
        
        st.markdown("**Probabilidades por clase:**")
        for i, (clase, prob) in enumerate(zip(clases, probabilidades)):
            st.progress(float(prob), text=f"{emojis[clase]} {clase}: {prob*100:.1f}%")
    else:
        st.info("👈 Sube una imagen para clasificar")

st.divider()
st.markdown("""
**Sobre el modelo:**
- Arquitectura: CNN con 3 bloques convolucionales
- Dataset: CIFAR-10 (50,000 imágenes de entrenamiento)
- Accuracy en test: **76%**
- Entrenado con PyTorch en CPU
""")