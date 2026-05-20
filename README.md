# 🔍 VisionShop — Clasificador de Imágenes con Deep Learning

VisionShop nació de una pregunta concreta: ¿puede una red neuronal aprender a 
distinguir objetos visuales desde cero, sin usar modelos preentrenados? 
Este proyecto responde esa pregunta entrenando dos arquitecturas CNN diferentes 
sobre CIFAR-10 y comparando sus resultados.

---

## Demo en vivo

👉 [Probar VisionShop] https://huggingface.co/spaces/dart01/visionshop

---

## Cómo funciona

Subes cualquier imagen y la red neuronal la clasifica en una de 10 categorías: 
avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco o camión. 
El modelo muestra la predicción con su nivel de confianza y las probabilidades 
de todas las clases.

### La arquitectura CNN

Diseñé una CNN con 3 bloques convolucionales desde cero en PyTorch. Cada bloque 
aplica filtros convolucionales para detectar patrones visuales de lo simple a lo 
complejo — bordes en la primera capa, texturas en la segunda, formas completas 
en la tercera — seguido de MaxPooling para reducir dimensionalidad. Las capas 
fully connected al final clasifican entre las 10 categorías con Dropout para 
regularización.

---

## Resultados

| Modelo | Dataset | Épocas | Test Accuracy |
|---|---|---|---|
| CNN desde cero | 50K imágenes | 10 | **76%** |
| CNN mejorada | 5K imágenes | 10 | 54% |

El experimento con la CNN mejorada usando menos datos confirmó algo importante: 
**más datos supera a una arquitectura más compleja**. Con 10x menos imágenes, 
una red más profunda no logra superar a una más simple entrenada con el dataset 
completo.

El análisis de la matriz de confusión reveló que las categorías cat y dog 
tienen el f1-score más bajo (56% y 67% respectivamente) por su alta similitud 
visual — resultado consistente con benchmarks publicados en CIFAR-10.

---

## Stack técnico

| Capa | Herramienta |
|---|---|
| Deep Learning | PyTorch |
| Datos y transforms | torchvision |
| Métricas | scikit-learn |
| Visualizaciones | matplotlib |
| Demo interactiva | Streamlit |

---

## Estructura del proyecto

```text
VisionShop/
├── notebooks/
│   ├── 01_eda.ipynb                    ← exploración de CIFAR-10
│   ├── 02_cnn_scratch.ipynb            ← CNN desde cero
│   └── 03_transfer_learning.ipynb      ← comparación de arquitecturas
├── models/
│   └── cnn_scratch.pth                 ← modelo entrenado
├── src/
├── app.py                              ← demo interactiva Streamlit
├── requirements.txt
└── README.md
```

---

## Correr localmente

```bash
git clone https://github.com/dart01/visionshop.git
cd visionshop

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## Lo que aprendí

Entrenar desde cero en CPU fue lento pero me obligó a entender cada decisión 
de arquitectura — por qué BatchNorm ayuda, qué hace realmente el Dropout, 
cómo el tamaño del batch afecta la convergencia. Los resultados de la matriz 
de confusión fueron la parte más interesante: ver exactamente dónde falla el 
modelo y por qué es lo que separa un proyecto de juguete de uno de producción.

---

## Autor

**Diego Riaño**
https://www.linkedin.com/in/diegoandres001/ · [GitHub](https://github.com/dart01) · (https://huggingface.co/spaces/dart01/visionshop)
