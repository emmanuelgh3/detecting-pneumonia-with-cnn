# Clasificación de Neumonía a partir de Rayos X con Redes Convolucionales

Este proyecto tiene como objetivo clasificar imágenes de rayos X de tórax para detectar casos de **neumonía** utilizando una red neuronal convolucional (CNN) desarrollada en Keras con TensorFlow.

## Descripción del Proyecto

Utilizando el conjunto de datos **Chest X-Ray Images (Pneumonia)**, se entrenó una red CNN para identificar si una imagen de rayos X corresponde a una persona sana (**Normal**) o con neumonía (**Pneumonia**).

El modelo fue entrenado con datos reescalados en escala de grises, utilizando técnicas de regularización como Dropout y BatchNormalization para evitar el sobreajuste. Se utilizaron métricas como **Precisión**, **Recall** y **AUC** para evaluar el rendimiento del modelo.

## Sobre el Dataset

Este proyecto utiliza el conjunto de datos **Chest X-Ray Images (Pneumonia)**, disponible públicamente en [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

### Contexto Clínico

Las imágenes de rayos X fueron recolectadas de pacientes pediátricos (entre 1 y 5 años de edad) del Guangzhou Women and Children’s Medical Center en China. Todas las radiografías se tomaron en vista anteroposterior (AP) como parte del cuidado clínico rutinario. Este estudio fue publicado en la revista _Cell_ y puede consultarse en [este enlace](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5).

> - Las imágenes **normales** presentan pulmones despejados, sin opacidades anormales.
> - Las imágenes con **neumonía bacteriana** suelen mostrar consolidación lobar focal.
> - Las imágenes con **neumonía viral** muestran un patrón más difuso e intersticial.

### Estructura del Dataset

El conjunto contiene 5,863 imágenes en formato JPEG, divididas en tres carpetas:

- `train/` — imágenes para entrenamiento.
- `val/` — imágenes para validación.
- `test/` — imágenes para prueba final.

Cada carpeta contiene dos subcarpetas:

- `NORMAL/` — rayos X de tórax normales.
- `PNEUMONIA/` — rayos X con diagnóstico de neumonía.

### Validación y Control de Calidad

- Las imágenes fueron seleccionadas tras un proceso de control de calidad que eliminó escaneos defectuosos o ilegibles.
- Las etiquetas fueron asignadas por **dos médicos especialistas**.
- Un tercer experto revisó el conjunto de evaluación para asegurar mayor precisión diagnóstica.

## Librerías y Requisitos

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib / Seaborn  
- Scikit-learn  

## Arquitectura del Modelo

### Arquitectura del Modelo

Se utilizó una red neuronal convolucional (CNN) construida desde cero con Keras. La arquitectura del modelo es la siguiente:

- `Conv2D` (32 filtros, 3x3, activación ReLU)
- `MaxPooling2D` (2x2)
- `Conv2D` (64 filtros, 3x3, activación ReLU)
- `MaxPooling2D` (2x2)
- `Conv2D` (128 filtros, 3x3, activación ReLU)
- `MaxPooling2D` (2x2)
- `Conv2D` (256 filtros, 3x3, activación ReLU)
- `MaxPooling2D` (2x2)
- `Flatten`
- `Dense` (128 unidades, activación ReLU)
- `Dropout` (0.5)
- `Dense` (1 unidad, activación Sigmoid)

Esta arquitectura permite detectar patrones espaciales en imágenes de rayos X para diferenciar entre casos de neumonía y casos normales.

### Filtros Elegidos

La red utiliza filtros convolucionales con tamaños de **3x3** debido a sus siguientes características:

1. **Eficiencia Computacional**: Un tamaño de filtro pequeño como el **3x3** es computacionalmente eficiente y al mismo tiempo sigue siendo capaz de capturar características importantes en las imágenes. 4 capas convolucionales son suficientes para imágenes de 150x150 px (mayor profundidad no mejoró resultados en pruebas preliminares), además de que 256 filtros en la última capa mantienen un balance (7.61MB de pesos).
   
2. **Características Locales**: Los filtros de **3x3** son efectivos para capturar patrones locales en imágenes, como bordes y texturas, lo que es esencial en el procesamiento de imágenes médicas donde los detalles y pequeños cambios pueden ser indicativos de enfermedades como lo es en este caso particular.

### Entrenamiento del Modelo

- **Función de pérdida:** `binary_crossentropy`  
- **Optimizador:** `Adam`  
- **Métricas de evaluación:** `accuracy`, `precision`, `recall`, `AUC`.  
- **Épocas:** 10  
- **Tamaño de batch:** 32  

El modelo fue entrenado con el conjunto de datos en la carpeta `train`, validado con `val`, y evaluado finalmente usando `test`.

## Procedimiento para Correr el Código

Este procedimiento describe los pasos necesarios para ejecutar el código del clasificador de neumonía utilizando el modelo entrenado.

### 1. Descarga los Archivos

Primero, debes descargar los archivos requeridos y asegurarte de tenerlos en una carpeta. Los archivos necesarios son:

- `pneumonia_classifier.ipynb` (Código general)
- `mejor_modelo.h5` (Modelo entrenado)
- `test.zip` (Conjunto de imágenes de prueba comprimido)

Coloca todos estos archivos en una carpeta en tu sistema (descomprimir el archivo .zip).

### 2. Instalar las Librerías Necesarias

Abre el archivo `pneumonia_classifier.ipynb` en Jupyter Notebook y ejecuta la primera celda, que contiene la importación de las librerías necesarias para ejecutar el código:

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.metrics import Precision, Recall, AUC
import seaborn as sns
```

### 3. Validación del Modelo

Dirigirse a la sección **Validacion del modelo** del notebook. En esta sección, cargarás el modelo previamente entrenado y las imágenes del conjunto de test para evaluar su rendimiento. 

Primero, debes cargar el modelo entrenado con el siguiente comando:
```python
model = load_model('mejor_modelo.h5')
```

A continuación, copiar la ruta de la carpeta ´test´ para definir el generador de imágenes para preprocesar y cargar las imágenes de test:
```python
test_dir = r'tu/ruta/chest_xray/test'  #Ruta de la carpeta de test
image_size = (150, 150)
batch_size = 32
test_datagen = ImageDataGenerator(rescale=1./255)  
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True  
)
```

Luego, correr las siguientes lineas de código hará que el modelo haga predicciones sobre el conjunto de test y muestra las métricas de rendimiento:
```python
loss, acc, precision, recall, auc = model.evaluate(test_generator)
print(f"\nPrecisión (accuracy): {acc:.4f}")
print(f"Precisión (precision): {precision:.4f}")
print(f"Sensibilidad (recall): {recall:.4f}")
print(f"AUC: {auc:.4f}")
```

Finalmente, la última línea de código hará que se muestre una parte de las predicciones que realizó el modelo sobre el conjunto de testeo.
```python
test_images, test_labels = next(test_generator)
predictions = model.predict(test_images)
predicted_labels = (predictions > 0.5).astype(int).flatten()
plt.figure(figsize=(15, 15))
for i in range(32):
    plt.subplot(8, 4, i+1)
    plt.imshow(test_images[i])
    plt.title(f"Real: {int(test_labels[i])}, Predicho: {predicted_labels[i]}")
    plt.axis('off')
plt.show()
```







