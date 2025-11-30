# Machine Learning con Árboles de Decisión

Este repositorio contiene un cuaderno de Jupyter (`Decision_Trees.ipynb`) que sirve como guía educativa y práctica sobre el uso de **Árboles de Decisión** en Machine Learning.

El notebook cubre desde conceptos básicos hasta un flujo de trabajo completo de Data Science, utilizando la librería `scikit-learn` en Python.

## Contenido del Notebook

El archivo `Decision_Trees.ipynb` está dividido en las siguientes secciones:

1.  **Clasificación (Iris Dataset):**

    - Introducción a los árboles de decisión para clasificar tipos de flores.
    - Visualización de la estructura del árbol.

2.  **Regresión (Diabetes Dataset):**

    - Uso de árboles de decisión para predecir valores numéricos continuos.
    - Importancia del hiperparámetro `max_depth` para evitar el sobreajuste (overfitting).

3.  **Selección de Características (Breast Cancer Dataset):**

    - Demostración de cómo los árboles seleccionan automáticamente las variables más relevantes.
    - Visualización de árboles más complejos.

4.  **Evaluación de Modelos:**

    - Implementación correcta de la división de datos en entrenamiento y prueba (`train_test_split`).
    - Métricas de evaluación: Accuracy y Reporte de Clasificación.

5.  **Proyecto Completo: Titanic:**

    - Un ejercicio "end-to-end" que simula un proyecto real.
    - **Carga de datos:** Descarga automática del dataset Titanic.
    - **Limpieza de datos:** Manejo de columnas irrelevantes y transformación de variables categóricas (One-Hot Encoding).
    - **Balanceo de clases:** Uso de `RandomUnderSampler` para tratar datos desbalanceados.
    - **Optimización:** Búsqueda de hiperparámetros óptimos con `GridSearchCV`.
    - **Interpretación:** Análisis de la importancia de las características (Feature Importance).
    - **Predicción:** Pronóstico de supervivencia para nuevos pasajeros.

6.  **Proyecto: Clasificación de Calidad de Autos (Car Evaluation):**
    - Un segundo proyecto práctico utilizando un dataset de Kaggle.
    - **Carga de datos:** Uso de la librería `kagglehub` para descargar el dataset.
    - **EDA:** Análisis exploratorio detallado con visualizaciones.
    - **Preprocesamiento:** Codificación de variables ordinales con `OrdinalEncoder` y balanceo de datos.
    - **Modelado:** Entrenamiento de un Árbol de Decisión con optimización de hiperparámetros.
    - **Evaluación:** Métricas de desempeño y matriz de confusión.

## Requisitos

Para ejecutar este notebook, necesitas tener instalado Python y las siguientes librerías:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (para `RandomUnderSampler`)
- category_encoders (para `OrdinalEncoder`)
- kagglehub (para descargar datasets de Kaggle)

Puedes instalar las dependencias usando pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn category_encoders kagglehub
```

## Cómo usar

1.  Clona este repositorio.
2.  Abre el archivo `Decision_Trees.ipynb` en Visual Studio Code, Jupyter Notebook o Google Colab.
3.  Ejecuta las celdas en orden secuencial.

## Datos

- Los datasets de Iris, Diabetes y Breast Cancer se cargan directamente desde `sklearn.datasets`.
- El dataset del Titanic se descarga automáticamente desde una URL pública de la Universidad de Stanford durante la ejecución del notebook.
- El dataset de Car Evaluation se descarga automáticamente desde Kaggle usando `kagglehub`.
