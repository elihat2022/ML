# Machine Learning con Árboles de Decisión y Random Forest

Este repositorio contiene cuadernos de Jupyter (`Decision_Trees.ipynb` y `Random_Forest.ipynb`) que sirven como guía educativa y práctica sobre el uso de algoritmos basados en árboles en Machine Learning.

Los notebooks cubren desde conceptos básicos hasta flujos de trabajo completos de Data Science, utilizando la librería `scikit-learn` en Python.

## Contenido de los Notebooks

### 1. Árboles de Decisión (`Decision_Trees.ipynb`)

Este archivo cubre los fundamentos de los árboles de decisión:

1.  **Clasificación (Iris Dataset):** Introducción y visualización.
2.  **Regresión (Diabetes Dataset):** Predicción numérica y control de overfitting (`max_depth`).
3.  **Selección de Características (Breast Cancer):** Cómo los árboles seleccionan variables importantes.
4.  **Evaluación:** Uso de `train_test_split` y métricas.
5.  **Proyecto Titanic:** Flujo completo (Limpieza, One-Hot Encoding, Balanceo, GridSearch).
6.  **Proyecto Car Evaluation:** Clasificación multicurso con datos de Kaggle.

### 2. Random Forest (`Random_Forest.ipynb`)

Este archivo explora el algoritmo de **Random Forest**, una técnica de ensamble que mejora la precisión y robustez:

1.  **Iris Dataset:** Ejemplo introductorio comparando con un solo árbol.
2.  **Car Evaluation:** Clasificación de calidad de autos utilizando un bosque de árboles.
3.  **Proyecto Telco Customer Churn:**
    - **Objetivo:** Predecir la fuga de clientes (Churn).
    - **Desafíos:** Manejo de datos desbalanceados y variables mixtas.
    - **Técnicas:** `RandomUnderSampler`, `OrdinalEncoder`, `GridSearchCV` para optimizar `n_estimators` y `max_depth`.
    - **Interpretación:** Análisis de importancia de características para decisiones de negocio.

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
2.  Abre los archivos `.ipynb` en Visual Studio Code, Jupyter Notebook o Google Colab.
3.  Ejecuta las celdas en orden secuencial.

## Datos

- Los datasets de Iris, Diabetes y Breast Cancer se cargan directamente desde `sklearn.datasets`.
- El dataset del Titanic se descarga automáticamente desde una URL pública.
- Los datasets de Car Evaluation y Telco Churn se descargan automáticamente desde Kaggle usando `kagglehub`.
