
# Predicción de Riesgo Materno.

## Descripción del proyecto.

Esta aplicacion web desarrollada mediante Streamlit para estimar
el nivel de riesgo materno mediante modelos de Machine Learning.
El modelo fue entrenado con un dataset desarrolladoe en Bangladesh,
Consta de 6 caracteristicas iniciales y una variable de tipo cualitativa
que es nuestra variable predictiva, al final se utilizó un total de 464
instancias despues del preprocesamiento de datos realizado.

El sistema utiliza seis características de salud para realizar predicciones 
y comparar los resultados de tres modelos de clasificación.

## Base de datos.

Maternal Health Risk Dataset, disponible en UCI
Machine Learning Repository.

- Instancias: 1014 iniciales
- Características predictoras: 6
- Variable objetivo: RiskLevel
- Clases: Low Risk, Mid Risk y High Risk

Fuente:
https://archive.ics.uci.edu/dataset/863/maternal+health+risk

## Modelos finales implementados.
Para el desarrollo de esta applicacion se tomaron en cuenta
numerosos modelos, sin embargo los modelos finales impplementados
con un mejor desempeño fueron los siguientes:

- Stacking C Optimizado
- Random Forest Optimizado
- Gradient Boosting Optimizado

## Funcionalidades

- Información general del proyecto.
- Comparación de métricas experimentales.
- Predicción mediante tres modelos de clasificación.
- Visualización de probabilidades estimadas.
- Interpretación de resultados y limitaciones.

## Tecnologías utilizadas

- Python
- Streamlit
- scikit-learn
- pandas
- NumPy
- Matplotlib
- joblib

## Ejecución local

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

## Advertencia

Esta aplicación fue desarrollada con fines académicos
Sus predicciones no constituyen un diagnóstico médico
ni sustituyen la valoración de profesionales de la salud.
Dado que este dataset contaba con numeroas instancias repetidas, 
cerca de un 50% y el procesamiento que le dimos a dichas instancias
nuestro moedelo no puede ser tomado como un diagnostico final.
La paciente necesita de una valoracion profesional para determinar
su estado de salud actual.


