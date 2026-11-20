import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Maternal Health Risk",
    page_icon="🏥",
    layout="wide"
)


# Menú lateral de navegación
st.sidebar.title("Predicción de Riesgo Materno")

pagina = st.sidebar.radio(
    "Selecciona un apartado:",
    [
        "Inicio",
        "Evaluación experimental",
        "Predicción",
        "Interpretación"
    ]
)

# Contenido de cada apartado
if pagina == "Inicio":

    st.title("Predicción de Riesgo Materno")


    st.write("**Alumno:** Abraham Payan Arzate")
    st.write("**Matrícula:** P403231")
    st.write("**Materia:** Machine Learning")
    st.write("**Institución:** Universidad Autonoma de Chihuahua")
    st.write("**Docente:** Dra. Olanda Prieto Ordaz")

    st.divider()
    st.header("Información del proyecto")
    st.subheader("Objetivo")
    st.write(
        "Desarrollar una aplicación web interactiva que permita mediante "
        "6 características realizar una evaluación preventiva de la salud materna. "
        "La aplicación compara las predicciones de tres modelos de clasificación "
        "entrenados y evaluados previamente. Su objetivo es servir como una "
        "herramienta de apoyo para el área médica, sin sustituir una evaluación "
        "clínica profesional."
)

    st.subheader("Base de datos")

    st.write(
        "El  dataset utilizado para el desarrollo de esta plataforma lleva el nombre de: Maternal Health Risk Dataset," 
        "fue recopilado en zonas rurales de Bangladesh entre los años de 2018 y 2020, publicandose finalemnet en el año 2023."
        "Esta investigación se realizó en distintos hospitales, clínicas comunitarias y centros de atención materna de la zona. "
        "La recopilación se realizó mediante un sistema de monitoreo de riesgo basado en IoT (Internet of Things) "
        "el cual consiste en un dispositivo remoto que envia informacion a bases de datos. Este dataset cuenta con un "
        "total de 1014 filas o instancias y 6 columnas o caracteristicas. La variable objetivo es RiskLevel, "
        "esta es una variable multiclase, capaz de identificar el nivel de riesgo materno (Low, Mid y High risk)."
)

    st.subheader("Modelos implementados")

    st.write(
       """
       Los modelos seleccionados durante la etapa experimental son:

       - **Stacking C Optimizado.** 

       - **Random Forest Optimizado.** 

       - **Gradient Boosting Optimizado.** 
       """
)

elif pagina == "Evaluación experimental":

    st.title("Evaluación experimental")

    st.write(
        "En este apartado se presentan los resultados obtenidos "
        "durante la evaluación de los tres modelos seleccionados, "
        "utilizando el conjunto de prueba (Test)."
    )

    st.subheader("Comparación de métricas")

    resultados = pd.DataFrame({
        "Modelo": [
            "Stacking C Optimizado",
            "Random Forest Optimizado",
            "Gradient Boosting Optimizado"
        ],
        "Accuracy": [0.7444, 0.7222, 0.7667],
        "Precision Macro": [0.7021, 0.6652, 0.7363],
        "Recall Macro": [0.7139, 0.6749, 0.7376],
        "F1 Macro": [0.7001, 0.6607, 0.7358],
        "ROC-AUC": [0.8359, 0.8290, 0.8486]
    })

    st.dataframe(
        resultados.style.format({
            "Accuracy": "{:.4f}",
            "Precision Macro": "{:.4f}",
            "Recall Macro": "{:.4f}",
            "F1 Macro": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        hide_index=True,
        use_container_width=True
    )



    st.subheader("Comparación gráfica de métricas")

    metrica = st.selectbox(
        "Selecciona la métrica que deseas comparar:",
        [
            "Accuracy",
            "Precision Macro",
            "Recall Macro",
            "F1 Macro",
            "ROC-AUC"
        ],
        index=3
    )

    st.write(
        f"Comparación de los tres modelos utilizando {metrica} "
        "en el conjunto de prueba (Test)."
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    barras = ax.barh(
        resultados["Modelo"],
        resultados[metrica],
        height=0.6
    )

    ax.set_xlabel(metrica)
    ax.set_ylabel("Modelo")
    ax.set_xlim(0, 1)

    ax.bar_label(
        barras,
        fmt="%.4f",
        padding=5
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)




elif pagina == "Predicción":

    st.title("Predicción de riesgo materno")

    st.write(
        "Ingrese las características de la paciente para "
        "obtener una estimación de su nivel de riesgo materno "
        "mediante los tres modelos de clasificación."
    )

    st.subheader("Datos de la paciente")

    col1, col2 = st.columns(2)

    with col1:

        edad = st.number_input(
            "Edad (años)",
            min_value=10,
            max_value=100,
            value=25
        )

        presion_sistolica = st.number_input(
            "Presión sistólica (mmHg)",
            min_value=50,
            max_value=250,
            value=120
        )

        presion_diastolica = st.number_input(
            "Presión diastólica (mmHg)",
            min_value=30,
            max_value=150,
            value=80
        )

    with col2:

        glucosa = st.number_input(
            "Nivel de glucosa (BS)",
            min_value=0.0,
            value=7.5,
            step=0.1
        )

        temperatura = st.number_input(
            "Temperatura corporal (°F)",
            min_value=90.0,
            max_value=110.0,
            value=98.0,
            step=0.1
        )

        frecuencia_cardiaca = st.number_input(
            "Frecuencia cardíaca (BPM)",
            min_value=30,
            max_value=200,
            value=75
        )

    datos = pd.DataFrame([{
        "Age": edad,
        "SystolicBP": presion_sistolica,
        "DiastolicBP": presion_diastolica,
        "BS": glucosa,
        "BodyTemp": temperatura,
        "HeartRate": frecuencia_cardiaca
    }])

    st.divider()

    st.subheader("Generar predicciones")

    if st.button("Generar predicciones"):
       
    

        modelos = {
            "Stacking C Optimizado": "modelos/stacking_c.pkl",
            "Random Forest Optimizado": "modelos/random_forest.pkl",
            "Gradient Boosting Optimizado": "modelos/gradient_boosting.pkl"
        }

        resultados_prediccion = []

        for nombre, ruta in modelos.items():

            modelo = joblib.load(ruta)

            prediccion = modelo.predict(datos)[0]
            probabilidades = modelo.predict_proba(datos)[0]

            clases = modelo.classes_

            prob_por_clase = dict(zip(clases, probabilidades))

            resultados_prediccion.append({
                "Modelo": nombre,
                "Predicción": prediccion,
                "Low Risk (%)": prob_por_clase.get("low risk", 0) * 100,
                "Mid Risk (%)": prob_por_clase.get("mid risk", 0) * 100,
                "High Risk (%)": prob_por_clase.get("high risk", 0) * 100
            })

        tabla_predicciones = pd.DataFrame(resultados_prediccion)

        st.subheader("Resultados de las predicciones")

        st.dataframe(
            tabla_predicciones.style.format({
                "Low Risk (%)": "{:.2f}",
                "Mid Risk (%)": "{:.2f}",
                "High Risk (%)": "{:.2f}"
            }),
            hide_index=True,
            use_container_width=True
        )


    
        st.subheader("Comparación de probabilidades")

        st.write(
            "La siguiente gráfica compara las probabilidades "
            "estimadas por los tres modelos para cada nivel de riesgo."
        )

        probabilidades_grafica = tabla_predicciones.set_index("Modelo")[
            ["Low Risk (%)", "Mid Risk (%)", "High Risk (%)"]
        ]

        fig, ax = plt.subplots(figsize=(12, 5))

        probabilidades_grafica.plot(
            kind="bar",
            ax=ax,
            width=0.75
        )

        ax.set_xlabel("Modelo")
        ax.set_ylabel("Probabilidad estimada (%)")
        ax.set_ylim(0, 100)

        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Nivel de riesgo")

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)



elif pagina == "Interpretación":

    st.title("Interpretación de las predicciones")

    st.write(
        "En este apartado se explica el significado de los "
        "niveles de riesgo materno, la interpretación de las "
        "probabilidades y las limitaciones de los modelos."
    )

    st.subheader("Niveles de riesgo materno")

    st.markdown("""
    **Low Risk (Riesgo bajo)**

   Indica que las características ingresadas corresponden a patrones 
   asociados con un menor nivel de riesgo materno dentro del conjunto
   de datos. Sin embargo, esta clasificación no garantiza la ausencia
   de complicaciones médicas.  

    **Mid Risk (Riesgo moderado)**

   Apartir de este nivel de riesgo de salud, es oportuno una valoracion mas 
   exaustiva para determinar potenciales riesgos en la salud de la paciente.

    **High Risk (Riesgo alto)**

    Indica que las características ingresadas corresponden a patrones asociados
    con un mayor nivel de riesgo materno. Se recomienda una valoración médica
    profesional para determinar la condición de salud de la paciente y las 
    medidas correspondientes..
    """)

    st.divider()

    st.subheader("Interpretación de probabilidades")

    st.write(
        "Cada modelo proporciona una estimación porcentual "
        "para los tres niveles de riesgo. La clase con el "
        "porcentaje más alto se selecciona como predicción."
    )

    st.write(
        "Por ejemplo, si un modelo asigna 60 % a Low Risk, "
        "30 % a Mid Risk y 10 % a High Risk, su predicción "
        "será Low Risk."
    )

    st.divider()

    st.subheader("Limitaciones de los modelos")

    st.markdown("""

    - Durante la evaluación experimental se observaron
      dificultades para identificar correctamente algunas
      instancias de riesgo moderado (Mid Risk).


    - Los modelos utilizan únicamente seis características,
      por lo que no consideran todos los factores que pueden
      influir en la salud materna.

    - Sus predicciones no sustituyen una evaluación médica y
    deben de ser utilizados unicamente como una herramienta de apoyo.
    """)
