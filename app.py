# -*- coding: utf-8 -*-
"""
Estimador de riesgo de deserción de clientes
ACA Final · Fundamentos de Inteligencia de Negocios (EAD1039)
CUN · Especialización en Analítica Avanzada de Datos

Integrantes: Marcos Yair Vela Pulido, Jerszinho Fray Rafael García Forero,
Fabio Ariel Cabrera Estrella
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
AZUL, ROJO, GRIS = "#1F6FB2", "#C0453F", "#8A9099"
TINTA, TINTA2, REJILLA = "#22262A", "#5C646B", "#E3E6E8"

st.set_page_config(page_title="Riesgo de deserción de clientes",
                   page_icon="📉", layout="wide")


# ------------------------------------------------------------------
# Carga
# ------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    with open(BASE / "modelo_churn.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def cargar_datos():
    return pd.read_csv(BASE / "clientes_churn.csv")


modelo = cargar_modelo()
datos = cargar_datos()

NUMERICAS = ["edad", "antiguedad_meses", "valor_poliza",
             "reclamaciones", "pqr", "retrasos_pago"]
CATEGORICAS = ["genero", "ciudad", "tipo_poliza", "canal_atencion"]

# Umbral de trabajo: decil superior de la cartera de referencia
PROBA_REFERENCIA = modelo.predict_proba(datos[NUMERICAS + CATEGORICAS])[:, 1]
UMBRAL_DECIL = float(np.quantile(PROBA_REFERENCIA, 0.90))
TASA_BASE = float(datos["churn"].mean())


def explicar(entrada):
    """Descompone la prediccion en el aporte de cada variable.

    En la regresion logistica el log-odds es la suma del intercepto mas cada
    coeficiente por su valor estandarizado. Como la descomposicion es exacta,
    sirve para mostrar que empujo el resultado hacia arriba y que lo bajo.
    """
    prep = modelo.named_steps["preprocesamiento"]
    clf = modelo.named_steps["clasificador"]
    vector = prep.transform(entrada)
    if hasattr(vector, "toarray"):
        vector = vector.toarray()
    nombres = prep.get_feature_names_out()
    aportes = vector[0] * clf.coef_[0]

    legible = []
    for n in nombres:
        n = n.replace("num__", "").replace("cat__", "").replace("_", " ")
        legible.append(n.capitalize())

    tabla = (pd.DataFrame({"variable": legible, "aporte": aportes})
             .query("aporte != 0")
             .assign(magnitud=lambda d: d["aporte"].abs())
             .sort_values("magnitud", ascending=False))
    return tabla


# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("Estimador de riesgo de deserción")
st.caption("Compañía aseguradora · Modelo de regresión logística entrenado sobre "
           f"{len(datos):,} clientes · Tasa de deserción observada: {TASA_BASE:.1%}".replace(",", "."))

tab_cliente, tab_cartera, tab_modelo = st.tabs(
    ["Evaluar un cliente", "Panorama de la cartera", "Estado del modelo"])

# ==================================================================
# 1. Evaluar un cliente
# ==================================================================
with tab_cliente:
    st.subheader("Datos del cliente")

    c1, c2, c3 = st.columns(3)
    with c1:
        edad = st.slider("Edad", 18, 75, 45)
        antiguedad = st.slider("Antigüedad en meses", 1, 120, 36)
    with c2:
        valor = st.number_input("Valor de la póliza (COP)", 80_000, 1_000_000,
                                500_000, step=10_000)
        reclamaciones = st.slider("Reclamaciones", 0, 6, 1)
    with c3:
        pqr = st.slider("PQR radicadas", 0, 5, 1)
        retrasos = st.slider("Retrasos de pago", 0, 5, 1)

    c4, c5, c6, c7 = st.columns(4)
    genero = c4.selectbox("Género", sorted(datos["genero"].unique()))
    ciudad = c5.selectbox("Ciudad", sorted(datos["ciudad"].unique()))
    tipo = c6.selectbox("Tipo de póliza", sorted(datos["tipo_poliza"].unique()))
    canal = c7.selectbox("Canal de atención", sorted(datos["canal_atencion"].unique()))

    entrada = pd.DataFrame([{
        "edad": edad, "antiguedad_meses": antiguedad, "valor_poliza": valor,
        "reclamaciones": reclamaciones, "pqr": pqr, "retrasos_pago": retrasos,
        "genero": genero, "ciudad": ciudad, "tipo_poliza": tipo,
        "canal_atencion": canal}])

    proba = float(modelo.predict_proba(entrada)[0, 1])
    percentil = float((PROBA_REFERENCIA < proba).mean())
    prioritario = proba >= UMBRAL_DECIL

    st.divider()
    r1, r2, r3 = st.columns([1, 1, 2])
    r1.metric("Riesgo estimado", f"{proba:.1%}")
    r2.metric("Posición en la cartera", f"Percentil {percentil:.0%}")
    if prioritario:
        r3.error("**Entra en el decil de mayor riesgo.** "
                 "Corresponde incluirlo en la lista de contacto preventivo.")
    else:
        r3.info("**Fuera del decil de mayor riesgo.** "
                "No requiere contacto preventivo en este ciclo.")

    # ---------------- Explicabilidad ----------------
    st.subheader("Por qué el modelo llegó a ese resultado")
    st.caption("Cada barra es el aporte de esa característica al riesgo, medido sobre "
               "el log-odds. A la derecha lo aumentan; a la izquierda lo reducen.")

    tabla = explicar(entrada).head(8).iloc[::-1]
    fig, eje = plt.subplots(figsize=(7.5, 3.4))
    colores = [ROJO if v > 0 else AZUL for v in tabla["aporte"]]
    eje.barh(tabla["variable"], tabla["aporte"], color=colores, height=0.6)
    eje.axvline(0, color=TINTA, linewidth=1)
    eje.set_xlabel("Aporte al riesgo estimado", fontsize=9, color=TINTA2)
    for lado in ["top", "right", "left", "bottom"]:
        eje.spines[lado].set_visible(False)
    eje.tick_params(length=0, labelsize=9, colors=TINTA2)
    eje.grid(axis="x", color=REJILLA, linewidth=0.8)
    eje.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Cómo se debe leer esta estimación"):
        st.markdown(
            "- El resultado es una **probabilidad estimada**, no una certeza sobre "
            "el comportamiento futuro de una persona.\n"
            "- Sirve para **priorizar** a quién contactar primero cuando la capacidad "
            "del equipo de retención es limitada. No sirve para negar renovaciones, "
            "encarecer primas ni restringir coberturas.\n"
            "- La precisión del modelo sobre la clase que deserta es baja, de modo que "
            "**la mayoría de las alertas son falsas**. Una alerta no es un juicio "
            "sobre el cliente.\n"
            "- El cliente tiene derecho a conocer y rectificar la información que se "
            "usa para evaluarlo, conforme a la Ley 1581 de 2012.")

# ==================================================================
# 2. Panorama de la cartera
# ==================================================================
with tab_cartera:
    st.subheader("Qué distingue a quien deserta")
    for archivo, pie in [
        ("viz1_comparacion.png",
         "Comparación de las variables de comportamiento entre ambos grupos."),
        ("viz2_distribucion.png",
         "Distribución del riesgo estimado sobre la cartera."),
        ("viz3_composicion.png",
         "Desertores alcanzados según el porcentaje de cartera contactado."),
    ]:
        ruta = BASE / archivo
        if ruta.exists():
            st.image(str(ruta), use_container_width=True)
            st.caption(pie)
            st.write("")

# ==================================================================
# 3. Estado del modelo (monitoreo)
# ==================================================================
with tab_modelo:
    st.subheader("Desempeño registrado en la evaluación")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC en prueba", "0,625")
    m2.metric("ROC-AUC fuera de muestra", "0,722")
    m3.metric("Exhaustividad", "47,8 %")
    m4.metric("Lift del primer decil", "2,98")

    st.caption("El ROC-AUC fuera de muestra proviene de validación cruzada anidada, "
               "que vuelve a seleccionar el hiperparámetro dentro de cada pliegue.")

    st.divider()
    st.subheader("Monitoreo y actualización")
    st.markdown(
        "El modelo se entrenó sobre un corte fijo de la cartera. Su desempeño se "
        "degrada a medida que cambian el comportamiento de los clientes y la mezcla "
        "de productos, de modo que requiere seguimiento.")

    st.markdown("**Señales que obligan a reentrenar**")
    st.dataframe(pd.DataFrame([
        {"Señal": "Deriva en las entradas",
         "Cómo se mide": "Comparar la distribución de las variables evaluadas contra la de entrenamiento",
         "Umbral sugerido": "Diferencia de medias superior a media desviación estándar"},
        {"Señal": "Caída del lift",
         "Cómo se mide": "Tasa real de deserción en el decil contactado frente a la cartera",
         "Umbral sugerido": "Lift por debajo de 2,0 durante dos ciclos seguidos"},
        {"Señal": "Cambio en la tasa base",
         "Cómo se mide": "Tasa de deserción trimestral de la cartera completa",
         "Umbral sugerido": "Variación superior a 3 puntos porcentuales"},
        {"Señal": "Antigüedad del entrenamiento",
         "Cómo se mide": "Meses transcurridos desde el último ajuste",
         "Umbral sugerido": "Reentrenamiento programado cada 6 meses"},
    ]), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Deriva de las variables evaluadas en esta sesión")
    st.caption("Compara lo que se ha ingresado en la aplicación contra la cartera "
               "con la que se entrenó el modelo. Con pocas consultas el resultado es "
               "apenas indicativo.")

    if "historial" not in st.session_state:
        st.session_state.historial = []
    st.session_state.historial.append(entrada.iloc[0].to_dict())
    historial = pd.DataFrame(st.session_state.historial)

    filas = []
    for var in NUMERICAS:
        media_ref = datos[var].mean()
        desv_ref = datos[var].std()
        media_ses = historial[var].mean()
        z = (media_ses - media_ref) / desv_ref if desv_ref else 0
        filas.append({"Variable": var,
                      "Media en entrenamiento": round(media_ref, 2),
                      "Media en esta sesión": round(media_ses, 2),
                      "Desviaciones de diferencia": round(z, 2),
                      "Alerta": "Revisar" if abs(z) > 0.5 else "Normal"})
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
    st.caption(f"Consultas registradas en esta sesión: {len(historial)}")

st.divider()
st.caption("ACA Final · Fundamentos de Inteligencia de Negocios (EAD1039) · "
           "Instructor: Kevin Pérez Cantero · CUN, septiembre de 2026")
