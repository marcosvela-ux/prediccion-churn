# Estimador de riesgo de deserción de clientes

Aplicación del ACA Final de Fundamentos de Inteligencia de Negocios (EAD1039).
Corporación Unificada Nacional de Educación Superior, Especialización en Analítica
Avanzada de Datos.

**Integrantes:** Marcos Yair Vela Pulido, Jerszinho Fray Rafael García Forero,
Fabio Ariel Cabrera Estrella.
**Instructor:** Kevin Pérez Cantero. Septiembre de 2026.

## Qué hace

Toma los datos de un cliente de la aseguradora y estima su probabilidad de
deserción con el modelo de regresión logística entrenado en la ACA 1. Además de
la cifra, muestra qué características empujaron el resultado hacia arriba y
cuáles hacia abajo, y ubica al cliente dentro de la cartera.

La aplicación tiene tres pestañas:

- **Evaluar un cliente.** Formulario, estimación de riesgo, posición en la
  cartera y descomposición del resultado por variable.
- **Panorama de la cartera.** Las tres visualizaciones del informe.
- **Estado del modelo.** Métricas de la evaluación y las señales que obligarían
  a reentrenar.

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `app.py` | Código de la aplicación |
| `modelo_churn.pkl` | Pipeline completo entrenado: imputación, escalado, codificación y clasificador |
| `clientes_churn.csv` | Cartera de referencia, usada para ubicar al cliente en percentiles |
| `viz1_comparacion.png` | Comparación entre quien deserta y quien permanece |
| `viz2_distribucion.png` | Distribución del riesgo estimado |
| `viz3_composicion.png` | Curva de ganancia acumulada |
| `requirements.txt` | Versiones de las librerías |

El `.pkl` guarda el pipeline entero, con imputación, escalado y codificación
adentro. La aplicación le pasa los datos crudos del formulario y el objeto hace
el resto igual que en el entrenamiento. Guardando solo el clasificador habría
que repetir esos pasos a mano en la app, y cualquier diferencia daría
predicciones equivocadas sin avisar.

## Ejecutar en el computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre en http://localhost:8501

## Publicar en Streamlit Cloud

1. Subir esta carpeta a un repositorio de GitHub
2. Entrar a https://share.streamlit.io e iniciar sesión con GitHub
3. **New app**, elegir el repositorio y poner `app.py` como archivo principal
4. **Deploy**

El despliegue tarda unos minutos la primera vez. Streamlit Cloud lee
`requirements.txt` e instala las dependencias solo.

## Advertencia de uso

El resultado es una probabilidad estimada, no una certeza sobre el
comportamiento futuro de una persona. Sirve para priorizar a quién contactar
primero cuando la capacidad del equipo de retención es limitada. No debe usarse
para negar renovaciones, encarecer primas ni restringir coberturas. La precisión
del modelo sobre la clase que deserta es baja, de modo que la mayoría de las
alertas son falsas y ninguna constituye un juicio sobre el cliente. El
tratamiento de datos personales se rige por la Ley 1581 de 2012.
