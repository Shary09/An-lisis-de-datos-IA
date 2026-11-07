# discoverR — Laboratorios 01 y 02 | EIF420O Inteligencia Artificial

Aplicación de Python y Streamlit basada en el framework MVC desarrollado durante el curso. Incluye exploración de datos y aprendizaje no supervisado (Lab 01), además de experimentos de **clasificación supervisada binaria** (Lab 02).

## Estructura del proyecto

| Archivo | Función |
|---|---|
| `app_lab01.py` | Interfaz Streamlit de ambos laboratorios. |
| `DataFrame.py`, `model.py`, `Controller.py`, `view.py` | Framework MVC de clase. La interfaz solicita clasificación al controlador, que delega al modelo y a `Supervisados`. |
| `supervisados.py` | Validación de datos, preparación, entrenamiento, evaluación, configuración estándar y búsqueda de hiperparámetros. |
| `analisis_supervisados.py` | Interpretación narrativa de resultados, matrices y configuraciones. |
| `guardar_resultados.py` | Guardado automático de los experimentos globales por CSV y ejecución. |
| `validar_modelos.py` | Script de experimentación por consola con el CSV de ejemplo; guarda sus resultados en `resultados_lab02/`. |
| `acp.py`, `clustering.py` | Métodos del Laboratorio 01. |
| `titanic4_2026.csv` | CSV de ejemplo para clasificación binaria (`survived`). |
| `requirements.txt` | Dependencias del proyecto. |

## Instalación en Anaconda / Windows

1. Descomprimir el proyecto.
2. Abrir **Anaconda Prompt** y entrar a la carpeta `discoverR`, donde está `app_lab01.py`.
3. Activar un entorno con Python compatible (recomendado 3.10 o superior).
4. Instalar dependencias y ejecutar la interfaz:

```bash
python -m pip install -r requirements.txt
streamlit run app_lab01.py
```

También se puede usar `iniciar_windows.bat` si Python está configurado para ese script. La aplicación se abre normalmente en `http://localhost:8501`.

## Laboratorio 01 — Datos y no supervisados

La interfaz mantiene exploración de datos, estadísticas, correlaciones, ACP, K-Means, HAC y otras visualizaciones. Estas funciones se conservan al agregar el Laboratorio 02.

## Laboratorio 02 — Clasificación supervisada

### Algoritmos

- **KNN (K vecinos más cercanos)**
- **Árbol de Decisión**
- **Random Forest**
- **XGBoost**
- **AdaBoost**

### Pasos en la interfaz

1. Cargar un CSV (por ejemplo `titanic4_2026.csv`).
2. Abrir **Clasificación supervisada**.
3. Elegir como variable objetivo una columna de **exactamente dos clases** (`survived` para Titanic).
4. Elegir **Análisis individual** o **Comparación global (5 algoritmos)**.
5. Ejecutar el experimento y revisar las métricas, configuraciones, matrices de confusión y explicación narrativa.
6. La comparación global guarda automáticamente resultados por dataset y por ejecución en `resultados_lab02/`.

### Metodología y reproducibilidad

- División estratificada de 80 % para entrenamiento y 20 % para prueba; semilla aleatoria `42`.
- Validación cruzada estratificada de cinco particiones.
- Preprocesamiento mediante pipelines: imputación por mediana para numéricos, por moda para categóricos, escalado numérico y codificación one-hot.
- Búsqueda de hiperparámetros con `GridSearchCV`, evaluada mediante **F1 macro**.
- Ejecución adicional del algoritmo con su configuración estándar para comparar con la mejor variación.
- Selección del ganador por F1 macro promedio en validación cruzada; conjunto de prueba reservado para evaluación final.
- Métricas en prueba: accuracy, precision macro, recall macro, F1 macro y matriz de confusión.

### Archivos generados

Por cada ejecución de **comparación global** se crea una carpeta independiente, por ejemplo:

```text
resultados_lab02/
  titanic4_2026/
    <fecha_hora_id_ejecucion>/
      comparacion_final.csv
      variantes_validacion_cruzada.csv
      matriz_*.png
      analisis_completo.txt
      informacion_ejecucion.json
```

El guardado automático desde Streamlit **no sobrescribe** ejecuciones previas. Los archivos que ya venían incluidos en la raíz de `resultados_lab02/` son ejemplos anteriores y no se actualizan al utilizar la interfaz.

### Compatibilidad con CSV

El laboratorio 02 admite otros CSV **de clasificación binaria**. No sirve directamente para regresión o clasificación multiclase. Para evitar fallos frecuentes, el código valida cantidad de clases y registros, elimina las columnas predictoras completamente vacías **según el conjunto de entrenamiento**, convierte infinitos numéricos en valores ausentes y normaliza los tipos mezclados en columnas categóricas. El preprocesamiento se ajusta usando datos de entrenamiento dentro de cada pipeline.

**Limitaciones importantes:** el proyecto no infiere automáticamente qué columnas son identificadores, fechas, texto libre, información filtrada del objetivo ni cuáles podrían causar fuga de datos. Conviene revisarlas antes de entrenar. CSV muy pequeños o variables categóricas de cardinalidad extrema también pueden requerir preparación manual.

### Ejecutar experimentación por consola

Desde la misma carpeta:

```bash
python validar_modelos.py
```

Este script está preparado para `titanic4_2026.csv` y guarda resultados separados de las ejecuciones con Streamlit. Para otro CSV, utiliza la interfaz o adapta los valores `ARCHIVO` y `objetivo` del script.

## Entrega académica

El código **no sustituye el informe**. El Laboratorio 02 pide marco teórico de los cinco algoritmos, experimentación, análisis de modelos y un documento LaTeX/Overleaf con la plantilla del curso. El enunciado menciona los conjuntos Potabilidad y Diabetes: el CSV Titanic es solo el ejemplo disponible en esta entrega, por lo que debe verificarse su aceptación si se utiliza en lugar de los conjuntos indicados.

## Resolución de problemas

- Si aparece `ModuleNotFoundError`, ejecutar `python -m pip install -r requirements.txt` en el mismo entorno que ejecuta Streamlit.
- Si no se reconoce `streamlit`, verificar el entorno activo en Anaconda Prompt.
- Si no se puede entrenar, revisar que la columna objetivo tenga exactamente dos clases, suficientes ejemplos en ambas clases y variables predictoras útiles.
- Los mensajes de error muestran causas habituales relacionadas con datos vacíos o insuficientes.
