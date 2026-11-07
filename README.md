# AI Analytics Framework — Advance 3 (MVC)

This advance reorganizes the project into an MVC-oriented structure while keeping the original Streamlit dashboard as the application entry point. The original source is located in `framework_ia/compatibility/` during gradual migration; there is no `legacy/lab2` directory.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Corrections included in this advance

- Replaced the implementation-oriented `describe()` heading with **Estadísticas descriptivas** and a plain-language explanation.
- Added help text explaining the difference between analyzing raw data and applying selected preparation/cleaning operations.
- Added a tooltip explaining `k-means++` versus `random`.
- Improved the PCA individuals plot with a distinct palette and clarified what the overlay represents.
- Kept the previous dashboard as the active UI to avoid replacing its layout during migration.

## Architecture

- `framework_ia/models/data/`: data loading and preparation
- `framework_ia/models/eda/`: exploratory analysis
- `framework_ia/models/supervised/`: supervised models
- `framework_ia/models/unsupervised/`: PCA, UMAP, t-SNE, K-Means and HAC
- `framework_ia/models/evaluation/`: metrics and benchmarking
- `framework_ia/controllers/`: coordination between view and models
- `framework_ia/views/`: interface components and pages
- `framework_ia/agents/`: agent base, EDA and specialized agent scaffolds
- `framework_ia/services/`: visualization, experiment and export services
- `framework_ia/compatibility/`: original source retained during migration; this is not the final MVC destination
- `outputs/previous_results/`: previous result artifacts

## Migration status

This is an incremental migration. The existing dashboard is preserved as the active interface, and its source has been moved to a neutral compatibility package so it can continue to run while features are extracted into MVC modules. The full dashboard has not yet been rewritten to call every new MVC module. Regression benchmarking and agent-driven interpretation require further integration and end-to-end validation.

## Avance 3: migración MVC y regresión

La interfaz Streamlit existente se mantiene como entrada de la aplicación. `app.py` ejecuta el dashboard original preservando su estilo y navegación; los módulos nuevos se incorporan gradualmente para no reemplazarlo por una pantalla de prueba.

### Arquitectura

- `framework_ia/models/data/`: carga y preparación de datos.
- `framework_ia/models/eda/`: estadísticas y análisis exploratorio.
- `framework_ia/models/supervised/`: registro de modelos de clasificación y regresión, además del benchmarking de regresión.
- `framework_ia/models/unsupervised/`: reducción dimensional (PCA, t-SNE, UMAP) y clustering (K-Means, HAC).
- `framework_ia/models/evaluation/`: métricas de evaluación.
- `framework_ia/controllers/`: coordinación de operaciones.
- `framework_ia/views/`: presentación y componentes de interfaz.
- `framework_ia/agents/`: agentes interpretativos iniciales y orquestador.
- `framework_ia/services/`: visualización, exportación y experimentos.
- `framework_ia/compatibility/`: módulos preexistentes que aún utiliza la interfaz durante la migración. Esta carpeta no se llama `legacy` ni `lab2`; se mantiene temporalmente para no romper imports. La migración puede completar su eliminación cuando cada dependencia se haya trasladado y probado.

### Correcciones del dashboard

- Estadísticas descriptivas: el título visible es descriptivo y no expone el nombre interno `describe()`.
- Preparación y limpieza: explica la diferencia entre analizar los datos originales y aplicar transformaciones a una copia, además de comparar filas, columnas, nulos y duplicados.
- PCA: el plano de individuos usa una paleta distintiva y permite colorear por una variable seleccionada; la sobreposición mantiene diferenciados individuos y variables.
- K-Means: explica `k-means++` y `random` en el selector.
- Clasificación: los predictores pueden ser numéricos, binarios o categóricos según la preparación del algoritmo; la variable objetivo define las clases.
- Regresión: nueva sección con los ocho algoritmos solicitados, preprocesamiento dentro de pipelines para evitar ajustar transformaciones con el conjunto de prueba, métricas MAE/MSE/RMSE/R² e interpretación inicial del agente.

### Ejecución

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

### Pruebas

```bash
python -m pytest -q
```

El benchmarking usa una partición de prueba reproducible, pero antes de afirmar que un modelo es el mejor para uso real se recomienda añadir validación cruzada sistemática y diagnóstico de residuos. La interpretación agéntica actual es una primera versión basada en resultados estructurados; no requiere proveedor externo de LLM.
