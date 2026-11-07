# Agentes por apartado de Advance 3

Cada apartado analítico presenta **Generar análisis con OpenAI** y permite descargar un **PDF del apartado** con el informe y las evidencias cuantitativas. La guía de utilización no es un análisis, por lo que no tiene agente.

| Apartado | Agente | Evidencia recopilada |
|---|---|---|
| Datos | DataQualityAgent | tipos, faltantes, duplicados, valores únicos |
| Estadísticas básicas | DescriptiveStatisticsAgent | descriptivas, distribución, boxplots, correlaciones, categorías |
| ACP | PCAAgent | varianza, proyección, aportes de variables |
| K-Means | KMeansAgent | silhouette/inercia, centroides, tamaños |
| HAC | HACAgent | dendrograma (configuración), enlace, grupos, centroides |
| t-SNE | TSNEAgent | perplexity, KL y resumen de proyección |
| UMAP | UMAPAgent | n_neighbors, min_dist, resumen de proyección |
| Clasificación supervisada | ClassificationSectionAgent | métricas CV/prueba, matrices de confusión, variantes |
| Regresión supervisada | RegressionSectionAgent | benchmarking, MAE, RMSE, R² y CV cuando exista |

Se envían **estadísticas y métricas agregadas**, no registros del archivo CSV ni píxeles de los gráficos. El informe interpreta las evidencias disponibles; no afirma analizar imágenes directamente. En los métodos de proyección, que carecen de métricas robustas de separación, el agente debe reconocer el límite.

## API

En la raíz del proyecto crea `.streamlit/secrets.toml` con:

```toml
OPENAI_API_KEY = "sk-tu-clave-aqui"
OPENAI_MODEL = "gpt-4.1-mini"
```

No publiques la clave ni subas el archivo a repositorios. Puedes usar alternativamente variables de entorno. La generación solo se inicia con el botón de cada apartado; no se ejecuta al navegar.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Limitación de base:** esta entrega modifica la última versión que estaba accesible, `Advance_3_OpenAI_Agents_Integrated`, basada en el proyecto original MVC. Para conservar las mejoras del ZIP `Advance_3_Lab03_Enhanced` se necesita fusionar sus cambios cuando esté disponible su archivo real.
