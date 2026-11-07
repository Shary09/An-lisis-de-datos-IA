# Agentes científico-técnicos: instalación y API

**Origen de esta edición:** el ZIP generado `Advance_3_Lab03_Enhanced.zip` no estaba disponible como copia recuperable. Por eso esta integración se aplica sobre `Advance_3_MVC_Framework_Migrated.zip` (base original). No reemplaza los añadidos de benchmarking y otras correcciones de la versión más reciente. Para integrar sobre ella de forma segura, debe facilitarse ese ZIP original.

## Instalar y ejecutar

```powershell
cd AI_Analytics_Framework_MVC_Lab3
python -m pip install -r requirements.txt
streamlit run app.py
```

## Dónde poner tu API key

1. En la carpeta raíz del proyecto, entra en `.streamlit/`.
2. Copia `secrets.toml.example` a `secrets.toml`.
3. Abre **`.streamlit/secrets.toml`** y sustituye el valor de `OPENAI_API_KEY` por tu clave real.
4. NO envíes `secrets.toml` ni la clave por WhatsApp, capturas, GitHub, ni por este chat. Está incluido en `.gitignore`.

Ejemplo **ficticio** de contenido:

```toml
OPENAI_API_KEY = "sk-tu-clave-aqui"
OPENAI_MODEL = "gpt-4.1-mini"
```

Alternativamente configura la variable de entorno `OPENAI_API_KEY` en Windows antes de abrir Streamlit.

## Uso

1. Carga un CSV y ejecuta el tipo de análisis deseado.
2. Abre el panel `Análisis técnico-científico con agentes`.
3. Elige EDA, Regresión, Clasificación o Clustering.
4. Activa `Generar también interpretación con OpenAI` si has configurado la API.
5. Pulsa `Interpretar resultados`. **Cada clic con OpenAI habilitado puede facturarse.**

La IA interpreta métricas y resúmenes; no recibe filas completas del CSV, no entrena, no hace predicciones y no corrige resultados numéricos. Los cálculos siguen en los modelos. El respaldo determinista funciona sin API y sigue disponible cuando la API falla.

**Advertencia sobre esta base:** su motor de regresión original solo devuelve métricas de prueba y no incluye CV. Por eso el agente señala que el menor RMSE de prueba no es una selección definitiva válida. El benchmarking CV del ZIP Enhanced debe conservarse al trasladar esta implementación.

## Archivos principales

- `framework_ia/services/openai_interpretation_service.py`: cliente Responses API, contexto limitado, manejo de errores.
- `framework_ia/agents/scientific_agent.py`: comparación científica y reglas de selección, sin estimadores.
- `framework_ia/agents/{eda,classification,regression,clustering}_agent.py`: agentes por especialidad.
- `framework_ia/agents/agent_orchestrator.py`: enrutamiento.
- `framework_ia/compatibility/app_lab01.py`: panel de Streamlit.
- `tests/test_scientific_agents.py`: validaciones sin costo de API.
