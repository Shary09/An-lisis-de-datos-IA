# Legacy-to-target file map

| Legacy file | Target location / action |
|---|---|
| `DataFrame.py` | `framework_ia/models/data/` (split loading, validation, preprocessing) |
| `model.py` | `framework_ia/models/` (split by domain; remove oversized orchestration responsibilities) |
| `Controller.py` | `framework_ia/controllers/` (split EDA, supervised, unsupervised and agent controllers) |
| `view.py`, `app_lab01.py` | `framework_ia/views/` and root `app.py` (migrate UI page by page) |
| `supervisados.py` | `framework_ia/models/supervised/supervised_models.py` and evaluation modules |
| `analisis_supervisados.py` | `framework_ia/agents/classification_agent.py` after converting outputs to structured data |
| `acp.py` | `framework_ia/models/unsupervised/dimensionality_reduction.py` (PCA implementation to migrate) |
| `clustering.py` | `framework_ia/models/unsupervised/clustering_models.py` and visualization service |
| `guardar_resultados.py` | `framework_ia/services/export_service.py` / experiment persistence |
| `validar_modelos.py` | tests or reproducible experiment scripts |
| `titanic4_2026.csv` | `data/input/` after validating licensing and data provenance |

The legacy files are intentionally retained unchanged under `framework_ia/compatibility` until each migration is tested.
