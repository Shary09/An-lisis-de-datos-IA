# Architecture and migration plan

## Principles
- MVC is the application structure: models contain analysis/algorithm logic, views render the UI, and controllers coordinate requests.
- Supervised learning contains classification and regression; they remain independent modules.
- Unsupervised learning contains dimensionality reduction (PCA, UMAP, t-SNE) and clustering (K-Means, HAC).
- EDA is an independent analysis capability and does not require model training.
- Visualization consumes results and does not train algorithms.
- Agents interpret structured results and should cite the metrics/data they use. They do not replace the statistical models.

## Target package map
- `framework_ia/models/data`: loading and data preparation.
- `framework_ia/models/eda`: exploratory analysis.
- `framework_ia/models/supervised/supervised_models.py`: supervised estimator factories. Lab 2 classification and Lab 3 regression are separate classes in one module.
- `framework_ia/models/unsupervised`: dimensionality reduction and clustering.
- `framework_ia/models/evaluation`: task-specific metrics and benchmarking.
- `framework_ia/views`: Streamlit pages and reusable UI components.
- `framework_ia/controllers`: orchestration between views and model services.
- `framework_ia/agents`: specialized interpretation agents and prompts.
- `framework_ia/services`: plotting, export and experiment persistence.

## UMAP and t-SNE
UMAP and t-SNE belong in `models/unsupervised/dimensionality_reduction.py`, alongside PCA as dimensionality-reduction / embedding techniques. They are not clustering algorithms. Their two-dimensional embeddings can be visualized and then optionally colored by K-Means/HAC cluster labels. The visualization service should label axes as `UMAP 1/2` or `t-SNE 1/2`; unlike PCA axes, they do not directly represent explained variance.

## Migration strategy
1. Keep `framework_ia/compatibility` as the known-good reference until regression tests confirm equivalent behavior.
2. Move one capability at a time and add tests before deleting or renaming legacy modules.
3. Preserve the Lab 2 classification workflow; add the eight Lab 3 regressors in the supervised model registry.
4. Define stable result schemas before connecting LLM-backed agents.
5. Do not allow an agent to silently alter data or launch expensive experiments; make recommendations and require explicit user action for consequential operations.

## Current state of this ZIP
This is an architectural starter, not a claim that every legacy workflow has already been migrated. The original Lab 2 source, dataset, reports and example outputs are preserved under `framework_ia/compatibility`. The EDA summary and deterministic EDA agent are a working first prototype. Classification/regression integration, remaining dashboards and migration tests are next steps.
