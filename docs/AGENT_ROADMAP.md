# Agent roadmap

## First milestone: EDA Agent
- Input: structured EDA summary, not screenshots alone.
- Output: findings, evidence, recommendations, limitations.
- Current prototype: deterministic rules in `framework_ia/agents/eda_agent.py`.
- Next: add correlation/outlier/distribution evidence and unit tests.

## Later milestones
- Clustering Agent: interpret PCA/UMAP/t-SNE embeddings, K-Means metrics and HAC linkage/dendrogram metadata.
- Classification Agent: explain validation results, confusion matrices and macro-F1 without inventing causes.
- Regression Agent: compare MAE, RMSE, R² and cross-validation distributions.
- Orchestrator: combine results, identify follow-up analyses and ask for user approval before running them.

LLM integration is deliberately provider-neutral and not required for the deterministic prototype. Configure secrets via environment variables, never commit API keys.
