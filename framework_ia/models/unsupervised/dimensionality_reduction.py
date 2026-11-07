"""Dimensionality reduction methods: PCA, t-SNE, and UMAP.

These methods belong to unsupervised learning, but are kept separate from
clustering because dimensionality reduction and clustering are distinct tasks.
"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def reduce_dimensions(data, method="pca", n_components=2, random_state=42, **kwargs):
    """Scale numeric data and return (embedding, fitted_transformer).

    For t-SNE, perplexity must be lower than the number of input samples.
    UMAP is imported lazily so the rest of the framework works without it.
    """
    import numpy as np
    values = data.select_dtypes(include="number") if hasattr(data, "select_dtypes") else data
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Expected at least two rows and one numeric feature.")
    values = StandardScaler().fit_transform(values)
    method = method.lower().replace("-", "")
    if method == "pca":
        transformer = PCA(n_components=n_components, random_state=random_state, **kwargs)
    elif method in {"tsne", "t-sne"}:
        perplexity = kwargs.pop("perplexity", min(30.0, max(1.0, (len(values)-1)/3)))
        if perplexity >= len(values):
            raise ValueError("t-SNE perplexity must be smaller than the number of samples.")
        transformer = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, **kwargs)
    elif method == "umap":
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError("Install umap-learn to use UMAP: pip install umap-learn") from exc
        transformer = UMAP(n_components=n_components, random_state=random_state, **kwargs)
    else:
        raise ValueError("method must be one of: 'pca', 'tsne', 'umap'.")
    embedding = transformer.fit_transform(values)
    return embedding, transformer
