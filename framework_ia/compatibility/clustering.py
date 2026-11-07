"""
Módulo de métodos no supervisados.
Incluye K-Means, HAC, PCA (sklearn), t-SNE, UMAP y utilidades de evaluación/visualización.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil, floor, pi
from seaborn import color_palette
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class Clustering:
    """
    Clase de métodos no supervisados: K-Means, HAC, PCA, t-SNE, UMAP
    y gráficos asociados.
    """

    # ------------------------------------------------------------------
    # Utilidades generales
    # ------------------------------------------------------------------
    @staticmethod
    def centroide(num_cluster, datos, clusters):
        ind = clusters == num_cluster
        return pd.DataFrame(datos[ind].mean()).T

    @staticmethod
    def recodificar(col, nuevo_codigo):
        col_cod = pd.Series(col, copy=True)
        for llave, valor in nuevo_codigo.items():
            col_cod.replace(llave, valor, inplace=True)
        return col_cod

    @staticmethod
    def evaluar_clustering(datos, etiquetas):
        """
        Calcula métricas de evaluación interna de clustering.
        Retorna un diccionario con silhouette, calinski-harabasz y davies-bouldin.
        """
        n_clusters = len(np.unique(etiquetas))
        if n_clusters < 2 or n_clusters >= len(datos):
            return {
                "silhouette": None,
                "calinski_harabasz": None,
                "davies_bouldin": None,
                "n_clusters": n_clusters
            }
        return {
            "silhouette": float(silhouette_score(datos, etiquetas)),
            "calinski_harabasz": float(calinski_harabasz_score(datos, etiquetas)),
            "davies_bouldin": float(davies_bouldin_score(datos, etiquetas)),
            "n_clusters": n_clusters
        }

    # ------------------------------------------------------------------
    # Visualizaciones
    # ------------------------------------------------------------------
    @staticmethod
    def bar_plot(centros, labels, scale=False, cluster=None, var=None):
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=120)
        centros = np.copy(centros)

        if scale:
            for col in range(centros.shape[1]):
                max_val = max(centros[:, col]) if max(centros[:, col]) != 0 else 1
                centros[:, col] = centros[:, col] / max_val

        colores = color_palette()
        minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0

        def inside_plot(valores, labels, titulo):
            plt.barh(range(len(valores)), valores, 1/1.5, color=colores)
            plt.xlim(minimo, ceil(centros.max()) + 0.1)
            plt.title(titulo)

        if var is not None:
            centros = np.array([n[[x in var for x in labels]] for n in centros])
            colores = [colores[x % len(colores)] for x, i in enumerate(labels) if i in var]
            labels = labels[[x in var for x in labels]]

        if cluster is None:
            for i in range(centros.shape[0]):
                plt.subplot(1, centros.shape[0], i + 1)
                inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
                plt.yticks(range(len(labels)), labels) if i == 0 else plt.yticks([])
        else:
            pos = 1
            for i in cluster:
                plt.subplot(1, len(cluster), pos)
                inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
                plt.yticks(range(len(labels)), labels) if pos == 1 else plt.yticks([])
                pos += 1
        plt.tight_layout()
        plt.show()

    @staticmethod
    def radar_plot(centros, labels):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=120)
        centros = np.array([
            ((n - min(n)) / (max(n) - min(n)) * 100) if max(n) != min(n) else (n / n * 50)
            for n in centros.T
        ])
        angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
        angulos += angulos[:1]
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angulos[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks(
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
            color="grey", size=8
        )
        plt.ylim(-10, 100)

        for i in range(centros.shape[1]):
            valores = centros[:, i].tolist()
            valores += valores[:1]
            ax.plot(angulos, valores, linewidth=1.5, linestyle='solid', label='Cluster ' + str(i))
            ax.fill(angulos, valores, alpha=0.25)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dendrograma(Z, titulo="Dendrograma HAC", max_d=None, truncate_mode=None, p=30):
        """
        Dibuja el dendrograma a partir de la matriz de linkage.
        """
        plt.figure(figsize=(12, 6), dpi=120)
        plt.title(titulo)
        plt.xlabel("Índice de muestra o (tamaño del clúster)")
        plt.ylabel("Distancia")
        dendrogram(
            Z,
            truncate_mode=truncate_mode,
            p=p,
            leaf_rotation=90.,
            leaf_font_size=8.,
            show_contracted=True
        )
        if max_d is not None:
            plt.axhline(y=max_d, c='r', ls='--', label=f'Corte en {max_d:.2f}')
            plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # K-Means
    # ------------------------------------------------------------------
    @staticmethod
    def kmeans(datos, n_clusters=3, init="k-means++", n_init=10, max_iter=300,
               random_state=42, visualizar=True):
        """
        Aplica K-Means clustering.
        Parámetros:
            - n_clusters: número de clústeres
            - init: método de inicialización ('k-means++', 'random')
            - n_init: número de inicializaciones
            - max_iter: máximo de iteraciones
            - random_state: semilla
            - visualizar: si True muestra bar_plot y radar_plot
        Retorna: etiquetas, métricas, modelo
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        etiquetas = kmeans.fit_predict(datos)
        metricas = Clustering.evaluar_clustering(datos, etiquetas)

        if visualizar:
            Clustering.bar_plot(kmeans.cluster_centers_, datos.columns)
            Clustering.radar_plot(kmeans.cluster_centers_, datos.columns)

        return etiquetas, metricas, kmeans

    @staticmethod
    def kmeans_elbow(datos, k_range=range(2, 11), random_state=42):
        """
        Calcula inercias y silhouette para el método del codo / silhouette.
        Retorna DataFrame con resultados y genera gráfico.
        """
        resultados = []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            etiq = km.fit_predict(datos)
            sil = silhouette_score(datos, etiq) if k > 1 else None
            resultados.append({
                "k": k,
                "inercia": km.inertia_,
                "silhouette": sil
            })
        df_res = pd.DataFrame(resultados)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
        axes[0].plot(df_res["k"], df_res["inercia"], marker="o")
        axes[0].set_xlabel("Número de clústeres (k)")
        axes[0].set_ylabel("Inercia")
        axes[0].set_title("Método del codo")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df_res["k"], df_res["silhouette"], marker="o", color="green")
        axes[1].set_xlabel("Número de clústeres (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette vs k")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return df_res

    # ------------------------------------------------------------------
    # HAC - Agrupamiento Jerárquico
    # ------------------------------------------------------------------
    @staticmethod
    def hac(datos, n_clusters=3, linkage_method="ward", metric="euclidean",
            distance_threshold=None, visualizar=True, plot_dendro=True):
        """
        Aplica Agrupamiento Jerárquico Aglomerativo (HAC).
        Parámetros:
            - n_clusters: número de clústeres (si distance_threshold es None)
            - linkage_method: 'ward', 'complete', 'average', 'single'
            - metric: métrica de distancia (ignorada si linkage='ward')
            - distance_threshold: umbral de distancia (alternativa a n_clusters)
            - visualizar: muestra bar_plot y radar de centroides
            - plot_dendro: muestra dendrograma
        Retorna: etiquetas, métricas, modelo, matriz de linkage Z
        """
        # Para ward solo se admite euclidean
        if linkage_method == "ward":
            metric = "euclidean"

        # Matriz de linkage para dendrograma (scipy)
        if metric == "euclidean":
            Z = linkage(datos, method=linkage_method, metric=metric)
        else:
            dist_matrix = pdist(datos, metric=metric)
            Z = linkage(dist_matrix, method=linkage_method)

        # Modelo sklearn
        if distance_threshold is not None:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage=linkage_method,
                metric=metric if linkage_method != "ward" else "euclidean"
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
                metric=metric if linkage_method != "ward" else "euclidean"
            )

        etiquetas = model.fit_predict(datos)
        metricas = Clustering.evaluar_clustering(datos, etiquetas)

        # Centroides aproximados (media de cada clúster)
        centros = []
        for c in sorted(np.unique(etiquetas)):
            centros.append(datos[etiquetas == c].mean().values)
        centros = np.array(centros)

        if plot_dendro:
            Clustering.plot_dendrograma(
                Z,
                titulo=f"Dendrograma HAC (linkage={linkage_method})",
                truncate_mode="lastp",
                p=30
            )

        if visualizar and len(centros) > 0:
            Clustering.bar_plot(centros, datos.columns)
            Clustering.radar_plot(centros, datos.columns)

        return etiquetas, metricas, model, Z

    @staticmethod
    def hac_completo(datos, n_clusters=3):
        """
        HAC al estilo del paquete del profesor (PaqNOSup.HAC):
        1. Dendrogramas de ward, average, single y complete
        2. Clustering con ward y k fijo + bar_plot de centroides
        3. Para cada linkage: Silhouette con k=2..10 y gráfico
        4. Retorna tabla benchmark (algoritmo, mejor k, silhouette)
        """
        from scipy.cluster.hierarchy import ward, average, single, complete, fcluster

        # 1. Dendrogramas (4 métodos)
        metodos_scipy = [
            ("ward", ward(datos)),
            ("average", average(datos)),
            ("single", single(datos)),
            ("complete", complete(datos)),
        ]
        for nombre, res in metodos_scipy:
            print(f"Método {nombre}:")
            Clustering.plot_dendrograma(
                res,
                titulo=f"Dendrograma HAC – {nombre}",
                truncate_mode="lastp",
                p=30,
            )

        # 2. Clustering final con ward y n_clusters fijo (como el profesor)
        grupos = fcluster(
            linkage(datos, method="ward", metric="euclidean"),
            n_clusters,
            criterion="maxclust",
        )
        grupos = grupos - 1  # pasar a 0-index

        centros = []
        for c in range(n_clusters):
            centros.append(Clustering.centroide(c, datos, grupos).values.ravel())
        centros = np.array(centros)

        Clustering.bar_plot(centros, datos.columns, scale=True)
        Clustering.radar_plot(centros, datos.columns)

        metricas_ward = Clustering.evaluar_clustering(datos, grupos)

        # 3. Silhouette por linkage (k = 2..10), como el profesor
        metodos = ["ward", "average", "single", "complete"]
        benchmark = []
        for metodo in metodos:
            silhouette_scores = []
            for k in range(2, 11):
                modelo = AgglomerativeClustering(
                    n_clusters=k, linkage=metodo, metric="euclidean"
                )
                etiquetas = modelo.fit_predict(datos)
                score = silhouette_score(datos, etiquetas)
                silhouette_scores.append(score)

            mejor_k = int(np.argmax(silhouette_scores) + 2)
            mejor_score = float(max(silhouette_scores))

            plt.figure(figsize=(10, 6), dpi=120)
            plt.plot(range(2, 11), silhouette_scores, marker="o")
            plt.title(f"Silhouette Score ({metodo})")
            plt.xlabel("Número de clusters")
            plt.ylabel("Silhouette Score")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            benchmark.append({
                "Algoritmo": f"HAC-{metodo}",
                "Num de Clusters": mejor_k,
                "Silhouette Score": round(mejor_score, 4),
            })

        df_benchmark = pd.DataFrame(benchmark)
        print("\n=== Benchmark HAC (estilo profesor) ===")
        print(df_benchmark.to_string(index=False))

        return grupos, metricas_ward, df_benchmark


    # ------------------------------------------------------------------
    # PCA (sklearn)
    # ------------------------------------------------------------------
    @staticmethod
    def pca(datos, n_components=2, whiten=False, svd_solver="auto", random_state=42):
        """PCA con sklearn. Retorna coords, modelo, df de varianza."""
        modelo = PCA(
            n_components=n_components,
            whiten=whiten,
            svd_solver=svd_solver,
            random_state=random_state
        )
        coords = modelo.fit_transform(datos)
        var_exp = modelo.explained_variance_ratio_ * 100
        df_coords = pd.DataFrame(
            coords,
            index=datos.index if hasattr(datos, "index") else None,
            columns=[f"PC{i+1}" for i in range(coords.shape[1])]
        )
        df_var = pd.DataFrame({
            "Componente": [f"PC{i+1}" for i in range(len(var_exp))],
            "Varianza explicada (%)": np.round(var_exp, 2),
            "Varianza acumulada (%)": np.round(np.cumsum(var_exp), 2)
        })
        return df_coords, modelo, df_var

    @staticmethod
    def pca_variaciones(datos, configs=None):
        """Prueba varias configuraciones de PCA y retorna tabla comparativa."""
        if configs is None:
            configs = [
                {"n_components": 2, "whiten": False, "svd_solver": "auto", "nombre": "estándar"},
                {"n_components": 3, "whiten": False, "svd_solver": "auto", "nombre": "3 componentes"},
                {"n_components": 2, "whiten": True, "svd_solver": "auto", "nombre": "whiten=True"},
                {"n_components": 2, "whiten": False, "svd_solver": "full", "nombre": "svd_solver=full"},
                {"n_components": 0.95, "whiten": False, "svd_solver": "full", "nombre": "95% varianza"},
            ]
        resultados = []
        for cfg in configs:
            nombre = cfg.pop("nombre", "config")
            try:
                coords, modelo, df_var = Clustering.pca(datos, **cfg)
                var_total = df_var["Varianza explicada (%)"].sum()
                resultados.append({
                    "config": nombre,
                    "n_components": modelo.n_components_,
                    "varianza_total (%)": round(var_total, 2),
                    "var_PC1 (%)": df_var.iloc[0]["Varianza explicada (%)"] if len(df_var) > 0 else None,
                    "var_PC2 (%)": df_var.iloc[1]["Varianza explicada (%)"] if len(df_var) > 1 else None,
                })
            except Exception as e:
                resultados.append({"config": nombre, "error": str(e)})
            finally:
                cfg["nombre"] = nombre
        return pd.DataFrame(resultados)

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------
    @staticmethod
    def tsne(datos, n_components=2, perplexity=30, learning_rate="auto",
             n_iter=1000, metric="euclidean", init="pca", random_state=42,
             visualizar=True, color_by=None):
        """Ejecuta t-SNE. Retorna coordenadas (DataFrame) y modelo."""
        n_samples = len(datos)
        max_perp = max(5, (n_samples - 1) // 3)
        if perplexity >= max_perp:
            perplexity = min(30, max_perp)

        modelo = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=n_iter,
            metric=metric,
            init=init,
            random_state=random_state,
            n_jobs=-1
        )
        coords = modelo.fit_transform(datos)
        df_coords = pd.DataFrame(
            coords,
            index=datos.index if hasattr(datos, "index") else None,
            columns=[f"TSNE{i+1}" for i in range(n_components)]
        )
        if visualizar:
            Clustering._scatter_2d(
                df_coords, titulo=f"t-SNE (perplexity={perplexity})", color_by=color_by
            )
        return df_coords, modelo

    @staticmethod
    def tsne_variaciones(datos, configs=None, color_by=None):
        """Varias configuraciones de t-SNE."""
        if configs is None:
            configs = [
                {"perplexity": 5, "n_iter": 1000, "init": "pca", "nombre": "perp=5"},
                {"perplexity": 15, "n_iter": 1000, "init": "pca", "nombre": "perp=15"},
                {"perplexity": 30, "n_iter": 1000, "init": "pca", "nombre": "estándar (perp=30)"},
                {"perplexity": 50, "n_iter": 1000, "init": "pca", "nombre": "perp=50"},
                {"perplexity": 30, "n_iter": 1500, "init": "random", "nombre": "init=random"},
            ]
        resultados = []
        for cfg in configs:
            nombre = cfg.pop("nombre", "config")
            try:
                coords, modelo = Clustering.tsne(datos, visualizar=False, **cfg)
                kl = getattr(modelo, "kl_divergence_", None)
                resultados.append({
                    "config": nombre,
                    "perplexity": cfg.get("perplexity"),
                    "n_iter": cfg.get("n_iter", 1000),
                    "init": cfg.get("init", "pca"),
                    "kl_divergence": round(kl, 4) if kl is not None else None
                })
                Clustering._scatter_2d(coords, titulo=f"t-SNE – {nombre}", color_by=color_by)
            except Exception as e:
                resultados.append({"config": nombre, "error": str(e)})
            finally:
                cfg["nombre"] = nombre
        return pd.DataFrame(resultados)

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------
    @staticmethod
    def umap_reduce(datos, n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="euclidean", random_state=42, visualizar=True,
                    color_by=None):
        """Ejecuta UMAP. Retorna coordenadas y modelo."""
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn no está instalado. pip install umap-learn")
        modelo = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        coords = modelo.fit_transform(datos)
        df_coords = pd.DataFrame(
            coords,
            index=datos.index if hasattr(datos, "index") else None,
            columns=[f"UMAP{i+1}" for i in range(n_components)]
        )
        if visualizar:
            Clustering._scatter_2d(
                df_coords,
                titulo=f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})",
                color_by=color_by
            )
        return df_coords, modelo

    @staticmethod
    def umap_variaciones(datos, configs=None, color_by=None):
        """Varias configuraciones de UMAP."""
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn no está instalado.")
        if configs is None:
            configs = [
                {"n_neighbors": 5, "min_dist": 0.1, "nombre": "n_neighbors=5"},
                {"n_neighbors": 15, "min_dist": 0.1, "nombre": "estándar (15, 0.1)"},
                {"n_neighbors": 30, "min_dist": 0.1, "nombre": "n_neighbors=30"},
                {"n_neighbors": 15, "min_dist": 0.0, "nombre": "min_dist=0.0"},
                {"n_neighbors": 15, "min_dist": 0.5, "nombre": "min_dist=0.5"},
            ]
        resultados = []
        for cfg in configs:
            nombre = cfg.pop("nombre", "config")
            try:
                coords, modelo = Clustering.umap_reduce(datos, visualizar=False, **cfg)
                resultados.append({
                    "config": nombre,
                    "n_neighbors": cfg.get("n_neighbors"),
                    "min_dist": cfg.get("min_dist"),
                })
                Clustering._scatter_2d(coords, titulo=f"UMAP – {nombre}", color_by=color_by)
            except Exception as e:
                resultados.append({"config": nombre, "error": str(e)})
            finally:
                cfg["nombre"] = nombre
        return pd.DataFrame(resultados)

    # ------------------------------------------------------------------
    # Visualización 2D auxiliar
    # ------------------------------------------------------------------
    @staticmethod
    def _scatter_2d(df_coords, titulo="Proyección 2D", color_by=None):
        plt.figure(figsize=(8, 6), dpi=120)
        x = df_coords.iloc[:, 0]
        y = df_coords.iloc[:, 1]
        if color_by is not None:
            if isinstance(color_by, (pd.Series, pd.DataFrame)):
                color_by = color_by.values.ravel()
            scatter = plt.scatter(x, y, c=color_by, cmap="tab10", alpha=0.7, s=20)
            plt.colorbar(scatter, label="Grupo / etiqueta")
        else:
            plt.scatter(x, y, alpha=0.6, s=20, c="steelblue")
        plt.xlabel(df_coords.columns[0])
        plt.ylabel(df_coords.columns[1])
        plt.title(titulo)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
