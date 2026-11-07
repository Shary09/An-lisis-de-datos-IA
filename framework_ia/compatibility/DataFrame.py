import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from acp import ACP
from clustering import Clustering

class DataFrame:
    """
    Clase personalizada que encapsula un pandas.DataFrame
    y expone métodos de EDA, transformaciones, ACP, clustering,
    t-SNE y UMAP.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.__df = dataframe.copy()

    def __str__(self):
        return str(self.__df.info())

    # --- Métodos básicos ---
    def mostrar(self):
        return self.__df

    def primeras_filas(self, n=10):
        return self.__df.head(n)

    def ultimas_filas(self, n=10):
        return self.__df.tail(n)

    def obtener_fila(self, indice):
        return self.__df.iloc[indice]

    def obtener_columna(self, nombre):
        return self.__df[nombre]

    def dimensiones(self):
        return self.__df.shape

    def nombres_columnas(self):
        return list(self.__df.columns)

    def tipos_datos(self):
        return self.__df.dtypes

    # --- Limpieza ---
    def contar_nulos(self):
        return self.__df.isnull().sum()

    def eliminar_nulos(self):
        return self.__df.dropna()

    def reemplazar_nulos(self, valor=0):
        return self.__df.fillna(valor)

    def detectar_duplicados(self):
        return self.__df[self.__df.duplicated()]

    def eliminar_duplicados(self):
        return self.__df.drop_duplicates()

    # --- Estadísticas y correlación ---
    def estadisticas(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        return datos_num.describe()

    def correlacion(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        return datos_num.corr()

    # --- Visualizaciones ---
    def histogramas(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        datos_num.hist(figsize=(10, 8))
        plt.show()

    def boxplots(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        datos_num.plot(kind="box", figsize=(10, 6))
        plt.show()
        
    def scatterplots(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        sns.pairplot(datos_num)
        plt.show()
        
    def mapa_calor(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(datos_num.corr(), annot=True, cmap="coolwarm")
        plt.show()

    # --- Transformaciones ---
    def normalizar(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        return (datos_num - datos_num.min()) / (datos_num.max() - datos_num.min())
        
    def estandarizar(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        return (datos_num - datos_num.mean()) / datos_num.std()
        
    def imputar_nulos(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        return datos_num.fillna(datos_num.mean())

    # --- Selección y exportación ---
    def seleccionar_variables(self, columnas):
        return self.__df[columnas]

    def guardar_csv(self, ruta="dataframe_guardado.csv"):
        self.__df.to_csv(ruta, index=False)
        return ruta

    def exportar_resultados(self, ruta="resultados_eda.csv"):
        self.__df.describe().to_csv(ruta)
        return ruta

    # --- Datos numéricos preparados ---
    def obtener_datos_numericos(self, estandarizar=True, imputar=True):
        """
        Retorna solo columnas numéricas, opcionalmente imputadas y estandarizadas.
        """
        datos = self.__df.select_dtypes(include=["float64", "int64", "float32", "int32"]).copy()
        if imputar:
            datos = datos.fillna(datos.mean())
        if estandarizar:
            datos = (datos - datos.mean()) / datos.std()
        return datos

    # --- ACP (prince + sklearn) ---
    def aplicar_acp(self, n_componentes=2, usar_prince=True, estandarizar=True):
        """
        Ejecuta ACP.
        - Si usar_prince=True usa la clase ACP (prince) y retorna el objeto completo.
        - Si usar_prince=False usa sklearn y retorna coordenadas + varianza.
        """
        datos_num = self.obtener_datos_numericos(estandarizar=estandarizar)

        if usar_prince:
            acp = ACP(datos_num, n_componentes=n_componentes)
            return acp
        else:
            coords, modelo, df_var = Clustering.pca(
                datos_num, n_components=n_componentes
            )
            return coords, modelo, df_var

    def acp_variaciones(self):
        """Prueba varias configuraciones de PCA (sklearn) y retorna tabla comparativa."""
        datos_num = self.obtener_datos_numericos(estandarizar=True)
        return Clustering.pca_variaciones(datos_num)

    def plot_acp_completo(self, n_componentes=5, ejes=[0, 1]):
        """
        Ejecuta ACP con prince y genera los tres gráficos clásicos:
        plano principal, círculo de correlación y sobreposición.
        """
        datos_num = self.obtener_datos_numericos(estandarizar=True)
        acp = ACP(datos_num, n_componentes=n_componentes)

        plt.figure(figsize=(8, 6))
        acp.plot_plano_principal(ejes=ejes, ind_labels=False)
        plt.show()

        plt.figure(figsize=(8, 6))
        acp.plot_circulo(ejes=ejes)
        plt.show()

        plt.figure(figsize=(8, 6))
        acp.plot_sobreposicion(ejes=ejes, ind_labels=False)
        plt.show()

        return acp

    # --- Clustering: K-Means ---
    def clustering_kmeans(self, n_clusters=3, init="k-means++", n_init=10,
                          max_iter=300, random_state=42, visualizar=True,
                          estandarizar=True):
        """
        Aplica K-Means. Retorna etiquetas, métricas y modelo.
        """
        datos = self.obtener_datos_numericos(estandarizar=estandarizar)
        etiquetas, metricas, modelo = Clustering.kmeans(
            datos, n_clusters=n_clusters, init=init, n_init=n_init,
            max_iter=max_iter, random_state=random_state, visualizar=visualizar
        )
        return etiquetas, metricas, modelo

    def kmeans_elbow(self, k_range=range(2, 11), estandarizar=True):
        """Método del codo + silhouette para elegir k."""
        datos = self.obtener_datos_numericos(estandarizar=estandarizar)
        return Clustering.kmeans_elbow(datos, k_range=k_range)

    # --- Clustering: HAC (estilo profesor: todo en un solo flujo) ---
    def clustering_hac(self, n_clusters=3, estandarizar=True):
        """
        HAC completo al estilo del profesor (PaqNOSup.HAC):
        dendrogramas de 4 linkages, clustering ward, Silhouette k=2..10.
        Retorna: etiquetas, métricas ward, tabla benchmark.
        """
        datos = self.obtener_datos_numericos(estandarizar=estandarizar)
        return Clustering.hac_completo(datos, n_clusters=n_clusters)

    # --- t-SNE ---
    def aplicar_tsne(self, n_components=2, perplexity=30, learning_rate="auto",
                     n_iter=1000, metric="euclidean", init="pca", random_state=42,
                     visualizar=True, color_by=None, estandarizar=True):
        """Ejecuta t-SNE y retorna coordenadas + modelo."""
        datos = self.obtener_datos_numericos(estandarizar=estandarizar)
        return Clustering.tsne(
            datos, n_components=n_components, perplexity=perplexity,
            learning_rate=learning_rate, n_iter=n_iter, metric=metric,
            init=init, random_state=random_state, visualizar=visualizar,
            color_by=color_by
        )

    def tsne_variaciones(self, color_by=None, estandarizar=True):
        """Prueba varias configuraciones de t-SNE."""
        datos = self.obtener_datos_numericos(estandarizar=estandarizar)
        return Clustering.tsne_variaciones(datos, color_by=color_by)

    # --- UMAP ---
    def aplicar_umap(self, n_components=2, n_neighbors=15, min_dist=0.1,
                     metric="euclidean", random_state=42, visualizar=True,
                     color_by=None, estandarizar=True):
        """Ejecuta UMAP y retorna coordenadas + modelo."""
        datos = self.obtener_datos_numericos(estandarizar=estandarizar)
        return Clustering.umap_reduce(
            datos, n_components=n_components, n_neighbors=n_neighbors,
            min_dist=min_dist, metric=metric, random_state=random_state,
            visualizar=visualizar, color_by=color_by
        )

    def umap_variaciones(self, color_by=None, estandarizar=True):
        """Prueba varias configuraciones de UMAP."""
        datos = self.obtener_datos_numericos(estandarizar=estandarizar)
        return Clustering.umap_variaciones(datos, color_by=color_by)

    # --- Métodos adicionales ---
    def ordenar(self, columna):
        return self.__df.sort_values(by=columna)

    def filtrar(self, columna, operador, valor):
        if operador == "==":
            return self.__df[self.__df[columna] == valor]
        elif operador == "!=":
            return self.__df[self.__df[columna] != valor]
        elif operador == ">":
            return self.__df[self.__df[columna] > valor]
        elif operador == ">=":
            return self.__df[self.__df[columna] >= valor]
        elif operador == "<":
            return self.__df[self.__df[columna] < valor]
        elif operador == "<=":
            return self.__df[self.__df[columna] <= valor]

    def agregar_columna(self, nombre, valor):
        self.__df[nombre] = valor
        return self.__df

    def eliminar_columna(self, nombre):
        self.__df = self.__df.drop(columns=[nombre])
        return self.__df

    def visualizar_datos(self):
        print(self.__df.head(10))
        print("\nDimensiones:", self.__df.shape)

    def frecuencias(self):
        freqs = {}
        for col in self.__df.columns:
            freqs[col] = self.__df[col].value_counts().to_frame(name="frecuencia")
        return freqs

    def convertir_tipos(self, conversiones):
        return self.__df.astype(conversiones)

    def distribucion_variables(self):
        return {
            "numericas": self.__df.select_dtypes(include=["float64", "int64"]).describe(),
            "categoricas": self.__df.select_dtypes(include=["object"]).describe()
        }

    def detectar_outliers(self):
        outliers = {}
        for col in self.__df.select_dtypes(include=["float64", "int64"]).columns:
            q1 = self.__df[col].quantile(0.25)
            q3 = self.__df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (self.__df[col] < (q1 - 1.5 * iqr)) | (self.__df[col] > (q3 + 1.5 * iqr))
            outliers[col] = self.__df[mask]
        return outliers
