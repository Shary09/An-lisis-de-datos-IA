import pandas as pd
from DataFrame import DataFrame
from supervisados import Supervisados


class DataFrameModel:
    """
    Modelo que encapsula la lógica de negocio sobre el DataFrame.
    Se conecta con la clase DataFrame y expone métodos para el Controller.
    """

    def __init__(self, ruta_csv=None, dataframe=None):
        if dataframe is not None:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("dataframe debe ser un pandas.DataFrame")
            datos = dataframe.copy()
        elif ruta_csv is not None:
            datos = pd.read_csv(ruta_csv)
        else:
            raise ValueError("Indique ruta_csv o dataframe")
        self.df = DataFrame(datos)

    # --- Métodos básicos ---
    def mostrar(self):
        return self.df.mostrar()

    def primeras_filas(self, n=10):
        return self.df.primeras_filas(n)

    def ultimas_filas(self, n=10):
        return self.df.ultimas_filas(n)

    def obtener_fila(self, indice):
        return self.df.obtener_fila(indice)

    def obtener_columna(self, nombre):
        return self.df.obtener_columna(nombre)

    def dimensiones(self):
        return self.df.dimensiones()

    def nombres_columnas(self):
        return self.df.nombres_columnas()

    def tipos_datos(self):
        return self.df.tipos_datos()

    # --- Limpieza ---
    def contar_nulos(self):
        return self.df.contar_nulos()

    def eliminar_nulos(self):
        return self.df.eliminar_nulos()

    def reemplazar_nulos(self, valor=0):
        return self.df.reemplazar_nulos(valor)

    def detectar_duplicados(self):
        return self.df.detectar_duplicados()

    def eliminar_duplicados(self):
        return self.df.eliminar_duplicados()

    # --- Estadísticas y correlación ---
    def estadisticas(self):
        return self.df.estadisticas()

    def correlacion(self):
        return self.df.correlacion()

    # --- Visualizaciones ---
    def histogramas(self):
        self.df.histogramas()

    def boxplots(self):
        self.df.boxplots()

    def scatterplots(self):
        self.df.scatterplots()

    def mapa_calor(self):
        self.df.mapa_calor()

    # --- Transformaciones ---
    def normalizar(self):
        return self.df.normalizar()

    def estandarizar(self):
        return self.df.estandarizar()

    def imputar_nulos(self):
        return self.df.imputar_nulos()

    # --- Selección y exportación ---
    def seleccionar_variables(self, columnas):
        return self.df.seleccionar_variables(columnas)

    def guardar_csv(self, ruta="dataframe_guardado.csv"):
        return self.df.guardar_csv(ruta)

    def exportar_resultados(self, ruta="resultados_eda.csv"):
        return self.df.exportar_resultados(ruta)

    # --- ACP ---
    def aplicar_acp(self, n_componentes=2, usar_prince=True, estandarizar=True):
        return self.df.aplicar_acp(
            n_componentes=n_componentes,
            usar_prince=usar_prince,
            estandarizar=estandarizar,
        )

    def acp_variaciones(self):
        return self.df.acp_variaciones()

    def plot_acp_completo(self, n_componentes=5, ejes=[0, 1]):
        return self.df.plot_acp_completo(
            n_componentes=n_componentes,
            ejes=ejes,
        )

    # --- K-Means ---
    def clustering_kmeans(
        self,
        n_clusters=3,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
        visualizar=True,
        estandarizar=True,
    ):
        return self.df.clustering_kmeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            visualizar=visualizar,
            estandarizar=estandarizar,
        )

    def kmeans_elbow(self, k_range=range(2, 11), estandarizar=True):
        return self.df.kmeans_elbow(
            k_range=k_range,
            estandarizar=estandarizar,
        )

    # --- HAC ---
    def clustering_hac(self, n_clusters=3, estandarizar=True):
        return self.df.clustering_hac(
            n_clusters=n_clusters,
            estandarizar=estandarizar,
        )

    # --- t-SNE ---
    def aplicar_tsne(
        self,
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        n_iter=1000,
        metric="euclidean",
        init="pca",
        random_state=42,
        visualizar=True,
        color_by=None,
        estandarizar=True,
    ):
        return self.df.aplicar_tsne(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            metric=metric,
            init=init,
            random_state=random_state,
            visualizar=visualizar,
            color_by=color_by,
            estandarizar=estandarizar,
        )

    def tsne_variaciones(self, color_by=None, estandarizar=True):
        return self.df.tsne_variaciones(
            color_by=color_by,
            estandarizar=estandarizar,
        )

    # --- UMAP ---
    def aplicar_umap(
        self,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        visualizar=True,
        color_by=None,
        estandarizar=True,
    ):
        return self.df.aplicar_umap(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            visualizar=visualizar,
            color_by=color_by,
            estandarizar=estandarizar,
        )

    def umap_variaciones(self, color_by=None, estandarizar=True):
        return self.df.umap_variaciones(
            color_by=color_by,
            estandarizar=estandarizar,
        )

    # --- Métodos adicionales para Controller ---
    def ordenar(self, columna):
        return self.df.ordenar(columna)

    def filtrar(self, columna, operador, valor):
        return self.df.filtrar(columna, operador, valor)

    def agregar_columna(self, nombre, valor):
        return self.df.agregar_columna(nombre, valor)

    def eliminar_columna(self, nombre):
        return self.df.eliminar_columna(nombre)

    def visualizar_datos(self):
        return self.df.visualizar_datos()

    def frecuencias(self):
        return self.df.frecuencias()

    def convertir_tipos(self, conversiones):
        return self.df.convertir_tipos(conversiones)

    def distribucion_variables(self):
        return self.df.distribucion_variables()

    def detectar_outliers(self):
        return self.df.detectar_outliers()

    # --- Clasificación supervisada (Laboratorio 2) ---
    def clasificacion_supervisada(self, objetivo, algoritmo="KNN", semilla=42):
        datos = self.df.mostrar().copy()
        laboratorio = Supervisados(datos, objetivo=objetivo, semilla=semilla)
        return laboratorio.comparar_modelo(algoritmo=algoritmo)


    def comparar_clasificadores(self, objetivo, semilla=42):
        """Ejecuta los cinco algoritmos con la misma partición y devuelve sus detalles."""
        datos = self.df.mostrar().copy()
        laboratorio = Supervisados(datos, objetivo=objetivo, semilla=semilla)
        algoritmos = ["KNN", "Arbol de Decision", "Random Forest", "XGBoost", "AdaBoost"]
        resultados = {}
        for algoritmo in algoritmos:
            resultados[algoritmo] = laboratorio.comparar_modelo(algoritmo=algoritmo)
        return resultados
