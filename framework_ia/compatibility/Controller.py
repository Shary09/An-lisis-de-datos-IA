import pandas as pd


class DataFrameController:
    """
    Controller que recibe el Model y el View.
    Gestiona eventos de la interfaz y decide qué método del Model ejecutar.
    """

    def __init__(self, model, view):
        self.model = model
        self.view = view

    def iniciar(self):
        # Flujo inicial
        self.view.mostrar_mensaje("Iniciando controlador...")
        self.view.mostrar_mensaje("--- Primeras filas del dataset ---")
        self.view.mostrar(self.model.primeras_filas(10))
        self.view.mostrar_mensaje(
            f"Dimensiones del dataset: {self.model.dimensiones()}"
        )

    def ejecutar(self, metodo, **kwargs):
        try:
            if metodo == "Comparación global supervisada":
                return self.model.comparar_clasificadores(
                    objetivo=kwargs.get("objetivo"),
                    semilla=kwargs.get("semilla", 42),
                )

            if metodo == "Clasificación supervisada":
                # Devuelve el diccionario para que Streamlit muestre sus métricas.
                return self.model.clasificacion_supervisada(
                    objetivo=kwargs.get("objetivo"),
                    algoritmo=kwargs.get("algoritmo", "KNN"),
                    semilla=kwargs.get("semilla", 42),
                )

            elif metodo == "Información":
                self.view.mostrar_mensaje(self.model.dimensiones())

            elif metodo == "Mostrar":
                self.view.mostrar(self.model.mostrar().head(20))

            elif metodo == "Primeras filas":
                self.view.mostrar(self.model.primeras_filas(10))

            elif metodo == "Últimas filas":
                self.view.mostrar(self.model.ultimas_filas(10))

            elif metodo == "Obtener fila":
                indice = kwargs.get("indice", 0)
                self.view.mostrar(self.model.obtener_fila(indice))

            elif metodo == "Obtener columna":
                columna = kwargs.get("columna")
                self.view.mostrar(self.model.obtener_columna(columna).head(20))

            elif metodo == "Estadísticas":
                self.view.mostrar_estadisticas(self.model.estadisticas())

            elif metodo == "Correlación":
                self.view.mostrar_correlacion(self.model.correlacion())

            elif metodo == "Frecuencias":
                self.view.mostrar_frecuencias(self.model.frecuencias())

            elif metodo == "Detectar duplicados":
                self.view.mostrar_duplicados(self.model.detectar_duplicados())

            elif metodo == "Eliminar duplicados":
                self.view.mostrar_eliminados(self.model.eliminar_duplicados())

            elif metodo == "Ordenar":
                columna = kwargs.get("columna")
                self.view.mostrar(self.model.ordenar(columna))

            elif metodo == "Filtrar":
                columna = kwargs.get("columna")
                operador = kwargs.get("operador")
                valor = kwargs.get("valor")
                self.view.mostrar(self.model.filtrar(columna, operador, valor))

            elif metodo == "Agregar columna":
                nombre = kwargs.get("nombre", "nueva_columna")
                valor = kwargs.get("valor", 1)
                self.view.mostrar(self.model.agregar_columna(nombre, valor))

            elif metodo == "Eliminar columna":
                columna = kwargs.get("columna")
                self.view.mostrar(self.model.eliminar_columna(columna))

            elif metodo == "Detectar outliers":
                self.view.mostrar_outliers(self.model.detectar_outliers())

            elif metodo == "Distribución variables":
                distribucion = self.model.distribucion_variables()
                self.view.mostrar_distribucion(distribucion)

            # --- Visualizaciones ---
            elif metodo == "Histogramas":
                self.view.mostrar_mensaje("Histograma de variables numéricas")
                self.model.histogramas()

            elif metodo == "Boxplots":
                self.view.mostrar_mensaje("Boxplots de variables numéricas")
                self.model.boxplots()

            elif metodo == "Scatterplots":
                self.view.mostrar_mensaje("Scatterplots (matriz de dispersión)")
                self.model.scatterplots()

            elif metodo == "Mapa calor":
                self.view.mostrar_mensaje("Mapa de calor de correlaciones")
                self.model.mapa_calor()

            # --- Transformaciones ---
            elif metodo == "Normalizar":
                self.view.mostrar_mensaje("Normalización de variables")
                self.view.mostrar(self.model.normalizar())

            elif metodo == "Estandarizar":
                self.view.mostrar_mensaje("Estandarización de variables")
                self.view.mostrar(self.model.estandarizar())

            elif metodo == "Imputar nulos":
                self.view.mostrar_mensaje("Imputación de valores nulos")
                self.view.mostrar(self.model.imputar_nulos())

            # --- ACP ---
            elif metodo == "ACP":
                resultado = self.model.aplicar_acp(
                    n_componentes=kwargs.get("n_componentes", 2),
                    usar_prince=kwargs.get("usar_prince", False),
                    estandarizar=kwargs.get("estandarizar", True),
                )

                # Si se usó sklearn retorna (coords, modelo, df_var)
                if isinstance(resultado, tuple):
                    self.view.mostrar_var_explicada(resultado[2])
                else:
                    # Objeto ACP (prince)
                    self.view.mostrar_mensaje(
                        "ACP (prince) ejecutado correctamente."
                    )
                    self.view.mostrar_var_explicada(
                        pd.DataFrame(
                            {
                                "Componente": [
                                    f"PC{i+1}"
                                    for i in range(len(resultado.var_explicada))
                                ],
                                "Varianza explicada (%)": resultado.var_explicada.round(2),
                            }
                        )
                    )

            elif metodo == "ACP variaciones":
                tabla = self.model.acp_variaciones()
                self.view.mostrar(tabla)

            elif metodo == "ACP completo":
                self.model.plot_acp_completo(
                    n_componentes=kwargs.get("n_componentes", 5),
                    ejes=kwargs.get("ejes", [0, 1]),
                )
                self.view.mostrar_mensaje("Gráficos de ACP generados.")

            # --- K-Means ---
            elif metodo == "Clustering" or metodo == "KMeans":
                etiquetas, metricas, modelo = self.model.clustering_kmeans(
                    n_clusters=kwargs.get("n_clusters", 3),
                    init=kwargs.get("init", "k-means++"),
                    n_init=kwargs.get("n_init", 10),
                    max_iter=kwargs.get("max_iter", 300),
                    random_state=kwargs.get("random_state", 42),
                    visualizar=kwargs.get("visualizar", True),
                    estandarizar=kwargs.get("estandarizar", True),
                )
                self.view.mostrar_clusters(etiquetas, metricas)

            # Se conserva disponible por si se desea usar más adelante.
            # elif metodo == "KMeans elbow":
            #     df_elbow = self.model.kmeans_elbow(
            #         k_range=kwargs.get("k_range", range(2, 11)),
            #         estandarizar=kwargs.get("estandarizar", True),
            #     )
            #     self.view.mostrar(df_elbow)

            # --- HAC ---
            elif metodo == "HAC":
                etiquetas, metricas, benchmark = self.model.clustering_hac(
                    n_clusters=kwargs.get("n_clusters", 3),
                    estandarizar=kwargs.get("estandarizar", True),
                )
                self.view.mostrar_clusters(etiquetas, metricas)
                self.view.mostrar(benchmark)

            # --- t-SNE ---
            elif metodo == "tSNE":
                coordenadas, modelo = self.model.aplicar_tsne(
                    n_components=kwargs.get("n_components", 2),
                    perplexity=kwargs.get("perplexity", 30),
                    learning_rate=kwargs.get("learning_rate", "auto"),
                    n_iter=kwargs.get("n_iter", 1000),
                    metric=kwargs.get("metric", "euclidean"),
                    init=kwargs.get("init", "pca"),
                    random_state=kwargs.get("random_state", 42),
                    visualizar=kwargs.get("visualizar", True),
                    color_by=kwargs.get("color_by"),
                    estandarizar=kwargs.get("estandarizar", True),
                )

                self.view.mostrar(coordenadas.head())

                if hasattr(modelo, "kl_divergence_"):
                    self.view.mostrar_mensaje(
                        f"KL divergence: {modelo.kl_divergence_:.4f}"
                    )

            elif metodo == "tSNE variaciones":
                tabla = self.model.tsne_variaciones(
                    color_by=kwargs.get("color_by"),
                    estandarizar=kwargs.get("estandarizar", True),
                )
                self.view.mostrar(tabla)

            # --- UMAP ---
            elif metodo == "UMAP":
                coordenadas, modelo = self.model.aplicar_umap(
                    n_components=kwargs.get("n_components", 2),
                    n_neighbors=kwargs.get("n_neighbors", 15),
                    min_dist=kwargs.get("min_dist", 0.1),
                    metric=kwargs.get("metric", "euclidean"),
                    random_state=kwargs.get("random_state", 42),
                    visualizar=kwargs.get("visualizar", True),
                    color_by=kwargs.get("color_by"),
                    estandarizar=kwargs.get("estandarizar", True),
                )

                self.view.mostrar(coordenadas.head())
                self.view.mostrar_mensaje("UMAP ejecutado correctamente.")

            elif metodo == "UMAP variaciones":
                tabla = self.model.umap_variaciones(
                    color_by=kwargs.get("color_by"),
                    estandarizar=kwargs.get("estandarizar", True),
                )
                self.view.mostrar(tabla)

            # --- Exportación ---
            elif metodo == "Guardar CSV":
                ruta = kwargs.get("ruta", "resultado_final.csv")
                self.view.mostrar_mensaje(
                    f"Archivo guardado en: {self.model.guardar_csv(ruta)}"
                )

            elif metodo == "Exportar resultados":
                ruta = kwargs.get("ruta", "estadisticas_finales.csv")
                self.view.mostrar_mensaje(
                    f"Resultados exportados en: {self.model.exportar_resultados(ruta)}"
                )

            else:
                self.view.mostrar_mensaje(f"Método '{metodo}' no reconocido.")

        except Exception as e:
            if metodo in ("Clasificación supervisada", "Comparación global supervisada"):
                # Streamlit captura y presenta el error en su propia interfaz.
                raise
            self.view.mostrar_mensaje(f"ERROR: {type(e).__name__}: {e}")
