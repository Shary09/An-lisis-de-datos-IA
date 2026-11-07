"""Nine independent analytical roles; no training, fitting or predictive code."""
from framework_ia.agents.scientific_agent import ScientificAgent, RegressionScientificAgent, ClassificationScientificAgent

class DataQualityAgent(ScientificAgent):
    name = "data_quality_agent"
    def _fallback(self, evidence, selection):
        return ["Examinar tipos de columna, valores faltantes y duplicados antes de interpretar resultados."]

class DescriptiveStatisticsAgent(ScientificAgent):
    name = "descriptive_statistics_agent"
    def _fallback(self, evidence, selection):
        return ["Comparar distribución, dispersión, valores extremos y asociaciones observadas."]

class PCAAgent(ScientificAgent):
    name = "pca_agent"
    def _fallback(self, evidence, selection):
        return ["Evaluar la varianza explicada y las contribuciones de las variables, sin asumir causalidad."]

class KMeansAgent(ScientificAgent):
    name = "kmeans_agent"
    def _fallback(self, evidence, selection):
        return ["Analizar silhouette, inercia, tamaños de grupos y perfiles de centroides."]

class HACAgent(ScientificAgent):
    name = "hac_agent"
    def _fallback(self, evidence, selection):
        return ["Evaluar los tamaños de grupos, su separación y el método de enlace utilizado."]

class TSNEAgent(ScientificAgent):
    name = "tsne_agent"
    def _fallback(self, evidence, selection):
        return ["Revisar perplexity y divergencia KL; las distancias globales no son una medida directa de similitud."]

class UMAPAgent(ScientificAgent):
    name = "umap_agent"
    def _fallback(self, evidence, selection):
        return ["Interpretar la proyección según n_neighbors/min_dist; evitar inferir agrupamientos validados solo visualmente."]

class ClassificationSectionAgent(ClassificationScientificAgent):
    name = "classification_section_agent"

class RegressionSectionAgent(RegressionScientificAgent):
    name = "regression_section_agent"

SECTION_AGENTS = {
    "Datos": DataQualityAgent,
    "Estadísticas básicas": DescriptiveStatisticsAgent,
    "ACP": PCAAgent,
    "K-Means": KMeansAgent,
    "HAC": HACAgent,
    "t-SNE": TSNEAgent,
    "UMAP": UMAPAgent,
    "Clasificación supervisada": ClassificationSectionAgent,
    "Regresión supervisada": RegressionSectionAgent,
}
