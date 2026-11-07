"""Controller for dataset loading and preparation actions."""
from framework_ia.models.data.preprocessing import prepare_dataframe
from framework_ia.models.eda.eda_analysis import analyze_dataframe

class DataController:
    def prepare(self, dataframe, **options):
        return prepare_dataframe(dataframe, **options)
    def profile(self, dataframe):
        return analyze_dataframe(dataframe)
