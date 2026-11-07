"""Coverage for one independent interpreter per analytical screen and PDF evidence."""
import pandas as pd
from framework_ia.agents.section_agents import SECTION_AGENTS
from framework_ia.services.section_evidence_service import build_evidence
from framework_ia.services.scientific_pdf_service import create_pdf


def test_nine_separate_agents():
    assert len(SECTION_AGENTS)==9
    assert len(set(SECTION_AGENTS.values()))==9
    assert 'Guía de utilización' not in SECTION_AGENTS
    for key, cls in SECTION_AGENTS.items():
        assert callable(cls.interpret),key


def test_section_evidence_and_pdf():
    data=pd.DataFrame({'age':[22,30,44,55,66], 'fare':[12,14,21,30,44], 'sex':['F','M','F','F','M']})
    for section in ('Datos','Estadísticas básicas'):
        ev=build_evidence(section,data,{'filename':'example.csv'})
        assert ev['tables'] and ev['charts']
        pdf=create_pdf(section,ev,'Análisis descriptivo verificable.','openai')
        assert pdf.startswith(b'%PDF') and len(pdf)>1500


def test_no_results_for_clustering():
    d=pd.DataFrame({'age':[10,20]})
    assert build_evidence('HAC',d,{}) is None
    assert build_evidence('UMAP',d,{}) is None


def test_models_evidence_is_not_raw_csv():
    d=pd.DataFrame({'age':[10,20,30], 'fare':[25,35,45]})
    r=pd.DataFrame([{'model':'Ridge','RMSE':3.5, 'MAE':2.1,'R2':0.6}])
    e=build_evidence('Regresión supervisada',d,{'regression_results':r,'regression_target':'fare'})
    assert len(e['models'])==1
    assert 'records' not in e
    assert 'regression_model_benchmark' in e['tables']
