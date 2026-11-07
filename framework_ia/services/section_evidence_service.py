"""Bounded, read-only summaries of evidence shown on each dashboard page.
No training, fitting or prediction occurs here. Chart evidence describes the real data
used to render the plots, never asks an LLM to infer pixels or manufacture findings.
"""
import math
import pandas as pd
import numpy as np


def _json(value):
    if isinstance(value, dict): return {str(k): _json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_json(x) for x in value]
    if isinstance(value, (np.ndarray, pd.Index)): return _json(value.tolist())
    if isinstance(value, np.generic): return _json(value.item())
    if isinstance(value, float): return round(value, 6) if math.isfinite(value) else None
    if pd.isna(value): return None
    if isinstance(value, (str, int, bool)) or value is None: return value
    return str(value)


def _table(df, max_rows=30, max_cols=14):
    if not isinstance(df, pd.DataFrame): return []
    return _json(df.iloc[:max_rows, :max_cols].reset_index().to_dict('records'))


def _counts(values):
    if values is None: return {}
    return {str(k):int(v) for k,v in pd.Series(values).value_counts().sort_index().items()}


def build_evidence(section, df, state):
    if not isinstance(df, pd.DataFrame) or df.empty: return None
    evidence = {"section": section, "dataset": state.get('filename','CSV cargado'),
                "shape": {"rows":len(df), "columns":len(df.columns)},
                "column_types": {str(k):str(v) for k,v in df.dtypes.items()},
                "tables":{}, "charts":{}, "limitations":[]}
    missing = {str(k):int(v) for k,v in df.isna().sum().items() if int(v)>0}
    evidence['data_quality'] = {'missing_by_column':missing, 'duplicates':int(df.duplicated().sum())}
    num = df.select_dtypes(include='number').replace([np.inf,-np.inf],np.nan)
    if section == 'Datos':
        evidence['tables']['data_types_missing_unique'] = _table(pd.DataFrame({'variable':df.columns,'type':df.dtypes.astype(str).values,'missing':df.isna().sum().values,'unique':df.nunique(dropna=True).values}))
        evidence['charts']['missing_values'] = {'missing_by_column':missing, 'total_missing':int(df.isna().sum().sum())}
    elif section == 'Estadísticas básicas':
        if not num.empty:
            desc = num.describe().T.reset_index().rename(columns={'index':'variable'})
            evidence['tables']['descriptive_statistics'] = _table(desc)
            evidence['charts']['numeric_distributions_boxplots'] = _table(desc[['variable','min','25%','50%','75%','max','mean','std']])
            if num.shape[1]>=2:
                correlations=num.corr(numeric_only=True)
                evidence['tables']['pearson_correlation_matrix'] = _table(correlations.reset_index())
                pairs=[]
                for i,a in enumerate(correlations.columns):
                    for b in correlations.columns[i+1:]:
                        v=correlations.loc[a,b]
                        if pd.notna(v): pairs.append({'x':str(a),'y':str(b),'pearson_r':float(v)})
                evidence['charts']['correlation_heatmap'] = sorted(pairs,key=lambda x:abs(x['pearson_r']),reverse=True)[:25]
        cat=df.select_dtypes(exclude='number')
        evidence['charts']['categorical_frequencies'] = {str(c):_counts(cat[c].dropna()) for c in cat.columns[:8]}
    elif section == 'ACP':
        if state.get('acp_var') is None: return None
        values=np.asarray(state['acp_var'],dtype=float).flatten()
        if len(values) and np.nanmax(values)<=1.00001: values=values*100
        evidence['tables']['variance_by_component'] = [{'component':i+1,'variance_pct':round(float(x),4)} for i,x in enumerate(values[:30])]
        evidence['charts']['scree_plot'] = {'variance_percent':_json(values[:30]),'cumulative_variance_percent':_json(np.cumsum(values[:30]))}
        corr=state.get('acp_corr')
        if isinstance(corr,pd.DataFrame): evidence['tables']['variable_correlations_with_components']=_table(corr)
        coords=state.get('acp_coords')
        if coords is not None:
            cc=pd.DataFrame(coords)
            evidence['charts']['individuals_projection']={'dimensions':list(map(str,cc.columns[:3])),'coordinate_ranges':_table(cc.iloc[:,:3].describe().T.reset_index())}
    elif section in ('K-Means','HAC'):
        key='km' if section=='K-Means' else 'hac'
        labels=state.get(key+'_labels')
        if labels is None: return None
        evidence['charts']['cluster_scatter']={'cluster_sizes':_counts(labels),'number_of_clusters':len(set(labels))}
        evidence['tables']['clustering_metrics']=_json(state.get(key+'_metricas') or {})
        centers=state.get(key+'_centers')
        if centers is not None:
            evidence['tables']['cluster_centers']=_table(pd.DataFrame(np.asarray(centers),columns=state.get(key+'_cols') or None))
        if section=='K-Means':
            evidence['charts']['elbow_and_silhouette']={'metrics':_json(state.get(key+'_metricas') or {}),'k':state.get('km_k')}
        else:
            evidence['charts']['hierarchical_dendrogram']={'linkage':state.get('hac_linkage'),'number_of_clusters':state.get('hac_k')}
        evidence['limitations'].append('Los grupos no poseen etiquetas de verdad terreno. Las proyecciones no demuestran causalidad.')
    elif section in ('t-SNE','UMAP'):
        key='tsne' if section=='t-SNE' else 'umap'
        coords=state.get(key+'_coords')
        if coords is None: return None
        cc=pd.DataFrame(coords).iloc[:,:2]
        evidence['charts']['embedding_2d_scatter']={'axis_statistics':_table(cc.describe().T.reset_index()),'rows':len(cc)}
        evidence['tables']['embedding_configuration'] = ({'perplexity':state.get('tsne_perp'),'kl_divergence':state.get('tsne_kl')} if section=='t-SNE' else {'n_neighbors':state.get('umap_nn'),'min_dist':state.get('umap_md')})
        evidence['limitations'].append('No puede inferirse validez de clústeres ni distancias globales del gráfico bidimensional por sí solo.')
    elif section == 'Clasificación supervisada':
        results=state.get('sup_global')
        one=state.get('sup_resultado')
        if results:
            models=[]
            for name,r in results.items():
                m=r.get('metricas') or {}
                models.append({'model':name,'f1_macro_cv':r.get('f1_macro_cv'), 'accuracy_test':m.get('accuracy'),
                               'precision_macro_test':m.get('precision_macro'),'recall_macro_test':m.get('recall_macro'),
                               'f1_macro_test':m.get('f1_macro')})
                if 'matriz_confusion' in m:
                    evidence['charts']['confusion_matrix_'+str(name)] = _json(np.asarray(m['matriz_confusion']).tolist())
                if isinstance(r.get('variaciones'),pd.DataFrame):
                    evidence['tables']['hyperparameter_variants_'+str(name)] = _table(r['variaciones'],15)
            evidence['models']=_json(models)
            evidence['tables']['model_comparison']=_json(models)
            evidence['charts']['classification_metric_comparison']=_json(models)
        elif one:
            m=one.get('metricas') or {}
            evidence['models']=[{'model':one.get('algoritmo'),'f1_macro_cv':one.get('f1_macro_cv'),
                                 'f1_macro_test':m.get('f1_macro'),'accuracy_test':m.get('accuracy')}]
            if 'matriz_confusion' in m: evidence['charts']['confusion_matrix']=_json(np.asarray(m['matriz_confusion']).tolist())
            if isinstance(one.get('variaciones'),pd.DataFrame): evidence['tables']['hyperparameter_variants']=_table(one['variaciones'])
        else: return None
        evidence['target']=state.get('sup_objetivo')
    elif section == 'Regresión supervisada':
        results=state.get('regression_results')
        if results is None or not isinstance(results,pd.DataFrame) or results.empty:return None
        evidence['target']=state.get('regression_target')
        evidence['models']=_table(results)
        evidence['tables']['regression_model_benchmark']=_table(results)
        evidence['charts']['rmse_mae_r2_model_comparison']=_table(results)
        evidence['limitations'].append('No inventar residuos ni gráficos reales-vs-predicho si no se proporcionan. Si solo hay métricas de prueba, no usar esas métricas para seleccionar sin CV.')
    return _json(evidence)
