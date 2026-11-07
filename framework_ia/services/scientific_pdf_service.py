"""Generate a portable, evidenced scientific interpretation PDF, without any API call."""
import io
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def _evidence_figures(evidence, doc_width):
    """Render a few factual figures directly from supplied aggregate statistics."""
    figures=[]
    charts=evidence.get('charts',{})
    candidates=[]
    if isinstance(charts.get('correlation_heatmap'),list):
        pairs=charts['correlation_heatmap'][:12]
        candidates.append(('Correlaciones de Pearson', [p.get('x','')+' / '+p.get('y','') for p in pairs], [p.get('pearson_r',0) for p in pairs]))
    for key, matrix in charts.items():
        if 'confusion_matrix' in key and isinstance(matrix,list) and matrix and isinstance(matrix[0],list):
            fig,ax=plt.subplots(figsize=(5.4,4.0))
            ax.imshow(matrix,cmap='Blues')
            for i,row in enumerate(matrix):
                for j,val in enumerate(row):
                    ax.text(j,i,str(val),ha='center',va='center',color='#222222')
            ax.set_xlabel('Predicho');ax.set_ylabel('Real');ax.set_title(key.replace('_',' ').title())
            fig.tight_layout();b=io.BytesIO();fig.savefig(b,format='png',dpi=135);plt.close(fig);b.seek(0)
            figures.append(Image(b,width=doc_width*.62,height=doc_width*.48))
    for key in ('cluster_scatter',):
        obj=charts.get(key,{})
        if isinstance(obj,dict) and obj.get('cluster_sizes'):
            candidates.append(('Tamano de los grupos',list(obj['cluster_sizes']),list(obj['cluster_sizes'].values())))
    for key in ('variance_by_component',):
        tbl=evidence.get('tables',{}).get(key,[])
        if isinstance(tbl,list) and tbl:
            candidates.append(('Varianza explicada (%)',[str(p['component']) for p in tbl[:20]],[p['variance_pct'] for p in tbl[:20]]))
    model_rows=evidence.get('models',[])
    if isinstance(model_rows,list) and model_rows:
        metric=next((m for m in ['cv_rmse_mean','RMSE','f1_macro_cv'] if any(isinstance(r.get(m),(float,int)) for r in model_rows if isinstance(r,dict))),None)
        if metric:
            items=[r for r in model_rows if isinstance(r.get(metric),(float,int))]
            candidates.append(('Comparacion de modelos: '+metric,[str(x.get('model','?')) for x in items],[x[metric] for x in items]))
    for title,names,values in candidates[:3]:
        if not names: continue
        fig,ax=plt.subplots(figsize=(7.8,3.2))
        ax.barh([str(n)[:25] for n in names],values,color='#287D82')
        ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.2)
        fig.tight_layout()
        b=io.BytesIO();fig.savefig(b,format='png',dpi=140,bbox_inches='tight');plt.close(fig)
        b.seek(0)
        im=Image(b,width=doc_width,height=doc_width*0.43)
        figures.append(im)
    return figures

def create_pdf(section, evidence, interpretation, mode):
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,leftMargin=1.9*cm,rightMargin=1.9*cm,topMargin=2*cm,bottomMargin=1.7*cm)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SectionReport',parent=styles['Heading2'],spaceBefore=14,spaceAfter=8,textColor=colors.HexColor('#183F56')))
    styles.add(ParagraphStyle(name='SmallReport',parent=styles['BodyText'],fontSize=8.5,leading=12,spaceAfter=5))
    elements=[]
    elements.append(Paragraph('Informe técnico-científico', styles['Title']))
    elements.append(Paragraph(escape(section),styles['Heading2']))
    elements.append(Paragraph('Datos: '+escape(str(evidence.get('dataset','Archivo CSV'))),styles['Normal']))
    elements.append(Paragraph('Interpretación: '+('OpenAI' if mode=='openai' else 'reglas locales (sin IA generativa)'),styles['Normal']))
    elements.append(Spacer(1,14))
    elements.append(Paragraph('Resultados e interpretación',styles['SectionReport']))
    for part in str(interpretation).split('\n'):
        part=part.strip()
        if not part: elements.append(Spacer(1,5));continue
        if part.startswith('#'):elements.append(Paragraph(escape(part.lstrip('# ')),styles['Heading3']))
        else:elements.append(Paragraph(escape(part).replace('**',''),styles['SmallReport']))
    elements.append(Paragraph('Visualizaciones de las evidencias',styles['SectionReport']))
    for fig in _evidence_figures(evidence,doc.width):
        elements.append(fig)
        elements.append(Spacer(1,10))
    elements.append(Paragraph('Evidencias de las tablas',styles['SectionReport']))
    for name, data in evidence.get('tables',{}).items():
        elements.append(Paragraph(escape(name.replace('_',' ').title()),styles['Heading3']))
        if isinstance(data,list) and data and isinstance(data[0],dict):
            keys=list(data[0].keys())[:6]
            values=[[Paragraph(escape(str(k))[:32],styles['SmallReport']) for k in keys]]
            for item in data[:14]:
                values.append([Paragraph(escape(str(item.get(k,'')))[:65],styles['SmallReport']) for k in keys])
            t=Table(values,repeatRows=1,hAlign='LEFT',colWidths=[doc.width/len(keys)]*len(keys))
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E3EFF3')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#D2DDE0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
            elements.append(t)
        else:
            elements.append(Paragraph(escape(str(data)[:1200]),styles['SmallReport']))
        elements.append(Spacer(1,9))
    elements.append(Paragraph('Evidencias utilizadas para interpretar los gráficos',styles['SectionReport']))
    for name, data in evidence.get('charts',{}).items():
        elements.append(Paragraph(escape(name.replace('_',' ').title()),styles['Heading3']))
        elements.append(Paragraph(escape(str(data)[:1300]),styles['SmallReport']))
    elements.append(Spacer(1,10))
    elements.append(Paragraph('Alcance: la interpretación usa resúmenes cuantitativos reales del análisis, no imágenes enviadas a la API. Las correlaciones no implican causalidad.',styles['SmallReport']))
    def pages(canvas, document):
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.9*cm,1*cm,'Advance 3 | Informe de análisis')
        canvas.drawRightString(A4[0]-1.9*cm,1*cm,str(document.page))
    doc.build(elements,onFirstPage=pages,onLaterPages=pages)
    return buf.getvalue()
