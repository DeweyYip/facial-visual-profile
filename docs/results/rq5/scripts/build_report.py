"""Build the six-page evidence review from audited tables and figures."""
from pathlib import Path
import json
import pandas as pd
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib import font_manager
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parents[1]; W,H=A4; M=42; CW=W-2*M
pdfmetrics.registerFont(TTFont('DV',font_manager.findfont('DejaVu Sans')))
pdfmetrics.registerFont(TTFont('DVB',font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))))
styles={
 'title':ParagraphStyle('title',fontName='DVB',fontSize=21,leading=25,spaceAfter=9),
 'h1':ParagraphStyle('h1',fontName='DVB',fontSize=14,leading=18,spaceAfter=9),
 'h2':ParagraphStyle('h2',fontName='DVB',fontSize=10,leading=14,spaceBefore=10,spaceAfter=5),
 'body':ParagraphStyle('body',fontName='DV',fontSize=8.6,leading=12,spaceAfter=6),
 'small':ParagraphStyle('small',fontName='DV',fontSize=7.5,leading=10,spaceAfter=5),
 'table':ParagraphStyle('table',fontName='DV',fontSize=7.2,leading=10),
}
S=[]
def p(text,style='body'):return Paragraph(text,styles[style])
def add(text,style='body'):S.append(p(text,style))
def table(rows,widths,font=7.5):
 converted=[[p(escape(str(v)),'table') for v in row] for row in rows]
 t=Table(converted,colWidths=widths,hAlign='LEFT',repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EDF1F3')),('VALIGN',(0,0),(-1,-1),'TOP'),
  ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
  ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#555555')),
  ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#999999')),
  ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
 S.append(t)
def figure(name,maxheight=480):
 path=ROOT/'figures'/f'{name}.png'
 w,h=PILImage.open(path).size
 scale=min(CW/w,maxheight/h)
 S.append(Image(str(path),width=w*scale,height=h*scale))
def page(title):S.append(PageBreak());add(title,'h1')
summary=pd.read_csv(ROOT/'tables/scores_mean_sample_sd.csv',header=[0,1],index_col=[0,1])
paired=pd.read_csv(ROOT/'tables/paired_mean_sample_sd.csv',header=[0,1],index_col=[0,1])
training=pd.read_csv(ROOT/'tables/training_summary.csv')
audit=json.loads((ROOT/'audit_report.json').read_text())
cap=(ROOT/'figure_captions.md').read_text().split('## ')
CAP={chunk.split('\n',1)[0]:chunk.split('\n',1)[1].strip() for chunk in cap if chunk.strip()}
add('Facial Visual Profile','title')
add('RQ5 evidence review · 27 October 2026','h1')
add('Nonlinear predicted-geometry fusion and auxiliary landmark supervision. Nine new runs (seeds 0–2), compared with three archived RQ4-A runs. Official CelebA test: 19,962 images, 24 attributes, clean plus 20 corrupted conditions.')
add('Test performance','h2')
rows=[['Scope','Model','mAP (%)','Frozen macro-F1 (%)','F1@0.5 (%)']]
for (scope,model),row in summary.iterrows():
 vals=[f"{100*row[(m,'mean')]:.3f} ± {100*row[(m,'std')]:.3f}" for m in ['mAP','frozen_F1','F1_at_05']]
 rows.append([scope,model]+vals)
table(rows,[65,42,126,145,CW-378])
add('Mean ± sample SD over three seeds. Corrupted scores are equal means over the 20 non-clean conditions within each seed. F1 thresholds were selected on clean validation and frozen for testing.','small')
add('Primary contrasts on corrupted inputs','h2')
rows=[['Contrast','mAP difference (pp)','Frozen-F1 difference (pp)']]
for contrast in ['N1-N0','MTL-A']:
 r=paired.loc[(contrast,'corrupted')]
 rows.append([contrast]+[f"{r[(f'delta_{m}_pp','mean')]:+.4f} ± {r[(f'delta_{m}_pp','std')]:.4f}" for m in ['mAP','frozen_F1']])
table(rows,[75,205,CW-280])
add('N1 does not show a consistent aggregate gain beyond the matched nonlinear RGB head. MTL lowers corrupted mAP in all three seeds; its mean frozen-F1 gain is positive in two of three seeds. These metrics do not support a uniform improvement claim.')
add('Definitions and verification','h2')
add('A: archived linear RGB R18. N0: RGB with nonlinear head and zero geometry placeholder. N1: predicted five-point geometry with the matched nonlinear head. MTL: RGB-only attribute inference with auxiliary GT landmark supervision during training. Frozen λ = 1.89811.')
add('109 source files were downloaded. All 252 condition-level and 6,048 attribute-level records reconcile with raw per-run JSON; threshold/test hashes and recorded checkpoint references agree. Weight binaries and per-image predictions were not independently re-evaluated.','small')
add('Historical A_seed0 ran 6 epochs although the current stopping rule would stop at epoch 4; both select epoch 2. The archived RQ4 environment also differs in Python patch version. These differences limit strict protocol equivalence.','small')

page('1. Paired effects across seeds')
figure('fig01_paired_effects',450)
add(CAP['fig01_paired_effects'],'small')
add('Interpretation','h2')
add('The N1-N0 mAP effect changes sign and is close to zero on average. The MTL-A corrupted mAP effect is negative for every seed. Frozen-F1 moves differently from mAP; this is a metric-dependent empirical result, not evidence identifying calibration or another mechanism.')
add('Primary evidence: tables/paired_seed_contrasts.csv and paired_mean_sample_sd.csv. Pairing is by nominal seed; identical RNG trajectories are not guaranteed.','small')

page('2. Corruption severity and absolute performance')
figure('fig02_corruption_curves',300)
add(CAP['fig02_corruption_curves'],'small')
add('Frozen severity settings','h2')
table([['Family','S1','S2','S3','S4'],['Blur radius (px)',1,2,3,4],['Brightness factor',.8,.6,.4,.2],['Rotation magnitude (degrees)',5,10,15,20],['Occluder side / 224',.1,.2,.3,.4],['JPEG quality',75,50,25,10]],[190]+[(CW-190)/4]*4)
add('C is clean. Rotation sign, occluder location and JPEG settings follow the frozen implementation. Each condition starts from the clean canvas.','small')
add('Why plot both levels and differences?','h2')
add('The absolute curves show that corruption severity dominates the overall score changes. The paired-effect plots resolve the much smaller differences between models without confusing those differences with the large clean-to-corrupted decline.')
add('Primary evidence: raw/condition_results.csv; plotted means and SD are exported to tables/curve_plot_data.csv.','small')

page('3. Corruption-family paired effects')
figure('fig03_family_effects',425)
add(CAP['fig03_family_effects'],'small')
add('Use in the paper','h2')
add('Use this figure to describe heterogeneity across corruption families. Do not select a favorable family as the headline outcome. The primary corrupted average includes every family and severity, and all seeds remain included.')
add('Primary evidence: tables/family_paired_differences.csv and family_contrast_summary.csv.','small')

page('4. Attribute-level diagnostics')
figure('fig04_attribute_effects',510)
add(CAP['fig04_attribute_effects'],'small')
add('Each row is one attribute; columns retain all three seeds. Large color changes in F1 can coexist with small AP changes. These diagnostics are post-hoc and should be placed in supplementary analysis unless they directly support a pre-specified claim.','small')
add('Primary evidence: tables/attribute_paired_differences.csv and attribute_contrast_summary.csv.','small')

page('5. Training, early stopping and evidence limits')
figure('fig05_training_curves',225)
add(CAP['fig05_training_curves'],'small')
rows=[['Run','Epochs run','Best epoch','Best val. mAP (%)']]
for _,r in training[training.model!='A'].iterrows():
 rows.append([f"{r.model}_seed{int(r.seed)}",int(r.epochs),int(r.best_epoch),f'{100*r.best_validation_mAP:.4f}'])
table(rows,[140,100,100,CW-340])
add('All nine new runs satisfy minimum 4 epochs and patience 2 with strict improvement in clean-validation mAP. New runs used 40 epochs in total. No test score selected a checkpoint or refit a threshold.','small')
add('Analysis limits','h2')
add('No confidence intervals or significance tests are claimed from three seeds. Corruption conditions and attributes are not independent replicates. RQ5 follows earlier test inspection on the same benchmark. A single training-only λ was used; the experiments do not rule out other loss weights or architectures. NME is an auxiliary MTL diagnostic without a matched landmark-regression baseline.','small')
add('Reproducibility: raw/ retains original source records; source_manifest.json stores Drive IDs, timestamps and SHA256. audit_report.json records checks and limits. scripts/build_evidence.py rebuilds the tables and figures. paper_findings_en.md and figure_captions.md provide manuscript text; no paper file was edited.','small')

def footer(canvas,doc):
 canvas.saveState();canvas.setStrokeColor(colors.HexColor('#CCCCCC'));canvas.line(M,32,W-M,32)
 canvas.setFont('DV',7);canvas.setFillColor(colors.HexColor('#555555'))
 canvas.drawString(M,21,'Facial Visual Profile · RQ5 · Three seeds · Sample SD, not CI')
 canvas.drawRightString(W-M,21,str(doc.page));canvas.restoreState()
SimpleDocTemplate(str(ROOT/'RQ5_Evidence_Report.pdf'),pagesize=A4,rightMargin=M,leftMargin=M,
    topMargin=38,bottomMargin=43,title='Facial Visual Profile RQ5 Evidence Review',author='Dewey').build(S,onFirstPage=footer,onLaterPages=footer)
print('Report created:',ROOT/'RQ5_Evidence_Report.pdf')
