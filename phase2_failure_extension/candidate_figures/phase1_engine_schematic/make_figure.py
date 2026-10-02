"""Source-grounded schematic of the frozen Phase-I atomistic instrument.

Vector drawing only; no simulation data or results are synthesized. See
source_map.json and figure_environment.tex for the scope of the abstraction.
"""
from pathlib import Path
import math
from reportlab.graphics.shapes import Drawing, Rect, Circle, Polygon, String, Path as VPath
from reportlab.graphics import renderPDF, renderSVG
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white

HERE = Path(__file__).resolve().parent
W, H = 744, 690
FONT_DIR = Path('/System/Library/Fonts/Supplemental')
for name, filename in [('Arial', 'Arial.ttf'), ('Arial-Bold', 'Arial Bold.ttf')]:
    from matplotlib.font_manager import FontProperties, findfont
    font_path = FONT_DIR / filename
    if not font_path.exists():
        font_path = Path(findfont(FontProperties(family="DejaVu Sans", weight="bold" if "Bold" in name else "normal", style="italic" if "Italic" in name else "normal")))
    pdfmetrics.registerFont(TTFont(name, str(font_path)))
INK = HexColor('#20272d')
GRAY = HexColor('#6d7b84')
LINE = HexColor('#bcc6cc')
BLUE = HexColor('#216a97')
PALE_BLUE = HexColor('#edf5fa')
PALE_GRAY = HexColor('#f5f6f7')
ORANGE = HexColor('#bb6a23')
PALE_ORANGE = HexColor('#fcf4eb')
GREEN = HexColor('#467765')
d = Drawing(W, H)
d.add(Rect(0, 0, W, H, fillColor=white, strokeColor=None))

def text(x, y, s, size=11.5, bold=False, color=INK, anchor='middle'):
    d.add(String(x, y, s, fontName='Arial-Bold' if bold else 'Arial',
                 fontSize=size, fillColor=color, textAnchor=anchor))

def rich_center(x,y,runs,size=10.8,color=INK):
    """runs is [(text, relative_size, vertical_offset), ...]."""
    widths=[pdfmetrics.stringWidth(s,'Arial',size*scale) for s,scale,dy in runs]
    xx=x-sum(widths)/2
    for (s,scale,dy),ww in zip(runs,widths):
        text(xx,y+dy,s,size*scale,color=color,anchor='start')
        xx+=ww

def path(points, color=GRAY, width=1, dash=None):
    p=VPath(strokeColor=color, strokeWidth=width, fillColor=None,
            strokeLineJoin=1, strokeLineCap=1)
    p.moveTo(*points[0])
    for xy in points[1:]: p.lineTo(*xy)
    if dash: p.strokeDashArray=dash
    d.add(p)

def head(a, b, color=BLUE, size=4):
    dx,dy=b[0]-a[0],b[1]-a[1]
    n=math.hypot(dx,dy);dx/=n;dy/=n
    d.add(Polygon([*b,b[0]-size*dx-size*.5*dy,b[1]-size*dy+size*.5*dx,
                   b[0]-size*dx+size*.5*dy,b[1]-size*dy-size*.5*dx],
                  fillColor=color,strokeColor=None))

def arrow(points, color=BLUE, width=1.25, dash=None, both=False):
    path(points,color,width,dash)
    head(points[-2],points[-1],color)
    if both: head(points[1],points[0],color)

def box(x,y,w,h,fill=white,edge=LINE):
    d.add(Rect(x,y,w,h,rx=3,ry=3,fillColor=fill,strokeColor=edge,strokeWidth=.8))

def panel(x,y,letter,title):
    text(x,y,letter,14,True,anchor='start')
    text(x+19,y,title,12.5,True,anchor='start')

# a. The interactive backend and process queue both spawn run_batch.py.
# reproduce_run.py rebuilds the stored design and calls AQSRunner directly.
panel(16,669,'a','Entry points')
for x in (16,260,504):box(x,584,224,67,PALE_GRAY)
text(128,631,'Browser / REST API',12,True)
text(128,610,'POST /api/generate',11)
text(128,595,'POST /api/simulate',11)
text(372,631,'Campaign / command line',12,True)
text(372,610,'JSON specifications',11)
text(372,595,'run_queue.py  →  worker processes',10.8)
text(616,631,'Replay a stored run',12,True)
text(616,610,'reproduce_run.py',11)
text(616,595,'Geometry, seed and saved settings',10.8)
box(110,542,300,27,PALE_BLUE,BLUE)
text(260,551,'run_batch.py  •  one batch specification',11.5,True,color=BLUE)
arrow([(128,584),(128,576),(190,576),(190,569)])
arrow([(372,584),(372,569)])
path([(260,542),(260,532),(616,532),(616,584)],BLUE,1.25)
arrow([(260,532),(125,532),(125,521)])
box(16,474,712,47,white)
text(125,502,'Generate graphene geometry',11.5,True)
text(125,485,'Family + parameters + seed',10.6,color=GRAY)
arrow([(227,498),(257,498)])
text(366,502,'Pack independent structures',11.5,True)
text(366,485,'BatchedSystem: positions, cells, IDs',10.6,color=GRAY)
arrow([(480,498),(510,498)])
text(620,502,'AQS loading',11.5,True)
text(620,485,'AQSRunner (b)',10.6,color=BLUE)
path([(16,458),(728,458)],LINE,.65)

# b. Coarse-grained AQS control loop. Early/late first-damage rollback is
# represented by one dashed retry loop; the caption states the full scope.
panel(16,437,'b','Quasi-static loading')
text(35,420,'Discovery campaign',10.8,color=GRAY,anchor='start')
box(90,369,275,34,PALE_GRAY)
text(227.5,382,'Relax initial atoms and cell',11.5,True)
arrow([(227.5,369),(227.5,358),(122,358),(122,345)])
box(40,292,164,53,PALE_BLUE,BLUE)
text(122,325,'Apply strain increment',11.5,True)
text(122,307,'Δε = 0.010 / 0.005',11,color=BLUE)
box(246,292,164,53,PALE_BLUE,BLUE)
text(328,325,'FIRE minimization',11.5,True)
text(328,307,'Force target: 0.02 eV/Å',10.8,color=BLUE)
arrow([(204,319),(246,319)])
arrow([(328,292),(328,270),(122,270),(122,292)],GRAY,1,[3,2])
text(225,279,'First bond loss: refine and retry',10.1,color=GRAY)
arrow([(382,292),(382,255)])
box(246,202,164,53,PALE_BLUE,BLUE)
text(328,234,'Relax transverse cell',11.3,True)
rich_center(328,216,[('Secant updates toward σ',1,0),('yy',.75,-2.5),(' = 0',1,0)],10.2,BLUE)
arrow([(246,229),(204,229)])
box(40,202,164,53,PALE_GRAY)
text(122,234,'Accept and record',11.5,True)
text(122,216,'Stress, geometry and bonds',10.4)
arrow([(122,202),(122,180)])
box(40,103,370,77,PALE_ORANGE,ORANGE)
text(225,160,'Stop-rule check',11.8,True,color=ORANGE)
text(225,141,'Loss of spanning path, stress collapse or loading limit',10.8)
rich_center(225,122,[('Original limits: ε',1,0),('max',.75,-2.5),(' = 0.32; post-peak interval = 0.12',1,0)])
arrow([(40,140),(25,140),(25,319),(40,319)],BLUE)
text(26,332,'No',10.5,color=BLUE)
arrow([(225,103),(225,79)],ORANGE)
text(239,91,'Yes',10.5,color=ORANGE,anchor='start')
box(40,20,370,59,PALE_GRAY)
text(225,59,'Run record + trajectory',11.8,True)
text(225,42,'Curves, metrics, fields, flags and stop reason',10.8)
text(225,27,'Database  →  browser, analysis and export',10.8,color=GRAY)

# c. One force kernel is shared by the AQS driver, optional MD and ASE.
panel(455,437,'c','Shared force engine')
text(474,419,'TorchRebo2Scr',10.8,color=BLUE,anchor='start')
box(472,364,242,41,PALE_GRAY)
text(593,389,'Periodic neighbor lists',11.5,True)
text(593,373,'Screening + bond-order neighborhoods',10.5)
arrow([(593,364),(593,349)])
box(472,293,242,56,PALE_BLUE,BLUE)
text(593,331,'Screened REBO2 energy',11.8,True,color=BLUE)
text(593,313,'Radial + angular + screening terms',10.8)
text(593,300,'Checked against Atomistica',10.3,color=GREEN)
arrow([(593,293),(593,277)])
box(472,238,242,39,PALE_BLUE,BLUE)
text(593,261,'Differentiate the energy',11.5,True)
text(593,246,'PyTorch autograd',10.8,color=BLUE)
arrow([(593,238),(593,224)])
text(593,211,'Forces  •  virial  •  per-atom fields',11.3,True)
text(593,195,'AQS stress = virial / current area',10.8,color=BLUE)
arrow([(410,319),(472,319)],BLUE,1.4,both=True)
text(440,337,'Calls',10.4,color=BLUE)

# MD is a separate time integrator, never the campaign's loading driver.
box(472,101,242,72,PALE_ORANGE,ORANGE)
text(593,156,'Optional finite-temperature MD',11.3,True,color=ORANGE)
text(593,140,'run_md()',10.8)
text(593,125,'Verlet + Langevin; strain-rate loading',10.5)
text(593,110,'Time, temperature, energy and stress',10.2,color=GRAY)
arrow([(593,190),(593,173)],ORANGE,1.25,both=True)

box(472,20,242,57,PALE_GRAY)
text(593,60,'Python / ASE entry',11.5,True)
text(593,43,'evaluate_atoms() or ASE calculator',10.8)
text(593,28,'CPU / MPS / CUDA',10.8,color=GRAY)
arrow([(714,48),(728,48),(728,384),(714,384)],GRAY,1)

renderPDF.drawToFile(d,str(HERE/'fig_phase1_engine.pdf'))
renderSVG.drawToFile(d,str(HERE/'fig_phase1_engine.svg'))
print('Created fig_phase1_engine.pdf and fig_phase1_engine.svg')
