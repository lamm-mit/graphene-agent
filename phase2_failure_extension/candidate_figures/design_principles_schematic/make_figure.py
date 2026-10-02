"""Editable vector schematics; no simulated states or invented measurements.

Rebuild with Python containing reportlab and PyMuPDF. All lengths below are
drawing coordinates, not atomic distances. Scientific qualifications and the
mapping to the manuscript are in figure_environment.tex and README.md.
"""
from pathlib import Path
import json
import math

from reportlab.graphics.shapes import Drawing, Group, Rect, Ellipse, Circle, Line, Path as VectorPath, Polygon, String
from reportlab.graphics import renderPDF, renderSVG
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color, white

HERE = Path(__file__).resolve().parent
W, H = 744, 510
INK = HexColor('#20272d')
EDGE = HexColor('#7e8990')
SHEET = HexColor('#e6e9eb')
BLUE = HexColor('#216a97')
PALE_BLUE = HexColor('#bed6e5')
RED = HexColor('#c4443f')
ORANGE = HexColor('#e38429')
FAINT = HexColor('#c9cfd3')

FONT_DIR = Path('/System/Library/Fonts/Supplemental')
for name, file in [('Arial', 'Arial.ttf'), ('Arial-Bold', 'Arial Bold.ttf'), ('Arial-Italic', 'Arial Italic.ttf')]:
    from matplotlib.font_manager import FontProperties, findfont
    font_path = FONT_DIR / file
    if not font_path.exists():
        font_path = Path(findfont(FontProperties(family="DejaVu Sans", weight="bold" if "Bold" in name else "normal", style="italic" if "Italic" in name else "normal")))
    pdfmetrics.registerFont(TTFont(name, str(font_path)))

d = Drawing(W, H)
d.add(Rect(0, 0, W, H, fillColor=white, strokeColor=None))

def text(g, x, y, value, size=10, bold=False, anchor='middle', color=INK, italic=False):
    font = 'Arial-Italic' if italic else ('Arial-Bold' if bold else 'Arial')
    for i, line in enumerate(value.split('\n')):
        g.add(String(x, y-i*(size+2), line, fontName=font, fontSize=size,
                     textAnchor=anchor, fillColor=color))

def line(g, points, color=EDGE, width=1, dash=None):
    p = VectorPath(strokeColor=color, strokeWidth=width, fillColor=None,
             strokeLineCap=1, strokeLineJoin=1)
    p.moveTo(*points[0])
    for point in points[1:]: p.lineTo(*point)
    if dash: p.strokeDashArray = dash
    g.add(p)
    return p

def arrow(g, points, color=BLUE, width=1.5, head=4):
    line(g, points, color, width)
    a, b = points[-2:]
    vx, vy = b[0]-a[0], b[1]-a[1]
    n = math.hypot(vx, vy)
    vx, vy = vx/n, vy/n
    points = [b[0], b[1], b[0]-head*vx-head*.5*vy,
              b[1]-head*vy+head*.5*vx, b[0]-head*vx+head*.5*vy,
              b[1]-head*vy-head*.5*vx]
    g.add(Polygon(points, fillColor=color, strokeColor=None))

def plate(g, x, y, w, h):
    g.add(Rect(x, y, w, h, fillColor=SHEET, strokeColor=EDGE, strokeWidth=.65))

def pull(g, x, y, w, h):
    arrow(g, [(x-1,y+h/2), (x-9,y+h/2)], head=3)
    arrow(g, [(x+w+1,y+h/2), (x+w+9,y+h/2)], head=3)

def slit(g, a, b, width=4.5):
    line(g, [a,b], EDGE, width+1)
    line(g, [a,b], white, width)

def break_path(g, points, width=2):
    line(g, points, white, width+1.8)
    line(g, points, RED, width)

def heading(g, letter, title):
    text(g, 0, 221, letter, 13, bold=True, anchor='start')
    text(g, 18, 221, title, 11.8, bold=True, anchor='start')

def panel(col, row, letter, title):
    g = Group()
    g.translate(14+244*col, 16+250*(1-row))
    heading(g,letter,title)
    d.add(g)
    return g

# a: Equal removed area; orientation changes continuous load-bearing section.
g=panel(0,0,'a','Continuous load paths')
for j in range(2):
    x,y,w,h=10+116*j,60,90,110
    plate(g,x,y,w,h)
    for cx in (24,66):
        for cy in (28,82):
            rx,ry=(18,6) if j==0 else (6,18)
            g.add(Ellipse(x+cx,y+cy,rx,ry,fillColor=white,strokeColor=EDGE,strokeWidth=.6))
    pull(g,x,y,w,h)
    if j==0:
        for yy in (12,55,98):
            arrow(g,[(x+4,y+yy),(x+86,y+yy)],width=1.35,head=3.5)
    else:
        arrow(g,[(x+4,y+20),(x+12,y+20),(x+24,y+6),(x+44,y+20),
                 (x+52,y+20),(x+66,y+6),(x+86,y+20)],width=1.35,head=3.5)
        arrow(g,[(x+4,y+55),(x+86,y+55)],width=1.35,head=3.5)
        arrow(g,[(x+4,y+90),(x+12,y+90),(x+24,y+104),(x+44,y+90),
                 (x+52,y+90),(x+66,y+104),(x+86,y+90)],width=1.35,head=3.5)
        line(g,[(x+24,y-5),(x+24,y+h+5)],INK,.65,[2,3])
    text(g,x+w/2,39,'Aligned ligaments' if j==0 else 'Narrow bridges',9.5)

# b: Three idealized slit geometries, no universal angle thresholds.
g=panel(1,0,'b','Slit interactions')
for j,label in enumerate(['Tip linking','Ligament\nrotation','Bridge\nbending']):
    x,y,w,h=2+j*78,65,66,107
    plate(g,x,y,w,h)
    if j==0:
        slit(g,(x+5,y+20),(x+44,y+34))
        slit(g,(x+24,y+48),(x+61,y+62))
        slit(g,(x+5,y+77),(x+44,y+91))
        break_path(g,[(x+44,y+34),(x+39,y+38),(x+33,y+40),(x+29,y+45),(x+24,y+48)],1.65)
        break_path(g,[(x+44,y+91),(x+49,y+86),(x+52,y+76),(x+61,y+62)],1.65)
    elif j==1:
        slit(g,(x+5,y+13),(x+33,y+50))
        slit(g,(x+31,y+58),(x+58,y+94))
        arrow(g,[(x+12,y+69),(x+17,y+79),(x+29,y+82),(x+41,y+77)],width=1.5)
        arrow(g,[(x+53,y+39),(x+49,y+29),(x+38,y+25),(x+26,y+29)],width=1.5)
        line(g,[(x+13,y+62),(x+53,y+43)],BLUE,1.5)
    else:
        slit(g,(x+20,y+32),(x+20,y+98))
        slit(g,(x+46,y+9),(x+46,y+72))
        arrow(g,[(x+3,y+53),(x+9,y+51),(x+10,y+22),(x+30,y+22),
                 (x+35,y+82),(x+56,y+82),(x+59,y+53),(x+64,y+53)],width=1.5,head=3.5)
    text(g,x+w/2,41,label,9.4)

# c: Tracked neighbor exchange. Lattice fragments are schematic, not saved atoms.
g=panel(2,0,'c','Evolving connectivity')
def fragment(g,x,y,shift=0,upper=False):
    # A true honeycomb graph: interior degree three, bond lengths all eight.
    # The moving fragment is viewed through the same schematic window.
    nodes,edges=set(),set()
    for i in range(-1,7):
        for k in range(-2,6):
            cx,cy=6+12*i,4+math.sqrt(3)*8*(k+(i%2)*.5)
            vv=[(round(cx+8*math.cos(t*math.pi/3),5),
                 round(cy+8*math.sin(t*math.pi/3),5)) for t in range(6)]
            for t in range(6):
                a,b=vv[t],vv[(t+1)%6]
                if all(0<=xx<=66 and 0<=yy<=43 for xx,yy in (a,b)):
                    nodes.update([a,b]);edges.add(tuple(sorted([a,b])))
    def position(p):
        return x+p[0]+shift,y+(43-p[1] if upper else p[1])
    kept={p for p in nodes if 0<=p[0]+shift<=66}
    for a,b in edges:
        if a in kept and b in kept: line(g,[position(a),position(b)],FAINT,.8)
    for p in kept:
        g.add(Circle(*position(p),1.5,fillColor=EDGE,strokeColor=None))

for j,shift in enumerate([0,-12,-24]):
    x,y=j*79,78
    fragment(g,x,y)
    fragment(g,x,y+43,shift,upper=True)
    p=(x+26,y+38.64102)
    q=(x+26+shift,y+47.35898)
    r=(x+50+shift,y+47.35898)
    if j==0: line(g,[p,q],INK,1.7)
    elif j==1:
        line(g,[p,(p[0]-3,p[1]+2)],RED,1.7)
        line(g,[(q[0]+3,q[1]-2),q],RED,1.7)
    else: line(g,[p,r],ORANGE,2.2)
    for pt in [p,q,r]: g.add(Circle(*pt,2.35,fillColor=INK,strokeColor=white,strokeWidth=.4))
    for value,pt,dx,dy in [('p',p,5,-5),('q',q,-6,7),('r',r,5,7)]:
        g.add(Circle(pt[0]+dx,pt[1]+dy+2,4.1,fillColor=white,strokeColor=None))
        text(g,pt[0]+dx,pt[1]+dy,value,9,italic=True)
    if j<2: arrow(g,[(x+67,y+43),(x+76,y+43)],EDGE,.8,3)
    if j==1:
        arrow(g,[(x+43,y+98),(x+19,y+98)],BLUE,1.25,3.5)
        arrow(g,[(x+19,y-9),(x+43,y-9)],BLUE,1.25,3.5)

def hierarchy(g,x,y,w=94,h=110):
    plate(g,x,y,w,h)
    # Continuous horizontal veins and transverse boundaries form compartments.
    for yy in (0,51,102):
        g.add(Rect(x,y+yy,w,8,fillColor=PALE_BLUE,strokeColor=None))
    for xx in (0,44,86):
        g.add(Rect(x+xx,y,8,h,fillColor=PALE_BLUE,strokeColor=None))
    for cx in (17,35,61,79):
        for cy in (19,38,70,89):
            g.add(Ellipse(x+cx,y+cy,5.5,3.7,fillColor=white,strokeColor=EDGE,strokeWidth=.35))
    return w,h

def domain_crack(g,x,y,cellx,celly):
    xx=x+24+44*cellx
    yy=y+8+51*celly
    break_path(g,[(xx,yy),(xx-3,yy+10),(xx+1,yy+18),(xx-2,yy+30),(xx,yy+43)],1.5)

# d: Primary veins remain connected while fine domains fail in sequence.
g=panel(0,1,'d','Hierarchical load sharing')
for j in range(2):
    x,y=8+j*120,62
    w,h=hierarchy(g,x,y)
    domain_crack(g,x,y,0,0)
    if j:
        domain_crack(g,x,y,1,1)
        domain_crack(g,x,y,0,1)
    for yy in (4,55,106): arrow(g,[(x+3,y+yy),(x+w-3,y+yy)],width=1.15,head=3)
    text(g,x+w/2,40,'First domain' if j==0 else 'Sequential damage',9.5)
arrow(g,[(106,117),(121,117)],EDGE,.8,3.4)

# e: Same initial crack length; position determines which vein is interrupted.
g=panel(1,1,'e','Flaw location')
for j in range(2):
    x,y=8+j*120,62
    w,h=hierarchy(g,x,y)
    cy=y+(30 if j==0 else 56)
    xx=x+25
    # A black slot denotes the initial flaw; red denotes its subsequent growth.
    slit(g,(xx,cy-12),(xx,cy+12),3.5)
    line(g,[(xx,cy-12),(xx,cy+12)],INK,1.35)
    if j==0:
        break_path(g,[(xx,cy+12),(xx+3,y+48),(xx-1,y+51),(xx+11,y+51)],1.35)
        break_path(g,[(xx,cy-12),(xx-2,y+13),(xx+1,y+8),(xx-10,y+8)],1.35)
        arrow(g,[(x+3,y+55),(x+w-3,y+55)],width=1.15,head=3)
    else:
        break_path(g,[(xx,cy+12),(xx+3,y+81),(xx-1,y+91)],1.35)
        break_path(g,[(xx,cy-12),(xx-3,y+35),(xx+1,y+23)],1.35)
        arrow(g,[(x+3,y+55),(xx-5,y+55)],width=1.15,head=3)
        arrow(g,[(xx+5,y+55),(x+w-3,y+55)],width=1.15,head=3)
    pull(g,x,y,w,h)
    text(g,x+w/2,40,'Inside a domain' if j==0 else 'Across a vein',9.5)

# f: Constituent architecture changes damage sequence, not a universal ranking.
g=panel(2,1,'f','Vein architecture')
for j,name in enumerate(['Solid','Bundle','Sub-veins']):
    x=3+j*79
    text(g,x+31,187,name,10)
    if j==0:
        plate(g,x,132,62,27)
        plate(g,x,50,62,27)
        break_path(g,[(x+31,48),(x+28,59),(x+34,66),(x+30,79)],2)
    elif j==1:
        for k in range(4):
            yy=132+k*8
            g.add(Rect(x,yy,62,4.5,fillColor=SHEET,strokeColor=EDGE,strokeWidth=.5))
            yy=50+k*8
            g.add(Rect(x,yy,62,4.5,fillColor=PALE_BLUE if k<2 else SHEET,strokeColor=EDGE,strokeWidth=.5))
            if k>=2: break_path(g,[(x+25+4*k,yy-1),(x+23+4*k,yy+5.5)],1.4)
            else: arrow(g,[(x+3,yy+2.25),(x+59,yy+2.25)],width=.85,head=2.8)
    else:
        for yy in (119,37):
            plate(g,x,yy,62,53)
            for ry in (0,24,48):
                g.add(Rect(x,yy+ry,62,5,fillColor=PALE_BLUE,strokeColor=None))
            for rx in (0,29,58):
                g.add(Rect(x+rx,yy,4,53,fillColor=PALE_BLUE,strokeColor=None))
            for cx in (15,46):
                for cy in (14,38):
                    g.add(Ellipse(x+cx,yy+cy,8,4,fillColor=white,strokeColor=EDGE,strokeWidth=.4))
        break_path(g,[(x+12,42),(x+10,48),(x+12,55),(x+9,61)],1.35)
        break_path(g,[(x+45,66),(x+43,74),(x+46,81),(x+43,85)],1.35)
    arrow(g,[(x+31,112),(x+31,97)],EDGE,.9,3.6)

# Fine separators aid scanning without enclosing panels in heavy boxes.
for x in (249,493): d.add(Line(x,20,x,491,strokeColor=FAINT,strokeWidth=.45))
d.add(Line(15,255,730,255,strokeColor=FAINT,strokeWidth=.45))

pdf=HERE/'fig_design_principles.pdf'
renderPDF.drawToFile(d,str(pdf),title='Architecture, evolving connectivity and the control of failure',
                     author='Schematic synthesis for graphene mechanics manuscript')
renderSVG.drawToFile(d,str(HERE/'fig_design_principles.svg'))
manifest={
    'purpose':'Idealized mechanism and design-rule synthesis; not simulation data',
    'page_size_pt':[W,H],
    'panel_a_equal_pore_area':{'each_sheet_area':90*110,'each_total_removed_area':4*math.pi*18*6},
    'panel_e_initial_crack_length_each':24,
    'source_manuscript':'paper_latest_20260924',
    'scope':{
        'a':'Figs 4,5,11,14: load paths, section and alignment; same idealized removed area.',
        'b':'Figs 6,7,14: mechanisms; no invariant angle thresholds or strength ranking.',
        'c':'Figs 8-10: observed geometric neighbor exchange; no healing or defect-free-slip claim.',
        'd':'Figs 12,13: tested hierarchical geometries; no isolated causal scale-separation law.',
        'e':'Fig 13: contained versus vein-interrupting cracks; no universal arrest criterion.',
        'f':'SI Fig S31: building-block dependent post-peak sequence; no exact retention claims.'}
}
(HERE/'schematic_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Created vector PDF and SVG')
