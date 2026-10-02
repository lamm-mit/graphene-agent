"""Render the final vector PDF and check page bounds, fonts and raster use."""
from pathlib import Path
import json
import fitz

HERE=Path(__file__).resolve().parent
doc=fitz.open(HERE/'fig_phase1_engine.pdf')
page=doc[0]
page.get_pixmap(matrix=fitz.Matrix(2.5,2.5),alpha=False).save(HERE/'fig_phase1_engine.png')
spans=[s for b in page.get_text('dict')['blocks'] if b['type']==0
       for l in b['lines'] for s in l['spans']]
outside=[s['text'] for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
fonts=page.get_fonts(full=True)
qa={'pages':len(doc),'page_size_pt':[page.rect.width,page.rect.height],
    'raster_images':len(page.get_images()), 'fonts':fonts,
    'text_outside_page':outside,'minimum_font_size_pt':min(s['size'] for s in spans),
    'missing_glyphs':[s['text'] for s in spans if '\ufffd' in s['text']],
    'note':'Programmatic checks supplement visual review of the PNG.'}
(HERE/'qa.json').write_text(json.dumps(qa,indent=2))
assert len(doc)==1 and not outside and not qa['missing_glyphs']
assert not page.get_images(), 'Figure must remain vector artwork'
print(json.dumps(qa,indent=2))
