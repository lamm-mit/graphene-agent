"""Render and inspect the generated PDF using PyMuPDF."""
import json
from pathlib import Path
import fitz

root=Path(__file__).resolve().parent
doc=fitz.open(root/'fig_design_principles.pdf')
doc[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5),alpha=False).save(root/'fig_design_principles.png')
spans=[s for b in doc[0].get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
outside=[s for s in spans if not doc[0].rect.contains(fitz.Rect(s['bbox']))]
qa={'pages':len(doc),'page_size_pt':list(doc[0].rect),
    'fonts':sorted(set(s['font'] for s in spans)),
    'font_records':doc[0].get_fonts(full=True),
    'raster_images_in_pdf':len(doc[0].get_images()),
    'text_outside_page':outside}
(root/'qa.json').write_text(json.dumps(qa,indent=2)+'\n')
assert len(doc)==1 and not outside
assert not doc[0].get_images()
print(json.dumps({k:v for k,v in qa.items() if k!='font_records'}))
