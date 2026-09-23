# Talk slides (phase 2)

Three slides made on request during phase 2, with their generators:

| slide | file | content |
|---|---|---|
| 2 | `slide2_mechanisms.pptx` | the four mechanisms, each as a schematic next to an atomistic snapshot, with bullets |
| 3 | `slide3_mechanisms_data.pptx` | the same mechanisms as pure schematics next to one simple chart of real data each |
| 4 | `slide4_engine.pptx` | how the force engine was written from scratch, batched for the GPU, and validated |

`*_preview.png` are renderings of the slides. The tiles (`tiles/`, `tiles3/`, `tiles4/`) are generated from the
experiment database by `make_slide{2,3,4}_tiles.py`; the decks are assembled by `build_slide{2,3,4}.js` with
[pptxgenjs](https://github.com/gitbrent/PptxGenJS):

```bash
python make_slide2_tiles.py && python make_slide3_tiles.py && python make_slide4_tiles.py
npm install && node build_slide2.js && node build_slide3.js && node build_slide4.js
```

Slide 3 deliberately shows two mechanisms as new and two as known physics sharpened by the campaign; the distinction is
explained in the paper.
