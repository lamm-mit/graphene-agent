# Third-party components and sources

**Interatomic potential.** The force engine in `carbon_discovery/potentials/rebo2scr/` is an independent PyTorch
implementation of the published screened second-generation REBO potential: functional form and parameters from
D. W. Brenner et al., *J. Phys.: Condens. Matter* 14, 783 (2002), and the screening functions from L. Pastewka et al.,
*Phys. Rev. B* 78, 161402(R) (2008) and 87, 205410 (2013). The Fortran reference implementation in
[Atomistica](https://github.com/Atomistica/atomistica) (v1.2.7, GPL-2.0-or-later) was consulted for conventions and
the default spline tables, and is used at run time, through its Python package, only by the validation scripts as the
numerical reference. Atomistica itself is not redistributed here.

**Runtime dependencies** (not bundled): PyTorch (BSD-3-Clause), ASE (LGPL-2.1-or-later), NumPy and SciPy (BSD),
Matplotlib (PSF-based), FastAPI and uvicorn (MIT/BSD), pytest (MIT), PyMuPDF (AGPL-3.0; used only to export one TikZ
figure to SVG), huggingface_hub (Apache-2.0).

**Vendored** in `carbon_discovery/app/frontend/static/vendor/`: three.js `three.module.js` and `OrbitControls.js`,
© 2010–2024 Three.js Authors, MIT license.

**Slides:** `slides/` uses pptxgenjs (MIT) via npm; not bundled.

**Reference images** in `carbon_discovery/reference_images/` were supplied by the human author with the prompt.

**AI agent.** The code, analyses and text in this repository were produced by Claude Fable 5.1 (Anthropic) in Claude
Code, in the two phases described in `README.md` and `PROVENANCE.md`.
