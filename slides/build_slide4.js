// Slide 4 of the talk: how the force engine was written from scratch and validated.
// Run: node build_slide4.js   ->  slide4_engine.pptx
const pptxgen = require("pptxgenjs");
const path = require("path");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
const DARK = "222222", MUTED = "6B6B6B", RED = "C0392B", TINT = "F3F4F6", BLUE = "1F77B4";
const FONT = "Calibri";
const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

slide.addText("The force engine: written from scratch, validated to machine precision", {
  x: 0.5, y: 0.22, w: 12.3, h: 0.5, fontFace: FONT, fontSize: 24, bold: true, color: DARK, isTextBox: true, margin: 0,
});
slide.addText("New PyTorch implementation of the published screened REBO2 potential, batched for the GPU, checked against the Atomistica reference before any result was produced.", {
  x: 0.5, y: 0.72, w: 12.3, h: 0.3, fontFace: FONT, fontSize: 12, italic: true, color: MUTED, isTextBox: true, margin: 0,
});

const tiles = [
  { n: "A", head: "Written from scratch, not adapted", img: "tileA.png",
    cap: "The functional form and parameter tables were transcribed from the papers and the reference source; the code itself (potential, neighbour list, minimiser, loading driver, MD, analysis, app) is new. Parameters unchanged; their checksum is stored in every run record." },
  { n: "B", head: "Vectorised and batched for the GPU; honest speed", img: "tileB.png",
    cap: "All atoms, pairs and triplets of several structures are one padded tensor; forces and per-atom virials come from autograd. Per call the port is 3× slower than the Fortran reference; the gains are batching, exact per-atom stresses and several processes sharing the GPU." },
  { n: "C", head: "Validated against the reference, through fracture", img: "tileC.png",
    cap: "Energies agree to 6×10⁻¹⁴ eV/atom and forces to 4×10⁻¹³ eV/Å on ordinary and difficult configurations; a complete loading curve to fracture is identical when the same minimiser is used, and within 0.14 N/m with the GPU float32 fast mode and the batched minimiser." },
];
const gx = 0.5, gy = 1.1, tw = 4.05, th = 4.55, gapx = 0.22;
const imgW = tw - 0.3, imgH = imgW * 3.6 / 4.0;
tiles.forEach((t, i) => {
  const x = gx + i * (tw + gapx), y = gy;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: tw, h: th, fill: { color: TINT }, line: { color: TINT, width: 0 }, rectRadius: 0.08 });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.1, y: y + 0.09, w: 0.26, h: 0.26, fill: { color: DARK }, line: { color: DARK, width: 0 } });
  slide.addText(t.n, { x: x + 0.1, y: y + 0.09, w: 0.26, h: 0.26, fontFace: FONT, fontSize: 10, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addText(t.head, { x: x + 0.44, y: y + 0.07, w: tw - 0.54, h: 0.3, fontFace: FONT, fontSize: 11.5, bold: true, color: DARK, valign: "middle", isTextBox: true, margin: 0 });
  slide.addImage({ path: path.join(__dirname, "tiles4", t.img), x: x + 0.15, y: y + 0.42, w: imgW, h: imgH });
  slide.addText(t.cap, { x: x + 0.12, y: y + 0.42 + imgH + 0.04, w: tw - 0.24, h: th - (0.42 + imgH + 0.08), fontFace: FONT, fontSize: 8.3, color: DARK, valign: "top", isTextBox: true, margin: 0 });
});

// bottom band: the 20 tests, grouped, plus the anecdote
const by = gy + th + 0.12, bh = 1.28;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: by, w: 12.33, h: bh, fill: { color: "FFFFFF" }, line: { color: "D0D0D0", width: 0.75 }, rectRadius: 0.06 });
slide.addText("20 validation tests, all passed (validation/results/validation_table.json: expected, observed, error, tolerance)", { x: 0.65, y: by + 0.06, w: 12.0, h: 0.26, fontFace: FONT, fontSize: 10.5, bold: true, color: DARK, isTextBox: true, margin: 0 });
const cols = [
  ["Exactness vs reference (T1, T2, T19)", "energies 6×10⁻¹⁴ eV/atom · forces 4×10⁻¹³ eV/Å · bond rupture E(r), F(r) to 10⁻¹¹ eV/Å (dimer, in-lattice bond, crack tip, affine stretch)"],
  ["Internal consistency (T3, T9–T12, T17)", "finite-difference forces and stress · neighbour list identical to brute force · translation and periodic-image invariance · force balance 10⁻¹⁴ · repeatability on the GPU"],
  ["Fast mode, physics, protocol (T4–T8, T13–T16, T18, T20)", "float32 GPU within 7×10⁻⁶ eV/atom, 1.5×10⁻⁴ eV/Å, 4×10⁻⁵ N/m · lattice constant, cohesive energy, elastic constants, chirality · FIRE-tolerance, strain-step and cell-size convergence · full curve through fracture · export round-trips"],
];
const cw = 2.95, cx0 = 0.65;
cols.forEach(([h, t], i) => {
  const x = cx0 + i * (cw + 0.12);
  slide.addText([{ text: h, options: { bold: true, color: BLUE, breakLine: true } }, { text: t, options: { color: DARK } }],
    { x, y: by + 0.36, w: cw, h: bh - 0.42, fontFace: FONT, fontSize: 8.2, valign: "top", isTextBox: true, margin: 0 });
});
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx0 + 3 * (cw + 0.12) + 0.05, y: by + 0.36, w: 12.83 - (cx0 + 3 * (cw + 0.12) + 0.05) - 0.12, h: bh - 0.46, fill: { color: "FDECEA" }, line: { color: "FDECEA", width: 0 }, rectRadius: 0.06 });
slide.addText([{ text: "The last 10⁻⁸ eV/atom. ", options: { bold: true, color: RED } }, { text: "The port first agreed with the reference only to 10⁻⁸. The residual was traced to the Fortran source storing the bond-order spline knots as single-precision literals; reproducing that rounding closed the gap to 10⁻¹³.", options: { color: DARK } }],
  { x: cx0 + 3 * (cw + 0.12) + 0.15, y: by + 0.42, w: 12.83 - (cx0 + 3 * (cw + 0.12) + 0.15) - 0.2, h: bh - 0.56, fontFace: FONT, fontSize: 8.2, valign: "top", isTextBox: true, margin: 0 });

slide.addText("Reference: Atomistica 1.2.7 Rebo2Scr (Pastewka et al.), used only for validation; production runs use the PyTorch port on the Apple GPU in float32 with the float64 CPU mode as the precision reference. No machine-learned surrogate anywhere in the mechanics.",
  { x: 0.5, y: 7.1, w: 12.3, h: 0.3, fontFace: FONT, fontSize: 8.5, color: MUTED, isTextBox: true, margin: 0 });
slide.addNotes("Engine slide: what was written (pipeline and lines of code), how it runs on the GPU (batched super-system tensor, autograd, multi-process concurrency) with honest speed numbers, and how it was validated (per-configuration agreement with the reference, full stress-strain overlay, the 20-test suite).");
pres.writeFile({ fileName: path.join(__dirname, "slide4_engine.pptx") }).then(f => console.log("wrote", f));
