// Slide 2 of the talk: four mechanisms (schematic + simulation snapshot per tile) and summary bullets.
// Run: node build_slide2.js   ->  slide2_mechanisms.pptx
const pptxgen = require("pptxgenjs");
const path = require("path");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
const DARK = "222222", MUTED = "6B6B6B", RED = "C0392B", TINT = "F3F4F6";
const FONT = "Calibri";
const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

// title
slide.addText("Four mechanisms set the strength and fracture mode of architected graphene", {
  x: 0.5, y: 0.22, w: 12.3, h: 0.52, fontFace: FONT, fontSize: 26, bold: true, color: DARK, isTextBox: true, margin: 0,
});
slide.addText("96 architectures, ten families, one validated model, matched porosity — what actually moves strength", {
  x: 0.5, y: 0.74, w: 12.3, h: 0.3, fontFace: FONT, fontSize: 12.5, italic: true, color: MUTED, isTextBox: true, margin: 0,
});

// mechanism tiles: 2 x 2 grid on the left (each: header, image, caption)
const tiles = [
  { n: "1", head: "Weakest section × ligament strength", img: "tile1.png",
    cap: "Pore shape is irrelevant at the coarse scale; ligament width and lattice orientation set σ_lig (armchair ligaments −20%)." },
  { n: "2", head: "Alignment has a cliff, not a slope", img: "tile2.png",
    cap: "Parallel slits 25 N/m, perpendicular 4 N/m; a 20° rotation already collapses to 9 N/m, below the 45° array (18.5)." },
  { n: "3", head: "Hierarchy pays only when aligned", img: "tile3.png",
    cap: "Nested meshes at matched mass never beat the single-level mesh unless a level is aligned with the load; aligning both levels adds the two effects (19.4 N/m, best nested design)." },
  { n: "4", head: "Fracture mode = load-transfer property", img: "tile4.png",
    cap: "8–10 parallel ligaments fail row by row and keep 20–60% of peak load; three wide strips cascade even with 15% dispersion." },
];
const gx = 0.5, gy = 1.12, tw = 3.9, th = 2.87, gapx = 0.22, gapy = 0.12;
tiles.forEach((t, i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = gx + col * (tw + gapx), y = gy + row * (th + gapy);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: tw, h: th, fill: { color: TINT }, line: { color: TINT, width: 0 }, rectRadius: 0.08 });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.1, y: y + 0.09, w: 0.26, h: 0.26, fill: { color: DARK }, line: { color: DARK, width: 0 } });
  slide.addText(t.n, { x: x + 0.1, y: y + 0.09, w: 0.26, h: 0.26, fontFace: FONT, fontSize: 10, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addText(t.head, { x: x + 0.44, y: y + 0.07, w: tw - 0.52, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: DARK, valign: "middle", isTextBox: true, margin: 0 });
  slide.addImage({ path: path.join(__dirname, "tiles", t.img), x: x + 0.1, y: y + 0.4, w: tw - 0.2, h: (tw - 0.2) * 2.2 / 4.0 });
  slide.addText(t.cap, { x: x + 0.12, y: y + th - 0.4, w: tw - 0.24, h: 0.38, fontFace: FONT, fontSize: 8.5, color: DARK, valign: "top", isTextBox: true, margin: 0 });
});

// summary bullets on the right
const bx = gx + 2 * tw + gapx + 0.3, bw = 13.33 - bx - 0.5;
slide.addText("What is new", { x: bx, y: gy + 0.02, w: bw, h: 0.34, fontFace: FONT, fontSize: 15, bold: true, color: DARK, isTextBox: true, margin: 0 });
const bullets = [
  ["Strength is not a rule of mixtures; modulus nearly is. ", "At equal mass, specific strength spans a factor of 6 across families; specific modulus stays within 110–275 N/m."],
  ["Load paths, not mass, separate the families. ", "Only load-parallel slit arrays keep >80% of the pristine specific strength."],
  ["Alignment fails abruptly once slit tips overlap ", "(en-echelon coalescence), a mechanism absent from all training data."],
  ["Hierarchy is not a free lunch: ", "aligned levels add, unaligned levels only cost mass."],
  ["Disorder, gradients and pore halos are costs; ", "apparent optima vanished under seed replication."],
  ["Pre-registered holdouts: ", "median strength error 14%, mode right in 10/12; the misses were exactly the new mechanisms."],
];
const runs = [];
bullets.forEach(([b, r], i) => {
  runs.push({ text: b, options: { bold: true, color: DARK, bullet: { indent: 14 }, breakLine: false } });
  runs.push({ text: r, options: { color: DARK, breakLine: i < bullets.length - 1, paraSpaceAfter: 6 } });
});
slide.addText(runs, { x: bx, y: gy + 0.42, w: bw, h: 5.4, fontFace: FONT, fontSize: 12, color: DARK, valign: "top", isTextBox: true, margin: 0, paraSpaceAfter: 6 });

// scope line
slide.addText("Model results only: screened REBO2 (validated against the Atomistica reference), athermal quasi-static uniaxial tension, 2D stress in N/m; work to failure is a proxy, not a fracture toughness. Red circles in snapshots: atoms that lost a bond.",
  { x: 0.5, y: 7.06, w: 12.3, h: 0.32, fontFace: FONT, fontSize: 8.5, color: MUTED, isTextBox: true, margin: 0 });

slide.addNotes("Mechanism slide: each tile pairs a schematic of the idea with a real simulation snapshot from the database. Numbers are 2D strengths in N/m for the screened REBO2 model under quasi-static tension.");
pres.writeFile({ fileName: path.join(__dirname, "slide2_mechanisms.pptx") }).then(f => console.log("wrote", f));
