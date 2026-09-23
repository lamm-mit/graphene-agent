// Slide 3 of the talk: the four mechanisms as pure schematics, each next to one simple chart of real data.
// Run: node build_slide3.js   ->  slide3_mechanisms_data.pptx
const pptxgen = require("pptxgenjs");
const path = require("path");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
const DARK = "222222", MUTED = "6B6B6B", RED = "C0392B", TINT = "F3F4F6", GREYTAG = "8A8A8A";
const FONT = "Calibri";
const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

slide.addText("The four mechanisms: sketch on the left, one measurement on the right", {
  x: 0.5, y: 0.22, w: 12.3, h: 0.5, fontFace: FONT, fontSize: 26, bold: true, color: DARK, isTextBox: true, margin: 0,
});
slide.addText("Every number is one simulation from the campaign database (screened REBO2, athermal quasi-static tension, 2D strength in N/m); designs compared at the same porosity.", {
  x: 0.5, y: 0.72, w: 12.3, h: 0.3, fontFace: FONT, fontSize: 12, italic: true, color: MUTED, isTextBox: true, margin: 0,
});

const tiles = [
  { n: "1", tag: "known, sharpened", tagColor: GREYTAG, head: "Strength = weakest section × ligament strength", img: "tile1.png",
    cap: "Halving the ligament width at the same porosity costs a third of the strength; changing the pore shape at the same net section costs nothing; armchair ligaments are 20% weaker than zigzag." },
  { n: "2", tag: "new", tagColor: RED, head: "Alignment has a cliff, not a slope", img: "tile2.png",
    cap: "Strength is not monotonic in the slit angle: a 20° tilt is worse than 45° because the slit tips of neighbouring rows overlap and link en echelon." },
  { n: "3", tag: "new", tagColor: RED, head: "Hierarchy pays only when aligned, and adds up", img: "tile3.png",
    cap: "At matched mass the nested mesh is weaker than the single-level one until a level is aligned with the load; aligning both levels adds the two gains." },
  { n: "4", tag: "known, sharpened", tagColor: GREYTAG, head: "Fracture mode = load transfer between paths", img: "tile4.png",
    cap: "Many ligaments hand over a small share when one breaks and fail row by row; three wide strips hand over a large share and fail in one avalanche, dispersion or not." },
];
const gx = 0.5, gy = 1.08, tw = 6.05, th = 2.92, gapx = 0.23, gapy = 0.12;
const imgW = tw - 0.6, imgH = imgW * 2.35 / 6.0;
tiles.forEach((t, i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = gx + col * (tw + gapx), y = gy + row * (th + gapy);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: tw, h: th, fill: { color: TINT }, line: { color: TINT, width: 0 }, rectRadius: 0.08 });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.1, y: y + 0.09, w: 0.26, h: 0.26, fill: { color: DARK }, line: { color: DARK, width: 0 } });
  slide.addText(t.n, { x: x + 0.1, y: y + 0.09, w: 0.26, h: 0.26, fontFace: FONT, fontSize: 10, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
  slide.addText(t.head, { x: x + 0.44, y: y + 0.07, w: tw - 1.9, h: 0.3, fontFace: FONT, fontSize: 12.5, bold: true, color: DARK, valign: "middle", isTextBox: true, margin: 0 });
  slide.addText(t.tag.toUpperCase(), { x: x + tw - 1.45, y: y + 0.1, w: 1.35, h: 0.24, fontFace: FONT, fontSize: 8, bold: true, color: t.tagColor, align: "right", valign: "middle", charSpacing: 1, isTextBox: true, margin: 0 });
  slide.addImage({ path: path.join(__dirname, "tiles3", t.img), x: x + (tw - imgW) / 2, y: y + 0.4, w: imgW, h: imgH });
  slide.addText(t.cap, { x: x + 0.12, y: y + 0.4 + imgH + 0.03, w: tw - 0.24, h: 0.34, fontFace: FONT, fontSize: 9, color: DARK, valign: "top", isTextBox: true, margin: 0 });
});

slide.addText("Tiles 2 and 3 are findings we could not locate in the prior literature; tiles 1 and 4 are established principles (net-section scaling, fibre-bundle load sharing) confirmed and refined at the atomic scale. Work to failure is a proxy, not a fracture toughness.",
  { x: 0.5, y: 7.12, w: 12.3, h: 0.3, fontFace: FONT, fontSize: 8.5, color: MUTED, isTextBox: true, margin: 0 });
slide.addNotes("Schematic + data slide: each tile is a sketch of the mechanism and one simple chart of real campaign data (bar charts of 2D strength, strength versus slit angle, and two stress-strain curves).");
pres.writeFile({ fileName: path.join(__dirname, "slide3_mechanisms_data.pptx") }).then(f => console.log("wrote", f));
