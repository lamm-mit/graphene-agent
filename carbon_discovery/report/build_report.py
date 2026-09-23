"""Assemble and compile the LaTeX report from the stored data (validation table, campaign tables,
hypotheses, holdout evaluation, top structures) -- every number in the generated snippets comes from
JSON/CSV produced by the simulations."""
from __future__ import annotations
import re
import sys, os, json, glob, subprocess, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from experiments import db
GEN = os.path.join(ROOT, "report", "generated")
os.makedirs(GEN, exist_ok=True)



def _fix_longtable(s):
    """longtable: keep caption+label on the first page only; repeat the column header on continuation pages."""
    out = []
    for line in s.split("\n"):
        if "\\endhead" in line and "\\endfirsthead" not in line and line.lstrip().startswith("\\toprule"):
            out.append(line.replace("\\endhead", "\\endfirsthead")); out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def tex_escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("^", "\\^{}").replace(">", "$>$").replace("<", "$<$").replace("|", "$|$")


def fmt(v, d=2):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return "--"
        if isinstance(v, (int, np.integer)): return str(v)
        return f"{v:.{d}f}"
    except Exception:
        return tex_escape(v)


def validation_table():
    p = os.path.join(ROOT, "validation", "results", "validation_table.json")
    rows = json.load(open(p)) if os.path.exists(p) else []
    out = ["\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{longtable}{>{\\raggedright\\arraybackslash}p{1.1cm}>{\\raggedright\\arraybackslash}p{4.4cm}>{\\raggedright\\arraybackslash}p{2.7cm}>{\\raggedright\\arraybackslash}p{4.1cm}>{\\raggedright\\arraybackslash}p{1.2cm}p{0.9cm}}",
           "\\caption{Validation suite: expected result, observed result, numerical error, tolerance and status (PASS/FAIL). Details in \\texttt{validation/results/}.}\\label{tab:validation}\\\\", "\\toprule test & name & expected & observed & error & status\\\\ \\midrule\\endhead"]
    for r in rows:
        err = r["error"]; err = f"{err:.1e}" if isinstance(err, (int, float)) else tex_escape(err)
        out.append(f"{tex_escape(r['test'])} & {tex_escape(r['name'])} & {tex_escape(r['expected'])} & {tex_escape(r['observed'])} & {err} (tol {r['tolerance']:.0e}) & \\textbf{{{r['status']}}}\\\\")
    out.append("\\bottomrule\\end{longtable}\\endgroup")
    open(os.path.join(GEN, "validation_table.tex"), "w").write(_fix_longtable("\n".join(out)))


def runs_table(recs, path, caption, label, cols):
    out = ["\\begin{landscape}\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{longtable}{" + "l" * len(cols) + "}", f"\\caption{{{caption}}}\\label{{{label}}}\\\\", "\\toprule " + " & ".join(tex_escape(c[1]) for c in cols) + "\\\\ \\midrule\\endhead"]
    for r in recs:
        vals = []
        for key, _, d in cols:
            if key.startswith("m."): v = r.get("metrics", {}).get(key[2:])
            elif key.startswith("d."): v = r.get("descriptors", {}).get(key[2:])
            else: v = r.get(key)
            vals.append(tex_escape(v) if isinstance(v, str) else fmt(v, d))
        out.append(" & ".join(vals) + "\\\\")
    out.append("\\bottomrule\\end{longtable}\\end{landscape}")
    open(path, "w").write(_fix_longtable("\n".join(out)))


def numbers(recs):
    lines = []
    def cmd(name, val): lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    cmd("nRuns", len(recs)); cmd("nCompleted", sum(1 for r in recs if r.get("status") == "completed"))
    for st in sorted(set(str(r.get("stage")) for r in recs)):
        n = sum(1 for r in recs if str(r.get("stage")) == st and r.get("status") == "completed")
        cmd("n" + "".join(ch for ch in st.title() if ch.isalpha()), n)
    via = {}
    for r in recs:
        via[r.get("viability")] = via.get(r.get("viability"), 0) + 1
    cmd("viabilitySummary", "; ".join(f"{tex_escape(k)}: {v}" for k, v in via.items()))
    env = db.environment_record()
    cmd("envOS", tex_escape(env["os"])); cmd("envCPU", tex_escape(env["cpu"])); cmd("envGPU", tex_escape(env["gpu"])); cmd("envTorch", env["torch"]); cmd("envASE", env["ase"]); cmd("envAtomistica", env["atomistica"]); cmd("envPython", env["python"])
    open(os.path.join(GEN, "numbers.tex"), "w").write("\n".join(lines))


def hypotheses_tex():
    p = os.path.join(ROOT, "experiments", "predictions", "hypotheses.json")
    if not os.path.exists(p):
        open(os.path.join(GEN, "hypotheses.tex"), "w").write("Hypotheses not yet recorded."); return
    H = json.load(open(p))
    out = []
    for h in H["hypotheses"]:
        out.append(f"\\paragraph{{{tex_escape(h['id'])}: {tex_escape(h['title'])}}} \\textbf{{Observation}}: {tex_escape(h['observation'])} \\textbf{{Proposed mechanism}}: {tex_escape(h['mechanism'])} \\textbf{{Competing explanation}}: {tex_escape(h['competing'])} \\textbf{{Predicted if mechanism holds}}: {tex_escape(h['prediction_if_true'])} \\textbf{{Predicted if competing explanation holds}}: {tex_escape(h['prediction_if_competing'])} \\textbf{{Discriminating experiments}}: {tex_escape(h.get('experiments',''))}" + (f" \\textbf{{Outcome}}: {tex_escape(h['outcome'])}" if h.get('outcome') else ""))
    open(os.path.join(GEN, "hypotheses.tex"), "w").write("\n\n".join(out))


def holdout_tex():
    p = os.path.join(ROOT, "experiments", "holdouts", "holdout_evaluation.json")
    if not os.path.exists(p):
        open(os.path.join(GEN, "holdout_table.tex"), "w").write("Holdout evaluation not yet available."); return
    E = json.load(open(p))
    out = ["\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{longtable}{>{\\raggedright\\arraybackslash}p{3.6cm}rrrrrrp{3.2cm}}", "\\caption{Holdout designs: predictions recorded before simulation (file hash in text) versus observations. Strength in N/m, work in J/m$^2$; ``mode'' = predicted/observed fracture-mode class (abrupt = single avalanche; stepwise = few events; progressive = multiple events with load retained).}\\label{tab:holdout}\\\\",
           "\\toprule design & $\\sigma_\\mathrm{pred}$ & $\\sigma_\\mathrm{obs}$ & $\\varepsilon^f_\\mathrm{pred}$ & $\\varepsilon^f_\\mathrm{obs}$ & $W_\\mathrm{pred}$ & $W_\\mathrm{obs}$ & mode pred/obs\\\\ \\midrule\\endhead"]
    for r in E["rows"]:
        p_ = r["predicted"]; o = r.get("observed", {})
        nm = tex_escape(r['name']).replace("\\_", "\\_\\allowbreak{}")
        oc = str(o.get('fracture_mode') or '--').split(' ')[0]
        out.append(f"{nm} & {fmt(p_.get('strength_Nm'))} & {fmt(o.get('strength_Nm'))} & {fmt(p_.get('failure_strain'),3)} & {fmt(o.get('failure_strain'),3)} & {fmt(p_.get('work_to_failure_J_m2'))} & {fmt(o.get('work_to_failure_J_m2'))} & {tex_escape(p_.get('fracture_mode_class'))}/{tex_escape(oc)}\\\\")
    out.append("\\bottomrule\\end{longtable}\\endgroup")
    s = E["summary"]
    nice = {"modulus_2d_Nm": "2D modulus", "strength_Nm": "strength", "failure_strain": "failure strain", "work_to_failure_J_m2": "work to failure", "damage_localization": "damage localisation"}
    out.append("Prediction errors (median absolute relative error): " + ", ".join(f"{nice.get(k, tex_escape(k))}: {v['median_abs_rel_error']*100:.0f}\\% (n={v['n']})" for k, v in s.items() if isinstance(v, dict)) + (f"; fracture-mode class accuracy {s['fracture_mode_class_accuracy']*100:.0f}\\%." if s.get("fracture_mode_class_accuracy") is not None else "."))
    out.append(f" Prediction file \\path{{{E['prediction_file']}}} written {E['prediction_written']} (SHA-256 {E['prediction_sha256'][:16]}\\ldots), evaluated {E['evaluated']}.")
    open(os.path.join(GEN, "holdout_table.tex"), "w").write(_fix_longtable("\n".join(out)))


def top_tex():
    p = os.path.join(ROOT, "final_designs", "top_structures.json")
    if not os.path.exists(p):
        open(os.path.join(GEN, "top_structures.tex"), "w").write("Top-structure analysis not yet available."); return
    T = json.load(open(p))
    out = []
    for i, s in enumerate(T["selected"]):
        out.append(f"\\subsection{{{tex_escape(s['title'])}}}\n\\begin{{figure}}[H]\\centering\\includegraphics[width=\\textwidth]{{{s['panel'].replace('figures/','')}}}\\caption{{{tex_escape(s['caption'])}}}\\label{{fig:top{i}}}\\end{{figure}}\n{tex_escape(s['interpretation'])}\n")
        if s.get("before_after"):
            out.append(f"\\begin{{figure}}[H]\\centering\\includegraphics[width=0.95\\textwidth]{{{s['before_after'].replace('figures/','')}}}\\caption{{Before/after comparison for {tex_escape(s['title'])}: relaxed, peak load and final configuration (colour: energy change per atom; red circles: atoms that lost a bond).}}\\end{{figure}}\n")
    open(os.path.join(GEN, "top_structures.tex"), "w").write(_fix_longtable("\n".join(out)))


def main(compile=True):
    recs = db.all_records()
    validation_table(); numbers(recs); hypotheses_tex(); holdout_tex(); top_tex()
    comp = [r for r in recs if r.get("status") == "completed" and r.get("campaign") == "discovery"]
    comp.sort(key=lambda r: (str(r.get("stage")), r.get("name")))
    def _short(r):
        r = dict(r); m = dict(r.get("metrics") or {})
        st = str(r.get("stage") or ""); mm = re.match(r"stage(\d)", st); r["stage"] = ("S" + mm.group(1)) if mm else st[:6]
        r["viability"] = {"stable": "stable", "reconstructed but viable": "reconstr.", "strongly reconstructed": "strongly rec."}.get(r.get("viability"), str(r.get("viability"))[:12])
        fm = str(m.get("fracture_mode") or ""); m["fracture_mode"] = fm.split(" ")[0] if fm else "--"; r["metrics"] = m
        return r
    comp = [_short(r) for r in comp]
    runs_table(comp, os.path.join(GEN, "runs_table.tex"), "Complete experiment database (discovery campaign): one row per simulation. $Y$ = 2D modulus (N/m), $\\sigma$ = strength (N/m), $\\varepsilon_p$ strain at peak, $\\varepsilon_{fd}$ first-damage strain, $\\varepsilon_f$ failure strain, $W$ work to failure (J/m$^2$), $L$ damage localisation.", "tab:database",
               [("name", "name", 0), ("stage", "stage", 0), ("n_atoms", "N", 0), ("porosity", "phi", 3), ("hierarchy_levels", "H", 0), ("viability", "viability", 0), ("m.modulus_2d_Nm", "Y", 1), ("m.strength_Nm", "sigma", 2), ("m.strain_at_peak", "eps_p", 3), ("m.first_damage_strain", "eps_fd", 3), ("m.failure_strain", "eps_f", 3), ("m.work_to_failure_J_m2", "W", 2), ("m.damage_localization", "L", 2), ("m.fracture_mode", "mode", 0)])
    if compile:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-silent", "report.tex"], cwd=os.path.join(ROOT, "report"), check=False)
        print("compiled:", os.path.exists(os.path.join(ROOT, "report", "report.pdf")))


if __name__ == "__main__":
    main(compile="--no-compile" not in sys.argv)
