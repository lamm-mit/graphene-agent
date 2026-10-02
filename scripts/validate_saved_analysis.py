"""Regenerate loading comparisons and verify them against the saved research values."""
from pathlib import Path
import hashlib, json, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
def canonical_hash(path):
    data=json.loads(path.read_text())
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    subprocess.run([sys.executable,str(ROOT/'phase2_failure_extension/candidate_figures/scripts/build_data.py')],check=True,cwd=ROOT)
    expected=json.loads((ROOT/'release/expected_analysis_hashes.json').read_text())
    for rel,h in expected.items():
        if canonical_hash(ROOT/rel)!=h:raise AssertionError(f'Regenerated numerical values differ: {rel}')
    print(f'PASS: {len(expected)} regenerated numerical tables match the research results (JSON key order ignored).')
if __name__=='__main__':main()
