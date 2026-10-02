"""Execute explicitly recorded conditional numerical checks, separate from primary jobs."""
import json,sys
from study_common import ROOT
from run_case import run
jobs=json.loads((ROOT/'protocol/quality_checks.json').read_text())
run(next(j for j in jobs if j['case_id']==sys.argv[1]))
