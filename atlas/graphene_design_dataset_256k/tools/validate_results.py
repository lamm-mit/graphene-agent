"""Validate a future mechanical-results JSONL against the geometry registry.
Usage: python validate_results.py results.jsonl /local/dataset/root
Does not launch runs, overwrite records or manufacture property values.
"""
import argparse,json
from pathlib import Path
import pyarrow.parquet as pq
import jsonschema

def main():
 p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('dataset',type=Path);args=p.parse_args()
 schema=json.loads((args.dataset/'mechanics/results.schema.json').read_text());validator=jsonschema.Draft202012Validator(schema)
 registry={r['design_id']:r for r in pq.read_table(args.dataset/'mechanics/design_inventory.parquet',columns=['design_id','coordinates_sha256']).to_pylist()};seen=set();n=0
 for line in args.results.read_text().splitlines():
  if not line.strip():continue
  r=json.loads(line);validator.validate(r)
  if r['run_id'] in seen:raise ValueError('Duplicate run_id: '+r['run_id'])
  seen.add(r['run_id']);expected=registry.get(r['design_id'])
  if not expected or expected['coordinates_sha256']!=r['coordinates_sha256']:raise ValueError('Unknown or mismatched initial geometry: '+r['design_id'])
  n+=1
 print(f'Validated {n} run records; no files changed')
if __name__=='__main__':main()
