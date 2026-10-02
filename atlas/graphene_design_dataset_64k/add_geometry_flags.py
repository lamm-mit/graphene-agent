"""Attach measured coordination fractions and cleanup flags to the catalog."""
import json,shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features,Value,Dataset
from pathlib import Path
ROOT=Path(__file__).resolve().parent;D=ROOT/'release/dataset'
def main():
 geom=np.load(ROOT/'assets/geometric_descriptors.npz')['geometry']
 for path in sorted((D/'data').glob('*.parquet')):
  t=pq.read_table(path);ids=np.array(t['index']);features=Features.from_arrow_schema(t.schema)
  for name,values,type in [('fraction_coordination_2',geom[ids,4],'float32'),('fraction_coordination_lt2',geom[ids,5],'float32'),('geometric_cleanup_flag',geom[ids,5]>0,'bool')]:
   if name in t.column_names:assert np.array_equal(t[name].to_numpy(),values);continue
   features[name]=Value(type);t=t.append_column(name,pa.array(values))
  t=t.cast(features.arrow_schema);partial=path.with_suffix('.parquet.partial');pq.write_table(t,partial,compression='zstd',row_group_size=64,write_page_index=True);partial.replace(path)
 ids=np.flatnonzero(geom[:,5]>0).tolist();assert len(ids)==4
 record=dict(descriptor_source='geometric_descriptors.npz geometry columns 4 and 5',flagged_ids=ids,flag_meaning='At least one retained atom has fewer than two geometric graph neighbors after bounded cleanup; not an energy/force criterion',all_structures_unrelaxed=True)
 (D/'provenance/geometric_cleanup_flags.json').write_text(json.dumps(record,indent=2))
 for path in [ROOT/'output/dataset_qa.json',D/'provenance/dataset_qa.json']:
  data=json.loads(path.read_text());data['catalog_bytes']=sum(p.stat().st_size for p in (D/'data').glob('*.parquet'));data['catalog_includes_coordination_quality_flags']=True;path.write_text(json.dumps(data,indent=2))
 sample=Dataset.from_parquet(str(D/'data/designs-0000.parquet'))[0];assert sample['image'].size==(384,384) and isinstance(sample['geometric_cleanup_flag'],bool)
 shutil.copy2(__file__,D/'code/add_geometry_flags.py');print(record)
if __name__=='__main__':main()
