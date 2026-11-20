import importlib.util,sys
from pathlib import Path
run=Path(__file__).resolve().parents[1]
origin=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location("audit_rw",origin)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=run.parent/"repo"
module.OUT=run/"test3_raw_alignment"
module.run()
