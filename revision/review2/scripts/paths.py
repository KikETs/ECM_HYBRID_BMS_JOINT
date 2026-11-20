"""Locations used by the revision scripts.
REPO  repository root
R1    revision/review1 (stored outputs of the 2026-11-16 analyses)
WORK  writable work directory: baseline/arms from `nested/reproduce.sh score` (RCSOP_WORK=$WORK/baseline) and out/ for results.
      Default revision/review2; override with RCSOP_REVISION_WORK."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REV2 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(REV2))
R1 = os.path.join(REPO, 'revision', 'review1')
WORK = os.environ.get('RCSOP_REVISION_WORK', REV2)
