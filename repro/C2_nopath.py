"""C2: run the bench with compaction rule 3b ('path' dependencies) dropped.
usage: python3 C2_nopath.py --tasks 200 --modes treejit+ok+compact --out DIR [--seed N]"""
import sys
from treejit import compaction
_orig = compaction.depended_on
def no_path(view, cfg, eids, replayed):
    current, _path = _orig(view, cfg, eids, replayed)
    return current, set()
compaction.depended_on = no_path
from treejit_bench.__main__ import main
main(sys.argv[1:])
