import sys
from L2_patches import apply
apply(sys.argv[1])
import L2_repro as L
for tpl in ("delete {} carefully", "fix the typo in {}"):
    L.B_TEMPLATE = tpl
    print("class B template:", repr(tpl))
    for N in [0, 4, 8, 20, 40]:
        L.run(N)
