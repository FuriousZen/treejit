"""T1b: the wrapped client's `.messages` namespace only has create(): messages.stream(), count_tokens, batches vanish."""
import tempfile
from treejit import TreeJIT
class M:
    def create(self, **k): return {}
    def stream(self, **k): return "stream-manager"
    def count_tokens(self, **k): return 1
    batches = "batches"
class C:
    messages = M()
with tempfile.TemporaryDirectory() as t:
    w = TreeJIT(t + "/x.db").wrap(C())
    for attr in ("create", "stream", "count_tokens", "batches"):
        try:
            getattr(w.messages, attr); print(f"messages.{attr}: ok")
        except AttributeError as e:
            print(f"messages.{attr}: AttributeError {e}")
