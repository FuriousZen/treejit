"""R1: /v1/responses is not a model route (plain passthrough); Codex-style argv shell args are never read-only."""
from treejit import proxy, policy, dialects
from treejit.config import Config
from treejit.model import ToolCall
print("model routes:", proxy.ROUTES)
print("'/v1/responses' in ROUTES:", "/v1/responses" in proxy.ROUTES, "| dialects:", list(dialects.DIALECTS))
cfg = Config()
import inspect
fn = policy.is_readonly
print("is_readonly signature:", inspect.signature(fn))
for args in ({"command": "git status"}, {"command": ["bash", "-lc", "git status"]}):
    try:
        print(f"  shell {args} -> read-only={fn('shell', args, cfg)}")
    except Exception as e:
        print(f"  shell {args} -> {type(e).__name__}: {e}")
