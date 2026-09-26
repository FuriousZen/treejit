"""S1: opaque executors (interpreters, scripts, task runners, aliases) are not commit points."""
import sys, tempfile, os
sys.path[:0] = ["/home/user/treejit/src", "/home/user/treejit/tests"]
from treejit.config import Config
from treejit.policy import is_commit_point, is_readonly
from treejit.shellwords import command_heads
CFG = Config()
CMDS = [
    "python -c 'import os; os.system(\"git push\")'", "python3 scripts/release.py", "python -c 'import subprocess; subprocess.run([\"git\",\"push\"])'",
    "node -e 'require(\"child_process\").execSync(\"git push\")'", "perl -e 'system(\"git push\")'",
    "ruby -e 'system(\"git push\")'", "make deploy", "make release", "npm run release", "npm run deploy",
    "yarn release", "pnpm run publish:all", "./deploy.sh", "bash scripts/push.sh", "sh deploy.sh",
    "git p", "git pub origin main", "gh api -X POST repos/o/r/issues", "gh repo delete o/r --yes",
    "gh workflow run deploy.yml", "gh issue create -t x", "gh pr merge 1", "curl -X POST https://x",
    "http POST https://x", "npx vercel --prod", "fly deploy", "aws s3 sync . s3://b", "gcloud app deploy",
    "heroku releases:rollback", "git push", "python -m pytest -q", "python -m twine upload dist/*",
]
print(f"{'command':62} {'ro':3} {'commit':6} heads")
for c in CMDS:
    a = {"command": c}
    print(f"{c[:62]:62} {int(is_readonly('Bash', a, CFG)):3} {int(is_commit_point('Bash', a, CFG)):6} {command_heads(c)}")

# end to end: approve '*' + 2 passing runs => ./deploy.sh replays unattended (git push would need promote_runs+1 AND is a commit)
from conftest import Model, calls_of, replayed_ids
from test_engine import fs_exec, train
from treejit import TreeJIT
from treejit.operate import approve
for cmd in ["./deploy.sh", "python -c 'import os; os.system(\"git push origin main\")'", "git push origin main"]:
    d = tempfile.mkdtemp()
    jit = TreeJIT(os.path.join(d, "t.db"))
    def policy(task, hist, body, cmd=cmd):
        plan = [("Bash", {"command": "git status --short"}), ("Read", {"file_path": "src/" + task.split()[-1] + ".py"}),
                ("Bash", {"command": cmd})]
        return plan[len(hist)] if len(hist) < len(plan) else None
    files = {f"src/m{i}.py": "x\n" for i in range(10)}
    model, ex = Model(policy), fs_exec(files)
    approve(jit.store, "*", "")
    train(jit, model, [f"ship m{i}" for i in range(2)], ex)
    res = []
    for k in range(2, 5):
        before = model.calls
        [m] = train(jit, model, [f"ship m{k}"], ex, run_prefix=f"r{k}")
        rep = replayed_ids(m)
        ids = [b["id"] for msg in m if msg["role"] == "assistant" for b in msg["content"]
               if b["type"] == "tool_use" and b["input"].get("command") == cmd]
        res.append((k + 1, any(i in rep for i in ids), model.calls - before))
    row = jit.store.q("SELECT ne.commit_point, ne.blocked, ne.tier FROM node_edges ne JOIN edges e ON e.id=ne.edge "
                      "WHERE e.tool='Bash' AND e.template NOT LIKE '%status%' AND e.tool='Bash'")
    print(f"[e2e approve '*'] {cmd!r}: (run#, target replayed, model calls) = {res}; edge rows={[dict(r) for r in row]}")
    jit.close()
