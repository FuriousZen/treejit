"""S3: realistic commands vs commit-point detection. expected: 1 = should be a commit point, 0 = local/reversible."""
import sys
sys.path.insert(0, "/home/user/treejit/src")
from treejit.config import Config
from treejit.policy import is_commit_point, is_readonly
CFG = Config()
ROWS = [  # (command, expected commit)
    ("git stash push", 0), ("git stash push -m wip src/x.py", 0), ("git checkout push", 0), ("git branch -D push", 0),
    ("git fetch origin push", 0), ("git log origin/push", 0), ("git worktree add ../push", 0), ("git rebase origin/push", 0),
    ("git push --dry-run", 0), ("git push -n origin main", 0),
    ("apt install curl", 0), ("apt-get install -y curl wget", 0), ("pip install requests curl", 0), ("brew install rsync", 0),
    ("npm install ssh2 && npm run build", 0), ("man ssh", 0), ("mkdir ssh", 0), ("rm -rf wget", 0), ("chmod 700 ssh", 0),
    ("echo push > f", 0), ("echo 'git push' > NOTES", 0), ("npm run push-docs", 0), ("sed -i 's/curl/wget/' Dockerfile", 0),
    ("curl -s localhost:8000/health", 0), ("curl -sSf http://127.0.0.1:3000/ -o /dev/null", 0), ("wget -qO- http://localhost:8080", 0),
    ("rsync -a src/ build/", 0), ("gh pr view 12", 0), ("gh pr list", 0), ("gh pr diff 12", 0), ("gh pr checks 12", 0),
    ("gh release list", 0), ("gh release view v1", 0), ("kubectl apply --dry-run=client -f x.yaml", 0),
    ("npm publish --dry-run", 0), ("cargo publish --dry-run", 0), ("docker compose push", 1),
    ("git push origin main", 1), ("docker push img:1", 1), ("gh pr create -f", 1), ("gh pr merge 3", 1), ("curl -X POST https://api.x/y", 1),
    ("ssh prod 'systemctl restart app'", 1), ("scp dist.tgz prod:/srv", 1), ("kubectl delete pod x", 1),
    ("python -m pytest -q", 0), ("pytest -q", 0), ("npm test", 0), ("cargo test", 0), ("go test ./...", 0), ("make test", 0),
]
fp = fn = 0
print(f"{'command':48} {'ro':2} {'commit':6} {'want':4} verdict")
for c, want in ROWS:
    a = {"command": c}
    ro, cp = int(is_readonly("Bash", a, CFG)), int(is_commit_point("Bash", a, CFG))
    v = "ok" if cp == want else ("FALSE POSITIVE" if cp else "FALSE NEGATIVE")
    fp += v == "FALSE POSITIVE"; fn += v == "FALSE NEGATIVE"
    print(f"{c:48} {ro:2} {cp:6} {want:4} {v}")
print(f"\n{len(ROWS)} commands: {fp} false positives, {fn} false negatives")
