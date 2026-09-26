"""Evaluate the prototype (S_proto_policy) against the S1/S2/S3 sets. Old = current repo policy."""
import sys, os, subprocess, tempfile
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "/home/user/treejit/src"]
import treejit.policy as P
from treejit.config import Config
CFG = Config()
old = P.is_commit_point
import S_proto_policy as NP
def row(c, want=None):
    a = {"command": c}
    o, n = int(old("Bash", a, CFG)), int(NP.is_commit_point("Bash", a, CFG))
    flag = "" if want is None else ("" if n == want else "  <-- WRONG")
    print(f"  {c[:52]!r:56} old={o} new={n} {NP.why(c, CFG)[:24]:24}{flag}")
    return n == want if want is not None else True
print("S1 opaque executors (want commit=1 except local runners):")
S1 = [("python -c 'import os; os.system(\"git push\")'", 1), ("python3 scripts/release.py", 1), ("node -e 'x()'", 1),
      ("perl -e 'system(1)'", 1), ("ruby -e 'x'", 1), ("make deploy", 1), ("npm run release", 1), ("yarn release", 1),
      ("./deploy.sh", 1), ("bash scripts/push.sh", 1), ("git p", 1), ("git -c alias.p=push p", 1),
      ("gh repo delete o/r --yes", 1), ("gh api -X POST repos/o/r/issues", 1), ("gh workflow run d.yml", 1),
      ("curl -X POST https://x", 1), ("npx vercel --prod", 1), ("uv run scripts/pub.py", 1),
      ("python -m pytest -q", 0), ("python -m pytest -q --lf", 0), ("pytest -x", 0), ("python -m unittest", 0),
      ("tox -e py311", 0), ("cargo test", 0), ("go test ./...", 0), ("npm test", 0), ("npm run test:unit", 0),
      ("npm run build", 0), ("make test", 0), ("make", 1), ("uv run pytest", 0), ("poetry run pytest -q", 0),
      ("git add x", 0), ("git commit -m 'x'", 0), ("git rm x", 0), ("git status", 0), ("mkdir -p a && touch a/b", 0),
      ("sed -i s/a/b/ f", 0), ("cp a b", 0), ("pip install -e .", 0)]
ok1 = sum(row(c, w) for c, w in S1)
print("S2 bash-parity (all push in bash => want 1):")
S2 = [r"$'\147it' push", r"$'\x67it' push", r"git $'\160ush'", '$"git" push', "{git,push}", "git {push,}",
      "git${IFS}push", "a=(git); a+=(push); \"${a[@]}\"", "x=push; git $x", "g=git; $g push", "git \"$(echo push)\"",
      "git `echo push`", "f(){ git \"$@\"; }; f push", "bash <<< 'git push'", "echo Z2l0IHB1c2g= | base64 -d | sh",
      "git send-pack origin main", r"bash -c $'\147it push'", "sh -c 'x=push; git $x'", "eval \"git \\${IFS}push\"",
      "env git${IFS}push", "git -C repo push", "xargs -n1 git push < r", "find . -exec git push \;"]
ok2 = sum(row(c, 1) for c in S2)
print("S3 false-positive table:")
sys.argv = ["x"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import S3_false_positives as S3
ok3 = sum(row(c, w) for c, w in S3.ROWS)
print(f"\ncorrect: S1 {ok1}/{len(S1)}  S2 {ok2}/{len(S2)}  S3 {ok3}/{len(S3.ROWS)}")
