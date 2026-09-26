"""S2: fuzz the read-only side. Generate commands from fragments, keep those is_readonly accepts, run them in
bash inside an empty dir with logging stubs for side-effecting programs; report any write or stub run."""
import itertools, os, random, subprocess, sys, tempfile, shutil
sys.path.insert(0, "/home/user/treejit/src")
from treejit.config import Config
from treejit.policy import is_readonly
CFG = Config()
stub = tempfile.mkdtemp(); log = os.path.join(stub, "log")
for prog in ["rm", "touch", "sh", "bash", "python", "evil", "x", "id"]:
    p = os.path.join(stub, prog); open(p, "w").write(f'#!/bin/dash\necho "{prog} $*" >> {log}\n'); os.chmod(p, 0o755)
FR = ["ls", "cat f", "echo", "#", "\\", "\n", "'", '"', "$'", "\\n", ";", "&", "|", ">", "<", "<<", "<<-", "EOF", "\t",
      "touch x", "evil", "(", ")", "{", "}", "$", "`", "=", "x=1", "-", "2>&1", "/dev/null", ">&", "<<<", "!", "#x",
      "sed -n p", "awk 1", "find .", "git status", "git log", "rm y", "*", "[", "]", "~", "%", ",", "..", "\\\n", "\"'\"",
      "$(", "${", "}", " ", " ", " ", " "]
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
seen, hits, tested = set(), [], 0
for _ in range(N):
    c = "".join(random.choice(FR) + random.choice(["", " "]) for _ in range(random.randint(2, 7)))
    if c in seen: continue
    seen.add(c)
    try:
        if not is_readonly("Bash", {"command": c}, CFG): continue
    except Exception as e:
        continue
    tested += 1
    cwd = tempfile.mkdtemp(); open(os.path.join(cwd, "f"), "w").write("a\n")
    if os.path.exists(log): os.remove(log)
    try:
        subprocess.run(["/bin/bash", "-c", c], env={"PATH": stub + ":/usr/bin:/bin", "HOME": cwd}, cwd=cwd, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
    except subprocess.TimeoutExpired:
        pass
    files = sorted(set(os.listdir(cwd)) - {"f"})
    ran = open(log).read().strip() if os.path.exists(log) else ""
    if files or ran:
        hits.append((c, files, ran))
    shutil.rmtree(cwd, ignore_errors=True)
print(f"generated={len(seen)} readonly-accepted={tested} bypasses={len(hits)}")
for h in hits[:30]: print(repr(h))
