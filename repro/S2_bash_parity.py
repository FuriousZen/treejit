"""S2: run each command in real bash with a stub `git` on PATH and compare with treejit's view."""
import os, subprocess, sys, tempfile
sys.path.insert(0, "/home/user/treejit/src")
from treejit.config import Config
from treejit.policy import is_commit_point, is_readonly
from treejit.shellwords import command_heads, tokenize
CFG = Config()
stub = tempfile.mkdtemp()
log = os.path.join(stub, "log")
for prog in ["git", "rm", "touch"]:
    p = os.path.join(stub, prog)
    open(p, "w").write(f'#!/bin/sh\necho "{prog} $*" >> {log}\n')
    os.chmod(p, 0o755)
CMDS = [
    "git push", r"$'\147it' push", r"$'\x67it' push", r"git $'\160ush'", r"git $'\x70ush'", '$"git" push', 'git $"push"',
    "gi\\\nt push", "git \\\npush", "{git,push}", "git {push,}", "g\"\"it push", "gi\\t push", '"git" push', "'git' push",
    "command git push", "exec -a name git push", "env -S 'git push'", "git${IFS}push", "git$IFS'push'", "git\tpush",
    "a=(git push); \"${a[@]}\"", "a=(git); a+=(push); \"${a[@]}\"", "x=push; git $x", "p=pu; git ${p}sh",
    "g=git; $g push", "git \"$(echo push)\"", "git $(printf push)", "git `echo push`", "f(){ git \"$@\"; }; f push",
    "bash <<< 'git push'", "sh<<<'git push'", "echo Z2l0IHB1c2g= | base64 -d | sh", "printf 'git push' | bash",
    "git -c alias.p=push p", "git send-pack origin main", "git --git-dir=.git push", "bash -c $'\\147it push'",
    "bash -c \"\\$'\\\\147it' push\"", "env git${IFS}push", "sh -c 'x=push; git $x'", "eval \"git \\${IFS}push\"",
    "\\git push", "git pus\\h", "git 'pu'sh",
]
print(f"{'command':40} {'bash runs':16} {'ro':2} {'commit':6} heads  EXPLOIT")
for c in CMDS:
    if os.path.exists(log): os.remove(log)
    subprocess.run(["bash", "-c", c], env={**os.environ, "PATH": stub + ":" + os.environ["PATH"]}, cwd=stub,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    ran = open(log).read().strip().replace("\n", "|") if os.path.exists(log) else "-"
    a = {"command": c}
    ro, cp = is_readonly("Bash", a, CFG), is_commit_point("Bash", a, CFG)
    pushed = "git push" in ran or "send-pack" in ran
    tag = ("RO-BYPASS " if ro and ran != "-" else "") + ("COMMIT-MISS" if pushed and not cp else "")
    print(f"{c!r:40} {ran[:16]:16} {int(ro):2} {int(cp):6} {command_heads(c)} {tag}")
