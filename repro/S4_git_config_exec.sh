#!/bin/bash
# S4: "read-only" git commands executing repo-controlled programs. Each check: does treejit call it read-only,
# and does running it create a marker file?
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d); M=$W/markers; mkdir -p $M
export GIT_CONFIG_NOSYSTEM=1 HOME=$W  # isolate from the host's config
git config --global user.email a@b; git config --global user.name a; git config --global init.defaultBranch main
R=$W/repo; git init -q $R; cd $R
echo a > f.txt; echo 'secret' > s.dat; echo '*.dat diff=evil filter=evil' > .gitattributes
git add .; git commit -qm init; echo b >> f.txt; echo more >> s.dat
ro() { python3 -c "import sys; sys.path.insert(0,'/home/user/treejit/src'); from treejit.policy import is_readonly; from treejit.config import Config; print(int(is_readonly('Bash', {'command': sys.argv[1]}, Config())))" "$1"; }
check() {  # name, config key, value, command
  rm -f $M/*; (git config "$2" "$3"; eval "$4" </dev/null >/dev/null 2>&1; git config --unset "$2")
  printf '%-44s %-26s ro=%s executed=%s\n' "$4" "$2" "$(ro "$4")" "$(ls $M | tr '\n' ' ' | sed 's/ $//;s/^$/no/')"
}
check fsmonitor core.fsmonitor "touch $M/fsmonitor; false" "git status"
check fsmonitor2 core.fsmonitor "touch $M/fsmonitor" "git status --short"
check fsmonitor3 core.fsmonitor "touch $M/fsmonitor" "git diff"
check fsmonitor4 core.fsmonitor "touch $M/fsmonitor" "git ls-files -m"
check extdiff diff.external "sh -c 'touch $M/diff.external'" "git diff"
check extdiff2 diff.external "sh -c 'touch $M/diff.external'" "git log -p -1"
check textconv diff.evil.textconv "sh -c 'touch $M/textconv; cat \$0'" "git diff"
check textconv2 diff.evil.textconv "sh -c 'touch $M/textconv; cat \$0'" "git show HEAD"
check textconv3 diff.evil.textconv "sh -c 'touch $M/textconv; cat \$0'" "git blame s.dat"
check clean filter.evil.clean "sh -c 'touch $M/filter.clean; cat'" "git status"
check clean2 filter.evil.clean "sh -c 'touch $M/filter.clean; cat'" "git diff --stat"
check pager core.pager "touch $M/pager; cat" "git -p log -1"
check pager2 core.pager "touch $M/pager; cat" "git --paginate status"
check pager3 pager.log "touch $M/pager.log; cat" "git -p log -1"
printf "#!/bin/sh\ntouch $M/gpg.program\nexit 1\n" > $W/gpg.sh; chmod +x $W/gpg.sh
check gpg gpg.program "$W/gpg.sh" "git log --show-signature -1"
git config gpg.program $W/gpg.sh; git -c gpg.format=openpgp commit -qm signed-ish --allow-empty 2>/dev/null
check gpg2 log.showSignature true "git log -1"
# embedded bare repo committed into the tree (transported by clone, no .git/config access needed)
rm -f $M/*; B=$R/vendor/pkg; mkdir -p $B; git init -q --bare $B; git -C $B config core.fsmonitor "touch $M/bare-fsmonitor; false"
git -C $B config core.bare false; git -C $B config core.worktree ..
(cd $B && git status >/dev/null 2>&1); printf '%-44s %-26s ro=%s executed=%s\n' "cd vendor/pkg && git status" "(embedded bare repo)" "$(ro 'cd vendor/pkg && git status')" "$(ls $M | tr '\n' ' ' | sed 's/^$/no/')"
rm -f $M/*; (cd $B && git -c safe.bareRepository=explicit status >/dev/null 2>&1); printf '%-44s %-26s executed=%s\n' "  same, with -c safe.bareRepository=explicit" "" "$(ls $M | sed 's/^$/no/')"
# proposed mitigation: hardening flags prepended by treejit on replay
echo "--- with hardening: git -c core.fsmonitor=false -c core.pager=cat -c diff.external= -c log.showSignature=false --no-pager ..."
H="-c core.fsmonitor=false -c core.pager=cat -c diff.external= -c log.showSignature=false -c gpg.program=false --no-pager"
check h1 core.fsmonitor "touch $M/fsmonitor" "git $H status"
check h2 diff.external "sh -c 'touch $M/diff.external'" "git $H diff"
check h3 diff.evil.textconv "sh -c 'touch $M/textconv; cat \$0'" "git $H diff --no-textconv"
check h4 filter.evil.clean "sh -c 'touch $M/filter.clean; cat'" "git $H diff --stat"
check h4b diff.evil.textconv "sh -c 'touch $M/textconv; cat \$0'" "git $H show HEAD~1"
echo "--- env-based: GIT_CONFIG_GLOBAL=/dev/null does not disable repo config; GIT_CONFIG_COUNT injection:"
check e1 core.fsmonitor "touch $M/fsmonitor" "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.fsmonitor GIT_CONFIG_VALUE_0=false git status"
