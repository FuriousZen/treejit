"""C3: the compactions table grows by one row per compacted call id, forever; `treejit prune` never touches it."""
import os, sqlite3, subprocess, sys, tempfile, time
db = sys.argv[1] if len(sys.argv) > 1 else "C2_out_s0/treejit+ok+compact.db"
c = sqlite3.connect(db)
q = lambda s: c.execute(s).fetchone()
runs = q("SELECT COUNT(*) FROM runs")[0]
rows, dig, saved = q("SELECT COUNT(*), COALESCE(SUM(LENGTH(digest)),0), COALESCE(SUM(saved),0) FROM compactions")
print(f"db={db}\nruns={runs} compaction rows={rows} ({rows/runs:.2f}/run) digest bytes={dig} (~{(dig+rows*80)/rows:.0f} B/row incl. id+hash)")
# growth curve: rows by run order (ts)
ts = [r[0] for r in c.execute("SELECT ts FROM compactions ORDER BY ts")]
rts = [r[0] for r in c.execute("SELECT created FROM runs ORDER BY created")]
for k in (50, 100, 150, 200):
    if k <= len(rts):
        cut = rts[k - 1]
        print(f"  after run {k:3d}: {sum(1 for t in ts if t <= cut + 1e-6)} rows")
# rows whose call id belongs to a run that is finished (has an outcome): never read again by a live conversation
fin = q("SELECT COUNT(*) FROM compactions WHERE call_id IN (SELECT s.call_id FROM steps s JOIN runs r ON r.id=s.run_id WHERE r.outcome IS NOT NULL)")[0]
orphan = q("SELECT COUNT(*) FROM compactions WHERE call_id NOT IN (SELECT call_id FROM steps)")[0]
print(f"rows belonging to runs with an outcome: {fin}/{rows}; rows with no step row: {orphan}")
c.close()
# prune does not touch it
with tempfile.TemporaryDirectory() as tmp:
    cp = os.path.join(tmp, "x.db")
    subprocess.run(["sqlite3", db, f".backup {cp}"], check=False) if False else None
    import shutil; shutil.copy(db, cp)
    before = sqlite3.connect(cp).execute("SELECT COUNT(*) FROM compactions").fetchone()[0]
    out = subprocess.run([sys.executable, "-m", "treejit", "--db", cp, "prune", "--days", "0", "--min-hits", "1000000"],
                         capture_output=True, text=True)
    print("treejit prune --days 0 --min-hits 1e6 ->", (out.stdout + out.stderr).strip())
    after = sqlite3.connect(cp).execute("SELECT COUNT(*) FROM compactions").fetchone()[0]
    print(f"compaction rows before/after prune: {before}/{after}")
