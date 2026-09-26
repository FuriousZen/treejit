"""SQLite storage (one file, WAL). The cold trace log (runs, steps, requests) is the
source of truth; the tree tables (edges, nodes, node_edges) are a materialized view
rebuilt from it. Operator state (pins, approvals, evictions) and hit counters live
in their own tables so rebuilds never lose them."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Iterable

from .util import now

SCHEMA = """
CREATE TABLE IF NOT EXISTS families(
  id TEXT PRIMARY KEY, tools_hash TEXT, prefix TEXT, dialect TEXT,
  created REAL, updated REAL, built_at REAL DEFAULT 0, dirty INTEGER DEFAULT 1);
CREATE INDEX IF NOT EXISTS families_tools ON families(tools_hash);

CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY, family TEXT, task TEXT, task_hash TEXT, created REAL, updated REAL,
  n_steps INTEGER DEFAULT 0, outcome TEXT, reason TEXT, outcome_at REAL);
CREATE INDEX IF NOT EXISTS runs_family ON runs(family);

CREATE TABLE IF NOT EXISTS steps(
  run_id TEXT, idx INTEGER, call_id TEXT, tool TEXT, args TEXT, obs TEXT, is_error INTEGER,
  replayed INTEGER DEFAULT 0, ts REAL, PRIMARY KEY(run_id, idx));

CREATE TABLE IF NOT EXISTS requests(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, family TEXT, run_id TEXT, dialect TEXT, tier TEXT,
  node TEXT, n_calls INTEGER DEFAULT 0, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
  cache_read INTEGER DEFAULT 0, cache_write INTEGER DEFAULT 0, latency_ms REAL, status INTEGER, note TEXT,
  call_ids TEXT);
CREATE INDEX IF NOT EXISTS requests_run ON requests(run_id);
CREATE INDEX IF NOT EXISTS requests_family ON requests(family);

CREATE TABLE IF NOT EXISTS edges(
  id TEXT PRIMARY KEY, family TEXT, tool TEXT, shape TEXT, template TEXT, n INTEGER, label TEXT);
CREATE INDEX IF NOT EXISTS edges_family ON edges(family);

CREATE TABLE IF NOT EXISTS nodes(
  id TEXT PRIMARY KEY, family TEXT, kind TEXT, ctx TEXT, depth INTEGER, parent TEXT, via TEXT,
  n_runs INTEGER, n_pass INTEGER, stump TEXT, last_seen REAL);
CREATE INDEX IF NOT EXISTS nodes_family ON nodes(family);

CREATE TABLE IF NOT EXISTS node_edges(
  node TEXT, edge TEXT, family TEXT, n INTEGER, pass_runs INTEGER, fail_runs INTEGER, pass_n INTEGER,
  blamed REAL, tomb INTEGER, live INTEGER, replayable INTEGER, tier TEXT, purity REAL, success REAL, conf REAL,
  bindings TEXT, holes TEXT, guard TEXT, post TEXT, ref TEXT, reasons TEXT, commit_point INTEGER,
  savings REAL, latency_ms REAL, score REAL, fillable INTEGER DEFAULT 0, PRIMARY KEY(node, edge));
CREATE INDEX IF NOT EXISTS node_edges_family ON node_edges(family);

CREATE TABLE IF NOT EXISTS pins(node TEXT, edge TEXT, ts REAL, PRIMARY KEY(node, edge));
CREATE TABLE IF NOT EXISTS approvals(edge TEXT, node TEXT, ts REAL, PRIMARY KEY(edge, node));
CREATE TABLE IF NOT EXISTS evictions(node TEXT PRIMARY KEY, ts REAL);
CREATE TABLE IF NOT EXISTS hits(node TEXT PRIMARY KEY, hits INTEGER, last_hit REAL);
"""

OBS_CAP = 64 * 1024


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(node_edges)")}
        if "fillable" not in cols:
            self.db.execute("ALTER TABLE node_edges ADD COLUMN fillable INTEGER DEFAULT 0")
            self.db.execute("UPDATE families SET dirty=1")

    def close(self) -> None:
        self.db.close()

    # -------------------------------------------------------------- basics
    def q(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, tuple(args)).fetchall()

    def q1(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Row | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def x(self, sql: str, args: Iterable[Any] = ()) -> int:
        with self.lock:
            cur = self.db.execute(sql, tuple(args))
            return cur.lastrowid or 0

    def xmany(self, sql: str, rows: list[tuple]) -> None:
        with self.lock:
            self.db.executemany(sql, rows)

    def transaction(self) -> "_Tx":
        return _Tx(self)

    # -------------------------------------------------------------- runs & steps
    def upsert_run(self, run_id: str, family: str, task: str, task_hash: str) -> None:
        t = now()
        self.x(
            "INSERT INTO runs(id, family, task, task_hash, created, updated) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET updated=excluded.updated",
            (run_id, family, task, task_hash, t, t),
        )

    def run(self, run_id: str) -> sqlite3.Row | None:
        return self.q1("SELECT * FROM runs WHERE id=?", (run_id,))

    def n_steps(self, run_id: str) -> int:
        r = self.q1("SELECT n_steps FROM runs WHERE id=?", (run_id,))
        return int(r["n_steps"]) if r else 0

    def write_steps(self, run_id: str, first_idx: int, rows: list[tuple]) -> None:
        """rows: (idx, call_id, tool, args_json, obs_text|None, is_error, replayed)"""
        t = now()
        with self.transaction():
            self.db.executemany(
                "INSERT OR REPLACE INTO steps(run_id, idx, call_id, tool, args, obs, is_error, replayed, ts) VALUES(?,?,?,?,?,?,?,?,?)",
                [(run_id, i, cid, tool, args, (obs[:OBS_CAP] if obs is not None else None), err, rep, t)
                 for (i, cid, tool, args, obs, err, rep) in rows],
            )
            n = first_idx + len(rows)
            self.db.execute("UPDATE runs SET n_steps=MAX(n_steps, ?), updated=? WHERE id=?", (n, t, run_id))

    def steps(self, run_id: str) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM steps WHERE run_id=? ORDER BY idx", (run_id,))

    def set_outcome(self, run_id: str, outcome: str, reason: str | None) -> list[str]:
        if run_id == "latest":
            r = self.q1("SELECT id FROM runs WHERE outcome IS NULL ORDER BY updated DESC LIMIT 1") or \
                self.q1("SELECT id FROM runs ORDER BY updated DESC LIMIT 1")
            if r is None:
                return []
            run_id = r["id"]
        rows = self.q("SELECT id, family FROM runs WHERE id=? OR (id LIKE ? AND outcome IS NULL)", (run_id, run_id + ".%"))
        t = now()
        for r in rows:
            self.x("UPDATE runs SET outcome=?, reason=?, outcome_at=? WHERE id=?", (outcome, reason, t, r["id"]))
            self.x("UPDATE families SET dirty=1 WHERE id=?", (r["family"],))
        return [r["id"] for r in rows]

    # -------------------------------------------------------------- requests
    def log_request(self, **kw: Any) -> int:
        kw.setdefault("ts", now())
        cols = ",".join(kw)
        return self.x(f"INSERT INTO requests({cols}) VALUES({','.join('?' * len(kw))})", kw.values())

    def update_request(self, rid: int, **kw: Any) -> None:
        if not kw:
            return
        sets = ",".join(f"{k}=?" for k in kw)
        self.x(f"UPDATE requests SET {sets} WHERE id=?", [*kw.values(), rid])

    # -------------------------------------------------------------- operator state
    def pins(self) -> set[tuple[str, str]]:
        return {(r["node"], r["edge"]) for r in self.q("SELECT node, edge FROM pins")}

    def approvals(self) -> set[tuple[str, str]]:
        return {(r["edge"], r["node"]) for r in self.q("SELECT edge, node FROM approvals")}

    def evictions(self) -> dict[str, float]:
        return {r["node"]: r["ts"] for r in self.q("SELECT node, ts FROM evictions")}

    def hit(self, node: str) -> None:
        self.x("INSERT INTO hits(node, hits, last_hit) VALUES(?,1,?) ON CONFLICT(node) DO UPDATE SET hits=hits+1, last_hit=excluded.last_hit",
               (node, now()))


class _Tx:
    def __init__(self, store: Store) -> None:
        self.store = store

    def __enter__(self) -> "_Tx":
        self.store.lock.acquire()
        self.store.db.execute("BEGIN")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.store.db.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.store.lock.release()


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
