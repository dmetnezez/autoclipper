"""Tiny SQLite store for jobs and clips."""
import json
import sqlite3
import threading
import time
import uuid

from .config import DB_PATH

_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            created REAL,
            source TEXT,
            title TEXT,
            status TEXT,          -- queued | running | done | error
            stage TEXT,
            progress REAL,
            error TEXT,
            settings TEXT,
            duration REAL,
            ai_used INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY,
            job_id TEXT,
            idx INTEGER,
            start REAL,
            end REAL,
            title TEXT,
            hook TEXT,
            score INTEGER,
            reason TEXT,
            description TEXT,
            file TEXT,
            thumb TEXT
        );
        """)
        # Jobs that were mid-run when the app closed can't resume
        c.execute("UPDATE jobs SET status='error', error='App was closed while this was processing. Delete it and try again.' "
                  "WHERE status IN ('running','queued')")


def create_job(source: str, settings: dict) -> str:
    jid = uuid.uuid4().hex[:10]
    with _lock, _conn() as c:
        c.execute("INSERT INTO jobs (id, created, source, title, status, stage, progress, settings) "
                  "VALUES (?,?,?,?,?,?,?,?)",
                  (jid, time.time(), source, source, "queued", "Waiting in line", 0, json.dumps(settings)))
    return jid


def update_job(jid: str, **fields):
    if not fields:
        return
    keys = ", ".join(f"{k}=?" for k in fields)
    with _lock, _conn() as c:
        c.execute(f"UPDATE jobs SET {keys} WHERE id=?", (*fields.values(), jid))


def add_clip(job_id: str, **f):
    cid = uuid.uuid4().hex[:10]
    with _lock, _conn() as c:
        c.execute("INSERT INTO clips (id, job_id, idx, start, end, title, hook, score, reason, description, file, thumb) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (cid, job_id, f["idx"], f["start"], f["end"], f["title"], f["hook"], f["score"],
                   f["reason"], f["description"], f["file"], f["thumb"]))
    return cid


def get_job(jid: str):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row:
            return None
        job = dict(row)
        job["settings"] = json.loads(job["settings"] or "{}")
        job["clips"] = [dict(r) for r in c.execute(
            "SELECT * FROM clips WHERE job_id=? ORDER BY score DESC, idx", (jid,))]
    return job


def list_jobs():
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT j.*, (SELECT COUNT(*) FROM clips WHERE job_id=j.id) AS clip_count "
            "FROM jobs j ORDER BY created DESC").fetchall()
    return [dict(r) for r in rows]


def delete_job(jid: str):
    with _lock, _conn() as c:
        c.execute("DELETE FROM clips WHERE job_id=?", (jid,))
        c.execute("DELETE FROM jobs WHERE id=?", (jid,))
