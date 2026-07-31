"""
store.py — the append-only event log (Layer 5).

The store is the single source of truth. In production this is a Postgres
`events` table (BIGSERIAL seq, run_id, type, node_id, payload JSONB, created_at);
here it is SQLite so the PoC runs with zero external services. The interface is
an adapter (EventStore) so swapping SQLite -> Postgres is a one-class change and
nothing above this layer knows the difference.

Two responsibilities:
  1. Assign the monotonic, gap-free `seq` that defines the replay order.
  2. Enforce provenance-at-write: a NODE_CREATED / EDGE_CREATED event is REJECTED
     unless its payload carries a valid Provenance tuple. This is where NULL
     provenance becomes structurally impossible.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from typing import Iterator

from .core import Event, EventType, Provenance, ProvenanceError


_CREATION_EVENTS = {EventType.NODE_CREATED, EventType.EDGE_CREATED}


def _assert_provenance(event: Event) -> None:
    if event.type not in _CREATION_EVENTS:
        return
    prov = event.payload.get("provenance")
    if prov is None:
        raise ProvenanceError(f"{event.type.value} for {event.node_id} has no provenance")
    # Rebuild + validate; accepts either a Provenance or its dict form.
    if isinstance(prov, Provenance):
        prov.validate()
    else:
        Provenance(
            run_id=prov.get("run_id", ""),
            batch_id=prov.get("batch_id", ""),
            generation=prov.get("generation"),
            created_at=prov.get("created_at", ""),
            expander_id=prov.get("expander_id"),
            parent_ids=tuple(prov.get("parent_ids", ())),
        ).validate()


class EventStore(ABC):
    @abstractmethod
    def append(self, event: Event) -> Event: ...
    @abstractmethod
    def read(self, run_id: str | None = None) -> Iterator[Event]: ...
    @abstractmethod
    def count(self, run_id: str | None = None) -> int: ...


class SQLiteEventStore(EventStore):
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                seq        INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     TEXT NOT NULL,
                type       TEXT NOT NULL,
                node_id    TEXT,
                payload    TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._db.commit()

    def append(self, event: Event) -> Event:
        _assert_provenance(event)  # <-- the guard. Rejects lineage-less creation.
        cur = self._db.execute(
            "INSERT INTO events (run_id, type, node_id, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                event.run_id,
                event.type.value,
                event.node_id,
                json.dumps(event.payload, default=_json_default),
                event.created_at,
            ),
        )
        self._db.commit()
        event.seq = cur.lastrowid
        return event

    def read(self, run_id: str | None = None) -> Iterator[Event]:
        if run_id is None:
            rows = self._db.execute("SELECT * FROM events ORDER BY seq ASC")
        else:
            rows = self._db.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY seq ASC", (run_id,)
            )
        for seq, rid, typ, node_id, payload, created_at in rows:
            yield Event(
                run_id=rid,
                type=EventType(typ),
                payload=json.loads(payload),
                node_id=node_id,
                created_at=created_at,
                seq=seq,
            )

    def count(self, run_id: str | None = None) -> int:
        if run_id is None:
            return self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return self._db.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()[0]


def _json_default(obj):
    if isinstance(obj, Provenance):
        return obj.as_dict()
    if hasattr(obj, "value"):   # Enums
        return obj.value
    return str(obj)
