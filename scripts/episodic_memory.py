"""
Episodic Memory: Hybrid Vector + Graph memory for Hermes Agent.
Pure Python fallback: SQLite graph + in-memory vector (no external deps).
Production: swap in Neo4j + FAISS when available.
"""

import json
import logging
import sqlite3
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import faiss

from config import load_config

logger = logging.getLogger("episodic_memory")


# ---------------------------------------------------------------------------
# Embedding (swap with sentence-transformers in production)
# ---------------------------------------------------------------------------
def _simple_embed(text: str, dim: int = 384, seed: int = 42) -> np.ndarray:
    """Deterministic random-projection embedding (stable, non-zero, no external deps)."""
    import hashlib
    rng = np.random.RandomState(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) ^ seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        vec[0] = 1e-6
        norm = 1e-6
    vec = vec / norm
    return vec


def _get_embedder():
    """
    Production: sentence-transformers with offline-first loading.
    Falls back to _simple_embed if model download or any init step fails.
    """
    try:
        import os
        # Force offline mode to avoid any network calls during init
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        # Try to load from cache only
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        def embed(texts):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return model.encode(texts, convert_to_numpy=True)
        return embed, 384
    except Exception:
        # Fallback: deterministic random projection, no network needed
        def embed(texts):
            return np.stack([_simple_embed(t, 384) for t in texts])
        return embed, 384


# -----------------------------------------------------------------------
# Vector store — FAISS-backed
# -----------------------------------------------------------------------
class VectorStore:
    """
    FAISS-based L2-distance vector store with persistent ID mapping.

    FAISS uses integer indices internally; we maintain a parallel `ids` list
    so the external string IDs are never lost.
    """

    def __init__(self, dim: int, index_path: Optional[Path] = None):
        self.dim = dim
        self.index_path = index_path
        # Index: either restored from disk or created fresh
        if index_path and index_path.exists():
            self._index = faiss.read_index(str(index_path))
            # Load parallel id list
            ids_path = index_path.with_suffix(".ids")
            if ids_path.exists():
                with open(ids_path) as f:
                    self.ids = json.load(f)
            else:
                self.ids = []
                logger.warning("FAISS index loaded but no .ids file found")
        else:
            self._index = faiss.IndexFlatL2(dim)
            self.ids = []

    def add(self, id: str, vec: np.ndarray):
        """Add a vector. FAISS requires float32 contiguous array."""
        v = vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(v)           # L2-normalize like the old code did
        self._index.add(v)
        self.ids.append(id)

    def search(self, query: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        """Return (indices, distances) for k nearest neighbors."""
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        k = min(k, self._index.ntotal)
        if k == 0:
            return [], []
        dists, idxs = self._index.search(q, k)
        # FAISS returns float indices; -1 means "no result" (out-of-bounds)
        # Filter out -1 entries and convert to int
        result_idxs, result_dists = [], []
        for d, i in zip(dists[0], idxs[0]):
            if int(i) >= 0:
                result_idxs.append(int(i))
                result_dists.append(float(d))
        return result_idxs, result_dists

    def save(self, path: Path):
        """Persist index to disk plus a parallel .ids file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        ids_path = Path(str(path) + ".ids")
        with open(ids_path, "w") as f:
            json.dump(self.ids, f)

    def load(self, path: Path):
        if not path.exists():
            return
        self._index = faiss.read_index(str(path))
        ids_path = Path(str(path) + ".ids")
        if ids_path.exists():
            with open(ids_path) as f:
                self.ids = json.load(f)
        else:
            self.ids = []
            logger.warning("FAISS index loaded but no .ids file found — IDs will be empty")


# ---------------------------------------------------------------------------
# Graph store (SQLite fallback for Neo4j)
# ---------------------------------------------------------------------------
class GraphStore:
    """SQLite-backed graph store. Swap with Neo4j for production."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                title TEXT,
                summary TEXT,
                created_at TEXT,
                tags TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                src TEXT,
                rel TEXT,
                dst TEXT,
                PRIMARY KEY (src, rel, dst)
            );
            CREATE INDEX IF NOT EXISTS idx_ep_created ON episodes(created_at);
            CREATE INDEX IF NOT EXISTS idx_edge_src ON edges(src);
        """)
        self.conn.commit()

    def upsert_episode(self, id: str, title: str, summary: str, created_at: str, tags: list[str]):
        self.conn.execute("""
            INSERT OR REPLACE INTO episodes (id, title, summary, created_at, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (id, title, summary, created_at, json.dumps(tags)))
        self.conn.commit()

    def link(self, src: str, rel: str, dst: str):
        self.conn.execute("""
            INSERT OR IGNORE INTO edges (src, rel, dst) VALUES (?, ?, ?)
        """, (src, rel, dst))
        self.conn.commit()

    def get_neighbors(self, id: str, depth: int = 2) -> list[str]:
        """Get neighboring episode IDs up to depth hops."""
        results = {id}
        current = {id}
        for _ in range(depth):
            if not current:
                break
            ids = list(current)
            placeholders = ",".join("?" * len(ids))
            dst_rows = self.conn.execute(f"""
                SELECT dst FROM edges WHERE src IN ({placeholders})
            """, ids).fetchall()
            src_rows = self.conn.execute(f"""
                SELECT src FROM edges WHERE dst IN ({placeholders}) AND rel != 'NEXT'
            """, ids).fetchall()
            current = {r[0] for r in dst_rows + src_rows if r[0] != id}
            results.update(current)
        return list(results - {id})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Episode:
    id: str
    title: str
    summary: str
    messages: list[Message]
    embedding: np.ndarray = field(default=None, repr=False)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["embedding"] = self.embedding.tolist() if self.embedding is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        d = d.copy()
        if d.get("embedding"):
            d["embedding"] = np.array(d["embedding"], dtype=np.float32)
        d["messages"] = [Message(**m) for m in d["messages"]]
        return cls(**d)


# ---------------------------------------------------------------------------
# Main EpisodicMemory class
# ---------------------------------------------------------------------------
class EpisodicMemory:
    """
    Hybrid episodic memory with pluggable stores.
    Default: SQLite graph + in-memory vectors.
    Production: Neo4j + FAISS.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        base = Path(self.config["storage"]["base"])
        base.mkdir(parents=True, exist_ok=True)
        self.db_path = base / "graph.db"
        self.index_path = base / "vectors.npy"
        self.state_path = base / "state.json"

        self._graph = GraphStore(self.db_path)
        self._vectors = VectorStore(self.config["vector_store"]["dim"], index_path=self.index_path)
        self._embed_fn, self._dim = _get_embedder()

        self._episode_map: dict[str, Episode] = {}
        self._current_episode: Optional[Episode] = None
        self._pending_messages: list[Message] = []

        # Neo4j sync health tracking
        self._neo4j_consecutive_failures: int = 0
        self._neo4j_last_failure_reason: Optional[str] = None

        self._load_state()

    # --- Neo4j swap-in (only used if available) ---
    def _try_neo4j_sync(self, ep: Episode, prev_id: Optional[str]):
        """If neo4j is available, sync episode to Neo4j as well."""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            return
        cfg = self.config["neo4j"]
        try:
            driver = GraphDatabase.driver(
                cfg["uri"], auth=(cfg["user"], cfg["password"]), encrypted=False
            )
            with driver.session() as sess:
                sess.run("""
                    MERGE (e:Episode {id: $id})
                    SET e.title=$title, e.summary=$summary, e.created_at=$created_at
                """, id=ep.id, title=ep.title, summary=ep.summary, created_at=ep.created_at)
                if prev_id:
                    sess.run("""
                        MATCH (prev:Episode {id: $pid}), (curr:Episode {id: $cid})
                        MERGE (prev)-[:NEXT]->(curr)
                    """, pid=prev_id, cid=ep.id)
            driver.close()
            # Reset failure counter on success
            if self._neo4j_consecutive_failures > 0:
                logger.info("Neo4j sync recovered after %d consecutive failures", self._neo4j_consecutive_failures)
                self._neo4j_consecutive_failures = 0
                self._neo4j_last_failure_reason = None
        except Exception as exc:
            # Track consecutive failures for progressive alerting.
            self._neo4j_consecutive_failures += 1
            self._neo4j_last_failure_reason = str(exc)[:120]
            logger.error(
                "Neo4j sync failed for episode %s (consecutive_failures=%d): %s",
                ep.id, self._neo4j_consecutive_failures, exc,
            )
            if self._neo4j_consecutive_failures == 3:
                logger.warning(
                    "[ALERT] Neo4j sync has failed 3 times consecutively. "
                    "Last error: %s. Check Neo4j container: docker ps | grep hermes-neo4j",
                    self._neo4j_last_failure_reason,
                )
            if self._neo4j_consecutive_failures > 10:
                logger.warning(
                    "[ALERT] Neo4j sync failing for %d consecutive episodes. "
                    "Consider disabling Neo4j sync or restarting the container.",
                    self._neo4j_consecutive_failures,
                )

    # --- Neo4j catch-up sync (for episodes created before Neo4j was available) ---
    def sync_to_neo4j(self):
        """
        Sync all episodes from the local episode_map to Neo4j if they're not there yet.
        Called at gateway:startup to catch up on episodes that exist locally but
        weren't synced to Neo4j (e.g. because the config was wrong at the time).
        """
        if not self._graph:
            return  # No Neo4j available
        try:
            from neo4j import GraphDatabase
        except ImportError:
            return

        cfg = self.config["neo4j"]
        try:
            driver = GraphDatabase.driver(
                cfg["uri"], auth=(cfg["user"], cfg["password"]), encrypted=False
            )
        except Exception as exc:
            logger.warning("Could not connect to Neo4j for catch-up sync: %s", exc)
            return

        try:
            with driver.session() as sess:
                for ep_id, ep in self._episode_map.items():
                    # Check if already in Neo4j
                    existing = sess.run(
                        "MATCH (e:Episode {id: $id}) RETURN e.id",
                        id=ep_id,
                    ).single()
                    if existing:
                        continue
                    # Insert missing episode
                    sess.run(
                        """
                        MERGE (e:Episode {id: $id})
                        SET e.title=$title, e.summary=$summary,
                            e.created_at=$created_at, e.tags=$tags
                        """,
                        id=ep.id, title=ep.title, summary=ep.summary,
                        created_at=ep.created_at, tags=ep.tags,
                    )
                # Rebuild NEXT edges in order
                ordered = sorted(
                    self._episode_map.values(),
                    key=lambda e: e.created_at
                )
                for i, ep in enumerate(ordered[:-1]):
                    sess.run(
                        """
                        MATCH (prev:Episode {id: $pid}), (curr:Episode {id: $cid})
                        MERGE (prev)-[:NEXT]->(curr)
                        """,
                        pid=ep.id, cid=ordered[i + 1].id,
                    )
            logger.info("Neo4j catch-up sync complete: %d episodes", len(self._episode_map))
        except Exception as exc:
            logger.warning("Neo4j catch-up sync failed: %s", exc)
        finally:
            driver.close()

    # --- Persistence ---
    def _save_state(self):
        state = {
            "current_episode": self._current_episode.to_dict() if self._current_episode else None,
            "episode_map": {k: v.to_dict() for k, v in self._episode_map.items()},
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, default=str)
        self._vectors.save(self.index_path)

    def _load_state(self):
        if self.state_path.exists():
            with open(self.state_path) as f:
                state = json.load(f)
            if state.get("current_episode"):
                self._current_episode = Episode.from_dict(state["current_episode"])
            for ep_id, ep_dict in state.get("episode_map", {}).items():
                self._episode_map[ep_id] = Episode.from_dict(ep_dict)
            self._vectors.load(self.index_path)

    # --- Public API ---
    def add_message(self, role: str, content: str):
        self._pending_messages.append(Message(role=role, content=content))

    def cut_episode(self, title: str = "", summary: str = "", tags: Optional[list[str]] = None) -> str:
        """Close current episode, persist to stores."""
        if not self._pending_messages:
            return ""

        ep_id = str(uuid.uuid4())[:8]
        text = " ".join([title, summary] + [m.content for m in self._pending_messages])
        embedding = self._embed_fn([text])[0].astype(np.float32)  # list of 1 text → 1 vector

        ep = Episode(
            id=ep_id,
            title=title or f"Episode-{ep_id}",
            summary=summary,
            messages=self._pending_messages.copy(),
            embedding=embedding,
            tags=tags or [],
        )

        prev_id = self._current_episode.id if self._current_episode else None

        # Persist
        self._graph.upsert_episode(ep.id, ep.title, ep.summary, ep.created_at, ep.tags)
        if prev_id:
            self._graph.link(prev_id, "NEXT", ep.id)
        self._vectors.add(ep.id, ep.embedding)

        if prev_id and self._current_episode:
            self._episode_map[prev_id] = self._current_episode

        self._current_episode = ep
        self._pending_messages = []
        self._save_state()

        # Async sync to Neo4j if available
        self._try_neo4j_sync(ep, prev_id)

        return ep_id

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        """Hybrid search: vector similarity + graph expansion."""
        query_vec = self._embed_fn([text])[0].astype(np.float32)

        # Vector search
        if not self._vectors.ids:
            return []
        idxs, dists = self._vectors.search(query_vec, top_k * 2)

        # Build candidate_ids with bounds check (FAISS indices may be stale after reload)
        candidate_ids = []
        for idx in idxs:
            if 0 <= idx < len(self._vectors.ids):
                candidate_ids.append(self._vectors.ids[idx])
            else:
                break  # stale index after vector store reload; stop here

        # Graph expansion — collect neighbor episode IDs
        graph_neighbors: set[str] = set()
        for cid in candidate_ids[:top_k]:
            neighbors = self._graph.get_neighbors(cid, depth=2)
            graph_neighbors.update(neighbors)

        all_ids = list(set(candidate_ids[:top_k]) | graph_neighbors)
        scored = []
        # Build a quick lookup: FAISS_idx → dist for known episode IDs
        idx_to_dist = {cid: dists[i] for i, cid in enumerate(candidate_ids) if i < len(dists)}
        for cid in all_ids:
            if cid not in self._episode_map:
                continue
            ep = self._episode_map[cid]
            vec_score = idx_to_dist.get(cid, 999.0)
            scored.append({
                "id": ep.id,
                "title": ep.title,
                "summary": ep.summary,
                "tags": ep.tags,
                "messages": [{"role": m.role, "content": m.content} for m in ep.messages[-5:]],
                "vector_score": float(vec_score),
            })

        scored.sort(key=lambda x: x["vector_score"])
        return scored[:top_k]

    def get_current_context(self) -> dict:
        current = None
        if self._current_episode:
            current = {
                "id": self._current_episode.id,
                "title": self._current_episode.title,
                "summary": self._current_episode.summary,
                "messages": [{"role": m.role, "content": m.content} for m in self._current_episode.messages],
            }
        return {
            "current_episode": current,
            "pending_messages": [{"role": m.role, "content": m.content} for m in self._pending_messages],
            "total_episodes": len(self._episode_map),
        }
