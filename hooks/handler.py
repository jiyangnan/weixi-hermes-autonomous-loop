"""
Episodic Memory Gateway Hook

Hooks:
  - session:start  → Recall relevant episodes, write to session recall file
  - agent:end      → Record completed session as an episode (background thread)

The session:start handler is SYNCHRONOUS — it writes recall results to a file
before the agent starts processing, so the agent can read them immediately.

Files written:
  ~/.hermes/autonomous-loop/recall/{session_key}.json  — recall results for session
"""

import hashlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("episodic-memory-hook")

# Ensure skill scripts are importable
SKILL_DIR = os.path.expanduser("~/.hermes/skills/autonomous-loop/scripts")
if SKILL_DIR not in sys.path:
    sys.path.insert(0, SKILL_DIR)

# Recall results directory
RECALL_DIR = Path.home() / ".hermes" / "autonomous-loop" / "recall"
RECALL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Global singleton (initialized at gateway:startup, one per gateway process)
# ------------------------------------------------------------------
_em_instance: Optional[Any] = None
_em_init_lock = threading.Lock()


def _get_em() -> Any:
    """Lazily init and return the EpisodicMemory singleton."""
    global _em_instance
    if _em_instance is not None:
        return _em_instance

    with _em_init_lock:
        if _em_instance is not None:
            return _em_instance

        try:
            from episodic_memory import EpisodicMemory
            _em_instance = EpisodicMemory()
            logger.info("EpisodicMemory singleton ready")
        except Exception as exc:
            logger.warning("Failed to init EpisodicMemory: %s — episodes will not be recorded", exc)
            _em_instance = _NoOpEM()
        return _em_instance


class _NoOpEM:
    """Fallback when EpisodicMemory can't be loaded."""
    def add_message(self, *a, **kw): pass
    def cut_episode(self, *a, **kw): return None
    def query(self, *a, **kw): return []


# ------------------------------------------------------------------
# Recall logic
# ------------------------------------------------------------------

def _platform_tag(platform: str) -> str:
    """Return the tag used for a platform in episodes."""
    return platform.lower() if platform else ""


def _user_tag(user_id: str) -> str:
    """Return the hashed user tag stored in episode tags."""
    if not user_id:
        return ""
    return f"user-{hashlib.sha256(user_id.encode()).hexdigest()[:6]}"


def _recall_for_session(platform: str, user_id: str, session_id: str, session_key: str) -> Dict[str, Any]:
    """
    Recall episodes relevant to a new session.
    Strategy:
      1. Platform + user filter (exact match on tags)
      2. If no platform/user match, fall back to recent episodes across all topics
      3. Return top-5 most relevant via vector search
    """
    em = _get_em()
    if isinstance(em, _NoOpEM):
        return {"episodes": [], "source": "noop"}

    episodes = []
    candidates = []

    # Strategy 1: try platform + user filtered search
    target_tags = {_platform_tag(platform), _user_tag(user_id)}
    if target_tags:
        for ep_id, ep in em._episode_map.items():
            ep_tags = set(ep.tags)
            if target_tags & ep_tags:  # intersection
                candidates.append((ep_id, ep))

    # Strategy 2: if no filtered candidates, use recent episodes
    if not candidates:
        candidates = sorted(
            em._episode_map.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )[:10]

    # Compute a relevance score based on recency and tag overlap
    scored = []
    for ep_id, ep in candidates[:15]:  # top 15 candidates scored
        ep_tags = set(ep.tags)
        tag_overlap = len(target_tags & ep_tags) if target_tags else 0
        recency_score = 1.0  # TODO: time-decay based on age
        scored.append((ep_id, ep, tag_overlap + recency_score))

    scored.sort(key=lambda x: x[2], reverse=True)
    top_eps = scored[:5]

    for ep_id, ep, score in top_eps:
        msgs = ep.messages[-5:] if ep.messages else []
        episodes.append({
            "id": ep.id,
            "title": ep.title,
            "summary": ep.summary,
            "tags": ep.tags,
            "created_at": ep.created_at,
            "messages": [{"role": m.role, "content": m.content} for m in msgs],
            "relevance_score": float(score),
        })

    return {
        "episodes": episodes,
        "source": "filtered" if target_tags else "recent",
        "platform": platform,
        "user_id_hash": _user_tag(user_id),
        "session_id": session_id,
    }


def _write_recall_file(session_key: str, recall_data: Dict[str, Any]) -> None:
    """Write recall results to session-specific file."""
    try:
        recall_file = RECALL_DIR / f"{session_key}.json"
        with open(recall_file, "w") as f:
            json.dump(recall_data, f, default=str)
        logger.info("Recall file written: %s (%d episodes)", recall_file.name, len(recall_data.get("episodes", [])))
    except Exception as exc:
        logger.warning("Failed to write recall file for %s: %s", session_key, exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _read_session_messages_from_db(session_id: str) -> List[Dict[str, Any]]:
    """Read all messages for a session from the SQLite session DB."""
    try:
        import sqlite3
        from hermes_constants import get_hermes_home

        db_path = get_hermes_home() / "state.db"
        conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """SELECT role, content, tool_calls, tool_name, timestamp, finish_reason
               FROM messages
               WHERE session_id = ?
               ORDER BY id ASC""",
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "role": row["role"],
                "content": row["content"] or "",
                "tool_calls": row["tool_calls"],
                "tool_name": row["tool_name"],
                "timestamp": row["timestamp"],
                "finish_reason": row["finish_reason"],
            }
            for row in rows
        ]
    except Exception as exc:
        logger.warning("Could not read session messages from DB for %s: %s", session_id, exc)
        return []


def _make_title(messages: List[Dict]) -> str:
    """Derive a short title from the first user message in the list."""
    for msg in messages:
        if msg.get("role") == "user":
            text = (msg.get("content") or "").replace("```", "").replace("`", "").strip()
            if len(text) > 60:
                return text[:57] + "..."
            return text or "(empty)"
    return "(no user message)"


def _make_tags(context: Dict[str, Any]) -> List[str]:
    """Derive tags from hook context for episode categorization."""
    tags = []
    if context.get("platform"):
        tags.append(context["platform"].lower())
    if context.get("user_id"):
        tags.append(_user_tag(context["user_id"]))
    return tags


def _record_episode(session_id: str, context: Dict[str, Any]) -> None:
    """
    Read messages from DB and add them as one episode to EpisodicMemory.
    Runs in a background thread so it never blocks the gateway event loop.
    """
    def _bg():
        try:
            em = _get_em()
            if isinstance(em, _NoOpEM):
                return

            messages = _read_session_messages_from_db(session_id)
            if not messages:
                logger.debug("No messages found for session %s", session_id)
                return

            for msg in messages:
                role = msg.get("role", "")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if role in ("user", "assistant"):
                    em.add_message(role, content)

            title = _make_title(messages)
            em.cut_episode(
                title=title,
                summary=f"{len(messages)} messages from {context.get('platform', '?')} session",
                tags=_make_tags(context),
            )
            logger.info("Episode recorded: session=%s title=%r", session_id, title)

        except Exception as exc:
            logger.warning("Error recording episode for %s: %s", session_id, exc)

    t = threading.Thread(target=_bg, daemon=True, name=f"episodic-record-{session_id[:8]}")
    t.start()


# ------------------------------------------------------------------
# Hook handler
# ------------------------------------------------------------------

def _is_autonomy_enabled(gate: str, default: bool = False) -> bool:
    """Check if an autonomy gate is enabled. Defaults to conservative if config missing."""
    try:
        from config import check_autonomy_gate
        return check_autonomy_gate(gate)
    except Exception:
        return default


def _detect_topic_drift(em: Any) -> float:
    """
    Compute topic drift score for the current episode.
    Returns cosine similarity between:
      - embedding of first user message (episode start)
      - embedding of last user message (episode end)

    High drift (low similarity) → topic shifted during this episode.
    Returns 1.0 if not enough messages to compare.
    """
    try:
        if not em._current_episode or len(em._current_episode.messages) < 4:
            return 1.0  # not enough data

        msgs = em._current_episode.messages

        # First and last user messages
        first_user = next((m for m in msgs if m.role == "user"), None)
        last_user = next((m for m in reversed(msgs) if m.role == "user"), None)
        if not first_user or not last_user:
            return 1.0

        # If same user message, drift is 0
        if first_user.content == last_user.content:
            return 0.0

        embed_fn, _ = _get_embedder()
        vec_first = embed_fn([first_user.content])[0].astype(np.float32)
        vec_last = embed_fn([last_user.content])[0].astype(np.float32)

        # Cosine similarity (vectors are already normalized by embedder)
        similarity = float(np.dot(vec_first, vec_last))
        return 1.0 - similarity  # drift = 1 - similarity
    except Exception as exc:
        logger.warning("[autonomous-loop] Drift detection error: %s", exc)
        return 0.0


def _autonomous_loop_controller(session_id: str, context: Dict[str, Any]) -> None:
    """
    Autonomous loop: observe → reason → act.

    Runs after _record_episode in the agent:end background thread.
    Checks topic drift. If drift is high and auto_episode_cut is enabled,
    cuts the episode and starts a fresh one.
    """
    def _bg():
        try:
            em = _get_em()
            if isinstance(em, _NoOpEM):
                return

            # Check auto_episode_cut gate
            if not _is_autonomy_enabled("auto_episode_cut", default=False):
                return

            drift = _detect_topic_drift(em)
            # Threshold: 0.7 similarity → 0.3 drift (clearly different topics)
            DRIFT_THRESHOLD = 0.3

            if drift >= DRIFT_THRESHOLD:
                title = f"auto-cut-{session_id[:8]}"
                logger.info(
                    "[autonomous-loop] Topic drift detected (score=%.3f ≥ %.3f) — auto_episode_cut=true, cutting episode",
                    drift, DRIFT_THRESHOLD,
                )
                em.cut_episode(
                    title=title,
                    summary=f"Auto-cut: topic drift={drift:.3f}",
                    tags=["auto-cut", "topic-drift"],
                )
                logger.info("[autonomous-loop] Episode cut, fresh episode started")
            else:
                logger.debug(
                    "[autonomous-loop] Topic drift=%.3f < %.3f, no action",
                    drift, DRIFT_THRESHOLD,
                )

        except Exception as exc:
            logger.warning("[autonomous-loop] Controller error: %s", exc)

    t = threading.Thread(target=_bg, daemon=True, name=f"autonomous-loop-{session_id[:8]}")
    t.start()


def handle(event_type: str, context: Dict[str, Any]) -> None:
    """
    Gateway hook handler.

    - session:start  → Synchronously recall and write recall file (blocking, gated by auto_recall)
    - agent:end      → Spawn background thread to record the episode (+ autonomous loop controller)
    """
    if event_type == "session:start":
        # auto_recall gate: if disabled, skip recall entirely
        if not _is_autonomy_enabled("auto_recall", default=True):
            logger.info("[episodic-memory] auto_recall=false, skipping recall")
            return

        platform = context.get("platform", "")
        user_id = context.get("user_id", "")
        session_id = context.get("session_id", "")
        session_key = context.get("session_key", session_id)

        logger.info("[episodic-memory] session:start — platform=%s user_id=%s session_id=%s",
                     platform, user_id[:8] if user_id else "", session_id[:8])

        recall_data = _recall_for_session(platform, user_id, session_id, session_key)
        _write_recall_file(session_key, recall_data)
        return

    if event_type == "agent:end":
        session_id = context.get("session_id")
        if not session_id:
            logger.debug("[episodic-memory] agent:end with no session_id, skipping")
            return
        _record_episode(session_id, context)
        # Run autonomous loop controller (topic drift check) after recording
        _autonomous_loop_controller(session_id, context)
        return
