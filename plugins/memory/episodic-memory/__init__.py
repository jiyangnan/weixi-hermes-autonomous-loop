"""
EpisodicMemoryProvider — MemoryProvider plugin for Hermes Agent.

Lifecycle:
  initialize()       — lazy-load EpisodicMemory singleton
  system_prompt_block() — static header (empty; context is injected via prefetch)
  prefetch(query, session_id)  — INJECT recall context on FIRST TURN ONLY
  sync_turn(user, asst)       — write turn to EpisodicMemory
  on_session_end(messages)     — cut episode and persist

Architecture:
  Hooks (session:start, agent:end) handle recording.
  This provider handles INJECTION — reads recall files and returns
  formatted context for the first turn of each session.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# EpisodicMemory singleton (lazy import from skill scripts)
# -----------------------------------------------------------------------
_EM_INSTANCE: Optional[Any] = None
_EM_INIT_LOCK = threading.Lock()
_EM_INIT_DONE = False


def _get_episodic_memory() -> Any:
    """Lazily init and return the EpisodicMemory singleton."""
    global _EM_INSTANCE, _EM_INIT_DONE
    if _EM_INIT_DONE and _EM_INSTANCE is not None:
        return _EM_INSTANCE

    with _EM_INIT_LOCK:
        if _EM_INIT_DONE:
            return _EM_INSTANCE
        _EM_INIT_DONE = True

        skill_scripts = os.path.expanduser("~/.hermes/skills/autonomous-loop/scripts")
        if skill_scripts not in sys.path:
            sys.path.insert(0, skill_scripts)

        try:
            from episodic_memory import EpisodicMemory
            _EM_INSTANCE = EpisodicMemory()
            logger.info("EpisodicMemory singleton ready")
        except Exception as exc:
            logger.warning("Failed to init EpisodicMemory: %s — provider inactive", exc)
            _EM_INSTANCE = None
        return _EM_INSTANCE


# -----------------------------------------------------------------------
# Recall file reader
# -----------------------------------------------------------------------
RECALL_DIR = Path.home() / ".hermes" / "autonomous-loop" / "recall"


def _read_recall_file(session_key: str) -> Optional[Dict[str, Any]]:
    """Read recall file for a session. Returns None if not found."""
    try:
        path = RECALL_DIR / f"{session_key}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read recall file for %s: %s", session_key, exc)
        return None


def _format_recall_context(recall_data: Dict[str, Any], max_episodes: int = 5) -> str:
    """
    Format recall data into a readable context block for injection.

    Returns empty string if no episodes.
    """
    episodes = recall_data.get("episodes", [])
    if not episodes:
        return ""

    # Respect max_episodes gate
    from episodic_memory import EpisodicMemory as _EM

    # Use config gate for max episodes
    try:
        skill_scripts = os.path.expanduser("~/.hermes/skills/autonomous-loop/scripts")
        if skill_scripts not in sys.path:
            sys.path.insert(0, skill_scripts)
        from config import check_autonomy_gate
        max_eps = check_autonomy_gate("max_recall_episodes", default=5)
    except Exception:
        max_eps = 5

    episodes = episodes[:max_eps]

    lines = ["## Relevant Memory (from previous sessions)\n"]
    for ep in episodes:
        title = ep.get("title", "(no title)")
        summary = ep.get("summary", "")
        tags = ep.get("tags", [])
        score = ep.get("relevance_score", 0)

        lines.append(f"### {title} (relevance={score:.2f})")
        if summary:
            lines.append(f"_{summary}_")
        if tags:
            lines.append(f"_tags: {', '.join(tags)}_")
        # Show last 2 messages as context snippets
        msgs = ep.get("messages", [])
        if msgs:
            lines.append("Recent exchange:")
            for m in msgs[-2:]:
                role = m.get("role", "?")
                content = (m.get("content") or "")[:200]
                lines.append(f"  **{role}**: {content}")
        lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# EpisodicMemoryProvider
# -----------------------------------------------------------------------
class EpisodicMemoryProvider(MemoryProvider):
    """
    Memory provider that injects relevant episodic context on the first turn
    of each session, then stays silent for the rest of the conversation.

    - INJECTION: prefetch() returns recall content on turn 0 only
    - RECORDING: sync_turn() writes each turn to EpisodicMemory
    - SESSION END: on_session_end() cuts the episode
    """

    def __init__(self):
        self._turn_count = 0
        self._first_turn_injected = False
        self._session_key: str = ""
        self._prefetch_cache: str = ""
        self._prefetch_lock = threading.Lock()

    # ------------------------------------------------------------------
    # MemoryProvider interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "episodic-memory"

    def is_available(self) -> bool:
        """Always available — no external service needed."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """Reset per-session state. Called at the start of each session."""
        self._turn_count = 0
        self._first_turn_injected = False
        self._session_key = session_id
        self._prefetch_cache = ""

        # Try to pre-load recall data into cache
        try:
            self._prefetch_cache = self._load_recall_for_session(session_id, kwargs)
        except Exception as exc:
            logger.debug("Recall prefetch init failed: %s", exc)
            self._prefetch_cache = ""

    def system_prompt_block(self) -> str:
        """No static text — context comes via prefetch injection."""
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """
        Return recall context ONLY on the first turn (turn_count == 0).

        After the first turn, returns empty string — this provider
        injects once per session and then stays silent.
        """
        if self._turn_count > 0:
            return ""

        with self._prefetch_lock:
            if self._first_turn_injected:
                return ""
            self._first_turn_injected = True

        if not self._prefetch_cache:
            return ""

        logger.info("[episodic-memory] Injecting recall context (session=%s)", session_id[:8])
        return self._prefetch_cache

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Write a completed turn to EpisodicMemory."""
        em = _get_episodic_memory()
        if em is None:
            return

        self._turn_count += 1

        def _bg():
            try:
                if user_content:
                    em.add_message("user", user_content)
                if assistant_content:
                    em.add_message("assistant", assistant_content)
            except Exception as exc:
                logger.warning("[episodic-memory] sync_turn error: %s", exc)

        t = threading.Thread(target=_bg, daemon=True, name=f"episodic-sync-{session_id[:8]}")
        t.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """
        Cut the episode when the session truly ends.
        Runs in background to avoid blocking the gateway shutdown path.
        """
        em = _get_episodic_memory()
        if em is None:
            return

        def _bg():
            try:
                # Replay all messages from the ending session into EpisodicMemory
                for msg in messages:
                    role = msg.get("role", "")
                    content = (msg.get("content") or "").strip()
                    if role in ("user", "assistant") and content:
                        em.add_message(role, content)

                # Derive title from first user message
                title = "(session end)"
                for msg in messages:
                    if msg.get("role") == "user":
                        text = (msg.get("content") or "").replace("```", "").replace("`", "").strip()
                        title = text[:60] + ("…" if len(text) > 60 else "")
                        break

                em.cut_episode(
                    title=title,
                    summary=f"Session ended, {len(messages)} messages",
                    tags=["session-end"],
                )
                logger.info("[episodic-memory] Episode recorded on session end (%d messages)", len(messages))
            except Exception as exc:
                logger.warning("[episodic-memory] on_session_end error: %s", exc)

        t = threading.Thread(target=_bg, daemon=True, name="episodic-session-end")
        t.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _session_key_for(self, session_id: str, context: Dict[str, Any]) -> str:
        """
        Compute the session_key used by the hook when writing recall files.
        Same logic as the hook's _recall_for_session().
        Format: {platform}-{user_id_hash}-new
        """
        import hashlib

        platform = context.get("platform", "")
        user_id = context.get("user_id", "")

        if platform and user_id:
            user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:6]
            return f"{platform.lower()}-{user_hash}-new"

        return session_id

    def _load_recall_for_session(self, session_id: str, context: Dict[str, Any]) -> str:
        """
        Load and format recall data for the current session.
        Session key is derived from platform+user_id (same as hook).
        """
        session_key = self._session_key_for(session_id, context)
        recall = _read_recall_file(session_key)

        if recall is None:
            recall = _read_recall_file(session_id)

        if recall is None:
            return ""

        return _format_recall_context(recall)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """No tools — context-only provider."""
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError(f"EpisodicMemoryProvider has no tool: {tool_name}")

    def shutdown(self) -> None:
        """Flush any pending writes."""
        # sync_turn uses fire-and-forget threads; nothing to flush
        pass
