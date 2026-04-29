# weixi-hermes-autonomous-loop

**Episodic Memory System for Hermes Agent** — gives your AI companion long-term memory, semantic recall, and autonomous self-improvement.

Built for [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research.

---

## What It Does

When you talk to your AI across sessions, it normally starts each conversation with zero context of what came before. This system fixes that.

```
Session 1: User asks about building an Agent platform
Session 2: AI recalls "previously we discussed the skill marketplace..."
Session 3: AI knows the full history of that project — no prompting needed
```

Every meaningful topic shift gets captured as an **Episode**. Episodes are stored persistently with semantic embeddings, so future sessions can search and retrieve relevant context automatically.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Hermes Agent                          │
│                                                          │
│  session:start ──→ Recall Hook (Route A) ──→ inject ctx   │
│  agent:end   ──→ Record + Drift Detection               │
└──────┬──────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              EpisodicMemory (Python)                    │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐    │
│  │ FAISS       │  │ SQLite      │  │ Neo4j         │    │
│  │ VectorStore │  │ GraphStore  │  │ GraphStore    │    │
│  │ (semantic)  │  │ (local)     │  │ (production)  │    │
│  └─────────────┘  └─────────────┘  └───────────────┘    │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────┐          │
│  │ sentence-transformers (all-MiniLM-L6-v2)    │          │
│  │ 384-dim embeddings, offline-first            │          │
│  └─────────────────────────────────────────────┘          │
│                                                          │
│  ┌─────────────────────────────────────────────┐          │
│  │ Autonomous Loop Controller                   │          │
│  │ Observe → Reason → Act (topic drift检测)      │          │
│  └─────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

**Data flow:**

1. `session:start` → read recall file → inject relevant episodes as context
2. Every message → appended to current pending episode
3. `agent:end` → Autonomous Loop Controller checks topic drift
4. `cut_episode()` → episode written to FAISS + SQLite + Neo4j (sync)
5. `query()` → FAISS vector search → graph expansion → ranked results

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Semantic Recall** | Real embeddings via sentence-transformers, not hash-based. Finds conceptually related episodes even without keyword overlap. |
| **Persistent Vector Index** | FAISS IndexFlatL2 with on-disk persistence. Survives restarts. |
| **Topic Drift Detection** | Cosine similarity between first/last user message in an episode. Drift ≥ 0.3 = topic shift. |
| **Autonomy Gates** | All autonomous actions gated by `config.yaml`. Human stays in control. |
| **Fail-Safe Neo4j Sync** | SQLite always succeeds. Neo4j sync failures are logged progressively and never crash the system. |
| **Offline-First** | `HF_HUB_OFFLINE=1`. Sentence-transformers model cached locally. Degrades gracefully to RandomState projection if needed. |

---

## Installation

### Prerequisites

- Hermes Agent (this system is a skill/plugin for it)
- Python 3.11+ with `python3 -m pip`
- Docker (for Neo4j; optional but recommended for production)

### Dependencies

```bash
# Install into hermes-agent venv
python3 -m pip install faiss-cpu sentence-transformers

# Verify
python3 -c "import faiss; from sentence_transformers import SentenceTransformer; print('OK')"
```

### Neo4j (optional)

```bash
docker run -d \
  --name hermes-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

### Gateway Hook Setup

Copy hooks into your hermes-agent hooks directory:

```bash
cp -r hooks/* ~/.hermes/hooks/
# or symlink:
ln -s /path/to/weixi-hermes-autonomous-loop/hooks/* ~/.hermes/hooks/
```

Then restart the gateway.

---

## Configuration

All autonomy gates are in `scripts/config.yaml`:

```yaml
auto_recall: true          # session:start writes recall file
max_recall_episodes: 5     # max episodes injected per session
auto_episode_cut: false    # Agent can auto-cut episodes (OFF by default)
auto_reflection: false     # Agent can write learning conclusions (OFF)
recall_min_score: 0.5      # silently skip episodes below this threshold
```

**Principle:** Autonomous actions are *gated* by these flags. The Agent checks gates before acting. Everything defaults to conservative/off until you explicitly enable it.

---

## Directory Structure

```
weixi-hermes-autonomous-loop/
├── README.md
├── SKILL.md                        # Full technical reference
├── run.sh                          # Quick test runner
├── scripts/
│   ├── episodic_memory.py           # Core: EpisodicMemory, VectorStore, GraphStore
│   ├── config.py                   # Config loader + check_autonomy_gate()
│   ├── config.yaml                 # Autonomy gates
│   └── check_deps.py               # Dependency checker
└── hooks/
    ├── HOOK.yaml                   # Hook registration (session:start, agent:end)
    └── handler.py                  # Hook handlers: recall + autonomous controller
```

---

## Storage Paths

| Path | Content |
|------|---------|
| `~/.hermes/autonomous-loop/` | SQLite DB, FAISS index, recall files |
| `~/.hermes/hooks/episodic-memory/` | Hook configuration (symlink from here) |
| `~/.hermes/skills/autonomous-loop/` | This repository |

---

## Quick Start

```python
import sys
sys.path.insert(0, '/path/to/scripts')
from episodic_memory import EpisodicMemory

em = EpisodicMemory()

# Log a conversation
em.add_message('user', '我想做一个 Agent 雇佣平台')
em.add_message('assistant', '核心功能有哪些？')
em.add_message('user', '技能市场、员工管理、一键部署')
em.cut_episode(title='Agent平台需求', summary='讨论核心功能', tags=['产品', '平台'])

# Next session — semantic search
results = em.query('平台怎么盈利', top_k=3)
for r in results:
    print(f"[{r['title']}] score={r['vector_score']:.3f}")
```

---

## Roadmap

- [ ] Graph expansion via NEXT edges in recall query
- [ ] Self-reflection module (analyze episodes → write learning conclusions)
- [ ] FAISS IVF index for billion-scale vectors
- [ ] External alerting (Telegram/Feishu) for Neo4j failures
- [ ] Web UI for browsing episodes

---

## References

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — Nous Research
- [sentence-transformers](https://sbert.net/) — all-MiniLM-L6-v2
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Semantic Search
