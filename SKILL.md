# Autonomous-Loop Episodic Memory System

## 概念

**Episodic Index**：把对话流切分为独立的情景单元（Episode），每个 Episode 包含：
- 时间戳、主题、摘要、关键消息
- Episode 之间用图组织：时序链（NEXT）、因果链、引用链

**Hybrid-Vector-Graph**：每个 Episode 同时有：
- Vector embedding（语义相似性搜索）
- Graph edges（关系/时序检索）
- 查询时融合两者：向量相似度 + 图扩展

**Autonomous-Loop**：Agent 自主循环——观察→记忆→推理→行动，基于历史 Episodes 自动推理上下文，减少对 session 的依赖。

## 架构

```
EpisodicMemory (episodic_memory.py)
├── SQLite GraphStore   ← 时序/因果边（本地持久化）
├── Neo4j GraphStore    ← 增量同步到 Neo4j（生产环境）
├── FAISS VectorStore   ← 持久化向量索引（L2-normalized，百万级支持）
├── sentence-transformers  ← 真实语义 embedding（离线优先，故障时降级到随机投影）
└── HybridQueryEngine   ← 向量检索 → 图扩展 → 融合排序
```

**存储路径：** `~/.hermes/autonomous-loop/`

## 依赖状态

| 依赖 | 状态 | 说明 |
|------|------|------|
| Neo4j | ✅ 运行中 | `docker ps \| grep hermes-neo4j` |
| neo4j Python 驱动 | ✅ 已安装 | `python3.11 -c "from neo4j import GraphDatabase"` |
| numpy | ✅ 系统自带 | 2.4.3 |
| sentence-transformers | ✅ 已安装 | `all-MiniLM-L6-v2`，384维，离线缓存 |
| faiss-cpu | ✅ 已安装 | `IndexFlatL2`，持久化到 `vectors.npy` |

## 使用

```python
import sys
sys.path.insert(0, '~/.hermes/skills/autonomous-loop/scripts')
from episodic_memory import EpisodicMemory

em = EpisodicMemory()

# 添加消息到当前 pending episode
em.add_message('user', '我想做一个 Agent 平台')
em.add_message('assistant', '你想要什么功能？')

# 触发 episode 切分（话题完成或对话结束）
em.cut_episode(
    title='产品需求讨论',
    summary='确定核心功能和目标用户',
    tags=['产品', '需求']
)

# 继续下一个 episode
em.add_message('user', '多租户怎么做')
em.add_message('assistant', '数据隔离是关键')
em.cut_episode(title='多租户架构', summary='讨论隔离方案', tags=['架构'])

# 混合检索
results = em.query('Agent 平台功能', top_k=5)
for r in results:
    print(f'{r["title"]} (score={r["vector_score"]:.3f})')
    for m in r['messages']:
        print(f'  {m["role"]}: {m["content"]}')

# 获取当前上下文（用于新会话初始化）
ctx = em.get_current_context()
print(f'当前 episode: {ctx["current_episode"]["title"]}')
print(f'待处理消息: {len(ctx["pending_messages"])}')
print(f'历史 episodes: {ctx["total_episodes"]}')
```

## 检索流程

1. **向量搜索**：FAISS/内存向量找到语义最相似的 Episodes
2. **图扩展**：从相似 Episodes 出发，沿 NEXT 边扩展 1-2 跳
3. **融合排序**：综合向量距离和图距离
4. **去重摘要**：合并高度相关内容

## Neo4j 管理

```bash
# 查看容器状态
docker ps | grep hermes-neo4j

# Neo4j Browser
open http://localhost:7474

# 查询所有 episodes（ Cypher）
MATCH (e:Episode) RETURN e.id, e.title, e.created_at ORDER BY e.created_at

# 清理所有数据
MATCH (e:Episode) DETACH DELETE e

# 查看边
MATCH ()-[r:NEXT]->() RETURN r.src, r.dst

# 查看 sync 健康状态（从 Python）
python -c "
import sys; sys.path.insert(0, '~/.hermes/skills/autonomous-loop/scripts')
from episodic_memory import EpisodicMemory
em = EpisodicMemory()
print('consecutive_failures:', em._neo4j_consecutive_failures)
print('last_failure_reason:', em._neo4j_last_failure_reason)
"
```

**错误处理：** Neo4j sync 失败时，本地 SQLite 写入不受影响（fail-safe）。连续失败会计数并在第 3 次、第 10 次触发 logger.warning 告警。成功同步后计数器自动归零。

## Autonomy Config Layer

`~/.hermes/skills/autonomous-loop/scripts/config.yaml` — human controls all scope gates here.

| Gate | Default | Meaning |
|------|---------|---------|
| `auto_recall` | `true` | `session:start` hook 自动写 recall 文件 |
| `max_recall_episodes` | `5` | 最多注入多少条 episodes |
| `auto_episode_cut` | `false` | Agent 能否主动 cut_episode（提交到长期记忆） |
| `auto_reflection` | `false` | Agent 能否分析 episodes 并写回学习结论 |
| `recall_min_score` | `0.5` | 低于此分数的 episode 静默跳过 |

**行为原则：** 所有 autonomous 行动在被执行前，代码必须调用 `check_autonomy_gate("gate_name")`。如果 gate 关闭，行为跳过或降级，不报错。

**BotLearn 参照：** 模式同 BotLearn `config.json` — human 定义范围，agent 在范围内自主行动，超出范围则等待确认。

## Autonomous Loop Controller

自动 episode 切分逻辑（运行在 `agent:end` 后台线程）：

```
Observe  → 提取当前 episode 的首尾 user 消息 embedding
Reason   → 计算余弦相似度，drift = 1 - similarity
Act      → 如果 drift ≥ 0.3 且 auto_episode_cut=true，执行 cut_episode
```

**Topic Drift 检测：**
- drift ≥ 0.3：话题明显转移，触发 auto-cut
- drift < 0.3：话题一致，不行动
- 消息 < 4 条时：数据不足，不行动

**注意：** `auto_episode_cut` 默认 false，所以即使 drift 高也不会自动 cut。等你确认需要这个行为后再开。

## 新会话开场流程

当开始一个**新会话**时（即 session:start 事件触发后，agent 处理第一条消息之前）：

1. **读取 recall 文件**：`~/.hermes/autonomous-loop/recall/{session_key}.json`
   - `session_key` 从当前会话上下文获取（格式如 `discord-{user_id}-new`）
   - 如果文件存在且有 `episodes`，继续下一步
   - 如果文件不存在或 episodes 为空，跳过

2. **注入相关上下文**：从 episodes 中提取与当前话题最相关的内容，以简短摘要形式告诉 agent，例如：
   ```
   [相关记忆] 你们之前讨论过「Agent雇佣平台」，
   确定核心功能是技能市场+员工管理+一键部署。
   最近一次谈到「多租户隔离」方案。
   ```

3. **选择性引用**：如果 recall 中有高度相关的具体消息（relevance_score 高），可以让 agent 参考那些对话

**为什么这样做：** recall 文件已经通过 semantic search + tag filtering 找好了相关 episodes，agent 只需要读文件并决定怎么用。这比改 gateway 核心代码更稳定，升级不丢。

## 升级计划

- [x] 安装 sentence-transformers（真实语义 embedding，消除 hash collision）
- [x] 接入 Hermes Agent session 生命周期（session:start → 写 recall 文件）
- [x] 新会话开场流程（agent 读 recall 文件注入上下文）
- [x] Config Layer（autonomy gates，BotLearn 模式）
- [x] Neo4j 错误处理（fail-safe + 渐进告警 + 失败计数）
- [x] VectorStore 升级为 FAISS（持久化 + L2-normalized search）
- [x] Autonomous loop controller（topic drift 检测 + auto_episode_cut）
