# 🧠 Context Engineer

### Advanced Context Engineering for Production AI Agents

🚀 <a href="https://context-engineer.streamlit.app/" target="_blank" rel="noopener noreferrer">Live App</a>

👉 https://context-engineer.streamlit.app/

> **Production-grade context management for LLM agents** — A LangGraph system that actively monitors, classifies, offloads, and retrieves context so your AI agent never loses track of what matters — even across multiple sessions.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io)
[![LangSmith](https://img.shields.io/badge/LangSmith-Observability-purple.svg)](https://smith.langchain.com)
[![SQLite](https://img.shields.io/badge/SQLite-Persistence-lightblue.svg)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [The 7 Techniques](#-the-7-techniques)
- [Cross-Session Persistent Memory](#-cross-session-persistent-memory)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Observability](#-observability)
- [Red-Team Testing](#-red-team-testing)
- [Key Design Decisions](#-key-design-decisions)
- [Author](#-author)

---

## 🎯 The Problem

Most AI agents break **silently** when their context window fills up. The model starts hallucinating earlier facts, forgetting key constraints, and degrading in quality — with no warning to the user. By the time you notice, the conversation is already corrupted.

The standard fix (truncate old messages) is destructive. You throw away context the agent might still need.

**This project solves it differently:**

The agent monitors its own context health, classifies every message by importance, offloads low-priority content to a SQLite store before the window fills, and retrieves relevant history on demand — using sub-agents as tools. The result is **effective unbounded memory with a fixed active context window**, and the whole system is fully observable via LangSmith.

---

## ⚙️ The 7 Techniques

| #   | Technique                    | What It Does                                                                                          | Where in Code                     |
| --- | ---------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------- |
| T1  | 🔴 **Pre-Rot Threshold**     | Triggers cleanup at 70% capacity — proactively, before quality degrades                               | `nodes.py → monitor_tokens_node`  |
| T2  | 🟡 **Layered Action Space**  | Classifies every message as CRITICAL / WORKING / BACKGROUND — CRITICAL messages are never evicted     | `layer_manager.py`                |
| T3  | 🟠 **Context Offloading**    | Moves BACKGROUND messages out of active state into SQLite long-term storage                           | `nodes.py → offload_context_node` |
| T4  | 🔵 **Agent-as-Tool**         | Sub-agents (`retrieve_from_memory`, `summarise_context`) are tools the reasoning node calls on demand | `sub_agents.py`                   |
| T5  | 🟢 **Token Budgeting**       | Fixed token allowances tracked in state — the agent always knows exactly how full its window is       | `token_counter.py`                |
| T6  | 🟣 **Scratchpad Management** | Reasoning trace maintained separately from conversation — never pollutes message history              | `nodes.py → reason_node`          |
| T7  | ⚪ **RAG Compression**       | Retrieved chunks are compressed to their most information-dense form before re-injection              | `context_manager.py`              |

---

## 💾 Cross-Session Persistent Memory

Beyond within-session context management, the system maintains a **true long-term memory store** across sessions:

| Feature                 | How It Works                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 🔐 **Critical Memory**  | CRITICAL messages (identity, key facts, explicit instructions) written to SQLite `critical_memory` table and restored on return |
| 📦 **Session Flush**    | All conversation messages flushed to the offload store at session close — not just when memory pressure forces it               |
| 👤 **Returning Users**  | Never have to re-introduce themselves — prior history is searchable via `retrieve_from_memory` from message one                 |
| 🕐 **Session Tracking** | Every session registered with timestamp and message count — full audit trail of user activity                                   |

This is the architectural pattern used in production customer support agents, legal document assistants, and long-running research pipelines.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER MESSAGE                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              🏷️  CLASSIFY INPUT NODE  (T2)                      │
│  • Labels message: CRITICAL / WORKING / BACKGROUND              │
│  • CRITICAL = identity facts, key constraints, system prompt    │
│  • Detects and blocks memory poisoning attacks                  │
│  • Promotes short user messages out of BACKGROUND tier          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              📊  MONITOR TOKENS NODE  (T1, T5)                  │
│  • Counts tokens across all active messages                     │
│  • Checks Pre-Rot Threshold (default 70%)                       │
│  • Sets needs_offload flag — deterministically, not via LLM     │
│  • Records token snapshot to scratchpad (T6)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
          needs_offload=True           needs_offload=False
              │                               │
              ▼                               │
┌─────────────────────────┐                  │
│  📤 OFFLOAD CONTEXT  (T3)│                  │
│  • Moves BACKGROUND msgs │                  │
│    from state to SQLite  │                  │
│  • Preserves all CRITICAL│                  │
│  • Updates token count   │                  │
└─────────────────────────┘                  │
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              🔍  RETRIEVE CONTEXT NODE  (T7)                    │
│  • Searches offload store for messages relevant to current query│
│  • Compresses retrieved chunks via RAG compression              │
│  • Injects compressed context back into reasoning window        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              🧠  REASON NODE  (T4, T6)                          │
│  • Claude API call with optimised context window                │
│  • Has access to sub-agent tools (Agent-as-Tool):               │
│    ├── retrieve_from_memory — semantic search over offload store │
│    └── summarise_context   — compress a set of messages         │
│  • Appends timestamped entry to scratchpad on every turn        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              💬  RESPOND NODE                                    │
│  • Writes final response to message history                     │
│  • Persists CRITICAL messages to cross-session store            │
│  • Updates session activity counter                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FINAL RESPONSE                            │
│     Layer badge · Scratchpad trace · Token dashboard update     │
└─────────────────────────────────────────────────────────────────┘
```

### Sub-Agents (T4 — Agent-as-Tool)

The reasoning node has access to two tools it calls autonomously:

| Tool                      | Purpose                                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------------------- |
| 🔎 `retrieve_from_memory` | Keyword-scored search over offloaded messages — pulls relevant history back into context when needed |
| 📝 `summarise_context`    | Produces a compressed summary of a set of messages when the full text is too large to re-inject      |

---

## 🛠 Tech Stack

| Component           | Technology                                 |
| ------------------- | ------------------------------------------ |
| **LLM**             | Anthropic Claude Sonnet                    |
| **Agent Framework** | LangGraph 0.2                              |
| **Persistence**     | SQLite (built-in Python)                   |
| **Token Counting**  | tiktoken `cl100k_base` + 10% safety buffer |
| **Observability**   | LangSmith (EU instance)                    |
| **Frontend**        | Streamlit                                  |
| **Testing**         | pytest — 64 tests, no API key required     |
| **Language**        | Python 3.11+                               |

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- An [Anthropic API key](https://console.anthropic.com)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ojey-egwuda/context-engineer
cd context_engineer
```

### Step 2: Create Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Demo mode — set low to trigger offloading quickly
TOKEN_BUDGET=3000

# Optional — observability (recommended)
LANGSMITH_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=context-engineer
LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
```

**Get API Keys:**

- Anthropic API Key: [console.anthropic.com](https://console.anthropic.com)
- LangSmith Key: [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys

---

## 🚀 Usage

### Streamlit App (Recommended)

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`. Enter your name to start a session. The token dashboard updates in real time — every message shows its layer classification, and the scratchpad records the full reasoning trace.

**Demo tip:** Set `TOKEN_BUDGET=3000` in `.env` to trigger offloading within a few messages and watch T1, T3, and T4 all activate.

### Running Tests

```bash
# Full suite — no API key required, fully deterministic
pytest tests/ -v

# Boundary tests only (64K / 100K / 128K thresholds)
pytest tests/test_boundaries.py -v
```

### Running the Red-Team Evaluator

```bash
# Requires LANGSMITH_API_KEY in .env
python evaluators/red_team_evaluator.py
```

Results appear in LangSmith under `Datasets → red-team-cases → Experiments`.

---

## 📁 Project Structure

```
context_engineer/
├── app.py                          # Streamlit UI — token dashboard, scratchpad, chat
├── requirements.txt
├── setup.sh                        # One-command setup
├── .env.example
├── evaluators/
│   └── red_team_evaluator.py       # LangSmith automated evaluation suite (5 categories)
├── src/
│   ├── config.py                   # All tunable values — token budget, thresholds, model
│   ├── context/
│   │   ├── token_counter.py        # tiktoken counting with 10% safety buffer
│   │   ├── layer_manager.py        # CRITICAL / WORKING / BACKGROUND classification
│   │   ├── offload_store.py        # SQLite long-term memory + cross-session persistence
│   │   └── context_manager.py      # Context window assembly + RAG compression
│   ├── graph/
│   │   ├── state.py                # AgentState TypedDict schema
│   │   ├── nodes.py                # All graph nodes + identity protection
│   │   └── graph.py                # Graph wiring + conditional edges
│   └── agents/
│       ├── main_agent.py           # Public interface: create_session, chat, get_context_health
│       └── sub_agents.py           # retrieve_from_memory + summarise_context tools
└── tests/
    ├── test_token_counter.py
    ├── test_offload_store.py
    ├── test_layer_manager.py
    └── test_boundaries.py          # 64 tests — boundary validation at 64K / 100K / 128K
```

---

## 📡 Observability

Every graph execution is traced in **LangSmith** (EU instance). Each run captures:

| Signal                      | What It Tells You                                 |
| --------------------------- | ------------------------------------------------- |
| 🔢 Token usage per turn     | How fast the window is filling                    |
| 📤 Offload events           | When and what was evicted from active context     |
| 🔧 Tool calls               | Which sub-agents were invoked and with what query |
| 🏷️ Layer classifications    | How each message was categorised                  |
| 🔍 Retrieved context length | Whether prior memory was successfully recalled    |

---

## 🛡️ Red-Team Testing

The system was tested against **20 adversarial attack categories** across two rounds. All 20 passed.

---

## 🔑 Key Design Decisions

**SQLite over Redis.** SQLite is built into Python and requires zero infrastructure. The `offload_store.py` interface is designed so you can swap in Redis or a vector database with a single file change — no other code changes needed.

**Heuristic layer classification.** Fast, transparent, and debuggable. Every classification decision is visible in the scratchpad. In production, replace with a trained lightweight classifier or a structured LLM call with output validation.

**10% token safety buffer.** Claude uses its own internal tokeniser. tiktoken's `cl100k_base` is within ~5% on most content, but the buffer ensures we never undercount and hit hard API limits unexpectedly.

**Flush-on-exit persistence.** Messages are written to the offload store both when memory pressure forces offloading during a session, and on session close — so the full conversation history is always available to returning users, regardless of whether the token budget was ever hit.

**Deterministic offload decision.** The graph decides when to offload — not the LLM. This is critical for production reliability. A model cannot talk itself out of or into an offload cycle.

---

## 👨‍💻 Author

**Ojonugwa Egwuda** — AI Engineer, Oxford UK

- LinkedIn: [linkedin.com/in/egwudaojonugwa](https://www.linkedin.com/in/egwudaojonugwa/)
- GitHub: [github.com/Ojey-egwuda](https://github.com/Ojey-egwuda)
- Portfolio: [ojey-egwuda.github.io](https://ojey-egwuda.github.io)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built for engineers who care about production reliability**

🧠 🔧 🚀

</div>
