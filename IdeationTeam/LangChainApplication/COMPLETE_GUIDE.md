# Multi-Agent Literature Sourcing System - Complete Guide

A LangChain-based application with **four specialized agents** that gather literature through **two rounds with human-in-the-loop feedback**.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [The Four Agents](#the-four-agents)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Two-Round Process](#two-round-process)
6. [Human Feedback](#human-feedback)
7. [Output Files](#output-files)
8. [Architecture](#architecture)
9. [Customization](#customization)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

```powershell
# 1. Setup
cd c:\Users\zwang3\Documents\GitHub\awesome-agentic-econ-workflows\IdeationTeam\LangChainApplication
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: OPENAI_API_KEY=your_key_here

# 3. Run
python 1-SourcingStage.py
```

---

## The Four Agents

### 1. **TrendSurfer** 🌊
- **Focus**: Recent trends (last 2-3 years)
- **Sources**: arXiv (sorted by date)
- **Best For**: Cutting-edge research

### 2. **TopicCrawler** 🕷️
- **Focus**: Comprehensive academic literature
- **Sources**: Semantic Scholar
- **Best For**: Broad coverage

### 3. **ScholarSearcher** 📚
- **Focus**: Highly-cited foundational papers (>50 citations)
- **Sources**: Semantic Scholar (sorted by citations)
- **Best For**: Field foundations

### 4. **GreyScout** 🔍
- **Focus**: Grey literature (working papers, reports)
- **Sources**: arXiv, institutional repositories
- **Best For**: Policy-relevant work

---

## Installation

### Step 1: Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Core Dependencies**:
- `langchain==1.0.7`
- `langchain-openai>=1.0.0`
- `openai>=1.109.1`
- `arxiv>=2.1.0`
- `pandas>=2.0.0`

### Step 3: Configure API Key

```powershell
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

---

## Usage

### Interactive Mode

```powershell
python 1-SourcingStage.py
```

**Process**:
1. Round 1: All agents search
2. Display top 15 papers
3. Collect your feedback
4. Round 2: Refined search
5. Save results to CSV

**Runtime**: 10-15 minutes

### Programmatic Mode

```python
from importlib import import_module
sourcing = import_module("1-SourcingStage")
MultiAgentOrchestrator = sourcing.MultiAgentOrchestrator
HumanFeedback = sourcing.HumanFeedback

orchestrator = MultiAgentOrchestrator()

# Round 1
round1_results = orchestrator.run_search_round(
    research_topic="Agent-based modeling in macroeconomics",
    round_number=1,
    feedback=None,
    max_results_per_agent=8
)

orchestrator.print_summary(round_number=1, top_n=15)

# Feedback
feedback = HumanFeedback(
    round_number=1,
    relevant_papers=["Paper A", "Paper B"],
    irrelevant_papers=["Paper X"],
    missing_topics=["network effects", "behavioral economics"],
    additional_keywords=["heterogeneity", "bounded rationality"],
    comments="Focus more on empirical applications"
)

# Round 2
round2_results = orchestrator.run_search_round(
    research_topic="Agent-based modeling in macroeconomics",
    round_number=2,
    feedback=feedback,
    max_results_per_agent=8
)

orchestrator.save_results("all_results.csv")
```

---

## Two-Round Process

### Visual Workflow

```
ROUND 1: Initial Exploration
  ↓
All 4 agents search independently
  ↓
Deduplicate & Rank by relevance
  ↓
Present top papers
  ↓
HUMAN FEEDBACK
  ↓
ROUND 2: Refined Search
  ↓
Agents incorporate feedback
  ↓
Targeted search on missing topics
  ↓
Final results (combined)
```

### Round 1: Initial Exploration
1. Each agent generates 3-5 specialized queries using GPT-4
2. Agents search their respective sources
3. Results deduplicated by title similarity
4. LLM ranks papers by relevance (0-1 scale)
5. Top papers presented with agent attribution

### Round 2: Refined Search
1. Agents incorporate feedback into prompts
2. Focus on missing topics and new keywords
3. Avoid patterns from irrelevant papers
4. Build on successful papers from Round 1
5. Merge with Round 1 for final results

---

## Human Feedback

### Interactive Collection

```
Enter relevant paper titles (comma-separated): 
> Paper A, Paper B

Enter irrelevant paper titles (comma-separated): 
> Paper X

Enter missing topics (comma-separated): 
> network effects, behavioral economics

Enter additional keywords (comma-separated): 
> DSGE, calibration, bounded rationality

General comments: 
> Focus more on empirical applications
```

### Programmatic Feedback

```python
feedback = HumanFeedback(
    round_number=1,
    relevant_papers=["Paper A", "Paper B"],
    irrelevant_papers=["Paper X"],
    missing_topics=["network effects", "behavioral economics"],
    additional_keywords=["heterogeneity", "bounded rationality"],
    comments="Focus more on empirical applications"
)
```

### How Feedback is Used

- **Missing topics** → Added to search queries
- **Additional keywords** → Included in keyword lists
- **Comments** → Incorporated into agent prompts
- **Relevant papers** → Boost similar papers in Round 2
- **Irrelevant papers** → Penalize similar patterns

---

## Output Files

### CSV Files

| File | Description |
|------|-------------|
| `round1_literature_results.csv` | Round 1 results |
| `round2_literature_results.csv` | Round 2 results (with feedback) |
| `literature_results_all_rounds.csv` | Combined unique results |
| `round1_feedback.json` | Structured feedback |

### CSV Columns

- `title`, `authors`, `abstract`, `url`
- `source` - Database (arXiv, Semantic Scholar)
- `agent` - Which agent found it
- `year`, `literature_type` (academic/grey/preprint)
- `citation_count`, `relevance_score` (0-1)

### Example Output

```
======================================================================
ROUND 1: Multi-Agent Literature Search
Topic: Agent-based modeling in macroeconomics and monetary policy
======================================================================

--- TrendSurfer ---
Generated 5 queries
[TrendSurfer] Searching for recent trends...
  Found 8 items

--- TopicCrawler ---
Generated 4 queries
[TopicCrawler] Comprehensive academic search...
  Found 15 items

--- ScholarSearcher ---
Generated 3 queries
[ScholarSearcher] Searching for foundational papers...
  Found 10 items

--- GreyScout ---
Generated 3 queries
[GreyScout] Searching for grey literature...
  Found 9 items

[Orchestrator] Total unique items: 38
[Orchestrator] Ranking papers...

======================================================================
ROUND 1 - TOP 10 PAPERS
======================================================================

1. [ScholarSearcher] Agent-Based Computational Economics
   Authors: Tesfatsion, L., Judd, K.
   Source: Semantic Scholar | Year: 2006 | Citations: 1250
   Relevance: 0.95

2. [TrendSurfer] Recent Advances in Agent-Based Macroeconomics
   Authors: Chen, S., Wang, Y.
   Source: arXiv | Year: 2024 | Citations: 0
   Relevance: 0.92

Agent Contributions:
  TopicCrawler: 15 papers
  ScholarSearcher: 10 papers
  TrendSurfer: 8 papers
  GreyScout: 5 papers
```

---

## Architecture

### System Components

```
MultiAgentOrchestrator
  ├── TrendSurfer (Recent trends)
  ├── TopicCrawler (Comprehensive academic)
  ├── ScholarSearcher (Highly-cited foundational)
  └── GreyScout (Grey literature)
```

### Key Classes

**MultiAgentOrchestrator**:
- `run_search_round()` - Execute one round
- `collect_human_feedback()` - Interactive feedback
- `save_results()` - Save to CSV
- `print_summary()` - Display top papers

**BaseAgent** (abstract):
- `refine_query()` - Generate queries
- `search()` - Execute searches

### Data Models

```python
class LiteratureItem(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    url: str
    source: str
    agent: str
    year: int
    literature_type: str
    citation_count: int
    relevance_score: float

class SearchQuery(BaseModel):
    queries: List[str]
    keywords: List[str]
    focus_areas: List[str]

class HumanFeedback(BaseModel):
    round_number: int
    relevant_papers: List[str]
    irrelevant_papers: List[str]
    missing_topics: List[str]
    additional_keywords: List[str]
    comments: str
```

---

## Customization

### Change LLM Model

```python
self.llm = ChatOpenAI(
    model="gpt-4-turbo",  # or "gpt-3.5-turbo"
    temperature=0.3,
    openai_api_key=self.api_key
)
```

### Adjust Agent Behavior

**TrendSurfer time window**:
```python
if paper.published.year >= datetime.now().year - 5:  # Last 5 years
```

**ScholarSearcher citation threshold**:
```python
if citation_count and citation_count > 100:  # Higher threshold
```

### Add New Agents

```python
class PolicyAnalyzer(BaseAgent):
    def __init__(self, openai_api_key: str):
        super().__init__("PolicyAnalyzer", openai_api_key)
    
    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        # Custom query generation
        pass
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        # Custom search logic
        pass

# Add to orchestrator
self.agents["PolicyAnalyzer"] = PolicyAnalyzer(self.api_key)
```

### Extend to More Rounds

```python
for round_num in range(1, 4):  # 3 rounds
    feedback = orchestrator.collect_human_feedback(round_num - 1) if round_num > 1 else None
    results = orchestrator.run_search_round(
        research_topic=topic,
        round_number=round_num,
        feedback=feedback,
        max_results_per_agent=8
    )
```

---

## Troubleshooting

### Common Errors

**`ModuleNotFoundError: No module named 'langchain'`**
```powershell
.\venv\Scripts\Activate.ps1
python 1-SourcingStage.py
```

**`ValueError: OpenAI API key is required`**
```powershell
cp .env.example .env
# Edit .env and add your key
```

**`OutputParserException: Failed to parse SearchQuery`**
- Check OpenAI API key is valid
- Try GPT-4 instead of GPT-3.5
- Reduce `max_results_per_agent`

### Performance Issues

**Too many results**: Reduce `max_results_per_agent=5`

**Missing relevant papers**: Provide more specific feedback

**Too much grey literature**: Filter by `literature_type` in post-processing

### Verification Checklist

- [ ] Virtual environment activated?
- [ ] Packages installed?
- [ ] `.env` file with valid API key?
- [ ] Running from correct directory?

---

## Key Features

### Advantages Over Single-Agent

| Feature | Single Agent | Multi-Agent |
|---------|-------------|-------------|
| Strategy | One approach | Four specialized |
| Feedback | None | Two-round loop |
| Coverage | Limited | Comprehensive |
| Specialization | Generic | Clear roles |
| Adaptability | Static | Iterative |

### Success Metrics

- ✅ Four distinct specialized agents
- ✅ Two-round feedback process
- ✅ Multiple literature sources
- ✅ Agent attribution
- ✅ Relevance ranking with feedback
- ✅ Structured CSV export
- ✅ Interactive and programmatic modes

---

## Tips for Effective Use

**Providing Feedback**:
1. Be specific with paper titles
2. Identify gaps in coverage
3. Add technical keywords
4. Explain why papers are useful
5. Don't over-constrain

**Next Steps**:
1. Synthesize for literature review
2. Analyze citation patterns
3. Identify research gaps
4. Extract methodologies
5. Build theoretical frameworks

---

## Files in This Project

- `1-SourcingStage.py` - Main multi-agent system
- `example_programmatic_feedback.py` - Non-interactive example
- `COMPLETE_GUIDE.md` - This guide
- `RUN_INSTRUCTIONS.md` - Quick reference
- `requirements.txt` - Dependencies
- `.env.example` - Configuration template
- `test_setup.py` - Verification script

---

## Technical Stack

- **LangChain 1.0.7** - Agent framework
- **OpenAI GPT-4** - Query refinement and ranking
- **arXiv API** - Preprints
- **Semantic Scholar API** - Academic papers
- **Pydantic 2.7+** - Data validation
- **Pandas 2.0+** - Data export

---

**Note**: Original single-agent version backed up as `1-SourcingStage-SingleAgent-Backup.py`
