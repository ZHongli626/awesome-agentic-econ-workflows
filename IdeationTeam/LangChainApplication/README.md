# Multi-Agent Literature Sourcing with Human-in-the-Loop

A LangChain-based application with **four specialized agents** that gather literature through **two rounds with human feedback**.

## Features

- **Four Specialized Agents**: TrendSurfer, TopicCrawler, ScholarSearcher, and GreyScout
- **Human-in-the-Loop**: Two-round process with feedback between rounds
- **AI-Powered Query Refinement**: Each agent uses GPT-4 to generate specialized queries
- **Multi-Source Search**: arXiv, Semantic Scholar, and grey literature sources
- **Intelligent Ranking**: LLM-based relevance scoring with feedback incorporation
- **Comprehensive Coverage**: Academic papers, preprints, and grey literature
- **Export Options**: Save results to CSV with agent attribution

## Installation

### 1. Create Virtual Environment

```powershell
# Navigate to project directory
cd c:\Users\zwang3\Documents\GitHub\awesome-agentic-econ-workflows\IdeationTeam\LangChainApplication

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

## The Four Agents

### 1. **TrendSurfer** 🌊
Identifies emerging trends and recent developments (last 2-3 years)

### 2. **TopicCrawler** 🕷️
Comprehensive academic literature search across all time periods

### 3. **ScholarSearcher** 📚
Finds highly-cited foundational papers (citation count > 50)

### 4. **GreyScout** 🔍
Discovers grey literature (working papers, reports, policy documents)

## Usage

### Interactive Mode (Two Rounds with Feedback)

```powershell
python 1-SourcingStage.py
```

This will:
1. Run Round 1 with all four agents
2. Display top papers
3. Prompt for your feedback
4. Run Round 2 with refined queries based on feedback
5. Save results to CSV files

### Programmatic Mode (No Interactive Prompts)

```powershell
python example_programmatic_feedback.py
```

Or in your code:

```python
from importlib import import_module
sourcing = import_module("1-SourcingStage")
MultiAgentOrchestrator = sourcing.MultiAgentOrchestrator
HumanFeedback = sourcing.HumanFeedback

# Initialize
orchestrator = MultiAgentOrchestrator()

# Round 1
round1_results = orchestrator.run_search_round(
    research_topic="Your research topic",
    round_number=1,
    feedback=None,
    max_results_per_agent=8
)

# View results
orchestrator.print_summary(round_number=1, top_n=15)

# Provide feedback
feedback = HumanFeedback(
    round_number=1,
    relevant_papers=["Paper A", "Paper B"],
    irrelevant_papers=["Paper X"],
    missing_topics=["topic1", "topic2"],
    additional_keywords=["keyword1", "keyword2"],
    comments="Focus more on empirical work"
)

# Round 2 with feedback
round2_results = orchestrator.run_search_round(
    research_topic="Your research topic",
    round_number=2,
    feedback=feedback,
    max_results_per_agent=8
)

# Save all results
orchestrator.save_results("all_results.csv")
```

## Key Components

### `MultiAgentOrchestrator`

Main orchestration class with methods:

- **`run_search_round(topic, round_number, feedback, max_results_per_agent)`**: Execute one round of multi-agent search
- **`collect_human_feedback(round_number)`**: Interactive feedback collection
- **`save_results(filename, round_number)`**: Save results to CSV
- **`print_summary(round_number, top_n)`**: Display top papers

### Agent Classes

Each agent inherits from `BaseAgent` and implements:
- **`refine_query(topic, feedback)`**: Generate specialized search queries
- **`search(query, max_results)`**: Execute searches in their domain

### Data Models

- **`LiteratureItem`**: Paper with title, authors, abstract, URL, source, **agent**, year, literature_type, citation_count, relevance_score
- **`SearchQuery`**: Refined queries, keywords, and focus areas
- **`HumanFeedback`**: Relevant/irrelevant papers, missing topics, additional keywords, comments

## Dependencies

### Core Packages
- `langchain==1.0.7` - LangChain framework
- `langchain-community>=0.4.0` - Community integrations
- `langchain-openai>=1.0.0` - OpenAI integration
- `openai>=1.109.1` - OpenAI API client

### Literature Sources
- `arxiv>=2.1.0` - arXiv API
- `scholarly>=1.7.0` - Google Scholar scraping

### Data Processing
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `pypdf>=3.17.0` - PDF processing

### Vector Stores (Optional)
- `chromadb>=0.4.0` - Vector database
- `faiss-cpu>=1.9.0` - Facebook AI Similarity Search

## Output Files

### CSV Files Generated

- **`round1_literature_results.csv`** - Results from Round 1
- **`round2_literature_results.csv`** - Results from Round 2 (with feedback)
- **`literature_results_all_rounds.csv`** - Combined results from both rounds
- **`round1_feedback.json`** - Structured feedback from Round 1

### CSV Columns

Each row includes:
- `title`, `authors`, `abstract`, `url`
- `source` - Database/repository
- `agent` - Which agent found it (TrendSurfer, TopicCrawler, ScholarSearcher, GreyScout)
- `year`, `literature_type` (academic/grey/preprint)
- `citation_count`, `relevance_score`

## Example Output

```
======================================================================
ROUND 1: Multi-Agent Literature Search
Topic: Agent-based modeling in macroeconomics and monetary policy
======================================================================

--- TrendSurfer ---
Generated 3 queries
Keywords: agent-based, recent, emerging, 2022-2024...
[TrendSurfer] Searching for recent trends: agent-based macro models
  Found 8 items for query: agent-based macro models...

--- TopicCrawler ---
Generated 4 queries
Keywords: macroeconomics, ABM, comprehensive...
[TopicCrawler] Comprehensive academic search: agent-based modeling
  Found 15 items for query: agent-based modeling...

--- ScholarSearcher ---
Generated 3 queries
Keywords: foundational, seminal, highly-cited...
[ScholarSearcher] Searching for foundational papers: ABM economics
  Found 10 items for query: ABM economics...

--- GreyScout ---
Generated 3 queries
Keywords: working papers, policy, reports...
[GreyScout] Searching for grey literature: agent-based policy
  Found 10 items for query: agent-based policy...

[Orchestrator] Total unique items found: 38
[Orchestrator] Ranking 38 papers...

======================================================================
ROUND 1 - TOP 10 PAPERS
======================================================================

1. [ScholarSearcher] Agent-Based Computational Economics
   Authors: Tesfatsion, L., ...
   Source: Semantic Scholar | Year: 2006 | Type: academic
   Citations: 1250
   Relevance: 0.95
   URL: https://...

2. [TrendSurfer] Recent Advances in Agent-Based Macroeconomics
   Authors: Chen, S., Wang, Y., ...
   Source: arXiv | Year: 2024 | Type: preprint
   Relevance: 0.92
   URL: https://arxiv.org/abs/...

Agent Contributions:
  TopicCrawler: 15 papers
  ScholarSearcher: 10 papers
  TrendSurfer: 8 papers
  GreyScout: 5 papers
```

## Customization

### Change LLM Model

Modify the `__init__` method in `LiteratureSourcingAgent`:

```python
self.llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # or "gpt-4-turbo"
    temperature=0.3,
    openai_api_key=self.api_key
)
```

### Add More Sources

Extend the class with additional search methods:

```python
def search_pubmed(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
    # Implementation for PubMed search
    pass
```

## Troubleshooting

### Import Errors
Make sure the virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
```

### API Key Issues
Verify your `.env` file contains a valid OpenAI API key.

### Rate Limiting
If you encounter rate limits, reduce `max_results_per_source` or add delays between API calls.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
