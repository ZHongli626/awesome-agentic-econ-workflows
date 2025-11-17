# Multi-Agent Literature Sourcing with Human-in-the-Loop

## Overview

The refactored `1-SourcingStage.py` now implements a **multi-agent system** with **human-in-the-loop feedback** across two rounds of literature gathering.

## The Four Specialized Agents

### 1. **TrendSurfer** 🌊
- **Focus**: Emerging trends and recent developments
- **Time Range**: Last 2-3 years
- **Targets**:
  - New methodologies and approaches
  - Recent empirical findings
  - Emerging debates
  - Latest technological applications
- **Sources**: arXiv (sorted by submission date)

### 2. **TopicCrawler** 🕷️
- **Focus**: Comprehensive academic literature
- **Coverage**: Broad and thorough
- **Targets**:
  - Core theoretical frameworks
  - Empirical studies and methodologies
  - Review articles and meta-analyses
  - Cross-disciplinary connections
- **Sources**: Semantic Scholar

### 3. **ScholarSearcher** 📚
- **Focus**: Highly-cited foundational papers
- **Criteria**: Citation count > 50
- **Targets**:
  - Seminal theoretical contributions
  - Landmark empirical studies
  - Classic papers that defined the field
  - Highly-cited review articles
- **Sources**: Semantic Scholar (sorted by citations)

### 4. **GreyScout** 🔍
- **Focus**: Grey literature
- **Types**: Non-peer-reviewed but authoritative
- **Targets**:
  - Working papers and preprints
  - Policy reports and white papers
  - Technical reports from institutions
  - Conference proceedings
  - Think tank publications
- **Sources**: arXiv, institutional repositories

## Two-Round Process with Human Feedback

### Round 1: Initial Exploration
1. All four agents search independently
2. Each agent generates 2-3 specialized queries
3. Results are deduplicated and ranked by relevance
4. Top papers are presented to the researcher

### Human Feedback Collection
After Round 1, the system collects:
- **Relevant papers**: Titles of papers that are on-target
- **Irrelevant papers**: Titles of papers to avoid
- **Missing topics**: Areas that need more coverage
- **Additional keywords**: New search terms to include
- **General comments**: Qualitative guidance

### Round 2: Refined Search
1. Agents incorporate human feedback into query generation
2. Search focuses on missing topics and new keywords
3. Avoids patterns from irrelevant papers
4. Builds on successful papers from Round 1
5. Final results combine both rounds

## Usage

### Basic Usage

```python
from 1-SourcingStage import MultiAgentOrchestrator

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
orchestrator.save_results("literature_results.csv", round_number=1)

# Collect feedback (interactive)
feedback = orchestrator.collect_human_feedback(round_number=1)

# Round 2 with feedback
round2_results = orchestrator.run_search_round(
    research_topic="Your research topic",
    round_number=2,
    feedback=feedback,
    max_results_per_agent=8
)

# View final results
orchestrator.print_summary(round_number=2, top_n=15)
orchestrator.save_results("literature_results_all_rounds.csv")
```

### Run the Complete Two-Round Process

```powershell
python 1-SourcingStage.py
```

This will:
1. Run Round 1 with all agents
2. Display top 15 papers
3. Prompt for human feedback (interactive)
4. Run Round 2 with refined queries
5. Display final results
6. Save all results to CSV files

## Feedback Format

When prompted for feedback, you can provide:

```
Relevant paper titles: Paper A, Paper B, Paper C
Irrelevant paper titles: Paper X, Paper Y
Missing topics: heterogeneous agents, network effects
Additional keywords: DSGE, calibration, simulation
General comments: Focus more on empirical applications
```

You can press Enter to skip any field.

## Output Files

### CSV Files
- `round1_literature_results.csv` - Round 1 results
- `round2_literature_results.csv` - Round 2 results
- `literature_results_all_rounds.csv` - Combined results

### Feedback File
- `round1_feedback.json` - Structured feedback from Round 1

## Data Model

Each literature item includes:

```python
{
    "title": str,
    "authors": List[str],
    "abstract": str,
    "url": str,
    "source": str,
    "agent": str,  # Which agent found it
    "year": int,
    "literature_type": str,  # "academic", "grey", "preprint"
    "citation_count": int,
    "relevance_score": float  # 0-1
}
```

## Programmatic Feedback

Instead of interactive feedback, you can provide it programmatically:

```python
from 1-SourcingStage import HumanFeedback

feedback = HumanFeedback(
    round_number=1,
    relevant_papers=["Paper A", "Paper B"],
    irrelevant_papers=["Paper X"],
    missing_topics=["network effects", "behavioral economics"],
    additional_keywords=["heterogeneity", "bounded rationality"],
    comments="Need more recent empirical work"
)

# Use in Round 2
round2_results = orchestrator.run_search_round(
    research_topic="Your topic",
    round_number=2,
    feedback=feedback,
    max_results_per_agent=8
)
```

## Customization

### Adjust Agent Behavior

Modify agent parameters in the class definitions:

```python
# In TrendSurfer.search()
if paper.published.year >= datetime.now().year - 5:  # Last 5 years instead of 3

# In ScholarSearcher.search()
if citation_count and citation_count > 100:  # Higher citation threshold
```

### Add More Agents

Create a new agent class:

```python
class PolicyAnalyzer(BaseAgent):
    """Agent focused on policy-oriented research."""
    
    def __init__(self, openai_api_key: str):
        super().__init__("PolicyAnalyzer", openai_api_key)
    
    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        # Implementation
        pass
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        # Implementation
        pass

# Add to orchestrator
self.agents["PolicyAnalyzer"] = PolicyAnalyzer(self.api_key)
```

### Extend to More Rounds

```python
# Round 3
feedback2 = orchestrator.collect_human_feedback(round_number=2)
round3_results = orchestrator.run_search_round(
    research_topic=research_topic,
    round_number=3,
    feedback=feedback2,
    max_results_per_agent=8
)
```

## Agent Statistics

After each round, view which agents contributed most:

```
Agent Contributions:
  TopicCrawler: 45 papers
  TrendSurfer: 32 papers
  ScholarSearcher: 28 papers
  GreyScout: 25 papers
```

## Benefits of Multi-Agent Approach

1. **Specialization**: Each agent focuses on its expertise area
2. **Coverage**: Comprehensive search across different literature types
3. **Quality**: Mix of foundational and cutting-edge research
4. **Adaptability**: Agents adjust based on human feedback
5. **Transparency**: Clear attribution of which agent found each paper

## Tips for Effective Feedback

1. **Be Specific**: Name exact papers that are relevant/irrelevant
2. **Identify Gaps**: Point out missing subtopics or methodologies
3. **Add Keywords**: Include technical terms the agents might have missed
4. **Provide Context**: Explain why certain papers are more useful
5. **Balance**: Don't over-constrain - let agents explore

## Troubleshooting

### Too Many Results
Reduce `max_results_per_agent` parameter:
```python
orchestrator.run_search_round(..., max_results_per_agent=5)
```

### Missing Relevant Papers
In feedback, add:
- More specific keywords
- Names of key authors
- Specific methodologies or datasets

### Too Much Grey Literature
Adjust GreyScout's weight or filter by `literature_type` in post-processing

## Next Steps

After gathering literature:
1. **Synthesis**: Use results for literature review
2. **Citation Network**: Analyze citation patterns
3. **Gap Analysis**: Identify research opportunities
4. **Methodology Review**: Extract common methods
5. **Theory Building**: Synthesize theoretical frameworks

---

**Note**: The original single-agent version is backed up as `1-SourcingStage-SingleAgent-Backup.py`
