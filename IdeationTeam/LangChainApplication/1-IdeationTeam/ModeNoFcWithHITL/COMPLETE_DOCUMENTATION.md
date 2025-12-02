# Complete Documentation: ModeNoFcWithHITL
# Research Question Generation Pipeline with Human-in-the-Loop

**Last Updated**: December 2, 2025  
**Version**: 1.0  
**Status**: ✅ Fully Functional

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Human-in-the-Loop Process](#human-in-the-loop-process)
4. [LangChain Integration](#langchain-integration)
5. [How to Run](#how-to-run)
6. [Configuration & Environment](#configuration--environment)
7. [Output Files](#output-files)
8. [Feedback Guidelines](#feedback-guidelines)
9. [Troubleshooting](#troubleshooting)
10. [Comparison with No-HITL Mode](#comparison-with-no-hitl-mode)

---

## Overview

### What is ModeNoFcWithHITL?

This mode runs a **human-guided pipeline** from research keywords to finalized research questions with:
- ❌ **No FireCrawl**: Uses only arXiv and Semantic Scholar APIs
- ✅ **Human-in-the-Loop (HITL)**: Collects feedback after each round
- ✅ **Two-round processing**: Initial generation + feedback-guided refinement
- ✅ **LangChain integration**: Uses GPT-4o-mini for intelligent processing
- ✅ **Interactive feedback**: User guides the research direction

### Key Features

- **Guided Exploration**: Your feedback shapes the research direction
- **Iterative Refinement**: Two rounds per stage with feedback between them
- **Quality Control**: Human oversight ensures relevance and quality
- **Flexible**: Skip feedback fields you don't need
- **Comprehensive**: 6 total rounds (2 per stage × 3 stages)

### Pipeline Flow

```
Research Keywords/Ideas
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Literature Sourcing (2 Rounds)                     │
│ File: 1-SourcingStage.py                                    │
│ Method: run_search_round()                                  │
│ Agents: TrendSurfer, TopicCrawler, ScholarSearcher,         │
│         GreyScout                                            │
│                                                              │
│ Round 1: Initial search → Display results                   │
│          ↓                                                   │
│       [HUMAN FEEDBACK]                                       │
│          ↓                                                   │
│ Round 2: Refined search with feedback                       │
│                                                              │
│ Output: literature_results_all_rounds.csv                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Research Question Refinement (2 Rounds)            │
│ File: 2-RefinementStage.py                                  │
│ Method: run_refinement_round()                              │
│ Agents: Ideator → Refiner                                   │
│                                                              │
│ Round 1: Initial concepts & questions → Display             │
│          ↓                                                   │
│       [HUMAN FEEDBACK]                                       │
│          ↓                                                   │
│ Round 2: Refined concepts & questions with feedback         │
│                                                              │
│ Output: refinement_results_all_rounds.json                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Question Integration & Prioritization (2 Rounds)   │
│ File: 3-IntegrationStage.py                                 │
│ Method: run_integration_round()                             │
│ Agents: Contextualizer → Finalizer                          │
│                                                              │
│ Round 1: Initial prioritization → Display                   │
│          ↓                                                   │
│       [HUMAN FEEDBACK]                                       │
│          ↓                                                   │
│ Round 2: Final prioritization with feedback                 │
│                                                              │
│ Output: finalized_research_questions.json                   │
└─────────────────────────────────────────────────────────────┘
    ↓
Final Prioritized Research Questions (Top 5)
```

---

## Pipeline Architecture

### Stage 1: Literature Sourcing (2 Rounds)

**File**: `1-SourcingStage.py`  
**Method**: `run_search_round()`

#### Round 1: Initial Search
- **Input**: Research topic
- **Process**: 4 agents search in parallel
- **Output**: ~30-40 papers
- **Feedback Collected**:
  - Relevant paper titles
  - Irrelevant paper titles
  - Missing topics to explore
  - Additional keywords
  - General comments

#### Round 2: Refined Search
- **Input**: Research topic + Round 1 feedback
- **Process**: Agents adjust queries based on feedback
- **Output**: ~30-40 additional papers
- **Total**: ~60-80 papers across both rounds

#### Multi-Agent System

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ TrendSurfer  │  │TopicCrawler  │  │ScholarSearcher│  │  GreyScout   │
│  (Recent     │  │(Comprehensive│  │(Highly-cited) │  │ (Working     │
│   Trends)    │  │   Coverage)  │  │               │  │  Papers)     │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
       │                 │                  │                  │
       └─────────────────┴──────────────────┴──────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Deduplicate Results │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Rank by Relevance (LLM)│
                    │  (considers feedback)  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Save to CSV          │
                    └───────────────────────┘
```

---

### Stage 2: Research Question Refinement (2 Rounds)

**File**: `2-RefinementStage.py`  
**Method**: `run_refinement_round()`

#### Round 1: Initial Generation
- **Input**: Literature papers from Stage 1
- **Process**: Ideator → Refiner
- **Output**: ~8 concepts, ~6 questions
- **Feedback Collected**:
  - Promising concepts/questions
  - Weak or unfeasible items
  - Missing angles or perspectives
  - Directions to focus on
  - General comments

#### Round 2: Refined Generation
- **Input**: Literature + Round 1 feedback
- **Process**: Agents incorporate feedback
- **Output**: ~8 concepts, ~6 questions
- **Total**: ~16 concepts, ~12 questions across both rounds

#### Two-Agent Pipeline

```
Literature Papers (CSV)
        │
        ▼
┌──────────────────┐
│     Ideator      │  ← Generates research concepts
│  (Concept Gen)   │     from literature patterns
└────────┬─────────┘     (considers feedback)
         │ ~8 concepts
         ▼
┌──────────────────┐
│     Refiner      │  ← Formulates specific research
│ (Question Form)  │     questions from concepts
└────────┬─────────┘     (considers feedback)
         │ ~6 questions
         ▼
refinement_results_round{N}.json
```

---

### Stage 3: Question Integration & Prioritization (2 Rounds)

**File**: `3-IntegrationStage.py`  
**Method**: `run_integration_round()`

#### Round 1: Initial Prioritization
- **Input**: Research questions from Stage 2
- **Process**: Contextualizer → Finalizer
- **Output**: ~5 prioritized questions
- **Feedback Collected**:
  - Question pairs to merge
  - Questions to discard
  - Questions to prioritize
  - Additional guidance

#### Round 2: Final Prioritization
- **Input**: Round 1 questions + feedback
- **Process**: Agents refine based on feedback
- **Output**: ~5 final prioritized questions

#### Two-Agent Pipeline

```
Research Questions (JSON)
        │
        ▼
┌──────────────────────┐
│   Contextualizer     │  ← Adds theoretical framework
│ (Theoretical Frame)  │     and contextual grounding
└──────────┬───────────┘     (considers feedback)
           │ Contextualized questions
           ▼
┌──────────────────────┐
│      Finalizer       │  ← Synthesizes, merges/discards,
│  (Synthesis & Rank)  │     and prioritizes questions
└──────────┬───────────┘     (applies feedback)
           │ ~5 final questions
           ▼
finalized_research_questions.json
finalized_research_questions.txt
```

---

## Human-in-the-Loop Process

### Feedback Collection Points

The pipeline pauses **6 times** to collect your feedback:
1. **Stage 1, Round 1**: After initial literature search
2. **Stage 1, Round 2**: (No feedback - proceeds to Stage 2)
3. **Stage 2, Round 1**: After initial concept/question generation
4. **Stage 2, Round 2**: (No feedback - proceeds to Stage 3)
5. **Stage 3, Round 1**: After initial question prioritization
6. **Stage 3, Round 2**: (No feedback - pipeline completes)

### Feedback Types by Stage

#### Stage 1: Literature Sourcing Feedback

```
HUMAN FEEDBACK - Round 1
======================================================================

Please provide feedback on the results:
(Press Enter to skip any field)

Relevant paper titles (comma-separated): Paper A, Paper B
Irrelevant paper titles (comma-separated): Paper X, Paper Y
Missing topics to explore (comma-separated): topic1, topic2
Additional keywords (comma-separated): keyword1, keyword2
General comments: Focus more on recent papers
```

**What it does**:
- **Relevant papers**: Agents find more papers like these
- **Irrelevant papers**: Agents avoid similar topics
- **Missing topics**: Agents search for these specifically
- **Additional keywords**: Added to search queries
- **Comments**: General guidance for agents

---

#### Stage 2: Refinement Feedback

```
FEEDBACK COLLECTION - Round 1
======================================================================

Please provide feedback on the concepts and questions generated.
Press Enter to skip any field.

Enter promising concepts/questions (comma-separated): concept1, question2
Enter weak or unfeasible items (comma-separated): concept3
Enter missing angles or perspectives (comma-separated): angle1, angle2
Enter directions to focus on (comma-separated): direction1
General comments: Focus on empirical feasibility
```

**What it does**:
- **Promising items**: Agents expand on these
- **Weak items**: Agents avoid similar approaches
- **Missing angles**: Agents explore these perspectives
- **Focus directions**: Agents prioritize these areas
- **Comments**: General guidance for refinement

---

#### Stage 3: Integration Feedback

```
INTEGRATION FEEDBACK - Round 1
======================================================================

Please provide feedback on the prioritized research questions.
Enter question numbers (1-indexed) for each action.
Press Enter to skip any field.

Enter pairs of questions to merge (e.g., '1,2 3,4'): 1,2
Enter questions to discard (comma-separated, e.g., '2,5'): 5
Enter questions to prioritize (comma-separated, e.g., '1,3'): 1,3
Additional guidance for refinement: Emphasize policy implications
```

**What it does**:
- **Merge questions**: Combines related questions
- **Discard questions**: Removes low-priority questions
- **Prioritize questions**: Boosts ranking of specific questions
- **Guidance**: Shapes final synthesis

---

## LangChain Integration

### ✅ Confirmation: LangChain is Fully Integrated

The pipeline uses LangChain throughout for intelligent query generation, refinement, and ranking, with feedback incorporated into prompts.

### LangChain Components Used

#### 1. ChatOpenAI (LLM)
- **Model**: `gpt-4o-mini` (configurable to `gpt-4o`)
- **Temperature**: Varies by agent (0.3-0.7)
- **API Key**: Loaded from `.env` file via `os.getenv("OPENAI_API_KEY")`

```python
from langchain_openai import ChatOpenAI

self.llm = ChatOpenAI(
    model="gpt-4o-mini",  # gpt-4o-mini, gpt-4o
    temperature=0.3,
    openai_api_key=openai_api_key
)
```

#### 2. ChatPromptTemplate with Feedback Context
- **Usage**: Structured prompts that incorporate human feedback
- **Agents Using It**: All agents in all three stages

```python
from langchain_core.prompts import ChatPromptTemplate

feedback_context = ""
if feedback:
    feedback_context = f"""
    Consider this feedback:
    - Relevant papers: {', '.join(feedback.relevant_papers)}
    - Avoid topics like: {', '.join(feedback.irrelevant_papers)}
    - Focus more on: {', '.join(feedback.missing_topics)}
    """

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are TrendSurfer..."),
    ("user", """Generate search queries.
    
    Research Topic: {topic}
    {feedback_context}
    ...""")
])
```

#### 3. PydanticOutputParser
- **Usage**: Ensures LLM outputs are structured as Pydantic models
- **Models Parsed**: `SearchQuery`, `ResearchConcept`, `ResearchQuestion`, etc.

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=SearchQuery)
chain = prompt | self.llm | parser
result = chain.invoke({"topic": research_topic, "feedback_context": feedback_context})
```

### How Feedback is Integrated

1. **Stage 1**: Feedback is converted to text and added to agent prompts
   - Relevant/irrelevant papers guide query generation
   - Missing topics become explicit search targets
   - Additional keywords are incorporated

2. **Stage 2**: Feedback shapes concept and question generation
   - Promising items are expanded upon
   - Weak items are avoided
   - Missing angles are explored

3. **Stage 3**: Feedback directly modifies question list
   - Questions are merged programmatically
   - Questions are discarded before synthesis
   - Priority boosts are applied

---

## How to Run

### Prerequisites

1. **Python Environment**: Python 3.8+
2. **API Key**: OpenAI API key in `.env` file
3. **Dependencies**: Install required packages

```bash
pip install langchain-openai langchain-core python-dotenv pydantic pandas arxiv requests beautifulsoup4
```

### Running the Full Pipeline

```bash
cd c:\Users\zwang3\Documents\GitHub\awesome-agentic-econ-workflows\IdeationTeam\LangChainApplication\1-IdeationTeam\ModeNoFcWithHITL
python 0-MasterOrchestrator.py
```

**What happens**:
1. Prompts for research topic
2. Runs Stage 1 Round 1 → collects feedback
3. Runs Stage 1 Round 2
4. Runs Stage 2 Round 1 → collects feedback
5. Runs Stage 2 Round 2
6. Runs Stage 3 Round 1 → collects feedback
7. Runs Stage 3 Round 2
8. Displays final questions

**Total time**: ~30-60 minutes (including feedback time)

### Running Individual Stages

```bash
# Stage 1 only (2 rounds with feedback)
python 1-SourcingStage.py

# Stage 2 only (requires Stage 1 output)
python 2-RefinementStage.py

# Stage 3 only (requires Stage 2 output)
python 3-IntegrationStage.py
```

---

## Configuration & Environment

### Environment Variables

Create a `.env` file in the `ModeNoFcWithHITL` directory:

```bash
OPENAI_API_KEY=sk-your_openai_api_key_here
```

### Configuration Parameters

Edit these in `0-MasterOrchestrator.py` or individual stage files:

#### Stage 1: Literature Sourcing
```python
max_results_per_agent = 8   # Papers per agent per round (default: 8)
```

#### Stage 2: Refinement
```python
num_concepts = 8            # Research concepts per round (default: 8)
num_questions = 6           # Questions per round (default: 6)
```

#### Stage 3: Integration
```python
max_final_questions = 5     # Final prioritized questions (default: 5)
```

### Model Configuration

To change the LLM model, edit the `ChatOpenAI` initialization in each stage file:

```python
# Current (default)
self.llm = ChatOpenAI(
    model="gpt-4o-mini",  # Fast and cost-effective
    temperature=0.3,
    openai_api_key=openai_api_key
)

# Alternative (more powerful)
self.llm = ChatOpenAI(
    model="gpt-4o",       # More capable, higher cost
    temperature=0.3,
    openai_api_key=openai_api_key
)
```

---

## Output Files

### Expected Output Files in ModeNoFcWithHITL Folder

After running the pipeline, you should see these files:

#### Stage 1 Outputs
- `round1_literature_results.csv` - Round 1 papers
- `round1_feedback.json` - Your Round 1 feedback
- `round2_literature_results.csv` - Round 2 papers
- `literature_results_all_rounds.csv` - All papers combined

#### Stage 2 Outputs
- `round1_refinement_results.json` - Round 1 concepts & questions
- `round1_refinement_feedback.json` - Your Round 1 feedback
- `round2_refinement_results.json` - Round 2 concepts & questions
- `refinement_results_all_rounds.json` - All results combined

#### Stage 3 Outputs
- `round1_integration_results.json` - Round 1 prioritized questions
- `round1_integration_feedback.json` - Your Round 1 feedback
- `round2_integration_results.json` - Round 2 prioritized questions
- `integration_results_all_rounds.json` - All results combined
- `finalized_research_questions.json` - Final questions (JSON)
- `finalized_research_questions.txt` - Final questions (readable)

---

## Feedback Guidelines

### Best Practices for Providing Feedback

#### Stage 1: Literature Sourcing

**DO**:
- ✅ Identify specific paper titles that are highly relevant
- ✅ Point out papers that are off-topic or low-quality
- ✅ Suggest missing research areas or subtopics
- ✅ Add technical keywords or author names

**DON'T**:
- ❌ Provide vague feedback like "find better papers"
- ❌ List too many items (keep to 3-5 per category)
- ❌ Contradict yourself (e.g., marking same topic as both relevant and irrelevant)

**Example Good Feedback**:
```
Relevant papers: "Agent-based models in monetary policy", "HANK models"
Irrelevant papers: "Stock market prediction", "Cryptocurrency analysis"
Missing topics: central bank digital currencies, heterogeneous expectations
Additional keywords: DSGE, macroprudential policy
Comments: Focus on papers from 2020 onwards
```

---

#### Stage 2: Refinement

**DO**:
- ✅ Highlight concepts that are novel and feasible
- ✅ Identify questions that are too broad or too narrow
- ✅ Suggest missing theoretical perspectives
- ✅ Guide toward empirical or theoretical focus

**DON'T**:
- ❌ Reject everything (be constructive)
- ❌ Provide conflicting guidance
- ❌ Focus only on negative feedback

**Example Good Feedback**:
```
Promising: "How do heterogeneous agents affect policy transmission?", concept on network effects
Weak: "What is the future of economics?" (too broad)
Missing angles: behavioral aspects, institutional constraints
Focus on: empirically testable questions, policy-relevant research
Comments: Emphasize questions that can use existing datasets
```

---

#### Stage 3: Integration

**DO**:
- ✅ Merge questions that overlap significantly
- ✅ Discard questions that are infeasible or redundant
- ✅ Prioritize questions with high impact potential
- ✅ Provide guidance on theoretical framing

**DON'T**:
- ❌ Merge unrelated questions
- ❌ Discard all questions
- ❌ Provide contradictory instructions

**Example Good Feedback**:
```
Merge: 1,2 (both about agent heterogeneity)
Discard: 5 (too similar to question 3)
Prioritize: 1,3 (high policy relevance)
Guidance: Emphasize empirical feasibility and data availability
```

---

### Skipping Feedback

You can press Enter to skip any feedback field. The pipeline will continue with the information you provide.

**When to skip**:
- You're satisfied with the current results
- You don't have specific guidance for that category
- You want to see what the next round produces without intervention

---

## Troubleshooting

### Common Issues & Solutions

#### Issue: "OpenAI API key is required"
**Solution**: Ensure `.env` file exists with `OPENAI_API_KEY=sk-...`

```bash
# Check if .env file exists
ls .env

# If not, create it
echo "OPENAI_API_KEY=sk-your_key_here" > .env
```

---

#### Issue: Import errors
**Solution**: Install dependencies:
```bash
pip install langchain-openai langchain-core python-dotenv pydantic pandas arxiv requests beautifulsoup4
```

---

#### Issue: API Rate Limiting (429 errors)
**Solution**: This is expected from Semantic Scholar API. The pipeline continues with results from other sources (arXiv).

**Mitigation**:
- Reduce `max_results_per_agent` parameter
- Add delays between API calls
- Use caching for repeated queries

---

#### Issue: Feedback not being applied
**Possible causes**:
1. Feedback format is incorrect
2. Feedback is too vague
3. Feedback contradicts itself

**Solutions**:
1. Follow the format examples exactly
2. Be specific (mention paper titles, concept names, etc.)
3. Review your feedback for consistency

---

#### Issue: Pipeline takes too long
**Causes**:
- Many API calls to LLM
- Large number of papers to process
- Waiting for user feedback

**Solutions**:
- Reduce `max_results_per_agent` (e.g., from 8 to 5)
- Reduce `num_concepts` and `num_questions`
- Prepare feedback in advance by reviewing results quickly

---

#### Issue: Low-quality results despite feedback
**Possible causes**:
1. Research topic is too broad or vague
2. Feedback is not specific enough
3. Model parameters need tuning

**Solutions**:
1. Make research topic more specific
2. Provide detailed, specific feedback
3. Try `gpt-4o` instead of `gpt-4o-mini`
4. Increase number of concepts/questions generated

---

## Comparison with No-HITL Mode

### Mode Comparison Table

| Feature | ModeNoFcWithHITL | ModeNoFcNoHITL |
|---------|------------------|----------------|
| **FireCrawl** | ❌ | ❌ |
| **Human Feedback** | ✅ (6 collection points) | ❌ |
| **Rounds per Stage** | 2 | 1 |
| **Total Rounds** | 6 | 3 |
| **Runtime** | ~30-60 min | ~5-10 min |
| **User Interaction** | High | None |
| **Customization** | High | Low |
| **Quality Control** | Human-guided | LLM-only |
| **Use Case** | Guided exploration | Rapid prototyping |

---

### When to Use ModeNoFcWithHITL

**✅ Use this mode when**:
- You want to guide the research direction
- Quality is more important than speed
- You have domain expertise to contribute
- You want to ensure relevance and feasibility
- You're exploring a new research area
- You have 30-60 minutes to dedicate

**❌ Don't use this mode when**:
- You need quick initial exploration
- You're processing multiple topics in batch
- You don't have time for feedback
- You trust LLM judgment completely
- You want fully automated operation

---

### Key Differences

#### With Human-in-the-Loop (This Mode)
- **Rounds**: 2 rounds per stage with feedback between them
- **Methods**: `run_search_round()`, `run_refinement_round()`, `run_integration_round()`
- **Feedback**: `collect_human_feedback()` after Round 1 of each stage
- **Output**: Results from both rounds saved separately
- **Customization**: High - user guides the process at each stage
- **Time**: ~30-60 minutes (including user feedback time)

#### Without Human-in-the-Loop (ModeNoFcNoHITL)
- **Rounds**: Single automated round per stage
- **Methods**: `run_automated_search()`, `run_automated_refinement()`, `run_automated_integration()`
- **Feedback**: None - fully automated
- **Output**: Single set of results
- **Customization**: Low - relies on LLM intelligence
- **Time**: ~5-10 minutes (fully automated)

---

## Summary

### What You Get

✅ **Human-guided pipeline** from keywords to research questions  
✅ **Multi-agent system** with specialized literature sourcing  
✅ **LangChain integration** for intelligent processing  
✅ **6 feedback collection points** for maximum control  
✅ **Structured outputs** in JSON and text formats  
✅ **~30-60 minute runtime** including feedback time  
✅ **High-quality results** shaped by your expertise  

### What You Need

📋 **OpenAI API key** in `.env` file  
📋 **Python 3.8+** with required packages  
📋 **Research topic** or keywords  
📋 **30-60 minutes** for full pipeline  
📋 **Domain knowledge** for effective feedback  

### What You'll Get

📄 **~60-80 literature papers** with metadata (2 rounds)  
📄 **~16 research concepts** from literature analysis (2 rounds)  
📄 **~12 research questions** with rationale (2 rounds)  
📄 **~5 final prioritized questions** with full context (2 rounds)  
📄 **All feedback saved** for reproducibility  

---

**Version**: 1.0  
**Status**: ✅ Production Ready  
**Last Updated**: December 2, 2025  
**Model**: GPT-4o-mini (configurable to GPT-4o)

---

*End of Complete Documentation*
