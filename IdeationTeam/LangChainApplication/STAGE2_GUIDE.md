# Stage 2: Research Question Refinement - Guide

## Overview

Stage 2 takes the literature gathered in Stage 1 and uses **two specialized agents** to generate refined research questions through a **two-round feedback process**.

## The Two Agents

### 1. **Ideator** 💡
**Mission**: Generate innovative research concepts from literature

**Process**:
- Analyzes literature from Stage 1
- Identifies research gaps
- Synthesizes insights across papers
- Proposes novel angles and perspectives
- Scores concepts by novelty

**Output**: 8 research concepts with:
- Concept title
- Description
- Key themes
- Literature support
- Novelty score (0-1)

---

### 2. **Refiner** 🎯
**Mission**: Formulate precise, actionable research questions

**Process**:
- Takes concepts from Ideator
- Formulates specific research questions
- Provides rationale for each question
- Suggests methodological approaches
- Links questions to concepts
- Scores questions by feasibility

**Output**: 6 research questions with:
- Question statement
- Rationale
- Methodology hints
- Related concepts
- Feasibility score (0-1)

---

## Workflow

```
Literature CSV (from Stage 1)
         ↓
    ROUND 1
         ↓
    Ideator → Generates 8 concepts
         ↓
    Refiner → Formulates 6 questions
         ↓
    Display concepts & questions
         ↓
  HUMAN FEEDBACK
         ↓
    ROUND 2
         ↓
    Ideator → Refined concepts (based on feedback)
         ↓
    Refiner → Refined questions (based on feedback)
         ↓
    Final output (JSON files)
```

---

## Usage

### Prerequisites

1. **Complete Stage 1 first**:
   ```powershell
   python 1-SourcingStage.py
   ```
   This creates `literature_results_all_rounds.csv`

2. **Activate virtual environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

### Run Stage 2

```powershell
python 2-RefinementStage.py
```

### What Happens

1. **Load Literature**: Reads `literature_results_all_rounds.csv`
2. **Round 1**: 
   - Ideator generates 8 concepts
   - Refiner formulates 6 questions
   - Display results
3. **Feedback Collection**: Interactive prompts for your input
4. **Round 2**:
   - Agents incorporate feedback
   - Generate refined concepts and questions
5. **Save Results**: JSON files with all outputs

**Runtime**: 5-10 minutes per round

---

## Human Feedback

### What You'll Be Asked

```
Enter promising concepts/questions (comma-separated): 
> Concept 1, Question 3

Enter weak or unfeasible items (comma-separated): 
> Concept 5, Question 2

Enter missing angles or perspectives (comma-separated): 
> behavioral aspects, policy implications

Enter directions to focus on (comma-separated): 
> empirical validation, computational methods

General comments: 
> Focus more on practical applications
```

### How Feedback is Used

**Promising items** → Agents emphasize similar themes in Round 2

**Weak items** → Agents avoid similar patterns

**Missing angles** → Agents incorporate into new concepts/questions

**Focus directions** → Agents prioritize these areas

**Comments** → Incorporated into agent prompts

---

## Output Files

### Round 1
- `round1_refinement_results.json` - Concepts and questions
- `round1_refinement_feedback.json` - Your feedback

### Round 2
- `round2_refinement_results.json` - Refined concepts and questions

### Combined
- `refinement_results_all_rounds.json` - All concepts and questions from both rounds

---

## Output Format

### Concepts JSON Structure

```json
{
  "concepts": [
    {
      "concept_title": "Heterogeneous Expectations in Monetary Policy",
      "description": "Exploring how diverse agent expectations influence central bank policy effectiveness...",
      "key_themes": ["heterogeneity", "expectations", "monetary policy", "agent-based models"],
      "literature_support": ["Paper A", "Paper B", "Paper C"],
      "novelty_score": 0.8
    }
  ]
}
```

### Questions JSON Structure

```json
{
  "questions": [
    {
      "question": "How do heterogeneous expectations among economic agents affect the transmission mechanism of monetary policy?",
      "rationale": "This question addresses a critical gap in understanding policy effectiveness...",
      "methodology_hints": ["Agent-based modeling", "Survey data analysis", "DSGE with heterogeneity"],
      "related_concepts": ["Heterogeneous Expectations in Monetary Policy"],
      "feasibility_score": 0.75
    }
  ]
}
```

---

## Example Output

```
======================================================================
ROUND 1: Research Question Refinement
======================================================================

[Ideator] Generating research concepts from 43 papers...
[Ideator] Generated 8 concepts

[Refiner] Formulating research questions from 8 concepts...
[Refiner] Formulated 6 questions

======================================================================
ROUND 1 - RESEARCH CONCEPTS (Top 8)
======================================================================

1. Heterogeneous Expectations in Monetary Policy
   Novelty: 0.8
   Description: Exploring how diverse agent expectations influence...
   Key Themes: heterogeneity, expectations, monetary policy, ABM
   Literature Support: Paper A, Paper B, Paper C

2. Network Effects in Financial Contagion
   Novelty: 0.8
   Description: Investigating how network structures amplify...
   Key Themes: networks, contagion, systemic risk, complexity
   Literature Support: Paper D, Paper E

...

======================================================================
ROUND 1 - RESEARCH QUESTIONS (Top 6)
======================================================================

1. How do heterogeneous expectations among economic agents affect the transmission mechanism of monetary policy?
   Feasibility: 0.75
   Rationale: This question addresses a critical gap...
   Methodologies: Agent-based modeling, Survey data analysis, DSGE
   Related Concepts: Heterogeneous Expectations in Monetary Policy

2. What role do network structures play in amplifying financial contagion during crises?
   Feasibility: 0.75
   Rationale: Understanding network effects is crucial...
   Methodologies: Network analysis, Simulation, Empirical studies
   Related Concepts: Network Effects in Financial Contagion

...
```

---

## Programmatic Usage

```python
from importlib import import_module
import pandas as pd

# Import Stage 2 module
stage2 = import_module("2-RefinementStage")
RefinementOrchestrator = stage2.RefinementOrchestrator
HumanFeedback = stage2.HumanFeedback

# Initialize
orchestrator = RefinementOrchestrator()

# Load literature
lit_df = pd.read_csv("literature_results_all_rounds.csv")

# Round 1
concepts1, questions1 = orchestrator.run_refinement_round(
    literature_df=lit_df,
    round_number=1,
    feedback=None,
    num_concepts=8,
    num_questions=6
)

# View results
orchestrator.print_concepts(round_number=1, top_n=8)
orchestrator.print_questions(round_number=1, top_n=6)

# Programmatic feedback
feedback = HumanFeedback(
    round_number=1,
    promising_items=["Concept 1", "Question 3"],
    weak_items=["Concept 5"],
    missing_angles=["behavioral aspects", "policy implications"],
    focus_directions=["empirical validation"],
    comments="Focus more on practical applications"
)

# Round 2
concepts2, questions2 = orchestrator.run_refinement_round(
    literature_df=lit_df,
    round_number=2,
    feedback=feedback,
    num_concepts=8,
    num_questions=6
)

# Save all results
orchestrator.save_results("refinement_results_all_rounds.json")
```

---

## Customization

### Adjust Number of Outputs

```python
# Generate more concepts and questions
orchestrator.run_refinement_round(
    literature_df=lit_df,
    round_number=1,
    feedback=None,
    num_concepts=12,  # More concepts
    num_questions=10  # More questions
)
```

### Change LLM Temperature

```python
# In Ideator or Refiner __init__
self.llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.9,  # Higher for more creativity
    openai_api_key=self.api_key
)
```

### Use Different Literature File

```python
# In main()
literature_csv = "custom_literature.csv"
```

---

## Tips for Effective Feedback

1. **Be Specific**: Reference exact concept titles or question text
2. **Identify Gaps**: Point out missing perspectives clearly
3. **Prioritize**: Focus on the most important directions
4. **Balance**: Don't over-constrain - allow agents to explore
5. **Iterate**: Use Round 2 to refine based on Round 1 insights

---

## Integration with Stage 1

### Full Pipeline

```powershell
# Stage 1: Literature Sourcing
python 1-SourcingStage.py
# Output: literature_results_all_rounds.csv

# Stage 2: Question Refinement
python 2-RefinementStage.py
# Input: literature_results_all_rounds.csv
# Output: refinement_results_all_rounds.json
```

### Data Flow

```
1-SourcingStage.py
    ↓ (CSV with papers)
2-RefinementStage.py
    ↓ (JSON with concepts & questions)
[Next Stage: Research Design]
```

---

## Troubleshooting

### Error: `literature_results_all_rounds.csv not found`

**Solution**: Run Stage 1 first:
```powershell
python 1-SourcingStage.py
```

### Error: `KeyError: 'relevance_score'`

**Solution**: The CSV is missing expected columns. Ensure you're using output from Stage 1.

### Too Generic Concepts/Questions

**Solution**: 
- Provide more specific feedback
- Increase literature summary size in `_prepare_literature_summary()`
- Use higher temperature for more creativity

### Low Novelty/Feasibility Scores

**Solution**: These are placeholder heuristics. You can:
- Implement better scoring using embeddings
- Manually review and adjust
- Use feedback to guide Round 2

---

## Next Steps

After Stage 2, you'll have:
- ✅ Curated literature (from Stage 1)
- ✅ Research concepts (from Ideator)
- ✅ Refined research questions (from Refiner)

**Potential Stage 3**: Research Design
- Detailed methodology
- Data requirements
- Analysis plan
- Expected contributions

---

## Key Features

- **Two-Agent Pipeline**: Ideator → Refiner
- **Human-in-the-Loop**: Two rounds with feedback
- **Literature-Grounded**: Based on Stage 1 results
- **Structured Output**: JSON with all metadata
- **Scoring**: Novelty and feasibility metrics
- **Flexible**: Programmatic or interactive use
