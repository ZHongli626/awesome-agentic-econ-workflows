"""
Research Question Integration Stage with Human-in-the-Loop
This script integrates refined research questions through theoretical framing and synthesis.

Agents:
- Contextualizer: Provides theoretical framing for research questions
- Finalizer: Synthesizes and prioritizes research questions

Input: Refined questions from 2-RefinementStage.py (JSON file)
Output: Finalized, prioritized research questions after two-round feedback process
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class ResearchQuestion(BaseModel):
    """Model for a research question."""
    question: str = Field(description="The research question")
    rationale: str = Field(description="Why this question is important")
    methodology_hints: List[str] = Field(description="Suggested methodological approaches")
    related_concepts: List[str] = Field(description="Related concept titles")
    feasibility_score: Optional[float] = Field(description="Feasibility score (0-1)", default=None)


class ContextualizedQuestion(BaseModel):
    """Model for a contextualized research question with theoretical framing."""
    question: str = Field(description="The research question")
    theoretical_framework: str = Field(description="Theoretical framework and positioning")
    literature_gaps: List[str] = Field(description="Specific gaps in literature this addresses")
    contribution: str = Field(description="Expected contribution to the field")
    related_theories: List[str] = Field(description="Related theoretical perspectives")
    priority_score: Optional[float] = Field(description="Priority score (0-1)", default=None)


class PrioritizedQuestion(BaseModel):
    """Model for a prioritized research question after synthesis."""
    question: str = Field(description="The synthesized research question")
    theoretical_framework: str = Field(description="Theoretical framework")
    rationale: str = Field(description="Comprehensive rationale")
    methodology: List[str] = Field(description="Recommended methodologies")
    expected_impact: str = Field(description="Expected impact and contribution")
    feasibility: str = Field(description="Feasibility assessment")
    priority_rank: int = Field(description="Priority ranking")
    priority_score: float = Field(description="Priority score (0-1)")


class IntegrationFeedback(BaseModel):
    """Model for human feedback during integration stage."""
    round_number: int = Field(description="Feedback round number")
    questions_to_merge: List[Tuple[int, int]] = Field(
        description="Pairs of question indices to merge", 
        default_factory=list
    )
    questions_to_discard: List[int] = Field(
        description="Indices of questions to discard", 
        default_factory=list
    )
    questions_to_prioritize: List[int] = Field(
        description="Indices of questions to prioritize", 
        default_factory=list
    )
    additional_guidance: str = Field(
        description="Additional guidance for refinement", 
        default=""
    )


class ContextualizedQuestionList(BaseModel):
    """Model for a list of contextualized questions."""
    questions: List[ContextualizedQuestion] = Field(description="List of contextualized questions")


class PrioritizedQuestionList(BaseModel):
    """Model for a list of prioritized questions."""
    questions: List[PrioritizedQuestion] = Field(description="List of prioritized questions")


class Contextualizer:
    """Agent that provides theoretical framing for research questions."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "Contextualizer"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # gpt-4o-mini, gpt-4o
            temperature=0.5,
            openai_api_key=self.api_key
        )
    
    def contextualize_questions(
        self,
        questions: List[ResearchQuestion],
        feedback: Optional[IntegrationFeedback] = None
    ) -> List[ContextualizedQuestion]:
        """Provide theoretical framing for research questions."""
        
        print(f"\n[{self.agent_name}] Contextualizing {len(questions)} research questions...")
        
        # Prepare questions summary
        questions_summary = self._prepare_questions_summary(questions)
        
        # Prepare feedback context
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Questions to prioritize: {feedback.questions_to_prioritize}
            - Questions to discard: {feedback.questions_to_discard}
            - Additional guidance: {feedback.additional_guidance}
            """
        
        parser = PydanticOutputParser(pydantic_object=ContextualizedQuestionList)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Contextualizer, an expert at providing theoretical framing and positioning for research questions. You must respond with valid JSON only."),
            ("user", """Provide theoretical framing for the following research questions.
            
            Research Questions:
            {questions_summary}
            
            {feedback_context}
            
            For each question, provide:
            - Theoretical framework: Position the question within existing theoretical perspectives
            - Literature gaps: Identify specific gaps this question addresses
            - Contribution: Articulate the expected contribution to the field
            - Related theories: List relevant theoretical perspectives (3-5)
            
            Respond with a JSON object containing:
            - "questions": array of question objects, each with "question", "theoretical_framework", "literature_gaps", "contribution", "related_theories"
            
            Example format:
            {{
              "questions": [
                {{
                  "question": "How does X affect Y?",
                  "theoretical_framework": "This question builds on institutional theory...",
                  "literature_gaps": ["Gap 1", "Gap 2"],
                  "contribution": "This research will contribute by...",
                  "related_theories": ["Theory 1", "Theory 2", "Theory 3"]
                }}
              ]
            }}
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm | parser
        result = chain.invoke({
            "questions_summary": questions_summary,
            "feedback_context": feedback_context
        })
        
        # Score priority based on theoretical richness
        contextualized_questions = result.questions
        for question in contextualized_questions:
            question.priority_score = self._score_priority(question)
        
        print(f"[{self.agent_name}] Contextualized {len(contextualized_questions)} questions")
        return contextualized_questions
    
    def _prepare_questions_summary(self, questions: List[ResearchQuestion]) -> str:
        """Prepare a summary of questions for the prompt."""
        summary_parts = []
        for i, q in enumerate(questions, 1):
            summary = f"""
Question {i}: {q.question}
Rationale: {q.rationale}
Methodologies: {', '.join(q.methodology_hints)}
Related Concepts: {', '.join(q.related_concepts)}
Feasibility: {q.feasibility_score}
"""
            summary_parts.append(summary)
        return "\n".join(summary_parts)
    
    def _score_priority(self, question: ContextualizedQuestion) -> float:
        """Score the priority of a contextualized question."""
        # Heuristic: more literature gaps and theories = higher priority
        gap_score = min(len(question.literature_gaps) / 5.0, 1.0)
        theory_score = min(len(question.related_theories) / 5.0, 1.0)
        return round((gap_score + theory_score) / 2.0, 2)


class Finalizer:
    """Agent that synthesizes and prioritizes research questions."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "Finalizer"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # gpt-4o-mini, gpt-4o
            temperature=0.3,
            openai_api_key=self.api_key
        )
    
    def synthesize_questions(
        self,
        contextualized_questions: List[ContextualizedQuestion],
        feedback: Optional[IntegrationFeedback] = None,
        max_questions: int = 5
    ) -> List[PrioritizedQuestion]:
        """Synthesize and prioritize research questions."""
        
        print(f"\n[{self.agent_name}] Synthesizing {len(contextualized_questions)} contextualized questions...")
        
        # Apply feedback: merge and discard questions
        processed_questions = self._apply_feedback(contextualized_questions, feedback)
        
        # Prepare questions summary
        questions_summary = self._prepare_contextualized_summary(processed_questions)
        
        # Prepare feedback context
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Questions merged: {len(feedback.questions_to_merge)} pairs
            - Questions discarded: {len(feedback.questions_to_discard)}
            - Additional guidance: {feedback.additional_guidance}
            """
        
        parser = PydanticOutputParser(pydantic_object=PrioritizedQuestionList)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Finalizer, an expert at synthesizing and prioritizing research questions. You must respond with valid JSON only."),
            ("user", """Synthesize and prioritize the following contextualized research questions.
            
            Contextualized Questions:
            {questions_summary}
            
            {feedback_context}
            
            Generate up to {max_questions} prioritized research questions that:
            - Synthesize related questions into coherent research directions
            - Prioritize based on impact, feasibility, and theoretical contribution
            - Provide clear, actionable formulations
            - Include comprehensive rationale and methodology
            
            For each prioritized question:
            - Question: Clear, synthesized research question
            - Theoretical framework: Integrated theoretical positioning
            - Rationale: Comprehensive justification
            - Methodology: Recommended research methods (3-5)
            - Expected impact: Anticipated contribution and impact
            - Feasibility: Assessment of research feasibility
            - Priority rank: Ranking from 1 (highest) to {max_questions}
            - Priority score: Numerical score (0-1)
            
            Respond with a JSON object containing:
            - "questions": array of question objects with all fields above
            
            Example format:
            {{
              "questions": [
                {{
                  "question": "Synthesized question...",
                  "theoretical_framework": "Framework...",
                  "rationale": "Rationale...",
                  "methodology": ["Method 1", "Method 2"],
                  "expected_impact": "Impact...",
                  "feasibility": "Feasibility assessment...",
                  "priority_rank": 1,
                  "priority_score": 0.95
                }}
              ]
            }}
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm | parser
        result = chain.invoke({
            "questions_summary": questions_summary,
            "feedback_context": feedback_context,
            "max_questions": max_questions
        })
        
        prioritized_questions = result.questions
        
        # Sort by priority rank
        prioritized_questions.sort(key=lambda x: x.priority_rank)
        
        print(f"[{self.agent_name}] Synthesized into {len(prioritized_questions)} prioritized questions")
        return prioritized_questions
    
    def _apply_feedback(
        self,
        questions: List[ContextualizedQuestion],
        feedback: Optional[IntegrationFeedback]
    ) -> List[ContextualizedQuestion]:
        """Apply human feedback to merge and discard questions."""
        if not feedback:
            return questions
        
        # Create a working copy
        processed = list(questions)
        
        # Discard questions (in reverse order to maintain indices)
        for idx in sorted(feedback.questions_to_discard, reverse=True):
            if 0 <= idx < len(processed):
                print(f"[{self.agent_name}] Discarding question {idx + 1}")
                processed.pop(idx)
        
        # Merge questions
        for idx1, idx2 in feedback.questions_to_merge:
            if 0 <= idx1 < len(processed) and 0 <= idx2 < len(processed):
                print(f"[{self.agent_name}] Merging questions {idx1 + 1} and {idx2 + 1}")
                q1, q2 = processed[idx1], processed[idx2]
                
                # Create merged question
                merged = ContextualizedQuestion(
                    question=f"{q1.question} (merged with: {q2.question})",
                    theoretical_framework=f"{q1.theoretical_framework} | {q2.theoretical_framework}",
                    literature_gaps=list(set(q1.literature_gaps + q2.literature_gaps)),
                    contribution=f"{q1.contribution} Additionally, {q2.contribution}",
                    related_theories=list(set(q1.related_theories + q2.related_theories)),
                    priority_score=max(q1.priority_score or 0, q2.priority_score or 0)
                )
                
                # Replace first, remove second
                processed[idx1] = merged
                if idx2 > idx1:
                    processed.pop(idx2)
                else:
                    processed.pop(idx2)
                    # Adjust idx1 if needed
        
        return processed
    
    def _prepare_contextualized_summary(self, questions: List[ContextualizedQuestion]) -> str:
        """Prepare a summary of contextualized questions for the prompt."""
        summary_parts = []
        for i, q in enumerate(questions, 1):
            summary = f"""
Question {i}: {q.question}
Theoretical Framework: {q.theoretical_framework}
Literature Gaps: {', '.join(q.literature_gaps)}
Contribution: {q.contribution}
Related Theories: {', '.join(q.related_theories)}
Priority Score: {q.priority_score}
"""
            summary_parts.append(summary)
        return "\n".join(summary_parts)


class IntegrationOrchestrator:
    """Orchestrates the integration process with two-round human feedback."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.contextualizer = Contextualizer(self.api_key)
        self.finalizer = Finalizer(self.api_key)
        
        self.round_results = {}
    
    def load_refinement_results(self, json_path: str) -> List[ResearchQuestion]:
        """Load refined questions from Stage 2."""
        print(f"\n[Orchestrator] Loading refinement results from {json_path}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract all questions from the refinement stage
        questions = []
        if 'all_questions' in data:
            for q_dict in data['all_questions']:
                questions.append(ResearchQuestion(**q_dict))
        elif 'questions' in data:
            for q_dict in data['questions']:
                questions.append(ResearchQuestion(**q_dict))
        
        print(f"[Orchestrator] Loaded {len(questions)} refined questions")
        return questions
    
    def run_integration_round(
        self,
        questions: List[ResearchQuestion],
        round_number: int,
        feedback: Optional[IntegrationFeedback] = None,
        max_final_questions: int = 5
    ) -> Tuple[List[ContextualizedQuestion], List[PrioritizedQuestion]]:
        """Run one round of integration (Contextualizer → Finalizer)."""
        
        print(f"\n{'='*70}")
        print(f"INTEGRATION ROUND {round_number}")
        print(f"{'='*70}")
        
        # Step 1: Contextualizer provides theoretical framing
        contextualized = self.contextualizer.contextualize_questions(
            questions=questions,
            feedback=feedback
        )
        
        # Step 2: Finalizer synthesizes and prioritizes
        prioritized = self.finalizer.synthesize_questions(
            contextualized_questions=contextualized,
            feedback=feedback,
            max_questions=max_final_questions
        )
        
        # Store results
        self.round_results[round_number] = {
            'contextualized': contextualized,
            'prioritized': prioritized
        }
        
        return contextualized, prioritized
    
    def print_contextualized_questions(self, round_number: int):
        """Print contextualized questions from a specific round."""
        if round_number not in self.round_results:
            print(f"No results for round {round_number}")
            return
        
        questions = self.round_results[round_number]['contextualized']
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_number} - CONTEXTUALIZED QUESTIONS")
        print(f"{'='*70}\n")
        
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q.question}")
            print(f"   Priority Score: {q.priority_score}")
            print(f"   Theoretical Framework: {q.theoretical_framework[:200]}...")
            print(f"   Literature Gaps: {', '.join(q.literature_gaps)}")
            print(f"   Related Theories: {', '.join(q.related_theories)}")
            print()
    
    def print_prioritized_questions(self, round_number: int):
        """Print prioritized questions from a specific round."""
        if round_number not in self.round_results:
            print(f"No results for round {round_number}")
            return
        
        questions = self.round_results[round_number]['prioritized']
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_number} - PRIORITIZED RESEARCH QUESTIONS")
        print(f"{'='*70}\n")
        
        for q in questions:
            print(f"RANK {q.priority_rank} (Score: {q.priority_score})")
            print(f"Question: {q.question}")
            print(f"Theoretical Framework: {q.theoretical_framework[:200]}...")
            print(f"Rationale: {q.rationale[:200]}...")
            print(f"Methodology: {', '.join(q.methodology)}")
            print(f"Expected Impact: {q.expected_impact[:150]}...")
            print(f"Feasibility: {q.feasibility[:150]}...")
            print()
    
    def collect_integration_feedback(
        self,
        round_number: int,
        num_questions: int
    ) -> IntegrationFeedback:
        """Collect human feedback for integration stage."""
        print(f"\n{'='*70}")
        print(f"INTEGRATION FEEDBACK - Round {round_number}")
        print(f"{'='*70}\n")
        
        print("Please provide feedback on the prioritized research questions.")
        print("Enter question numbers (1-indexed) for each action.")
        print("Press Enter to skip any field.\n")
        
        # Questions to merge
        merge_input = input("Enter pairs of questions to merge (e.g., '1,2 3,4'): ").strip()
        questions_to_merge = []
        if merge_input:
            pairs = merge_input.split()
            for pair in pairs:
                try:
                    idx1, idx2 = pair.split(',')
                    questions_to_merge.append((int(idx1) - 1, int(idx2) - 1))
                except:
                    print(f"Invalid pair format: {pair}")
        
        # Questions to discard
        discard_input = input("Enter questions to discard (comma-separated, e.g., '2,5'): ").strip()
        questions_to_discard = []
        if discard_input:
            try:
                questions_to_discard = [int(x.strip()) - 1 for x in discard_input.split(',')]
            except:
                print("Invalid discard format")
        
        # Questions to prioritize
        prioritize_input = input("Enter questions to prioritize (comma-separated, e.g., '1,3'): ").strip()
        questions_to_prioritize = []
        if prioritize_input:
            try:
                questions_to_prioritize = [int(x.strip()) - 1 for x in prioritize_input.split(',')]
            except:
                print("Invalid prioritize format")
        
        # Additional guidance
        guidance = input("Additional guidance for refinement: ").strip()
        
        feedback = IntegrationFeedback(
            round_number=round_number,
            questions_to_merge=questions_to_merge,
            questions_to_discard=questions_to_discard,
            questions_to_prioritize=questions_to_prioritize,
            additional_guidance=guidance
        )
        
        # Save feedback to JSON
        feedback_file = f"round{round_number}_integration_feedback.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback.model_dump(), f, indent=2)
        
        print(f"\n[Orchestrator] Feedback saved to {feedback_file}")
        
        return feedback
    
    def convert_prioritized_to_research_questions(
        self,
        prioritized: List[PrioritizedQuestion]
    ) -> List[ResearchQuestion]:
        """Convert prioritized questions back to ResearchQuestion format for next round."""
        questions = []
        for pq in prioritized:
            q = ResearchQuestion(
                question=pq.question,
                rationale=pq.rationale,
                methodology_hints=pq.methodology,
                related_concepts=[],  # Not applicable in this stage
                feasibility_score=pq.priority_score
            )
            questions.append(q)
        return questions
    
    def run_automated_integration(
        self,
        questions: List[ResearchQuestion],
        max_final_questions: int = 5
    ) -> List[PrioritizedQuestion]:
        """Run automated single-round integration without human feedback."""
        print(f"\n{'='*70}")
        print(f"AUTOMATED QUESTION INTEGRATION & PRIORITIZATION")
        print(f"{'='*70}")
        
        # Step 1: Contextualizer provides theoretical framing
        contextualized = self.contextualizer.contextualize_questions(
            questions=questions,
            feedback=None
        )
        
        # Step 2: Finalizer synthesizes and prioritizes
        prioritized = self.finalizer.synthesize_questions(
            contextualized_questions=contextualized,
            feedback=None,
            max_questions=max_final_questions
        )
        
        # Store results
        self.round_results[1] = {
            'contextualized': contextualized,
            'prioritized': prioritized
        }
        
        print(f"\n[Orchestrator] Generated {len(prioritized)} prioritized questions")
        
        return prioritized
    
    def print_prioritized_questions(self):
        """Print prioritized questions."""
        if not self.round_results:
            print("No results to print")
            return
        
        last_round = max(self.round_results.keys())
        questions = self.round_results[last_round]['prioritized']
        
        print(f"\n{'='*70}")
        print(f"PRIORITIZED RESEARCH QUESTIONS")
        print(f"{'='*70}\n")
        
        for q in questions:
            print(f"RANK {q.priority_rank} (Score: {q.priority_score})")
            print(f"Question: {q.question}")
            print(f"Theoretical Framework: {q.theoretical_framework[:200]}...")
            print(f"Rationale: {q.rationale[:200]}...")
            print(f"Methodology: {', '.join(q.methodology)}")
            print(f"Expected Impact: {q.expected_impact[:150]}...")
            print(f"Feasibility: {q.feasibility[:150]}...")
            print()
    
    def save_results(self, filename: str, round_number: Optional[int] = None):
        """Save integration results to JSON file."""
        if round_number:
            # Save specific round
            if round_number not in self.round_results:
                print(f"No results for round {round_number}")
                return
            
            data = {
                'round': round_number,
                'contextualized': [q.model_dump() for q in self.round_results[round_number]['contextualized']],
                'prioritized': [q.model_dump() for q in self.round_results[round_number]['prioritized']]
            }
            output_file = f"round{round_number}_{filename}"
        else:
            # Save all rounds
            data = {
                'by_round': {
                    str(rnd): {
                        'contextualized': [q.model_dump() for q in results['contextualized']],
                        'prioritized': [q.model_dump() for q in results['prioritized']]
                    }
                    for rnd, results in self.round_results.items()
                }
            }
            output_file = filename
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Orchestrator] Results saved to {output_file}")
    
    def save_final_questions(self, filename: str = "finalized_research_questions.json"):
        """Save the final prioritized questions from the last round."""
        if not self.round_results:
            print("No results to save")
            return
        
        # Get the last round's prioritized questions
        last_round = max(self.round_results.keys())
        final_questions = self.round_results[last_round]['prioritized']
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_rounds': len(self.round_results),
            'final_questions': [q.model_dump() for q in final_questions]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n[Orchestrator] Final questions saved to {filename}")
        
        # Also save as readable text
        text_filename = filename.replace('.json', '.txt')
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("FINALIZED RESEARCH QUESTIONS\n")
            f.write("="*70 + "\n\n")
            
            for q in final_questions:
                f.write(f"RANK {q.priority_rank} (Priority Score: {q.priority_score})\n")
                f.write(f"{'='*70}\n\n")
                f.write(f"QUESTION:\n{q.question}\n\n")
                f.write(f"THEORETICAL FRAMEWORK:\n{q.theoretical_framework}\n\n")
                f.write(f"RATIONALE:\n{q.rationale}\n\n")
                f.write(f"METHODOLOGY:\n{', '.join(q.methodology)}\n\n")
                f.write(f"EXPECTED IMPACT:\n{q.expected_impact}\n\n")
                f.write(f"FEASIBILITY:\n{q.feasibility}\n\n")
                f.write("-"*70 + "\n\n")
        
        print(f"[Orchestrator] Final questions also saved to {text_filename}")


def main():
    """Main function with two-round integration process."""
    
    # Configuration
    refinement_results = "refinement_results_all_rounds.json"  # Output from Stage 2
    max_final_questions = 5
    
    print("="*70)
    print("STARTING TWO-ROUND RESEARCH QUESTION INTEGRATION")
    print("="*70)
    
    # Initialize orchestrator
    orchestrator = IntegrationOrchestrator()
    
    # Load refined questions from Stage 2
    try:
        initial_questions = orchestrator.load_refinement_results(refinement_results)
    except FileNotFoundError:
        print(f"\nError: {refinement_results} not found!")
        print("Please run 2-RefinementStage.py first to generate refinement results.")
        return
    
    # ROUND 1: Initial contextualization and synthesis
    contextualized1, prioritized1 = orchestrator.run_integration_round(
        questions=initial_questions,
        round_number=1,
        feedback=None,
        max_final_questions=max_final_questions
    )
    
    # Display Round 1 results
    orchestrator.print_contextualized_questions(round_number=1)
    orchestrator.print_prioritized_questions(round_number=1)
    
    # Save Round 1 results
    orchestrator.save_results("integration_results.json", round_number=1)
    
    # Collect human feedback for Round 1
    feedback1 = orchestrator.collect_integration_feedback(
        round_number=1,
        num_questions=len(prioritized1)
    )
    
    # Convert prioritized questions back to ResearchQuestion format for Round 2
    questions_for_round2 = orchestrator.convert_prioritized_to_research_questions(prioritized1)
    
    # ROUND 2: Refined contextualization and synthesis based on feedback
    contextualized2, prioritized2 = orchestrator.run_integration_round(
        questions=questions_for_round2,
        round_number=2,
        feedback=feedback1,
        max_final_questions=max_final_questions
    )
    
    # Display Round 2 results
    orchestrator.print_contextualized_questions(round_number=2)
    orchestrator.print_prioritized_questions(round_number=2)
    
    # Save Round 2 results
    orchestrator.save_results("integration_results.json", round_number=2)
    
    # Save all results
    orchestrator.save_results("integration_results_all_rounds.json")
    
    # Save final questions
    orchestrator.save_final_questions("finalized_research_questions.json")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Initial questions from Stage 2: {len(initial_questions)}")
    print(f"Round 1 prioritized questions: {len(prioritized1)}")
    print(f"Round 2 prioritized questions: {len(prioritized2)}")
    
    print(f"\n{'='*70}")
    print("INTEGRATION PROCESS COMPLETE")
    print(f"{'='*70}")
    print("Files created:")
    print("  - round1_integration_results.json")
    print("  - round1_integration_feedback.json")
    print("  - round2_integration_results.json")
    print("  - integration_results_all_rounds.json")
    print("  - finalized_research_questions.json")
    print("  - finalized_research_questions.txt")
    
    print(f"\n{'='*70}")
    print("FINALIZED RESEARCH QUESTIONS")
    print(f"{'='*70}\n")
    
    for q in prioritized2:
        print(f"RANK {q.priority_rank}: {q.question}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
