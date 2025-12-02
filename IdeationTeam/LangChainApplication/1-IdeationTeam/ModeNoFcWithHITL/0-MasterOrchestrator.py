"""
Master Orchestrator for Research Question Generation with Human-in-the-Loop (No FireCrawl)
This script runs all three stages sequentially with human feedback between rounds.

Pipeline:
1. Sourcing Stage: Gather literature from research keywords (2 rounds with feedback)
2. Refinement Stage: Generate research concepts and questions (2 rounds with feedback)
3. Integration Stage: Contextualize and prioritize final questions (2 rounds with feedback)

Input: Research keywords or basic research ideas
Output: Finalized, prioritized research questions
Human Interaction: Feedback collection after each round in each stage
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Main orchestrator that runs all three stages with human-in-the-loop."""
    
    # Change to script directory to ensure outputs are saved there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # ========== CONFIGURATION ==========
    research_topic = input("Enter your research topic or keywords: ").strip()
    if not research_topic:
        research_topic = "Agent-based modeling in macroeconomics and monetary policy"
        print(f"Using default topic: {research_topic}")
    
    print("\n" + "="*70)
    print("RESEARCH QUESTION GENERATION PIPELINE WITH HUMAN-IN-THE-LOOP")
    print("Mode: No FireCrawl, With Human Feedback")
    print("="*70)
    print(f"Research Topic: {research_topic}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # ========== STAGE 1: LITERATURE SOURCING (2 ROUNDS) ==========
    print("\n" + "="*70)
    print("STAGE 1: LITERATURE SOURCING WITH HUMAN FEEDBACK")
    print("="*70)
    
    try:
        # Import and run Stage 1
        from importlib import import_module
        stage1 = import_module('1-SourcingStage')
        
        # Initialize orchestrator
        orchestrator1 = stage1.MultiAgentOrchestrator()
        
        # ROUND 1: Initial search
        print("\n[Stage 1] Starting Round 1...")
        round1_results = orchestrator1.run_search_round(
            research_topic=research_topic,
            round_number=1,
            feedback=None,
            max_results_per_agent=8
        )
        
        # Display and save Round 1 results
        orchestrator1.print_summary(round_number=1, top_n=15)
        orchestrator1.save_results("literature_results.csv", round_number=1)
        
        # Collect human feedback
        print("\n" + "="*70)
        print("Please review the Round 1 results above.")
        print("="*70)
        feedback1 = orchestrator1.collect_human_feedback(round_number=1)
        
        # ROUND 2: Refined search with feedback
        print("\n[Stage 1] Starting Round 2 with your feedback...")
        round2_results = orchestrator1.run_search_round(
            research_topic=research_topic,
            round_number=2,
            feedback=feedback1,
            max_results_per_agent=8
        )
        
        # Display and save Round 2 results
        orchestrator1.print_summary(round_number=2, top_n=15)
        orchestrator1.save_results("literature_results.csv", round_number=2)
        
        # Save all results
        literature_file = "literature_results_all_rounds.csv"
        orchestrator1.save_results(literature_file)
        
        print(f"\n[Stage 1] Complete:")
        print(f"  Round 1: {len(round1_results)} papers")
        print(f"  Round 2: {len(round2_results)} papers")
        print(f"  Total unique: {len(orchestrator1.all_results)} papers")
        print(f"  Results saved to: {literature_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 1-SourcingStage.py is properly configured.")
        return
    
    # ========== STAGE 2: REFINEMENT (2 ROUNDS) ==========
    print("\n" + "="*70)
    print("STAGE 2: RESEARCH QUESTION REFINEMENT WITH HUMAN FEEDBACK")
    print("="*70)
    
    try:
        # Import and run Stage 2
        stage2 = import_module('2-RefinementStage')
        
        # Initialize orchestrator
        orchestrator2 = stage2.RefinementOrchestrator()
        literature_df = orchestrator2.load_literature(literature_file)
        
        # ROUND 1: Initial concept generation and question formulation
        print("\n[Stage 2] Starting Round 1...")
        concepts1, questions1 = orchestrator2.run_refinement_round(
            literature_df=literature_df,
            round_number=1,
            feedback=None,
            num_concepts=8,
            num_questions=6
        )
        
        # Display Round 1 results
        orchestrator2.print_concepts(round_number=1, top_n=8)
        orchestrator2.print_questions(round_number=1, top_n=6)
        orchestrator2.save_results("refinement_results.json", round_number=1)
        
        # Collect human feedback
        print("\n" + "="*70)
        print("Please review the Round 1 concepts and questions above.")
        print("="*70)
        feedback2 = orchestrator2.collect_human_feedback(round_number=1)
        
        # ROUND 2: Refined concepts and questions based on feedback
        print("\n[Stage 2] Starting Round 2 with your feedback...")
        concepts2, questions2 = orchestrator2.run_refinement_round(
            literature_df=literature_df,
            round_number=2,
            feedback=feedback2,
            num_concepts=8,
            num_questions=6
        )
        
        # Display Round 2 results
        orchestrator2.print_concepts(round_number=2, top_n=8)
        orchestrator2.print_questions(round_number=2, top_n=6)
        orchestrator2.save_results("refinement_results.json", round_number=2)
        
        # Save all results
        refinement_file = "refinement_results_all_rounds.json"
        orchestrator2.save_results(refinement_file)
        
        print(f"\n[Stage 2] Complete:")
        print(f"  Round 1: {len(concepts1)} concepts, {len(questions1)} questions")
        print(f"  Round 2: {len(concepts2)} concepts, {len(questions2)} questions")
        print(f"  Total: {len(orchestrator2.all_concepts)} concepts, {len(orchestrator2.all_questions)} questions")
        print(f"  Results saved to: {refinement_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 2-RefinementStage.py is properly configured.")
        return
    
    # ========== STAGE 3: INTEGRATION (2 ROUNDS) ==========
    print("\n" + "="*70)
    print("STAGE 3: QUESTION INTEGRATION & PRIORITIZATION WITH HUMAN FEEDBACK")
    print("="*70)
    
    try:
        # Import and run Stage 3
        stage3 = import_module('3-IntegrationStage')
        
        # Initialize orchestrator
        orchestrator3 = stage3.IntegrationOrchestrator()
        initial_questions = orchestrator3.load_refinement_results(refinement_file)
        
        # ROUND 1: Initial contextualization and synthesis
        print("\n[Stage 3] Starting Round 1...")
        contextualized1, prioritized1 = orchestrator3.run_integration_round(
            questions=initial_questions,
            round_number=1,
            feedback=None,
            max_final_questions=5
        )
        
        # Display Round 1 results
        orchestrator3.print_contextualized_questions(round_number=1)
        orchestrator3.print_prioritized_questions(round_number=1)
        orchestrator3.save_results("integration_results.json", round_number=1)
        
        # Collect human feedback
        print("\n" + "="*70)
        print("Please review the Round 1 prioritized questions above.")
        print("="*70)
        feedback3 = orchestrator3.collect_integration_feedback(
            round_number=1,
            num_questions=len(prioritized1)
        )
        
        # Convert prioritized questions back to ResearchQuestion format for Round 2
        questions_for_round2 = orchestrator3.convert_prioritized_to_research_questions(prioritized1)
        
        # ROUND 2: Refined contextualization and synthesis based on feedback
        print("\n[Stage 3] Starting Round 2 with your feedback...")
        contextualized2, prioritized2 = orchestrator3.run_integration_round(
            questions=questions_for_round2,
            round_number=2,
            feedback=feedback3,
            max_final_questions=5
        )
        
        # Display Round 2 results
        orchestrator3.print_contextualized_questions(round_number=2)
        orchestrator3.print_prioritized_questions(round_number=2)
        orchestrator3.save_results("integration_results.json", round_number=2)
        
        # Save all results
        orchestrator3.save_results("integration_results_all_rounds.json")
        
        # Save final questions
        final_file = "finalized_research_questions.json"
        orchestrator3.save_final_questions(final_file)
        
        print(f"\n[Stage 3] Complete:")
        print(f"  Round 1: {len(prioritized1)} prioritized questions")
        print(f"  Round 2: {len(prioritized2)} prioritized questions")
        print(f"  Final results saved to: {final_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 3 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 3-IntegrationStage.py is properly configured.")
        return
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Research Topic: {research_topic}")
    print(f"Literature Papers: {len(orchestrator1.all_results)}")
    print(f"Research Concepts: {len(orchestrator2.all_concepts)}")
    print(f"Research Questions: {len(orchestrator2.all_questions)}")
    print(f"Final Prioritized Questions: {len(prioritized2)}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    print("\n" + "="*70)
    print("FINALIZED RESEARCH QUESTIONS")
    print("="*70 + "\n")
    
    for q in prioritized2:
        print(f"RANK {q.priority_rank}: {q.question}")
        print(f"   Priority Score: {q.priority_score}")
        print()
    
    print("="*70)
    print("All results saved in current directory.")
    print("View detailed questions in: finalized_research_questions.txt")
    print("="*70)


if __name__ == "__main__":
    main()
