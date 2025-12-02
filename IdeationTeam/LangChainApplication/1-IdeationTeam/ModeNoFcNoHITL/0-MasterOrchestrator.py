"""
Master Orchestrator for Automated Research Question Generation (No Human-in-the-Loop, No FireCrawl)
This script runs all three stages sequentially in a fully automated pipeline.

Pipeline:
1. Sourcing Stage: Gather literature from research keywords
2. Refinement Stage: Generate research concepts and questions
3. Integration Stage: Contextualize and prioritize final questions

Input: Research keywords or basic research ideas
Output: Finalized, prioritized research questions
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Main orchestrator that runs all three stages automatically."""
    
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
    print("AUTOMATED RESEARCH QUESTION GENERATION PIPELINE")
    print("Mode: No FireCrawl, No Human-in-the-Loop")
    print("="*70)
    print(f"Research Topic: {research_topic}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # ========== STAGE 1: LITERATURE SOURCING ==========
    print("\n" + "="*70)
    print("STAGE 1: LITERATURE SOURCING")
    print("="*70)
    
    try:
        # Import and run Stage 1
        from importlib import import_module
        stage1 = import_module('1-SourcingStage')
        
        # Run automated sourcing
        orchestrator1 = stage1.MultiAgentOrchestrator()
        literature_results = orchestrator1.run_automated_search(
            research_topic=research_topic,
            max_results_per_agent=15
        )
        
        # Save results
        literature_file = "literature_results_automated.csv"
        orchestrator1.save_results(literature_file)
        
        print(f"\n[Stage 1] Complete: {len(literature_results)} papers found")
        print(f"[Stage 1] Results saved to: {literature_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 1 failed: {e}")
        print("Please ensure 1-SourcingStage.py is properly configured.")
        return
    
    # ========== STAGE 2: REFINEMENT ==========
    print("\n" + "="*70)
    print("STAGE 2: RESEARCH QUESTION REFINEMENT")
    print("="*70)
    
    try:
        # Import and run Stage 2
        stage2 = import_module('2-RefinementStage')
        
        # Run automated refinement
        orchestrator2 = stage2.RefinementOrchestrator()
        literature_df = orchestrator2.load_literature(literature_file)
        
        refined_questions = orchestrator2.run_automated_refinement(
            literature_df=literature_df,
            num_concepts=10,
            num_questions=8
        )
        
        # Save results
        refinement_file = "refinement_results_automated.json"
        orchestrator2.save_results(refinement_file)
        
        print(f"\n[Stage 2] Complete: {len(refined_questions)} questions generated")
        print(f"[Stage 2] Results saved to: {refinement_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 2 failed: {e}")
        print("Please ensure 2-RefinementStage.py is properly configured.")
        return
    
    # ========== STAGE 3: INTEGRATION ==========
    print("\n" + "="*70)
    print("STAGE 3: QUESTION INTEGRATION & PRIORITIZATION")
    print("="*70)
    
    try:
        # Import and run Stage 3
        stage3 = import_module('3-IntegrationStage')
        
        # Run automated integration
        orchestrator3 = stage3.IntegrationOrchestrator()
        initial_questions = orchestrator3.load_refinement_results(refinement_file)
        
        final_questions = orchestrator3.run_automated_integration(
            questions=initial_questions,
            max_final_questions=5
        )
        
        # Save final results
        final_file = "finalized_research_questions_automated.json"
        orchestrator3.save_final_questions(final_file)
        
        print(f"\n[Stage 3] Complete: {len(final_questions)} final questions")
        print(f"[Stage 3] Results saved to: {final_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 3 failed: {e}")
        print("Please ensure 3-IntegrationStage.py is properly configured.")
        return
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Research Topic: {research_topic}")
    print(f"Literature Papers: {len(literature_results)}")
    print(f"Refined Questions: {len(refined_questions)}")
    print(f"Final Questions: {len(final_questions)}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    print("\n" + "="*70)
    print("FINALIZED RESEARCH QUESTIONS")
    print("="*70 + "\n")
    
    for i, q in enumerate(final_questions, 1):
        print(f"{i}. {q.question}")
        print(f"   Priority Score: {q.priority_score}")
        print()
    
    print("="*70)
    print(f"\nAll results saved in current directory.")
    print(f"View detailed questions in: {final_file.replace('.json', '.txt')}")
    print("="*70)


if __name__ == "__main__":
    main()
