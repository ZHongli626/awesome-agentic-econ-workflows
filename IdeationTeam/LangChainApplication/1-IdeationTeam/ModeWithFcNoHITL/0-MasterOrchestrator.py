"""
Master Orchestrator for Automated Research Question Generation with Firecrawl (No Human-in-the-Loop)
This script runs all three stages sequentially in a fully automated pipeline with Firecrawl web scraping.

Pipeline:
1. Sourcing Stage: Gather literature from APIs + Firecrawl web scraping (single automated round)
2. Refinement Stage: Generate research concepts and questions (single automated round)
3. Integration Stage: Contextualize and prioritize final questions (single automated round)

Input: Research keywords or basic research ideas
Output: Finalized, prioritized research questions
Firecrawl: TrendSurfer and TopicCrawler use Firecrawl for web scraping
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Clear any cached modules to avoid import errors
for module_name in ['1-SourcingStage', '2-RefinementStage', '3-IntegrationStage']:
    if module_name in sys.modules:
        del sys.modules[module_name]


def main():
    """Main orchestrator that runs all three stages automatically with Firecrawl."""
    
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
    print("AUTOMATED RESEARCH QUESTION GENERATION PIPELINE WITH FIRECRAWL")
    print("Mode: With FireCrawl, No Human-in-the-Loop")
    print("="*70)
    print(f"Research Topic: {research_topic}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Check for Firecrawl API key
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    if firecrawl_key:
        print("✅ Firecrawl API key found - web scraping enabled")
    else:
        print("⚠️  Firecrawl API key not found - web scraping disabled")
    print()
    
    # ========== STAGE 1: LITERATURE SOURCING ==========
    print("\n" + "="*70)
    print("STAGE 1: AUTOMATED LITERATURE SOURCING WITH FIRECRAWL")
    print("="*70)
    
    try:
        # Import and run Stage 1
        from importlib import import_module
        stage1 = import_module('1-SourcingStage')
        
        # Initialize orchestrator with Firecrawl support
        orchestrator1 = stage1.MultiAgentOrchestrator()
        
        # Run automated search (single round)
        print("\n[Stage 1] Running automated search with Firecrawl...")
        results = orchestrator1.run_automated_search(
            research_topic=research_topic,
            max_results_per_agent=8
        )
        
        # Display and save results
        orchestrator1.print_summary(top_n=15)
        literature_file = "literature_results_automated.csv"
        orchestrator1.save_results(literature_file)
        
        print(f"\n[Stage 1] Complete:")
        print(f"  Total papers found: {len(results)}")
        print(f"  Results saved to: {literature_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 1-SourcingStage.py is properly configured.")
        return
    
    # ========== STAGE 2: REFINEMENT ==========
    print("\n" + "="*70)
    print("STAGE 2: AUTOMATED RESEARCH QUESTION REFINEMENT")
    print("="*70)
    
    try:
        # Import and run Stage 2
        stage2 = import_module('2-RefinementStage')
        
        # Initialize orchestrator
        orchestrator2 = stage2.RefinementOrchestrator()
        literature_df = orchestrator2.load_literature(literature_file)
        
        # Run automated refinement (single round)
        print("\n[Stage 2] Running automated refinement...")
        questions = orchestrator2.run_automated_refinement(
            literature_df=literature_df,
            num_concepts=10,
            num_questions=8
        )
        
        # Display and save results
        orchestrator2.print_concepts(top_n=10)
        orchestrator2.print_questions(top_n=8)
        
        refinement_file = "refinement_results_automated.json"
        orchestrator2.save_results(refinement_file)
        
        print(f"\n[Stage 2] Complete:")
        print(f"  Concepts generated: {len(orchestrator2.all_concepts)}")
        print(f"  Questions generated: {len(questions)}")
        print(f"  Results saved to: {refinement_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 2-RefinementStage.py is properly configured.")
        return
    
    # ========== STAGE 3: INTEGRATION ==========
    print("\n" + "="*70)
    print("STAGE 3: AUTOMATED QUESTION INTEGRATION & PRIORITIZATION")
    print("="*70)
    
    try:
        # Import and run Stage 3
        stage3 = import_module('3-IntegrationStage')
        
        # Initialize orchestrator
        orchestrator3 = stage3.IntegrationOrchestrator()
        
        # Run automated integration (single round)
        print("\n[Stage 3] Running automated integration...")
        prioritized_questions = orchestrator3.run_automated_integration(
            questions=questions,
            max_final_questions=5
        )
        
        # Display and save results
        orchestrator3.print_prioritized_questions()
        
        integration_file = "integration_results_automated.json"
        orchestrator3.save_results(integration_file)
        
        final_file = "finalized_research_questions_automated.json"
        orchestrator3.save_final_questions(final_file)
        
        print(f"\n[Stage 3] Complete:")
        print(f"  Final prioritized questions: {len(prioritized_questions)}")
        print(f"  Results saved to: {final_file}")
        
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
    print(f"Final Prioritized Questions: {len(prioritized_questions)}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    print("\n" + "="*70)
    print("FINALIZED RESEARCH QUESTIONS")
    print("="*70 + "\n")
    
    for q in prioritized_questions:
        print(f"RANK {q.priority_rank}: {q.question}")
        print(f"   Priority Score: {q.priority_score}")
        print()
    
    print("="*70)
    print("All results saved in current directory.")
    print("View detailed questions in: finalized_research_questions_automated.txt")
    print("="*70)


if __name__ == "__main__":
    main()
