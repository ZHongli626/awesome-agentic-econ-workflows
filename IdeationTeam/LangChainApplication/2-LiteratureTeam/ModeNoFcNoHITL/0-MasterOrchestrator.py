"""
Master Orchestrator for Literature Team Pipeline (No Firecrawl, No HITL)
This script runs all three stages sequentially in a fully automated pipeline.

Pipeline:
1. Literature Gathering Stage: Retrieve and analyze literature based on research questions
2. Gap Detection Stage: Identify research gaps and construct knowledge graph
3. Synthesis Stage: Generate literature review and research plan

Input: Research questions (JSON from IdeationTeam or manual input)
Output: Literature review, research plan, and bibliography
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
    print("="*70)
    print("LITERATURE TEAM PIPELINE")
    print("Mode: No FireCrawl, No Human-in-the-Loop")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Load research questions
    questions_file = input("Enter path to research questions JSON (or press Enter to create sample): ").strip()
    
    if questions_file and os.path.exists(questions_file):
        print(f"\nLoading research questions from: {questions_file}")
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            # Handle different formats
            if isinstance(questions_data, list):
                research_questions = questions_data
            elif isinstance(questions_data, dict):
                # Try to extract questions from various possible structures
                if 'questions' in questions_data:
                    research_questions = questions_data['questions']
                elif 'prioritized' in questions_data:
                    research_questions = questions_data['prioritized']
                else:
                    research_questions = [questions_data]
            else:
                raise ValueError("Unexpected JSON format")
            
            print(f"Loaded {len(research_questions)} research questions")
        except Exception as e:
            print(f"Error loading questions: {e}")
            print("Creating sample questions instead...")
            research_questions = create_sample_questions()
    else:
        print("\nCreating sample research questions...")
        research_questions = create_sample_questions()
    
    print(f"\nResearch Questions to Process:")
    for i, q in enumerate(research_questions[:5], 1):
        question_text = q.get('question', str(q)) if isinstance(q, dict) else str(q)
        print(f"  {i}. {question_text[:80]}...")
    if len(research_questions) > 5:
        print(f"  ... and {len(research_questions) - 5} more")
    print()
    
    # ========== STAGE 1: LITERATURE GATHERING ==========
    print("\n" + "="*70)
    print("STAGE 1: LITERATURE GATHERING")
    print("="*70)
    
    try:
        # Import and run Stage 1
        from importlib import import_module
        stage1 = import_module('1-LiteratureGatheringStage')
        
        # Convert questions to proper format
        formatted_questions = []
        for q in research_questions:
            if isinstance(q, dict):
                formatted_questions.append(stage1.ResearchQuestion(**q))
            else:
                # Create a basic question object
                formatted_questions.append(stage1.ResearchQuestion(
                    question=str(q),
                    priority_rank=1,
                    priority_score=0.8
                ))
        
        # Run literature gathering
        orchestrator1 = stage1.LiteratureGatheringOrchestrator()
        literature_batch = orchestrator1.run_gathering_pipeline(
            research_questions=formatted_questions,
            max_papers_per_question=15
        )
        
        # Save results
        batch_file = "literature_batch.json"
        csv_file = "literature_items.csv"
        orchestrator1.save_literature_batch(batch_file)
        orchestrator1.save_literature_csv(csv_file)
        
        print(f"\n[Stage 1] Complete:")
        print(f"  - Literature Items: {len(literature_batch.literature_items)}")
        print(f"  - Trends Identified: {len(literature_batch.trend_analyses)}")
        print(f"  - Citations Created: {len(literature_batch.citations)}")
        print(f"  - Insights Extracted: {len(literature_batch.insights)}")
        print(f"  - Results saved to: {batch_file}, {csv_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 1-LiteratureGatheringStage.py is properly configured.")
        return
    
    # ========== STAGE 2: GAP DETECTION ==========
    print("\n" + "="*70)
    print("STAGE 2: GAP DETECTION & KNOWLEDGE GRAPH CONSTRUCTION")
    print("="*70)
    
    try:
        # Import and run Stage 2
        stage2 = import_module('2-GapDetectionStage')
        
        # Load literature batch
        with open(batch_file, 'r', encoding='utf-8') as f:
            literature_batch_dict = json.load(f)
        
        # Run gap detection
        orchestrator2 = stage2.GapDetectionOrchestrator()
        gap_analysis = orchestrator2.run_gap_detection_pipeline(literature_batch_dict)
        
        # Save results
        gap_file = "gap_analysis_results.json"
        gaps_csv = "research_gaps.csv"
        graph_file = "knowledge_graph.json"
        orchestrator2.save_gap_analysis(gap_file)
        orchestrator2.save_gaps_csv(gaps_csv)
        orchestrator2.save_graph_json(graph_file)
        
        print(f"\n[Stage 2] Complete:")
        print(f"  - Papers Analyzed: {len(gap_analysis.paper_structures)}")
        print(f"  - Gaps Identified: {len(gap_analysis.research_gaps)}")
        print(f"  - Graph Nodes: {len(gap_analysis.knowledge_graph.nodes)}")
        print(f"  - Graph Edges: {len(gap_analysis.knowledge_graph.edges)}")
        print(f"  - Results saved to: {gap_file}, {gaps_csv}, {graph_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 2-GapDetectionStage.py is properly configured.")
        return
    
    # ========== STAGE 3: SYNTHESIS ==========
    print("\n" + "="*70)
    print("STAGE 3: LITERATURE REVIEW & RESEARCH PLAN SYNTHESIS")
    print("="*70)
    
    try:
        # Import and run Stage 3
        stage3 = import_module('3-SynthesisStage')
        
        # Load gap analysis
        with open(gap_file, 'r', encoding='utf-8') as f:
            gap_analysis_dict = json.load(f)
        
        # Run synthesis
        orchestrator3 = stage3.SynthesisOrchestrator()
        synthesis_result = orchestrator3.run_synthesis_pipeline(gap_analysis_dict)
        
        # Save results
        synthesis_file = "synthesis_results.json"
        review_file = "literature_review.txt"
        plan_file = "research_plan.txt"
        bib_file = "bibliography.txt"
        orchestrator3.save_synthesis_result(synthesis_file)
        orchestrator3.save_literature_review(review_file)
        orchestrator3.save_research_plan(plan_file)
        orchestrator3.save_bibliography(bib_file)
        
        print(f"\n[Stage 3] Complete:")
        print(f"  - Review Sections: {len(synthesis_result.literature_review.sections)}")
        print(f"  - Research Objectives: {len(synthesis_result.research_plan.research_objectives)}")
        print(f"  - Bibliography Entries: {synthesis_result.bibliography.total_references}")
        print(f"  - Results saved to: {synthesis_file}, {review_file}, {plan_file}, {bib_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 3 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 3-SynthesisStage.py is properly configured.")
        return
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Research Questions Processed: {len(research_questions)}")
    print(f"Literature Items Gathered: {len(literature_batch.literature_items)}")
    print(f"Research Gaps Identified: {len(gap_analysis.research_gaps)}")
    print(f"Literature Review Sections: {len(synthesis_result.literature_review.sections)}")
    print(f"Research Objectives: {len(synthesis_result.research_plan.research_objectives)}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Print key outputs
    print("\n" + "="*70)
    print("KEY OUTPUTS")
    print("="*70)
    
    print(f"\nLiterature Review Title:")
    print(f"  {synthesis_result.literature_review.title}")
    
    print(f"\nTop 5 Research Gaps (by severity):")
    for i, gap in enumerate(gap_analysis.research_gaps[:5], 1):
        print(f"  {i}. [{gap.gap_type.upper()}] {gap.gap_title}")
        print(f"     Severity: {gap.severity} | Addressability: {gap.addressability}")
    
    print(f"\nResearch Objectives:")
    for obj in synthesis_result.research_plan.research_objectives:
        print(f"  - {obj.objective_title} (Priority: {obj.priority})")
    
    print("\n" + "="*70)
    print("OUTPUT FILES SUMMARY")
    print("="*70)
    print("\nStage 1 - Literature Gathering:")
    print(f"  - {batch_file} (complete literature batch)")
    print(f"  - {csv_file} (literature items CSV)")
    
    print("\nStage 2 - Gap Detection:")
    print(f"  - {gap_file} (complete gap analysis)")
    print(f"  - {gaps_csv} (research gaps CSV)")
    print(f"  - {graph_file} (knowledge graph)")
    
    print("\nStage 3 - Synthesis:")
    print(f"  - {synthesis_file} (complete synthesis)")
    print(f"  - {review_file} (formatted literature review)")
    print(f"  - {plan_file} (formatted research plan)")
    print(f"  - {bib_file} (formatted bibliography)")
    
    print("\n" + "="*70)
    print("All results saved in current directory.")
    print(f"View the literature review in: {review_file}")
    print(f"View the research plan in: {plan_file}")
    print("="*70)


def create_sample_questions():
    """Create sample research questions for demonstration."""
    return [
        {
            "question": "How do agent-based models improve monetary policy analysis compared to traditional DSGE models?",
            "priority_rank": 1,
            "priority_score": 0.95,
            "theoretical_framework": "Agent-based computational economics",
            "methodology": ["Agent-based modeling", "Comparative analysis", "Simulation"]
        },
        {
            "question": "What are the computational challenges in scaling agent-based macroeconomic models to realistic population sizes?",
            "priority_rank": 2,
            "priority_score": 0.88,
            "theoretical_framework": "Computational economics",
            "methodology": ["High-performance computing", "Parallel processing", "Algorithmic optimization"]
        },
        {
            "question": "How can heterogeneous agent models capture the distributional effects of monetary policy?",
            "priority_rank": 3,
            "priority_score": 0.82,
            "theoretical_framework": "Heterogeneous agent macroeconomics",
            "methodology": ["HANK models", "Distributional analysis", "Empirical validation"]
        }
    ]


if __name__ == "__main__":
    main()
