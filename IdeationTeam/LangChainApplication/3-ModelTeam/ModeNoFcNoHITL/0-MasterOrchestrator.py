"""
Master Orchestrator for Automated Model Development (No Human-in-the-Loop, No FireCrawl)
This script runs all three ModelTeam stages sequentially in a fully automated pipeline.

Pipeline:
1. Theory Stage: Develop theoretical frameworks from research questions and literature
2. Model Design Stage: Design computational models based on theoretical frameworks
3. Calibration Stage: Define calibration strategy and parameter identification

Input: Research questions + Literature review from LiteratureTeam
Output: Complete model specification with theoretical framework, design, and calibration strategy
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Main orchestrator that runs all three ModelTeam stages automatically."""
    
    # Change to script directory to ensure outputs are saved there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # ========== CONFIGURATION ==========
    print("\n" + "="*70)
    print("AUTOMATED MODEL DEVELOPMENT PIPELINE")
    print("Mode: No FireCrawl, No Human-in-the-Loop")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Check for required input files
    research_questions_file = input("Enter path to research questions JSON file (or press Enter for default): ").strip()
    if not research_questions_file:
        # Try to find default files from IdeationTeam or LiteratureTeam
        default_paths = [
            "../../1-IdeationTeam/ModeNoFcNoHITL/finalized_research_questions_automated.json",
            "../../../1-IdeationTeam/ModeNoFcNoHITL/finalized_research_questions_automated.json",
            "research_questions.json"
        ]
        for path in default_paths:
            if os.path.exists(path):
                research_questions_file = path
                print(f"Using default research questions file: {research_questions_file}")
                break
        
        if not research_questions_file or not os.path.exists(research_questions_file):
            print("\n[WARNING] No research questions file found.")
            print("Please provide a valid path to research questions JSON file.")
            return
    
    literature_file = input("Enter path to literature review file (or press Enter to skip): ").strip()
    if not literature_file:
        # Try to find default files from LiteratureTeam
        default_lit_paths = [
            "../../2-LiteratureTeam/ModeNoFcNoHITL/literature_review.txt",
            "../../../2-LiteratureTeam/ModeNoFcNoHITL/literature_review.txt",
            "literature_review.txt"
        ]
        for path in default_lit_paths:
            if os.path.exists(path):
                literature_file = path
                print(f"Using default literature file: {literature_file}")
                break
    
    print("\n" + "="*70)
    print("INPUT FILES")
    print("="*70)
    print(f"Research Questions: {research_questions_file}")
    print(f"Literature Review: {literature_file if literature_file else 'Not provided'}")
    print("="*70 + "\n")
    
    # ========== STAGE 1: THEORY DEVELOPMENT ==========
    print("\n" + "="*70)
    print("STAGE 1: THEORETICAL FRAMEWORK DEVELOPMENT")
    print("="*70)
    
    try:
        # Import and run Stage 1
        from importlib import import_module
        stage1 = import_module('1-TheoryStage')
        
        # Run theory development
        orchestrator1 = stage1.TheoryStageOrchestrator()
        
        # Load research questions
        with open(research_questions_file, 'r', encoding='utf-8') as f:
            rq_data = json.load(f)
        
        # Extract questions (handle different JSON structures)
        if isinstance(rq_data, dict):
            questions = rq_data.get('questions', rq_data.get('final_questions', []))
        else:
            questions = rq_data
        
        # Load literature if available
        literature_data = None
        if literature_file and os.path.exists(literature_file):
            if literature_file.endswith('.json'):
                with open(literature_file, 'r', encoding='utf-8') as f:
                    literature_data = json.load(f)
            elif literature_file.endswith('.txt'):
                with open(literature_file, 'r', encoding='utf-8') as f:
                    literature_text = f.read()
                    literature_data = {"review_text": literature_text}
        
        # Run theory stage
        theory_output = orchestrator1.run_theory_pipeline(
            research_questions=questions,
            literature_batch=literature_data or {}
        )
        
        # Save results
        theory_file = "theory_output.json"
        orchestrator1.save_theory_output(theory_file)
        
        frameworks = theory_output.theoretical_frameworks if theory_output else []
        
        print(f"\n[Stage 1] Complete: {len(frameworks)} theoretical frameworks developed")
        print(f"[Stage 1] Results saved to: {theory_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 1-TheoryStage.py is properly configured.")
        return
    
    # ========== STAGE 2: MODEL DESIGN ==========
    print("\n" + "="*70)
    print("STAGE 2: COMPUTATIONAL MODEL DESIGN")
    print("="*70)
    
    try:
        # Import and run Stage 2
        stage2 = import_module('2-ModelDesignStage')
        
        # Run model design
        orchestrator2 = stage2.ModelDesignOrchestrator()
        
        # Load theory output
        with open(theory_file, 'r', encoding='utf-8') as f:
            theory_data = json.load(f)
        
        # Run model design stage
        design_output = orchestrator2.run_design_pipeline(
            theory_output=theory_data
        )
        
        # Save results
        design_file = "model_design_output.json"
        orchestrator2.save_design_output(design_file)
        
        models = design_output.formal_models if design_output else []
        
        print(f"\n[Stage 2] Complete: {len(models)} computational models designed")
        print(f"[Stage 2] Results saved to: {design_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 2-ModelDesignStage.py is properly configured.")
        return
    
    # ========== STAGE 3: CALIBRATION STRATEGY ==========
    print("\n" + "="*70)
    print("STAGE 3: CALIBRATION & PARAMETER IDENTIFICATION")
    print("="*70)
    
    try:
        # Import and run Stage 3
        stage3 = import_module('3-CalibrationStage')
        
        # Run calibration strategy
        orchestrator3 = stage3.CalibrationOrchestrator()
        
        # Load model design output
        with open(design_file, 'r', encoding='utf-8') as f:
            design_data = json.load(f)
        
        # Run calibration stage
        calibration_output = orchestrator3.run_calibration_pipeline(
            model_design_output=design_data
        )
        
        # Save final results
        calibration_file = "calibration_output.json"
        orchestrator3.save_calibration_output(calibration_file)
        
        calibration_strategies = calibration_output.calibrated_models if calibration_output else []
        
        print(f"\n[Stage 3] Complete: {len(calibration_strategies)} calibration strategies developed")
        print(f"[Stage 3] Results saved to: {calibration_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 3 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 3-CalibrationStage.py is properly configured.")
        return
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Research Questions: {len(questions) if questions else 0}")
    print(f"Theoretical Frameworks: {len(frameworks) if frameworks else 0}")
    print(f"Model Designs: {len(models) if models else 0}")
    print(f"Calibrated Models: {len(calibration_strategies) if calibration_strategies else 0}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # ========== OUTPUT SUMMARY ==========
    print("\n" + "="*70)
    print("OUTPUT FILES SUMMARY")
    print("="*70)
    print(f"1. Theory Output: {theory_file}")
    print(f"2. Model Design Output: {design_file}")
    print(f"3. Calibration Output: {calibration_file}")
    print("="*70)
    
    # Display brief summary of first framework
    if frameworks and len(frameworks) > 0:
        print("\n" + "="*70)
        print("SAMPLE OUTPUT: First Theoretical Framework")
        print("="*70)
        first_framework = frameworks[0] if isinstance(frameworks[0], dict) else frameworks[0].__dict__
        print(f"\nTitle: {first_framework.get('framework_title', 'N/A')}")
        print(f"Research Question: {first_framework.get('research_question', 'N/A')}")
        print(f"Theoretical Approach: {first_framework.get('theoretical_approach', 'N/A')}")
        print(f"\nAssumptions: {len(first_framework.get('assumptions', []))}")
        print(f"Conceptual Components: {len(first_framework.get('conceptual_components', []))}")
        print(f"Mathematical Formulations: {len(first_framework.get('mathematical_formulations', []))}")
        print("="*70)
    
    print(f"\n✅ All ModelTeam stages completed successfully!")
    print(f"📁 All results saved in: {os.getcwd()}")
    print("="*70)


if __name__ == "__main__":
    main()
