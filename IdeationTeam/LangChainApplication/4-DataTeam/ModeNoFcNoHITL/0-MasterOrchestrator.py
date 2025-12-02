"""
Master Orchestrator for Automated Data Pipeline (No Human-in-the-Loop, No FireCrawl)
This script runs all three DataTeam stages sequentially in a fully automated pipeline.

Pipeline:
1. Data Source Stage: Discover and acquire economic datasets
2. Data Cleaning Stage: Clean, integrate, and engineer features
3. Quality Assurance Stage: Validate, document, and assess reproducibility

Input: Data requirements from ModelTeam (calibration needs)
Output: High-quality, validated, and documented datasets ready for analysis
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Main orchestrator that runs all three DataTeam stages automatically."""
    
    # Change to script directory to ensure outputs are saved there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # ========== CONFIGURATION ==========
    print("\n" + "="*70)
    print("AUTOMATED DATA PIPELINE")
    print("Mode: No FireCrawl, No Human-in-the-Loop")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Check for required input files
    data_requirements_file = input("Enter path to data requirements JSON file (or press Enter for default): ").strip()
    if not data_requirements_file:
        # Try to find default files from ModelTeam
        default_paths = [
            "../../3-ModelTeam/ModeNoFcNoHITL/calibration_output.json",
            "../../../3-ModelTeam/ModeNoFcNoHITL/calibration_output.json",
            "data_requirements.json"
        ]
        for path in default_paths:
            if os.path.exists(path):
                data_requirements_file = path
                print(f"Using default data requirements file: {data_requirements_file}")
                break
        
        if not data_requirements_file or not os.path.exists(data_requirements_file):
            print("\n[WARNING] No data requirements file found.")
            print("Creating example data requirements...")
            # Create example requirements
            example_requirements = [
                {
                    "requirement_id": "REQ1",
                    "variable_name": "GDP",
                    "description": "Real GDP data for economic analysis",
                    "frequency": "quarterly",
                    "time_period": "1990-2023",
                    "geographic_coverage": "US",
                    "unit_of_measurement": "Billions of chained 2012 dollars",
                    "priority": "High"
                },
                {
                    "requirement_id": "REQ2",
                    "variable_name": "Unemployment Rate",
                    "description": "Civilian unemployment rate",
                    "frequency": "monthly",
                    "time_period": "1990-2023",
                    "geographic_coverage": "US",
                    "unit_of_measurement": "Percent",
                    "priority": "High"
                },
                {
                    "requirement_id": "REQ3",
                    "variable_name": "Inflation Rate",
                    "description": "Consumer Price Index inflation rate",
                    "frequency": "monthly",
                    "time_period": "1990-2023",
                    "geographic_coverage": "US",
                    "unit_of_measurement": "Percent change",
                    "priority": "Medium"
                }
            ]
            data_requirements_file = "example_data_requirements.json"
            with open(data_requirements_file, 'w', encoding='utf-8') as f:
                json.dump({"data_requirements": example_requirements}, f, indent=2)
            print(f"Created example file: {data_requirements_file}")
    
    print("\n" + "="*70)
    print("INPUT FILES")
    print("="*70)
    print(f"Data Requirements: {data_requirements_file}")
    print("="*70 + "\n")
    
    # ========== STAGE 1: DATA SOURCE DISCOVERY & ACQUISITION ==========
    print("\n" + "="*70)
    print("STAGE 1: DATA SOURCE DISCOVERY & ACQUISITION")
    print("="*70)
    
    try:
        # Import and run Stage 1
        from importlib import import_module
        stage1 = import_module('1-DataSourceStage')
        
        # Run data source stage
        orchestrator1 = stage1.DataSourceOrchestrator()
        
        # Load data requirements
        with open(data_requirements_file, 'r', encoding='utf-8') as f:
            req_data = json.load(f)
        
        # Extract requirements (handle different JSON structures)
        if isinstance(req_data, dict):
            requirements_list = req_data.get('data_requirements', req_data.get('requirements', []))
        else:
            requirements_list = req_data
        
        # Convert to DataRequirement objects
        requirements = []
        for req in requirements_list:
            if isinstance(req, dict):
                req_obj = stage1.DataRequirement(**req)
            else:
                req_obj = req
            requirements.append(req_obj)
        
        # Run data source stage
        source_output = orchestrator1.run_source_pipeline(
            data_requirements=requirements
        )
        
        # Save results
        source_file = "data_source_output.json"
        orchestrator1.save_source_output(source_file)
        
        data_sources = source_output.discovered_sources if source_output else []
        raw_datasets = source_output.raw_datasets if source_output else []
        
        print(f"\n[Stage 1] Complete: {len(data_sources)} data sources identified, {len(raw_datasets)} datasets acquired")
        print(f"[Stage 1] Results saved to: {source_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 1-DataSourceStage.py is properly configured.")
        return
    
    # ========== STAGE 2: DATA CLEANING & FEATURE ENGINEERING ==========
    print("\n" + "="*70)
    print("STAGE 2: DATA CLEANING & FEATURE ENGINEERING")
    print("="*70)
    
    try:
        # Import and run Stage 2
        stage2 = import_module('2-DataCleaningStage')
        
        # Run data cleaning stage
        orchestrator2 = stage2.DataCleaningOrchestrator()
        
        # Load source output
        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        
        # Run data cleaning stage
        cleaning_output = orchestrator2.run_cleaning_pipeline(
            data_source_output=source_data
        )
        
        # Save results
        cleaning_file = "data_cleaning_output.json"
        orchestrator2.save_cleaning_output(cleaning_file)
        
        cleaned_datasets = cleaning_output.cleaned_datasets if cleaning_output else []
        feature_datasets = cleaning_output.feature_enhanced_datasets if cleaning_output else []
        
        print(f"\n[Stage 2] Complete: {len(cleaned_datasets)} datasets cleaned, {len(feature_datasets)} feature-enhanced datasets created")
        print(f"[Stage 2] Results saved to: {cleaning_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 2-DataCleaningStage.py is properly configured.")
        return
    
    # ========== STAGE 3: QUALITY ASSURANCE ==========
    print("\n" + "="*70)
    print("STAGE 3: QUALITY ASSURANCE & VALIDATION")
    print("="*70)
    
    try:
        # Import and run Stage 3
        stage3 = import_module('3-QualityAssuranceStage')
        
        # Run quality assurance stage
        orchestrator3 = stage3.QualityAssuranceOrchestrator()
        
        # Load cleaning output
        with open(cleaning_file, 'r', encoding='utf-8') as f:
            cleaning_data = json.load(f)
        
        # Run quality assurance stage
        qa_output = orchestrator3.run_qa_pipeline(
            data_cleaning_output=cleaning_data
        )
        
        # Save final results
        qa_file = "quality_assurance_output.json"
        orchestrator3.save_qa_output(qa_file)
        
        documented_datasets = qa_output.documented_datasets if qa_output else []
        
        print(f"\n[Stage 3] Complete: {len(documented_datasets)} documented datasets")
        print(f"[Stage 3] Results saved to: {qa_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Stage 3 failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure 3-QualityAssuranceStage.py is properly configured.")
        return
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Data Requirements: {len(requirements) if requirements else 0}")
    print(f"Data Sources Identified: {len(data_sources) if data_sources else 0}")
    print(f"Raw Datasets Acquired: {len(raw_datasets) if raw_datasets else 0}")
    print(f"Cleaned Datasets: {len(cleaned_datasets) if cleaned_datasets else 0}")
    print(f"Feature-Enhanced Datasets: {len(feature_datasets) if feature_datasets else 0}")
    print(f"Documented Datasets: {len(documented_datasets) if documented_datasets else 0}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # ========== OUTPUT SUMMARY ==========
    print("\n" + "="*70)
    print("OUTPUT FILES SUMMARY")
    print("="*70)
    print(f"1. Data Source Output: {source_file}")
    print(f"2. Data Cleaning Output: {cleaning_file}")
    print(f"3. Quality Assurance Output: {qa_file}")
    print("="*70)
    
    # Display brief summary of first dataset
    if feature_datasets and len(feature_datasets) > 0:
        print("\n" + "="*70)
        print("SAMPLE OUTPUT: First Feature-Enhanced Dataset")
        print("="*70)
        first_dataset = feature_datasets[0] if isinstance(feature_datasets[0], dict) else feature_datasets[0].__dict__
        print(f"\nDataset ID: {first_dataset.get('dataset_id', 'N/A')}")
        print(f"Dataset Name: {first_dataset.get('dataset_name', 'N/A')}")
        print(f"Source: {first_dataset.get('source', 'N/A')}")
        print(f"\nVariables: {first_dataset.get('num_variables', 'N/A')}")
        print(f"Observations: {first_dataset.get('num_observations', 'N/A')}")
        print(f"Time Period: {first_dataset.get('time_period', 'N/A')}")
        print(f"Engineered Features: {len(first_dataset.get('engineered_features', []))}")
        print("="*70)
    
    # Display documented datasets summary
    if documented_datasets and len(documented_datasets) > 0:
        print("\n" + "="*70)
        print("DOCUMENTED DATASETS SUMMARY")
        print("="*70)
        for i, dataset in enumerate(documented_datasets[:3], 1):
            dataset_dict = dataset if isinstance(dataset, dict) else dataset.__dict__
            print(f"\n{i}. Dataset: {dataset_dict.get('dataset_id', 'N/A')}")
            print(f"   Dataset Name: {dataset_dict.get('dataset_name', 'N/A')}")
            doc = dataset_dict.get('documentation', {})
            if isinstance(doc, dict):
                print(f"   Documentation Sections: {len(doc.get('sections', []))}")
        print("="*70)
    
    print(f"\n✅ All DataTeam stages completed successfully!")
    print(f"📁 All results saved in: {os.getcwd()}")
    print("="*70)


if __name__ == "__main__":
    main()
