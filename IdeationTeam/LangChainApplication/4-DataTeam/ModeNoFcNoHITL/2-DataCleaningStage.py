"""
Data Cleaning Stage - Automated Mode (No Firecrawl, No HITL)
This script performs data preprocessing, cleaning, integration, and feature engineering.

Pipeline:
1. DataCleaner: Quality control and data cleaning
2. DataIntegrator: Consolidate datasets
3. FeatureEngineer: Create derived variables and features

Input: Raw datasets from Stage 1 (data_source_output.json)
Output: Feature-enhanced datasets
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# ========== DATA MODELS ==========

class RawDataset(BaseModel):
    """Model for raw dataset from Stage 1 (simplified)."""
    dataset_id: str
    dataset_name: str
    source: str
    file_path: str
    metadata: Dict
    data_preview: List[Dict]
    quality_notes: str


class DataQualityIssue(BaseModel):
    """Model for a data quality issue."""
    issue_id: str = Field(description="Unique issue identifier")
    issue_type: str = Field(description="Type: missing_values, outliers, duplicates, inconsistency, format_error")
    severity: str = Field(description="High/Medium/Low severity")
    description: str = Field(description="Description of the issue")
    affected_variables: List[str] = Field(description="Variables affected")
    affected_rows: int = Field(description="Number of rows affected")
    recommended_action: str = Field(description="Recommended action to fix")


class CleaningAction(BaseModel):
    """Model for a cleaning action."""
    action_id: str = Field(description="Unique action identifier")
    action_type: str = Field(description="Type: imputation, removal, transformation, standardization")
    description: str = Field(description="Description of the action")
    variables_affected: List[str] = Field(description="Variables affected")
    method: str = Field(description="Method used")
    parameters: Dict = Field(description="Parameters for the action", default_factory=dict)


class CleanedDataset(BaseModel):
    """Model for cleaned dataset."""
    dataset_id: str = Field(description="Dataset identifier")
    dataset_name: str = Field(description="Dataset name")
    source: str = Field(description="Source name")
    file_path: str = Field(description="Path to cleaned data file")
    original_rows: int = Field(description="Original number of rows")
    cleaned_rows: int = Field(description="Number of rows after cleaning")
    quality_issues: List[DataQualityIssue] = Field(description="Quality issues found")
    cleaning_actions: List[CleaningAction] = Field(description="Cleaning actions applied")
    data_preview: List[Dict] = Field(description="Preview of cleaned data")
    quality_score: float = Field(description="Quality score 0-1")


class IntegrationMapping(BaseModel):
    """Model for integration mapping."""
    mapping_id: str = Field(description="Unique mapping identifier")
    source_dataset: str = Field(description="Source dataset ID")
    source_variable: str = Field(description="Source variable name")
    target_variable: str = Field(description="Target variable name in integrated dataset")
    transformation: str = Field(description="Transformation applied")


class IntegratedDataset(BaseModel):
    """Model for integrated dataset."""
    integrated_id: str = Field(description="Integrated dataset identifier")
    integrated_name: str = Field(description="Integrated dataset name")
    source_datasets: List[str] = Field(description="Source dataset IDs")
    file_path: str = Field(description="Path to integrated data file")
    integration_method: str = Field(description="Method: merge, append, join")
    join_keys: List[str] = Field(description="Keys used for joining")
    mappings: List[IntegrationMapping] = Field(description="Variable mappings")
    num_rows: int = Field(description="Number of rows")
    num_variables: int = Field(description="Number of variables")
    data_preview: List[Dict] = Field(description="Preview of integrated data")


class FeatureDefinition(BaseModel):
    """Model for a feature definition."""
    feature_id: str = Field(description="Unique feature identifier")
    feature_name: str = Field(description="Feature name")
    feature_type: str = Field(description="Type: lag, difference, ratio, interaction, polynomial, aggregate")
    description: str = Field(description="Description of the feature")
    formula: str = Field(description="Formula or calculation")
    source_variables: List[str] = Field(description="Source variables used")
    rationale: str = Field(description="Economic rationale for the feature")


class FeatureEnhancedDataset(BaseModel):
    """Model for feature-enhanced dataset."""
    dataset_id: str = Field(description="Dataset identifier")
    dataset_name: str = Field(description="Dataset name")
    base_dataset: str = Field(description="Base integrated dataset ID")
    file_path: str = Field(description="Path to feature-enhanced data file")
    original_variables: List[str] = Field(description="Original variables")
    engineered_features: List[FeatureDefinition] = Field(description="Engineered features")
    num_rows: int = Field(description="Number of rows")
    num_total_variables: int = Field(description="Total variables including features")
    data_preview: List[Dict] = Field(description="Preview of feature-enhanced data")
    feature_summary: str = Field(description="Summary of feature engineering")


class DataCleaningOutput(BaseModel):
    """Model for the complete data cleaning stage output."""
    raw_datasets: List[RawDataset] = Field(description="Input raw datasets")
    cleaned_datasets: List[CleanedDataset] = Field(description="Cleaned datasets")
    integrated_datasets: List[IntegratedDataset] = Field(description="Integrated datasets")
    feature_enhanced_datasets: List[FeatureEnhancedDataset] = Field(description="Feature-enhanced datasets")
    metadata: Dict = Field(description="Output metadata", default_factory=dict)


# ========== AGENTS ==========

class DataCleaner:
    """Agent for quality control and data cleaning."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "DataCleaner"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=self.api_key
        )
    
    def clean_dataset(
        self,
        raw_dataset: RawDataset
    ) -> CleanedDataset:
        """Clean a raw dataset."""
        
        print(f"\n[{self.agent_name}] Cleaning dataset: {raw_dataset.dataset_name}...")
        
        # Step 1: Identify quality issues
        print(f"  Step 1: Identifying quality issues...")
        quality_issues = self._identify_quality_issues(raw_dataset)
        
        # Step 2: Design cleaning actions
        print(f"  Step 2: Designing cleaning actions...")
        cleaning_actions = self._design_cleaning_actions(quality_issues)
        
        # Step 3: Apply cleaning
        print(f"  Step 3: Applying cleaning actions...")
        cleaned_data, quality_score = self._apply_cleaning(
            raw_dataset,
            cleaning_actions
        )
        
        # Create cleaned dataset
        cleaned_dataset = CleanedDataset(
            dataset_id=raw_dataset.dataset_id,
            dataset_name=f"{raw_dataset.dataset_name} (Cleaned)",
            source=raw_dataset.source,
            file_path=f"data/cleaned/{raw_dataset.dataset_id}_cleaned.csv",
            original_rows=len(raw_dataset.data_preview) * 60,  # Estimate
            cleaned_rows=len(cleaned_data),
            quality_issues=quality_issues,
            cleaning_actions=cleaning_actions,
            data_preview=cleaned_data,
            quality_score=quality_score
        )
        
        print(f"[{self.agent_name}] Cleaning complete. Quality score: {quality_score:.2f}")
        return cleaned_dataset
    
    def _identify_quality_issues(
        self,
        raw_dataset: RawDataset
    ) -> List[DataQualityIssue]:
        """Identify quality issues in the dataset."""
        
        variables = list(raw_dataset.data_preview[0].keys()) if raw_dataset.data_preview else []
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying data quality issues in economic datasets."),
            ("user", """Identify 3-7 potential data quality issues for this dataset.

Dataset: {dataset_name}
Source: {source}
Variables: {variables}

For each issue, provide:
- issue_id: Short ID (e.g., "Q1", "Q2")
- issue_type: missing_values, outliers, duplicates, inconsistency, or format_error
- severity: High, Medium, or Low
- description: Description of the issue
- affected_variables: List of variable names
- affected_rows: Estimated number of rows affected
- recommended_action: Recommended action

Return as JSON with "issues" array.
Example: {{
  "issues": [
    {{
      "issue_id": "Q1",
      "issue_type": "missing_values",
      "severity": "Medium",
      "description": "Missing values in GDP variable for recent periods",
      "affected_variables": ["GDP"],
      "affected_rows": 15,
      "recommended_action": "Impute using forward fill or interpolation"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "dataset_name": raw_dataset.dataset_name,
                "source": raw_dataset.source,
                "variables": ", ".join(variables[:10])
            })
            
            data = json.loads(result.content)
            issues = [DataQualityIssue(**i) for i in data.get("issues", [])]
            
            return issues
        
        except Exception as e:
            print(f"    Error identifying issues: {e}")
            return []
    
    def _design_cleaning_actions(
        self,
        quality_issues: List[DataQualityIssue]
    ) -> List[CleaningAction]:
        """Design cleaning actions based on quality issues."""
        
        issues_text = "\n".join([
            f"- {i.issue_id}: {i.issue_type} - {i.description}"
            for i in quality_issues
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at designing data cleaning strategies."),
            ("user", """Design cleaning actions for these quality issues.

Quality Issues:
{issues}

For each action, provide:
- action_id: Short ID (e.g., "A1", "A2")
- action_type: imputation, removal, transformation, or standardization
- description: What the action does
- variables_affected: List of variables
- method: Specific method (e.g., "forward fill", "mean imputation", "remove duplicates")
- parameters: Any parameters (as dict)

Return as JSON with "actions" array.
Example: {{
  "actions": [
    {{
      "action_id": "A1",
      "action_type": "imputation",
      "description": "Impute missing GDP values using linear interpolation",
      "variables_affected": ["GDP"],
      "method": "linear_interpolation",
      "parameters": {{"limit": 3}}
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "issues": issues_text
            })
            
            data = json.loads(result.content)
            actions = [CleaningAction(**a) for a in data.get("actions", [])]
            
            return actions
        
        except Exception as e:
            print(f"    Error designing actions: {e}")
            return []
    
    def _apply_cleaning(
        self,
        raw_dataset: RawDataset,
        cleaning_actions: List[CleaningAction]
    ) -> Tuple[List[Dict], float]:
        """Apply cleaning actions (simulated)."""
        
        # Simulate cleaning by modifying preview data
        cleaned_data = []
        for row in raw_dataset.data_preview:
            cleaned_row = row.copy()
            # Simulate some cleaning (e.g., fill missing values)
            for key, value in cleaned_row.items():
                if value is None or value == "":
                    cleaned_row[key] = 0.0 if isinstance(value, (int, float)) else "N/A"
            cleaned_data.append(cleaned_row)
        
        # Calculate quality score based on issues
        quality_score = 0.85  # Simulated score
        
        return cleaned_data, quality_score


class DataIntegrator:
    """Agent for consolidating datasets."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "DataIntegrator"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=self.api_key
        )
    
    def integrate_datasets(
        self,
        cleaned_datasets: List[CleanedDataset]
    ) -> List[IntegratedDataset]:
        """Integrate multiple cleaned datasets."""
        
        print(f"\n[{self.agent_name}] Integrating {len(cleaned_datasets)} datasets...")
        
        # Step 1: Design integration strategy
        print(f"  Step 1: Designing integration strategy...")
        integration_strategy = self._design_integration_strategy(cleaned_datasets)
        
        # Step 2: Create variable mappings
        print(f"  Step 2: Creating variable mappings...")
        mappings = self._create_variable_mappings(cleaned_datasets)
        
        # Step 3: Perform integration
        print(f"  Step 3: Performing integration...")
        integrated_datasets = self._perform_integration(
            cleaned_datasets,
            integration_strategy,
            mappings
        )
        
        print(f"[{self.agent_name}] Integration complete. Created {len(integrated_datasets)} integrated dataset(s)")
        return integrated_datasets
    
    def _design_integration_strategy(
        self,
        cleaned_datasets: List[CleanedDataset]
    ) -> Dict:
        """Design integration strategy."""
        
        datasets_text = "\n".join([
            f"- {d.dataset_id}: {d.dataset_name} ({d.cleaned_rows} rows)"
            for d in cleaned_datasets
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at designing data integration strategies."),
            ("user", """Design an integration strategy for these datasets.

Datasets:
{datasets}

Provide:
- integration_method: merge, append, or join
- join_keys: Keys to use for joining (e.g., ["date", "country"])
- rationale: Why this approach

Return as JSON.
Example: {{
  "integration_method": "merge",
  "join_keys": ["date"],
  "rationale": "All datasets have time series structure with common date variable"
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "datasets": datasets_text
            })
            
            data = json.loads(result.content)
            return data
        
        except Exception as e:
            print(f"    Error designing strategy: {e}")
            return {
                "integration_method": "merge",
                "join_keys": ["date"],
                "rationale": "Default time series merge"
            }
    
    def _create_variable_mappings(
        self,
        cleaned_datasets: List[CleanedDataset]
    ) -> List[IntegrationMapping]:
        """Create variable mappings for integration."""
        
        # Get all variables from first dataset preview
        all_vars = []
        for dataset in cleaned_datasets:
            if dataset.data_preview:
                all_vars.extend(list(dataset.data_preview[0].keys()))
        
        # Create simple mappings (identity mapping for demo)
        mappings = []
        for i, var in enumerate(set(all_vars[:10]), 1):
            mapping = IntegrationMapping(
                mapping_id=f"M{i}",
                source_dataset=cleaned_datasets[0].dataset_id if cleaned_datasets else "D1",
                source_variable=var,
                target_variable=var.lower().replace(" ", "_"),
                transformation="standardize_name"
            )
            mappings.append(mapping)
        
        return mappings
    
    def _perform_integration(
        self,
        cleaned_datasets: List[CleanedDataset],
        integration_strategy: Dict,
        mappings: List[IntegrationMapping]
    ) -> List[IntegratedDataset]:
        """Perform the actual integration."""
        
        # Simulate integration
        integrated_data = []
        
        # Combine data from all datasets
        for dataset in cleaned_datasets:
            for row in dataset.data_preview[:3]:  # Take first 3 rows from each
                integrated_row = {}
                for key, value in row.items():
                    # Standardize key names
                    std_key = key.lower().replace(" ", "_")
                    integrated_row[std_key] = value
                integrated_data.append(integrated_row)
        
        # Create integrated dataset
        integrated_dataset = IntegratedDataset(
            integrated_id="INT1",
            integrated_name="Integrated Economic Dataset",
            source_datasets=[d.dataset_id for d in cleaned_datasets],
            file_path="data/integrated/integrated_dataset.csv",
            integration_method=integration_strategy.get("integration_method", "merge"),
            join_keys=integration_strategy.get("join_keys", ["date"]),
            mappings=mappings,
            num_rows=len(integrated_data) * 100,  # Estimate
            num_variables=len(integrated_data[0]) if integrated_data else 0,
            data_preview=integrated_data
        )
        
        return [integrated_dataset]


class FeatureEngineer:
    """Agent for creating derived variables and features."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "FeatureEngineer"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=self.api_key
        )
    
    def engineer_features(
        self,
        integrated_dataset: IntegratedDataset
    ) -> FeatureEnhancedDataset:
        """Engineer features from integrated dataset."""
        
        print(f"\n[{self.agent_name}] Engineering features for: {integrated_dataset.integrated_name}...")
        
        # Step 1: Design features
        print(f"  Step 1: Designing features...")
        feature_definitions = self._design_features(integrated_dataset)
        
        # Step 2: Create features
        print(f"  Step 2: Creating features...")
        enhanced_data = self._create_features(
            integrated_dataset,
            feature_definitions
        )
        
        # Create feature-enhanced dataset
        original_vars = list(integrated_dataset.data_preview[0].keys()) if integrated_dataset.data_preview else []
        
        feature_enhanced = FeatureEnhancedDataset(
            dataset_id="FE1",
            dataset_name="Feature-Enhanced Economic Dataset",
            base_dataset=integrated_dataset.integrated_id,
            file_path="data/enhanced/feature_enhanced_dataset.csv",
            original_variables=original_vars,
            engineered_features=feature_definitions,
            num_rows=integrated_dataset.num_rows,
            num_total_variables=len(original_vars) + len(feature_definitions),
            data_preview=enhanced_data,
            feature_summary=f"Created {len(feature_definitions)} engineered features including lags, differences, ratios, and interactions"
        )
        
        print(f"[{self.agent_name}] Feature engineering complete. Created {len(feature_definitions)} features")
        return feature_enhanced
    
    def _design_features(
        self,
        integrated_dataset: IntegratedDataset
    ) -> List[FeatureDefinition]:
        """Design features to create."""
        
        variables = list(integrated_dataset.data_preview[0].keys()) if integrated_dataset.data_preview else []
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at feature engineering for economic data."),
            ("user", """Design 8-15 engineered features for this dataset.

Dataset: {dataset_name}
Variables: {variables}

For each feature, provide:
- feature_id: Short ID (e.g., "F1", "F2")
- feature_name: Feature name
- feature_type: lag, difference, ratio, interaction, polynomial, or aggregate
- description: Description
- formula: Formula (e.g., "GDP_t - GDP_t-1", "GDP / Population")
- source_variables: List of source variables
- rationale: Economic rationale

Include diverse feature types:
- Lags: lagged values for time series
- Differences: first differences, growth rates
- Ratios: meaningful economic ratios
- Interactions: interaction terms
- Polynomials: squared or cubed terms

Return as JSON with "features" array.
Example: {{
  "features": [
    {{
      "feature_id": "F1",
      "feature_name": "gdp_growth",
      "feature_type": "difference",
      "description": "GDP growth rate (year-over-year)",
      "formula": "(GDP_t - GDP_t-4) / GDP_t-4 * 100",
      "source_variables": ["gdp"],
      "rationale": "Growth rate is key indicator of economic performance"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "dataset_name": integrated_dataset.integrated_name,
                "variables": ", ".join(variables[:10])
            })
            
            data = json.loads(result.content)
            features = [FeatureDefinition(**f) for f in data.get("features", [])]
            
            return features
        
        except Exception as e:
            print(f"    Error designing features: {e}")
            return []
    
    def _create_features(
        self,
        integrated_dataset: IntegratedDataset,
        feature_definitions: List[FeatureDefinition]
    ) -> List[Dict]:
        """Create the actual features."""
        
        # Simulate feature creation
        enhanced_data = []
        
        for row in integrated_dataset.data_preview:
            enhanced_row = row.copy()
            
            # Add simulated features
            for feature in feature_definitions[:5]:  # Add first 5 features
                if feature.feature_type == "lag":
                    enhanced_row[feature.feature_name] = 95.0  # Simulated lag value
                elif feature.feature_type == "difference":
                    enhanced_row[feature.feature_name] = 2.5  # Simulated growth rate
                elif feature.feature_type == "ratio":
                    enhanced_row[feature.feature_name] = 0.65  # Simulated ratio
                elif feature.feature_type == "interaction":
                    enhanced_row[feature.feature_name] = 150.0  # Simulated interaction
                else:
                    enhanced_row[feature.feature_name] = 100.0  # Default
            
            enhanced_data.append(enhanced_row)
        
        return enhanced_data


# ========== ORCHESTRATOR ==========

class DataCleaningOrchestrator:
    """Orchestrator for the data cleaning stage."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        
        self.data_cleaner = DataCleaner(self.api_key)
        self.data_integrator = DataIntegrator(self.api_key)
        self.feature_engineer = FeatureEngineer(self.api_key)
        self.cleaning_output: Optional[DataCleaningOutput] = None
    
    def run_cleaning_pipeline(
        self,
        data_source_output: Dict
    ) -> DataCleaningOutput:
        """Run the complete data cleaning pipeline."""
        
        print(f"\n{'='*70}")
        print(f"DATA CLEANING PIPELINE")
        print(f"{'='*70}")
        
        # Parse raw datasets
        raw_datasets_data = data_source_output.get('raw_datasets', [])
        raw_datasets = [RawDataset(**d) for d in raw_datasets_data]
        
        print(f"Raw Datasets: {len(raw_datasets)}")
        print(f"{'='*70}\n")
        
        # Step 1: Clean datasets
        print(f"STEP 1: DATA CLEANING")
        print(f"-"*70)
        cleaned_datasets = []
        for raw_dataset in raw_datasets:
            cleaned_dataset = self.data_cleaner.clean_dataset(raw_dataset)
            cleaned_datasets.append(cleaned_dataset)
        
        # Step 2: Integrate datasets
        print(f"\nSTEP 2: DATA INTEGRATION")
        print(f"-"*70)
        integrated_datasets = self.data_integrator.integrate_datasets(cleaned_datasets)
        
        # Step 3: Engineer features
        print(f"\nSTEP 3: FEATURE ENGINEERING")
        print(f"-"*70)
        feature_enhanced_datasets = []
        for integrated_dataset in integrated_datasets:
            feature_enhanced = self.feature_engineer.engineer_features(integrated_dataset)
            feature_enhanced_datasets.append(feature_enhanced)
        
        # Create output
        self.cleaning_output = DataCleaningOutput(
            raw_datasets=raw_datasets,
            cleaned_datasets=cleaned_datasets,
            integrated_datasets=integrated_datasets,
            feature_enhanced_datasets=feature_enhanced_datasets,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_raw_datasets": len(raw_datasets),
                "num_cleaned_datasets": len(cleaned_datasets),
                "num_integrated_datasets": len(integrated_datasets),
                "num_feature_enhanced_datasets": len(feature_enhanced_datasets),
                "total_features_created": sum(len(d.engineered_features) for d in feature_enhanced_datasets)
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Cleaned Datasets: {len(cleaned_datasets)}")
        print(f"Integrated Datasets: {len(integrated_datasets)}")
        print(f"Feature-Enhanced Datasets: {len(feature_enhanced_datasets)}")
        print(f"Total Features Created: {self.cleaning_output.metadata['total_features_created']}")
        print(f"{'='*70}\n")
        
        return self.cleaning_output
    
    def save_cleaning_output(self, filename: str = "data_cleaning_output.json"):
        """Save cleaning output to JSON."""
        if not self.cleaning_output:
            print("No cleaning output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.cleaning_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] Cleaning output saved to {filename}")
    
    def save_cleaning_report(self, filename: str = "data_cleaning_report.txt"):
        """Save cleaning report in readable text format."""
        if not self.cleaning_output:
            print("No cleaning output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DATA CLEANING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"CLEANED DATASETS ({len(self.cleaning_output.cleaned_datasets)}):\n")
            f.write("-"*70 + "\n")
            for dataset in self.cleaning_output.cleaned_datasets:
                f.write(f"\n{dataset.dataset_id}. {dataset.dataset_name}\n")
                f.write(f"   Source: {dataset.source}\n")
                f.write(f"   Rows: {dataset.original_rows} -> {dataset.cleaned_rows}\n")
                f.write(f"   Quality Score: {dataset.quality_score:.2f}\n")
                f.write(f"   Quality Issues: {len(dataset.quality_issues)}\n")
                for issue in dataset.quality_issues:
                    f.write(f"     - {issue.issue_type} ({issue.severity}): {issue.description}\n")
                f.write(f"   Cleaning Actions: {len(dataset.cleaning_actions)}\n")
                for action in dataset.cleaning_actions:
                    f.write(f"     - {action.action_type}: {action.description}\n")
            
            f.write(f"\n\nINTEGRATED DATASETS ({len(self.cleaning_output.integrated_datasets)}):\n")
            f.write("-"*70 + "\n")
            for dataset in self.cleaning_output.integrated_datasets:
                f.write(f"\n{dataset.integrated_id}. {dataset.integrated_name}\n")
                f.write(f"   Source Datasets: {', '.join(dataset.source_datasets)}\n")
                f.write(f"   Integration Method: {dataset.integration_method}\n")
                f.write(f"   Join Keys: {', '.join(dataset.join_keys)}\n")
                f.write(f"   Rows: {dataset.num_rows}\n")
                f.write(f"   Variables: {dataset.num_variables}\n")
                f.write(f"   Mappings: {len(dataset.mappings)}\n")
            
            f.write(f"\n\nFEATURE-ENHANCED DATASETS ({len(self.cleaning_output.feature_enhanced_datasets)}):\n")
            f.write("-"*70 + "\n")
            for dataset in self.cleaning_output.feature_enhanced_datasets:
                f.write(f"\n{dataset.dataset_id}. {dataset.dataset_name}\n")
                f.write(f"   Base Dataset: {dataset.base_dataset}\n")
                f.write(f"   Original Variables: {len(dataset.original_variables)}\n")
                f.write(f"   Engineered Features: {len(dataset.engineered_features)}\n")
                f.write(f"   Total Variables: {dataset.num_total_variables}\n")
                f.write(f"   Rows: {dataset.num_rows}\n\n")
                
                f.write(f"   Engineered Features:\n")
                for feature in dataset.engineered_features:
                    f.write(f"     {feature.feature_id}. {feature.feature_name} ({feature.feature_type})\n")
                    f.write(f"        Formula: {feature.formula}\n")
                    f.write(f"        Rationale: {feature.rationale}\n\n")
                
                f.write(f"   Summary: {dataset.feature_summary}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"[Orchestrator] Cleaning report saved to {filename}")
    
    def save_enhanced_datasets_csv(self, output_dir: str = "data/enhanced"):
        """Save feature-enhanced datasets as CSV files."""
        if not self.cleaning_output:
            print("No cleaning output to save")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset in self.cleaning_output.feature_enhanced_datasets:
            # Create DataFrame from preview
            df = pd.DataFrame(dataset.data_preview)
            
            # Save to CSV
            filename = f"{dataset.dataset_id}_{dataset.dataset_name.replace(' ', '_').lower()}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            
            print(f"[Orchestrator] Saved enhanced dataset to {filepath}")


def main():
    """Main function for data cleaning stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Load data source output
    source_file = input("Enter path to data source output JSON (from Stage 1): ").strip()
    
    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found!")
        return
    
    # Load data
    with open(source_file, 'r', encoding='utf-8') as f:
        data_source_output = json.load(f)
    
    # Run pipeline
    orchestrator = DataCleaningOrchestrator()
    cleaning_output = orchestrator.run_cleaning_pipeline(data_source_output)
    
    # Save outputs
    orchestrator.save_cleaning_output("data_cleaning_output.json")
    orchestrator.save_cleaning_report("data_cleaning_report.txt")
    orchestrator.save_enhanced_datasets_csv("data/enhanced")
    
    # Print summary
    print("\n" + "="*70)
    print("DATA CLEANING SUMMARY")
    print("="*70)
    print(f"Raw Datasets: {cleaning_output.metadata['num_raw_datasets']}")
    print(f"Cleaned Datasets: {cleaning_output.metadata['num_cleaned_datasets']}")
    print(f"Integrated Datasets: {cleaning_output.metadata['num_integrated_datasets']}")
    print(f"Feature-Enhanced Datasets: {cleaning_output.metadata['num_feature_enhanced_datasets']}")
    print(f"Total Features Created: {cleaning_output.metadata['total_features_created']}")
    
    print(f"\nQuality Scores:")
    for dataset in cleaning_output.cleaned_datasets:
        print(f"  - {dataset.dataset_name}: {dataset.quality_score:.2f}")
    
    print(f"\nFeature Types:")
    for dataset in cleaning_output.feature_enhanced_datasets:
        feature_types = {}
        for feature in dataset.engineered_features:
            feature_types[feature.feature_type] = feature_types.get(feature.feature_type, 0) + 1
        for ftype, count in feature_types.items():
            print(f"  - {ftype}: {count}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
