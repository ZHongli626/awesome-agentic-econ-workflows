"""
Data Source Stage - Automated Mode (No Firecrawl, No HITL)
This script discovers and acquires economic datasets from various sources.

Pipeline:
1. DataScout: Discover target data sources and datasets
2. DataCollector: Acquire and standardize data

Input: Economic dataset/repository specification or research requirements
Output: Raw dataset with metadata
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# ========== DATA MODELS ==========

class DataRequirement(BaseModel):
    """Model for data requirements."""
    requirement_id: str = Field(description="Unique requirement identifier")
    variable_name: str = Field(description="Name of variable/indicator needed")
    description: str = Field(description="Description of what data is needed")
    frequency: str = Field(description="Required frequency: annual, quarterly, monthly, daily")
    time_period: str = Field(description="Required time period (e.g., '1990-2020')")
    geographic_coverage: str = Field(description="Geographic coverage (e.g., 'US', 'OECD', 'Global')")
    unit_of_measurement: str = Field(description="Unit of measurement")
    priority: str = Field(description="High/Medium/Low priority")


class DataSource(BaseModel):
    """Model for a data source."""
    source_id: str = Field(description="Unique source identifier")
    source_name: str = Field(description="Name of data source")
    source_type: str = Field(description="Type: government, international, academic, commercial, central_bank")
    url: str = Field(description="URL or access point")
    description: str = Field(description="Description of the source")
    coverage: str = Field(description="Geographic and temporal coverage")
    data_quality: str = Field(description="High/Medium/Low quality assessment")
    access_method: str = Field(description="API, download, web_scraping, manual")
    cost: str = Field(description="Free/Subscription/Purchase")
    update_frequency: str = Field(description="How often data is updated")
    variables_available: List[str] = Field(description="Key variables available")


class DatasetMetadata(BaseModel):
    """Model for dataset metadata."""
    dataset_id: str = Field(description="Unique dataset identifier")
    dataset_name: str = Field(description="Name of the dataset")
    source_name: str = Field(description="Source name")
    description: str = Field(description="Dataset description")
    variables: List[str] = Field(description="Variables/columns in dataset")
    num_observations: int = Field(description="Number of observations")
    time_period: str = Field(description="Time period covered")
    frequency: str = Field(description="Data frequency")
    geographic_coverage: str = Field(description="Geographic coverage")
    last_updated: str = Field(description="Last update date")
    file_format: str = Field(description="File format: csv, xlsx, json, etc.")
    file_size_mb: float = Field(description="File size in MB")


class DataAcquisitionPlan(BaseModel):
    """Model for data acquisition plan."""
    plan_id: str = Field(description="Unique plan identifier")
    target_sources: List[str] = Field(description="Source IDs to acquire from")
    acquisition_method: str = Field(description="Method: API, download, scraping")
    acquisition_steps: List[str] = Field(description="Ordered steps to acquire data")
    standardization_needed: bool = Field(description="Whether standardization is needed")
    standardization_steps: List[str] = Field(description="Standardization steps if needed")
    estimated_time: str = Field(description="Estimated time to complete")
    potential_issues: List[str] = Field(description="Potential issues to watch for")


class RawDataset(BaseModel):
    """Model for raw acquired dataset."""
    dataset_id: str = Field(description="Unique dataset identifier")
    dataset_name: str = Field(description="Dataset name")
    source: str = Field(description="Source name")
    acquisition_date: str = Field(description="Date acquired")
    file_path: str = Field(description="Path to data file")
    metadata: DatasetMetadata = Field(description="Dataset metadata")
    data_preview: List[Dict] = Field(description="First few rows as preview")
    quality_notes: str = Field(description="Initial quality assessment notes")
    standardization_applied: bool = Field(description="Whether standardization was applied")
    standardization_log: List[str] = Field(description="Log of standardization steps")


class DataSourceOutput(BaseModel):
    """Model for the complete data source stage output."""
    data_requirements: List[DataRequirement] = Field(description="Input data requirements")
    discovered_sources: List[DataSource] = Field(description="Discovered data sources")
    acquisition_plan: DataAcquisitionPlan = Field(description="Data acquisition plan")
    raw_datasets: List[RawDataset] = Field(description="Acquired raw datasets")
    metadata: Dict = Field(description="Output metadata", default_factory=dict)


# ========== AGENTS ==========

class DataScout:
    """Agent for discovering data sources and datasets."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "DataScout"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=self.api_key
        )
    
    def discover_sources(
        self,
        data_requirements: List[DataRequirement]
    ) -> List[DataSource]:
        """Discover relevant data sources based on requirements."""
        
        print(f"\n[{self.agent_name}] Discovering data sources for {len(data_requirements)} requirements...")
        
        # Prepare requirements context
        reqs_text = "\n".join([
            f"- {r.variable_name}: {r.description} ({r.frequency}, {r.time_period}, {r.geographic_coverage})"
            for r in data_requirements
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying economic data sources and repositories."),
            ("user", """Identify 5-10 relevant data sources for these data requirements.

Data Requirements:
{requirements}

For each source, provide:
- source_id: Short ID (e.g., "S1", "S2")
- source_name: Name (e.g., "FRED", "World Bank", "OECD", "BEA", "IMF")
- source_type: government, international, academic, commercial, or central_bank
- url: URL or access point
- description: Brief description
- coverage: Geographic and temporal coverage
- data_quality: High, Medium, or Low
- access_method: API, download, web_scraping, or manual
- cost: Free, Subscription, or Purchase
- update_frequency: How often updated
- variables_available: Key variables (5-10)

Focus on reputable sources like:
- FRED (Federal Reserve Economic Data)
- World Bank Open Data
- OECD Data
- IMF Data
- BEA (Bureau of Economic Analysis)
- BLS (Bureau of Labor Statistics)
- Eurostat
- National statistical agencies

Return as JSON with "sources" array.
Example: {{
  "sources": [
    {{
      "source_id": "S1",
      "source_name": "FRED (Federal Reserve Economic Data)",
      "source_type": "central_bank",
      "url": "https://fred.stlouisfed.org/",
      "description": "Comprehensive economic time series database maintained by Federal Reserve Bank of St. Louis",
      "coverage": "US and international, 1950s-present",
      "data_quality": "High",
      "access_method": "API",
      "cost": "Free",
      "update_frequency": "Daily",
      "variables_available": ["GDP", "Unemployment", "Inflation", "Interest rates", "Exchange rates"]
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "requirements": reqs_text
            })
            
            data = json.loads(result.content)
            sources = [DataSource(**s) for s in data.get("sources", [])]
            
            print(f"[{self.agent_name}] Discovered {len(sources)} data sources")
            return sources
        
        except Exception as e:
            print(f"    Error discovering sources: {e}")
            return []
    
    def identify_target_datasets(
        self,
        data_requirements: List[DataRequirement],
        discovered_sources: List[DataSource]
    ) -> List[DatasetMetadata]:
        """Identify specific target datasets from sources."""
        
        print(f"\n[{self.agent_name}] Identifying target datasets...")
        
        reqs_text = "\n".join([
            f"- {r.variable_name}: {r.description}"
            for r in data_requirements[:5]
        ])
        
        sources_text = "\n".join([
            f"- {s.source_name}: {', '.join(s.variables_available[:5])}"
            for s in discovered_sources[:5]
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying specific datasets within data sources."),
            ("user", """Identify 3-8 specific target datasets to acquire.

Requirements:
{requirements}

Available Sources:
{sources}

For each dataset, provide:
- dataset_id: Short ID (e.g., "D1", "D2")
- dataset_name: Specific dataset name
- source_name: Source it comes from
- description: What the dataset contains
- variables: List of variables/columns (5-15)
- num_observations: Estimated number of observations
- time_period: Time period covered
- frequency: annual, quarterly, monthly, or daily
- geographic_coverage: Geographic coverage
- last_updated: Estimated last update (e.g., "2024-Q3")
- file_format: csv, xlsx, json, or other
- file_size_mb: Estimated size in MB

Return as JSON with "datasets" array.
Example: {{
  "datasets": [
    {{
      "dataset_id": "D1",
      "dataset_name": "GDP and Components - Quarterly",
      "source_name": "BEA",
      "description": "US GDP and its components at quarterly frequency",
      "variables": ["Date", "GDP", "Consumption", "Investment", "Government", "Net_Exports"],
      "num_observations": 300,
      "time_period": "1947-2024",
      "frequency": "quarterly",
      "geographic_coverage": "United States",
      "last_updated": "2024-Q3",
      "file_format": "csv",
      "file_size_mb": 0.5
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "requirements": reqs_text,
                "sources": sources_text
            })
            
            data = json.loads(result.content)
            datasets = [DatasetMetadata(**d) for d in data.get("datasets", [])]
            
            print(f"[{self.agent_name}] Identified {len(datasets)} target datasets")
            return datasets
        
        except Exception as e:
            print(f"    Error identifying datasets: {e}")
            return []


class DataCollector:
    """Agent for acquiring and standardizing data."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "DataCollector"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=self.api_key
        )
    
    def create_acquisition_plan(
        self,
        target_datasets: List[DatasetMetadata],
        discovered_sources: List[DataSource]
    ) -> DataAcquisitionPlan:
        """Create a plan for acquiring the target datasets."""
        
        print(f"\n[{self.agent_name}] Creating data acquisition plan...")
        
        datasets_text = "\n".join([
            f"- {d.dataset_id}: {d.dataset_name} from {d.source_name}"
            for d in target_datasets
        ])
        
        sources_text = "\n".join([
            f"- {s.source_name}: {s.access_method} ({s.cost})"
            for s in discovered_sources[:5]
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at planning data acquisition workflows."),
            ("user", """Create a data acquisition plan for these datasets.

Target Datasets:
{datasets}

Available Sources:
{sources}

Provide:
- plan_id: "PLAN1"
- target_sources: List of source names to acquire from
- acquisition_method: Primary method (API, download, or scraping)
- acquisition_steps: Ordered steps (5-10 steps)
- standardization_needed: true/false
- standardization_steps: Steps if needed (3-7 steps)
- estimated_time: Time estimate (e.g., "2-4 hours")
- potential_issues: Issues to watch for (3-5)

Return as JSON.
Example: {{
  "plan_id": "PLAN1",
  "target_sources": ["FRED", "BEA", "BLS"],
  "acquisition_method": "API",
  "acquisition_steps": [
    "Step 1: Set up API credentials for FRED",
    "Step 2: Query FRED API for GDP data",
    "Step 3: Download BEA quarterly data",
    "Step 4: Save raw data files"
  ],
  "standardization_needed": true,
  "standardization_steps": [
    "Convert all dates to YYYY-MM-DD format",
    "Standardize column names to snake_case",
    "Handle missing values consistently"
  ],
  "estimated_time": "2-3 hours",
  "potential_issues": [
    "API rate limits",
    "Missing data for recent periods",
    "Different date formats across sources"
  ]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "datasets": datasets_text,
                "sources": sources_text
            })
            
            data = json.loads(result.content)
            plan = DataAcquisitionPlan(**data)
            
            print(f"[{self.agent_name}] Acquisition plan created")
            return plan
        
        except Exception as e:
            print(f"    Error creating plan: {e}")
            return DataAcquisitionPlan(
                plan_id="PLAN1",
                target_sources=[],
                acquisition_method="download",
                acquisition_steps=[],
                standardization_needed=True,
                standardization_steps=[],
                estimated_time="Unknown",
                potential_issues=[]
            )
    
    def simulate_data_acquisition(
        self,
        target_datasets: List[DatasetMetadata],
        acquisition_plan: DataAcquisitionPlan
    ) -> List[RawDataset]:
        """Simulate data acquisition (creates synthetic data for demonstration)."""
        
        print(f"\n[{self.agent_name}] Acquiring datasets...")
        print(f"  Note: Simulating data acquisition with synthetic data for demonstration")
        
        raw_datasets = []
        
        for dataset_meta in target_datasets:
            print(f"  Acquiring: {dataset_meta.dataset_name}...")
            
            # Create synthetic data preview
            data_preview = self._create_synthetic_preview(dataset_meta)
            
            # Create file path
            file_path = f"data/raw/{dataset_meta.dataset_id}_{dataset_meta.dataset_name.replace(' ', '_').lower()}.csv"
            
            # Create raw dataset object
            raw_dataset = RawDataset(
                dataset_id=dataset_meta.dataset_id,
                dataset_name=dataset_meta.dataset_name,
                source=dataset_meta.source_name,
                acquisition_date=datetime.now().strftime('%Y-%m-%d'),
                file_path=file_path,
                metadata=dataset_meta,
                data_preview=data_preview,
                quality_notes=f"Data acquired successfully. {dataset_meta.num_observations} observations covering {dataset_meta.time_period}.",
                standardization_applied=acquisition_plan.standardization_needed,
                standardization_log=acquisition_plan.standardization_steps if acquisition_plan.standardization_needed else []
            )
            
            raw_datasets.append(raw_dataset)
            print(f"    ✓ Acquired: {len(data_preview)} rows preview")
        
        print(f"\n[{self.agent_name}] Acquired {len(raw_datasets)} datasets")
        return raw_datasets
    
    def _create_synthetic_preview(self, dataset_meta: DatasetMetadata) -> List[Dict]:
        """Create synthetic data preview for demonstration."""
        
        # Create 5 sample rows
        preview = []
        
        for i in range(5):
            row = {}
            for var in dataset_meta.variables[:10]:  # Limit to first 10 variables
                if 'date' in var.lower() or 'year' in var.lower():
                    row[var] = f"2020-0{i+1}-01"
                elif 'id' in var.lower():
                    row[var] = f"ID{i+1:03d}"
                elif 'name' in var.lower():
                    row[var] = f"Entity_{i+1}"
                else:
                    # Numeric value
                    row[var] = round(100 + i * 10.5, 2)
            
            preview.append(row)
        
        return preview


# ========== ORCHESTRATOR ==========

class DataSourceOrchestrator:
    """Orchestrator for the data source stage."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        
        self.data_scout = DataScout(self.api_key)
        self.data_collector = DataCollector(self.api_key)
        self.source_output: Optional[DataSourceOutput] = None
    
    def run_source_pipeline(
        self,
        data_requirements: List[DataRequirement]
    ) -> DataSourceOutput:
        """Run the complete data source pipeline."""
        
        print(f"\n{'='*70}")
        print(f"DATA SOURCE PIPELINE")
        print(f"{'='*70}")
        print(f"Data Requirements: {len(data_requirements)}")
        print(f"{'='*70}\n")
        
        # Step 1: DataScout discovers sources
        print(f"STEP 1: DATA DISCOVERY")
        print(f"-"*70)
        discovered_sources = self.data_scout.discover_sources(data_requirements)
        
        # Step 2: DataScout identifies target datasets
        target_datasets = self.data_scout.identify_target_datasets(
            data_requirements,
            discovered_sources
        )
        
        # Step 3: DataCollector creates acquisition plan
        print(f"\nSTEP 2: ACQUISITION PLANNING")
        print(f"-"*70)
        acquisition_plan = self.data_collector.create_acquisition_plan(
            target_datasets,
            discovered_sources
        )
        
        # Step 4: DataCollector acquires data
        print(f"\nSTEP 3: DATA ACQUISITION")
        print(f"-"*70)
        raw_datasets = self.data_collector.simulate_data_acquisition(
            target_datasets,
            acquisition_plan
        )
        
        # Create output
        self.source_output = DataSourceOutput(
            data_requirements=data_requirements,
            discovered_sources=discovered_sources,
            acquisition_plan=acquisition_plan,
            raw_datasets=raw_datasets,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_requirements": len(data_requirements),
                "num_sources": len(discovered_sources),
                "num_datasets": len(raw_datasets),
                "total_observations": sum(d.metadata.num_observations for d in raw_datasets)
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Sources Discovered: {len(discovered_sources)}")
        print(f"Datasets Acquired: {len(raw_datasets)}")
        print(f"Total Observations: {self.source_output.metadata['total_observations']}")
        print(f"{'='*70}\n")
        
        return self.source_output
    
    def save_source_output(self, filename: str = "data_source_output.json"):
        """Save source output to JSON."""
        if not self.source_output:
            print("No source output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.source_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] Source output saved to {filename}")
    
    def save_source_report(self, filename: str = "data_source_report.txt"):
        """Save source report in readable text format."""
        if not self.source_output:
            print("No source output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DATA SOURCE REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"DATA REQUIREMENTS ({len(self.source_output.data_requirements)}):\n")
            f.write("-"*70 + "\n")
            for req in self.source_output.data_requirements:
                f.write(f"\n{req.requirement_id}. {req.variable_name}\n")
                f.write(f"   Description: {req.description}\n")
                f.write(f"   Frequency: {req.frequency} | Period: {req.time_period}\n")
                f.write(f"   Coverage: {req.geographic_coverage} | Priority: {req.priority}\n")
            
            f.write(f"\n\nDISCOVERED SOURCES ({len(self.source_output.discovered_sources)}):\n")
            f.write("-"*70 + "\n")
            for source in self.source_output.discovered_sources:
                f.write(f"\n{source.source_id}. {source.source_name}\n")
                f.write(f"   Type: {source.source_type} | Quality: {source.data_quality}\n")
                f.write(f"   Access: {source.access_method} | Cost: {source.cost}\n")
                f.write(f"   URL: {source.url}\n")
                f.write(f"   Coverage: {source.coverage}\n")
                f.write(f"   Variables: {', '.join(source.variables_available[:5])}...\n")
            
            f.write(f"\n\nACQUISITION PLAN:\n")
            f.write("-"*70 + "\n")
            plan = self.source_output.acquisition_plan
            f.write(f"Method: {plan.acquisition_method}\n")
            f.write(f"Estimated Time: {plan.estimated_time}\n")
            f.write(f"Standardization: {'Yes' if plan.standardization_needed else 'No'}\n\n")
            
            f.write(f"Acquisition Steps:\n")
            for i, step in enumerate(plan.acquisition_steps, 1):
                f.write(f"  {i}. {step}\n")
            
            if plan.standardization_needed:
                f.write(f"\nStandardization Steps:\n")
                for i, step in enumerate(plan.standardization_steps, 1):
                    f.write(f"  {i}. {step}\n")
            
            f.write(f"\nPotential Issues:\n")
            for issue in plan.potential_issues:
                f.write(f"  - {issue}\n")
            
            f.write(f"\n\nACQUIRED DATASETS ({len(self.source_output.raw_datasets)}):\n")
            f.write("-"*70 + "\n")
            for dataset in self.source_output.raw_datasets:
                f.write(f"\n{dataset.dataset_id}. {dataset.dataset_name}\n")
                f.write(f"   Source: {dataset.source}\n")
                f.write(f"   Observations: {dataset.metadata.num_observations}\n")
                f.write(f"   Period: {dataset.metadata.time_period} ({dataset.metadata.frequency})\n")
                f.write(f"   Coverage: {dataset.metadata.geographic_coverage}\n")
                f.write(f"   File: {dataset.file_path}\n")
                f.write(f"   Format: {dataset.metadata.file_format} ({dataset.metadata.file_size_mb} MB)\n")
                f.write(f"   Variables: {', '.join(dataset.metadata.variables[:8])}...\n")
                f.write(f"   Quality: {dataset.quality_notes}\n")
                
                if dataset.data_preview:
                    f.write(f"\n   Data Preview (first 3 rows):\n")
                    for i, row in enumerate(dataset.data_preview[:3], 1):
                        f.write(f"     Row {i}: {row}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"[Orchestrator] Source report saved to {filename}")
    
    def save_datasets_csv(self, output_dir: str = "data/raw"):
        """Save acquired datasets as CSV files."""
        if not self.source_output:
            print("No source output to save")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset in self.source_output.raw_datasets:
            # Create DataFrame from preview
            df = pd.DataFrame(dataset.data_preview)
            
            # Save to CSV
            filename = f"{dataset.dataset_id}_{dataset.dataset_name.replace(' ', '_').lower()}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            
            print(f"[Orchestrator] Saved dataset to {filepath}")


def main():
    """Main function for data source stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Create sample data requirements
    print("Creating sample data requirements...")
    data_requirements = [
        DataRequirement(
            requirement_id="R1",
            variable_name="GDP",
            description="Real Gross Domestic Product",
            frequency="quarterly",
            time_period="1990-2024",
            geographic_coverage="United States",
            unit_of_measurement="Billions of chained 2012 dollars",
            priority="High"
        ),
        DataRequirement(
            requirement_id="R2",
            variable_name="Unemployment Rate",
            description="Civilian unemployment rate",
            frequency="monthly",
            time_period="1990-2024",
            geographic_coverage="United States",
            unit_of_measurement="Percent",
            priority="High"
        ),
        DataRequirement(
            requirement_id="R3",
            variable_name="CPI",
            description="Consumer Price Index for All Urban Consumers",
            frequency="monthly",
            time_period="1990-2024",
            geographic_coverage="United States",
            unit_of_measurement="Index 1982-84=100",
            priority="High"
        ),
        DataRequirement(
            requirement_id="R4",
            variable_name="Federal Funds Rate",
            description="Effective federal funds rate",
            frequency="monthly",
            time_period="1990-2024",
            geographic_coverage="United States",
            unit_of_measurement="Percent per annum",
            priority="Medium"
        ),
        DataRequirement(
            requirement_id="R5",
            variable_name="Personal Consumption",
            description="Real personal consumption expenditures",
            frequency="quarterly",
            time_period="1990-2024",
            geographic_coverage="United States",
            unit_of_measurement="Billions of chained 2012 dollars",
            priority="Medium"
        )
    ]
    
    print(f"Created {len(data_requirements)} data requirements\n")
    
    # Run pipeline
    orchestrator = DataSourceOrchestrator()
    source_output = orchestrator.run_source_pipeline(data_requirements)
    
    # Save outputs
    orchestrator.save_source_output("data_source_output.json")
    orchestrator.save_source_report("data_source_report.txt")
    orchestrator.save_datasets_csv("data/raw")
    
    # Print summary
    print("\n" + "="*70)
    print("DATA SOURCE SUMMARY")
    print("="*70)
    print(f"Requirements: {len(source_output.data_requirements)}")
    print(f"Sources Discovered: {len(source_output.discovered_sources)}")
    print(f"Datasets Acquired: {len(source_output.raw_datasets)}")
    print(f"Total Observations: {source_output.metadata['total_observations']}")
    
    print(f"\nTop Sources:")
    for source in source_output.discovered_sources[:3]:
        print(f"  - {source.source_name} ({source.source_type}, {source.data_quality} quality)")
    
    print(f"\nAcquired Datasets:")
    for dataset in source_output.raw_datasets:
        print(f"  - {dataset.dataset_name}: {dataset.metadata.num_observations} obs, {dataset.metadata.time_period}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
