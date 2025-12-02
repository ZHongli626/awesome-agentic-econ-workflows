"""
Synthesis Stage - Automated Mode (No Firecrawl, No HITL)
This script synthesizes gap analysis from Stage 2 into a literature review and research plan.

Pipeline:
1. KnowledgeWeaver: Knowledge integration (synthesizes insights, gaps, and trends)
2. CiteKeeper: Reference finalization (creates formatted bibliography)

Input: Gap analysis results from Stage 2 (gap_analysis_results.json)
Output: Literature review and research plan
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# ========== DATA MODELS ==========

class LiteratureSection(BaseModel):
    """Model for a section of the literature review."""
    section_title: str = Field(description="Section title")
    section_content: str = Field(description="Section content (2-4 paragraphs)")
    key_papers: List[str] = Field(description="Key papers cited in this section")
    related_gaps: List[str] = Field(description="Related gap IDs")


class LiteratureReview(BaseModel):
    """Model for the complete literature review."""
    title: str = Field(description="Review title")
    abstract: str = Field(description="Abstract (150-250 words)")
    introduction: str = Field(description="Introduction section")
    sections: List[LiteratureSection] = Field(description="Main review sections")
    synthesis: str = Field(description="Synthesis and conclusions")
    metadata: Dict = Field(description="Review metadata", default_factory=dict)


class ResearchObjective(BaseModel):
    """Model for a research objective."""
    objective_id: str = Field(description="Unique objective ID")
    objective_title: str = Field(description="Objective title")
    description: str = Field(description="Detailed description")
    addresses_gaps: List[str] = Field(description="Gap IDs this objective addresses")
    methodology: List[str] = Field(description="Proposed methodologies")
    expected_outcomes: List[str] = Field(description="Expected outcomes")
    timeline: str = Field(description="Estimated timeline")
    priority: str = Field(description="High/Medium/Low")


class ResearchPlan(BaseModel):
    """Model for the research plan."""
    plan_title: str = Field(description="Plan title")
    overview: str = Field(description="Plan overview (2-3 paragraphs)")
    research_objectives: List[ResearchObjective] = Field(description="Research objectives")
    methodology_overview: str = Field(description="Overall methodology approach")
    expected_contributions: List[str] = Field(description="Expected contributions")
    timeline_summary: str = Field(description="Timeline summary")
    resource_requirements: List[str] = Field(description="Required resources")


class FormattedReference(BaseModel):
    """Model for a formatted reference."""
    citation_key: str = Field(description="Citation key (e.g., Smith2023)")
    formatted_citation: str = Field(description="Full formatted citation")
    reference_type: str = Field(description="Type: journal, conference, preprint, etc.")
    importance: str = Field(description="High/Medium/Low")


class Bibliography(BaseModel):
    """Model for the complete bibliography."""
    references: List[FormattedReference] = Field(description="All references")
    total_references: int = Field(description="Total number of references")
    by_type: Dict = Field(description="Count by reference type", default_factory=dict)
    by_importance: Dict = Field(description="Count by importance", default_factory=dict)


class SynthesisResult(BaseModel):
    """Model for the complete synthesis output."""
    literature_review: LiteratureReview = Field(description="Literature review")
    research_plan: ResearchPlan = Field(description="Research plan")
    bibliography: Bibliography = Field(description="Bibliography")
    metadata: Dict = Field(description="Synthesis metadata", default_factory=dict)


# ========== AGENTS ==========

class KnowledgeWeaver:
    """Agent for knowledge integration and synthesis."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "KnowledgeWeaver"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.6,
            openai_api_key=self.api_key
        )
    
    def integrate_knowledge(
        self,
        gap_analysis: Dict
    ) -> tuple[LiteratureReview, ResearchPlan]:
        """Integrate knowledge from gap analysis into review and plan."""
        
        print(f"\n[{self.agent_name}] Integrating knowledge from gap analysis...")
        
        # Extract key components
        paper_structures = gap_analysis.get('paper_structures', [])
        research_gaps = gap_analysis.get('research_gaps', [])
        knowledge_graph = gap_analysis.get('knowledge_graph', {})
        
        print(f"  - Papers: {len(paper_structures)}")
        print(f"  - Gaps: {len(research_gaps)}")
        print(f"  - Graph nodes: {knowledge_graph.get('metadata', {}).get('total_nodes', 0)}")
        
        # Step 1: Generate literature review
        literature_review = self._generate_literature_review(
            paper_structures,
            research_gaps,
            knowledge_graph
        )
        
        # Step 2: Generate research plan
        research_plan = self._generate_research_plan(
            research_gaps,
            paper_structures,
            knowledge_graph
        )
        
        print(f"[{self.agent_name}] Knowledge integration complete")
        
        return literature_review, research_plan
    
    def _generate_literature_review(
        self,
        paper_structures: List[Dict],
        research_gaps: List[Dict],
        knowledge_graph: Dict
    ) -> LiteratureReview:
        """Generate comprehensive literature review."""
        
        print(f"\n  [{self.agent_name}] Generating literature review...")
        
        # Safety check for empty data
        if not paper_structures:
            paper_structures = []
        if not research_gaps:
            research_gaps = []
        
        # Prepare context
        papers_summary = self._prepare_papers_summary(paper_structures[:30])
        gaps_summary = self._prepare_gaps_summary(research_gaps[:15])
        
        # Generate title and abstract
        title, abstract = self._generate_title_and_abstract(papers_summary, gaps_summary)
        
        # Generate introduction
        introduction = self._generate_introduction(papers_summary, gaps_summary)
        
        # Generate main sections
        sections = self._generate_review_sections(paper_structures, research_gaps, knowledge_graph)
        
        # Generate synthesis
        synthesis = self._generate_synthesis(research_gaps, knowledge_graph)
        
        review = LiteratureReview(
            title=title,
            abstract=abstract,
            introduction=introduction,
            sections=sections,
            synthesis=synthesis,
            metadata={
                "total_papers_reviewed": len(paper_structures),
                "total_gaps_identified": len(research_gaps),
                "total_sections": len(sections),
                "created_at": datetime.now().isoformat()
            }
        )
        
        print(f"  [{self.agent_name}] Literature review generated ({len(sections)} sections)")
        
        return review
    
    def _generate_title_and_abstract(
        self,
        papers_summary: str,
        gaps_summary: str
    ) -> tuple[str, str]:
        """Generate review title and abstract."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at writing academic literature reviews."),
            ("user", """Based on the following papers and research gaps, generate a title and abstract for a literature review.
            
            Papers Summary:
            {papers}
            
            Research Gaps:
            {gaps}
            
            Generate:
            - title: Concise, descriptive title (10-15 words)
            - abstract: Comprehensive abstract (150-250 words) covering scope, key findings, gaps, and future directions
            
            Return as a JSON object with "title" and "abstract" fields.
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "papers": papers_summary[:2000],
                "gaps": gaps_summary[:1000]
            })
            
            data = json.loads(result.content)
            return data.get("title", "Literature Review"), data.get("abstract", "")
        
        except Exception as e:
            print(f"    Error generating title/abstract: {e}")
            return "Literature Review", "This review synthesizes recent research findings and identifies key gaps."
    
    def _generate_introduction(self, papers_summary: str, gaps_summary: str) -> str:
        """Generate introduction section."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at writing academic literature reviews."),
            ("user", """Write an introduction section (3-4 paragraphs) for a literature review.
            
            Papers Summary:
            {papers}
            
            Research Gaps:
            {gaps}
            
            The introduction should:
            - Establish the research context and importance
            - Outline the scope of the review
            - Preview the main themes and gaps
            - State the review's objectives
            
            Return as a JSON object with an "introduction" field containing the text.
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "papers": papers_summary[:2000],
                "gaps": gaps_summary[:1000]
            })
            
            data = json.loads(result.content)
            return data.get("introduction", "")
        
        except Exception as e:
            print(f"    Error generating introduction: {e}")
            return "This literature review examines recent research in the field."
    
    def _generate_review_sections(
        self,
        paper_structures: List[Dict],
        research_gaps: List[Dict],
        knowledge_graph: Dict
    ) -> List[LiteratureSection]:
        """Generate main review sections organized by themes."""
        
        # Group papers by methodology/theme
        sections = []
        
        # Extract unique methodologies and themes
        all_methods = set()
        for paper in paper_structures[:30]:
            all_methods.update(paper.get('methodologies', []))
        
        # Create sections for top methodologies/themes
        top_methods = list(all_methods)[:5]
        
        # If no methodologies found, return empty sections list
        if not top_methods:
            return sections
        
        for idx, method in enumerate(top_methods, 1):
            # Find papers using this method
            related_papers = [
                p.get('paper_title', 'Unknown')
                for p in paper_structures
                if method in p.get('methodologies', [])
            ][:10]
            
            # Find related gaps
            related_gaps = [
                g.get('gap_id', '')
                for g in research_gaps
                if method.lower() in g.get('gap_description', '').lower()
            ][:3]
            
            # Generate section content
            section_content = self._generate_section_content(
                method,
                related_papers,
                related_gaps,
                paper_structures
            )
            
            section = LiteratureSection(
                section_title=f"{method} Approaches",
                section_content=section_content,
                key_papers=related_papers[:5],
                related_gaps=related_gaps
            )
            
            sections.append(section)
        
        # Add a gaps section
        gaps_section = self._generate_gaps_section(research_gaps)
        sections.append(gaps_section)
        
        return sections
    
    def _generate_section_content(
        self,
        theme: str,
        related_papers: List[str],
        related_gaps: List[str],
        paper_structures: List[Dict]
    ) -> str:
        """Generate content for a review section."""
        
        papers_text = "\n".join([f"- {p}" for p in related_papers[:5]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at writing academic literature reviews."),
            ("user", """Write a section (2-4 paragraphs) for a literature review on the following theme.
            
            Theme: {theme}
            
            Key Papers:
            {papers}
            
            The section should:
            - Summarize key findings and contributions
            - Discuss methodological approaches
            - Identify patterns and trends
            - Note limitations and gaps
            
            Return as a JSON object with a "content" field containing the text.
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "theme": theme,
                "papers": papers_text
            })
            
            data = json.loads(result.content)
            return data.get("content", "")
        
        except Exception as e:
            print(f"    Error generating section content: {e}")
            return f"This section reviews research on {theme}."
    
    def _generate_gaps_section(self, research_gaps: List[Dict]) -> LiteratureSection:
        """Generate section on research gaps."""
        
        gaps_text = "\n".join([
            f"- [{g.get('gap_type', 'unknown').upper()}] {g.get('gap_title', 'Unknown')}: {g.get('gap_description', '')[:150]}..."
            for g in research_gaps[:10]
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at writing academic literature reviews."),
            ("user", """Write a section (2-3 paragraphs) summarizing research gaps.
            
            Research Gaps:
            {gaps}
            
            The section should:
            - Categorize gaps by type
            - Discuss their significance
            - Suggest priorities for future research
            
            Return as a JSON object with a "content" field containing the text.
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({"gaps": gaps_text})
            data = json.loads(result.content)
            content = data.get("content", "")
        except Exception as e:
            print(f"    Error generating gaps section: {e}")
            content = "Several research gaps have been identified in the literature."
        
        return LiteratureSection(
            section_title="Research Gaps and Future Directions",
            section_content=content,
            key_papers=[],
            related_gaps=[g.get('gap_id', '') for g in research_gaps[:10]]
        )
    
    def _generate_synthesis(
        self,
        research_gaps: List[Dict],
        knowledge_graph: Dict
    ) -> str:
        """Generate synthesis and conclusions."""
        
        gaps_summary = self._prepare_gaps_summary(research_gaps[:10])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at writing academic literature reviews."),
            ("user", """Write a synthesis and conclusions section (2-3 paragraphs) for a literature review.
            
            Research Gaps:
            {gaps}
            
            The synthesis should:
            - Summarize key themes and findings
            - Highlight the most critical gaps
            - Provide recommendations for future research
            - Discuss broader implications
            
            Return as a JSON object with a "synthesis" field containing the text.
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({"gaps": gaps_summary})
            data = json.loads(result.content)
            return data.get("synthesis", "")
        
        except Exception as e:
            print(f"    Error generating synthesis: {e}")
            return "This review has identified several important research gaps and future directions."
    
    def _generate_research_plan(
        self,
        research_gaps: List[Dict],
        paper_structures: List[Dict],
        knowledge_graph: Dict
    ) -> ResearchPlan:
        """Generate research plan based on gaps."""
        
        print(f"\n  [{self.agent_name}] Generating research plan...")
        
        # Generate plan overview
        overview = self._generate_plan_overview(research_gaps)
        
        # Generate research objectives
        objectives = self._generate_research_objectives(research_gaps[:10])
        
        # Generate methodology overview
        methodology_overview = self._generate_methodology_overview(paper_structures, research_gaps)
        
        # Generate expected contributions
        expected_contributions = self._generate_expected_contributions(research_gaps)
        
        # Generate timeline and resources
        timeline_summary = self._generate_timeline_summary(objectives)
        resource_requirements = self._generate_resource_requirements(objectives)
        
        plan = ResearchPlan(
            plan_title="Research Plan: Addressing Key Gaps",
            overview=overview,
            research_objectives=objectives,
            methodology_overview=methodology_overview,
            expected_contributions=expected_contributions,
            timeline_summary=timeline_summary,
            resource_requirements=resource_requirements
        )
        
        print(f"  [{self.agent_name}] Research plan generated ({len(objectives)} objectives)")
        
        return plan
    
    def _generate_plan_overview(self, research_gaps: List[Dict]) -> str:
        """Generate research plan overview."""
        
        gaps_text = "\n".join([
            f"- {g.get('gap_title', 'Unknown')}"
            for g in research_gaps[:5]
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at writing research plans."),
            ("user", """Write an overview (2-3 paragraphs) for a research plan that addresses these gaps:
            
            {gaps}
            
            The overview should:
            - State the plan's purpose and scope
            - Explain how it addresses the gaps
            - Outline the expected impact
            
            Return as a JSON object with an "overview" field.
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({"gaps": gaps_text})
            data = json.loads(result.content)
            return data.get("overview", "")
        except Exception as e:
            print(f"    Error generating plan overview: {e}")
            return "This research plan addresses key gaps identified in the literature."
    
    def _generate_research_objectives(
        self,
        research_gaps: List[Dict]
    ) -> List[ResearchObjective]:
        """Generate research objectives from gaps."""
        
        objectives = []
        
        # Group gaps by type
        gap_types = {}
        for gap in research_gaps:
            gap_type = gap.get('gap_type', 'unknown')
            if gap_type not in gap_types:
                gap_types[gap_type] = []
            gap_types[gap_type].append(gap)
        
        # Create objectives for each gap type
        for idx, (gap_type, gaps) in enumerate(gap_types.items(), 1):
            gap_titles = [g.get('gap_title', 'Unknown') for g in gaps[:3]]
            gap_ids = [g.get('gap_id', '') for g in gaps[:3]]
            
            # Determine priority based on severity
            severities = [g.get('severity', 'Low') for g in gaps]
            priority = "High" if "Critical" in severities or "High" in severities else "Medium"
            
            objective = ResearchObjective(
                objective_id=f"OBJ_{idx:03d}",
                objective_title=f"Address {gap_type.capitalize()} Gaps",
                description=f"This objective focuses on addressing {gap_type} gaps including: {', '.join(gap_titles[:2])}.",
                addresses_gaps=gap_ids,
                methodology=gaps[0].get('suggested_approaches', [])[:3] if gaps else [],
                expected_outcomes=[
                    f"Improved understanding of {gap_type} aspects",
                    "Novel methodological contributions",
                    "Empirical validation of findings"
                ],
                timeline="12-18 months",
                priority=priority
            )
            
            objectives.append(objective)
        
        return objectives
    
    def _generate_methodology_overview(
        self,
        paper_structures: List[Dict],
        research_gaps: List[Dict]
    ) -> str:
        """Generate methodology overview."""
        
        # Collect methodologies from papers
        all_methods = set()
        for paper in paper_structures[:20]:
            all_methods.update(paper.get('methodologies', []))
        
        methods_text = ", ".join(list(all_methods)[:5])
        
        return f"The research will employ a mixed-methods approach, building on established methodologies including {methods_text}. Novel methodological contributions will address identified gaps in current approaches."
    
    def _generate_expected_contributions(self, research_gaps: List[Dict]) -> List[str]:
        """Generate expected contributions."""
        
        contributions = [
            "Comprehensive literature synthesis addressing key research gaps",
            "Novel methodological frameworks for future research",
            "Empirical evidence to validate theoretical propositions"
        ]
        
        # Add gap-specific contributions
        for gap in research_gaps[:3]:
            impact = gap.get('potential_impact', '')
            if impact:
                contributions.append(impact)
        
        return contributions[:5]
    
    def _generate_timeline_summary(self, objectives: List[ResearchObjective]) -> str:
        """Generate timeline summary."""
        
        high_priority = sum(1 for obj in objectives if obj.priority == "High")
        total = len(objectives)
        
        return f"The research plan spans 18-24 months, with {high_priority} high-priority objectives to be completed in the first 12 months, followed by {total - high_priority} medium-priority objectives."
    
    def _generate_resource_requirements(self, objectives: List[ResearchObjective]) -> List[str]:
        """Generate resource requirements."""
        
        return [
            "Access to academic databases and literature",
            "Computational resources for data analysis",
            "Research team with expertise in relevant methodologies",
            "Collaboration with domain experts",
            "Funding for data collection and dissemination"
        ]
    
    def _prepare_papers_summary(self, paper_structures: List[Dict]) -> str:
        """Prepare summary of papers."""
        
        if not paper_structures:
            return "No papers available for summary."
        
        summary_parts = []
        for paper in paper_structures[:20]:
            summary = f"- {paper.get('paper_title', 'Unknown')}: {', '.join(paper.get('key_findings', [])[:2])}"
            summary_parts.append(summary)
        
        return "\n".join(summary_parts) if summary_parts else "No paper summaries available."
    
    def _prepare_gaps_summary(self, research_gaps: List[Dict]) -> str:
        """Prepare summary of gaps."""
        
        if not research_gaps:
            return "No research gaps identified."
        
        summary_parts = []
        for gap in research_gaps:
            summary = f"- [{gap.get('gap_type', 'unknown').upper()}] {gap.get('gap_title', 'Unknown')}: {gap.get('gap_description', '')[:100]}..."
            summary_parts.append(summary)
        
        return "\n".join(summary_parts) if summary_parts else "No gap summaries available."


class CiteKeeper:
    """Agent for reference finalization and bibliography management."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "CiteKeeper"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=self.api_key
        )
    
    def finalize_references(
        self,
        gap_analysis: Dict,
        literature_review: LiteratureReview,
        research_plan: ResearchPlan
    ) -> Bibliography:
        """Finalize references and create formatted bibliography."""
        
        print(f"\n[{self.agent_name}] Finalizing references and bibliography...")
        
        # Extract all cited papers
        cited_papers = self._extract_cited_papers(literature_review, research_plan)
        
        # Get paper details from gap analysis
        paper_structures = gap_analysis.get('paper_structures', [])
        
        # Create formatted references
        references = self._create_formatted_references(cited_papers, paper_structures)
        
        # Sort references alphabetically
        references.sort(key=lambda x: x.citation_key)
        
        # Create bibliography statistics
        by_type = {}
        by_importance = {}
        
        for ref in references:
            by_type[ref.reference_type] = by_type.get(ref.reference_type, 0) + 1
            by_importance[ref.importance] = by_importance.get(ref.importance, 0) + 1
        
        bibliography = Bibliography(
            references=references,
            total_references=len(references),
            by_type=by_type,
            by_importance=by_importance
        )
        
        print(f"[{self.agent_name}] Bibliography finalized ({len(references)} references)")
        
        return bibliography
    
    def _extract_cited_papers(
        self,
        literature_review: LiteratureReview,
        research_plan: ResearchPlan
    ) -> set:
        """Extract all cited papers from review and plan."""
        
        cited_papers = set()
        
        # From review sections
        for section in literature_review.sections:
            cited_papers.update(section.key_papers)
        
        return cited_papers
    
    def _create_formatted_references(
        self,
        cited_papers: set,
        paper_structures: List[Dict]
    ) -> List[FormattedReference]:
        """Create formatted references for cited papers."""
        
        references = []
        
        # Create a mapping of paper titles to structures
        paper_map = {p.get('paper_title', ''): p for p in paper_structures}
        
        for paper_title in cited_papers:
            if not paper_title:
                continue
            
            paper = paper_map.get(paper_title)
            if not paper:
                # Create basic reference if paper not found
                ref = FormattedReference(
                    citation_key=self._generate_citation_key(paper_title, None),
                    formatted_citation=f"{paper_title}. (n.d.).",
                    reference_type="unknown",
                    importance="Medium"
                )
                references.append(ref)
                continue
            
            # Generate citation key
            citation_key = self._generate_citation_key(
                paper_title,
                paper.get('research_objectives', [])
            )
            
            # Format citation (simplified APA style)
            formatted_citation = self._format_citation(paper)
            
            # Determine reference type
            ref_type = self._determine_reference_type(paper)
            
            # Determine importance
            importance = self._determine_importance(paper)
            
            ref = FormattedReference(
                citation_key=citation_key,
                formatted_citation=formatted_citation,
                reference_type=ref_type,
                importance=importance
            )
            
            references.append(ref)
        
        return references
    
    def _generate_citation_key(self, paper_title: str, objectives: Optional[List[str]]) -> str:
        """Generate citation key from paper title."""
        
        # Extract first word from title
        words = paper_title.split()
        first_word = words[0] if words else "Unknown"
        
        # Clean and format
        key = first_word.strip(',:;.').capitalize()
        
        return f"{key}2024"  # Simplified - would use actual year
    
    def _format_citation(self, paper: Dict) -> str:
        """Format citation in APA style."""
        
        title = paper.get('paper_title', 'Unknown')
        
        # Simplified formatting
        citation = f"{title}. (2024). Research Paper."
        
        return citation
    
    def _determine_reference_type(self, paper: Dict) -> str:
        """Determine reference type."""
        
        methodologies = paper.get('methodologies', [])
        
        if any('empirical' in m.lower() for m in methodologies):
            return "journal"
        elif any('review' in m.lower() for m in methodologies):
            return "review"
        else:
            return "conference"
    
    def _determine_importance(self, paper: Dict) -> str:
        """Determine reference importance."""
        
        findings = paper.get('key_findings', [])
        contributions = paper.get('theoretical_contributions', [])
        
        if len(findings) >= 4 or len(contributions) >= 2:
            return "High"
        elif len(findings) >= 2 or len(contributions) >= 1:
            return "Medium"
        else:
            return "Low"


# ========== ORCHESTRATOR ==========

class SynthesisOrchestrator:
    """Orchestrates the synthesis pipeline."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        
        # Initialize agents
        self.knowledge_weaver = KnowledgeWeaver(self.api_key)
        self.cite_keeper = CiteKeeper(self.api_key)
        
        self.synthesis_result: Optional[SynthesisResult] = None
    
    def run_synthesis_pipeline(
        self,
        gap_analysis: Dict
    ) -> SynthesisResult:
        """Run the complete synthesis pipeline."""
        
        print(f"\n{'='*70}")
        print(f"SYNTHESIS PIPELINE")
        print(f"{'='*70}")
        print(f"Input: {gap_analysis.get('metadata', {}).get('total_gaps_identified', 0)} gaps")
        print(f"{'='*70}\n")
        
        # Step 1: KnowledgeWeaver - Knowledge integration
        literature_review, research_plan = self.knowledge_weaver.integrate_knowledge(gap_analysis)
        
        # Step 2: CiteKeeper - Reference finalization
        bibliography = self.cite_keeper.finalize_references(
            gap_analysis,
            literature_review,
            research_plan
        )
        
        # Create synthesis result
        self.synthesis_result = SynthesisResult(
            literature_review=literature_review,
            research_plan=research_plan,
            bibliography=bibliography,
            metadata={
                "created_at": datetime.now().isoformat(),
                "review_sections": len(literature_review.sections),
                "research_objectives": len(research_plan.research_objectives),
                "total_references": bibliography.total_references
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Literature Review: {len(literature_review.sections)} sections")
        print(f"Research Plan: {len(research_plan.research_objectives)} objectives")
        print(f"Bibliography: {bibliography.total_references} references")
        print(f"{'='*70}\n")
        
        return self.synthesis_result
    
    def load_gap_analysis(self, filepath: str) -> Dict:
        """Load gap analysis from Stage 2."""
        print(f"[Orchestrator] Loading gap analysis from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[Orchestrator] Loaded analysis with {data.get('metadata', {}).get('total_gaps_identified', 0)} gaps")
        return data
    
    def save_synthesis_result(self, filepath: str):
        """Save the complete synthesis result to JSON."""
        if not self.synthesis_result:
            print("No synthesis result to save.")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.synthesis_result.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"[Orchestrator] Synthesis result saved to: {filepath}")
    
    def save_literature_review(self, filepath: str):
        """Save literature review as formatted text."""
        if not self.synthesis_result:
            print("No synthesis result to save.")
            return
        
        review = self.synthesis_result.literature_review
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{review.title}\n")
            f.write("="*len(review.title) + "\n\n")
            
            f.write("ABSTRACT\n")
            f.write("-"*70 + "\n")
            f.write(f"{review.abstract}\n\n")
            
            f.write("INTRODUCTION\n")
            f.write("-"*70 + "\n")
            f.write(f"{review.introduction}\n\n")
            
            for section in review.sections:
                f.write(f"{section.section_title.upper()}\n")
                f.write("-"*70 + "\n")
                f.write(f"{section.section_content}\n\n")
            
            f.write("SYNTHESIS AND CONCLUSIONS\n")
            f.write("-"*70 + "\n")
            f.write(f"{review.synthesis}\n")
        
        print(f"[Orchestrator] Literature review saved to: {filepath}")
    
    def save_research_plan(self, filepath: str):
        """Save research plan as formatted text."""
        if not self.synthesis_result:
            print("No synthesis result to save.")
            return
        
        plan = self.synthesis_result.research_plan
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{plan.plan_title}\n")
            f.write("="*len(plan.plan_title) + "\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"{plan.overview}\n\n")
            
            f.write("RESEARCH OBJECTIVES\n")
            f.write("-"*70 + "\n")
            for obj in plan.research_objectives:
                f.write(f"\n{obj.objective_id}: {obj.objective_title} (Priority: {obj.priority})\n")
                f.write(f"{obj.description}\n")
                f.write(f"Methodology: {', '.join(obj.methodology)}\n")
                f.write(f"Timeline: {obj.timeline}\n")
            
            f.write(f"\nMETHODOLOGY OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"{plan.methodology_overview}\n\n")
            
            f.write("EXPECTED CONTRIBUTIONS\n")
            f.write("-"*70 + "\n")
            for contrib in plan.expected_contributions:
                f.write(f"- {contrib}\n")
            
            f.write(f"\nTIMELINE\n")
            f.write("-"*70 + "\n")
            f.write(f"{plan.timeline_summary}\n\n")
            
            f.write("RESOURCE REQUIREMENTS\n")
            f.write("-"*70 + "\n")
            for resource in plan.resource_requirements:
                f.write(f"- {resource}\n")
        
        print(f"[Orchestrator] Research plan saved to: {filepath}")
    
    def save_bibliography(self, filepath: str):
        """Save bibliography as formatted text."""
        if not self.synthesis_result:
            print("No synthesis result to save.")
            return
        
        bib = self.synthesis_result.bibliography
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("BIBLIOGRAPHY\n")
            f.write("="*70 + "\n\n")
            
            for ref in bib.references:
                f.write(f"[{ref.citation_key}] {ref.formatted_citation}\n\n")
        
        print(f"[Orchestrator] Bibliography saved to: {filepath}")
    
    def print_summary(self):
        """Print a summary of the synthesis."""
        if not self.synthesis_result:
            print("No synthesis result available.")
            return
        
        result = self.synthesis_result
        
        print(f"\n{'='*70}")
        print(f"SYNTHESIS SUMMARY")
        print(f"{'='*70}\n")
        
        # Literature review
        print(f"Literature Review:")
        print(f"  Title: {result.literature_review.title}")
        print(f"  Sections: {len(result.literature_review.sections)}")
        print(f"  Total Papers: {result.literature_review.metadata.get('total_papers_reviewed', 0)}")
        
        # Research plan
        print(f"\nResearch Plan:")
        print(f"  Objectives: {len(result.research_plan.research_objectives)}")
        high_priority = sum(1 for obj in result.research_plan.research_objectives if obj.priority == "High")
        print(f"  High Priority: {high_priority}")
        
        # Bibliography
        print(f"\nBibliography:")
        print(f"  Total References: {result.bibliography.total_references}")
        print(f"  By Type: {result.bibliography.by_type}")
        print(f"  By Importance: {result.bibliography.by_importance}")
        
        print(f"\n{'='*70}\n")


# ========== MAIN ==========

def main():
    """Main function to run the synthesis stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Initialize orchestrator
    orchestrator = SynthesisOrchestrator()
    
    # Load gap analysis from Stage 2
    gap_file = input("Enter path to gap analysis JSON (or press Enter for default): ").strip()
    if not gap_file:
        gap_file = "gap_analysis_results.json"
    
    try:
        gap_analysis = orchestrator.load_gap_analysis(gap_file)
    except Exception as e:
        print(f"Error loading gap analysis: {e}")
        print("Please run Stage 2 first to generate the gap analysis.")
        return
    
    # Run pipeline
    synthesis_result = orchestrator.run_synthesis_pipeline(gap_analysis)
    
    # Save results
    orchestrator.save_synthesis_result("synthesis_results.json")
    orchestrator.save_literature_review("literature_review.txt")
    orchestrator.save_research_plan("research_plan.txt")
    orchestrator.save_bibliography("bibliography.txt")
    
    # Print summary
    orchestrator.print_summary()
    
    print("\n" + "="*70)
    print("Synthesis complete!")
    print("Output files:")
    print("  - synthesis_results.json (complete synthesis)")
    print("  - literature_review.txt (formatted review)")
    print("  - research_plan.txt (formatted plan)")
    print("  - bibliography.txt (formatted references)")
    print("="*70)


if __name__ == "__main__":
    main()
