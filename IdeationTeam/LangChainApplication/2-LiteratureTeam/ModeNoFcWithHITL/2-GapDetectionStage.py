"""
Gap Detection Stage - Automated Mode (No Firecrawl, No HITL)
This script analyzes literature from Stage 1 to detect research gaps and construct knowledge graphs.

Pipeline:
1. PaperDecomposer: Structural analysis of papers (methods, findings, limitations)
2. GapFinder: Research opportunity detection (methodological, theoretical, empirical gaps)
3. KnowledgeWeaver: Graph construction (relationships between papers, concepts, gaps)

Input: Literature batch from Stage 1 (literature_batch.json)
Output: Analyzed literature with research gaps and knowledge graph
"""

import os
import json
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# ========== DATA MODELS ==========

class PaperStructure(BaseModel):
    """Model for decomposed paper structure."""
    paper_title: str = Field(description="Title of the paper")
    research_objectives: List[str] = Field(description="Main research objectives")
    methodologies: List[str] = Field(description="Methodologies used")
    key_findings: List[str] = Field(description="Key findings (3-5 points)")
    theoretical_contributions: List[str] = Field(description="Theoretical contributions")
    limitations: List[str] = Field(description="Stated limitations")
    future_work_suggestions: List[str] = Field(description="Suggested future work")
    data_sources: List[str] = Field(description="Data sources used", default_factory=list)
    assumptions: List[str] = Field(description="Key assumptions", default_factory=list)


class ResearchGap(BaseModel):
    """Model for identified research gap."""
    gap_id: str = Field(description="Unique gap identifier")
    gap_type: str = Field(description="Type: methodological, theoretical, empirical, data")
    gap_title: str = Field(description="Concise title for the gap")
    gap_description: str = Field(description="Detailed description")
    evidence_papers: List[str] = Field(description="Papers that reveal this gap")
    severity: str = Field(description="Critical/High/Medium/Low")
    addressability: str = Field(description="Easy/Moderate/Difficult")
    related_trends: List[str] = Field(description="Related trend names")
    potential_impact: str = Field(description="Expected impact if addressed")
    suggested_approaches: List[str] = Field(description="Suggested research approaches")


class KnowledgeNode(BaseModel):
    """Model for a node in the knowledge graph."""
    node_id: str = Field(description="Unique node identifier")
    node_type: str = Field(description="Type: paper, concept, method, gap, trend")
    label: str = Field(description="Node label/name")
    properties: Dict = Field(description="Additional properties", default_factory=dict)


class KnowledgeEdge(BaseModel):
    """Model for an edge in the knowledge graph."""
    source_id: str = Field(description="Source node ID")
    target_id: str = Field(description="Target node ID")
    relationship: str = Field(description="Relationship type: cites, uses_method, addresses_gap, etc.")
    weight: Optional[float] = Field(description="Edge weight/strength", default=1.0)
    properties: Dict = Field(description="Additional properties", default_factory=dict)


class KnowledgeGraph(BaseModel):
    """Model for the complete knowledge graph."""
    nodes: List[KnowledgeNode] = Field(description="Graph nodes")
    edges: List[KnowledgeEdge] = Field(description="Graph edges")
    metadata: Dict = Field(description="Graph metadata", default_factory=dict)


class GapAnalysisResult(BaseModel):
    """Model for the complete gap analysis output."""
    paper_structures: List[PaperStructure] = Field(description="Decomposed paper structures")
    research_gaps: List[ResearchGap] = Field(description="Identified research gaps")
    knowledge_graph: KnowledgeGraph = Field(description="Constructed knowledge graph")
    gap_summary: Dict = Field(description="Summary statistics", default_factory=dict)
    metadata: Dict = Field(description="Analysis metadata", default_factory=dict)


# ========== AGENTS ==========

class PaperDecomposer:
    """Agent for structural analysis of papers."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "PaperDecomposer"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=self.api_key
        )
    
    def decompose_papers(
        self,
        literature_batch: Dict
    ) -> List[PaperStructure]:
        """Decompose papers into structural components."""
        literature_items = literature_batch.get('literature_items', [])
        
        print(f"\n[{self.agent_name}] Decomposing {len(literature_items)} papers...")
        
        paper_structures = []
        
        # Process papers in batches
        batch_size = 5
        for i in range(0, len(literature_items), batch_size):
            batch = literature_items[i:i+batch_size]
            
            print(f"  Processing papers {i+1}-{min(i+batch_size, len(literature_items))}...")
            
            for paper in batch:
                structure = self._decompose_single_paper(paper)
                if structure:
                    paper_structures.append(structure)
        
        print(f"[{self.agent_name}] Decomposed {len(paper_structures)} papers")
        return paper_structures
    
    def _decompose_single_paper(self, paper: Dict) -> Optional[PaperStructure]:
        """Decompose a single paper into structural components."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing academic papers and extracting their structural components."),
            ("user", """Analyze the following paper and extract its structural components.
            
            Title: {title}
            Authors: {authors}
            Year: {year}
            Abstract: {abstract}
            
            Extract the following:
            - research_objectives: Main research objectives (2-4 points)
            - methodologies: Methodologies used (2-5 methods)
            - key_findings: Key findings (3-5 points)
            - theoretical_contributions: Theoretical contributions (1-3 points)
            - limitations: Stated or implied limitations (2-4 points)
            - future_work_suggestions: Suggested future work (1-3 points)
            - data_sources: Data sources mentioned (if any)
            - assumptions: Key assumptions (if any)
            
            Return as a JSON object with these fields.
            Example: {{
              "paper_title": "...",
              "research_objectives": ["obj1", "obj2"],
              "methodologies": ["method1", "method2"],
              ...
            }}
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "title": paper.get('title', 'Unknown'),
                "authors": ', '.join(paper.get('authors', [])[:3]),
                "year": paper.get('year', 'N/A'),
                "abstract": paper.get('abstract', 'No abstract available')[:500]
            })
            
            structure_data = json.loads(result.content)
            structure = PaperStructure(**structure_data)
            
            return structure
        
        except Exception as e:
            print(f"    Error decomposing paper '{paper.get('title', 'Unknown')[:50]}...': {e}")
            return None


class GapFinder:
    """Agent for research opportunity detection."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "GapFinder"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.6,
            openai_api_key=self.api_key
        )
    
    def find_gaps(
        self,
        paper_structures: List[PaperStructure],
        literature_batch: Dict
    ) -> List[ResearchGap]:
        """Identify research gaps from paper structures."""
        
        print(f"\n[{self.agent_name}] Identifying research gaps from {len(paper_structures)} papers...")
        
        # Prepare analysis context
        trends = literature_batch.get('trend_analyses', [])
        insights = literature_batch.get('insights', [])
        
        # Identify different types of gaps
        methodological_gaps = self._find_methodological_gaps(paper_structures)
        theoretical_gaps = self._find_theoretical_gaps(paper_structures, insights)
        empirical_gaps = self._find_empirical_gaps(paper_structures, trends)
        data_gaps = self._find_data_gaps(paper_structures)
        
        all_gaps = methodological_gaps + theoretical_gaps + empirical_gaps + data_gaps
        
        # Deduplicate and rank gaps
        unique_gaps = self._deduplicate_gaps(all_gaps)
        ranked_gaps = self._rank_gaps(unique_gaps)
        
        print(f"[{self.agent_name}] Identified {len(ranked_gaps)} unique research gaps")
        print(f"  - Methodological: {len(methodological_gaps)}")
        print(f"  - Theoretical: {len(theoretical_gaps)}")
        print(f"  - Empirical: {len(empirical_gaps)}")
        print(f"  - Data: {len(data_gaps)}")
        
        return ranked_gaps
    
    def _find_methodological_gaps(self, paper_structures: List[PaperStructure]) -> List[ResearchGap]:
        """Find methodological gaps."""
        
        # Collect all methodologies and limitations
        all_methods = []
        all_limitations = []
        
        for paper in paper_structures:
            all_methods.extend(paper.methodologies)
            all_limitations.extend(paper.limitations)
        
        # Check if we have data to analyze
        if not all_methods and not all_limitations:
            return []
        
        # Prepare summary
        methods_summary = "\n".join([f"- {m}" for m in list(set(all_methods))[:20]]) if all_methods else "No methodologies found"
        limitations_summary = "\n".join([f"- {l}" for l in all_limitations[:30]]) if all_limitations else "No limitations found"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying methodological gaps in research."),
            ("user", """Based on the methodologies used and limitations stated, identify 3-5 methodological gaps.
            
            Methodologies Used:
            {methods}
            
            Stated Limitations:
            {limitations}
            
            For each gap, provide:
            - gap_id: Unique ID (e.g., "METH_GAP_001")
            - gap_type: "methodological"
            - gap_title: Concise title
            - gap_description: Detailed description (2-3 sentences)
            - evidence_papers: List of paper titles (use "Multiple papers" if general)
            - severity: "Critical", "High", "Medium", or "Low"
            - addressability: "Easy", "Moderate", or "Difficult"
            - related_trends: Empty list for now
            - potential_impact: Expected impact if addressed
            - suggested_approaches: List of 2-3 suggested approaches
            
            Return as a JSON array of gap objects.
            
            Respond with ONLY the JSON array, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "methods": methods_summary,
                "limitations": limitations_summary
            })
            
            gaps_data = json.loads(result.content)
            gaps = [ResearchGap(**gap) for gap in gaps_data]
            
            return gaps
        
        except Exception as e:
            print(f"    Error finding methodological gaps: {e}")
            return []
    
    def _find_theoretical_gaps(
        self,
        paper_structures: List[PaperStructure],
        insights: List[Dict]
    ) -> List[ResearchGap]:
        """Find theoretical gaps."""
        
        # Collect theoretical contributions
        all_contributions = []
        for paper in paper_structures:
            all_contributions.extend(paper.theoretical_contributions)
        
        # Check if we have data to analyze
        if not all_contributions and not insights:
            return []
        
        # Prepare insights summary
        insights_summary = "\n".join([
            f"- {ins.get('insight_title', 'N/A')}: {ins.get('insight_description', 'N/A')[:100]}..."
            for ins in insights[:10]
        ]) if insights else "No insights available"
        
        contributions_summary = "\n".join([f"- {c}" for c in all_contributions[:20]]) if all_contributions else "No theoretical contributions found"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying theoretical gaps in research."),
            ("user", """Based on theoretical contributions and insights, identify 2-4 theoretical gaps.
            
            Theoretical Contributions:
            {contributions}
            
            Research Insights:
            {insights}
            
            For each gap, provide the same structure as before with gap_type: "theoretical".
            
            Return as a JSON array of gap objects.
            
            Respond with ONLY the JSON array, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "contributions": contributions_summary,
                "insights": insights_summary
            })
            
            gaps_data = json.loads(result.content)
            gaps = [ResearchGap(**gap) for gap in gaps_data]
            
            return gaps
        
        except Exception as e:
            print(f"    Error finding theoretical gaps: {e}")
            return []
    
    def _find_empirical_gaps(
        self,
        paper_structures: List[PaperStructure],
        trends: List[Dict]
    ) -> List[ResearchGap]:
        """Find empirical gaps."""
        
        # Collect findings and future work
        all_findings = []
        all_future_work = []
        
        for paper in paper_structures:
            all_findings.extend(paper.key_findings)
            all_future_work.extend(paper.future_work_suggestions)
        
        # Check if we have data to analyze
        if not all_future_work and not trends:
            return []
        
        trends_summary = "\n".join([
            f"- {t.get('trend_name', 'N/A')}: {t.get('description', 'N/A')[:100]}..."
            for t in trends[:7]
        ]) if trends else "No trends available"
        
        future_work_summary = "\n".join([f"- {fw}" for fw in all_future_work[:20]]) if all_future_work else "No future work suggestions found"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying empirical gaps in research."),
            ("user", """Based on future work suggestions and trends, identify 2-4 empirical gaps.
            
            Future Work Suggestions:
            {future_work}
            
            Research Trends:
            {trends}
            
            For each gap, provide the same structure with gap_type: "empirical".
            
            Return as a JSON array of gap objects.
            
            Respond with ONLY the JSON array, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "future_work": future_work_summary,
                "trends": trends_summary
            })
            
            gaps_data = json.loads(result.content)
            gaps = [ResearchGap(**gap) for gap in gaps_data]
            
            return gaps
        
        except Exception as e:
            print(f"    Error finding empirical gaps: {e}")
            return []
    
    def _find_data_gaps(self, paper_structures: List[PaperStructure]) -> List[ResearchGap]:
        """Find data-related gaps."""
        
        # Collect data sources
        all_data_sources = []
        for paper in paper_structures:
            all_data_sources.extend(paper.data_sources)
        
        if not all_data_sources:
            return []
        
        data_summary = "\n".join([f"- {ds}" for ds in list(set(all_data_sources))[:15]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying data gaps in research."),
            ("user", """Based on data sources used, identify 1-3 data gaps.
            
            Data Sources Used:
            {data_sources}
            
            For each gap, provide the same structure with gap_type: "data".
            
            Return as a JSON array of gap objects.
            
            Respond with ONLY the JSON array, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "data_sources": data_summary
            })
            
            gaps_data = json.loads(result.content)
            gaps = [ResearchGap(**gap) for gap in gaps_data]
            
            return gaps
        
        except Exception as e:
            print(f"    Error finding data gaps: {e}")
            return []
    
    def _deduplicate_gaps(self, gaps: List[ResearchGap]) -> List[ResearchGap]:
        """Remove duplicate gaps based on similarity."""
        unique_gaps = []
        seen_titles = set()
        
        for gap in gaps:
            normalized_title = gap.gap_title.lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_gaps.append(gap)
        
        return unique_gaps
    
    def _rank_gaps(self, gaps: List[ResearchGap]) -> List[ResearchGap]:
        """Rank gaps by severity and addressability."""
        
        severity_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        addressability_order = {"Easy": 3, "Moderate": 2, "Difficult": 1}
        
        def gap_score(gap: ResearchGap) -> Tuple[int, int]:
            sev = severity_order.get(gap.severity, 0)
            addr = addressability_order.get(gap.addressability, 0)
            return (sev, addr)
        
        gaps.sort(key=gap_score, reverse=True)
        return gaps


class KnowledgeWeaver:
    """Agent for knowledge graph construction."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "KnowledgeWeaver"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            openai_api_key=self.api_key
        )
    
    def construct_graph(
        self,
        literature_batch: Dict,
        paper_structures: List[PaperStructure],
        research_gaps: List[ResearchGap]
    ) -> KnowledgeGraph:
        """Construct knowledge graph from literature and gaps."""
        
        print(f"\n[{self.agent_name}] Constructing knowledge graph...")
        
        nodes = []
        edges = []
        
        # Create nodes for papers
        paper_nodes = self._create_paper_nodes(literature_batch.get('literature_items', []))
        nodes.extend(paper_nodes)
        
        # Create nodes for concepts/methods
        concept_nodes = self._create_concept_nodes(paper_structures)
        nodes.extend(concept_nodes)
        
        # Create nodes for trends
        trend_nodes = self._create_trend_nodes(literature_batch.get('trend_analyses', []))
        nodes.extend(trend_nodes)
        
        # Create nodes for gaps
        gap_nodes = self._create_gap_nodes(research_gaps)
        nodes.extend(gap_nodes)
        
        # Create edges between papers and methods
        method_edges = self._create_method_edges(paper_structures, paper_nodes, concept_nodes)
        edges.extend(method_edges)
        
        # Create edges between papers and gaps
        gap_edges = self._create_gap_edges(research_gaps, paper_nodes, gap_nodes)
        edges.extend(gap_edges)
        
        # Create edges between papers and trends
        trend_edges = self._create_trend_edges(
            literature_batch.get('citations', []),
            paper_nodes,
            trend_nodes
        )
        edges.extend(trend_edges)
        
        # Create citation edges (if available)
        citation_edges = self._create_citation_edges(paper_nodes)
        edges.extend(citation_edges)
        
        graph = KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "node_types": self._count_node_types(nodes),
                "edge_types": self._count_edge_types(edges),
                "created_at": datetime.now().isoformat()
            }
        )
        
        print(f"[{self.agent_name}] Graph constructed:")
        print(f"  - Nodes: {len(nodes)}")
        print(f"  - Edges: {len(edges)}")
        print(f"  - Node types: {graph.metadata['node_types']}")
        
        return graph
    
    def _create_paper_nodes(self, literature_items: List[Dict]) -> List[KnowledgeNode]:
        """Create nodes for papers."""
        nodes = []
        
        for idx, paper in enumerate(literature_items):
            node = KnowledgeNode(
                node_id=f"PAPER_{idx:03d}",
                node_type="paper",
                label=paper.get('title', 'Unknown'),
                properties={
                    "authors": paper.get('authors', []),
                    "year": paper.get('year'),
                    "citation_count": paper.get('citation_count'),
                    "source": paper.get('source'),
                    "url": paper.get('url')
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _create_concept_nodes(self, paper_structures: List[PaperStructure]) -> List[KnowledgeNode]:
        """Create nodes for concepts and methods."""
        nodes = []
        seen_concepts = set()
        
        # Extract unique methods
        for paper in paper_structures:
            for method in paper.methodologies:
                if method and method not in seen_concepts:
                    seen_concepts.add(method)
                    node = KnowledgeNode(
                        node_id=f"METHOD_{len(nodes):03d}",
                        node_type="method",
                        label=method,
                        properties={}
                    )
                    nodes.append(node)
        
        return nodes
    
    def _create_trend_nodes(self, trends: List[Dict]) -> List[KnowledgeNode]:
        """Create nodes for trends."""
        nodes = []
        
        for idx, trend in enumerate(trends):
            node = KnowledgeNode(
                node_id=f"TREND_{idx:03d}",
                node_type="trend",
                label=trend.get('trend_name', 'Unknown'),
                properties={
                    "description": trend.get('description'),
                    "emergence_year": trend.get('emergence_year'),
                    "growth_trajectory": trend.get('growth_trajectory')
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _create_gap_nodes(self, research_gaps: List[ResearchGap]) -> List[KnowledgeNode]:
        """Create nodes for research gaps."""
        nodes = []
        
        for gap in research_gaps:
            node = KnowledgeNode(
                node_id=gap.gap_id,
                node_type="gap",
                label=gap.gap_title,
                properties={
                    "gap_type": gap.gap_type,
                    "severity": gap.severity,
                    "addressability": gap.addressability,
                    "description": gap.gap_description
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _create_method_edges(
        self,
        paper_structures: List[PaperStructure],
        paper_nodes: List[KnowledgeNode],
        method_nodes: List[KnowledgeNode]
    ) -> List[KnowledgeEdge]:
        """Create edges between papers and methods."""
        edges = []
        
        # Create mapping of paper titles to node IDs
        paper_map = {node.label: node.node_id for node in paper_nodes}
        method_map = {node.label: node.node_id for node in method_nodes}
        
        for paper_struct in paper_structures:
            paper_id = paper_map.get(paper_struct.paper_title)
            if not paper_id:
                continue
            
            for method in paper_struct.methodologies:
                method_id = method_map.get(method)
                if method_id:
                    edge = KnowledgeEdge(
                        source_id=paper_id,
                        target_id=method_id,
                        relationship="uses_method",
                        weight=1.0
                    )
                    edges.append(edge)
        
        return edges
    
    def _create_gap_edges(
        self,
        research_gaps: List[ResearchGap],
        paper_nodes: List[KnowledgeNode],
        gap_nodes: List[KnowledgeNode]
    ) -> List[KnowledgeEdge]:
        """Create edges between papers and gaps."""
        edges = []
        
        paper_map = {node.label: node.node_id for node in paper_nodes}
        
        for gap in research_gaps:
            gap_id = gap.gap_id
            
            for paper_title in gap.evidence_papers:
                paper_id = paper_map.get(paper_title)
                if paper_id:
                    edge = KnowledgeEdge(
                        source_id=paper_id,
                        target_id=gap_id,
                        relationship="reveals_gap",
                        weight=1.0
                    )
                    edges.append(edge)
        
        return edges
    
    def _create_trend_edges(
        self,
        citations: List[Dict],
        paper_nodes: List[KnowledgeNode],
        trend_nodes: List[KnowledgeNode]
    ) -> List[KnowledgeEdge]:
        """Create edges between papers and trends."""
        edges = []
        
        paper_map = {node.label: node.node_id for node in paper_nodes}
        trend_map = {node.label: node.node_id for node in trend_nodes}
        
        for citation in citations:
            paper_title = citation.get('paper_title')
            paper_id = paper_map.get(paper_title)
            
            if not paper_id:
                continue
            
            for trend_name in citation.get('related_trends', []):
                trend_id = trend_map.get(trend_name)
                if trend_id:
                    edge = KnowledgeEdge(
                        source_id=paper_id,
                        target_id=trend_id,
                        relationship="relates_to_trend",
                        weight=1.0
                    )
                    edges.append(edge)
        
        return edges
    
    def _create_citation_edges(self, paper_nodes: List[KnowledgeNode]) -> List[KnowledgeEdge]:
        """Create citation edges between papers (placeholder - would need citation data)."""
        # This is a placeholder - in a real implementation, you'd extract citation relationships
        return []
    
    def _count_node_types(self, nodes: List[KnowledgeNode]) -> Dict[str, int]:
        """Count nodes by type."""
        counts = {}
        for node in nodes:
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts
    
    def _count_edge_types(self, edges: List[KnowledgeEdge]) -> Dict[str, int]:
        """Count edges by relationship type."""
        counts = {}
        for edge in edges:
            counts[edge.relationship] = counts.get(edge.relationship, 0) + 1
        return counts


# ========== ORCHESTRATOR ==========

class GapDetectionOrchestrator:
    """Orchestrates the gap detection pipeline."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        
        # Initialize agents
        self.paper_decomposer = PaperDecomposer(self.api_key)
        self.gap_finder = GapFinder(self.api_key)
        self.knowledge_weaver = KnowledgeWeaver(self.api_key)
        
        self.gap_analysis_result: Optional[GapAnalysisResult] = None
    
    def run_gap_detection_pipeline(
        self,
        literature_batch: Dict
    ) -> GapAnalysisResult:
        """Run the complete gap detection pipeline."""
        
        print(f"\n{'='*70}")
        print(f"GAP DETECTION PIPELINE")
        print(f"{'='*70}")
        print(f"Input: {literature_batch.get('metadata', {}).get('total_papers', 0)} papers")
        print(f"{'='*70}\n")
        
        # Step 1: PaperDecomposer - Structural analysis
        paper_structures = self.paper_decomposer.decompose_papers(literature_batch)
        
        # Step 2: GapFinder - Research opportunity detection
        research_gaps = self.gap_finder.find_gaps(paper_structures, literature_batch)
        
        # Step 3: KnowledgeWeaver - Graph construction
        knowledge_graph = self.knowledge_weaver.construct_graph(
            literature_batch,
            paper_structures,
            research_gaps
        )
        
        # Create gap summary
        gap_summary = self._create_gap_summary(research_gaps)
        
        # Create result
        self.gap_analysis_result = GapAnalysisResult(
            paper_structures=paper_structures,
            research_gaps=research_gaps,
            knowledge_graph=knowledge_graph,
            gap_summary=gap_summary,
            metadata={
                "created_at": datetime.now().isoformat(),
                "total_papers_analyzed": len(paper_structures),
                "total_gaps_identified": len(research_gaps),
                "graph_nodes": len(knowledge_graph.nodes),
                "graph_edges": len(knowledge_graph.edges)
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Papers Analyzed: {len(paper_structures)}")
        print(f"Gaps Identified: {len(research_gaps)}")
        print(f"Knowledge Graph: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        print(f"{'='*70}\n")
        
        return self.gap_analysis_result
    
    def _create_gap_summary(self, research_gaps: List[ResearchGap]) -> Dict:
        """Create summary statistics for gaps."""
        
        gap_types = {}
        severities = {}
        addressabilities = {}
        
        for gap in research_gaps:
            gap_types[gap.gap_type] = gap_types.get(gap.gap_type, 0) + 1
            severities[gap.severity] = severities.get(gap.severity, 0) + 1
            addressabilities[gap.addressability] = addressabilities.get(gap.addressability, 0) + 1
        
        return {
            "total_gaps": len(research_gaps),
            "by_type": gap_types,
            "by_severity": severities,
            "by_addressability": addressabilities
        }
    
    def load_literature_batch(self, filepath: str) -> Dict:
        """Load literature batch from Stage 1."""
        print(f"[Orchestrator] Loading literature batch from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[Orchestrator] Loaded batch with {data.get('metadata', {}).get('total_papers', 0)} papers")
        return data
    
    def save_gap_analysis(self, filepath: str):
        """Save the complete gap analysis to JSON."""
        if not self.gap_analysis_result:
            print("No gap analysis to save.")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.gap_analysis_result.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"[Orchestrator] Gap analysis saved to: {filepath}")
    
    def save_gaps_csv(self, filepath: str):
        """Save research gaps to CSV for easy viewing."""
        if not self.gap_analysis_result:
            print("No gap analysis to save.")
            return
        
        data = [gap.model_dump() for gap in self.gap_analysis_result.research_gaps]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"[Orchestrator] Gaps CSV saved to: {filepath}")
    
    def save_graph_json(self, filepath: str):
        """Save knowledge graph to JSON."""
        if not self.gap_analysis_result:
            print("No gap analysis to save.")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.gap_analysis_result.knowledge_graph.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"[Orchestrator] Knowledge graph saved to: {filepath}")
    
    def print_summary(self):
        """Print a summary of the gap analysis."""
        if not self.gap_analysis_result:
            print("No gap analysis available.")
            return
        
        result = self.gap_analysis_result
        
        print(f"\n{'='*70}")
        print(f"GAP ANALYSIS SUMMARY")
        print(f"{'='*70}\n")
        
        # Gap summary
        print(f"Research Gaps ({result.gap_summary['total_gaps']}):")
        print(f"  By Type: {result.gap_summary['by_type']}")
        print(f"  By Severity: {result.gap_summary['by_severity']}")
        print(f"  By Addressability: {result.gap_summary['by_addressability']}")
        
        # Top gaps
        print(f"\nTop 10 Research Gaps (by severity):")
        for i, gap in enumerate(result.research_gaps[:10], 1):
            print(f"  {i}. [{gap.gap_type.upper()}] {gap.gap_title}")
            print(f"     Severity: {gap.severity} | Addressability: {gap.addressability}")
            print(f"     {gap.gap_description[:100]}...")
        
        # Knowledge graph
        print(f"\nKnowledge Graph:")
        print(f"  Nodes: {len(result.knowledge_graph.nodes)}")
        print(f"  Edges: {len(result.knowledge_graph.edges)}")
        print(f"  Node Types: {result.knowledge_graph.metadata.get('node_types', {})}")
        print(f"  Edge Types: {result.knowledge_graph.metadata.get('edge_types', {})}")
        
        print(f"\n{'='*70}\n")


# ========== MAIN ==========

def main():
    """Main function to run the gap detection stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Initialize orchestrator
    orchestrator = GapDetectionOrchestrator()
    
    # Load literature batch from Stage 1
    batch_file = input("Enter path to literature batch JSON (or press Enter for default): ").strip()
    if not batch_file:
        batch_file = "literature_batch.json"
    
    try:
        literature_batch = orchestrator.load_literature_batch(batch_file)
    except Exception as e:
        print(f"Error loading literature batch: {e}")
        print("Please run Stage 1 first to generate the literature batch.")
        return
    
    # Run pipeline
    gap_analysis = orchestrator.run_gap_detection_pipeline(literature_batch)
    
    # Save results
    orchestrator.save_gap_analysis("gap_analysis_results.json")
    orchestrator.save_gaps_csv("research_gaps.csv")
    orchestrator.save_graph_json("knowledge_graph.json")
    
    # Print summary
    orchestrator.print_summary()
    
    print("\n" + "="*70)
    print("Gap detection complete!")
    print("Output files:")
    print("  - gap_analysis_results.json (complete analysis)")
    print("  - research_gaps.csv (gaps only)")
    print("  - knowledge_graph.json (graph structure)")
    print("="*70)


if __name__ == "__main__":
    main()
