"""
Literature Gathering Stage - Automated Mode (No Firecrawl, No HITL)
This script gathers and processes literature based on research questions from IdeationTeam.

Pipeline:
1. TopicCrawler: Retrieves literature based on research questions
2. TrendTracker: Monitors field trends and emerging patterns
3. CiteKeeper: Manages bibliography (receives TrendTracker results)
4. InsightSummarizer: Extracts key knowledge from literature

Input: Research questions (JSON from IdeationTeam Stage 3)
Output: Initial literature batch with metadata, trends, citations, and insights
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import arxiv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import requests

# Load environment variables
load_dotenv()


# ========== DATA MODELS ==========

class ResearchQuestion(BaseModel):
    """Model for input research questions."""
    question: str = Field(description="The research question")
    priority_rank: Optional[int] = Field(description="Priority ranking", default=None)
    priority_score: Optional[float] = Field(description="Priority score", default=None)
    theoretical_framework: Optional[str] = Field(description="Theoretical framework", default="")
    methodology: Optional[List[str]] = Field(description="Suggested methodologies", default_factory=list)


class LiteratureItem(BaseModel):
    """Model for a literature item."""
    title: str = Field(description="Title of the paper/document")
    authors: List[str] = Field(description="List of authors")
    abstract: str = Field(description="Abstract or summary")
    url: str = Field(description="URL or identifier")
    source: str = Field(description="Source database/repository")
    year: Optional[int] = Field(description="Publication year", default=None)
    citation_count: Optional[int] = Field(description="Number of citations", default=None)
    literature_type: str = Field(description="Type: academic, grey, preprint", default="academic")
    research_question: str = Field(description="Associated research question", default="")
    relevance_score: Optional[float] = Field(description="Relevance score (0-1)", default=None)


class TrendAnalysis(BaseModel):
    """Model for trend analysis results."""
    trend_name: str = Field(description="Name of the trend")
    description: str = Field(description="Description of the trend")
    key_papers: List[str] = Field(description="Key papers related to this trend")
    emergence_year: Optional[int] = Field(description="Year trend emerged", default=None)
    growth_trajectory: str = Field(description="Growing/Stable/Declining", default="Growing")
    related_questions: List[str] = Field(description="Related research questions")


class CitationEntry(BaseModel):
    """Model for a citation entry."""
    citation_key: str = Field(description="Unique citation key (e.g., Author2023)")
    formatted_citation: str = Field(description="Formatted citation string")
    paper_title: str = Field(description="Paper title")
    authors: List[str] = Field(description="Authors")
    year: Optional[int] = Field(description="Publication year", default=None)
    url: str = Field(description="URL or DOI")
    citation_count: Optional[int] = Field(description="Number of citations", default=None)
    related_trends: List[str] = Field(description="Related trend names")


class KnowledgeInsight(BaseModel):
    """Model for extracted knowledge insights."""
    insight_title: str = Field(description="Title of the insight")
    insight_description: str = Field(description="Detailed description")
    supporting_papers: List[str] = Field(description="Papers supporting this insight")
    key_findings: List[str] = Field(description="Key findings (3-5 points)")
    methodologies_used: List[str] = Field(description="Methodologies mentioned")
    research_gaps: List[str] = Field(description="Identified research gaps")
    related_questions: List[str] = Field(description="Related research questions")


class LiteratureBatch(BaseModel):
    """Model for the complete literature batch output."""
    research_questions: List[ResearchQuestion] = Field(description="Input research questions")
    literature_items: List[LiteratureItem] = Field(description="Retrieved literature")
    trend_analyses: List[TrendAnalysis] = Field(description="Trend analysis results")
    citations: List[CitationEntry] = Field(description="Bibliography entries")
    insights: List[KnowledgeInsight] = Field(description="Extracted insights")
    metadata: Dict = Field(description="Batch metadata", default_factory=dict)


# ========== AGENTS ==========

class TopicCrawler:
    """Agent for literature retrieval based on research questions."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "TopicCrawler"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=self.api_key
        )
    
    def retrieve_literature(
        self,
        research_questions: List[ResearchQuestion],
        max_papers_per_question: int = 15
    ) -> List[LiteratureItem]:
        """Retrieve literature for each research question."""
        print(f"\n[{self.agent_name}] Retrieving literature for {len(research_questions)} research questions...")
        
        all_literature = []
        
        for rq in research_questions:
            print(f"\n  Processing: {rq.question[:80]}...")
            
            # Generate search queries from research question
            search_queries = self._generate_search_queries(rq)
            
            # Search multiple sources
            for query in search_queries[:2]:  # Top 2 queries per question
                # Search Semantic Scholar
                papers = self._search_semantic_scholar(query, max_papers_per_question // 2)
                
                # Search arXiv
                papers.extend(self._search_arxiv(query, max_papers_per_question // 2))
                
                # Associate with research question
                for paper in papers:
                    paper.research_question = rq.question
                
                all_literature.extend(papers)
                print(f"    Found {len(papers)} papers for query: {query[:50]}...")
        
        # Deduplicate
        unique_literature = self._deduplicate(all_literature)
        print(f"\n[{self.agent_name}] Total unique papers retrieved: {len(unique_literature)}")
        
        return unique_literature
    
    def _generate_search_queries(self, research_question: ResearchQuestion) -> List[str]:
        """Generate search queries from a research question."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at formulating academic search queries."),
            ("user", """Generate 3 effective search queries for academic databases based on this research question.
            
            Research Question: {question}
            Theoretical Framework: {framework}
            Methodologies: {methodologies}
            
            Return queries as a JSON array of strings.
            Example: ["query 1", "query 2", "query 3"]
            
            Respond with ONLY the JSON array, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "question": research_question.question,
                "framework": research_question.theoretical_framework or "N/A",
                "methodologies": ", ".join(research_question.methodology) if research_question.methodology else "N/A"
            })
            
            queries = json.loads(result.content)
            return queries
        except Exception as e:
            print(f"    Error generating queries: {e}")
            # Fallback: use the question itself
            return [research_question.question]
    
    def _search_semantic_scholar(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search Semantic Scholar API."""
        results = []
        
        try:
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,abstract,url,year,citationCount,publicationTypes"
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for paper in data.get("data", []):
                item = LiteratureItem(
                    title=paper.get("title", ""),
                    authors=[author.get("name", "") for author in paper.get("authors", [])],
                    abstract=paper.get("abstract", "No abstract available"),
                    url=paper.get("url", ""),
                    source="Semantic Scholar",
                    year=paper.get("year"),
                    literature_type="academic",
                    citation_count=paper.get("citationCount")
                )
                results.append(item)
        
        except Exception as e:
            print(f"    Error searching Semantic Scholar: {e}")
        
        return results
    
    def _search_arxiv(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search arXiv API."""
        results = []
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in search.results():
                item = LiteratureItem(
                    title=paper.title,
                    authors=[author.name for author in paper.authors],
                    abstract=paper.summary,
                    url=paper.entry_id,
                    source="arXiv",
                    year=paper.published.year if paper.published else None,
                    literature_type="preprint"
                )
                results.append(item)
        
        except Exception as e:
            print(f"    Error searching arXiv: {e}")
        
        return results
    
    def _deduplicate(self, papers: List[LiteratureItem]) -> List[LiteratureItem]:
        """Remove duplicate papers based on title."""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            normalized_title = paper.title.lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)
        
        return unique_papers


class TrendTracker:
    """Agent for field monitoring and trend analysis."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "TrendTracker"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            openai_api_key=self.api_key
        )
    
    def analyze_trends(
        self,
        literature: List[LiteratureItem],
        research_questions: List[ResearchQuestion]
    ) -> List[TrendAnalysis]:
        """Analyze trends in the literature."""
        print(f"\n[{self.agent_name}] Analyzing trends in {len(literature)} papers...")
        
        # Prepare literature summary
        lit_summary = self._prepare_literature_summary(literature)
        questions_text = "\n".join([f"- {rq.question}" for rq in research_questions])
        
        parser = PydanticOutputParser(pydantic_object=TrendAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at identifying research trends and patterns in academic literature."),
            ("user", """Analyze the following literature and identify 5-7 major research trends.
            
            Research Questions:
            {questions}
            
            Literature Summary:
            {literature}
            
            For each trend, provide:
            - trend_name: Clear, concise name
            - description: 2-3 sentence description
            - key_papers: List of 3-5 paper titles from the summary
            - emergence_year: Approximate year the trend emerged
            - growth_trajectory: "Growing", "Stable", or "Declining"
            - related_questions: Which research questions relate to this trend
            
            Return as a JSON array of trend objects.
            Example: [{{"trend_name": "...", "description": "...", "key_papers": [...], ...}}]
            
            Respond with ONLY the JSON array, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "questions": questions_text,
                "literature": lit_summary
            })
            
            trends_data = json.loads(result.content)
            trends = [TrendAnalysis(**trend) for trend in trends_data]
            
            print(f"[{self.agent_name}] Identified {len(trends)} trends")
            return trends
        
        except Exception as e:
            print(f"[{self.agent_name}] Error analyzing trends: {e}")
            return []
    
    def _prepare_literature_summary(self, literature: List[LiteratureItem], max_papers: int = 50) -> str:
        """Prepare a summary of literature for analysis."""
        summary_parts = []
        
        for idx, paper in enumerate(literature[:max_papers], 1):
            summary = f"""
Paper {idx}: {paper.title}
Authors: {', '.join(paper.authors[:3])}
Year: {paper.year or 'N/A'}
Citations: {paper.citation_count or 'N/A'}
Abstract: {paper.abstract[:250]}...
"""
            summary_parts.append(summary)
        
        return "\n".join(summary_parts)


class CiteKeeper:
    """Agent for bibliography management."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "CiteKeeper"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=self.api_key
        )
    
    def manage_bibliography(
        self,
        literature: List[LiteratureItem],
        trends: List[TrendAnalysis]
    ) -> List[CitationEntry]:
        """Create bibliography entries with trend associations."""
        print(f"\n[{self.agent_name}] Managing bibliography for {len(literature)} papers...")
        
        citations = []
        
        for paper in literature:
            # Generate citation key
            citation_key = self._generate_citation_key(paper)
            
            # Format citation
            formatted_citation = self._format_citation(paper)
            
            # Associate with trends
            related_trends = self._associate_with_trends(paper, trends)
            
            citation = CitationEntry(
                citation_key=citation_key,
                formatted_citation=formatted_citation,
                paper_title=paper.title,
                authors=paper.authors,
                year=paper.year,
                url=paper.url,
                citation_count=paper.citation_count,
                related_trends=related_trends
            )
            
            citations.append(citation)
        
        # Sort by citation count (descending)
        citations.sort(key=lambda x: x.citation_count or 0, reverse=True)
        
        print(f"[{self.agent_name}] Created {len(citations)} bibliography entries")
        return citations
    
    def _generate_citation_key(self, paper: LiteratureItem) -> str:
        """Generate a citation key (e.g., Smith2023)."""
        if not paper.authors:
            author_part = "Unknown"
        else:
            # Get first author's last name
            first_author = paper.authors[0]
            author_part = first_author.split()[-1] if first_author else "Unknown"
        
        year_part = str(paper.year) if paper.year else "XXXX"
        
        return f"{author_part}{year_part}"
    
    def _format_citation(self, paper: LiteratureItem) -> str:
        """Format citation in APA style."""
        authors_str = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += " et al."
        
        year_str = f"({paper.year})" if paper.year else "(n.d.)"
        
        citation = f"{authors_str} {year_str}. {paper.title}. {paper.source}. {paper.url}"
        
        return citation
    
    def _associate_with_trends(self, paper: LiteratureItem, trends: List[TrendAnalysis]) -> List[str]:
        """Associate paper with relevant trends."""
        related_trends = []
        
        paper_text = f"{paper.title} {paper.abstract}".lower()
        
        for trend in trends:
            # Check if paper is mentioned in trend's key papers
            if paper.title in trend.key_papers:
                related_trends.append(trend.trend_name)
            # Or check for keyword overlap
            elif any(keyword.lower() in paper_text for keyword in trend.trend_name.split()):
                related_trends.append(trend.trend_name)
        
        return related_trends


class InsightSummarizer:
    """Agent for extracting knowledge insights from literature."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "InsightSummarizer"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.6,
            openai_api_key=self.api_key
        )
    
    def extract_insights(
        self,
        literature: List[LiteratureItem],
        research_questions: List[ResearchQuestion],
        trends: List[TrendAnalysis]
    ) -> List[KnowledgeInsight]:
        """Extract key insights from literature."""
        print(f"\n[{self.agent_name}] Extracting insights from {len(literature)} papers...")
        
        # Group literature by research question
        insights = []
        
        for rq in research_questions:
            # Get papers related to this question
            related_papers = [p for p in literature if p.research_question == rq.question]
            
            if not related_papers:
                continue
            
            print(f"\n  Analyzing {len(related_papers)} papers for: {rq.question[:60]}...")
            
            # Extract insights for this question
            question_insights = self._extract_insights_for_question(rq, related_papers, trends)
            insights.extend(question_insights)
        
        print(f"\n[{self.agent_name}] Extracted {len(insights)} insights")
        return insights
    
    def _extract_insights_for_question(
        self,
        research_question: ResearchQuestion,
        papers: List[LiteratureItem],
        trends: List[TrendAnalysis]
    ) -> List[KnowledgeInsight]:
        """Extract insights for a specific research question."""
        
        # Prepare papers summary
        papers_summary = "\n\n".join([
            f"Paper: {p.title}\nAuthors: {', '.join(p.authors[:3])}\nAbstract: {p.abstract[:300]}..."
            for p in papers[:20]  # Top 20 papers
        ])
        
        # Prepare trends context
        trends_text = "\n".join([f"- {t.trend_name}: {t.description}" for t in trends[:5]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at synthesizing research insights from academic literature."),
            ("user", """Extract 2-3 key insights from the following papers related to the research question.
            
            Research Question: {question}
            
            Relevant Trends:
            {trends}
            
            Papers:
            {papers}
            
            For each insight, provide:
            - insight_title: Clear, concise title
            - insight_description: Detailed description (3-4 sentences)
            - supporting_papers: List of 2-4 paper titles
            - key_findings: List of 3-5 key findings
            - methodologies_used: List of methodologies mentioned
            - research_gaps: List of 2-3 identified gaps
            - related_questions: [The research question]
            
            Return as a JSON array of insight objects.
            Example: [{{"insight_title": "...", "insight_description": "...", ...}}]
            
            Respond with ONLY the JSON array, no other text.
            """)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "question": research_question.question,
                "trends": trends_text,
                "papers": papers_summary
            })
            
            insights_data = json.loads(result.content)
            insights = [KnowledgeInsight(**insight) for insight in insights_data]
            
            return insights
        
        except Exception as e:
            print(f"    Error extracting insights: {e}")
            return []


# ========== ORCHESTRATOR ==========

class LiteratureGatheringOrchestrator:
    """Orchestrates the literature gathering pipeline."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        
        # Initialize agents
        self.topic_crawler = TopicCrawler(self.api_key)
        self.trend_tracker = TrendTracker(self.api_key)
        self.cite_keeper = CiteKeeper(self.api_key)
        self.insight_summarizer = InsightSummarizer(self.api_key)
        
        self.literature_batch: Optional[LiteratureBatch] = None
    
    def run_gathering_pipeline(
        self,
        research_questions: List[ResearchQuestion],
        max_papers_per_question: int = 15
    ) -> LiteratureBatch:
        """Run the complete literature gathering pipeline."""
        
        print(f"\n{'='*70}")
        print(f"LITERATURE GATHERING PIPELINE")
        print(f"{'='*70}")
        print(f"Research Questions: {len(research_questions)}")
        print(f"{'='*70}\n")
        
        # Step 1: TopicCrawler - Retrieve literature
        literature = self.topic_crawler.retrieve_literature(
            research_questions=research_questions,
            max_papers_per_question=max_papers_per_question
        )
        
        # Step 2: TrendTracker - Analyze trends
        trends = self.trend_tracker.analyze_trends(
            literature=literature,
            research_questions=research_questions
        )
        
        # Step 3: CiteKeeper - Manage bibliography (receives TrendTracker results)
        citations = self.cite_keeper.manage_bibliography(
            literature=literature,
            trends=trends
        )
        
        # Step 4: InsightSummarizer - Extract insights
        insights = self.insight_summarizer.extract_insights(
            literature=literature,
            research_questions=research_questions,
            trends=trends
        )
        
        # Create literature batch
        self.literature_batch = LiteratureBatch(
            research_questions=research_questions,
            literature_items=literature,
            trend_analyses=trends,
            citations=citations,
            insights=insights,
            metadata={
                "created_at": datetime.now().isoformat(),
                "total_papers": len(literature),
                "total_trends": len(trends),
                "total_citations": len(citations),
                "total_insights": len(insights)
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Literature Items: {len(literature)}")
        print(f"Trends Identified: {len(trends)}")
        print(f"Citations Created: {len(citations)}")
        print(f"Insights Extracted: {len(insights)}")
        print(f"{'='*70}\n")
        
        return self.literature_batch
    
    def load_research_questions(self, filepath: str) -> List[ResearchQuestion]:
        """Load research questions from IdeationTeam output."""
        print(f"[Orchestrator] Loading research questions from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            questions = [ResearchQuestion(**q) for q in data]
        elif isinstance(data, dict) and 'questions' in data:
            questions = [ResearchQuestion(**q) for q in data['questions']]
        else:
            raise ValueError("Unexpected JSON format for research questions")
        
        print(f"[Orchestrator] Loaded {len(questions)} research questions")
        return questions
    
    def save_literature_batch(self, filepath: str):
        """Save the complete literature batch to JSON."""
        if not self.literature_batch:
            print("No literature batch to save.")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.literature_batch.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"[Orchestrator] Literature batch saved to: {filepath}")
    
    def save_literature_csv(self, filepath: str):
        """Save literature items to CSV for easy viewing."""
        if not self.literature_batch:
            print("No literature batch to save.")
            return
        
        data = [item.model_dump() for item in self.literature_batch.literature_items]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"[Orchestrator] Literature CSV saved to: {filepath}")
    
    def print_summary(self):
        """Print a summary of the literature batch."""
        if not self.literature_batch:
            print("No literature batch available.")
            return
        
        batch = self.literature_batch
        
        print(f"\n{'='*70}")
        print(f"LITERATURE BATCH SUMMARY")
        print(f"{'='*70}\n")
        
        # Research Questions
        print(f"Research Questions ({len(batch.research_questions)}):")
        for i, rq in enumerate(batch.research_questions, 1):
            print(f"  {i}. {rq.question}")
        
        # Top Papers
        print(f"\nTop 10 Papers (by citations):")
        sorted_papers = sorted(
            batch.literature_items,
            key=lambda x: x.citation_count or 0,
            reverse=True
        )
        for i, paper in enumerate(sorted_papers[:10], 1):
            print(f"  {i}. {paper.title}")
            print(f"     Citations: {paper.citation_count or 'N/A'} | Year: {paper.year or 'N/A'}")
        
        # Trends
        print(f"\nIdentified Trends ({len(batch.trend_analyses)}):")
        for i, trend in enumerate(batch.trend_analyses, 1):
            print(f"  {i}. {trend.trend_name} ({trend.growth_trajectory})")
            print(f"     {trend.description[:100]}...")
        
        # Insights
        print(f"\nKey Insights ({len(batch.insights)}):")
        for i, insight in enumerate(batch.insights, 1):
            print(f"  {i}. {insight.insight_title}")
            print(f"     {insight.insight_description[:100]}...")
        
        print(f"\n{'='*70}\n")


# ========== MAIN ==========

def main():
    """Main function to run the literature gathering stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Initialize orchestrator
    orchestrator = LiteratureGatheringOrchestrator()
    
    # Load research questions from IdeationTeam output
    # Default path - adjust as needed
    questions_file = input("Enter path to research questions JSON (or press Enter for default): ").strip()
    if not questions_file:
        questions_file = "../../1-IdeationTeam/ModeNoFcNoHITL/finalized_research_questions_automated.json"
    
    try:
        research_questions = orchestrator.load_research_questions(questions_file)
    except Exception as e:
        print(f"Error loading research questions: {e}")
        print("Using sample research questions for demonstration...")
        research_questions = [
            ResearchQuestion(
                question="How do agent-based models improve monetary policy analysis?",
                priority_rank=1,
                priority_score=0.95
            ),
            ResearchQuestion(
                question="What are the computational challenges in large-scale ABM simulations?",
                priority_rank=2,
                priority_score=0.88
            )
        ]
    
    # Run pipeline
    literature_batch = orchestrator.run_gathering_pipeline(
        research_questions=research_questions,
        max_papers_per_question=15
    )
    
    # Save results
    orchestrator.save_literature_batch("literature_batch.json")
    orchestrator.save_literature_csv("literature_items.csv")
    
    # Print summary
    orchestrator.print_summary()
    
    print("\n" + "="*70)
    print("Literature gathering complete!")
    print("Output files:")
    print("  - literature_batch.json (complete batch with all data)")
    print("  - literature_items.csv (literature items only)")
    print("="*70)


if __name__ == "__main__":
    main()
