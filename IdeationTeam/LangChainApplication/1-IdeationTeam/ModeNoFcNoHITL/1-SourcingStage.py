"""
Automated Multi-Agent Literature Sourcing Stage (No Human-in-the-Loop)
This script uses specialized LangChain agents to gather literature automatically.

Agents:
- TrendSurfer: Identifies emerging trends and recent developments
- TopicCrawler: Searches academic databases for peer-reviewed literature
- ScholarSearcher: Finds highly-cited foundational papers
- GreyScout: Discovers grey literature (reports, working papers, policy docs)

Input: Research keywords or basic research ideas
Output: Ranked literature results (CSV file)
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import arxiv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()


class LiteratureItem(BaseModel):
    """Model for a literature item."""
    title: str = Field(description="Title of the paper/document")
    authors: List[str] = Field(description="List of authors")
    abstract: str = Field(description="Abstract or summary")
    url: str = Field(description="URL or identifier")
    source: str = Field(description="Source database/repository")
    agent: str = Field(description="Agent that found this item")
    year: Optional[int] = Field(description="Publication year", default=None)
    literature_type: str = Field(description="Type: academic, grey, preprint", default="academic")
    citation_count: Optional[int] = Field(description="Number of citations", default=None)
    relevance_score: Optional[float] = Field(description="Relevance score (0-1)", default=None)


class SearchQuery(BaseModel):
    """Model for refined search queries."""
    queries: List[str] = Field(description="List of refined search queries")
    keywords: List[str] = Field(description="Key terms to search for")
    focus_areas: List[str] = Field(description="Specific focus areas")


class BaseAgent:
    """Base class for all literature sourcing agents."""
    
    def __init__(self, name: str, openai_api_key: str):
        self.name = name
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", # gpt-4o-mini, gpt-4o
            temperature=0.3,
            openai_api_key=openai_api_key
        )
        self.results: List[LiteratureItem] = []
    
    def refine_query(self, research_topic: str) -> SearchQuery:
        """Refine research query based on agent specialty and feedback."""
        raise NotImplementedError("Subclasses must implement refine_query")
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Execute search based on agent specialty."""
        raise NotImplementedError("Subclasses must implement search")


class TrendSurfer(BaseAgent):
    """Agent focused on identifying emerging trends and recent developments."""
    
    def __init__(self, openai_api_key: str):
        super().__init__("TrendSurfer", openai_api_key)
    
    def refine_query(self, research_topic: str) -> SearchQuery:
        """Generate queries focused on recent trends and developments."""
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are TrendSurfer, an expert at identifying emerging trends and recent developments in research. You must respond with valid JSON only."),
            ("user", """Generate search queries focused on RECENT (last 2-3 years) and EMERGING trends.
            
            Research Topic: {topic}
            
            Focus on:
            - New methodologies and approaches
            - Recent empirical findings
            - Emerging debates and controversies
            - Latest technological applications
            
            Respond with a JSON object containing:
            - "queries": array of 3-5 search query strings
            - "keywords": array of 5-10 keyword strings
            - "focus_areas": array of 3-5 focus area strings
            
            Example: {{"queries": ["query1", "query2"], "keywords": ["kw1", "kw2"], "focus_areas": ["area1", "area2"]}}
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm | parser
        result = chain.invoke({
            "topic": research_topic
        })
        
        return result
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search for recent papers on arXiv and Semantic Scholar."""
        print(f"[TrendSurfer] Searching for recent trends: {query}")
        results = []
        
        # Search arXiv (preprints - recent trends)
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate  # Most recent first
            )
            
            for paper in search.results():
                # Only include papers from last 3 years
                if paper.published and paper.published.year >= datetime.now().year - 3:
                    item = LiteratureItem(
                        title=paper.title,
                        authors=[author.name for author in paper.authors],
                        abstract=paper.summary,
                        url=paper.entry_id,
                        source="arXiv",
                        agent="TrendSurfer",
                        year=paper.published.year,
                        literature_type="preprint"
                    )
                    results.append(item)
        except Exception as e:
            print(f"[TrendSurfer] Error searching arXiv: {e}")
        
        return results


class TopicCrawler(BaseAgent):
    """Agent focused on comprehensive academic literature search."""
    
    def __init__(self, openai_api_key: str):
        super().__init__("TopicCrawler", openai_api_key)
    
    def refine_query(self, research_topic: str) -> SearchQuery:
        """Generate comprehensive academic search queries."""
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are TopicCrawler, an expert at comprehensive academic literature searches. You must respond with valid JSON only."),
            ("user", """Generate broad and comprehensive search queries for peer-reviewed academic literature.
            
            Research Topic: {topic}
            
            Focus on:
            - Core theoretical frameworks
            - Empirical studies and methodologies
            - Review articles and meta-analyses
            - Cross-disciplinary connections
            
            Respond with a JSON object containing:
            - "queries": array of 3-5 search query strings
            - "keywords": array of 5-10 keyword strings
            - "focus_areas": array of 3-5 focus area strings
            
            Example format:
            {{
              "queries": ["query 1", "query 2", "query 3"],
              "keywords": ["keyword1", "keyword2", "keyword3"],
              "focus_areas": ["area1", "area2", "area3"]
            }}
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm | parser
        result = chain.invoke({
            "topic": research_topic
        })
        
        return result
    
    def search(self, query: str, max_results: int = 15) -> List[LiteratureItem]:
        """Search academic databases comprehensively."""
        print(f"[TopicCrawler] Comprehensive academic search: {query}")
        results = []
        
        # Search Semantic Scholar
        try:
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,abstract,url,year,citationCount,publicationTypes"
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for paper in data.get("data", []):
                item = LiteratureItem(
                    title=paper.get("title", ""),
                    authors=[author.get("name", "") for author in paper.get("authors", [])],
                    abstract=paper.get("abstract", "No abstract available"),
                    url=paper.get("url", ""),
                    source="Semantic Scholar",
                    agent="TopicCrawler",
                    year=paper.get("year"),
                    literature_type="academic",
                    citation_count=paper.get("citationCount")
                )
                results.append(item)
        
        except Exception as e:
            print(f"[TopicCrawler] Error searching Semantic Scholar: {e}")
        
        return results


class ScholarSearcher(BaseAgent):
    """Agent focused on finding highly-cited foundational papers."""
    
    def __init__(self, openai_api_key: str):
        super().__init__("ScholarSearcher", openai_api_key)
    
    def refine_query(self, research_topic: str) -> SearchQuery:
        """Generate queries for foundational and highly-cited work."""
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are ScholarSearcher, an expert at finding seminal and highly-cited foundational papers. You must respond with valid JSON only."),
            ("user", """Generate search queries for FOUNDATIONAL and HIGHLY-CITED papers.
            
            Research Topic: {topic}
            
            Focus on:
            - Seminal theoretical contributions
            - Landmark empirical studies
            - Highly-cited review articles
            - Classic papers that defined the field
            
            Respond with a JSON object containing:
            - "queries": array of 3-5 search query strings
            - "keywords": array of 5-10 keyword strings
            - "focus_areas": array of 3-5 focus area strings
            
            Example: {{"queries": ["query1", "query2"], "keywords": ["kw1", "kw2"], "focus_areas": ["area1", "area2"]}}
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm | parser
        result = chain.invoke({
            "topic": research_topic
        })
        
        return result
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search for highly-cited papers."""
        print(f"[ScholarSearcher] Searching for foundational papers: {query}")
        results = []
        
        # Search Semantic Scholar sorted by citations
        try:
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,abstract,url,year,citationCount",
                "sort": "citationCount:desc"  # Sort by citations
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for paper in data.get("data", []):
                # Only include papers with significant citations
                citation_count = paper.get("citationCount", 0)
                if citation_count and citation_count > 50:  # Threshold for "highly-cited"
                    item = LiteratureItem(
                        title=paper.get("title", ""),
                        authors=[author.get("name", "") for author in paper.get("authors", [])],
                        abstract=paper.get("abstract", "No abstract available"),
                        url=paper.get("url", ""),
                        source="Semantic Scholar",
                        agent="ScholarSearcher",
                        year=paper.get("year"),
                        literature_type="academic",
                        citation_count=citation_count
                    )
                    results.append(item)
        
        except Exception as e:
            print(f"[ScholarSearcher] Error searching: {e}")
        
        return results


class GreyScout(BaseAgent):
    """Agent focused on grey literature (reports, working papers, policy documents)."""
    
    def __init__(self, openai_api_key: str):
        super().__init__("GreyScout", openai_api_key)
    
    def refine_query(self, research_topic: str) -> SearchQuery:
        """Generate queries for grey literature."""
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are GreyScout, an expert at finding grey literature like working papers, reports, and policy documents. You must respond with valid JSON only."),
            ("user", """Generate search queries for GREY LITERATURE (non-peer-reviewed but authoritative).
            
            Research Topic: {topic}
            
            Focus on:
            - Working papers and preprints
            - Policy reports and white papers
            - Technical reports from institutions
            - Conference proceedings
            - Think tank publications
            
            Respond with a JSON object containing:
            - "queries": array of 3-5 search query strings
            - "keywords": array of 5-10 keyword strings
            - "focus_areas": array of 3-5 focus area strings
            
            Example: {{"queries": ["query1", "query2"], "keywords": ["kw1", "kw2"], "focus_areas": ["area1", "area2"]}}
            
            Respond with ONLY the JSON object, no other text.
            """)
        ])
        
        chain = prompt | self.llm | parser
        result = chain.invoke({
            "topic": research_topic
        })
        
        return result
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search for grey literature."""
        print(f"[GreyScout] Searching for grey literature: {query}")
        results = []
        
        # Search arXiv for working papers
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
                    source="arXiv (Working Paper)",
                    agent="GreyScout",
                    year=paper.published.year if paper.published else None,
                    literature_type="grey"
                )
                results.append(item)
        
        except Exception as e:
            print(f"[GreyScout] Error searching: {e}")
        
        return results


class MultiAgentOrchestrator:
    """Orchestrates multiple agents for automated literature sourcing."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        
        # Initialize all agents
        self.agents = {
            "TrendSurfer": TrendSurfer(self.api_key),
            "TopicCrawler": TopicCrawler(self.api_key),
            "ScholarSearcher": ScholarSearcher(self.api_key),
            "GreyScout": GreyScout(self.api_key)
        }
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", # gpt-4
        temperature=0.3, openai_api_key=self.api_key)
        self.all_results: List[LiteratureItem] = []
        self.round_results: Dict[int, List[LiteratureItem]] = {}
    
    def run_automated_search(
        self,
        research_topic: str,
        max_results_per_agent: int = 15
    ) -> List[LiteratureItem]:
        """Run automated single-round search without human feedback."""
        print(f"\n{'='*70}")
        print(f"AUTOMATED LITERATURE SEARCH")
        print(f"Topic: {research_topic}")
        print(f"{'='*70}\n")
        
        round_results = []
        
        for agent_name, agent in self.agents.items():
            print(f"\n--- {agent_name} ---")
            
            # Refine query based on agent specialty
            search_query = agent.refine_query(research_topic)
            print(f"Generated {len(search_query.queries)} queries")
            print(f"Keywords: {', '.join(search_query.keywords[:5])}...")
            
            # Execute searches
            for query in search_query.queries[:2]:  # Use top 2 queries per agent
                results = agent.search(query, max_results_per_agent)
                round_results.extend(results)
                print(f"  Found {len(results)} items for query: {query[:50]}...")
        
        # Deduplicate
        unique_results = self._deduplicate(round_results)
        print(f"\n[Orchestrator] Total unique items found: {len(unique_results)}")
        
        # Rank by relevance
        ranked_results = self._rank_by_relevance(unique_results, research_topic)
        
        self.all_results = ranked_results
        
        return ranked_results
    
    def _deduplicate(self, papers: List[LiteratureItem]) -> List[LiteratureItem]:
        """Remove duplicate papers."""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            normalized_title = paper.title.lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _rank_by_relevance(
        self,
        papers: List[LiteratureItem],
        research_topic: str
    ) -> List[LiteratureItem]:
        """Rank papers by relevance using LLM."""
        if not papers:
            return []
        
        print(f"[Orchestrator] Ranking {len(papers)} papers...")
        
        batch_size = 5
        ranked_papers = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            
            papers_text = "\n\n".join([
                f"Paper {idx+1} (by {p.agent}):\nTitle: {p.title}\nAbstract: {p.abstract[:250]}..."
                for idx, p in enumerate(batch)
            ])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert research evaluator."),
                ("user", """Rate each paper's relevance to the research topic (0-1 scale).
                
                Research Topic: {topic}
                
                Papers:
                {papers}
                
                Provide scores as comma-separated (e.g., 0.9, 0.7, 0.5, 0.3, 0.8).
                Only numbers, no other text.
                """)
            ])
            
            chain = prompt | self.llm
            
            try:
                result = chain.invoke({
                    "topic": research_topic,
                    "papers": papers_text
                })
                
                scores = [float(s.strip()) for s in result.content.split(",")]
                
                for paper, score in zip(batch, scores):
                    paper.relevance_score = score
                    ranked_papers.append(paper)
            
            except Exception as e:
                print(f"[Orchestrator] Error ranking batch: {e}")
                ranked_papers.extend(batch)
        
        ranked_papers.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        return ranked_papers
    
    def save_results(self, filename: str):
        """Save results to CSV."""
        results = self.all_results
        
        if not results:
            print("No results to save.")
            return
        
        data = [item.model_dump() for item in results]
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"[Orchestrator] Results saved to {filename}")
    
    def print_summary(self, top_n: int = 10):
        """Print summary of results."""
        results = self.all_results
        title = f"TOP {top_n} PAPERS"
        
        print(f"\n{'='*70}")
        print(title)
        print(f"{'='*70}\n")
        
        for idx, paper in enumerate(results[:top_n], 1):
            print(f"{idx}. [{paper.agent}] {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   Source: {paper.source} | Year: {paper.year or 'N/A'} | Type: {paper.literature_type}")
            if paper.citation_count:
                print(f"   Citations: {paper.citation_count}")
            if paper.relevance_score:
                print(f"   Relevance: {paper.relevance_score:.2f}")
            print(f"   URL: {paper.url}")
            print(f"   Abstract: {paper.abstract[:150]}...")
            print()
        
        # Agent statistics
        agent_counts = {}
        for paper in results:
            agent_counts[paper.agent] = agent_counts.get(paper.agent, 0) + 1
        
        print(f"\nAgent Contributions:")
        for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {agent}: {count} papers")


def main():
    """Main function for automated literature sourcing (No HITL)."""
    # Example research topic
    research_topic = "Agent-based modeling in macroeconomics and monetary policy"
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Run automated search
    print("\n" + "="*70)
    print("AUTOMATED LITERATURE SOURCING (No Human-in-the-Loop)")
    print("="*70)
    
    results = orchestrator.run_automated_search(
        research_topic=research_topic,
        max_results_per_agent=15
    )
    
    # Display results
    orchestrator.print_summary(top_n=15)
    
    # Save results
    orchestrator.save_results("literature_results_automated.csv")
    
    # Final summary
    print(f"\n{'='*70}")
    print("SOURCING COMPLETE")
    print(f"{'='*70}")
    print(f"Total papers found: {len(results)}")
    print(f"Results saved to: literature_results_automated.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
