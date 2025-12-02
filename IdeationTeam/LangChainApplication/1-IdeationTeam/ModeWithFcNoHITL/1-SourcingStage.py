"""
Multi-Agent Literature Sourcing Stage with Firecrawl (No Human-in-the-Loop)
This script uses specialized LangChain agents to gather literature with Firecrawl web scraping.

Agents:
- TrendSurfer: Identifies emerging trends using arXiv + Firecrawl web scraping
- TopicCrawler: Searches academic databases + Firecrawl for comprehensive coverage
- ScholarSearcher: Finds highly-cited foundational papers
- GreyScout: Discovers grey literature (reports, working papers, policy docs)

Firecrawl Integration:
- TrendSurfer: Scrapes recent blog posts, news articles, and research websites
- TopicCrawler: Scrapes academic institution pages and research portals
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


class HumanFeedback(BaseModel):
    """Model for human feedback."""
    round_number: int = Field(description="Feedback round number")
    relevant_papers: List[str] = Field(description="Titles of relevant papers")
    irrelevant_papers: List[str] = Field(description="Titles of irrelevant papers")
    missing_topics: List[str] = Field(description="Topics that should be explored more")
    additional_keywords: List[str] = Field(description="Additional keywords to include")
    comments: str = Field(description="General comments and guidance")


class BaseAgent:
    """Base class for all literature sourcing agents."""
    
    def __init__(self, name: str, openai_api_key: str, firecrawl_api_key: Optional[str] = None):
        self.name = name
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # gpt-4o-mini, gpt-4o
            temperature=0.3,
            openai_api_key=openai_api_key
        )
        self.firecrawl_api_key = firecrawl_api_key
        self.results: List[LiteratureItem] = []
    
    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        """Refine research query based on agent specialty and feedback."""
        raise NotImplementedError("Subclasses must implement refine_query")
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Execute search based on agent specialty."""
        raise NotImplementedError("Subclasses must implement search")


class TrendSurfer(BaseAgent):
    """Agent focused on identifying emerging trends and recent developments using arXiv + Firecrawl."""
    
    def __init__(self, openai_api_key: str, firecrawl_api_key: Optional[str] = None):
        super().__init__("TrendSurfer", openai_api_key, firecrawl_api_key)
    
    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        """Generate queries focused on recent trends and developments."""
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Missing topics: {', '.join(feedback.missing_topics)}
            - Additional keywords: {', '.join(feedback.additional_keywords)}
            - Comments: {feedback.comments}
            """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are TrendSurfer, an expert at identifying emerging trends and recent developments in research. You must respond with valid JSON only."),
            ("user", """Generate search queries focused on RECENT (last 2-3 years) and EMERGING trends.
            
            Research Topic: {topic}
            {feedback_context}
            
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
            "topic": research_topic,
            "feedback_context": feedback_context
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
        
        # Add Firecrawl web scraping for recent trends
        if self.firecrawl_api_key:
            firecrawl_results = self._search_with_firecrawl(query, max_results=5)
            results.extend(firecrawl_results)
        
        return results
    
    def _search_with_firecrawl(self, query: str, max_results: int = 5) -> List[LiteratureItem]:
        """Use Firecrawl to scrape recent blog posts, news articles, and research websites."""
        print(f"[TrendSurfer] Using Firecrawl to scrape web content for: {query}")
        results = []
        
        try:
            # Firecrawl API endpoint for search
            firecrawl_url = "https://api.firecrawl.dev/v1/search"
            headers = {
                "Authorization": f"Bearer {self.firecrawl_api_key}",
                "Content-Type": "application/json"
            }
            
            # Search for recent content
            payload = {
                "query": query,
                "limit": max_results,
                "scrapeOptions": {
                    "formats": ["markdown", "html"],
                    "onlyMainContent": True
                }
            }
            
            response = requests.post(firecrawl_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for idx, item in enumerate(data.get("data", [])[:max_results]):
                    # Extract content from Firecrawl response
                    title = item.get("title", f"Web Content {idx + 1}")
                    url = item.get("url", "")
                    content = item.get("markdown", item.get("html", ""))
                    
                    # Extract abstract/summary (first 500 chars of content)
                    abstract = content[:500] + "..." if len(content) > 500 else content
                    
                    lit_item = LiteratureItem(
                        title=title,
                        authors=["Web Source"],
                        abstract=abstract,
                        url=url,
                        source="Firecrawl (Web)",
                        agent="TrendSurfer",
                        year=datetime.now().year,
                        literature_type="web"
                    )
                    results.append(lit_item)
                
                print(f"[TrendSurfer] Firecrawl found {len(results)} web sources")
            else:
                print(f"[TrendSurfer] Firecrawl API error: {response.status_code}")
        
        except Exception as e:
            print(f"[TrendSurfer] Error with Firecrawl: {e}")
        
        return results


class TopicCrawler(BaseAgent):
    """Agent focused on comprehensive academic literature search + Firecrawl."""
    
    def __init__(self, openai_api_key: str, firecrawl_api_key: Optional[str] = None):
        super().__init__("TopicCrawler", openai_api_key, firecrawl_api_key)
    
    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        """Generate comprehensive academic search queries."""
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Relevant papers found: {', '.join(feedback.relevant_papers[:3])}
            - Missing topics: {', '.join(feedback.missing_topics)}
            - Additional keywords: {', '.join(feedback.additional_keywords)}
            """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are TopicCrawler, an expert at comprehensive academic literature searches. You must respond with valid JSON only."),
            ("user", """Generate broad and comprehensive search queries for peer-reviewed academic literature.
            
            Research Topic: {topic}
            {feedback_context}
            
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
            "topic": research_topic,
            "feedback_context": feedback_context
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
        
        # Add Firecrawl web scraping for academic institution pages
        if self.firecrawl_api_key:
            firecrawl_results = self._search_with_firecrawl(query, max_results=3)
            results.extend(firecrawl_results)
        
        return results
    
    def _search_with_firecrawl(self, query: str, max_results: int = 3) -> List[LiteratureItem]:
        """Use Firecrawl to scrape academic institution pages and research portals."""
        print(f"[TopicCrawler] Using Firecrawl to scrape academic content for: {query}")
        results = []
        
        try:
            # Firecrawl API endpoint for search
            firecrawl_url = "https://api.firecrawl.dev/v1/search"
            headers = {
                "Authorization": f"Bearer {self.firecrawl_api_key}",
                "Content-Type": "application/json"
            }
            
            # Search for academic content
            payload = {
                "query": f"{query} site:edu OR site:ac.uk OR research",
                "limit": max_results,
                "scrapeOptions": {
                    "formats": ["markdown"],
                    "onlyMainContent": True
                }
            }
            
            response = requests.post(firecrawl_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for idx, item in enumerate(data.get("data", [])[:max_results]):
                    title = item.get("title", f"Academic Web Content {idx + 1}")
                    url = item.get("url", "")
                    content = item.get("markdown", "")
                    
                    # Extract abstract/summary
                    abstract = content[:500] + "..." if len(content) > 500 else content
                    
                    lit_item = LiteratureItem(
                        title=title,
                        authors=["Academic Institution"],
                        abstract=abstract,
                        url=url,
                        source="Firecrawl (Academic Web)",
                        agent="TopicCrawler",
                        year=datetime.now().year,
                        literature_type="web"
                    )
                    results.append(lit_item)
                
                print(f"[TopicCrawler] Firecrawl found {len(results)} academic web sources")
            else:
                print(f"[TopicCrawler] Firecrawl API error: {response.status_code}")
        
        except Exception as e:
            print(f"[TopicCrawler] Error with Firecrawl: {e}")
        
        return results


class ScholarSearcher(BaseAgent):
    """Agent focused on finding highly-cited foundational papers."""
    
    def __init__(self, openai_api_key: str):
        super().__init__("ScholarSearcher", openai_api_key)
    
    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        """Generate queries for foundational and highly-cited work."""
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Additional keywords: {', '.join(feedback.additional_keywords)}
            - Comments: {feedback.comments}
            """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are ScholarSearcher, an expert at finding seminal and highly-cited foundational papers. You must respond with valid JSON only."),
            ("user", """Generate search queries for FOUNDATIONAL and HIGHLY-CITED papers.
            
            Research Topic: {topic}
            {feedback_context}
            
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
            "topic": research_topic,
            "feedback_context": feedback_context
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
    
    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        """Generate queries for grey literature."""
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Missing topics: {', '.join(feedback.missing_topics)}
            - Additional keywords: {', '.join(feedback.additional_keywords)}
            """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are GreyScout, an expert at finding grey literature like working papers, reports, and policy documents. You must respond with valid JSON only."),
            ("user", """Generate search queries for GREY LITERATURE (non-peer-reviewed but authoritative).
            
            Research Topic: {topic}
            {feedback_context}
            
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
            "topic": research_topic,
            "feedback_context": feedback_context
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
    """Orchestrates multiple agents with Firecrawl integration (No Human-in-the-Loop)."""
    
    def __init__(self, openai_api_key: Optional[str] = None, firecrawl_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        
        # Load Firecrawl API key from environment if not provided
        self.firecrawl_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        
        # Initialize all agents with Firecrawl support for TrendSurfer and TopicCrawler
        self.agents = {
            "TrendSurfer": TrendSurfer(self.api_key, self.firecrawl_api_key),
            "TopicCrawler": TopicCrawler(self.api_key, self.firecrawl_api_key),
            "ScholarSearcher": ScholarSearcher(self.api_key),
            "GreyScout": GreyScout(self.api_key)
        }
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=self.api_key)  # gpt-4o-mini, gpt-4o
        self.all_results: List[LiteratureItem] = []
        self.round_results: Dict[int, List[LiteratureItem]] = {}
    
    def run_search_round(
        self,
        research_topic: str,
        round_number: int,
        feedback: Optional[HumanFeedback] = None,
        max_results_per_agent: int = 10
    ) -> List[LiteratureItem]:
        """Run one round of multi-agent search."""
        print(f"\n{'='*70}")
        print(f"ROUND {round_number}: Multi-Agent Literature Search")
        print(f"Topic: {research_topic}")
        print(f"{'='*70}\n")
        
        round_results = []
        
        for agent_name, agent in self.agents.items():
            print(f"\n--- {agent_name} ---")
            
            # Refine query based on agent specialty and feedback
            search_query = agent.refine_query(research_topic, feedback)
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
        ranked_results = self._rank_by_relevance(unique_results, research_topic, feedback)
        
        self.round_results[round_number] = ranked_results
        self.all_results.extend(ranked_results)
        
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
        research_topic: str,
        feedback: Optional[HumanFeedback] = None
    ) -> List[LiteratureItem]:
        """Rank papers by relevance using LLM."""
        if not papers:
            return []
        
        print(f"[Orchestrator] Ranking {len(papers)} papers...")
        
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Consider this feedback:
            - Relevant papers: {', '.join(feedback.relevant_papers[:3])}
            - Avoid topics like: {', '.join(feedback.irrelevant_papers[:3])}
            - Focus more on: {', '.join(feedback.missing_topics)}
            """
        
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
                {feedback_context}
                
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
                    "feedback_context": feedback_context,
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
    
    def run_automated_search(
        self,
        research_topic: str,
        max_results_per_agent: int = 10
    ) -> List[LiteratureItem]:
        """Run automated single-round search without human feedback."""
        print(f"\n{'='*70}")
        print(f"AUTOMATED LITERATURE SEARCH WITH FIRECRAWL")
        print(f"Topic: {research_topic}")
        print(f"{'='*70}\n")
        
        all_results = []
        
        for agent_name, agent in self.agents.items():
            print(f"\n--- {agent_name} ---")
            
            # Refine query based on agent specialty (no feedback)
            search_query = agent.refine_query(research_topic, feedback=None)
            print(f"Generated {len(search_query.queries)} queries")
            print(f"Keywords: {', '.join(search_query.keywords[:5])}...")
            
            # Execute searches
            for query in search_query.queries[:2]:  # Use top 2 queries per agent
                results = agent.search(query, max_results_per_agent)
                all_results.extend(results)
                print(f"  Found {len(results)} items for query: {query[:50]}...")
        
        # Deduplicate
        unique_results = self._deduplicate(all_results)
        print(f"\n[Orchestrator] Total unique items found: {len(unique_results)}")
        
        # Rank by relevance
        ranked_results = self._rank_by_relevance(unique_results, research_topic, feedback=None)
        
        # Store results
        self.all_results = ranked_results
        
        return ranked_results
    
    def collect_human_feedback(self, round_number: int) -> HumanFeedback:
        """Collect feedback from human researcher (interactive)."""
        print(f"\n{'='*70}")
        print(f"HUMAN FEEDBACK - Round {round_number}")
        print(f"{'='*70}\n")
        
        print("Please provide feedback on the results:")
        print("(Press Enter to skip any field)\n")
        
        relevant = input("Relevant paper titles (comma-separated): ").strip()
        irrelevant = input("Irrelevant paper titles (comma-separated): ").strip()
        missing = input("Missing topics to explore (comma-separated): ").strip()
        keywords = input("Additional keywords (comma-separated): ").strip()
        comments = input("General comments: ").strip()
        
        feedback = HumanFeedback(
            round_number=round_number,
            relevant_papers=[p.strip() for p in relevant.split(",") if p.strip()],
            irrelevant_papers=[p.strip() for p in irrelevant.split(",") if p.strip()],
            missing_topics=[t.strip() for t in missing.split(",") if t.strip()],
            additional_keywords=[k.strip() for k in keywords.split(",") if k.strip()],
            comments=comments
        )
        
        return feedback
    
    def save_results(self, filename: str, round_number: Optional[int] = None):
        """Save results to CSV."""
        if round_number:
            results = self.round_results.get(round_number, [])
            filename = f"round{round_number}_{filename}"
        else:
            results = self.all_results
        
        if not results:
            print("No results to save.")
            return
        
        data = [item.model_dump() for item in results]
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"[Orchestrator] Results saved to {filename}")
    
    def print_summary(self, round_number: Optional[int] = None, top_n: int = 10):
        """Print summary of results."""
        if round_number:
            results = self.round_results.get(round_number, [])
            title = f"ROUND {round_number} - TOP {top_n} PAPERS"
        else:
            results = self.all_results
            title = f"ALL ROUNDS - TOP {top_n} PAPERS"
        
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
    """Main function with two-round human-in-the-loop process."""
    # example
    research_topic = "Agent-based modeling in macroeconomics and monetary policy"
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # ROUND 1: Initial search
    print("\n" + "="*70)
    print("STARTING TWO-ROUND LITERATURE SEARCH WITH HUMAN FEEDBACK")
    print("="*70)
    
    round1_results = orchestrator.run_search_round(
        research_topic=research_topic,
        round_number=1,
        feedback=None,
        max_results_per_agent=8
    )
    
    # Display Round 1 results
    orchestrator.print_summary(round_number=1, top_n=15)
    orchestrator.save_results("literature_results.csv", round_number=1)
    
    # Collect human feedback
    feedback = orchestrator.collect_human_feedback(round_number=1)
    
    # Save feedback
    with open("round1_feedback.json", "w") as f:
        json.dump(feedback.model_dump(), f, indent=2)
    print("[Orchestrator] Feedback saved to round1_feedback.json")
    
    # ROUND 2: Refined search with feedback
    round2_results = orchestrator.run_search_round(
        research_topic=research_topic,
        round_number=2,
        feedback=feedback,
        max_results_per_agent=8
    )
    
    # Display Round 2 results
    orchestrator.print_summary(round_number=2, top_n=15)
    orchestrator.save_results("literature_results.csv", round_number=2)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Round 1: {len(round1_results)} papers")
    print(f"Round 2: {len(round2_results)} papers")
    print(f"Total unique papers: {len(orchestrator.all_results)}")
    
    # Save all results
    orchestrator.save_results("literature_results_all_rounds.csv")


if __name__ == "__main__":
    main()
