"""
Literature Sourcing Stage using LangChain
This script uses LangChain to gather relevant academic literature from various sources.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import pandas as pd
import arxiv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()


class LiteratureItem(BaseModel):
    """Model for a literature item."""
    title: str = Field(description="Title of the paper")
    authors: List[str] = Field(description="List of authors")
    abstract: str = Field(description="Abstract or summary")
    url: str = Field(description="URL or identifier")
    source: str = Field(description="Source (e.g., arXiv, Google Scholar)")
    year: Optional[int] = Field(description="Publication year", default=None)
    relevance_score: Optional[float] = Field(description="Relevance score (0-1)", default=None)


class SearchQuery(BaseModel):
    """Model for refined search queries."""
    queries: List[str] = Field(description="List of refined search queries")
    keywords: List[str] = Field(description="Key terms to search for")


class LiteratureSourcingAgent:
    """Agent for sourcing academic literature using LangChain."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the literature sourcing agent."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            openai_api_key=self.api_key
        )
        
        self.literature_items: List[LiteratureItem] = []
    
    def refine_research_query(self, research_topic: str) -> SearchQuery:
        """Use LLM to refine research topic into specific search queries."""
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert research librarian specializing in economics and social sciences."),
            ("user", """Given the following research topic, generate 3-5 specific search queries 
            and extract 5-10 key terms that would be effective for finding relevant academic literature.
            
            Research Topic: {topic}
            
            {format_instructions}
            """)
        ])
        
        chain = prompt | self.llm | parser
        
        result = chain.invoke({
            "topic": research_topic,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
    
    def search_arxiv(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search arXiv for relevant papers."""
        print(f"Searching arXiv for: {query}")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in search.results():
            item = LiteratureItem(
                title=paper.title,
                authors=[author.name for author in paper.authors],
                abstract=paper.summary,
                url=paper.entry_id,
                source="arXiv",
                year=paper.published.year if paper.published else None
            )
            results.append(item)
        
        return results
    
    def search_semantic_scholar(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search Semantic Scholar API for relevant papers."""
        print(f"Searching Semantic Scholar for: {query}")
        
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,url,year,citationCount"
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for paper in data.get("data", []):
                item = LiteratureItem(
                    title=paper.get("title", ""),
                    authors=[author.get("name", "") for author in paper.get("authors", [])],
                    abstract=paper.get("abstract", "No abstract available"),
                    url=paper.get("url", ""),
                    source="Semantic Scholar",
                    year=paper.get("year")
                )
                results.append(item)
            
            return results
        
        except Exception as e:
            print(f"Error searching Semantic Scholar: {e}")
            return []
    
    def rank_papers_by_relevance(self, papers: List[LiteratureItem], research_topic: str) -> List[LiteratureItem]:
        """Use LLM to rank papers by relevance to the research topic."""
        print(f"Ranking {len(papers)} papers by relevance...")
        
        # Process in batches to avoid token limits
        batch_size = 5
        ranked_papers = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            
            papers_text = "\n\n".join([
                f"Paper {idx+1}:\nTitle: {p.title}\nAbstract: {p.abstract[:300]}..."
                for idx, p in enumerate(batch)
            ])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert research evaluator."),
                ("user", """Rate the relevance of each paper to the research topic on a scale of 0-1.
                
                Research Topic: {topic}
                
                Papers:
                {papers}
                
                Provide relevance scores as a comma-separated list (e.g., 0.9, 0.7, 0.5, 0.3, 0.8).
                Only provide the numbers, no other text.
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
                print(f"Error ranking batch: {e}")
                # Add papers without scores
                ranked_papers.extend(batch)
        
        # Sort by relevance score
        ranked_papers.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        return ranked_papers
    
    def gather_literature(
        self,
        research_topic: str,
        max_results_per_source: int = 10,
        rank_results: bool = True
    ) -> List[LiteratureItem]:
        """Main method to gather literature from multiple sources."""
        print(f"\n{'='*60}")
        print(f"Starting literature sourcing for: {research_topic}")
        print(f"{'='*60}\n")
        
        # Step 1: Refine the research query
        print("Step 1: Refining research queries...")
        refined_queries = self.refine_research_query(research_topic)
        print(f"Generated {len(refined_queries.queries)} search queries")
        print(f"Keywords: {', '.join(refined_queries.keywords)}\n")
        
        # Step 2: Search multiple sources
        all_papers = []
        
        for query in refined_queries.queries[:3]:  # Use top 3 queries
            # Search arXiv
            arxiv_results = self.search_arxiv(query, max_results_per_source)
            all_papers.extend(arxiv_results)
            
            # Search Semantic Scholar
            semantic_results = self.search_semantic_scholar(query, max_results_per_source)
            all_papers.extend(semantic_results)
        
        # Remove duplicates based on title similarity
        unique_papers = self._deduplicate_papers(all_papers)
        print(f"\nFound {len(unique_papers)} unique papers")
        
        # Step 3: Rank papers by relevance
        if rank_results and unique_papers:
            ranked_papers = self.rank_papers_by_relevance(unique_papers, research_topic)
        else:
            ranked_papers = unique_papers
        
        self.literature_items = ranked_papers
        
        print(f"\n{'='*60}")
        print(f"Literature sourcing complete!")
        print(f"Total papers gathered: {len(ranked_papers)}")
        print(f"{'='*60}\n")
        
        return ranked_papers
    
    def _deduplicate_papers(self, papers: List[LiteratureItem]) -> List[LiteratureItem]:
        """Remove duplicate papers based on title similarity."""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Normalize title for comparison
            normalized_title = paper.title.lower().strip()
            
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)
        
        return unique_papers
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export literature items to a pandas DataFrame."""
        if not self.literature_items:
            return pd.DataFrame()
        
        data = [item.model_dump() for item in self.literature_items]
        df = pd.DataFrame(data)
        
        return df
    
    def save_to_csv(self, filename: str = "literature_results.csv"):
        """Save literature results to a CSV file."""
        df = self.export_to_dataframe()
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def print_summary(self, top_n: int = 10):
        """Print a summary of the top N papers."""
        print(f"\n{'='*60}")
        print(f"TOP {top_n} PAPERS")
        print(f"{'='*60}\n")
        
        for idx, paper in enumerate(self.literature_items[:top_n], 1):
            print(f"{idx}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   Source: {paper.source} | Year: {paper.year or 'N/A'}")
            if paper.relevance_score:
                print(f"   Relevance: {paper.relevance_score:.2f}")
            print(f"   URL: {paper.url}")
            print(f"   Abstract: {paper.abstract[:200]}...")
            print()


def main():
    """Main function to demonstrate the literature sourcing agent."""
    
    # Example usage
    research_topic = "Agent-based modeling in macroeconomics and monetary policy"
    
    # Initialize the agent
    agent = LiteratureSourcingAgent()
    
    # Gather literature
    papers = agent.gather_literature(
        research_topic=research_topic,
        max_results_per_source=10,
        rank_results=True
    )
    
    # Print summary
    agent.print_summary(top_n=10)
    
    # Save results
    agent.save_to_csv("economics_literature.csv")
    
    # Export to DataFrame for further analysis
    df = agent.export_to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
