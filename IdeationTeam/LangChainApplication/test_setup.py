"""
Quick test script to verify the setup is working correctly.
This script tests imports and basic functionality without making API calls.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import langchain
        print(f"✓ langchain {langchain.__version__}")
    except ImportError as e:
        print(f"✗ langchain import failed: {e}")
        return False
    
    try:
        import langchain_openai
        print(f"✓ langchain_openai")
    except ImportError as e:
        print(f"✗ langchain_openai import failed: {e}")
        return False
    
    try:
        import langchain_community
        print(f"✓ langchain_community")
    except ImportError as e:
        print(f"✗ langchain_community import failed: {e}")
        return False
    
    try:
        import openai
        print(f"✓ openai {openai.__version__}")
    except ImportError as e:
        print(f"✗ openai import failed: {e}")
        return False
    
    try:
        import arxiv
        print(f"✓ arxiv")
    except ImportError as e:
        print(f"✗ arxiv import failed: {e}")
        return False
    
    try:
        import pandas
        print(f"✓ pandas {pandas.__version__}")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print(f"✓ python-dotenv")
    except ImportError as e:
        print(f"✗ python-dotenv import failed: {e}")
        return False
    
    print("\nAll imports successful! ✓")
    return True


def test_data_models():
    """Test that Pydantic models can be instantiated."""
    print("\nTesting data models...")
    
    try:
        from pydantic import BaseModel, Field
        from typing import List, Optional
        
        class TestModel(BaseModel):
            title: str
            authors: List[str]
            year: Optional[int] = None
        
        test_item = TestModel(
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            year=2023
        )
        
        print(f"✓ Pydantic models working")
        print(f"  Test item: {test_item.title}")
        return True
    
    except Exception as e:
        print(f"✗ Pydantic model test failed: {e}")
        return False


def test_env_file():
    """Check if .env file exists."""
    print("\nChecking environment configuration...")
    
    import os
    from pathlib import Path
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✓ .env file exists")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            print("✓ OPENAI_API_KEY is set")
            return True
        else:
            print("⚠ OPENAI_API_KEY not set or using placeholder")
            print("  Please update your .env file with a valid API key")
            return False
    else:
        print("⚠ .env file not found")
        if env_example.exists():
            print("  Copy .env.example to .env and add your API key:")
            print("  cp .env.example .env")
        return False


def test_arxiv_search():
    """Test a simple arXiv search without API key."""
    print("\nTesting arXiv search (no API key required)...")
    
    try:
        import arxiv
        
        search = arxiv.Search(
            query="agent-based modeling",
            max_results=2
        )
        
        results = list(search.results())
        
        if results:
            print(f"✓ arXiv search working")
            print(f"  Found {len(results)} papers")
            print(f"  Example: {results[0].title[:60]}...")
            return True
        else:
            print("⚠ arXiv search returned no results")
            return False
    
    except Exception as e:
        print(f"✗ arXiv search failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("LangChain Literature Sourcing - Setup Test")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Data Models", test_data_models()))
    results.append(("Environment", test_env_file()))
    results.append(("arXiv Search", test_arxiv_search()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests passed! Your setup is ready to use.")
        print("\nNext steps:")
        print("1. Make sure your .env file has a valid OPENAI_API_KEY")
        print("2. Run: python 1-SourcingStage.py")
    else:
        print("Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("- Missing .env file: Copy .env.example to .env")
        print("- Missing API key: Add your OpenAI API key to .env")
        print("- Import errors: Make sure virtual environment is activated")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
