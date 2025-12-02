# How to Run the Multi-Agent Literature Sourcing System

## Important: Always Use the Virtual Environment!

The script requires packages installed in the virtual environment. **Do not run with system Python.**

## Step-by-Step Instructions

### 1. Open PowerShell in the Project Directory

```powershell
cd c:\Users\zwang3\Documents\GitHub\awesome-agentic-econ-workflows\IdeationTeam\LangChainApplication
```

### 2. Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the beginning of your prompt:
```
(venv) PS C:\Users\zwang3\Documents\GitHub\awesome-agentic-econ-workflows\IdeationTeam\LangChainApplication>
```

### 3. Set Your OpenAI API Key

Create a `.env` file if you haven't already:

```powershell
cp .env.example .env
```

Edit `.env` and add your API key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### 4. Run the Script

**Interactive mode (with feedback prompts):**
```powershell
python 1-SourcingStage.py
```

**Programmatic mode (no prompts):**
```powershell
python example_programmatic_feedback.py
```

### 5. When Done, Deactivate

```powershell
deactivate
```

## Common Errors and Solutions

### Error: `ModuleNotFoundError: No module named 'langchain'`

**Problem**: Running with system Python instead of virtual environment

**Solution**: 
1. Activate the virtual environment first: `.\venv\Scripts\Activate.ps1`
2. Then run the script: `python 1-SourcingStage.py`

### Error: `ModuleNotFoundError: No module named 'dotenv'`

**Problem**: Packages not installed in virtual environment

**Solution**:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Error: `ValueError: OpenAI API key is required`

**Problem**: Missing or invalid `.env` file

**Solution**:
1. Create `.env` file: `cp .env.example .env`
2. Edit `.env` and add your OpenAI API key
3. Make sure the file is in the same directory as the script

### Error: Script runs but uses wrong Python

**Problem**: VS Code or IDE using system Python

**Solution**:
- In VS Code: Select the virtual environment Python interpreter
  - Press `Ctrl+Shift+P`
  - Type "Python: Select Interpreter"
  - Choose `.\venv\Scripts\python.exe`

## Running from VS Code

### Option 1: Use Integrated Terminal

1. Open integrated terminal (`Ctrl+``)
2. Activate venv: `.\venv\Scripts\Activate.ps1`
3. Run: `python 1-SourcingStage.py`

### Option 2: Configure Python Interpreter

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `.\venv\Scripts\python.exe`
4. Click "Run" button or press `F5`

## Verifying Your Setup

Run this test to verify everything is working:

```powershell
.\venv\Scripts\Activate.ps1
python test_setup.py
```

All tests should pass except the environment test (if you haven't set up `.env` yet).

## Quick Reference

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Run interactive mode
python 1-SourcingStage.py

# Run programmatic mode
python example_programmatic_feedback.py

# Run tests
python test_setup.py

# Deactivate venv
deactivate
```

## What the Script Does

1. **Round 1**: All four agents search independently
   - TrendSurfer: Recent trends
   - TopicCrawler: Comprehensive academic search
   - ScholarSearcher: Highly-cited papers
   - GreyScout: Grey literature

2. **Feedback**: You provide input on results

3. **Round 2**: Agents refine search based on feedback

4. **Output**: CSV files with all results

## Expected Runtime

- **Round 1**: 2-5 minutes (depending on API response times)
- **Feedback**: As long as you need
- **Round 2**: 2-5 minutes
- **Total**: ~10-15 minutes for full two-round process

## Output Files

After running, you'll find:
- `round1_literature_results.csv`
- `round2_literature_results.csv`
- `literature_results_all_rounds.csv`
- `round1_feedback.json`

## Tips

1. **Always activate the virtual environment first**
2. **Make sure you have a valid OpenAI API key**
3. **Be patient** - API calls take time
4. **Provide detailed feedback** for better Round 2 results
5. **Check the CSV files** for complete results

## Troubleshooting Checklist

- [ ] Virtual environment activated? (see `(venv)` in prompt)
- [ ] Packages installed? (`pip list | Select-String langchain`)
- [ ] `.env` file exists with valid API key?
- [ ] Running from correct directory?
- [ ] Using correct Python? (`python --version` should show venv Python)

## Need Help?

1. Check `README.md` for detailed documentation
2. See `MULTI_AGENT_GUIDE.md` for usage examples
3. Review `IMPLEMENTATION_SUMMARY.md` for architecture details
