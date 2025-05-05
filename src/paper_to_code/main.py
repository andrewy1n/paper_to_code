#!/usr/bin/env python
import sys
import warnings
from crewai_tools import FileReadTool

from datetime import datetime

from paper_to_code.crew import PaperToCode

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """

    try:
        reader = PdfReader("1-s2.0-S0098300418305909-main.pdf")
        text = " ".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        raise RuntimeError(f"PDF reading failed: {str(e)}")
    
    inputs = {
        'research_paper_txt': text,
    }
    
    try:
        PaperToCode().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        PaperToCode().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        PaperToCode().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        PaperToCode().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
