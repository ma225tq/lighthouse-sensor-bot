import json
from pathlib import Path
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.tools.duckdb import DuckDbTools
from agno.utils.log import logger
import utils.duck
from typing import Optional, List
from app.helpers.CustomDuckDbTools import CustomDuckDbTools
from app.helpers.load_json_from_file import load_json_from_file
from dotenv import load_dotenv
import os
from textwrap import dedent
from app.services.huggingface_model import HuggingFaceModel

load_dotenv()


def initialize_agent(data_dir):
    """Initialize the agent with the necessary tools and configuration

    Args:
        data_dir: Path to the data directory

    Returns:
        The initialized agent and tools
    """
    semantic_model_data = load_json_from_file(data_dir.joinpath("semantic_model.json"))  # Här laddas semantic model från en JSON-fil
    if semantic_model_data is None:
        print("Error: Could not load semantic model. Exiting.")
        exit()

    semantic_instructions = utils.duck.get_default_instructions(semantic_model_data)  # Här hämtas standard instruktioner från semantic model

    # Create a custom system message with stronger formatting instructions
    custom_system_message = (
        "You are a Data Engineering expert designed to perform tasks using DuckDb.\n\n"
    )

    # Add explicit formatting instructions at the beginning for emphasis
    custom_system_message += dedent(
        """\
        ## CRITICAL OUTPUT FORMAT REQUIREMENT:
        You MUST structure your response in exactly this format:
        1. First section: Your reasoning and SQL queries (start with "## Analysis")
        2. Second section: ONLY the direct answer with NO planning, NO SQL, and NO explanation (start with "## Answer")
        
        """
    )

    # Add the rest of the standard system message
    standard_system_message = utils.duck.get_system_message(
        semantic_instructions, semantic_model_data
    )
    # Remove the initial "You are a Data Engineering expert..." part to avoid duplication
    if standard_system_message.startswith("You are a Data Engineering expert"):
        standard_system_message = "\n\n".join(standard_system_message.split("\n\n")[1:])

    custom_system_message += standard_system_message

    # OpenRouter Version
    # BASE_URL = os.getenv("OPENROUTER_BASE_URL")
    # API_KEY_OPENROUTER = os.getenv("OPENROUTER_API_KEY")

    # data_analyst = Agent(
    #     instructions=semantic_instructions,
    #     system_message=custom_system_message,
    #     tools=DuckDbTools(),  # DuckDB verktyg för att köra SQL-frågor lokalt
    #     show_tool_calls=False,
    #     model=OpenRouter(
    #         base_url=BASE_URL, api_key=API_KEY_OPENROUTER, id="qwen/qwen-2.5-72b-instruct"
    #     ),
    #     tool_choice="required",
    #     tool_call_limit=20,
    #     markdown=True,
    # )

    # Create our agent using the external HuggingFaceModel implementation
    data_analyst = Agent(
        instructions=semantic_instructions,
        system_message=custom_system_message,
        tools=DuckDbTools(),
        show_tool_calls=False,
        model=HuggingFaceModel(os.getenv("HUGGINGFACE_API_KEY")),
        tool_choice="required",
        tool_call_limit=20,
        markdown=True,
    )

    return data_analyst


# duck_tools


def get_system_message(instructions, semantic_model) -> List[str]:
    system_message = (
        "You are a Data Engineering expert designed to perform tasks using DuckDb."
    )
    system_message += "\n\n"

    # Add formatting instructions at the beginning for emphasis
    system_message += dedent(
        """\
        ## CRITICAL OUTPUT FORMAT REQUIREMENT:
        You MUST structure your response in exactly this format:
        1. First section: ONLY the direct answer with NO planning, NO SQL, and NO explanation
        2. Second section: Your reasoning and SQL queries (start with "## Analysis")
        
        """
    )

    # Arbetsflöde:
    # Agent tar emot frågan
    # LLM genererar SQL
    # DuckDbTools utför SQL-operationerna
    # Resultatet går tillbaka till LLM för formatering
  
    

