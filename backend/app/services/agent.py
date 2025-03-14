import json
from pathlib import Path
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckdb import DuckDbTools
from agno.utils.log import logger
import utils.duck
from typing import Optional
from app.helpers.CustomDuckDbTools import CustomDuckDbTools
from app.helpers.load_json_from_file import load_json_from_file


def initialize_agent(data_dir):
    """Initialize the agent with the necessary tools and configuration
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        The initialized agent and tools
    """
    semantic_model_data = load_json_from_file(data_dir.joinpath("semantic_model.json"))
    if semantic_model_data is None:
        print("Error: Could not load semantic model. Exiting.")
        exit()
    
    semantic_instructions = utils.duck.get_default_instructions(semantic_model_data)

    # Create a custom DuckDbTools that uses local paths
    # duck_tools = CustomDuckDbTools(data_dir=str(data_dir), semantic_model=semantic_model_data)

    data_analyst = Agent(  
        instructions=semantic_instructions,
        system_message=utils.duck.get_system_message(semantic_instructions, semantic_model_data),
        tools=DuckDbTools(),  # Initialize with DuckDbTools
        show_tool_calls=False,
        model=OpenAIChat(id="gpt-4o"), # or gpt-3.5-turbo if you prefer
        markdown=True
    )
    
    return data_analyst 
  # duck_tools
