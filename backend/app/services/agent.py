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
from huggingface_hub.inference_api import InferenceApi

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

    # HuggingFace Version using OpenAI-like interface (Agno expects certain interfaces that Hugging Face's API doesn't naturally provide.)
    class HuggingFaceModel:
        def __init__(self, api_token):
            self.api = InferenceApi(
                repo_id="Qwen/Qwen-72B-Chat",
                token=api_token
            )
            self.tools = None
            self.functions = None
            self.id = "Qwen/Qwen-72B-Chat"
            self.name = "Qwen-72B-Chat"
            self.provider = "huggingface"
            # Add these attributes for Agno compatibility
            self.assistant_message_role = "assistant"
            self.supports_vision = False
            self.supports_function_calling = False  # We're faking this functionality
            self.supports_tools = False  # We're faking this functionality
        
        def set_tools(self, tools):
            self.tools = tools
        
        def set_functions(self, functions):
            self.functions = functions
        
        def response(self, messages):
            return self.chat(messages)
        
        def chat(self, messages):
            try:
                formatted_messages = []
                print(f"Processing {len(messages)} messages")
                
                for msg in messages:
                    # Handle Agno Message objects
                    role = msg.role if hasattr(msg, 'role') else msg.get('role', '')
                    content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    
                    print(f"Message: role={role}, content preview={content[:30]}...")
                    
                    if role == 'system':
                        formatted_messages.append(f"<|im_start|>system\n{content}\n<|im_end|>")
                    elif role == 'user':
                        formatted_messages.append(f"<|im_start|>user\n{content}\n<|im_end|>")
                    elif role == 'assistant':
                        formatted_messages.append(f"<|im_start|>assistant\n{content}\n<|im_end|>")
                
                prompt = "\n".join(formatted_messages)
                prompt += "\n<|im_start|>assistant\n"
                
                print(f"Sending prompt to Hugging Face API (length: {len(prompt)})")
                print(f"First 100 chars: {prompt[:100]}...")
                
                try:
                    # Use raw_response parameter to get the raw response object
                    response = self.api(prompt, raw_response=True)
                    print(f"API response status code: {response.status_code}")
                    
                    # Get the text from the raw response
                    response_text = ""
                    if response.status_code == 200:
                        response_json = response.json()
                        print(f"Response JSON: {str(response_json)[:200]}...")
                        if isinstance(response_json, list) and len(response_json) > 0:
                            response_text = response_json[0].get('generated_text', '')
                        else:
                            response_text = str(response_json)
                    else:
                        print(f"Error response: {response.text}")
                        response_text = f"Error getting response: HTTP {response.status_code}"
                except Exception as api_error:
                    print(f"Error calling Hugging Face API: {str(api_error)}")
                    # Fallback to simpler API call if the raw_response approach fails
                    try:
                        response_text = self.api(prompt)
                        print(f"Got response using simple API call: {str(response_text)[:100]}...")
                    except Exception as fallback_error:
                        print(f"Fallback API call also failed: {str(fallback_error)}")
                        response_text = f"Error: {str(fallback_error)}"
                
                print(f"Final response text: {response_text[:100]}...")
                
                # Create a Message object that Agno expects
                from agno.agent import Message
                return Message(
                    role="assistant",
                    content=response_text
                )
                
            except Exception as e:
                print(f"Error in HuggingFaceModel chat: {str(e)}")
                from agno.agent import Message
                return Message(
                    role="assistant",
                    content=f"Error: {str(e)}"
                )

        def function_call(self, messages, functions=None):
            # This is a compatibility method for Agno
            print("function_call method called - using regular chat instead")
            return self.chat(messages)

        def __str__(self):
            return f"HuggingFaceModel({self.id})"

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
  
    

