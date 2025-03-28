import os
from pathlib import Path
from flask import Flask
from app.helpers.load_json_from_file import load_json_from_file
from dotenv import load_dotenv
import io
import logging
from contextlib import redirect_stdout
from flask_cors import CORS

from app.conf.websocket import socketio, init_socketio

load_dotenv()  # Load environment variables (for OpenAI API key, etc.)
from pathlib import Path

from agno.tools.duckdb import DuckDbTools
from agno.utils.log import logger  

from app.services.agent import initialize_agent

app = Flask(__name__)


FRONTEND_URL = os.getenv("FRONTEND_URL")
CORS(app, resources={r"/*": {"origins": [FRONTEND_URL, "http://localhost:3000"]}})


init_socketio(app, [FRONTEND_URL, "http://localhost:3000"])

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Get OpenAI API Key
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
  
# Set up directory paths
cwd = Path(__file__).parent.resolve()  # Current working directory
data_dir = cwd.parent.joinpath("data")  # Directory for data files
if not data_dir.exists():
    data_dir.mkdir(parents=True)

# Create output directory for saving query results
output_dir = cwd.parent.joinpath("output")
if not output_dir.exists():
    output_dir.mkdir(parents=True)

# --- Agent Initialization ---
# data_analyst = initialize_agent(data_dir)

# --- Register Routes ---
from app.routes.api import api_bp

# Register blueprints
app.register_blueprint(api_bp)

# Pass the necessary objects to the routes
# app.config['DATA_ANALYST'] = data_analyst
app.config['SEMANTIC_MODEL'] = load_json_from_file(data_dir.joinpath("semantic_model.json"))
app.config['DATA_DIR'] = data_dir
app.config['OUTPUT_DIR'] = output_dir

# Setup websocket routes
from app.routes.websocket import setup_websocket_routes
setup_websocket_routes(socketio)

# reminder for melker since he has the bad habit of always running wsl as root
print('REMEMBER TO USE THE AGENT WITH APPROPRIATE PERMISSIONS!!!!')
print('REMEMBER TO USE THE AGENT WITH APPROPRIATE PERMISSIONS!!!!')
print('REMEMBER TO USE THE AGENT WITH APPROPRIATE PERMISSIONS!!!!')
print('REMEMBER TO USE THE AGENT WITH APPROPRIATE PERMISSIONS!!!!')
print('REMEMBER TO USE THE AGENT WITH APPROPRIATE PERMISSIONS!!!!')

# backend/run.py or backend/main.py
from app.app import app, socketio

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', allow_unsafe_werkzeug=True)