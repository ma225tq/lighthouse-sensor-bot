from flask import Blueprint, request, jsonify, current_app
from app.helpers.load_json_from_file import load_json_from_file
from dotenv import load_dotenv
from app.services.query import query
from app.conf.CustomDuckDbTools import CustomDuckDbTools

load_dotenv()
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route("/")
def hello_world():
    return jsonify({"message": "Hello, World!"})

@api_bp.route("/query", methods=["GET", "POST"])
def query_endpoint():
    data = request.get_json()
    
    # Get the necessary objects from app config
    data_dir = current_app.config['DATA_DIR']
    output_dir = current_app.config['OUTPUT_DIR']
    
    # Get source_file from request if provided
    source_file = data.get('source_file')
    
    if not source_file:
        return jsonify({"error": "Source file is required"}), 400
    
    # Get model_id from request if provided
    llm_model_id = data.get('llm_model_id') 
    
    if not llm_model_id:
        return jsonify({"error": "LLM Model ID is required"}), 400
    
    # If source_file is provided, create a new agent with the source_file
    if source_file:
        semantic_model_data = load_json_from_file(data_dir.joinpath("semantic_model.json"))
        if semantic_model_data is None:
            print("Error: Could not load semantic model. Exiting.")
            exit()
    
        # Create a new instance of CustomDuckDbTools with the source_file
        duck_tools = CustomDuckDbTools(
            data_dir=str(data_dir),
            semantic_model=current_app.config['SEMANTIC_MODEL'],
            source_file=source_file
        )
        
        # Initialize a new agent for this request with the custom tools
        from app.services.agent import initialize_agent
        data_analyst = initialize_agent(data_dir, llm_model_id, [duck_tools])
        
        # Add source file specific instructions
        additional_instructions = [
            f"IMPORTANT: Use the file '{source_file}' as your primary data source.",
            f"When you need to create a table, use 'data' as the table name and it will automatically use the file '{source_file}'."
        ]
        data_analyst.instructions = data_analyst.instructions + additional_instructions
    
    # Call the query service
    return query(
        data=data, 
        data_dir=data_dir,
        data_analyst=data_analyst
    )

@api_bp.route("/test", methods=["GET"])
def test_connection():
    """Test endpoint to verify the connection between frontend and backend"""
    return jsonify({
        "content": "Backend connection test successful",
        "status": "online"
    })

@api_bp.route("/evaluate", methods=["POST"])
def evaluate_endpoint():
    data = request.get_json()
    model_id = data.get('model_id')
    question = data.get('question', '')
    response = data.get('response', '')
    
    if not model_id:
        return jsonify({"error": "Model ID is required"}), 400
        
    # Run the evaluation
    try:
        from app.ragas.scripts.ragas_tests import run_evaluation
        from app.helpers.save_query_to_db import save_query_with_eval_to_db
        import numpy as np
        import math
        
        results, df = run_evaluation(model_id)
        
        # Fix NaN and numpy types in results before serializing to JSON
        for key, value in results.items():
            # Replace NaN values with 0
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                results[key] = 0.0
            
            # Convert numpy types to Python native types
            if isinstance(value, np.bool_):
                results[key] = bool(value)
            elif isinstance(value, np.integer):
                results[key] = int(value)
            elif isinstance(value, np.floating):
                if math.isnan(value) or math.isinf(value):
                    results[key] = 0.0
                else:
                    results[key] = float(value)
            elif isinstance(value, np.ndarray):
                results[key] = value.tolist()
        
        # Check if there was an error
        if isinstance(results, dict) and "error" in results:
            # Still return 200 so frontend can display the error message
            return jsonify({
                "error": f"Evaluation encountered an error: {results['error']}",
                "results": results
            }), 200
        
        # Try to save to database but continue if it fails
        try:
            save_query_with_eval_to_db(
                query=question,
                direct_response=response,
                full_response=response,
                evaluation_results=results,
            )
        except Exception as db_error:
            print(f"Warning: Failed to save to database: {db_error}")
            # Continue even if database save fails
        
        return jsonify({"results": results})
    except Exception as e:
        error_message = f"Error during evaluation: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

 