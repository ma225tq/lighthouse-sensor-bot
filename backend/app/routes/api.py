import json
from flask import Blueprint, request, jsonify, current_app, Response
from app.helpers.load_json_from_file import load_json_from_file
from dotenv import load_dotenv
from app.services.query import query
from app.conf.CustomDuckDbTools import CustomDuckDbTools
import pandas as pd
from app.services.query_with_eval import query_with_eval
from app.conf.postgres import get_cursor
import logging
from pathlib import Path
from collections import OrderedDict

load_dotenv()
api_bp = Blueprint("api", __name__, url_prefix="/api")
logger = logging.getLogger(__name__)


@api_bp.route("/")
def hello_world():
    return jsonify({"message": "Hello, World!"})


@api_bp.route("/query", methods=["GET", "POST"])
def query_endpoint():
    data = request.get_json()

    # Get the necessary objects from app config
    data_dir = current_app.config["DATA_DIR"]
    # Get source_file from request if provided
    source_file = data.get("source_file")

    if not source_file:
        return jsonify({"error": "Source file is required"}), 400

    # Get model_id from request if provided
    llm_model_id = data.get("llm_model_id")

    if not llm_model_id:
        return jsonify({"error": "LLM Model ID is required"}), 400

    # If source_file is provided, create a new agent with the source_file
    if source_file:
        semantic_model_data = load_json_from_file(
            data_dir.joinpath("semantic_model.json")
        )
        if semantic_model_data is None:
            print("Error: Could not load semantic model. Exiting.")
            exit()

        # Create a new instance of CustomDuckDbTools with the source_file
        duck_tools = CustomDuckDbTools(
            data_dir=str(data_dir),
            semantic_model=current_app.config["SEMANTIC_MODEL"],
            source_file=source_file,
        )

        # Initialize a new agent for this request with the custom tools
        from app.services.agent import initialize_agent

        data_analyst = initialize_agent(data_dir, llm_model_id, [duck_tools])

        # Add source file specific instructions
        additional_instructions = [
            f"IMPORTANT: Use the file '{source_file}' as your primary data source.",
            f"When you need to create a table, use 'data' as the table name and it will automatically use the file '{source_file}'.",
        ]
        data_analyst.instructions = data_analyst.instructions + additional_instructions

    # Call the query service
    return query(data=data, data_dir=data_dir, data_analyst=data_analyst)


@api_bp.route("/test", methods=["GET"])
def test_connection():
    """Test endpoint to verify the connection between frontend and backend"""
    return jsonify(
        {"content": "Backend connection test successful", "status": "online"}
    )


@api_bp.route("/evaluate", methods=["POST"])
def evaluate_endpoint():
    data = request.get_json()
    model_id = data.get("model_id")

    if not model_id:
        return jsonify({"error": "Model ID is required"}), 400

    results, status_code = query_with_eval(model_id)
    return jsonify(results), status_code


@api_bp.route("/model-performance", methods=["GET"])
def model_performance():
    """Get aggregated model performance metrics."""
    try:
        model_type = request.args.get("type")

        with get_cursor() as cursor:
            query = """
            SELECT * FROM model_performance_metrics
            """

            params = []
            if model_type:
                query += " WHERE model_type = %s"
                params.append(model_type)

            query += " ORDER BY model_name"

            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            # Convert metrics to proper format for visualization
            for result in results:
                for key, value in result.items():
                    if key.startswith("avg_") and value is not None:
                        result[key] = float(value)

            return jsonify(
                {
                    "data": results,
                    "metrics": [
                        {
                            "id": "avg_factual_correctness",
                            "name": "Factual Correctness",
                        },
                        {
                            "id": "avg_semantic_similarity",
                            "name": "Semantic Similarity",
                        },
                        {"id": "avg_context_recall", "name": "Context Recall"},
                        {"id": "avg_faithfulness", "name": "Faithfulness"},
                        {"id": "avg_bleu_score", "name": "BLEU Score"},
                        {
                            "id": "avg_non_llm_string_similarity",
                            "name": "String Similarity",
                        },
                        {"id": "avg_rouge_score", "name": "ROUGE Score"},
                        {"id": "avg_string_present", "name": "String Present"},
                    ],
                }
            )

    except Exception as e:
        logger.error(f"Error fetching model performance: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/full-query-data", methods=["GET"])
def query_data():
    """Get full results for all evaluated queries."""
    try:

        with get_cursor() as cursor:
            query = """
            SELECT * FROM full_query_data
            """
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            return jsonify({"data": results})

    except Exception as e:
        logger.error(f"Error fetching model performance: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/test-cases", methods=["GET"])
def get_test_cases():
    try:
        # Load test cases from JSON file
        test_cases_path = Path("app/ragas/test_cases/synthetic_test_cases.json")
        with open(test_cases_path, "r") as f:
            test_cases = json.load(f)

        # Create an ordered list of test cases
        ordered_test_cases = []
        for test_case in test_cases:
            ordered_test_case = OrderedDict()
            ordered_test_case["user_input"] = test_case["user_input"]
            ordered_test_case["reference_contexts"] = test_case["reference_contexts"]
            ordered_test_case["reference"] = test_case["reference"]
            ordered_test_case["synthesizer_name"] = test_case["synthesizer_name"]
            ordered_test_cases.append(ordered_test_case)

        response_data = {"test_cases": ordered_test_cases}
        
        return Response(
            json.dumps(response_data, indent=2),
            mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Error fetching test cases: {e}")
        return jsonify({"error": str(e)}), 500