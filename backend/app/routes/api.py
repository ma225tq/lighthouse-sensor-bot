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
    number_of_runs = data.get("number_of_runs", 1)
    max_retries = data.get("max_retries", 3) 

    if not model_id:
        return jsonify({"error": "Model ID is required"}), 400

    results, status_code = query_with_eval(
        model_id, 
        number_of_runs=number_of_runs,
        max_retries=max_retries
    )
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
                        {"id": "avg_rogue_score", "name": "ROUGE Score"},
                        {"id": "avg_string_present", "name": "String Present"},
                    ],
                }
            )

    except Exception as e:
        logger.error(f"Error fetching model performance: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/export-chart", methods=["POST"])
def export_chart():
    """Generate a high-quality matplotlib chart from model performance data."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        import base64
        from matplotlib.figure import Figure
        
        # Get parameters from request
        data = request.get_json()
        chart_type = data.get('chartType', 'bar')
        metric_id = data.get('metricId', 'avg_factual_correctness')
        models = data.get('models', [])
        
        # If no data is provided, fetch all data
        if not models:
            # Use the same query as model_performance endpoint
            model_type = data.get('modelType')
            
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
                models = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                # Convert metrics to proper format for visualization
                for result in models:
                    for key, value in result.items():
                        if key.startswith("avg_") and value is not None:
                            result[key] = float(value)
        
        # Create a high-quality figure
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        # Create figure with better resolution
        fig = Figure(figsize=(12, 8), dpi=300)
        ax = fig.add_subplot(111)
        
        # Get metric name from ID
        metric_name_map = {
            "avg_factual_correctness": "Factual Correctness",
            "avg_semantic_similarity": "Semantic Similarity",
            "avg_context_recall": "Context Recall",
            "avg_faithfulness": "Faithfulness",
            "avg_bleu_score": "BLEU Score",
            "avg_non_llm_string_similarity": "String Similarity",
            "avg_rogue_score": "ROUGE Score",
            "avg_string_present": "String Present",
            "avg_prompt_tokens": "Prompt Tokens",
            "avg_completion_tokens": "Completion Tokens",
            "avg_total_tokens": "Total Tokens"
        }
        metric_name = metric_name_map.get(metric_id, metric_id)
        
        # Prepare data
        labels = [model['model_name'].split('/')[-1] for model in models]
        values = [model.get(metric_id, 0) for model in models]
        std_dev_id = metric_id.replace('avg_', 'stddev_')
        std_devs = [model.get(std_dev_id, 0) for model in models]
        
        # Set colors
        colors = [
            (31/255, 119/255, 180/255),
            (255/255, 127/255, 14/255),
            (44/255, 160/255, 44/255),
            (214/255, 39/255, 40/255),
            (148/255, 103/255, 189/255),
            (140/255, 86/255, 75/255),
            (227/255, 119/255, 194/255),
            (127/255, 127/255, 127/255),
            (188/255, 189/255, 34/255),
            (23/255, 190/255, 207/255)
        ]
        
        # For bar chart
        if chart_type == 'bar':
            bars = ax.bar(
                labels, 
                values, 
                color=[colors[i % len(colors)] for i in range(len(labels))],
                yerr=std_devs,
                capsize=5,
                alpha=0.8
            )
            
            # Add value labels on top of bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.02,
                    f"{value:.3f}",
                    ha='center', 
                    va='bottom',
                    fontweight='bold',
                    fontsize=10
                )
            
            # Set y-axis limit based on metric type
            is_token_metric = metric_id in ['avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens']
            if not is_token_metric:
                ax.set_ylim(0, 1.1)  # For score metrics
            
            # Add grid
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set title and labels
            ax.set_title(f"Model Performance: {metric_name}", fontweight='bold', pad=20)
            ax.set_xlabel("Models", labelpad=15)
            ax.set_ylabel("Score (0-1)" if not is_token_metric else "Average Tokens", labelpad=15)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
        # For radar chart
        elif chart_type == 'radar':
            # This requires more complex setup for matplotlib
            # Get all performance metrics
            performance_metrics = [
                {"id": "avg_factual_correctness", "name": "Factual Correctness"},
                {"id": "avg_semantic_similarity", "name": "Semantic Similarity"},
                {"id": "avg_context_recall", "name": "Context Recall"},
                {"id": "avg_faithfulness", "name": "Faithfulness"},
                {"id": "avg_bleu_score", "name": "BLEU Score"},
                {"id": "avg_non_llm_string_similarity", "name": "String Similarity"},
                {"id": "avg_rogue_score", "name": "ROUGE Score"},
                {"id": "avg_string_present", "name": "String Present"}
            ]
            
            # Get only performance metrics, no token metrics
            metrics = [m for m in performance_metrics 
                     if m["id"] not in ["avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens"]]
            
            # Set up radar chart
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            ax = fig.add_subplot(111, polar=True)
            
            # Plot each model
            for i, model in enumerate(models):
                values = [model.get(metric["id"], 0) for metric in metrics]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, 'o-', linewidth=2, color=colors[i % len(colors)], label=labels[i], alpha=0.8)
                ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
            
            # Set chart properties
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Set labels and add metric names
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([metric["name"] for metric in metrics])
            
            # Y-axis limits
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Add title
            ax.set_title("Model Performance Comparison (All Metrics)", fontweight='bold', pad=20)
        
        # Add a border and tight layout
        fig.tight_layout(pad=3.0)
        
        # Save figure to a bytes buffer and convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({
            "image": img_base64,
            "format": "png"
        })
    
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
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
            ordered_test_case["query"] = test_case["query"]
            ordered_test_case["reference_contexts"] = test_case["reference_contexts"]
            ordered_test_case["ground_truth"] = test_case["ground_truth"]
            ordered_test_case["synthesizer_name"] = test_case["synthesizer_name"]
            ordered_test_cases.append(ordered_test_case)

        response_data = {"test_cases": ordered_test_cases}

        return Response(
            json.dumps(response_data, indent=2), mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Error fetching test cases: {e}")
        return jsonify({"error": str(e)}), 500
