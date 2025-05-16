#!/usr/bin/env python3
"""
Examples of generating different types of charts using the ChartGenerator class.
This script demonstrates how to use the ChartGenerator programmatically.

Run this script from the project root directory:
    python backend/app/utils/chart_examples.py
"""

import sys
from pathlib import Path
import dotenv

# Ensure we can import from parent directory
script_dir = Path(__file__).resolve().parent
backend_app_dir = script_dir.parent
backend_dir = backend_app_dir.parent
sys.path.append(str(backend_dir))

# Load environment variables
dotenv.load_dotenv(backend_dir / ".env")

# Import after path setup
from app.utils.chart_generator import ChartGenerator


def main():
    """Generate example charts using ChartGenerator"""
    # Create output directory in the examples folder
    output_dir = backend_dir / "output" / "chart_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create chart generator instance
    chart_generator = ChartGenerator(output_dir=str(output_dir))

    print(f"Generating example charts in {output_dir}")
    print("=" * 50)

    # Example 1: Generate model comparison chart for factual correctness
    print("\n1. Model Comparison Chart (Factual Correctness)")
    chart_path = chart_generator.model_comparison_chart(
        metric_name="factual_correctness"
    )
    print(f"Chart saved: {chart_path}")

    # Example 2: Generate model comparison chart for BLEU score
    print("\n2. Model Comparison Chart (BLEU Score)")
    chart_path = chart_generator.model_comparison_chart(
        metric_name="bleu_score"
    )
    print(f"Chart saved: {chart_path}")

    # Example 3: Generate radar chart for a specific model
    model_name = "qwen/qwen-2.5-72b-instruct"  
    print(f"\n3. Model Metrics Radar Chart for {model_name}")
    try:
        chart_path = chart_generator.metric_comparison_chart(
            model_name=model_name
        )
        print(f"Chart saved: {chart_path}")
    except Exception as e:
        print(f"Error generating radar chart: {e}")
        print("Please check if the model name exists in your database")

    # Example 4: Generate model vs model comparison
    model1 = "anthropic/claude-3.7-sonnet"
    model2 = "qwen/qwen-2.5-72b-instruct"
    print(f"\n4. Model vs Model Comparison: {model1} vs {model2}")
    chart_path = chart_generator.model_vs_model_chart(
        model1=model1, 
        model2=model2
    )
    print(f"Chart saved: {chart_path}")

    # Example 5: Generate factual correctness matrix chart
    print("\n5. Factual Correctness Matrix Chart")
    chart_path = chart_generator.factual_correctness_matrix(
        max_questions=8,
        limit_models=6
    )
    print(f"Chart saved: {chart_path}")

    # Example 6: Generate heatmap for all models and metrics
    print("\n6. Metrics Heatmap")
    chart_path = chart_generator.metrics_heatmap()
    print(f"Chart saved: {chart_path}")

    # Example 7: Generate query performance chart for a specific model
    print(f"\n7. Query Performance Chart for {model_name}")
    try:
        chart_path = chart_generator.query_performance_chart(
            model_name=model_name,
            metric_name="factual_correctness",
            limit=5
        )
        print(f"Chart saved: {chart_path}")
    except Exception as e:
        print(f"Error generating query performance chart: {e}")
        print("Please check if the model name exists in your database")

    # Example 8: Generate model vs model chart
    # Note: Replace these with actual model names from your database
    model1 = "qwen/qwen-2.5-72b-instruct"
    model2 = "openai/gpt-4o-2024-11-20"
    print(f"\n8. Model vs Model Chart: {model1} vs {model2}")
    try:
        chart_path = chart_generator.model_vs_model_chart(
            model1=model1,
            model2=model2
        )
        print(f"Chart saved: {chart_path}")
    except Exception as e:
        print(f"Error generating model vs model chart: {e}")
        print("Please check if both model names exist in your database")

    # Example 9: Generate comprehensive chart of all models and metrics
    print("\n9. All Models All Metrics Chart")
    chart_path = chart_generator.all_models_all_metrics()
    print(f"Chart saved: {chart_path}")

    print("\n" + "=" * 50)
    print(f"All charts have been saved to {output_dir}")


if __name__ == "__main__":
    main() 