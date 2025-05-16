#!/usr/bin/env python3
"""
Command-line utility to generate charts from evaluation results.
This script allows you to generate various visualization charts from the 
PostgreSQL database that contains model evaluation results.

Usage:
    python generate_charts.py [--chart-type TYPE] [--model MODEL] [--metric METRIC] [--output DIR]

Options:
    --chart-type    Type of chart to generate (default: all)
                    Options: model-comparison, model-metrics, metrics-heatmap, 
                    query-performance, model-vs-model, all-models-all-metrics, all
    --model         Model name to analyze (required for model-metrics and query-performance)
    --model1        First model for model-vs-model comparison
    --model2        Second model for model-vs-model comparison
    --metric        Metric to use (default: factual_correctness)
    --limit         Maximum number of queries for query-performance chart (default: 10)
    --output        Output directory (default: project_root/output/charts)
"""

import argparse
import sys
from pathlib import Path
import os
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
    parser = argparse.ArgumentParser(description="Generate charts from evaluation results")
    
    parser.add_argument(
        "--chart-type",
        choices=[
            "model-comparison", 
            "model-metrics", 
            "metrics-heatmap", 
            "query-performance", 
            "model-vs-model", 
            "all-models-all-metrics",
            "ragas-radar-chart",
            "all"
        ],
        default="all",
        help="Type of chart to generate"
    )
    
    parser.add_argument(
        "--model",
        help="Model name to analyze (required for model-metrics and query-performance)"
    )
    
    parser.add_argument(
        "--model1",
        help="First model for model-vs-model comparison"
    )
    
    parser.add_argument(
        "--model2",
        help="Second model for model-vs-model comparison"
    )
    
    parser.add_argument(
        "--metric",
        default="factual_correctness",
        help="Metric to use (default: factual_correctness)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of queries for query-performance chart"
    )
    
    parser.add_argument(
        "--output",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Create chart generator
    chart_generator = ChartGenerator(output_dir=args.output)
    
    # Generate requested charts
    if args.chart_type in ["model-comparison", "all"]:
        print("Generating model comparison chart...")
        path = chart_generator.model_comparison_chart(metric_name=args.metric)
        print(f"Chart saved: {path}")
    
    if args.chart_type in ["model-metrics", "all"]:
        if args.model or args.chart_type != "all":
            if not args.model:
                parser.error("--model is required for model-metrics chart")
            print(f"Generating metrics radar chart for {args.model}...")
            path = chart_generator.metric_comparison_chart(model_name=args.model)
            print(f"Chart saved: {path}")
    
    if args.chart_type in ["metrics-heatmap", "all"]:
        print("Generating metrics heatmap...")
        path = chart_generator.metrics_heatmap()
        print(f"Chart saved: {path}")
    
    if args.chart_type in ["query-performance", "all"]:
        if args.model or args.chart_type != "all":
            if not args.model:
                parser.error("--model is required for query-performance chart")
            print(f"Generating query performance chart for {args.model}...")
            path = chart_generator.query_performance_chart(
                model_name=args.model,
                metric_name=args.metric,
                limit=args.limit
            )
            print(f"Chart saved: {path}")
    
    if args.chart_type in ["model-vs-model", "all"]:
        if args.model1 and args.model2 or args.chart_type != "all":
            if not args.model1 or not args.model2:
                parser.error("Both --model1 and --model2 are required for model-vs-model chart")
            print(f"Generating comparison chart for {args.model1} vs {args.model2}...")
            path = chart_generator.model_vs_model_chart(
                model1=args.model1,
                model2=args.model2
            )
            print(f"Chart saved: {path}")
    
    if args.chart_type in ["all-models-all-metrics", "all"]:
        print("Generating comprehensive chart of all models and metrics...")
        path = chart_generator.all_models_all_metrics()
        print(f"Chart saved: {path}")
        
    if args.chart_type in ["ragas-radar-chart", "all"]:
        print("Generating optimized RAGAS radar chart with all models and metrics...")
        path = chart_generator.ragas_radar_chart()
        print(f"Chart saved: {path}")


if __name__ == "__main__":
    main() 