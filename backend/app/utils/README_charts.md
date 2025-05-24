# Chart Generator for Model Evaluation Results

This utility generates visualizations from model evaluation results stored in PostgreSQL. These visualizations can be used for thesis writing, presentations, or analysis.

## Features

The chart generator supports the following chart types:

1. **Model Comparison Chart** - Bar chart comparing different models on a specific metric
2. **Model Metrics Chart** - Radar chart showing all metrics for a specific model
3. **Metrics Heatmap** - Heatmap comparing multiple models across multiple metrics
4. **Query Performance Chart** - Bar chart showing performance on individual queries for a specific model
5. **Model vs Model Chart** - Side-by-side bar chart comparing two models across multiple metrics
6. **All Models All Metrics** - Comprehensive grouped bar chart showing all models and all metrics
7. **RAGAS Radar Chart** - Optimized radar chart showing all models across all RAGAS metrics
8. **Factual Correctness Matrix** - Comparison matrix showing individual question scores for each model

## Usage

### Command Line Tool

You can generate charts directly using the command-line tool:

```bash
cd /path/to/lighthouse-sensor-bot
python backend/app/utils/generate_charts.py [options]
```

Options:
- `--chart-type`: Type of chart to generate (default: all)
  - Options: model-comparison, model-metrics, metrics-heatmap, query-performance, model-vs-model, all-models-all-metrics, all
- `--model`: Model name to analyze (required for model-metrics and query-performance)
- `--model1`: First model for model-vs-model comparison
- `--model2`: Second model for model-vs-model comparison  
- `--metric`: Metric to use (default: factual_correctness)
- `--limit`: Maximum number of queries for query-performance chart (default: 10)
- `--output`: Output directory (default: project_root/output/charts)
- `--png`: Generate PNG files in addition to PDF files (by default only PDF files are generated)

Examples:

```bash
# 1. Radar Chart (average metric values across all questions)
python backend/app/utils/generate_charts.py --chart-type ragas-radar-chart

# 2. Comprehensive Charts (all models and metrics except individual models)
# Comprehensive bar chart of all models and metrics
python backend/app/utils/generate_charts.py --chart-type all-models-all-metrics

# Heatmap showing average values for all metrics across all questions
python backend/app/utils/generate_charts.py --chart-type metrics-heatmap

# 3. Model Comparison with RAGAS Metrics (by individual metric)
# Factual Correctness
python backend/app/utils/generate_charts.py --chart-type model-comparison --metric factual_correctness

# Semantic Similarity
python backend/app/utils/generate_charts.py --chart-type model-comparison --metric semantic_similarity

# Faithfulness
python backend/app/utils/generate_charts.py --chart-type model-comparison --metric faithfulness

# BLEU Score
python backend/app/utils/generate_charts.py --chart-type model-comparison --metric bleu_score

# ROUGE Score
python backend/app/utils/generate_charts.py --chart-type model-comparison --metric rogue_score

# Non LLM String Similarity
python backend/app/utils/generate_charts.py --chart-type model-comparison --metric non_llm_string_similarity

# 4. Claude 3.7 Sonnet vs Open Source Models (side-by-side comparisons)
# Claude vs Llama 3.3 70B
python backend/app/utils/generate_charts.py --chart-type model-vs-model --model1 "anthropic/claude-3.7-sonnet" --model2 "meta-llama/llama-3.3-70b-instruct"

# Claude vs Llama 3.1 8B
python backend/app/utils/generate_charts.py --chart-type model-vs-model --model1 "anthropic/claude-3.7-sonnet" --model2 "meta-llama/llama-3.1-8b-instruct"

# Claude vs Mistral 8B
python backend/app/utils/generate_charts.py --chart-type model-vs-model --model1 "anthropic/claude-3.7-sonnet" --model2 "mistralai/ministral-8b"

# Claude vs Qwen 2.5 72B
python backend/app/utils/generate_charts.py --chart-type model-vs-model --model1 "anthropic/claude-3.7-sonnet" --model2 "qwen/qwen-2.5-72b-instruct"

# 5. Factual Correctness Matrix (individual question scores across models)
python backend/app/utils/generate_charts.py --chart-type factual-correctness-matrix

# 6. Generate PNG files in addition to PDF (optional)
python backend/app/utils/generate_charts.py --chart-type all-models-all-metrics --png
```

All generated charts will be saved to the `output/charts` directory in the project root.

### API Endpoints

The chart generator is also exposed via REST API endpoints:

1. **Model Comparison Chart**:
   ```
   GET /api/charts/model-comparison?metric=factual_correctness&models=model1,model2
   ```

2. **Model Metrics Chart**:
   ```
   GET /api/charts/model-metrics?model=qwen/qwen-2.5-72b-instruct
   ```

3. **Metrics Heatmap**:
   ```
   GET /api/charts/metrics-heatmap?models=model1,model2&metrics=metric1,metric2
   ```

4. **Query Performance Chart**:
   ```
   GET /api/charts/query-performance?model=qwen/qwen-2.5-72b-instruct&metric=factual_correctness&limit=10
   ```

5. **Model vs Model Chart**:
   ```
   GET /api/charts/model-vs-model?model1=anthropic/claude-3.7-sonnet&model2=meta-llama/llama-3.3-70b-instruct&metrics=metric1,metric2
   ```

6. **All Models All Metrics Chart**:
   ```
   GET /api/charts/all-models-all-metrics
   ```

### Programmatic Usage

You can also use the `ChartGenerator` class directly in your Python code:

```python
from app.utils.chart_generator import ChartGenerator

# Create a chart generator instance (PDF-only by default)
chart_generator = ChartGenerator(output_dir="/path/to/output/dir")

# Or create with PNG files as well
chart_generator = ChartGenerator(output_dir="/path/to/output/dir", pdf_only=False)

# Generate a model comparison chart
chart_path = chart_generator.model_comparison_chart(
    metric_name="factual_correctness",
    model_names=["qwen/qwen-2.5-72b-instruct", "anthropic/claude-3.7-sonnet","]
)

# Generate a radar chart for a specific model. Radar charts for visualizing all metrics for a single model
chart_path = chart_generator.metric_comparison_chart(
    model_name="qwen/qwen-2.5-72b-instruct"
)

# Generate a heatmap. Heatmaps for comparing multiple models across multiple metrics
chart_path = chart_generator.metrics_heatmap()

# Generate a comprehensive chart of all models and metrics
chart_path = chart_generator.all_models_all_metrics()
```

## Output

All charts are saved as PDF files by default to the specified output directory. The default directory is `output/charts` in the project root. PNG files can be generated in addition by using the `--png` flag or setting `pdf_only=False` in the ChartGenerator constructor.

## Metrics

The following metrics are available for visualization:

- `factual_correctness` - Measures the factual accuracy of the model's response
- `semantic_similarity` - Measures how semantically similar the response is to the ground truth
- `faithfulness` - Measures how faithful the model is to the retrieved context
- `bleu_score` - BLEU score for measuring translation quality
- `non_llm_string_similarity` - String similarity without using LLM
- `rogue_score` - ROUGE score (Recall-Oriented Understudy for Gisting Evaluation) for measuring summarization quality