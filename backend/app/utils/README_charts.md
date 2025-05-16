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

Examples:

```bash
# Generate all chart types
python backend/app/utils/generate_charts.py

# Generate model comparison chart for a specific metric
python backend/app/utils/generate_charts.py --chart-type model-comparison --metric bleu_score

# Generate radar chart for a specific model
python backend/app/utils/generate_charts.py --chart-type model-metrics --model "qwen/qwen-2.5-72b-instruct"

# Compare two models side by side
python backend/app/utils/generate_charts.py --chart-type model-vs-model --model1 "qwen/qwen-2.5-72b-instruct" --model2 "openai/gpt-4o-2024-11-20"
```

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
   GET /api/charts/model-vs-model?model1=qwen/qwen-2.5-72b-instruct&model2=openai/gpt-4o-2024-11-20&metrics=metric1,metric2
   ```

6. **All Models All Metrics Chart**:
   ```
   GET /api/charts/all-models-all-metrics
   ```

### Programmatic Usage

You can also use the `ChartGenerator` class directly in your Python code:

```python
from app.utils.chart_generator import ChartGenerator

# Create a chart generator instance
chart_generator = ChartGenerator(output_dir="/path/to/output/dir")

# Generate a model comparison chart
chart_path = chart_generator.model_comparison_chart(
    metric_name="factual_correctness",
    model_names=["qwen/qwen-2.5-72b-instruct", "openai/gpt-4o-2024-11-20"]
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

All charts are saved as PNG files to the specified output directory. The default directory is `output/charts` in the project root.

## Metrics

The following metrics are available for visualization:

- `factual_correctness` - Measures the factual accuracy of the model's response
- `semantic_similarity` - Measures how semantically similar the response is to the ground truth
- `context_recall` - Measures how well the model retrieves relevant context
- `faithfulness` - Measures how faithful the model is to the retrieved context
- `bleu_score` - BLEU score for measuring translation quality
- `non_llm_string_similarity` - String similarity without using LLM
- `rogue_score` - ROUGE score for measuring summarization quality
- `string_present` - Checks if specific strings are present in the response 