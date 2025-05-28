import pandas as pd
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Use non-interactive backend to prevent popups
matplotlib.use('Agg')
import os
from dotenv import load_dotenv
from scipy import stats

# Load environment variables
load_dotenv()

output_dir = "app/tests/test-output"
os.makedirs(output_dir, exist_ok=True)

# Use localhost for local development instead of postgres container name
db_host = os.environ.get("DB_HOST", "localhost")
if db_host == "postgres":
    db_host = "localhost"

conn = psycopg2.connect(
    dbname=os.environ.get("DB_NAME"),
    user=os.environ.get("DB_USER"),
    password=os.environ.get("DB_PASSWORD"),
    host=db_host,
    port=os.environ.get("DB_PORT")
)

# Current LLM pricing (per 1M tokens) - Based on December 2024 pricing
MODEL_PRICING = {
    "anthropic/claude-3.7-sonnet": {
        "input_cost_per_1m": 3.00,   # $3 per 1M input tokens
        "output_cost_per_1m": 15.00  # $15 per 1M output tokens
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "input_cost_per_1m": 0.20,   # $0.20 per 1M input tokens
        "output_cost_per_1m": 0.20   # $0.20 per 1M output tokens
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "input_cost_per_1m": 0.80,   # $0.80 per 1M input tokens
        "output_cost_per_1m": 0.80   # $0.80 per 1M output tokens
    },
    "mistralai/ministral-8b": {
        "input_cost_per_1m": 0.10,   # $0.10 per 1M input tokens
        "output_cost_per_1m": 0.10   # $0.10 per 1M output tokens
    },
    "qwen/qwen-2.5-72b-instruct": {
        "input_cost_per_1m": 0.50,   # $0.50 per 1M input tokens
        "output_cost_per_1m": 0.50   # $0.50 per 1M output tokens
    }
}

def calculate_query_cost(prompt_tokens, completion_tokens, model_name):
    """Calculate the cost for a single query based on token usage"""
    if model_name not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    
    # Convert tokens to millions and calculate cost
    input_cost = (prompt_tokens / 1_000_000) * pricing["input_cost_per_1m"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output_cost_per_1m"]
    
    return input_cost + output_cost

def run_friedman_test_cost(data_df):
    """Run Friedman test on the cost data"""
    # Check if we have enough data
    if data_df.shape[0] < 2 or data_df.shape[1] < 2:
        return {
            "error": "Not enough data for Friedman test",
            "rows": data_df.shape[0],
            "columns": data_df.shape[1]
        }
    
    # Extract data as a numpy array
    data = data_df.to_numpy()
    
    # Run Friedman test
    try:
        statistic, p_value = stats.friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'models': data_df.columns.tolist()
        }
    except Exception as e:
        return {
            "error": str(e),
            "rows": data_df.shape[0],
            "columns": data_df.shape[1]
        }

# Query to get token usage with model information
query = """
SELECT 
    lm.name AS model_name,
    qr.id AS query_id,
    qr.query AS query_text,
    tu.prompt_tokens,
    tu.completion_tokens,
    tu.total_tokens
FROM query_result qr
JOIN llm_models lm ON qr.llm_model_id = lm.id
JOIN token_usage tu ON qr.id = tu.query_result_id
WHERE tu.prompt_tokens IS NOT NULL 
  AND tu.completion_tokens IS NOT NULL
  AND tu.total_tokens IS NOT NULL
ORDER BY qr.query, lm.name
"""

# Load data into pandas DataFrame
df = pd.read_sql(query, conn)
conn.close()

# Calculate cost for each query
df['cost_usd'] = df.apply(
    lambda row: calculate_query_cost(
        row['prompt_tokens'], 
        row['completion_tokens'], 
        row['model_name']
    ), 
    axis=1
)

print(f"Total rows with token usage data: {len(df)}")
print(f"Unique queries by ID: {df['query_id'].nunique()}")
print(f"Unique queries by text: {df['query_text'].nunique()}")
print(f"Unique models: {df['model_name'].nunique()}")
print(f"Models in dataset: {df['model_name'].unique()}")

def create_model_aliases(models):
    """Create hardcoded aliases for model names"""
    model_name_map = {
        "anthropic/claude-3.7-sonnet": "Claude 3.7 Sonnet",
        "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B",
        "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
        "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
        "mistralai/ministral-8b": "Ministral 8B"
    }
    
    # Create aliases 
    aliases = {}
    for i, model in enumerate(models):
        if model in model_name_map:
            aliases[model] = model_name_map[model]
        else:
            # Fallback for any models not in our hardcoded list
            aliases[model] = f"Model {i+1}"
    
    return aliases

def prepare_cost_data(df):
    """Prepare cost data grouped by query and model"""
    print(f"\nAnalyzing cost data...")
    
    # Group by query text and model, getting average cost for each group
    grouped_df = df.groupby(['query_text', 'model_name'])['cost_usd'].mean().reset_index()
    
    # Pivot to get models as columns and queries as rows
    pivot_df = grouped_df.pivot_table(
        index='query_text', 
        columns='model_name',
        values='cost_usd'
    )
    
    print(f"Shape after pivot: {pivot_df.shape}")
    print(f"Missing values per model:")
    print(pivot_df.isna().sum())
    
    # Check before dropping NA values
    complete_rows = pivot_df.dropna()
    print(f"Shape after dropping incomplete rows: {complete_rows.shape}")
    
    return complete_rows

def create_cost_boxplot_chart(data_df, friedman_results):
    """Create an enhanced boxplot comparison of cost across models with Friedman test results"""
    
    # Define consistent order
    preferred_order = [
        "Claude 3.7 Sonnet", 
        "Llama 3.1 8B", 
        "Llama 3.3 70B", 
        "Ministral 8B", 
        "Qwen 2.5 72B"
    ]
    
    # Create aliases for model names
    model_aliases = create_model_aliases(data_df.columns)
    
    # Create reordered model list
    ordered_models = []
    for preferred in preferred_order:
        for model in data_df.columns:
            if preferred in model_aliases[model]:
                ordered_models.append(model)
                break
    
    # Add any remaining models
    for model in data_df.columns:
        if model not in ordered_models:
            ordered_models.append(model)
    
    plt.figure(figsize=(12, 6))
    
    # Create boxplot with ordered names
    plt.boxplot([data_df[col].values for col in ordered_models], 
               tick_labels=[model_aliases[col] for col in ordered_models])
    
    # Add statistical test information
    if 'error' not in friedman_results:
        stat = friedman_results['statistic']
        p_val = friedman_results['p_value']
        signif_text = "significant" if p_val < 0.05 else "not significant"
        title = f'Comparison of cost across models\n'
        title += f'Friedman χ² = {stat:.4f}, p = {p_val:.4f} ({signif_text})'
    else:
        title = f'Comparison of cost across models\n'
        title += f'Friedman test error: {friedman_results["error"]}'
    
    # Enhanced styling for title and labels
    plt.title(title, fontsize=20, fontweight='bold')
    plt.ylabel('Cost (USD)', fontsize=18, fontweight='bold')
    plt.xlabel('Model', fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=16, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    boxplot_path = os.path.join(output_dir, 'cost_boxplot.pdf')
    plt.savefig(boxplot_path)
    plt.close()
    
    print(f"Cost boxplot chart saved to: {boxplot_path}")

def create_cost_statistics_chart(data_df):
    """Create a bar chart showing cost statistics for each model"""
    
    # Define consistent order
    preferred_order = [
        "Claude 3.7 Sonnet", 
        "Llama 3.1 8B", 
        "Llama 3.3 70B", 
        "Ministral 8B", 
        "Qwen 2.5 72B"
    ]
    
    # Create aliases for model names
    model_aliases = create_model_aliases(data_df.columns)
    
    # Create reordered model list
    ordered_models = []
    for preferred in preferred_order:
        for model in data_df.columns:
            if preferred in model_aliases[model]:
                ordered_models.append(model)
                break
    
    # Add any remaining models
    for model in data_df.columns:
        if model not in ordered_models:
            ordered_models.append(model)
    
    # Extract ordered data
    ordered_data = []
    for model in ordered_models:
        values = data_df[model].values
        ordered_data.append({
            'model': model,
            'alias': model_aliases[model],
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values)
        })
    
    ordered_aliases = [item['alias'] for item in ordered_data]
    ordered_means = [item['mean'] for item in ordered_data]
    ordered_medians = [item['median'] for item in ordered_data]
    ordered_stds = [item['std'] for item in ordered_data]
    
    # Create plot
    x = np.arange(len(ordered_models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, ordered_means, width, label='Mean', yerr=ordered_stds, 
             capsize=5, color='skyblue', edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add median markers
    ax.scatter(x, ordered_medians, color='red', marker='D', s=50, label='Median')
    
    # Calculate the maximum height for proper spacing
    max_height = max(ordered_means)
    max_std = max(ordered_stds)
    top_space = max_height + max_std
    
    # Set y-axis limits with extra space at the top for labels
    ax.set_ylim(0, top_space * 1.4)  # Add 40% extra space at the top
    
    # Customize plot with enhanced font sizes
    ax.set_xlabel('Model', fontsize=18, fontweight='bold')
    ax.set_ylabel('Cost (USD)', fontsize=18, fontweight='bold')
    ax.set_title('Cost Statistics by Model', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_aliases, rotation=45, ha='right', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=14, prop={'weight': 'bold'})
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Enhanced Y-axis tick labels
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Add cost values above bars with enhanced styling and better positioning
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Position text higher above the error bars
        text_y = height + ordered_stds[i] + (top_space * 0.05)  # Add 5% of top space as buffer
        ax.text(bar.get_x() + bar.get_width()/2., text_y,
                f'Mean: ${ordered_means[i]:.3f}\nMedian: ${ordered_medians[i]:.3f}\nStd: ${ordered_stds[i]:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Add extra space at the top
    
    stats_path = os.path.join(output_dir, 'cost_statistics.pdf')
    plt.savefig(stats_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"Cost statistics chart saved to: {stats_path}")

def print_cost_summary(data_df):
    """Print cost summary statistics"""
    print(f"\n--- COST STATISTICS BY MODEL ---")
    
    # Create aliases for model names
    model_aliases = create_model_aliases(data_df.columns)
    
    stats_table = []
    for model in data_df.columns:
        values = data_df[model].values
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        total_cost = np.sum(values)
        
        model_alias = model_aliases[model]
        stats_table.append([
            model_alias,
            f"${mean_val:.4f}",
            f"${median_val:.4f}",
            f"${std_val:.4f}",
            f"${total_cost:.4f}"
        ])
    
    # Print as a formatted table
    headers = ["Model", "Mean Cost", "Median Cost", "Std Dev", "Total Cost"]
    
    # Calculate column widths
    col_widths = [max(len(headers[i]), max(len(row[i]) for row in stats_table)) for i in range(len(headers))]
    
    # Print header
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for row in stats_table:
        print(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)))
    
    # Save to file
    with open(os.path.join(output_dir, 'cost_statistics.txt'), 'w') as f:
        f.write("COST STATISTICS BY MODEL\n")
        f.write(header_row + "\n")
        f.write("-" * len(header_row) + "\n")
        for row in stats_table:
            f.write(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + "\n")

# Main execution
if len(df) > 0:
    print(f"\n{'='*50}")
    print(f"ANALYZING COST DATA")
    print(f"{'='*50}")
    
    # Prepare data
    cost_data = prepare_cost_data(df)
    
    if len(cost_data) > 0:
        # Print cost summary
        print_cost_summary(cost_data)
        
        # Run Friedman test on cost data
        friedman_results = run_friedman_test_cost(cost_data)
        
        if 'error' not in friedman_results:
            print(f"\nFriedman test statistic: {friedman_results['statistic']:.4f}")
            print(f"p-value: {friedman_results['p_value']:.4f}")
            print(f"Models compared: {friedman_results['models']}")
            
            if friedman_results['p_value'] < 0.05:
                print("Friedman test is significant! There are significant cost differences between models.")
            else:
                print("Friedman test is NOT significant. No significant cost differences between models.")
        else:
            print(f"Error running Friedman test: {friedman_results['error']}")
        
        # Create cost statistics chart (bar chart)
        create_cost_statistics_chart(cost_data)
        
        # Create cost boxplot chart
        create_cost_boxplot_chart(cost_data, friedman_results)
        
        # Calculate total costs
        total_cost_by_model = {}
        model_aliases = create_model_aliases(df['model_name'].unique())
        
        for model_name in df['model_name'].unique():
            model_df = df[df['model_name'] == model_name]
            total_cost = model_df['cost_usd'].sum()
            total_cost_by_model[model_aliases[model_name]] = total_cost
        
        print(f"\n--- TOTAL COSTS BY MODEL ---")
        for model, cost in sorted(total_cost_by_model.items(), key=lambda x: x[1], reverse=True):
            print(f"{model}: ${cost:.4f}")
        
        print(f"\nTotal cost across all models: ${sum(total_cost_by_model.values()):.4f}")
        
    else:
        print("Not enough cost data after filtering")
else:
    print("No token usage data available for cost analysis") 