import pandas as pd
import psycopg2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
# Use non-interactive backend to prevent popups
matplotlib.use('Agg')
from scikit_posthocs import posthoc_nemenyi_friedman
import os
from dotenv import load_dotenv
import itertools
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
import scikit_posthocs as sp

# Load environment variables
load_dotenv()

output_dir = "app/tests/test-output"
os.makedirs(output_dir, exist_ok=True)

conn = psycopg2.connect(
    dbname=os.environ.get("DB_NAME"),
    user=os.environ.get("DB_USER"),
    password=os.environ.get("DB_PASSWORD"),
    host=os.environ.get("DB_HOST"),
    port=os.environ.get("DB_PORT")
)

# Query to get evaluation metrics with model information and actual query text
query = """
SELECT 
    lm.name AS model_name,
    qr.id AS query_id,
    qr.query AS query_text,
    em.factual_correctness,
    em.semantic_similarity,
    em.faithfulness,
    em.bleu_score,
    em.non_llm_string_similarity,
    em.rogue_score,
    em.string_present,
    tu.prompt_tokens,
    tu.completion_tokens,
    tu.total_tokens
FROM query_result qr
JOIN llm_models lm ON qr.llm_model_id = lm.id
JOIN query_evaluation qe ON qr.id = qe.query_result_id
JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
LEFT JOIN token_usage tu ON qr.id = tu.query_result_id
ORDER BY qr.query, lm.name
"""

# Load data into pandas DataFrame
df = pd.read_sql(query, conn)
conn.close()

# database typo, cant be bothered to fix
df.rename(columns={'rogue_score': 'rouge_score'}, inplace=True)

def calculate_cost(row):
    """Calculate cost based on model pricing and token usage"""
    model_pricing = {
        "anthropic/claude-3.7-sonnet": {"input": 3.00, "output": 15.00},
        "qwen/qwen2.5-vl-72b-instruct": {"input": 0.25, "output": 0.75},
        "meta-llama/llama-3.3-70b-instruct": {"input": 0.07, "output": 0.25},
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.02, "output": 0.03},
        "mistralai/ministral-8b": {"input": 0.10, "output": 0.10}
    }
    
    alternative_models = {
        "qwen/qwen-2.5-72b-instruct": "qwen/qwen2.5-vl-72b-instruct",
        "qwen2.5-vl-72b-instruct": "qwen/qwen2.5-vl-72b-instruct"
    }
    
    model_name = row['model_name']
    
    if model_name in alternative_models:
        model_name = alternative_models[model_name]
    
    prompt_tokens = row['prompt_tokens']
    completion_tokens = row['completion_tokens']
    
    # if "llama" in model_name.lower():
    #     print(f"DEBUG - {model_name}: prompt={prompt_tokens}, completion={completion_tokens}")
    
    # Return NaN if missing token data or unknown model
    if pd.isna(prompt_tokens) or pd.isna(completion_tokens) or model_name not in model_pricing:
        print(f"Warning: Missing data or unknown model for {row['model_name']}")
        return np.nan
    
    pricing = model_pricing[model_name]
    
    # Calculate cost: (tokens / 1,000,000) * price_per_million
    cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
    
    if cost < 0.01 or "llama" in model_name.lower():
        print(f"Cost for {model_name}: ${cost:.6f} = ({prompt_tokens} × ${pricing['input']}/M + {completion_tokens} × ${pricing['output']}/M) ÷ 1M")
    
    return cost

def print_cost_legend():
    """Print model pricing information"""
    model_pricing = {
        "anthropic/claude-3.7-sonnet": {"input": 3.00, "output": 15.00, "alias": "Claude 3.7 Sonnet"},
        "qwen/qwen2.5-vl-72b-instruct": {"input": 0.25, "output": 0.75, "alias": "Qwen 2.5 72B"},
        "meta-llama/llama-3.3-70b-instruct": {"input": 0.07, "output": 0.25, "alias": "Llama 3.3 70B"},
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.02, "output": 0.03, "alias": "Llama 3.1 8B"},
        "mistralai/ministral-8b": {"input": 0.10, "output": 0.10, "alias": "Ministral 8B"}
    }
    
    print("\n" + "="*60)
    print("MODEL PRICING (per million tokens)")
    print("="*60)
    
    print(f"{'Model':<20} {'Input':<12} {'Output':<12}")
    print("-" * 45)
    
    for model, pricing in model_pricing.items():
        print(f"{pricing['alias']:<20} ${pricing['input']:<11.2f} ${pricing['output']:<11.2f}")
    
    # Save to file
    with open(os.path.join(output_dir, 'model_pricing_legend.txt'), 'w') as f:
        f.write("MODEL PRICING (per million tokens)\n")
        f.write("="*60 + "\n")
        f.write(f"{'Model':<20} {'Input':<12} {'Output':<12}\n")
        f.write("-" * 45 + "\n")
        for model, pricing in model_pricing.items():
            f.write(f"{pricing['alias']:<20} ${pricing['input']:<11.2f} ${pricing['output']:<11.2f}\n")

# Calculate cost for each row
df['cost'] = df.apply(calculate_cost, axis=1)

# Print cost legend
print_cost_legend()

# Print diagnostic information
print(f"Total rows in original dataset: {len(df)}")
print(f"Unique queries by ID: {df['query_id'].nunique()}")
print(f"Unique queries by text: {df['query_text'].nunique()}")
print(f"Unique models: {df['model_name'].nunique()}")
print(f"Models in dataset: {df['model_name'].unique()}")

print(f"Rows with cost data: {df['cost'].notna().sum()}")

print("\n==== TOKEN USAGE STATISTICS ====")
for model in df['model_name'].unique():
    model_data = df[df['model_name'] == model]
    print(f"\nModel: {model}")
    print(f"  Count: {len(model_data)}")
    print(f"  Prompt tokens: mean={model_data['prompt_tokens'].mean():.1f}, median={model_data['prompt_tokens'].median()}")
    print(f"  Completion tokens: mean={model_data['completion_tokens'].mean():.1f}, median={model_data['completion_tokens'].median()}")
    print(f"  Cost: mean=${model_data['cost'].mean():.4f}, median=${model_data['cost'].median():.4f}")

# Group by actual query text and model name
def prepare_data_for_metric(df, metric_name):
    print(f"\nAnalyzing metric: {metric_name}")
    
    # Group by query text and model, getting average metric value for each group
    grouped_df = df.groupby(['query_text', 'model_name'])[metric_name].mean().reset_index()
    
    # Pivot to get models as columns and queries as rows
    pivot_df = grouped_df.pivot_table(
        index='query_text', 
        columns='model_name',
        values=metric_name
    )
    
    print(f"Shape after pivot: {pivot_df.shape}")
    print(f"Missing values per model:")
    print(pivot_df.isna().sum())
    
    # Check before dropping NA values
    complete_rows = pivot_df.dropna()
    print(f"Shape after dropping incomplete rows: {complete_rows.shape}")
    
    # Return the filtered pivot table
    return complete_rows

# Run Friedman test on the properly grouped data
def run_friedman_test(data_df):
    """Run Friedman test on the properly grouped data"""
  
    print("\n====== ACTUAL VALUES USED FOR STATISTICAL TESTS ======")
    for col in data_df.columns:
        values = data_df[col].values
        print(f"\n{col}:")
        print(f"  Min: {np.min(values):.8f}")
        print(f"  Mean: {np.mean(values):.8f}")
        print(f"  Median: {np.median(values):.8f}")
        print(f"  Max: {np.max(values):.8f}")
        print(f"  First 3 values: {values[:3]}")
    print("======================================================\n")
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

# Run pairwise Wilcoxon tests if Friedman test is significant
def run_pairwise_wilcoxon(data_df):
    """
    Run pairwise Wilcoxon signed-rank tests using scikit_posthocs
    which provides better p-value adjustment methods
    """
    # Reshape data for scikit_posthocs
    # scikit_posthocs.posthoc_wilcoxon expects long-format data
    long_format_data = []
    models = data_df.columns
    
    # Create a list of model names and test values for posthoc_wilcoxon
    for query_idx in range(len(data_df)):
        for model in models:
            # Skip NaN values
            if pd.notna(data_df[model].iloc[query_idx]):
                long_format_data.append({
                    'model': model,
                    'score': data_df[model].iloc[query_idx]
                })
    
    # Convert to dataframe
    long_df = pd.DataFrame(long_format_data)
    
    print(f"\nRunning Wilcoxon tests with Bonferroni correction...")
    
    try:
        # Run the pairwise Wilcoxon tests with Bonferroni correction
        result_matrix = sp.posthoc_wilcoxon(
            long_df, 
            val_col='score',
            group_col='model',
            p_adjust='bonferroni',
            zero_method='wilcox'
        )
        
        # Convert matrix format to the list format expected by the plotting code
        results = []
        for i, model1 in enumerate(result_matrix.index):
            for j, model2 in enumerate(result_matrix.columns):
                if i < j:  # Only look at upper triangle to avoid duplicates
                    p_value = result_matrix.iloc[i, j]
                    
                    results.append({
                        'model1': model1,
                        'model2': model2,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'correction': 'bonferroni'
                    })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        print(f"Error running pairwise Wilcoxon tests: {e}")
        return pd.DataFrame({
            'error': [str(e)]
        })

# Analyze all metrics including cost (removed individual token metrics)
metrics = [
    'factual_correctness', 
    'semantic_similarity',
    'faithfulness',
    'bleu_score',
    'non_llm_string_similarity',
    'rouge_score',
    'string_present',
    'cost'
]

results_by_metric = {}

def create_model_aliases(models):
    """Create hardcoded aliases for model names"""
    model_name_map = {
        "anthropic/claude-3.7-sonnet": "Claude 3.7 Sonnet",
        "qwen/qwen2.5-vl-72b-instruct": "Qwen 2.5 72B",
        "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B",
        "qwen2.5-vl-72b-instruct": "Qwen 2.5 72B",
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
            print(f"Warning: Unknown model name: {model}")
    
    return aliases

def plot_and_save_metric_comparison(data_df, metric_name, friedman_results, posthoc_results=None, wilcoxon_results=None):
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
    
    # 1. Create metric statistics graph
    create_metric_stats_graph(data_df, metric_name, ordered_models, model_aliases)
    
    # 2. Create boxplot comparison graph
    create_boxplot_graph(data_df, metric_name, friedman_results, ordered_models, model_aliases)
    
    # 3. Create post-hoc tests graph if applicable
    if friedman_results.get('p_value', 1.0) < 0.05 and (posthoc_results is not None or wilcoxon_results is not None):
        create_posthoc_graph(data_df, metric_name, posthoc_results, wilcoxon_results, model_aliases)

def create_metric_stats_graph(data_df, metric_name, ordered_models, model_aliases):
    """Create a bar chart showing mean/median/std for each model"""
    # Extract ordered data
    ordered_data = []
    for model in ordered_models:
        ordered_data.append({
            'model': model,
            'alias': model_aliases[model],
            'mean': np.mean(data_df[model].values),
            'median': np.median(data_df[model].values),
            'std': np.std(data_df[model].values)
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
    
    # Customize plot
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(f'{metric_name.replace("_", " ").title()} Statistics by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_aliases, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add mean values above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Mean: {ordered_means[i]:.3f}\nMedian: {ordered_medians[i]:.3f}\nStd: {ordered_stds[i]:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    stats_path = os.path.join(output_dir, f'{metric_name}_statistics.pdf')
    plt.savefig(stats_path)
    plt.close()

def create_boxplot_graph(data_df, metric_name, friedman_results, ordered_models, model_aliases):
    """Create a boxplot comparison of models with Friedman test results"""
    plt.figure(figsize=(12, 6))
    
    # Create boxplot with ordered names
    plt.boxplot([data_df[col].values for col in ordered_models], 
               tick_labels=[model_aliases[col] for col in ordered_models])
    
    # Add statistical test information
    if 'error' not in friedman_results:
        stat = friedman_results['statistic']
        p_val = friedman_results['p_value']
        signif_text = "significant" if p_val < 0.05 else "not significant"
        title = f'Comparison of {metric_name} across models\n'
        title += f'Friedman χ² = {stat:.4f}, p = {p_val:.4f} ({signif_text})'
    else:
        title = f'Comparison of {metric_name} across models\n'
        title += f'Friedman test error: {friedman_results["error"]}'
    
    plt.title(title)
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.xlabel('Model')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    boxplot_path = os.path.join(output_dir, f'{metric_name}_boxplot.pdf')
    plt.savefig(boxplot_path)
    plt.close()

def create_combined_posthoc_table(posthoc_results, wilcoxon_results, model_aliases):
    """
    Create a combined table showing both Nemenyi and Wilcoxon test results
    """
    # Extract pairwise comparisons from Nemenyi matrix
    combined_results = []
    
    if posthoc_results is not None:
        models = posthoc_results.columns
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:  # Only upper triangle to avoid duplicates
                    nemenyi_p = posthoc_results.iloc[i, j]
                    nemenyi_sig = "✓" if nemenyi_p < 0.05 else ""
                    
                    # Find matching Wilcoxon result
                    wilcoxon_p = None
                    wilcoxon_sig = ""
                    
                    if wilcoxon_results is not None and not wilcoxon_results.empty and 'error' not in wilcoxon_results.columns:
                        # Find the matching pair in wilcoxon results
                        mask = ((wilcoxon_results['model1'] == model1) & (wilcoxon_results['model2'] == model2)) | \
                               ((wilcoxon_results['model1'] == model2) & (wilcoxon_results['model2'] == model1))
                        
                        if mask.any():
                            wilcoxon_row = wilcoxon_results[mask].iloc[0]
                            wilcoxon_p = wilcoxon_row['p_value']
                            wilcoxon_sig = "✓" if wilcoxon_row.get('significant', False) else ""
                    
                    combined_results.append({
                        'model1': model_aliases[model1],
                        'model2': model_aliases[model2],
                        'nemenyi_p': nemenyi_p,
                        'nemenyi_sig': nemenyi_sig,
                        'wilcoxon_p': wilcoxon_p,
                        'wilcoxon_sig': wilcoxon_sig
                    })
    
    return combined_results

def print_combined_posthoc_table(combined_results):
    """
    Print the combined post-hoc test results as a formatted table
    """
    if not combined_results:
        print("No post-hoc test results available")
        return
    
    print("\n--- PAIRWISE POST-HOC TEST RESULTS ---")
    
    # Create table headers
    headers = ["Model Pair", "Nemenyi p-value", "Nemenyi Sig", "Wilcoxon p-value", "Wilcoxon Sig"]
    
    # Prepare data rows
    table_data = []
    for result in combined_results:
        model_pair = f"{result['model1']} vs {result['model2']}"
        nemenyi_p = f"{result['nemenyi_p']:.4f}" if result['nemenyi_p'] is not None else "N/A"
        nemenyi_sig = result['nemenyi_sig']
        wilcoxon_p = f"{result['wilcoxon_p']:.4f}" if result['wilcoxon_p'] is not None else "N/A"
        wilcoxon_sig = result['wilcoxon_sig']
        
        table_data.append([model_pair, nemenyi_p, nemenyi_sig, wilcoxon_p, wilcoxon_sig])
    
    # Calculate column widths
    col_widths = [max(len(headers[i]), max(len(row[i]) for row in table_data)) for i in range(len(headers))]
    
    # Print header
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for row in table_data:
        print(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)))

def create_posthoc_graph(data_df, metric_name, posthoc_results, wilcoxon_results, model_aliases):
    """Create visualizations for post-hoc tests"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create combined table instead of separate visualizations
    combined_results = create_combined_posthoc_table(posthoc_results, wilcoxon_results, model_aliases)
    
    if combined_results:
        # ax.set_title("Pairwise Post-hoc Test Results", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Prepare table data for matplotlib
        table_data = []
        for result in combined_results:
            model_pair = f"{result['model1']} vs\n{result['model2']}"
            nemenyi_p = f"{result['nemenyi_p']:.4f}" if result['nemenyi_p'] is not None else "N/A"
            nemenyi_sig = result['nemenyi_sig']
            wilcoxon_p = f"{result['wilcoxon_p']:.4f}" if result['wilcoxon_p'] is not None else "N/A"
            wilcoxon_sig = result['wilcoxon_sig']
            
            table_data.append([model_pair, nemenyi_p, nemenyi_sig, wilcoxon_p, wilcoxon_sig])
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=['Model Pair', 'Nemenyi\np-value', 'Nemenyi\nSig (p<0.05)', 'Wilcoxon\np-value', 'Wilcoxon\nSig (p<0.05)'],
            loc='center',
            cellLoc='center',
            colWidths=[0.3, 0.15, 0.15, 0.15, 0.15]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Color code cells based on significance
        for i in range(len(table_data)):
            # Nemenyi significance
            if table_data[i][2] == "✓":  # Nemenyi significant
                table[(i+1, 2)].set_facecolor('#ccffcc')  
            
            # Wilcoxon significance  
            if table_data[i][4] == "✓":  # Wilcoxon significant
                table[(i+1, 4)].set_facecolor('#ccffcc')
        
        # Style header
        for j in range(5):
            table[(0, j)].set_facecolor('#e6f2ff') 
            table[(0, j)].set_text_props(weight='bold')
    
    plt.tight_layout()
    posthoc_path = os.path.join(output_dir, f'{metric_name}_posthoc_tests.pdf')
    plt.savefig(posthoc_path, bbox_inches='tight')
    plt.close()

def print_metric_stats(data_df, metric_name):
    """Print descriptive statistics for each model for a given metric"""
    print(f"\n--- {metric_name.upper()} STATISTICS BY MODEL ---")
    
    stats_table = []
    for model in data_df.columns:
        values = data_df[model].values
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        
        model_alias = create_model_aliases({model})[model]
        
        # Format values with appropriate precision based on magnitude
        if metric_name == 'cost':
            mean_str = format_cost_value(mean_val)
            median_str = format_cost_value(median_val)
            std_str = format_cost_value(std_val)
        else:
            mean_str = f"{mean_val:.4f}"
            median_str = f"{median_val:.4f}"
            std_str = f"{std_val:.4f}"
            
        stats_table.append([
            model_alias,
            mean_str,
            median_str,
            std_str
        ])
    
    # Print as a formatted table
    headers = ["Model", "Mean", "Median", "Std Dev"]
    
    # Calculate column widths
    col_widths = [max(len(headers[i]), max(len(row[i]) for row in stats_table)) for i in range(len(headers))]
    
    # Print header
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for row in stats_table:
        print(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)))
    
    if metric_name == 'cost':
        create_cost_statistics_pdf(stats_table, headers)
    else:
        # Add to summary file for other metrics
        with open(os.path.join(output_dir, f'{metric_name}_statistics.txt'), 'w') as f:
            f.write(f"{metric_name.upper()} STATISTICS BY MODEL\n")
            f.write(header_row + "\n")
            f.write("-" * len(header_row) + "\n")
            for row in stats_table:
                f.write(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + "\n")

def format_cost_value(value):
    """Format cost values with appropriate precision based on magnitude"""
    if value < 0.001:  # Very small values need more precision
        return f"{value:.6f}"
    elif value < 0.01:  # Small values
        return f"{value:.5f}"
    elif value < 0.1:   # Medium values
        return f"{value:.4f}"
    else:               # Larger values
        return f"{value:.4f}"

def create_cost_statistics_pdf(stats_table, headers):
    """Create a PDF with cost statistics and pricing legend"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    for i in range(len(stats_table)):
        row = stats_table[i]
        model_name = row[0]
        
        # Special handling for Llama models with very small values
        if "Llama" in model_name:
            for j in range(1, 4):  # For mean, median, std columns
                # Force scientific notation for extremely small values
                try:
                    value = float(row[j])
                    if value < 0.001:
                        stats_table[i][j] = f"{value:.2e}"  
                    else:
                        stats_table[i][j] = f"{value:.5f}"  
                except ValueError:
                    pass  # Keep as is if not a number
    
    # Top subplot: Cost statistics table
    ax1.set_title("Cost Statistics by Model (USD per query)", fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Create statistics table
    stats_table_display = ax1.table(
        cellText=stats_table,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.2, 0.2, 0.2]
    )
    
    stats_table_display.auto_set_font_size(False)
    stats_table_display.set_fontsize(11)
    stats_table_display.scale(1, 2)
    
    
    for j in range(len(headers)):
        stats_table_display[(0, j)].set_facecolor('#e6f2ff') 
        stats_table_display[(0, j)].set_text_props(weight='bold')
    
    # Bottom subplot: Pricing legend
    ax2.set_title("Model Pricing (per million tokens)", fontsize=14, fontweight='bold', pad=20)
    ax2.axis('off')
    
    # Pricing data
    model_pricing = [
        ["Claude 3.7 Sonnet", "$3.00", "$15.00"],
        ["Qwen 2.5 72B", "$0.25", "$0.75"],
        ["Llama 3.3 70B", "$0.07", "$0.25"],
        ["Llama 3.1 8B", "$0.02", "$0.03"],
        ["Ministral 8B", "$0.10", "$0.10"]
    ]
    
    pricing_headers = ["Model", "Input Tokens", "Output Tokens"]
    
    # Create pricing table
    pricing_table_display = ax2.table(
        cellText=model_pricing,
        colLabels=pricing_headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.3, 0.3]
    )
    
    pricing_table_display.auto_set_font_size(False)
    pricing_table_display.set_fontsize(11)
    pricing_table_display.scale(1, 2)
    
    
    for j in range(len(pricing_headers)):
        pricing_table_display[(0, j)].set_facecolor('#ffe6e6') 
        pricing_table_display[(0, j)].set_text_props(weight='bold')
    
    plt.tight_layout()
    cost_pdf_path = os.path.join(output_dir, 'cost_statistics.pdf')
    plt.savefig(cost_pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Cost statistics with pricing legend saved to: {cost_pdf_path}")


for metric in metrics:
    print(f"\n{'='*50}")
    print(f"ANALYZING {metric.upper()}")
    print(f"{'='*50}")
    
    # Prepare data
    data_df = prepare_data_for_metric(df, metric)
    
    if len(data_df) > 0:
        
        if metric in ['factual_correctness', 'cost']:
            print_metric_stats(data_df, metric)
        
        # Run Friedman test
        friedman_results = run_friedman_test(data_df)
        
        posthoc_results = None
        wilcoxon_results = None
        
        if 'error' not in friedman_results:
            print(f"Friedman test statistic: {friedman_results['statistic']:.4f}")
            print(f"p-value: {friedman_results['p_value']:.4f}")
            print(f"Models compared: {friedman_results['models']}")
            
            # If significant, run post-hoc tests
            if friedman_results['p_value'] < 0.05:
                print("\nFriedman test is significant! Running post-hoc tests...")
                try:
                    # Run Nemenyi post-hoc test
                    posthoc_results = posthoc_nemenyi_friedman(data_df)
                except Exception as e:
                    print(f"Error in Nemenyi test: {e}")
                    
                # Also run Wilcoxon signed-rank tests
                wilcoxon_results = run_pairwise_wilcoxon(data_df)
                
                
                if posthoc_results is not None or (wilcoxon_results is not None and not wilcoxon_results.empty):
                    model_aliases = create_model_aliases(data_df.columns)
                    combined_results = create_combined_posthoc_table(posthoc_results, wilcoxon_results, model_aliases)
            else:
                print("\nFriedman test is NOT significant. No need for post-hoc tests.")
        else:
            print(f"Error running Friedman test: {friedman_results['error']}")
            print(f"Data shape: {friedman_results['rows']} rows, {friedman_results['columns']} columns")
        
        # Visualize the data using the new function
        plot_and_save_metric_comparison(data_df, metric, friedman_results, posthoc_results, wilcoxon_results)
        
        # Save results
        results_by_metric[metric] = {
            'friedman': friedman_results,
            'data_shape': data_df.shape,
            'models': list(data_df.columns),
            'queries': len(data_df)
        }
    else:
        print(f"Not enough data for {metric} after filtering")

# Print summary of all results
print("\n\n=== SUMMARY OF STATISTICAL TESTS ===")
for metric, results in results_by_metric.items():
    if 'error' not in results['friedman']:
        sig_status = "SIGNIFICANT" if results['friedman']['p_value'] < 0.05 else "not significant"
        print(f"{metric}: p={results['friedman']['p_value']:.4f} ({sig_status}), {results['queries']} queries")
    else:
        print(f"{metric}: Error - {results['friedman']['error']}")

# Save summary to a text file
with open(os.path.join(output_dir, 'friedman_test_summary.txt'), 'w') as f:
    f.write("=== SUMMARY OF STATISTICAL TESTS ===\n")
    for metric, results in results_by_metric.items():
        if 'error' not in results['friedman']:
            sig_status = "SIGNIFICANT" if results['friedman']['p_value'] < 0.05 else "not significant"
            f.write(f"{metric}: p={results['friedman']['p_value']:.4f} ({sig_status}), {results['queries']} queries\n")
        else:
            f.write(f"{metric}: Error - {results['friedman']['error']}\n")