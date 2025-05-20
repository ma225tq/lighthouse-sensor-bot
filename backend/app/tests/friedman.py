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
    em.string_present
FROM query_result qr
JOIN llm_models lm ON qr.llm_model_id = lm.id
JOIN query_evaluation qe ON qr.id = qe.query_result_id
JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
ORDER BY qr.query, lm.name
"""

# Load data into pandas DataFrame
df = pd.read_sql(query, conn)
conn.close()

# database typo, cant be bothered to fix
df.rename(columns={'rogue_score': 'rouge_score'}, inplace=True)

# Print diagnostic information
print(f"Total rows in original dataset: {len(df)}")
print(f"Unique queries by ID: {df['query_id'].nunique()}")
print(f"Unique queries by text: {df['query_text'].nunique()}")
print(f"Unique models: {df['model_name'].nunique()}")
print(f"Models in dataset: {df['model_name'].unique()}")

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

# Analyze all metrics
metrics = [
    'factual_correctness', 
    'semantic_similarity',
    'faithfulness',
    'bleu_score',
    'non_llm_string_similarity',
    'rouge_score',  # Fixed typo here
    'string_present'
]

results_by_metric = {}

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

def create_posthoc_graph(data_df, metric_name, posthoc_results, wilcoxon_results, model_aliases):
    """Create visualizations for post-hoc tests"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Add Nemenyi post-hoc test results if available
    if posthoc_results is not None:
        axes[0].set_title("Post-hoc Nemenyi Test p-values")
        posthoc_vals = posthoc_results.values
        models = posthoc_results.columns
        
        # Create a heatmap-like table
        im = axes[0].imshow(posthoc_vals, cmap='coolwarm', vmin=0, vmax=0.1)
        
        # Add model aliases as labels
        axes[0].set_xticks(np.arange(len(models)))
        axes[0].set_yticks(np.arange(len(models)))
        axes[0].set_xticklabels([model_aliases[m] for m in models])
        axes[0].set_yticklabels([model_aliases[m] for m in models])
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add grid lines to create borders around each cell
        for edge, spine in axes[0].spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
            
        axes[0].set_xticks(np.arange(-.5, len(models), 1), minor=True)
        axes[0].set_yticks(np.arange(-.5, len(models), 1), minor=True)
        axes[0].grid(which='minor', color='black', linestyle='-', linewidth=1)
        
        # Add p-values in each cell
        for i in range(len(models)):
            for j in range(len(models)):
                if not np.isnan(posthoc_vals[i, j]):
                    text_color = "black" if posthoc_vals[i, j] > 0.05 else "white"
                    axes[0].text(j, i, f"{posthoc_vals[i, j]:.4f}", 
                               ha="center", va="center", color=text_color,
                               fontsize=8)
        
        # Add a colorbar
        plt.colorbar(im, ax=axes[0], label="p-value")
        axes[0].set_aspect('equal')
    
    # Add Wilcoxon test results if available
    if wilcoxon_results is not None and not wilcoxon_results.empty:
        axes[1].set_title("Pairwise Wilcoxon Signed-Rank Tests (Bonferroni Correction)")
        axes[1].axis('off')
        
        # Format Wilcoxon results as a table
        if 'error' in wilcoxon_results.columns:
            print(f"Error in Wilcoxon tests: {wilcoxon_results['error'].iloc[0]}")
        else:
            # Use the actual aliases instead of model numbers
            table_data = []
            for _, row in wilcoxon_results.iterrows():
                # Use the actual aliases
                m1_name = model_aliases[row['model1']]
                m2_name = model_aliases[row['model2']]
                
                sig_text = "✓" if row.get('significant', False) else ""
                table_data.append([
                    f"{m1_name} vs {m2_name}",
                    f"{row['p_value']:.4f}",
                    sig_text
                ])
            
            if table_data:
                table = axes[1].table(
                    cellText=table_data,
                    colLabels=['Comparison', 'p-value', 'Significant (p<0.05)'],
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.6, 0.2, 0.2]  # Control column widths
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
    
    plt.tight_layout()
    posthoc_path = os.path.join(output_dir, f'{metric_name}_posthoc_tests.pdf')
    plt.savefig(posthoc_path)
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
        stats_table.append([
            model_alias,
            f"{mean_val:.4f}",
            f"{median_val:.4f}",
            f"{std_val:.4f}"
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
    
    # Add to summary file
    with open(os.path.join(output_dir, f'{metric_name}_statistics.txt'), 'w') as f:
        f.write(f"{metric_name.upper()} STATISTICS BY MODEL\n")
        f.write(header_row + "\n")
        f.write("-" * len(header_row) + "\n")
        for row in stats_table:
            f.write(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + "\n")

# Modify the main loop to call this function for factual_correctness
for metric in metrics:
    print(f"\n{'='*50}")
    print(f"ANALYZING {metric.upper()}")
    print(f"{'='*50}")
    
    # Prepare data
    data_df = prepare_data_for_metric(df, metric)
    
    if len(data_df) > 0:
        # Print descriptive statistics for factual correctness
        if metric == 'factual_correctness':
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
                    print("\nPost-hoc Nemenyi test p-values:")
                    print(posthoc_results)
                except Exception as e:
                    print(f"Error in Nemenyi test: {e}")
                    
                # Also run Wilcoxon signed-rank tests
                wilcoxon_results = run_pairwise_wilcoxon(data_df)
                print("\nPairwise Wilcoxon signed-rank tests:")
                print(wilcoxon_results)
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