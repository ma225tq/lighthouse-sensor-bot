#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced chart styling with mock data.
This shows the improvements in text and data size without needing database connection.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import os
from matplotlib.colors import LinearSegmentedColormap

# Set up output directory
script_dir = Path(__file__).resolve().parent
output_dir = script_dir.parent.parent.parent / "output" / "enhanced_test_charts"
output_dir.mkdir(parents=True, exist_ok=True)

def test_enhanced_heatmap():
    """Test the enhanced heatmap styling"""
    
    # Create mock data similar to the real heatmap
    models = ["Claude 3.7 Sonnet", "Llama 3.1 8B", "Llama 3.3 70B", "Mistral 8B", "Qwen 2.5 72B"]
    metrics = ["Factual Correctness", "Semantic Similarity", "Faithfulness", "BLEU Score", "ROUGE Score", "Non Llm String Similarity"]
    
    # Create mock performance data (0-1 scale)
    np.random.seed(42)  # For reproducible results
    data = []
    for model in models:
        row = []
        for i, metric in enumerate(metrics):
            # Create realistic performance scores with some variation
            base_score = 0.6 + (i * 0.05) + np.random.uniform(-0.2, 0.3)
            score = max(0.0, min(1.0, base_score))  # Clamp between 0 and 1
            row.append(score)
        data.append(row)
    
    df = pd.DataFrame(data, index=models, columns=metrics)
    
    # Create figure with enhanced size and white background for better visibility
    model_count = len(df.index)
    metric_count = len(df.columns)
    
    # Enhanced figure sizing for better readability
    width = max(16, 12 + 1.0 * metric_count)  # Increased width scaling
    height = max(12, 8 + 1.5 * model_count)   # Significantly increased height for better spacing
    
    plt.figure(figsize=(width, height), facecolor='white', dpi=250)  # Higher DPI for better quality
    
    # Create custom light green colormap for good visibility and differentiation
    light_colors = ["#ffffff", "#f0fff0", "#e8f5e8", "#d4f1d4", "#b8e6b8", "#90d190", "#6bb66b", "#4caf50"]
    custom_light_cmap = LinearSegmentedColormap.from_list("light_greens", light_colors)
    
    # Generate heatmap with enhanced styling and lighter colors for better text visibility
    ax = sns.heatmap(
        df,
        annot=True,
        cmap=custom_light_cmap,  # Custom light green colormap for excellent text visibility
        fmt=".2f",
        linewidths=1.5,  # Thicker lines for better separation
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Score', 'shrink': 0.8},  # Better colorbar sizing
        annot_kws={"fontsize": 22, "fontweight": "bold", "color": "black"}  # Large text for maximum visibility
    )
    
    # Enhance colorbar text size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)  # Bigger colorbar tick labels
    cbar.set_label('Score', fontsize=18, fontweight='bold')  # Bigger colorbar label
    
    # Enhanced labels and title with better spacing
    plt.title('Model Performance Across All Metrics (Enhanced)', fontsize=24, fontweight='bold', pad=30)  # Even larger title
    plt.xlabel('Metrics', fontsize=20, fontweight='bold', labelpad=18)  # Larger font with padding
    plt.ylabel('Models', fontsize=20, fontweight='bold', labelpad=25)   # Larger font with more padding
    
    # Set x-axis labels with better formatting and spacing
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=16, fontweight='bold')  # Larger font
    
    # Set y-axis labels (model names) with enhanced spacing and larger font
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=17, fontweight='bold', rotation=0)  # Larger font
    
    # Add extra spacing around the heatmap
    ax.tick_params(axis='y', pad=10)  # Add padding between y-axis labels and heatmap
    ax.tick_params(axis='x', pad=8)   # Add padding between x-axis labels and heatmap
    
    # Adjust layout for optimal spacing
    plt.tight_layout()
    
    # Save the figure with high DPI for better quality
    save_path = output_dir / "enhanced_heatmap_test.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced heatmap saved: {save_path}")

def test_enhanced_bar_chart():
    """Test the enhanced bar chart styling"""
    
    # Create mock data for model comparison
    models = ["Claude 3.7 Sonnet", "Llama 3.1 8B", "Llama 3.3 70B", "Mistral 8B", "Qwen 2.5 72B"]
    scores = [0.86, 0.16, 0.36, 0.16, 0.53]
    query_counts = [150, 150, 150, 150, 150]
    
    # Close any existing plots and start fresh
    plt.close('all')
    
    # Create a new figure with improved size and white background
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    
    # Number of models
    num_models = len(models)
    
    # Define model-specific colors for better distinction
    colors = ['#ff7f0e', '#17becf', '#17becf', '#1f77b4', '#9467bd']  # Orange, Teal, Teal, Blue, Purple
    
    # Create x positions
    x = np.arange(num_models)
    
    # Create bars with increased width for better visibility
    bar_width = 0.7
    bars = ax.bar(
        x, 
        scores, 
        width=bar_width, 
        color=colors, 
        zorder=5,
        edgecolor='white',  # White edges for better definition
        linewidth=0.8,      # Subtle edge lines
        alpha=0.9           # Slight transparency for better look
    )
    
    # Add a light background color for better visual appeal
    ax.set_facecolor('#f8f9fa')  # Very light gray background
    
    # Enhanced labels and title with improved formatting
    ax.set_title('Average Factual Correctness by Model (Enhanced)', fontsize=24, fontweight='bold', pad=30)  # Larger title
    ax.set_xlabel('Model', fontsize=20, fontweight='bold', labelpad=18)  # Larger labels
    ax.set_ylabel('Average Factual Correctness', fontsize=20, fontweight='bold', labelpad=25)  # Larger labels
    
    # Add grid lines behind the bars
    ax.grid(axis='y', linestyle='-', alpha=0.2, color='gray', zorder=0)
    
    # Set x-tick positions and labels with proper rotation and enhanced font
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=16, fontweight='bold')  # Larger, bold tick labels
    
    # Enhance y-axis tick labels
    ax.tick_params(axis='y', labelsize=14, labelpad=8)  # Larger y-axis labels
    
    # Set y-axis limits for better visualization
    max_score = max(scores)
    ax.set_ylim(0, min(1.0, max_score * 1.1))  # Cap at 1.0 or 10% above max score
    
    # Improve axis appearance
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # Thicker border
        spine.set_color('#333333')  # Darker border for better definition
    
    # Add data labels on top of bars with enhanced styling
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Ensure non-zero values are visible with a minimum height but label shows actual value
        display_height = max(0.01, height) if height > 0 else 0
        if height != display_height:
            bar.set_height(display_height)
            
        ax.annotate(f'{height:.2f}', 
                  xy=(bar.get_x() + bar.get_width() / 2, display_height),
                  xytext=(0, 5),  # 5 points vertical offset for better spacing
                  textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=16, fontweight='bold', zorder=10,  # Larger, bolder text
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))  # Add subtle text background
    
    # Add the number of queries per model at the bottom of each bar with enhanced styling
    for i, count in enumerate(query_counts):
        ax.annotate(f'n={count}',
                  xy=(x[i], 0.01),
                   ha='center', va='bottom',
                   fontsize=14, color='#555555', zorder=10, fontweight='bold')  # Larger, bold text
    
    # Add more padding at the bottom for model names
    plt.subplots_adjust(bottom=0.25)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the figure with higher resolution
    save_path = output_dir / "enhanced_bar_chart_test.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced bar chart saved: {save_path}")

def main():
    """Generate test charts showing the enhanced styling"""
    print(f"Generating enhanced test charts in {output_dir}")
    print("=" * 60)
    
    # Test enhanced heatmap
    print("\n1. Enhanced Heatmap Test")
    test_enhanced_heatmap()
    
    # Test enhanced bar chart  
    print("\n2. Enhanced Bar Chart Test")
    test_enhanced_bar_chart()
    
    print("\n" + "=" * 60)
    print(f"All enhanced test charts have been saved to {output_dir}")
    print("\nKey enhancements applied:")
    print("• Font sizes increased significantly (titles: 24pt, labels: 20pt, data: 16-22pt)")
    print("• All text made bold for better visibility")
    print("• Higher DPI (250-300) for better quality")
    print("• Enhanced spacing and padding")
    print("• Better color schemes for text visibility")
    print("• Larger figure sizes for better readability")

if __name__ == "__main__":
    main() 