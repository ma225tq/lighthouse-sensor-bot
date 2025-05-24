import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import decimal
import json
from matplotlib.colors import LinearSegmentedColormap

# Import our custom database configuration
try:
    from .db_config import get_connection_params
    USE_CUSTOM_CONFIG = True
except ImportError:
    from app.conf.postgres import get_cursor
    USE_CUSTOM_CONFIG = False

# Set up logging
logger = logging.getLogger(__name__)

# Custom cursor context manager using our configuration
@contextmanager
def get_custom_cursor():
    """Get a database cursor using custom configuration."""
    conn = None
    try:
        db_params = get_connection_params()
        conn = psycopg2.connect(
            host=db_params['host'],
            dbname=db_params['dbname'],
            user=db_params['user'],
            password=db_params['password'],
            port=db_params['port']
        )
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

class ChartGenerator:
    """
    Utility class for generating charts from evaluation results stored in PostgreSQL.
    These charts can be used for thesis or presentation purposes.
    """
    
    def __init__(self, output_dir: str = None, pdf_only: bool = True):
        """
        Initialize the chart generator.
        
        Args:
            output_dir: Directory to save charts to. Defaults to 'output/charts' in the project root.
            pdf_only: If True, generate only PDF files (skip PNG generation)
        """
        if output_dir is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent.parent
            self.output_dir = project_root / "output" / "charts"
        else:
            self.output_dir = Path(output_dir)
        
        self.pdf_only = pdf_only
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set default plot style
        sns.set_theme(style="whitegrid")
        
    def _fetch_data(self, query: str) -> pd.DataFrame:
        """
        Fetch data from the database using the provided SQL query.
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame containing the query results
        """
        try:
            # Use custom cursor or default cursor based on configuration
            cursor_manager = get_custom_cursor if USE_CUSTOM_CONFIG else get_cursor
            
            with cursor_manager() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # Convert Decimal objects to floats for compatibility with matplotlib
            for col in df.select_dtypes(include=['object']).columns:
                # Check if the column contains Decimal objects
                if df[col].first_valid_index() is not None and isinstance(
                    df.loc[df[col].first_valid_index(), col], decimal.Decimal
                ):
                    df[col] = df[col].astype(float)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def _save_figure(self, filename: str, fig: plt.Figure = None, dpi=300, save_pdf=True, pdf_only=False) -> str:
        """
        Save the current figure or a provided figure to the output directory.
        
        Args:
            filename: Filename for the saved chart (without extension)
            fig: Figure to save, if None the current figure is saved
            dpi: Resolution for the saved image
            save_pdf: Whether to also save as PDF
            pdf_only: If True, only save PDF and skip PNG generation
            
        Returns:
            Path to the saved PNG or PDF file
        """
        if pdf_only and save_pdf:
            # Save only PDF
            pdf_path = self.output_dir / f"{filename}.pdf"
            
            if fig:
                fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
            else:
                plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
                
            plt.close()
            return str(pdf_path)
        else:
            # Save PNG (and optionally PDF)
            png_path = self.output_dir / f"{filename}.png"
            
            if fig:
                fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
                if save_pdf:
                    pdf_path = self.output_dir / f"{filename}.pdf"
                    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
            else:
                plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
                if save_pdf:
                    pdf_path = self.output_dir / f"{filename}.pdf"
                    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
                
            plt.close()
            return str(png_path)
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename by replacing unsafe characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Replace slashes and other problematic characters
        sanitized = filename.replace('/', '_').replace('\\', '_')
        sanitized = sanitized.replace(':', '_').replace('*', '_')
        sanitized = sanitized.replace('?', '_').replace('"', '_')
        sanitized = sanitized.replace('<', '_').replace('>', '_')
        sanitized = sanitized.replace('|', '_').replace(' ', '_')
        return sanitized
    
    def _shorten_model_name(self, model_name: str) -> str:
        """
        Convert long model names to shorter display names.
        
        Args:
            model_name: Original model name
            
        Returns:
            Shortened model name
        """
        # Define mappings for common model names
        name_mappings = {
            'anthropic/claude-3.7-sonnet': 'Claude 3.7 Sonnet',
            'meta-llama/llama-3.3-70b-instruct': 'Llama 3.3 70B', 
            'meta-llama/llama-3.1-8b-instruct': 'Llama 3.1 8B',
            'qwen/qwen-2.5-72b-instruct': 'Qwen 2.5 72B',
            'mistralai/ministral-8b': 'Ministral 8B',
            'openai/gpt-4o-2024-11-20': 'GPT-4o',
            'google/gemini-2.5-flash-preview': 'Gemini 2.5 Flash',
            'nvidia/llama-3.1-nemotron-70b-instruct': 'Nemotron 70B',
            'nousresearch/hermes-3-llama-3.1-70b': 'Hermes 3 70B'
        }
        
        # Return mapped name if available, otherwise return original
        return name_mappings.get(model_name, model_name)
    
    def model_comparison_chart(self, 
                              metric_name: str = 'factual_correctness',
                              model_names: List[str] = None) -> str:
        """
        Generate a bar chart comparing different models on a specific metric.
        
        Args:
            metric_name: The name of the metric to compare (e.g., 'factual_correctness')
            model_names: List of model names to include, if None all models are included
            
        Returns:
            Path to the saved chart image
        """
        # Define the base query
        query = f"""
        SELECT 
            m.name AS model_name,
            AVG(em.{metric_name}) AS avg_score,
            COUNT(*) AS query_count
        FROM 
            query_result qr
            JOIN llm_models m ON qr.llm_model_id = m.id
            JOIN query_evaluation qe ON qr.id = qe.query_result_id
            JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
        """
        
        # Add filter for specific models if provided
        if model_names and len(model_names) > 0:
            model_filters = ", ".join([f"'{model}'" for model in model_names])
            query += f"WHERE m.name IN ({model_filters})"
            
        # Group by model and sort by score
        query += """
        GROUP BY 
            m.name
        ORDER BY 
            avg_score DESC
        """
        
        # Fetch the data
        df = self._fetch_data(query)
        
        if df.empty:
            logger.warning(f"No data found for metric: {metric_name}")
            return "No data available"
        
        # Apply model name shortening
        df['model_name'] = df['model_name'].apply(self._shorten_model_name)
        
        # Close any existing plots and start fresh
        plt.close('all')
        
        # Create a new figure with improved size and white background
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
        
        # Get model names and scores
        models = df['model_name'].tolist()
        scores = df['avg_score'].tolist()
        query_counts = df['query_count'].tolist()
        
        # Number of models
        num_models = len(models)
        
        # Define a vibrant color palette - using custom colors for better clarity
        # Use a more visually appealing blue gradient
        base_color = '#1f77b4'  # Base blue color
        colors = []
        
        # Define model-specific colors
        for model in models:
            if 'claude' in model.lower():
                colors.append('#ff7f0e')  # Orange for Claude models
            elif 'gpt' in model.lower() or 'openai' in model.lower():
                colors.append('#e377c2')  # Pink for GPT/OpenAI models
            elif 'gemini' in model.lower():
                colors.append('#2ca02c')  # Green for Gemini models
            elif 'llama' in model.lower():
                colors.append('#17becf')  # Teal for Llama models (changed from red)
            elif 'qwen' in model.lower():
                colors.append('#9467bd')  # Purple for Qwen models
            else:
                colors.append(base_color)  # Default blue for other models
        
        # Create x positions
        x = np.arange(num_models)
        
        # Create bars with increased width for better visibility
        bar_width = 0.7
        # Create bars with enhanced visual appeal - add gradient effect
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
        
        # Add labels and title with improved formatting
        formatted_metric = metric_name
        if metric_name == 'rogue_score':
            formatted_metric = 'ROUGE Score'
        elif metric_name == 'bleu_score':
            formatted_metric = 'BLEU Score'
        else:
            formatted_metric = ' '.join(word.capitalize() for word in metric_name.split('_'))
        ax.set_title(f'Average {formatted_metric} by Model', fontsize=24, fontweight='bold', pad=30)  # Larger title like heatmap
        ax.set_xlabel('Model', fontsize=20, fontweight='bold', labelpad=18)  # Larger labels like heatmap
        ax.set_ylabel(f'Average {formatted_metric}', fontsize=20, fontweight='bold', labelpad=25)  # Larger labels like heatmap
        
        # Add grid lines behind the bars
        ax.grid(axis='y', linestyle='-', alpha=0.2, color='gray', zorder=0)
        
        # Set x-tick positions and labels with proper rotation and enhanced font
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=50, ha='right', fontsize=14, fontweight='bold')  # More slanted and slightly smaller for better spacing
        
        # Enhance y-axis tick labels
        ax.tick_params(axis='y', labelsize=16, pad=8)  # Larger y-axis labels (increased from 14)
        
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
                
            ax.annotate(f'{height:.3f}', 
                      xy=(bar.get_x() + bar.get_width() / 2, display_height),
                      xytext=(0, 5),  # 5 points vertical offset for better spacing
                      textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=20, fontweight='bold', zorder=10)  # Smaller data labels to prevent overlap (reduced from 28)
        
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
        return self._save_figure(f"model_comparison_{metric_name}", fig=fig, dpi=300, pdf_only=self.pdf_only)
    
    def metric_comparison_chart(self, model_name: str) -> str:
        """
        Generate a radar chart showing all metrics for a specific model.
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Path to the saved chart image
        """
        # Define metrics to include - use the same approach as ragas_radar_chart
        metrics = [
            'factual_correctness',
            'semantic_similarity',
            'context_recall',
            'faithfulness',
            'bleu_score',
            'rogue_score',
            'string_present'
        ]
        
        # Define the query with flexible column handling as in ragas_radar_chart
        try:
            # First try to get available columns from the database
            schema_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'evaluation_metrics'
            """
            
            available_columns = self._fetch_data(schema_query)
            available_metrics = [col[0] for col in available_columns.values]
            
            # Filter metrics to only include those that exist
            valid_metrics = [m for m in metrics if m in available_metrics]
            
            if not valid_metrics:
                logger.error("No valid metrics found in database")
                raise ValueError("No valid metrics available")
                
            # Build query with only valid metrics
            metrics_str = ', '.join([f'AVG(em.{metric}) AS {metric}' for metric in valid_metrics])
            
            query = f"""
            SELECT 
                {metrics_str},
                COUNT(*) AS query_count
            FROM 
                query_result qr
                JOIN llm_models m ON qr.llm_model_id = m.id
                JOIN query_evaluation qe ON qr.id = qe.query_result_id
                JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
            WHERE 
                m.name = '{model_name}'
            """
            
            # Fetch the data
            df = self._fetch_data(query)
            
            # Update metrics list to what's actually available
            metrics = valid_metrics
            
        except Exception as e:
            logger.error(f"Error querying data for {model_name}: {e}")
            
            # Fallback to basic query with minimal metrics
            basic_metrics = ['factual_correctness', 'semantic_similarity']
            metrics_str = ', '.join([f'AVG(em.{metric}) AS {metric}' for metric in basic_metrics])
            query = f"""
            SELECT 
                {metrics_str},
                COUNT(*) AS query_count
            FROM 
                query_result qr
                JOIN llm_models m ON qr.llm_model_id = m.id
                JOIN query_evaluation qe ON qr.id = qe.query_result_id
                JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
            WHERE 
                m.name = '{model_name}'
            """
            
            # Fetch the data
            df = self._fetch_data(query)
            metrics = basic_metrics
        
        # Create figure and polar axis
        fig = plt.figure(figsize=(12, 12), facecolor='white')
        ax = plt.subplot(111, polar=True)
        ax.set_facecolor('white')
        
        # Number of metrics
        N = len(metrics)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the polygon
        
        # Format metric names for display
        metric_labels = ['ROUGE Score' if metric == 'rogue_score' else ('BLEU Score' if metric == 'bleu_score' else ' '.join(word.capitalize() for word in metric.split('_'))) for metric in metrics]
        
        # Turn off the default grid
        ax.grid(False)
        
        # Add black axis lines from center to each vertex
        for angle in angles[:-1]:
            ax.plot([0, angle], [0, 1], 'black', linewidth=0.8, alpha=0.5)
            
        # Add concentric circles to create a spider web effect
        for r in [0.2, 0.4, 0.6, 0.8]:
            # Calculate points on the circle
            circle_angles = np.linspace(0, 2*np.pi, 100)
            # Draw the circle as a line
            ax.plot(circle_angles, [r] * len(circle_angles), color='black', linestyle='-', linewidth=0.6, alpha=0.3)
        
        # Set the angle labels
        plt.xticks(angles[:-1], metric_labels, fontsize=14, fontweight='bold')
        
        # Set y limits
        plt.ylim(0, 1)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], 
                 color="black", size=12)
        
        # Check if we have valid data
        if df.empty or df['query_count'].iloc[0] == 0:
            # Add a message in the center of the chart
            ax.text(0, 0, f"No data available for\n{model_name}", 
                   fontsize=18, fontweight='bold', ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.5'))
            
            # Add title indicating no data
            plt.title(f'Metrics Radar Chart for {model_name} (No Data)', fontsize=24, fontweight='bold', y=1.05, pad=30)  # Larger title like heatmap
            
            # Log a warning
            logger.warning(f"No data found for model: {model_name}")
            
            # Save the figure with sanitized filename
            safe_model_name = self._sanitize_filename(model_name)
            return self._save_figure(f"radar_chart_{safe_model_name}_no_data", dpi=300, pdf_only=self.pdf_only)
        
        # Get the query count
        query_count = int(df['query_count'].iloc[0])
        
        # Convert any remaining Decimal values to float
        for col in df.columns:
            if col != 'query_count' and col in metrics:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Process the data for radar chart - handle any missing columns
        values = [df.iloc[0][metric] if metric in df.columns else 0 for metric in metrics]
        
        # Debug print the actual values before adjustment
        print(f"\nValues for {model_name} before adjustment:")
        for m, v in zip(metrics, values):
            print(f"  - {m}: {v}")
        
        # Get original values before min adjustment
        original_values = values.copy()
        
        # Replace very small values with a minimum visible value
        values = [max(0.01, v) if v > 0 else 0 for v in values]
        
        # Add values to complete the polygon
        values += values[:1]
        original_values += original_values[:1]  # Also extend original values
        
        # Choose a vibrant color for the model - match the multi-model chart color for consistency
        model_color = '#1f77b4'  # Blue for nova-pro-v1
        if model_name.lower().startswith('claude'):
            model_color = '#ff7f0e'  # Orange for Claude models
        elif model_name.lower().startswith('gpt') or model_name.lower().startswith('openai'):
            model_color = '#e377c2'  # Pink for GPT/OpenAI models
        elif model_name.lower().startswith('gemini'):
            model_color = '#2ca02c'  # Green for Gemini models
        elif model_name.lower().startswith('llama'):
            model_color = '#17becf'  # Teal for Llama models (changed from red)
        
        # Plot the polygon with higher line width
        ax.plot(angles, values, linewidth=3.5, linestyle='solid', 
               label=model_name, color=model_color)
        
        # Fill with semi-transparent color
        ax.fill(angles, values, alpha=0.25, color=model_color)
        
        # Add data points at each vertex with bigger markers
        for i, value in enumerate(values[:-1]):
            ax.scatter(angles[i], value, s=120, color=model_color, 
                     edgecolor='black', linewidth=1.5, zorder=10)
            
            # Add value labels at each point
            ha = 'left' if angles[i] > np.pi else 'right'
            va = 'bottom' if angles[i] < np.pi/2 or angles[i] > 3*np.pi/2 else 'top'
            offset = 0.05
            x_offset = np.cos(angles[i]) * offset
            y_offset = np.sin(angles[i]) * offset
            
            # Add text with background for better visibility - ALWAYS show non-zero values
            # Use the original value (before min height adjustment) for the label
            original_value = original_values[i]
            if original_value > 0:  # Show ALL non-zero values
                print(f"Adding label for {metrics[i]}: {original_value}")
                ax.text(angles[i] + x_offset, value + y_offset, f'{original_value:.3f}', 
                       fontsize=12, fontweight='bold', ha=ha, va=va,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Add title with enhanced styling like heatmap
        plt.title(f'Metrics Radar Chart for {model_name} (n={query_count})', fontsize=24, fontweight='bold', y=1.05, pad=30)  # Larger title like heatmap
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure with sanitized filename
        safe_model_name = self._sanitize_filename(model_name)
        return self._save_figure(f"radar_chart_{safe_model_name}", dpi=300, pdf_only=self.pdf_only)
    
    def metrics_heatmap(self, 
                       model_names: List[str] = None,
                       metrics: List[str] = None) -> str:
        """
        Generate a heatmap comparing multiple models across multiple metrics.
        
        Args:
            model_names: List of model names to include
            metrics: List of metrics to include
            
        Returns:
            Path to the saved chart image
        """
        if metrics is None:
            # Include all RAGAS metrics by default
            metrics = [
                'factual_correctness',
                'semantic_similarity',
                'faithfulness',
                'bleu_score',
                'rogue_score',
                'non_llm_string_similarity'  # Using exact database column name
            ]
        
        # Try to get available columns from the database
        try:
            schema_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'evaluation_metrics'
            """
            
            available_columns = self._fetch_data(schema_query)
            available_metrics = [col[0] for col in available_columns.values]
            
            # Check for string similarity-related columns with flexible matching
            string_sim_variants = ['string_similarity', 'stringsimilarity', 'str_similarity', 'similarity_string']
            string_sim_cols = [col for col in available_metrics if any(variant in col.lower() for variant in string_sim_variants)]
            
            if string_sim_cols:
                # Replace our standard name with the actual column name from database
                metrics = [string_sim_cols[0] if m == 'string_similarity' else m for m in metrics]
            
            # Filter metrics to only include those that exist
            valid_metrics = [m for m in metrics if m in available_metrics]
            
            if not valid_metrics:
                logger.error("No valid metrics found in database")
                valid_metrics = ['factual_correctness']
                
            # Update metrics list to what's actually available
            metrics = valid_metrics
            
        except Exception as e:
            logger.error(f"Error querying schema for heatmap: {e}")
            # Continue with the provided metrics
        
        # Define the query
        metrics_str = ', '.join([f'AVG(em.{metric}) AS {metric}' for metric in metrics])
        query = f"""
        SELECT 
            m.name AS model_name,
            {metrics_str}
        FROM 
            query_result qr
            JOIN llm_models m ON qr.llm_model_id = m.id
            JOIN query_evaluation qe ON qr.id = qe.query_result_id
            JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
        """
        
        # Add filter for specific models if provided
        if model_names and len(model_names) > 0:
            model_filters = ", ".join([f"'{model}'" for model in model_names])
            query += f"WHERE m.name IN ({model_filters})"
            
        # Group by model
        query += """
        GROUP BY 
            m.name
        ORDER BY 
            model_name
        """
        
        # Fetch the data
        df = self._fetch_data(query)
        
        if df.empty:
            logger.warning("No data found for heatmap")
            return "No data available"
        
        # Apply model name shortening before setting as index
        df['model_name'] = df['model_name'].apply(self._shorten_model_name)
        
        # Set model name as index
        df = df.set_index('model_name')
        
        # Ensure all data is numeric
        df = df.astype(float)
        
        # Ensure very small values are preserved by setting a minimum display value
        for col in df.columns:
            # Replace extremely small values (but > 0) with a minimum visible value
            df[col] = df[col].apply(lambda x: max(0.001, x) if x > 0 else 0)
        
        # Create figure with enhanced size and white background for better visibility
        model_count = len(df.index)
        metric_count = len(df.columns)
        
        # Enhanced figure sizing for better readability
        width = max(16, 12 + 1.0 * metric_count)  # Increased width scaling
        height = max(12, 8 + 1.5 * model_count)   # Significantly increased height for better spacing
        
        plt.figure(figsize=(width, height), facecolor='white', dpi=250)  # Higher DPI for better quality
        
        # Create custom colormap with stronger contrast for better visibility of differences  
        # Use a green gradient for maximum contrast between low and high values
        strong_colors = ["#ffffff", "#f1f8e9", "#dcedc8", "#c5e1a5", "#aed581", "#9ccc65", "#8bc34a", "#689f38", "#558b2f", "#33691e"]
        custom_light_cmap = LinearSegmentedColormap.from_list("strong_greens", strong_colors)
        
        # Generate heatmap with enhanced styling and lighter colors for better text visibility
        ax = sns.heatmap(
            df,
            annot=True,
            cmap=custom_light_cmap,  # Custom light green colormap for excellent text visibility
            fmt=".3f",
            linewidths=1.5,  # Thicker lines for better separation
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Score', 'shrink': 0.8},  # Better colorbar sizing
            annot_kws={"fontsize": 32, "fontweight": "bold", "color": "black"}  # Much larger text for maximum visibility (increased from 26)
        )
        
        # Enhance colorbar text size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)  # Bigger colorbar tick labels
        cbar.set_label('Score', fontsize=18, fontweight='bold')  # Bigger colorbar label
        
        # Format metric names for display
        formatted_metrics = ['ROUGE Score' if metric == 'rogue_score' else ('BLEU Score' if metric == 'bleu_score' else ' '.join(word.capitalize() for word in metric.split('_'))) for metric in metrics]
        
        # Enhanced labels and title with better spacing
        plt.title('Model Performance Across All Metrics', fontsize=24, fontweight='bold', pad=30)  # Even larger title
        plt.xlabel('Metrics', fontsize=20, fontweight='bold', labelpad=18)  # Larger font with padding
        plt.ylabel('Models', fontsize=20, fontweight='bold', labelpad=25)   # Larger font with more padding
        
        # Set x-axis labels with better formatting and spacing
        ax.set_xticklabels(formatted_metrics, rotation=45, ha='right', fontsize=20, fontweight='bold')  # Larger RAGAS metrics labels (increased from 16)
        
        # Set y-axis labels (model names) with enhanced spacing and larger font
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=21, fontweight='bold', rotation=0)  # Larger model names (increased from 17)
        
        # Add extra spacing around the heatmap
        ax.tick_params(axis='y', pad=10)  # Add padding between y-axis labels and heatmap
        ax.tick_params(axis='x', pad=8)   # Add padding between x-axis labels and heatmap
        
        # Adjust layout for optimal spacing
        plt.tight_layout()
        
        # Save the figure with high DPI for better quality
        return self._save_figure("metrics_heatmap", dpi=300, pdf_only=self.pdf_only)
    
    def query_performance_chart(self, 
                              model_name: str,
                              metric_name: str = 'factual_correctness',
                              limit: int = 10) -> str:
        """
        Generate a bar chart showing performance on individual queries for a specific model.
        
        Args:
            model_name: Name of the model to analyze
            metric_name: Name of the metric to visualize
            limit: Maximum number of queries to include
            
        Returns:
            Path to the saved chart image
        """
        # Define the query
        query = f"""
        SELECT 
            qr.query,
            em.{metric_name} AS score
        FROM 
            query_result qr
            JOIN llm_models m ON qr.llm_model_id = m.id
            JOIN query_evaluation qe ON qr.id = qe.query_result_id
            JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
        WHERE 
            m.name = '{model_name}'
        ORDER BY 
            qr."timestamp" DESC
        LIMIT {limit}
        """
        
        # Fetch the data
        df = self._fetch_data(query)
        
        if df.empty:
            logger.warning(f"No data found for model: {model_name}")
            return "No data available"
        
        # Ensure score is numeric
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        
        # Truncate long queries
        df['short_query'] = df['query'].apply(lambda x: (x[:50] + '...') if len(x) > 50 else x)
        
        # Create figure with enhanced size like heatmap
        plt.figure(figsize=(16, 10), facecolor='white', dpi=250)  # Larger figure with higher DPI like heatmap
        
        # Create horizontal bar chart
        ax = sns.barplot(
            data=df,
            y='short_query',
            x='score',
            orient='h'
        )
        
        # Format metric name for display
        formatted_metric = ' '.join(word.capitalize() for word in metric_name.split('_'))
        
        # Add labels and title with enhanced styling like heatmap
        plt.title(f'{formatted_metric} Scores by Query for {model_name}', fontsize=24, fontweight='bold', pad=30)  # Larger title like heatmap
        plt.xlabel(f'{formatted_metric} Score', fontsize=20, fontweight='bold', labelpad=18)  # Larger labels like heatmap
        plt.ylabel('Query', fontsize=20, fontweight='bold', labelpad=25)  # Larger labels like heatmap
        
        # Enhance tick labels
        plt.xticks(fontsize=14, fontweight='bold')  # Larger x-axis tick labels
        plt.yticks(fontsize=14, fontweight='bold')  # Larger y-axis tick labels
        
        # Add data labels with enhanced styling
        for p in ax.patches:
            width = p.get_width()
            # Ensure non-zero values are visible with a minimum width but label shows actual value
            display_width = max(0.01, width) if width > 0 else 0
            if width != display_width:
                p.set_width(display_width)
                
            ax.annotate(f'{width:.3f}', 
                      (display_width, p.get_y() + p.get_height() / 2),
                      ha='left', va='center',
                      fontsize=16, fontweight='bold', xytext=(5, 0),  # Larger, bolder text like heatmap
                      textcoords='offset points',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))  # Add background like heatmap
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure with sanitized filename
        safe_model_name = self._sanitize_filename(model_name)
        return self._save_figure(f"query_performance_{safe_model_name}_{metric_name}", pdf_only=self.pdf_only)
    
    def model_vs_model_chart(self,
                           model1: str,
                           model2: str,
                           metrics: List[str] = None) -> str:
        """
        Generate a side-by-side bar chart comparing two models across multiple metrics.
        
        Args:
            model1: Name of the first model
            model2: Name of the second model
            metrics: List of metrics to include
            
        Returns:
            Path to the saved chart image
        """
        if metrics is None:
            metrics = [
                'factual_correctness',
                'semantic_similarity',
                'faithfulness',
                'bleu_score',
                'rogue_score',
                'non_llm_string_similarity'
            ]
        
        # Get available columns from the database to ensure we only query existing metrics
        try:
            schema_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'evaluation_metrics'
            """
            
            available_columns = self._fetch_data(schema_query)
            available_metrics = [col[0] for col in available_columns.values]
            
            # Filter metrics to only include those that exist
            valid_metrics = [m for m in metrics if m in available_metrics]
            
            if not valid_metrics:
                logger.error("No valid metrics found in database")
                return "No valid metrics available"
                
            # Update metrics list to what's actually available
            metrics = valid_metrics
            
        except Exception as e:
            logger.error(f"Error querying schema: {e}")
            # Continue with the provided metrics list
        
        # Define the query
        metrics_str = ', '.join([f'AVG(em.{metric}) AS {metric}' for metric in metrics])
        query = f"""
        SELECT 
            m.name AS model_name,
            {metrics_str}
        FROM 
            query_result qr
            JOIN llm_models m ON qr.llm_model_id = m.id
            JOIN query_evaluation qe ON qr.id = qe.query_result_id
            JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
        WHERE 
            m.name IN ('{model1}', '{model2}')
        GROUP BY 
            m.name
        """
        
        # Fetch the data
        df = self._fetch_data(query)
        
        if df.empty:
            logger.warning(f"No data found for models: {model1} and {model2}")
            return "No data available"
        
        # Ensure all data is numeric
        for col in metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Reshape data for grouped bar chart
        df_melted = pd.melt(
            df, 
            id_vars=['model_name'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Score'
        )
        
        # Format metric names for display
        df_melted['Metric'] = df_melted['Metric'].apply(
            lambda x: 'ROUGE Score' if x == 'rogue_score' else ('BLEU Score' if x == 'bleu_score' else ' '.join(word.capitalize() for word in x.split('_')))
        )
        
        # Completely fresh approach - use a custom plotting method
        
        # Close any existing plots and start fresh
        plt.close('all')
        
        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        
        # Get unique metrics and models
        metrics_display = df_melted['Metric'].unique()
        models = df_melted['model_name'].unique()
        num_metrics = len(metrics_display)
        
        # Define colors
        colors = ['#1f77b4', '#ff7f0e']  # Blue and orange for better contrast
        
        # Custom bar positioning
        width = 0.35  # Width of the bars
        x = np.arange(num_metrics)  # Metric positions on x-axis
        
        # Make sure models are in the correct order (model1 should be first)
        ordered_models = [model1, model2]
        
        # Custom bar plot to replace seaborn's barplot
        for i, model in enumerate(ordered_models):
            model_data = df_melted[df_melted['model_name'] == model]
            
            # Get scores for each metric
            scores = [model_data[model_data['Metric'] == metric]['Score'].values[0] 
                     for metric in metrics_display]
            
            # Plot bars with offset positions
            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, scores, width, label=model, color=colors[i], zorder=5)
            
            # Add text annotations for each bar with enhanced styling
            for j, bar in enumerate(bars):
                height = bar.get_height()
                # Ensure non-zero values are visible with a minimum height but label shows actual value
                display_height = max(0.01, height) if height > 0 else 0
                if height != display_height:
                    bar.set_height(display_height)
                
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, display_height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=16, fontweight='bold', zorder=10)  # Smaller data labels to prevent overlap (reduced from 28)
        
        # Configure the rest of the chart with enhanced styling like heatmap
        ax.set_title(f'Model Comparison: {model1} vs {model2}', fontsize=24, fontweight='bold', pad=30)  # Larger title like heatmap
        ax.set_xlabel('Metric', fontsize=20, fontweight='bold', labelpad=18)  # Larger labels like heatmap
        ax.set_ylabel('Score', fontsize=20, fontweight='bold', labelpad=25)  # Larger labels like heatmap
        
        # Place metrics at the center of each group with enhanced tick labels
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_display, rotation=30, ha='right', fontsize=16, fontweight='bold')  # More slanted metric names
        
        # Enhance y-axis tick labels
        ax.tick_params(axis='y', labelsize=14, pad=8)  # Larger y-axis labels (fixed: pad instead of labelpad)
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        
        # Ensure y-axis starts at 0 and ends at 1.0 for better comparison
        ax.set_ylim(0, 1.0)
        
        # Add legend with enhanced styling
        ax.legend(title='Model Name', loc='upper right', fontsize=14, title_fontsize=16)  # Larger legend text
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure with sanitized filename
        safe_model1 = self._sanitize_filename(model1)
        safe_model2 = self._sanitize_filename(model2)
        
        # Save the figure with the custom plotting approach
        return self._save_figure(f"model_comparison_{safe_model1}_vs_{safe_model2}", fig=fig, dpi=300, pdf_only=self.pdf_only)

    def all_models_all_metrics(self) -> str:
        """
        Generate a grouped bar chart showing all models and all metrics.
        
        Returns:
            Path to the saved chart image
        """
        # First, let's query the database to get all available model names
        model_query = """
        SELECT DISTINCT m.name 
        FROM llm_models m
        JOIN query_result qr ON qr.llm_model_id = m.id
        """
        
        try:
            model_df = self._fetch_data(model_query)
            print("\nAvailable model names in database:")
            for model in model_df['name']:
                print(f"  - \"{model}\"")
        except Exception as e:
            print(f"Error querying model names: {e}")
        
        # Define all metrics to include - same as radar chart
        metrics = [
            'factual_correctness',
            'semantic_similarity',
            'faithfulness',
            'bleu_score',
            'rogue_score',
            'non_llm_string_similarity',  # Using exact database column name
        ]
        
        # Try to get available columns from the database
        try:
            schema_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'evaluation_metrics'
            """
            
            available_columns = self._fetch_data(schema_query)
            available_metrics = [col[0] for col in available_columns.values]
            
            # Filter metrics to only include those that exist
            valid_metrics = [m for m in metrics if m in available_metrics]
            
            if not valid_metrics:
                logger.error("No valid metrics found in database")
                raise ValueError("No valid metrics available")
                
            # Build query with only valid metrics
            metrics_str = ', '.join([f'AVG(em.{metric}) AS {metric}' for metric in valid_metrics])
            
            # Update metrics list to what's actually available
            metrics = valid_metrics
        except Exception as e:
            logger.error(f"Error querying schema: {e}")
            # If schema query fails, use default metrics approach
            metrics_str = ', '.join([f'AVG(em.{metric}) AS {metric}' for metric in metrics])
        
        # Define the query
        query = f"""
        SELECT 
            m.name AS model_name,
            {metrics_str},
            COUNT(*) AS query_count
        FROM 
            query_result qr
            JOIN llm_models m ON qr.llm_model_id = m.id
            JOIN query_evaluation qe ON qr.id = qe.query_result_id
            JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
        GROUP BY 
            m.name
        ORDER BY 
            AVG(em.factual_correctness) DESC
        """
        
        # Fetch the data
        df = self._fetch_data(query)
        
        if df.empty:
            logger.warning("No data found")
            return "No data available"
        
        # Apply model name shortening
        df['model_name'] = df['model_name'].apply(self._shorten_model_name)
        
        # Ensure all data is numeric and handle NaN values
        for col in metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        # Ensure very small values are preserved with a minimum display value
        for col in metrics:
            if col in df.columns:
                # Replace extremely small values (but > 0) with a minimum visible value
                df[col] = df[col].apply(lambda x: max(0.001, x) if x > 0 else 0)
        
        # Reshape data for grouped bar chart
        df_melted = pd.melt(
            df, 
            id_vars=['model_name', 'query_count'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Score'
        )
        
        # Format metric names for display with longer, more descriptive names
        df_melted['Metric'] = df_melted['Metric'].apply(
            lambda x: 'ROUGE Similarity Score' if x == 'rogue_score' 
            else ('BLEU Similarity Score' if x == 'bleu_score' 
            else ('Non-LLM String Similarity' if 'string_similarity' in x.lower()
            else ('Semantic Similarity Score' if x == 'semantic_similarity'
            else ('Factual Correctness Score' if x == 'factual_correctness'
            else ('Faithfulness Score' if x == 'faithfulness'
            else ' '.join(word.capitalize() for word in x.split('_')))))))
        )
        
        # Close any existing figures
        plt.close('all')
        
        # Calculate dynamic figure size based on number of models and metrics - expanded for longer metric names
        model_count = len(df['model_name'].unique())
        metric_count = len(metrics)
        
        # Base width plus additional width for more metrics and longer names
        width = max(20, 16 + 0.6 * metric_count)  # Increased base width for longer metric names
        # Base height plus additional height for more models
        height = max(12, 8 + 0.45 * model_count)
        
        # Adjust cell size as question count increases
        cell_size_factor = 1.0 if metric_count <= 8 else (1.0 - min(0.3, (metric_count - 8) * 0.02))
        
        # Calculate appropriate figure sizes and spacing
        # Add extra space for legend and longer metric names
        height_with_legend = height + 1.2  # More space for larger legend
        
        # Set up the figure with a high-resolution DPI for better scaling and extra space
        fig = plt.figure(figsize=(width * cell_size_factor, height_with_legend), facecolor='white', dpi=200)
        
        # Create a gridspec for better layout control (like radar chart)
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 10])  # Title, legend, chart
        
        # Add title in separate subplot
        ax_title = fig.add_subplot(gs[0])
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, 'Model Performance Across All Metrics', 
                     fontsize=24, fontweight='bold', ha='center', va='center')
        
        # Add legend in separate subplot
        ax_legend = fig.add_subplot(gs[1])
        ax_legend.axis('off')
        
        # Create main chart in the bottom subplot
        ax = fig.add_subplot(gs[2])
        
        # Create a vibrant and diverse color palette for better distinction between metrics
        # Define the number of distinct colors needed (one per metric)
        n_colors = len(metrics)
        
        # Create a vibrant and diverse color palette instead of just blues
        # This palette provides better distinction between metrics
        vibrant_colors = [
            "#3498db",  # Blue
            "#e74c3c",  # Red
            "#2ecc71",  # Green
            "#9b59b6",  # Purple
            "#f39c12",  # Orange
            "#1abc9c",  # Turquoise
            "#d35400",  # Dark Orange
            "#34495e",  # Navy Blue
            "#16a085",  # Teal
            "#c0392b",  # Dark Red
            "#8e44ad",  # Violet
            "#27ae60"   # Emerald
        ]
        
        # Extend the palette if we have more metrics than colors
        if n_colors > len(vibrant_colors):
            vibrant_colors = vibrant_colors * (n_colors // len(vibrant_colors) + 1)
        
        # Use only as many colors as needed
        custom_colors = vibrant_colors[:n_colors]
        
        # Create grouped bar chart with custom colors
        sns.barplot(
            data=df_melted,
            x='model_name',
            y='Score',
            hue='Metric',
            palette=custom_colors,
            saturation=0.95,
            dodge=True,
            alpha=0.9,
            ax=ax  # Use the specific axis
        )
        
        # Add enhanced axis labels
        ax.set_xlabel('Model', fontsize=20, fontweight='bold', labelpad=18)  # Larger labels like heatmap
        ax.set_ylabel('Average Score', fontsize=26, fontweight='bold', labelpad=25)  # Much larger y-axis label (increased from 20)
        
        # Dynamic x-label rotation based on number of models with enhanced styling
        # Use slanted labels for better readability with larger font
        plt.xticks(rotation=25, ha='right', fontsize=22, fontweight='bold')  # Much larger, slanted model names (increased from 16)
        
        # Improve y-axis ticks with enhanced styling
        plt.yticks(fontsize=18, fontweight='bold')  # Larger y-axis labels (increased from 14)
        
        # Add the number of queries per model with better formatting and enhanced styling
        model_positions = {model: i for i, model in enumerate(df['model_name'].unique())}
        for model, count in zip(df['model_name'].unique(), df['query_count'].unique()):
            # Add counts at the bottom of the chart with improved visual appeal
            ax.annotate(f'n={count}',
                      xy=(model_positions[model], -0.02),
                      xytext=(0, -10),  # Offset below x-axis
                      textcoords='offset points',
                      ha='center', va='top',
                      fontsize=14, fontweight='bold',  # Larger, bolder text like heatmap
                      color='#555555',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
        
        # Remove legend from main chart and add it to the dedicated legend subplot
        ax.get_legend().remove()  # Remove the automatic legend
        
        # Get legend elements from the chart
        handles, labels = ax.get_legend_handles_labels()
        
        # Create legend in the dedicated legend subplot
        legend = ax_legend.legend(
            handles, labels,
            title='Evaluation Metrics', 
            loc='center',                    # Center the legend
            ncol=min(3, len(labels)),       # Fewer columns to accommodate longer names
            frameon=True,
            fontsize=22,  # Much larger legend text (increased from 18)
            title_fontsize=24,  # Much larger legend title (increased from 20)
            framealpha=0.95,
            edgecolor='#cccccc'
        )
        
        # Make legend text bold manually
        for text in legend.get_texts():
            text.set_fontweight('bold')
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        
        # Ensure all bars are visible by adjusting y-axis to start at 0
        # Add 10% padding at the top for labels
        max_score = df_melted['Score'].max()
        ax.set_ylim(0, min(1.0, max_score * 1.15))
        
        # Add value labels on top of each bar with improved formatting and enhanced styling like heatmap
        for p in ax.patches:
            height = p.get_height()
            # Set minimum display height for very small values but keep actual data value
            display_height = max(0.01, height) if height > 0 else 0
            p.set_height(display_height)
            
            # Only label bars with significant height
            if height > 0:  # Label all non-zero values
                ax.annotate(
                    f'{height:.3f}', 
                    xy=(p.get_x() + p.get_width() / 2, display_height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=24, fontweight='bold',  # Much larger, bolder text for data values (increased from 16)
                    color='#333333'
                )
        
        # Add border around the plot for better definition
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#333333')
        
        # Use tight_layout for proper spacing with gridspec
        plt.tight_layout()
        
        # Save the figure with high DPI for better quality
        return self._save_figure("all_models_all_metrics", dpi=300, pdf_only=self.pdf_only)

    def enhanced_metrics_radar_chart(self) -> str:
        """
        Generate an optimized radar chart showing all models and all metrics.
        The chart will display average scores for each model across all evaluation metrics.
        
        Returns:
            Path to the saved chart image
        """
        # Include all metrics from the table
        metrics = [
            'factual_correctness',   # Measures the factual accuracy
            'semantic_similarity',    # Measures semantic similarity to reference
            'faithfulness',          # Measures answer's faithfulness to the context
            'bleu_score',            # BLEU Score for lexical similarity
            'non_llm_string_similarity',      # String similarity measure (exact column name)
            'rogue_score',           # ROUGE Score for summary evaluation
        ]
        
        # Define the query with error handling for missing columns
        base_metrics = []
        for metric in metrics:
            base_metrics.append(f"AVG(CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'evaluation_metrics' AND column_name = '{metric}') THEN em.{metric} ELSE NULL END) AS {metric}")
        
        metrics_str = ', '.join(base_metrics)
        
        query = f"""
        SELECT 
            m.name AS model_name,
            {metrics_str},
            COUNT(*) AS query_count
        FROM 
            query_result qr
            JOIN llm_models m ON qr.llm_model_id = m.id
            JOIN query_evaluation qe ON qr.id = qe.query_result_id
            JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
        GROUP BY 
            m.name
        ORDER BY 
            AVG(em.factual_correctness) DESC
        """
        
        # Fetch the data
        try:
            df = self._fetch_data(query)
        except Exception as e:
            logger.error(f"Error with full metrics query: {e}")
            # Fallback to query only existing columns
            logger.info("Falling back to query for existing columns only")
            
            # Get available metrics from the database
            schema_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'evaluation_metrics'
            """
            
            available_columns = self._fetch_data(schema_query)
            available_metrics = [col[0] for col in available_columns.values]
            
            # Filter metrics to only include those that exist
            valid_metrics = [m for m in metrics if m in available_metrics]
            
            if not valid_metrics:
                logger.error("No valid metrics found in database")
                return "No valid metrics available"
                
            # Build new query with only valid metrics
            metrics_str = ', '.join([f'AVG(em.{metric}) AS {metric}' for metric in valid_metrics])
            query = f"""
            SELECT 
                m.name AS model_name,
                {metrics_str},
                COUNT(*) AS query_count
            FROM 
                query_result qr
                JOIN llm_models m ON qr.llm_model_id = m.id
                JOIN query_evaluation qe ON qr.id = qe.query_result_id
                JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
            GROUP BY 
                m.name
            ORDER BY 
                AVG(em.factual_correctness) DESC
            """
            
            df = self._fetch_data(query)
            
            # Update metrics list to what's actually available
            metrics = valid_metrics
        
        if df.empty:
            logger.warning("No data found for radar chart")
            return "No data available"
        
        # Apply model name shortening
        df['model_name'] = df['model_name'].apply(self._shorten_model_name)
        
        # Filter out columns that don't exist in the dataframe
        metrics = [m for m in metrics if m in df.columns]
        
        # Convert to numeric and handle NaN values
        for col in metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Format metric names for display
        metric_labels = ['ROUGE Score' if metric == 'rogue_score' else ('BLEU Score' if metric == 'bleu_score' else ' '.join(word.capitalize() for word in metric.split('_'))) for metric in metrics]
        
        # Create figure with white background and more space for the legend
        fig = plt.figure(figsize=(16, 16), facecolor='white')
        
        # Create a gridspec for the radar plot and legend
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 5])
        
        # Add a title at the top of the figure with enhanced styling like heatmap
        ax_title = fig.add_subplot(gs[0])
        ax_title.axis('off')  # Hide axis
        ax_title.text(0.5, 0.5, 'Complete Metrics Comparison', 
                     fontsize=28, fontweight='bold', ha='center', va='center')  # Larger title like heatmap
        
        # Create radar plot
        ax = fig.add_subplot(gs[1], polar=True)
        
        # Set background color
        ax.set_facecolor('white')
        
        # Number of metrics (axes)
        N = len(metrics)
        
        # Calculate angles for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the polygon
        
        # Set the labels for each axis with increased font size
        plt.xticks(angles[:-1], metric_labels, fontsize=18, fontweight='bold')  # Larger, bolder axis labels
        
        # Draw y-axis labels (grid lines) with better visibility
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], 
                   color="black", size=16, fontweight='bold')  # Larger, bolder y-axis labels
        plt.ylim(0, 1)
        
        # Remove the concentric circles for a cleaner look
        ax.grid(False)  # Turn off the default grid
        
        # Add black axis lines from center to each vertex 
        for angle in angles[:-1]:
            ax.plot([0, angle], [0, 1], 'black', linewidth=0.8, alpha=0.5)
            
        # Add concentric circles to create a spider web effect
        for r in [0.2, 0.4, 0.6, 0.8]:
            # Calculate points on the circle
            circle_angles = np.linspace(0, 2*np.pi, 100)
            x = r * np.cos(circle_angles)
            y = r * np.sin(circle_angles)
            
            # Draw the circle as a line
            ax.plot(circle_angles, [r] * len(circle_angles), color='black', linestyle='-', linewidth=0.6, alpha=0.3)
        
        # Define a custom, darker color palette - similar to the second image
        custom_colors = [
            '#1f77b4',  # Blue (nova-pro-v1)
            '#ff7f0e',  # Orange (claude-3.7-sonnet) 
            '#2ca02c',  # Green (gemini-2.5-flash-preview)
            '#17becf',  # Teal (llama-3.1-8b-instruct)
            '#9467bd',  # Purple (llama-3.3-70b-instruct)
            '#8c564b',  # Brown (mistral-8b)
            '#e377c2',  # Pink (gpt-4o-2024-11-20)
            '#7f7f7f',  # Gray (qwen-2.5-72b-instruct)
            '#bcbd22',  # Olive
            '#17becf',  # Teal
        ]
        
        # Create a color mapping dictionary to ensure consistent colors
        model_color_map = {}
        for i, model_name in enumerate(df['model_name']):
            color_idx = i % len(custom_colors)
            model_color_map[model_name] = custom_colors[color_idx]
        
        # Plot each model with enhanced visibility
        for i, (idx, row) in enumerate(df.iterrows()):
            model_name = row['model_name']
            color = model_color_map[model_name]
            
            # Get values for each metric, handling missing metrics
            original_values = [row[metric] if metric in row.index and pd.notna(row[metric]) else 0 for metric in metrics]
            values = original_values.copy()
            
            # Replace very small values with a minimum visible value for better display
            values = [max(0.01, v) if v > 0 else 0 for v in values]
            
            values += values[:1]  # Close the polygon
            original_values += original_values[:1]  # Also extend original values
            
            # Plot the model's polygon with higher line width for better visibility
            ax.plot(angles, values, linewidth=3.5, linestyle='solid', 
                   label=model_name, color=color)
            
            # Fill with semi-transparent color
            ax.fill(angles, values, alpha=0.25, color=color)
            
            # Add data points at each vertex with bigger markers
            for j, value in enumerate(values[:-1]):
                ax.scatter(angles[j], value, s=120, color=color, 
                          edgecolor='black', linewidth=1.5, zorder=10)
                
                # Add value labels at each point for better readability
                # Always show labels for ALL non-zero values
                original_value = original_values[j]
                if original_value > 0:  # Show ALL non-zero values, not just those > 0.05
                    ha = 'left' if angles[j] > np.pi else 'right'
                    va = 'bottom' if angles[j] < np.pi/2 or angles[j] > 3*np.pi/2 else 'top'
                    
                    # Position adjustment for better label placement
                    offset = 0.05
                    x_offset = np.cos(angles[j]) * offset
                    y_offset = np.sin(angles[j]) * offset
                    
                    # Add text with background for better visibility
                    ax.text(angles[j] + x_offset, value + y_offset, f'{original_value:.3f}', 
                           fontsize=12, fontweight='bold', ha=ha, va=va,  # Larger, bolder text like heatmap
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Create legend similar to the second image - horizontal colored boxes above the chart
        handles = []
        legend_labels = []
        
        for model_name in df['model_name']:
            count = int(df.loc[df['model_name'] == model_name, 'query_count'].iloc[0])
            color = model_color_map[model_name]
            # Create a rectangle patch for each model
            handle = plt.Rectangle((0, 0), 1, 1, color=color)
            handles.append(handle)
            legend_labels.append(f"{model_name} (n={count})")
        
        # Create legend without adding it to the plot yet
        legend = ax_title.legend(handles, legend_labels, loc='upper center', 
                               ncol=min(4, len(df)), fontsize=14, frameon=True,  # Larger legend font
                               bbox_to_anchor=(0.5, 0.2))
        
        # Adjust layout for better readability
        plt.tight_layout()
        
        # Save the figure with higher DPI for better quality
        return self._save_figure("enhanced_metrics_radar_chart", dpi=300, pdf_only=self.pdf_only)

    def factual_correctness_matrix(self, max_questions: int = 20, limit_models: int = 8) -> str:
        """
        Generate matrix chart(s) showing factual correctness scores for individual questions across different models.
        Automatically splits into two charts if more than 10 questions: Q1-Q10 and Q11-Q20.
        
        Args:
            max_questions: Maximum number of questions to include in the matrix (default 20)
            limit_models: Maximum number of models to include in the matrix
            
        Returns:
            Path to the saved chart image(s)
        """
        # Query to get ALL results (not just latest) for each model and question
        query = """
        SELECT 
            qr.query,
            m.name AS model_name,
            em.factual_correctness
        FROM 
            query_result qr
            JOIN llm_models m ON qr.llm_model_id = m.id
            JOIN query_evaluation qe ON qr.id = qe.query_result_id
            JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
        WHERE
            em.factual_correctness IS NOT NULL
        ORDER BY 
            m.name, qr.query
        """
        
        # Fetch the data
        try:
            df = self._fetch_data(query)
        except Exception as e:
            logger.error(f"Error fetching matrix data: {e}")
            return "Error fetching data"
        
        if df.empty:
            logger.warning("No data found for factual correctness matrix")
            return "No data available"
        
        # Apply model name shortening
        df['model_name'] = df['model_name'].apply(self._shorten_model_name)
        
        # Ensure all data is numeric
        df['factual_correctness'] = pd.to_numeric(df['factual_correctness'], errors='coerce').fillna(0)
        
        # Calculate average factual correctness for each model/question pair
        avg_df = df.groupby(['model_name', 'query'])['factual_correctness'].mean().reset_index()
        
        # Load the test cases to get the official test numbers
        try:
            test_cases_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                        'app', 'ragas', 'test_cases', 'synthetic_test_cases.json')
            with open(test_cases_path, 'r') as f:
                test_cases = json.load(f)
            
            # Create a mapping from query text to test_no
            query_to_test_no = {}
            for test in test_cases:
                if 'query' in test and 'test_no' in test:
                    query_to_test_no[test['query']] = test['test_no']
            
            # Add test_no to the dataframe
            avg_df['test_no'] = avg_df['query'].apply(lambda q: query_to_test_no.get(q, 999))
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
            # If we can't load the test cases, create a column with sequential numbers
            unique_queries = avg_df['query'].unique()
            query_to_test_no = {q: i+1 for i, q in enumerate(unique_queries)}
            avg_df['test_no'] = avg_df['query'].apply(lambda q: query_to_test_no.get(q, 999))
        
        # Get top models based on average factual correctness
        model_avg_scores = avg_df.groupby('model_name')['factual_correctness'].mean().reset_index()
        model_avg_scores = model_avg_scores.sort_values('factual_correctness', ascending=False)
        top_models = model_avg_scores.head(limit_models)['model_name'].tolist()
        
        # Filter data for just those models
        avg_df = avg_df[avg_df['model_name'].isin(top_models)]
        
        # Get questions sorted by test_no
        question_test_no = avg_df[['query', 'test_no']].drop_duplicates()
        question_test_no = question_test_no.sort_values('test_no')
        all_questions = question_test_no.head(max_questions)['query'].tolist()
        
        # Determine if we need to split into two charts
        if len(all_questions) > 10:
            # Split into two charts: Q1-Q10 and Q11-Q20
            first_10_questions = all_questions[:10]
            remaining_questions = all_questions[10:] if len(all_questions) > 10 else []
            
            # Generate first chart (Q1-Q10)
            chart1_path = self._generate_single_matrix_chart(
                avg_df, first_10_questions, query_to_test_no, top_models, 
                chart_suffix="_part1", title_suffix=" (Q1-Q10)"
            )
            
            # Generate second chart (Q11-Q20) if there are remaining questions
            if remaining_questions:
                chart2_path = self._generate_single_matrix_chart(
                    avg_df, remaining_questions, query_to_test_no, top_models, 
                    chart_suffix="_part2", title_suffix=f" (Q11-Q{len(all_questions)})"
                )
                return f"Charts saved: {chart1_path} and {chart2_path}"
            else:
                return chart1_path
        else:
            # Single chart for 10 or fewer questions
            return self._generate_single_matrix_chart(
                avg_df, all_questions, query_to_test_no, top_models
            )
    
    def _generate_single_matrix_chart(self, avg_df, questions, query_to_test_no, top_models, 
                                    chart_suffix="", title_suffix="") -> str:
        """
        Helper method to generate a single factual correctness matrix chart.
        
        Args:
            avg_df: DataFrame with averaged factual correctness data
            questions: List of questions to include in this chart
            query_to_test_no: Mapping from query text to test number
            top_models: List of top performing models
            chart_suffix: Suffix to add to the filename (e.g., "_part1")
            title_suffix: Suffix to add to the title (e.g., " (Q1-Q10)")
            
        Returns:
            Path to the saved chart image
        """
        # Filter data for just those models and questions
        filtered_df = avg_df[avg_df['query'].isin(questions)]
        
        # Create simplified question labels (Q1, Q2, etc.) that match the test_no
        question_display = {}
        for q in questions:
            test_no = query_to_test_no.get(q, 0)
            if test_no > 0 and test_no < 100:  # Reasonable test_no range
                question_display[q] = f"Q{test_no}"
            else:
                # Fallback to order in list if test_no is missing or invalid
                # For part 2 charts, we need to add offset based on the chart suffix
                base_offset = 10 if chart_suffix == "_part2" else 0
                question_display[q] = f"Q{questions.index(q) + 1 + base_offset}"
        
        # Pivot the data to create the matrix using the averaged values
        matrix_df = filtered_df.pivot(index='model_name', columns='query', values='factual_correctness')
        
        # Replace column names with Q1, Q2, etc. labels
        matrix_df = matrix_df.rename(columns=question_display)
        
        # Reorder columns by test number
        ordered_columns = sorted(matrix_df.columns, key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
        matrix_df = matrix_df[ordered_columns]
        
        # Create figure with enhanced sizing to prevent overlapping model names
        question_count = len(matrix_df.columns)
        model_count = len(matrix_df.index)
        
        # Significantly increased height calculations to prevent model name overlapping
        width = max(18, 12 + 0.7 * question_count)  # Increased base width 
        height = max(16, 10 + 1.2 * model_count)    # Significantly increased height for model names
        
        # Adjust cell size as question count increases
        cell_size_factor = 1.0 if question_count <= 8 else (1.0 - min(0.3, (question_count - 8) * 0.02))
        
        # Set up the figure with larger size and higher DPI for better quality
        fig = plt.figure(figsize=(width * cell_size_factor, height), facecolor='white', dpi=250)  # Higher DPI

        # Create custom colormap with stronger contrast for better visibility of differences
        # Use a blue gradient with darker colors for values over 0.99
        strong_blue_colors = ["#ffffff", "#f0f7ff", "#e3f2fd", "#c8e6fc", "#bbdefb", "#9fd2f8", "#90caf9", "#64b5f6", "#42a5f5", "#1976d2"]
        custom_light_cmap = LinearSegmentedColormap.from_list("strong_blues", strong_blue_colors)

        # Create the heatmap with enhanced settings for better readability
        ax = sns.heatmap(
            matrix_df,
            annot=True,
            cmap=custom_light_cmap,
            vmin=0,
            vmax=1,
            linewidths=max(0.8, 1.2 * cell_size_factor),  # Thicker lines for better separation
            linecolor='white',
            fmt='.3f',
            annot_kws={"color": "black", 
                      "fontweight": "bold",
                      "fontsize": max(14, 16 * cell_size_factor)},  # Significantly larger text for better readability
            cbar=False,  # Remove the colorbar completely
            square=True   # Force square cells
        )
        
        # Enhanced spacing and appearance for better model name visibility
        ax.set_aspect('equal', adjustable='box')
        
        # Customize the appearance with enhanced fonts and spacing like heatmap
        plt.title(f'Average Factual Correctness Score by Model and Question{title_suffix}', 
                 fontsize=24,  # Larger title font like heatmap (increased from 20)
                 fontweight='bold',
                 pad=30)  # More padding
        plt.xlabel('Questions', fontsize=20, fontweight='bold', labelpad=18)  # Larger font with padding like heatmap
        plt.ylabel('Models', fontsize=20, fontweight='bold', labelpad=25)     # Larger font with more padding like heatmap
        
        # Significantly improve y-axis labels (model names) to prevent overlapping
        plt.yticks(fontsize=16, fontweight='bold', rotation=0)  # Larger, horizontal model names like heatmap
        
        # Dynamic label rotation based on number of questions with enhanced styling
        # Use horizontal for few questions, angled for many
        if question_count <= 8:
            plt.xticks(rotation=0, ha='center', fontsize=18, fontweight='bold')  # Larger Q1-Q20 labels (increased from 16)
        else:
            # Use rotated labels for larger question counts
            rotation = min(45, max(20, question_count * 1.5))  # Scale rotation with question count
            plt.xticks(rotation=rotation, ha='right', fontsize=18, fontweight='bold')  # Larger Q1-Q20 labels (increased from 13)
        
        # Use tight_layout for optimal spacing
        plt.tight_layout()
        
        # Save the figure with high DPI for better quality
        filename = f"factual_correctness_matrix{chart_suffix}"
        return self._save_figure(filename, fig=fig, dpi=300, pdf_only=self.pdf_only) 