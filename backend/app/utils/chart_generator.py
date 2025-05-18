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
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the chart generator.
        
        Args:
            output_dir: Directory to save charts to. Defaults to 'output/charts' in the project root.
        """
        if output_dir is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent.parent
            self.output_dir = project_root / "output" / "charts"
        else:
            self.output_dir = Path(output_dir)
            
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
    
    def _save_figure(self, filename: str, fig: plt.Figure = None, dpi=300) -> str:
        """
        Save the current figure or a provided figure to the output directory.
        
        Args:
            filename: Filename for the saved chart (without extension)
            fig: Figure to save, if None the current figure is saved
            dpi: Resolution for the saved image
            
        Returns:
            Path to the saved file
        """
        file_path = self.output_dir / f"{filename}.png"
        
        if fig:
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        else:
            plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
            
        plt.close()
        return str(file_path)
    
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
        formatted_metric = ' '.join(word.capitalize() for word in metric_name.split('_'))
        ax.set_title(f'Average {formatted_metric} Score by Model', fontsize=20, fontweight='bold')
        ax.set_xlabel('Model', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'Average {formatted_metric}', fontsize=16, fontweight='bold')
        
        # Add grid lines behind the bars
        ax.grid(axis='y', linestyle='-', alpha=0.2, color='gray', zorder=0)
        
        # Set x-tick positions and labels with proper rotation
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=13)
        
        # Set y-axis limits for better visualization
        max_score = max(scores)
        ax.set_ylim(0, min(1.0, max_score * 1.1))  # Cap at 1.0 or 10% above max score
        
        # Improve axis appearance
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # Thicker border
            spine.set_color('#333333')  # Darker border for better definition
        
        # Add data labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 5),  # 5 points vertical offset for better spacing
                      textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=13, fontweight='bold', zorder=10,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))  # Add subtle text background
        
        # Add the number of queries per model at the bottom of each bar
        for i, count in enumerate(query_counts):
            ax.annotate(f'n={count}',
                      xy=(x[i], 0.01),
                       ha='center', va='bottom',
                       fontsize=11, color='#555555', zorder=10, fontweight='bold')
        
        # Add more padding at the bottom for model names
        plt.subplots_adjust(bottom=0.25)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure with higher resolution
        return self._save_figure(f"model_comparison_{metric_name}", fig=fig, dpi=300)
    
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
        metric_labels = [' '.join(word.capitalize() for word in metric.split('_')) for metric in metrics]
        
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
            plt.title(f'Metrics Radar Chart for {model_name} (No Data)', fontsize=18, fontweight='bold', y=1.05)
            
            # Log a warning
            logger.warning(f"No data found for model: {model_name}")
            
            # Save the figure with sanitized filename
            safe_model_name = self._sanitize_filename(model_name)
            return self._save_figure(f"radar_chart_{safe_model_name}_no_data", dpi=300)
        
        # Get the query count
        query_count = int(df['query_count'].iloc[0])
        
        # Convert any remaining Decimal values to float
        for col in df.columns:
            if col != 'query_count' and col in metrics:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Process the data for radar chart - handle any missing columns
        values = [df.iloc[0][metric] if metric in df.columns else 0 for metric in metrics]
        
        # Add values to complete the polygon
        values += values[:1]
        
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
        ax.plot(angles, values, linewidth=3, linestyle='solid', color=model_color)
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
            
            # Add text with background for better visibility
            ax.text(angles[i] + x_offset, value + y_offset, f'{value:.2f}', 
                   fontsize=12, fontweight='bold', ha=ha, va=va,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Add title
        plt.title(f'Metrics Radar Chart for {model_name} (n={query_count})', fontsize=18, fontweight='bold', y=1.05)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure with sanitized filename
        safe_model_name = self._sanitize_filename(model_name)
        return self._save_figure(f"radar_chart_{safe_model_name}", dpi=300)
    
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
        
        # Set model name as index
        df = df.set_index('model_name')
        
        # Ensure all data is numeric
        df = df.astype(float)
        
        # Create figure with white background for better visibility
        plt.figure(figsize=(14, 10), facecolor='white')
        
        # Generate heatmap with improved styling
        ax = sns.heatmap(
            df,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            linewidths=.5,
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Score'}
        )
        
        # Format metric names for display
        formatted_metrics = [' '.join(word.capitalize() for word in metric.split('_')) for metric in metrics]
        
        # Add labels and title
        plt.title('Model Performance Across All RAGAS Metrics', fontsize=18, fontweight='bold')
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Models', fontsize=14)
        
        # Set x-axis labels to formatted metrics
        ax.set_xticklabels(formatted_metrics, rotation=45, ha='right', fontsize=11)
        
        # Set y-axis labels (model names) with larger font
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
        
        # Adjust layout for better display
        plt.tight_layout()
        
        # Save the figure with higher DPI
        return self._save_figure("metrics_heatmap", dpi=300)
    
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
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Create horizontal bar chart
        ax = sns.barplot(
            data=df,
            y='short_query',
            x='score',
            orient='h'
        )
        
        # Format metric name for display
        formatted_metric = ' '.join(word.capitalize() for word in metric_name.split('_'))
        
        # Add labels and title
        plt.title(f'{formatted_metric} Scores by Query for {model_name}', fontsize=16)
        plt.xlabel(f'{formatted_metric} Score', fontsize=14)
        plt.ylabel('Query', fontsize=14)
        
        # Add data labels
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.2f}', 
                      (p.get_width(), p.get_y() + p.get_height() / 2),
                      ha='left', va='center',
                      fontsize=10, xytext=(5, 0),
                      textcoords='offset points')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure with sanitized filename
        safe_model_name = self._sanitize_filename(model_name)
        return self._save_figure(f"query_performance_{safe_model_name}_{metric_name}")
    
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
            lambda x: ' '.join(word.capitalize() for word in x.split('_'))
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
        
        # Custom bar plot to replace seaborn's barplot
        for i, model in enumerate(models):
            model_data = df_melted[df_melted['model_name'] == model]
            
            # Get scores for each metric
            scores = [model_data[model_data['Metric'] == metric]['Score'].values[0] 
                     for metric in metrics_display]
            
            # Plot bars with offset positions
            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, scores, width, label=model, color=colors[i], zorder=5)
            
            # Add text annotations for each bar
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=10, fontweight='bold', zorder=10)
        
        # Configure the rest of the chart
        ax.set_title(f'Model Comparison: {model1} vs {model2}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        
        # Place metrics at the center of each group
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_display)
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        
        # Ensure y-axis starts at 0 and ends at 1.0 for better comparison
        ax.set_ylim(0, 1.0)
        
        # Add legend
        ax.legend(title='Model Name', loc='upper right')
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure with sanitized filename
        safe_model1 = self._sanitize_filename(model1)
        safe_model2 = self._sanitize_filename(model2)
        
        # Save the figure with the custom plotting approach
        return self._save_figure(f"model_comparison_{safe_model1}_vs_{safe_model2}", fig=fig, dpi=300)

    def all_models_all_metrics(self) -> str:
        """
        Generate a comprehensive grouped bar chart showing all models and all metrics.
        
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
        
        # Define all RAGAS metrics to include - same as radar chart
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
            
            # Print available columns for debugging
            print("Available columns in evaluation_metrics table:")
            for col in available_metrics:
                print(f"  - {col}")
            
            # Check for string similarity-related columns with flexible matching
            string_sim_variants = ['string_similarity', 'stringsimilarity', 'str_similarity', 'similarity_string']
            string_sim_cols = [col for col in available_metrics if any(variant in col.lower() for variant in string_sim_variants)]
            
            if string_sim_cols:
                print(f"Found potential string similarity columns: {string_sim_cols}")
                # Replace our standard name with the actual column name from database
                metrics = [string_sim_cols[0] if m == 'string_similarity' else m for m in metrics]
            
            # Filter metrics to only include those that exist
            valid_metrics = [m for m in metrics if m in available_metrics]
            
            print(f"Using these metrics: {valid_metrics}")
            
            if not valid_metrics:
                logger.error("No valid metrics found in database")
                raise ValueError("No valid metrics available")
                
            # Build query with only valid metrics
            metrics_str = ', '.join([f'AVG(em.{metric}) AS {metric}' for metric in valid_metrics])
            
            # Update metrics list to what's actually available
            metrics = valid_metrics
        except Exception as e:
            logger.error(f"Error querying schema: {e}")
            print(f"Schema query error: {e}")
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
        
        # Print the dataframe columns to verify what data we have
        print("\nColumns in returned data:")
        print(df.columns.tolist())
        
        # Check if any string similarity column is present
        sim_cols = [col for col in df.columns if 'similarity' in col.lower() and 'semantic' not in col.lower()]
        if sim_cols:
            print(f"Found similarity columns in results: {sim_cols}")
            
            # If we found a string similarity column but it's not in our metrics list, add it
            for col in sim_cols:
                if col not in metrics and 'string' in col.lower():
                    print(f"Adding missing string similarity column: {col}")
                    metrics.append(col)
        
        # Ensure all data is numeric and handle NaN values
        for col in metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Reshape data for grouped bar chart
        df_melted = pd.melt(
            df, 
            id_vars=['model_name', 'query_count'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Score'
        )
        
        # Format metric names for display
        df_melted['Metric'] = df_melted['Metric'].apply(
            lambda x: ' '.join(word.capitalize() for word in x.split('_'))
        )
        
        # Create figure
        plt.figure(figsize=(16, 10), facecolor='white')
        
        # Use a more distinct color palette
        palette = plt.cm.Dark2(np.linspace(0, 1, len(metrics)))
        
        # Create grouped bar chart
        ax = sns.barplot(
            data=df_melted,
            x='model_name',
            y='Score',
            hue='Metric',
            palette=palette
        )
        
        # Add labels and title
        plt.title('Comprehensive Model Performance Across All RAGAS Metrics', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add the number of queries per model
        model_positions = {model: i for i, model in enumerate(df['model_name'].unique())}
        for model, count in zip(df['model_name'].unique(), df['query_count'].unique()):
            ax.annotate(f'n={count}',
                      (model_positions[model], 0.02),
                      ha='center', va='bottom',
                      fontsize=8, color='gray')
        
        # Add legend outside the plot with better formatting
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        
        # Add grid for easier reading
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ensure all bars are visible by adjusting y-axis to start at 0
        plt.ylim(0, 1.0)
        
        # Add value labels on top of each bar
        for p in ax.patches:
            if p.get_height() > 0.05:  # Only label bars with significant height
                ax.annotate(f'{p.get_height():.2f}', 
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='bottom',
                          fontsize=7, rotation=90)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        return self._save_figure("all_models_all_metrics", dpi=300)
        
    def ragas_radar_chart(self) -> str:
        """
        Generate an optimized radar chart showing all models and all RAGAS metrics.
        The chart will display average scores for each model across all RAGAS evaluation metrics.
        
        Returns:
            Path to the saved chart image
        """
        # Include all RAGAS metrics from the table
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
            logger.warning("No data found for RAGAS radar chart")
            return "No data available"
        
        # Filter out columns that don't exist in the dataframe
        metrics = [m for m in metrics if m in df.columns]
        
        # Convert to numeric and handle NaN values
        for col in metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Format metric names for display
        metric_labels = [' '.join(word.capitalize() for word in metric.split('_')) for metric in metrics]
        
        # Create figure with white background and more space for the legend
        fig = plt.figure(figsize=(16, 16), facecolor='white')
        
        # Create a gridspec for the radar plot and legend
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 5])
        
        # Add a title at the top of the figure
        ax_title = fig.add_subplot(gs[0])
        ax_title.axis('off')  # Hide axis
        ax_title.text(0.5, 0.5, 'Complete RAGAS Metrics Comparison', 
                     fontsize=24, fontweight='bold', ha='center', va='center')
        
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
        plt.xticks(angles[:-1], metric_labels, fontsize=16, fontweight='bold')
        
        # Draw y-axis labels (grid lines) with better visibility
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], 
                   color="black", size=14)
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
            values = [row[metric] if metric in row.index and pd.notna(row[metric]) else 0 for metric in metrics]
            values += values[:1]  # Close the polygon
            
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
                if value > 0.05:  # Only show values that are significant
                    ha = 'left' if angles[j] > np.pi else 'right'
                    va = 'bottom' if angles[j] < np.pi/2 or angles[j] > 3*np.pi/2 else 'top'
                    
                    # Position adjustment for better label placement
                    offset = 0.05
                    x_offset = np.cos(angles[j]) * offset
                    y_offset = np.sin(angles[j]) * offset
                    
                    # Add text with background for better visibility
                    ax.text(angles[j] + x_offset, value + y_offset, f'{value:.2f}', 
                           fontsize=10, fontweight='bold', ha=ha, va=va,
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
                               ncol=min(4, len(df)), fontsize=12, frameon=True,
                               bbox_to_anchor=(0.5, 0.2))
        
        # Adjust layout for better readability
        plt.tight_layout()
        
        # Save the figure with higher DPI for better quality
        return self._save_figure("enhanced_ragas_metrics_radar_chart", dpi=300)

    def factual_correctness_matrix(self, max_questions: int = 8, limit_models: int = 8) -> str:
        """
        Generate a matrix chart showing factual correctness scores for individual questions across different models.
        This displays the actual scores for each question rather than averages.
        
        Args:
            max_questions: Maximum number of questions to include in the matrix
            limit_models: Maximum number of models to include in the matrix
            
        Returns:
            Path to the saved chart image
        """
        # First, get the questions and latest responses from each model
        query = """
        WITH latest_results AS (
            SELECT 
                qr.query,
                m.name AS model_name,
                em.factual_correctness,
                ROW_NUMBER() OVER (PARTITION BY qr.query, m.name ORDER BY qr.timestamp DESC) as rn
            FROM 
                query_result qr
                JOIN llm_models m ON qr.llm_model_id = m.id
                JOIN query_evaluation qe ON qr.id = qe.query_result_id
                JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
            WHERE
                em.factual_correctness IS NOT NULL
        )
        SELECT 
            query, 
            model_name, 
            factual_correctness
        FROM 
            latest_results
        WHERE 
            rn = 1
        ORDER BY 
            model_name, query
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
        
        # Ensure all data is numeric
        df['factual_correctness'] = pd.to_numeric(df['factual_correctness'], errors='coerce').fillna(0)
        
        # Select top models based on average factual correctness
        top_models_query = """
        SELECT 
            m.name AS model_name,
            AVG(em.factual_correctness) AS avg_score,
            COUNT(DISTINCT qr.query) AS question_count
        FROM 
            query_result qr
            JOIN llm_models m ON qr.llm_model_id = m.id
            JOIN query_evaluation qe ON qr.id = qe.query_result_id
            JOIN evaluation_metrics em ON qe.evaluation_metrics_id = em.id
        WHERE
            em.factual_correctness IS NOT NULL
        GROUP BY 
            m.name
        ORDER BY 
            avg_score DESC
        LIMIT {}
        """.format(limit_models)
        
        top_models = self._fetch_data(top_models_query)
        model_list = top_models['model_name'].tolist()
        
        # Filter for just those models
        df = df[df['model_name'].isin(model_list)]
        
        # Limit questions - get the most common questions across all models
        question_counts = df['query'].value_counts().reset_index()
        question_counts.columns = ['query', 'count']
        common_questions = question_counts.head(max_questions)['query'].tolist()
        
        # Create simplified question labels (Q1, Q2, etc.)
        question_display = {}
        for i, q in enumerate(common_questions):
            question_display[q] = f"Q{i+1}"
        
        # Filter for just those questions
        df = df[df['query'].isin(common_questions)]
        
        # Pivot the data to create the matrix
        matrix_df = df.pivot(index='model_name', columns='query', values='factual_correctness')
        
        # Replace column names with Q1, Q2, etc. labels
        matrix_df = matrix_df.rename(columns=question_display)
        
        # Create figure with size based on number of questions
        question_count = len(matrix_df.columns)
        model_count = len(matrix_df.index)
        
        width = max(10, 8 + 0.4 * question_count)  # Base width plus adjustment for questions
        height = max(8, 6 + 0.3 * model_count)    # Base height plus adjustment for models
        
        plt.figure(figsize=(width, height), facecolor='white')
        
        # Create the heatmap with a color gradient
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        
        # Create the heatmap
        ax = sns.heatmap(
            matrix_df,
            annot=True,
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=1,
            linecolor='white',
            fmt='.2f',
            cbar_kws={'label': 'Factual Correctness Score'}
        )
        
        # Customize the appearance
        plt.title('RAGAS Factual Correctness Score by Model and Question', fontsize=16, fontweight='bold')
        plt.xlabel('Questions', fontsize=14)
        plt.ylabel('Models', fontsize=14)
        
        # Use horizontal alignment for x tick labels
        plt.xticks(rotation=0, ha='center', fontsize=12, fontweight='bold')
        
        # Add a legend box under the questions
        legend_handles = []
        legend_labels = []
        
        # Create custom color squares for the legend
        legend_colors = [
            (cmap(0.1), '0.0 - 0.2'),
            (cmap(0.3), '0.2 - 0.4'),
            (cmap(0.5), '0.4 - 0.6'),
            (cmap(0.7), '0.6 - 0.8'),
            (cmap(0.9), '0.8 - 1.0')
        ]
        
        for color, label in legend_colors:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            legend_labels.append(label)
        
        # Position the legend below the matrix
        legend = plt.legend(
            legend_handles, 
            legend_labels,
            title="Legend",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),  # Move legend further down
            ncol=len(legend_colors),
            frameon=True
        )
        
        # Adjust layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)  # Increase bottom padding significantly
        
        # Save the figure with higher DPI for better quality
        return self._save_figure("factual_correctness_matrix", dpi=300) 