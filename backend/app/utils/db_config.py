"""
Database configuration for chart generation.
This file provides connection parameters for PostgreSQL database access.
"""

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',  # Use localhost for local development
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'password',
    'port': '5432'
}

def get_connection_params():
    """
    Returns database connection parameters.
    You can modify these values to match your local PostgreSQL configuration.
    """
    return DB_CONFIG 