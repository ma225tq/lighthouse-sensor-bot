"""
Database configuration for chart generation.
This file provides connection parameters for PostgreSQL database access.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Database connection parameters from environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'dbname': os.getenv('DB_NAME', 'postgres'), 
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'port': os.getenv('DB_PORT', '5432')
}

def get_connection_params():
    """
    Returns database connection parameters from environment variables.
    Falls back to default values if environment variables are not set.
    """
    return DB_CONFIG 