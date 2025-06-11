#!/usr/bin/env python3
"""
Script to load ferry trips data from CSV into PostgreSQL database
"""

import os
import csv
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_database_connection():
    """Create database connection using environment variables"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def create_ferry_trips_table(cursor):
    """Create the ferry_trips table with appropriate schema"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS ferry_trips (
        id SERIAL PRIMARY KEY,
        route_id INTEGER,
        route_name VARCHAR(255),
        ferry_name VARCHAR(255),
        ferry_id INTEGER,
        terminal_departure VARCHAR(255),
        terminal_arrival VARCHAR(255),
        time_departure TIMESTAMP,
        cars_outbound INTEGER,
        trucks_outbound INTEGER,
        trucks_with_trailer_outbound INTEGER,
        motorcycles_outbound INTEGER,
        exemption_vehicles_outbound INTEGER,
        pedestrians_outbound INTEGER,
        buses_outbound INTEGER,
        vehicles_left_at_terminal_outbound INTEGER,
        cars_inbound INTEGER,
        trucks_inbound INTEGER,
        trucks_with_trailer_inbound INTEGER,
        motorcycles_inbound INTEGER,
        exemption_vehicles_inbound INTEGER,
        pedestrians_inbound INTEGER,
        buses_inbound INTEGER,
        vehicles_left_at_terminal_inbound INTEGER,
        trip_type VARCHAR(100),
        passenger_car_equivalent_outbound_and_inbound DECIMAL(10,2),
        tailored_trip INTEGER,
        full_ferry_outbound INTEGER,
        full_ferry_inbound INTEGER,
        passenger_car_equivalent_outbound DECIMAL(10,2),
        passenger_car_equivalent_inbound DECIMAL(10,2),
        fuelcons_outbound_l DECIMAL(10,2),
        distance_outbound_nm DECIMAL(10,6),
        start_time_outbound TIMESTAMP,
        end_time_outbound TIMESTAMP,
        fuelcons_inbound_l DECIMAL(10,2),
        distance_inbound_nm DECIMAL(10,6),
        start_time_inbound TIMESTAMP,
        end_time_inbound TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    cursor.execute(create_table_sql)
    logger.info("Ferry trips table created successfully")

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object"""
    if not timestamp_str or timestamp_str.strip() == '':
        return None
    try:
        return datetime.strptime(timestamp_str.strip(), '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None

def parse_number(value_str, number_type='int'):
    """Parse string to number, handling empty values"""
    if not value_str or value_str.strip() == '':
        return None
    try:
        if number_type == 'float':
            return float(value_str.strip())
        else:
            return int(value_str.strip())
    except ValueError:
        return None

def insert_batch(cursor, batch_data):
    """Insert a batch of ferry trip records"""
    
    insert_sql = """
    INSERT INTO ferry_trips (
        route_id, route_name, ferry_name, ferry_id, terminal_departure, terminal_arrival,
        time_departure, cars_outbound, trucks_outbound, trucks_with_trailer_outbound,
        motorcycles_outbound, exemption_vehicles_outbound, pedestrians_outbound,
        buses_outbound, vehicles_left_at_terminal_outbound, cars_inbound, trucks_inbound,
        trucks_with_trailer_inbound, motorcycles_inbound, exemption_vehicles_inbound,
        pedestrians_inbound, buses_inbound, vehicles_left_at_terminal_inbound, trip_type,
        passenger_car_equivalent_outbound_and_inbound, tailored_trip, full_ferry_outbound,
        full_ferry_inbound, passenger_car_equivalent_outbound, passenger_car_equivalent_inbound,
        fuelcons_outbound_l, distance_outbound_nm, start_time_outbound, end_time_outbound,
        fuelcons_inbound_l, distance_inbound_nm, start_time_inbound, end_time_inbound
    ) VALUES %s
    """
    
    values = []
    for row in batch_data:
        values.append((
            row['route_id'], row['route_name'], row['ferry_name'], row['ferry_id'],
            row['terminal_departure'], row['terminal_arrival'], row['time_departure'],
            row['cars_outbound'], row['trucks_outbound'], row['trucks_with_trailer_outbound'],
            row['motorcycles_outbound'], row['exemption_vehicles_outbound'], row['pedestrians_outbound'],
            row['buses_outbound'], row['vehicles_left_at_terminal_outbound'], row['cars_inbound'],
            row['trucks_inbound'], row['trucks_with_trailer_inbound'], row['motorcycles_inbound'],
            row['exemption_vehicles_inbound'], row['pedestrians_inbound'], row['buses_inbound'],
            row['vehicles_left_at_terminal_inbound'], row['trip_type'],
            row['passenger_car_equivalent_outbound_and_inbound'], row['tailored_trip'],
            row['full_ferry_outbound'], row['full_ferry_inbound'], row['passenger_car_equivalent_outbound'],
            row['passenger_car_equivalent_inbound'], row['fuelcons_outbound_l'], row['distance_outbound_nm'],
            row['start_time_outbound'], row['end_time_outbound'], row['fuelcons_inbound_l'],
            row['distance_inbound_nm'], row['start_time_inbound'], row['end_time_inbound']
        ))
    
    execute_values(cursor, insert_sql, values)

def load_ferry_data(csv_file_path, batch_size=1000):
    """Load ferry data from CSV file into database"""
    
    conn = get_database_connection()
    
    try:
        with conn.cursor() as cursor:
            # Create table
            create_ferry_trips_table(cursor)
            
            # Clear existing data
            cursor.execute("TRUNCATE TABLE ferry_trips RESTART IDENTITY")
            logger.info("Cleared existing ferry trips data")
            
            batch_data = []
            total_rows = 0
            successful_rows = 0
            
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                
                for row_num, row in enumerate(csv_reader, 1):
                    total_rows += 1
                    
                    try:
                        # Parse and validate data
                        parsed_row = {
                            'route_id': parse_number(row.get('route_id')),
                            'route_name': row.get('route_name', '').strip() or None,
                            'ferry_name': row.get('ferry_name', '').strip() or None,
                            'ferry_id': parse_number(row.get('ferry_id')),
                            'terminal_departure': row.get('terminal_departure', '').strip() or None,
                            'terminal_arrival': row.get('terminal_arrival', '').strip() or None,
                            'time_departure': parse_timestamp(row.get('time_departure')),
                            'cars_outbound': parse_number(row.get('cars_outbound')),
                            'trucks_outbound': parse_number(row.get('trucks_outbound')),
                            'trucks_with_trailer_outbound': parse_number(row.get('trucks_with_trailer_outbound')),
                            'motorcycles_outbound': parse_number(row.get('motorcycles_outbound')),
                            'exemption_vehicles_outbound': parse_number(row.get('exemption_vehicles_outbound')),
                            'pedestrians_outbound': parse_number(row.get('pedestrians_outbound')),
                            'buses_outbound': parse_number(row.get('buses_outbound')),
                            'vehicles_left_at_terminal_outbound': parse_number(row.get('vehicles_left_at_terminal_outbound')),
                            'cars_inbound': parse_number(row.get('cars_inbound')),
                            'trucks_inbound': parse_number(row.get('trucks_inbound')),
                            'trucks_with_trailer_inbound': parse_number(row.get('trucks_with_trailer_inbound')),
                            'motorcycles_inbound': parse_number(row.get('motorcycles_inbound')),
                            'exemption_vehicles_inbound': parse_number(row.get('exemption_vehicles_inbound')),
                            'pedestrians_inbound': parse_number(row.get('pedestrians_inbound')),
                            'buses_inbound': parse_number(row.get('buses_inbound')),
                            'vehicles_left_at_terminal_inbound': parse_number(row.get('vehicles_left_at_terminal_inbound')),
                            'trip_type': row.get('trip_type', '').strip() or None,
                            'passenger_car_equivalent_outbound_and_inbound': parse_number(row.get('passenger_car_equivalent_outbound_and_inbound'), 'float'),
                            'tailored_trip': parse_number(row.get('tailored_trip')),
                            'full_ferry_outbound': parse_number(row.get('full_ferry_outbound')),
                            'full_ferry_inbound': parse_number(row.get('full_ferry_inbound')),
                            'passenger_car_equivalent_outbound': parse_number(row.get('passenger_car_equivalent_outbound'), 'float'),
                            'passenger_car_equivalent_inbound': parse_number(row.get('passenger_car_equivalent_inbound'), 'float'),
                            'fuelcons_outbound_l': parse_number(row.get('fuelcons_outbound_l'), 'float'),
                            'distance_outbound_nm': parse_number(row.get('distance_outbound_nm'), 'float'),
                            'start_time_outbound': parse_timestamp(row.get('start_time_outbound')),
                            'end_time_outbound': parse_timestamp(row.get('end_time_outbound')),
                            'fuelcons_inbound_l': parse_number(row.get('fuelcons_inbound_l'), 'float'),
                            'distance_inbound_nm': parse_number(row.get('distance_inbound_nm'), 'float'),
                            'start_time_inbound': parse_timestamp(row.get('start_time_inbound')),
                            'end_time_inbound': parse_timestamp(row.get('end_time_inbound'))
                        }
                        
                        batch_data.append(parsed_row)
                        successful_rows += 1
                        
                        # Insert batch when it reaches batch_size
                        if len(batch_data) >= batch_size:
                            insert_batch(cursor, batch_data)
                            logger.info(f"Processed {successful_rows} rows...")
                            batch_data = []
                            
                    except Exception as e:
                        logger.warning(f"Error processing row {row_num}: {e}")
                        continue
                
                # Insert remaining data
                if batch_data:
                    insert_batch(cursor, batch_data)
            
            # Commit transaction
            conn.commit()
            
            logger.info(f"Data loading completed:")
            logger.info(f"  Total rows processed: {total_rows}")
            logger.info(f"  Successfully inserted: {successful_rows}")
            logger.info(f"  Failed rows: {total_rows - successful_rows}")
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during data loading: {e}")
        raise
    finally:
        conn.close()

def main():
    """Main function to run the data loading script"""
    
    # Determine CSV file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, '..', 'data', 'ferry_trips_data.csv')
    
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found: {csv_file_path}")
        return
    
    logger.info(f"Starting ferry data loading from: {csv_file_path}")
    
    try:
        load_ferry_data(csv_file_path)
        logger.info("Ferry data loading completed successfully!")
    except Exception as e:
        logger.error(f"Failed to load ferry data: {str(e)}")

if __name__ == "__main__":
    main()