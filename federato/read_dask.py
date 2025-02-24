#!/usr/bin/env python3

import dask.dataframe as dd
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime
import pytz
import time
import logging
from typing import Dict, Any
import json
from tqdm import tqdm
import numpy as np
import os
import psutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('federato/dask_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    logger.info(f"Current memory usage: {mem:.2f} MB")

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, description: str):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting: {self.description}")
        log_memory_usage()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Completed: {self.description} in {duration:.2f} seconds")
        log_memory_usage()
        return False

def get_essential_columns():
    """Define the essential columns to load."""
    return [
        'user_id',
        'session_id',
        'event_type',
        'client_event_time',
        'country',
        'region',
        'city',
        'device_family',
        'os_name'
    ]

def get_dtypes() -> Dict[str, Any]:
    """Define column dtypes to avoid mixed type warnings."""
    return {
        'user_id': 'string',
        'session_id': 'Int64',
        'event_type': 'category',
        'country': 'category',
        'region': 'category',
        'city': 'category',
        'device_family': 'category',
        'os_name': 'category'
    }

def load_and_process_data():
    """Load all CSV files from 2024 directory using Dask with optimized settings."""
    with Timer("Data Loading"):
        # Define the path pattern for CSV files
        csv_pattern = 'federato/2024/*_csv/*.csv'
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            logger.error(f"No CSV files found matching pattern: {csv_pattern}")
            return None
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Calculate optimal chunk size based on available memory
        available_memory = psutil.virtual_memory().available
        chunk_size = min(128 * 1024 * 1024, available_memory // (4 * len(csv_files)))  # Increased chunk size
        
        try:
            logger.info(f"Reading CSV files with chunk size: {chunk_size/1024/1024:.2f} MB")
            df = dd.read_csv(
                csv_pattern,
                usecols=get_essential_columns(),
                dtype=get_dtypes(),
                blocksize=chunk_size,
                assume_missing=True,
                parse_dates=['client_event_time'],  # Parse dates during loading
                storage_options={'anon': True}
            )
            
            # Early filtering of mobile devices
            logger.info("Filtering out mobile devices...")
            df = df[~df['device_family'].isin(['iOS', 'Android'])]
            
            # Drop rows with null user_ids early
            logger.info("Removing rows with null user_ids...")
            df = df.dropna(subset=['user_id'])
            
            # Add computed columns efficiently
            logger.info("Computing time-based columns...")
            df['hour'] = df['client_event_time'].dt.hour
            df['day'] = df['client_event_time'].dt.day_name()
            df['month'] = df['client_event_time'].dt.month
            
            # Optimize partitioning
            logger.info("Optimizing partitions...")
            n_partitions = max(1, os.cpu_count() * 2)  # At least 1 partition
            df = df.repartition(npartitions=n_partitions)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

def run_basic_analysis(df):
    """Run basic analysis with optimized computations."""
    with Timer("Basic Analysis"):
        try:
            logger.info("Computing metrics in stages...")
            
            # Stage 1: Basic counts (faster operations)
            logger.info("Stage 1: Computing basic counts...")
            basic_metrics = dd.compute(
                df.shape[0],  # total events
                df['user_id'].nunique(),  # unique users
                df['session_id'].nunique(),  # unique sessions
            )
            total_events, unique_users, unique_sessions = basic_metrics
            
            logger.info("\n=== Basic Statistics ===")
            logger.info(f"Total events: {total_events:,}")
            logger.info(f"Unique users: {unique_users:,}")
            logger.info(f"Unique sessions: {unique_sessions:,}")
            
            # Stage 2: Event type analysis (sample-based for faster processing)
            logger.info("\nStage 2: Analyzing event types...")
            sample_size = min(1_000_000, total_events)  # Cap at 1M events for sampling
            event_sample = df['event_type'].sample(n=sample_size, random_state=42)
            top_events = event_sample.value_counts().compute().nlargest(10)
            
            logger.info("\n=== Top 10 Event Types (Based on Sample) ===")
            for event_type, count in top_events.items():
                # Scale up the counts to estimate full dataset
                estimated_count = int(count * (total_events / sample_size))
                logger.info(f"{event_type}: ~{estimated_count:,} (estimated)")
            
            # Stage 3: Geographic distribution (using sampling)
            logger.info("\nStage 3: Analyzing geographic distribution...")
            country_sample = df['country'].sample(n=sample_size, random_state=42)
            top_countries = country_sample.value_counts().compute().nlargest(5)
            
            logger.info("\n=== Top 5 Countries (Based on Sample) ===")
            for country, count in top_countries.items():
                estimated_count = int(count * (total_events / sample_size))
                logger.info(f"{country}: ~{estimated_count:,} (estimated)")
            
            # Stage 4: OS distribution (using sampling)
            logger.info("\nStage 4: Analyzing OS distribution...")
            os_sample = df['os_name'].sample(n=sample_size, random_state=42)
            os_dist = os_sample.value_counts().compute()
            
            logger.info("\n=== OS Distribution (Based on Sample) ===")
            total_sample = os_dist.sum()
            for os_name, count in os_dist.items():
                percentage = (count / total_sample) * 100
                estimated_count = int(count * (total_events / sample_size))
                logger.info(f"{os_name}: ~{estimated_count:,} ({percentage:.2f}%)")
            
            # Stage 5: Time distribution (using efficient binning)
            logger.info("\nStage 5: Analyzing hourly distribution...")
            hour_sample = df['hour'].sample(n=sample_size, random_state=42)
            hourly_dist = hour_sample.value_counts().compute().sort_index()
            
            logger.info("\n=== Hourly Distribution (Based on Sample) ===")
            for hour in range(24):
                count = hourly_dist.get(hour, 0)
                estimated_count = int(count * (total_events / sample_size))
                logger.info(f"{hour:02d}:00 - ~{estimated_count:,} events (estimated)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in basic analysis: {str(e)}")
            raise

def main():
    """Main function with optimized execution flow."""
    total_start_time = time.time()
    logger.info("Starting optimized Dask analysis...")
    
    try:
        # Load and preprocess the data
        df = load_and_process_data()
        if df is None:
            return
        
        # Run basic analysis
        run_basic_analysis(df)
        
        # Log overall execution time
        total_time = time.time() - total_start_time
        logger.info(f"\nTotal analysis completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise
    finally:
        # Log final memory usage
        log_memory_usage()

if __name__ == "__main__":
    main() 