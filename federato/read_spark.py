#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import logging
import time
import glob
import psutil
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('federato/spark_analysis.log')
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

def create_spark_session():
    """Create and configure Spark session."""
    # Calculate memory settings based on available system memory
    available_memory = psutil.virtual_memory().available
    executor_memory = int(available_memory * 0.8 / 1024 / 1024)  # 80% of available memory in MB
    
    return (SparkSession.builder
            .appName("UserAnalytics")
            .config("spark.executor.memory", f"{executor_memory}m")
            .config("spark.driver.memory", f"{executor_memory}m")
            .config("spark.sql.shuffle.partitions", os.cpu_count() * 4)
            .config("spark.default.parallelism", os.cpu_count() * 4)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
            # Fix for Java security manager issue
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
            # Local mode settings
            .master("local[*]")
            .config("spark.driver.host", "localhost")
            .config("spark.driver.bindAddress", "localhost")
            .getOrCreate())

def get_schema():
    """Define the schema for CSV files."""
    return StructType([
        StructField("user_id", StringType(), True),
        StructField("session_id", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("client_event_time", TimestampType(), True),
        StructField("country", StringType(), True),
        StructField("region", StringType(), True),
        StructField("city", StringType(), True),
        StructField("device_family", StringType(), True),
        StructField("os_name", StringType(), True)
    ])

def load_and_process_data(spark):
    """Load and process CSV files using Spark."""
    with Timer("Data Loading"):
        try:
            # Get list of CSV files
            csv_pattern = 'federato/2024/*_csv/*.csv'
            csv_files = glob.glob(csv_pattern)
            
            if not csv_files:
                logger.error(f"No CSV files found matching pattern: {csv_pattern}")
                return None
            
            logger.info(f"Found {len(csv_files)} CSV files to process")
            
            # Read CSV files with schema
            df = (spark.read.format("csv")
                 .option("header", "true")
                 .schema(get_schema())
                 .load(csv_pattern))
            
            # Early filtering and processing
            df = (df.filter(~F.col("device_family").isin(["iOS", "Android"]))  # Remove mobile
                 .filter(F.col("user_id").isNotNull())  # Remove null users
                 .withColumn("hour", F.hour("client_event_time"))
                 .withColumn("day", F.date_format("client_event_time", "EEEE"))
                 .withColumn("month", F.month("client_event_time"))
                 .cache())  # Cache the filtered dataset
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

def analyze_data(df):
    """Run analysis on the Spark DataFrame."""
    with Timer("Data Analysis"):
        try:
            # Basic Statistics
            logger.info("\n=== Basic Statistics ===")
            basic_stats = df.agg(
                F.count("*").alias("total_events"),
                F.countDistinct("user_id").alias("unique_users"),
                F.countDistinct("session_id").alias("unique_sessions")
            ).collect()[0]
            
            logger.info(f"Total events: {basic_stats['total_events']:,}")
            logger.info(f"Unique users: {basic_stats['unique_users']:,}")
            logger.info(f"Unique sessions: {basic_stats['unique_sessions']:,}")
            
            # Event Types Distribution
            logger.info("\n=== Top 10 Event Types ===")
            df.groupBy("event_type") \
              .count() \
              .orderBy(F.col("count").desc()) \
              .limit(10) \
              .show(truncate=False)
            
            # Geographic Distribution
            logger.info("\n=== Top 5 Countries ===")
            df.groupBy("country") \
              .count() \
              .orderBy(F.col("count").desc()) \
              .limit(5) \
              .show()
            
            # OS Distribution
            logger.info("\n=== OS Distribution ===")
            total_events = df.count()
            (df.groupBy("os_name")
             .count()
             .withColumn("percentage", F.round((F.col("count") / total_events) * 100, 2))
             .orderBy(F.col("count").desc())
             .show())
            
            # Hourly Distribution
            logger.info("\n=== Hourly Distribution ===")
            (df.groupBy("hour")
             .count()
             .orderBy("hour")
             .show(24))

        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise

def main():
    """Main function to orchestrate the analysis."""
    total_start_time = time.time()
    logger.info("Starting Spark analysis...")
    spark = None
    
    try:
        # Create Spark session
        spark = create_spark_session()
        
        # Load and process data
        df = load_and_process_data(spark)
        if df is None:
            return
        
        # Run analysis
        analyze_data(df)
        
        # Log overall execution time
        total_time = time.time() - total_start_time
        logger.info(f"\nTotal analysis completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise
    finally:
        # Log final memory usage
        log_memory_usage()
        # Stop Spark session
        if spark:
            spark.stop()

if __name__ == "__main__":
    main() 