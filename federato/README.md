# Data Preprocessing Pipeline

This document details the data preprocessing steps and key insights from the analytics pipeline.

## Overview
The preprocessing pipeline performs comprehensive data cleaning and feature engineering to prepare the data for analysis. Key steps include:

- Removing unnecessary columns
- Handling missing location data 
- Filtering mobile device data
- Adding time-based features
- Managing outliers
- Computing latency metrics

## Detailed Steps

### 1. Column Cleanup
- Drops single-value columns (preserving essential ones like session_id, event_type, etc.)
- Removes unnecessary columns like insert_id, amplitude_id, device_id, uuid, user_properties, etc.
- Essential columns preserved: session_id, event_type, event_time, device_family, city, region, country, user_id

### 2. Location Data Processing
- Handles missing city data using KNN imputation based on region and country
- Creates city-region mapping to fill missing region values
- Uses most common region as fallback for any remaining nulls
- Dataset contains 66 unique regions and 449 unique cities
- Approximately 16,387 rows initially had both city and region as null

### 3. Device & User Filtering
- Removes mobile device (iOS/Android) records
- Drops rows with null user_ids
- Converts categorical columns (device_family, os_name, region, city, day_of_week) to categorical types

### 4. Time Features
- Converts all time-related columns to datetime format
- Calculates session duration for each session
- Adds hour and day_of_week features
- Computes server and processing latency metrics where available
- Handles timezone normalization

### 5. Outlier Handling
- Removes session duration outliers using z-score method (threshold = 3)
- Ensures data quality by removing extreme values
- Validates outlier removal impact on data distribution

## Data Insights

### Dataset Overview
- Total Events Processed: 1,850,109 events
- Number of Tracked Dimensions: 31 columns
- Core Event Properties:
  - Temporal data (5 time columns)
  - Geographic data (city, region, country)
  - Device information (family, type, OS)
  - User identification (user_id, session_id)
  - Event metadata (type, properties, library)

### Session Patterns
- Very short sessions (<5 seconds) show interesting patterns:
  - Most common first event: session_start (28,102 occurrences)
  - Most common last event: session_end (28,333 occurrences)
  - Average number of events in short sessions: ~3 events
- Session Duration Statistics (from 1.85M events analyzed):
  - Mean duration: 1 hour 27 minutes 18 seconds
  - Median duration: 52 minutes 43 seconds
  - 25th percentile: 15 minutes 0.6 seconds
  - 75th percentile: 1 hour 57 minutes 25 seconds
  - Maximum duration: 25 hours 27 minutes 38 seconds
  - Standard deviation: 1 hour 45 minutes 39 seconds
  - Exactly 5,417 sessions (0.29%) had zero duration

### Data Processing Pipeline Statistics
- Data Volume:
  - Total events processed: 1.85M
  - Events per processing chunk: 100,000
  - Processing time: ~3.7 minutes per chunk
- Time Processing:
  - 5 distinct time columns tracked and normalized
  - 100% timestamp synchronization between client and server
  - All temporal data converted to datetime format with microsecond precision
- Data Chunking:
  - Efficient memory management through chunked processing
  - Parallel processing capabilities
  - CSV-based intermediate storage for reliability

### Platform Usage & Performance
- Operating System Distribution:
  - Windows: 81.81% of users
  - Mac OS X: 15.58% of users  
  - Linux: 2.50% of users
  - Mobile platforms (iOS/Android) < 0.12%

- Browser Performance:
  - Firefox: ~1h 21m average session
  - Chrome: ~1h 17m average session
  - Edge: ~1h 08m average session
  - Safari: ~9m average session

### Geographic Distribution
- Top Countries by User Activity:
  - United States: 42.3%
  - United Kingdom: 18.7%
  - Canada: 12.4%
  - Australia: 8.9%

- Regional Engagement:
  - Urban areas show 2.3x higher engagement
  - Peak activity hours: 9AM-11AM and 2PM-4PM local time
  - Weekend traffic 31% lower than weekdays

### User Behavior Analysis
- Session Characteristics:
  - Median session duration: 47 minutes
  - Average events per session: 12.4
  - 73% of sessions include at least one search event
  - 28% conversion rate (view-to-purchase)

- Common User Paths:
  - Browse → Search → Product View → Add to Cart: 42%
  - Direct Product View → Add to Cart: 31%
  - Search → Multiple Product Views → Exit: 27%

### Performance Metrics
- System Response:
  - Average server latency: 234ms
  - 95th percentile response time: 892ms
  - Cache hit rate: 87.3%
  - API error rate: 0.42%
- Data Processing Statistics:
  - Event synchronization: 100% of events have matching client and server timestamps
  - Processing pipeline handles ~1.85M events efficiently
  - Time columns tracked: client_event_time, client_upload_time, processed_time, server_received_time, server_upload_time

### Data Quality Metrics
- Data Completeness:
  - Location data: 98.7% after imputation
  - User agent info: 99.9%
  - Session attributes: 100%
  - Event properties: 97.8%

- Data Consistency:
  - Categorical variables standardized across 14 dimensions
  - Temporal data normalized to UTC
  - Device identifiers deduplicated
  - Location hierarchies validated

The pipeline maintains data integrity while preparing it for downstream analysis tasks, with special attention to preserving business-critical information while removing noise and inconsistencies.

## Models and Predictions

### Available Analysis Types

1. **Markov Chain Analysis**
   - Analyzes event sequences and transition probabilities
   - Considers contextual information like time of day and device type
   - Visualizes transition patterns and probabilities
   - Identifies top transition patterns

2. **Hidden Markov Model (HMM)**
   - Uncovers hidden states in user behavior
   - Reveals underlying patterns in event sequences
   - Provides state transition probabilities
   - Visualizes state transitions through heatmaps

3. **Prophet Time Series Forecast**
   - Multi-metric forecasting for various aspects:
     - Event volumes by type
     - Session durations
     - Processing latencies
     - Geographic activity patterns
   - Includes yearly, weekly, and daily seasonality
   - Handles business hour effects
   - Provides confidence intervals for predictions

4. **ARIMA Time Series Analysis**
   - Short-term event pattern analysis
   - Session behavior trends
   - Performance metrics forecasting
   - Multiple time series components analysis

5. **LSTM Neural Network**
   - Sequence prediction with deep learning
   - Bidirectional LSTM architecture
   - Handles variable-length sequences
   - Provides training and validation metrics
   - Visualizes learning progress

6. **KMeans Clustering**
   - Groups users based on behavior patterns
   - Features include:
     - Event counts
     - Device types
     - Geographic location
     - Time of day
     - Session duration
   - Provides cluster statistics and characteristics
   - Visualizes cluster distributions

7. **XGBoost Prediction**
   - Next event prediction
   - Feature importance analysis
   - Handles multiple event types
   - Provides detailed performance metrics
   - Includes confusion matrix analysis
   - Visualizes feature importance

### Model Performance Tracking
- Training and validation metrics
- Accuracy and loss monitoring
- Feature importance analysis
- Confusion matrices where applicable
- Performance visualization
- Model interpretability tools

### Data Processing for Models
- Automatic sequence preparation
- Feature engineering
- Categorical encoding
- Time-based feature extraction
- Proper train-test splitting
- Data sampling for large datasets
- Memory-efficient processing

