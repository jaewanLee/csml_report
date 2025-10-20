"""
BTC Data Collector using ccxt
Collects historical OHLCV data for multiple timeframes
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional

# Import configuration
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.exchange_config import EXCHANGE_CONFIG, COLLECTION_CONFIG, VALIDATION_CONFIG


class BTCDataCollector:
    """
    Collects BTC historical data using ccxt library
    """

    def __init__(self, exchange_name: str = 'binance'):
        """
        Initialize the data collector
        
        Args:
            exchange_name: Name of the exchange to use ('binance' or 'coinbase')
        """
        self.exchange_name = exchange_name
        self.exchange = self._setup_exchange()
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.logs_dir = Path(__file__).parent.parent / 'logs'

        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_exchange(self):
        """Setup exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class(EXCHANGE_CONFIG[self.exchange_name])
            self.logger.info(f"Successfully connected to {self.exchange_name}")
            return exchange
        except Exception as e:
            self.logger.error(
                f"Failed to setup exchange {self.exchange_name}: {e}")
            raise

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.logs_dir / f"btc_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file),
                      logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

    def collect_timeframe_data(self,
                               timeframe: str,
                               start_date: str,
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Collect data for a specific timeframe
        
        Args:
            timeframe: Timeframe string (e.g., '4h', '1d', '1w')
            start_date: Start date in ISO format
            end_date: End date in ISO format (None for current date)
            
        Returns:
            DataFrame with OHLCV data
        """
        self.logger.info(
            f"Collecting {timeframe} data from {start_date} to {end_date or 'now'}"
        )

        try:
            # Convert dates to timestamps
            start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_timestamp = int(pd.Timestamp(end_date).timestamp() *
                                1000) if end_date else None

            # Collect data in batches
            all_data = []
            current_timestamp = start_timestamp

            while True:
                try:
                    # Fetch data batch
                    ohlcv = self.exchange.fetch_ohlcv(
                        COLLECTION_CONFIG['symbol'],
                        timeframe,
                        since=current_timestamp,
                        limit=COLLECTION_CONFIG['batch_size'])

                    if not ohlcv:
                        break

                    all_data.extend(ohlcv)

                    # Update timestamp for next batch
                    current_timestamp = ohlcv[-1][0] + 1

                    # Check if we've reached the end date
                    if end_timestamp and current_timestamp >= end_timestamp:
                        break

                    # Rate limiting
                    time.sleep(1 / COLLECTION_CONFIG['rateLimit'])

                except Exception as e:
                    self.logger.warning(
                        f"Error fetching batch at {current_timestamp}: {e}")
                    time.sleep(COLLECTION_CONFIG['retry_delay'])
                    continue

            # Convert to DataFrame
            df = pd.DataFrame(all_data,
                              columns=[
                                  'timestamp', 'open', 'high', 'low', 'close',
                                  'volume'
                              ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Data validation
            df = self._validate_data(df, timeframe)

            self.logger.info(
                f"Successfully collected {len(df)} records for {timeframe}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to collect {timeframe} data: {e}")
            raise

    def _validate_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Validate collected data
        
        Args:
            df: DataFrame to validate
            timeframe: Timeframe string
            
        Returns:
            Validated DataFrame
        """
        self.logger.info(f"Validating {timeframe} data...")

        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            self.logger.warning(
                f"Found {missing_count} missing values in {timeframe} data")
            df = df.dropna()

        # Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(
                f"Found {duplicates} duplicate timestamps in {timeframe} data")
            df = df[~df.index.duplicated(keep='first')]

        # Check for extreme price changes
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            price_changes = df[col].pct_change().abs()
            extreme_changes = (price_changes
                               > VALIDATION_CONFIG['max_price_change']).sum()
            if extreme_changes > 0:
                self.logger.warning(
                    f"Found {extreme_changes} extreme price changes in {col} for {timeframe}"
                )

        # Check for negative prices
        negative_prices = (df[price_cols] < 0).any().any()
        if negative_prices:
            self.logger.error(f"Found negative prices in {timeframe} data")
            raise ValueError("Negative prices detected")

        # Check for zero volume
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            self.logger.warning(
                f"Found {zero_volume} zero volume records in {timeframe} data")

        self.logger.info(f"Data validation completed for {timeframe}")
        return df

    def save_data(self, df: pd.DataFrame, timeframe: str):
        """
        Save data to Parquet format
        
        Args:
            df: DataFrame to save
            timeframe: Timeframe string
        """
        filename = f"btc_{timeframe}_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = self.data_dir / filename

        try:
            df.to_parquet(filepath, compression='snappy')
            self.logger.info(f"Saved {timeframe} data to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save {timeframe} data: {e}")
            raise

    def collect_all_timeframes(self):
        """
        Collect data for all configured timeframes
        """
        self.logger.info("Starting data collection for all timeframes")

        for timeframe in COLLECTION_CONFIG['timeframes']:
            try:
                # Collect data
                df = self.collect_timeframe_data(
                    timeframe=timeframe,
                    start_date=COLLECTION_CONFIG['start_date'],
                    end_date=COLLECTION_CONFIG['end_date'])

                # Save data
                self.save_data(df, timeframe)

                # Generate data quality report
                self._generate_quality_report(df, timeframe)

            except Exception as e:
                self.logger.error(f"Failed to collect {timeframe} data: {e}")
                continue

        self.logger.info("Data collection completed for all timeframes")

    def _generate_quality_report(self, df: pd.DataFrame, timeframe: str):
        """
        Generate data quality report
        
        Args:
            df: DataFrame to analyze
            timeframe: Timeframe string
        """
        report = {
            'timeframe': timeframe,
            'total_records': len(df),
            'date_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'price_statistics': {
                'min_price': df[['open', 'high', 'low', 'close']].min().min(),
                'max_price': df[['open', 'high', 'low', 'close']].max().max(),
                'mean_price': df[['open', 'high', 'low',
                                  'close']].mean().mean()
            },
            'volume_statistics': {
                'min_volume': df['volume'].min(),
                'max_volume': df['volume'].max(),
                'mean_volume': df['volume'].mean()
            }
        }

        # Save report
        report_file = self.logs_dir / f"quality_report_{timeframe}_{datetime.now().strftime('%Y%m%d')}.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Quality report saved for {timeframe}")


if __name__ == "__main__":
    # Example usage
    collector = BTCDataCollector(exchange_name='binance')
    collector.collect_all_timeframes()
