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

    def __init__(self, exchange_name: str = "binance"):
        """
        Initialize the data collector

        Args:
            exchange_name: Name of the exchange to use ('binance' or 'coinbase')
        """
        self.exchange_name = exchange_name
        self.data_dir = Path(__file__).parent.parent / "data"
        self.logs_dir = Path(__file__).parent.parent / "logs"

        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Setup logging BEFORE using self.logger
        self._setup_logging()

        # Setup exchange after logging is ready
        self.exchange = self._setup_exchange()

    def _setup_exchange(self):
        """Setup exchange connection with proper error handling"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class(EXCHANGE_CONFIG[self.exchange_name])

            # Test connection by loading markets
            exchange.load_markets()
            self.logger.info(f"Successfully connected to {self.exchange_name}")
            return exchange

        except Exception as e:
            self.logger.error(
                f"Failed to setup exchange {self.exchange_name}: {e}")
            raise

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = (
            self.logs_dir /
            f"btc_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file),
                      logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _retry_with_backoff(self,
                            func,
                            max_retries: int = 3,
                            base_delay: float = 1.0):
        """
        Retry a function with exponential backoff

        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds

        Returns:
            Function result or raises last exception
        """
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(
                        f"Max retries ({max_retries}) exceeded: {e}")
                    raise

                delay = base_delay * (2**attempt)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                )
                time.sleep(delay)

        raise Exception("Unexpected error in retry logic")

    def _check_exchange_health(self) -> bool:
        """
        Check if exchange is healthy and responsive

        Returns:
            True if exchange is healthy, False otherwise
        """
        try:
            # Test basic connectivity
            ticker = self.exchange.fetch_ticker(COLLECTION_CONFIG["symbol"])
            if not ticker or "last" not in ticker:
                return False

            # Check if we can fetch recent data
            recent_ohlcv = self.exchange.fetch_ohlcv(
                COLLECTION_CONFIG["symbol"], "1h", limit=1)

            if not recent_ohlcv:
                return False

            self.logger.info("Exchange health check passed")
            return True

        except Exception as e:
            self.logger.warning(f"Exchange health check failed: {e}")
            return False

    def collect_timeframe_data(self,
                               timeframe: str,
                               start_date: str,
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Collect data for a specific timeframe with improved pagination

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
            end_timestamp = (int(pd.Timestamp(end_date).timestamp() *
                                 1000) if end_date else None)

            # Collect data in batches with improved pagination
            all_data = []
            current_timestamp = start_timestamp
            max_retries = 3
            retry_count = 0

            while True:
                try:
                    # Fetch data batch with proper error handling
                    ohlcv = self.exchange.fetch_ohlcv(
                        COLLECTION_CONFIG["symbol"],
                        timeframe,
                        since=current_timestamp,
                        limit=COLLECTION_CONFIG["batch_size"],
                    )

                    if not ohlcv:
                        self.logger.info("No more data available")
                        break

                    all_data.extend(ohlcv)

                    # Update timestamp for next batch (avoid duplicates)
                    current_timestamp = ohlcv[-1][0] + 1

                    # Check if we've reached the end date
                    if end_timestamp and current_timestamp >= end_timestamp:
                        self.logger.info("Reached end date")
                        break

                    # Reset retry count on successful fetch
                    retry_count = 0

                    # Improved rate limiting using exchange's built-in rate limiting
                    if hasattr(self.exchange,
                               "rateLimit") and self.exchange.rateLimit:
                        time.sleep(1 / self.exchange.rateLimit)
                    else:
                        time.sleep(0.1)  # Fallback rate limiting

                except Exception as e:
                    retry_count += 1
                    self.logger.warning(
                        f"Error fetching batch at {current_timestamp} (attempt {retry_count}): {e}"
                    )

                    if retry_count >= max_retries:
                        self.logger.error(
                            f"Max retries ({max_retries}) exceeded for batch at {current_timestamp}"
                        )
                        break

                    # Exponential backoff for retries
                    time.sleep(COLLECTION_CONFIG["retry_delay"] *
                               (2**retry_count))
                    continue

            if not all_data:
                self.logger.warning(f"No data collected for {timeframe}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
                all_data,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume"
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

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
        Validate collected data with comprehensive checks

        Args:
            df: DataFrame to validate
            timeframe: Timeframe string

        Returns:
            Validated DataFrame
        """
        self.logger.info(f"Validating {timeframe} data...")

        if df.empty:
            self.logger.warning(f"Empty DataFrame for {timeframe}")
            return df

        original_length = len(df)

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
            df = df[~df.index.duplicated(keep="first")]

        # Check for extreme price changes
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                price_changes = df[col].pct_change().abs()
                extreme_changes = (
                    price_changes
                    > VALIDATION_CONFIG["max_price_change"]).sum()
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
        zero_volume = (df["volume"] == 0).sum()
        if zero_volume > 0:
            self.logger.warning(
                f"Found {zero_volume} zero volume records in {timeframe} data")

        # Check for logical price relationships (high >= low, etc.)
        invalid_ohlc = ((df["high"] < df["low"])
                        | (df["high"] < df["open"])
                        | (df["high"] < df["close"])
                        | (df["low"] > df["open"])
                        | (df["low"] > df["close"])).sum()

        if invalid_ohlc > 0:
            self.logger.warning(
                f"Found {invalid_ohlc} records with invalid OHLC relationships in {timeframe}"
            )

        # Check data continuity (gaps in time series)
        if len(df) > 1:
            time_diff = df.index.to_series().diff().dropna()
            expected_interval = pd.Timedelta(
                timeframe.replace("m", "min").replace("h",
                                                      "h").replace("d", "D"))
            gaps = (time_diff > expected_interval * 1.5).sum()
            if gaps > 0:
                self.logger.warning(
                    f"Found {gaps} potential time gaps in {timeframe} data")

        final_length = len(df)
        if final_length != original_length:
            self.logger.info(
                f"Data validation: {original_length} -> {final_length} records for {timeframe}"
            )

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
            df.to_parquet(filepath, compression="snappy")
            self.logger.info(f"Saved {timeframe} data to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save {timeframe} data: {e}")
            raise

    def collect_all_timeframes(self):
        """
        Collect data for all configured timeframes with improved error handling
        """
        self.logger.info("Starting data collection for all timeframes")

        successful_timeframes = []
        failed_timeframes = []

        for timeframe in COLLECTION_CONFIG["timeframes"]:
            try:
                self.logger.info(f"Starting collection for {timeframe}")

                # Use retry logic for data collection
                df = self._retry_with_backoff(
                    lambda: self.collect_timeframe_data(
                        timeframe=timeframe,
                        start_date=COLLECTION_CONFIG["start_date"],
                        end_date=COLLECTION_CONFIG["end_date"],
                    ),
                    max_retries=3,
                    base_delay=2.0,
                )

                if df.empty:
                    self.logger.warning(f"No data collected for {timeframe}")
                    failed_timeframes.append(timeframe)
                    continue

                # Save data
                self.save_data(df, timeframe)

                # Generate data quality report
                self._generate_quality_report(df, timeframe)

                successful_timeframes.append(timeframe)
                self.logger.info(f"Successfully completed {timeframe}")

            except Exception as e:
                self.logger.error(f"Failed to collect {timeframe} data: {e}")
                failed_timeframes.append(timeframe)
                continue

        # Summary report
        self.logger.info(f"Data collection completed:")
        self.logger.info(
            f"  Successful: {len(successful_timeframes)} timeframes - {successful_timeframes}"
        )
        if failed_timeframes:
            self.logger.warning(
                f"  Failed: {len(failed_timeframes)} timeframes - {failed_timeframes}"
            )

        return {
            "successful": successful_timeframes,
            "failed": failed_timeframes
        }

    # REMOVED: collect_timeframes method - not needed for 2-5 minute collection

    # REMOVED: get_available_timeframes method - not needed for simple collection

    # REMOVED: collect_single_timeframe method - not needed for simple collection

    # REMOVED: show_usage_examples method - simplified to focus on collect_all_timeframes only

    def _generate_quality_report(self, df: pd.DataFrame, timeframe: str):
        """
        Generate data quality report

        Args:
            df: DataFrame to analyze
            timeframe: Timeframe string
        """
        report = {
            "timeframe": timeframe,
            "total_records": len(df),
            "date_range": {
                "start": df.index.min().isoformat(),
                "end": df.index.max().isoformat(),
            },
            "missing_values": df.isnull().sum().to_dict(),
            "price_statistics": {
                "min_price": df[["open", "high", "low", "close"]].min().min(),
                "max_price": df[["open", "high", "low", "close"]].max().max(),
                "mean_price": df[["open", "high", "low",
                                  "close"]].mean().mean(),
            },
            "volume_statistics": {
                "min_volume": df["volume"].min(),
                "max_volume": df["volume"].max(),
                "mean_volume": df["volume"].mean(),
            },
        }

        # Save report
        report_file = (
            self.logs_dir /
            f"quality_report_{timeframe}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        import json

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Quality report saved for {timeframe}")

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about collected data

        Returns:
            Dictionary with collection statistics
        """
        stats = {
            "data_files": [],
            "total_records": 0,
            "date_ranges": {},
            "file_sizes": {},
        }

        try:
            for file_path in self.data_dir.glob("*.parquet"):
                if file_path.is_file():
                    df = pd.read_parquet(file_path)
                    stats["data_files"].append(file_path.name)
                    stats["total_records"] += len(df)
                    stats["date_ranges"][file_path.name] = {
                        "start": df.index.min().isoformat(),
                        "end": df.index.max().isoformat(),
                        "records": len(df),
                    }
                    stats["file_sizes"][
                        file_path.name] = file_path.stat().st_size

        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")

        return stats


if __name__ == "__main__":
    # Simple BTC data collection - collect all timeframes at once
    print("🚀 Starting BTC data collection...")
    print("=" * 50)

    collector = BTCDataCollector(exchange_name="binance")

    # Check exchange health before starting
    if not collector._check_exchange_health():
        print("❌ Exchange health check failed. Please check your connection.")
        exit(1)

    print("✅ Exchange health check passed")
    print("📊 Collecting all timeframes (4h, 1d, 1w, 1M)...")
    print("⏱️  Estimated time: 2-5 minutes")
    print()

    # Collect all timeframes
    result = collector.collect_all_timeframes()

    # Display results
    print("\n" + "=" * 50)
    print("📈 Collection Results:")
    print("=" * 50)
    print(
        f"✅ Successful: {len(result['successful'])} timeframes - {result['successful']}"
    )
    if result['failed']:
        print(
            f"❌ Failed: {len(result['failed'])} timeframes - {result['failed']}"
        )

    # Get collection statistics
    stats = collector.get_collection_stats()
    print(f"\n📊 Total records collected: {stats['total_records']:,}")
    print(f"📁 Data files created: {len(stats['data_files'])}")
    print("=" * 50)

    if result['successful']:
        print("🎉 Data collection completed successfully!")
    else:
        print("⚠️  Data collection had issues. Check logs for details.")
    result_all = collector.collect_all_timeframes()

    # Display final results
    print("\n" + "=" * 50)
    print("Final Collection Summary:")
    print("=" * 50)
    print(
        f"All timeframes - Successful: {len(result_all['successful'])} timeframes"
    )

    # Get collection statistics
    stats = collector.get_collection_stats()
    print(f"\nTotal records collected: {stats['total_records']}")
    print("=" * 50)
