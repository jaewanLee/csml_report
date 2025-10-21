"""
Exchange configuration for ccxt data collection
"""

# Exchange settings
EXCHANGE_CONFIG = {
    'binance': {
        'id': 'binance',
        'apiKey': None,  # Not needed for public data
        'secret': None,  # Not needed for public data
        'sandbox': False,
        'rateLimit': 1200,  # Respect rate limits
    },
    'coinbase': {
        'id': 'coinbasepro',
        'apiKey': None,
        'secret': None,
        'sandbox': False,
        'rateLimit': 1000,
    }
}

# Data collection settings - TIMEFRAME-SPECIFIC DATE RANGES
COLLECTION_CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframes': ['4h', '1d', '1w', '1M'],
    # Timeframe-specific date ranges for efficient data collection
    'timeframe_dates': {
        '4h': {
            'start_date': '2020-03-01T00:00:00Z',  # Keep current for buffer
            'end_date': '2025-10-19T23:59:59Z'
        },
        '1d': {
            'start_date': '2020-01-08T00:00:00Z',  # 2020-05-12 - 125 days
            'end_date': '2025-10-19T23:59:59Z'
        },
        '1w': {
            'start_date': '2019-02-11T00:00:00Z',  # 2020-05-12 - 65 weeks
            'end_date': '2025-10-19T23:59:59Z'
        },
        '1M': {
            'start_date': '2018-04-12T00:00:00Z',  # 2020-05-12 - 25 months
            'end_date': '2025-10-19T23:59:59Z'
        }
    },
    'batch_size': 1000,  # Records per API call
    'retry_attempts': 3,
    'retry_delay': 1,  # seconds
}

# Data validation settings
VALIDATION_CONFIG = {
    'min_records_per_day': 6,  # For 4h timeframe
    'max_price_change': 0.5,  # 50% max change between consecutive records
    'required_columns':
    ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
    'duplicate_threshold':
    0.95,  # Similarity threshold for duplicate detection
    # Timeframe-specific validation
    'timeframe_validation': {
        '4h': {
            'min_records_per_day': 6
        },
        '1d': {
            'min_records_per_day': 1
        },
        '1w': {
            'min_records_per_day': 1
        },
        '1M': {
            'min_records_per_day': 1
        }  # Monthly data - at least 1 record per month
    }
}
