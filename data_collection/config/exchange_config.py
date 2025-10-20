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

# Data collection settings
COLLECTION_CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframes': ['4h', '1d', '1w'],
    'start_date': '2020-05-12T00:00:00Z',
    'end_date': None,  # Current date
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
}
