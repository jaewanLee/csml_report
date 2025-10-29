from pathlib import Path


def save_data_to_parquet(data, file_name):
    data_dir = Path('../../features')
    data_dir.mkdir(exist_ok=True)

    data.to_parquet(data_dir / f'{file_name}.parquet')

    return data
