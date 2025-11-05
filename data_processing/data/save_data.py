from pathlib import Path


def save_data_to_parquet(data, file_name):
    # 프로젝트 루트 기준으로 features 폴더 경로 설정
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'features'
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / f'{file_name}.parquet'
    data.to_parquet(file_path)
    print(f"✅ Saved {file_name}.parquet to {file_path}")

    return data
