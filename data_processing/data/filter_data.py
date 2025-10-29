import pandas as pd


def filter_data(data, start_date, end_date):
    return data[(data.index >= start_date) & (data.index <= end_date)]
