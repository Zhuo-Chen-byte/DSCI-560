import datetime
from datetime import timedelta


class Config:
    def __init__(self, lookup_days: int=1):
        self.lookup_days = lookup_days
        self.stock_data_local_dir = 'Latest Stock Data'
