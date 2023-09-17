import datetime
from datetime import timedelta


class Config:
    def __init__(self, trading_duration_in_days: int = 1, fund: int=10000, lookup_days: int=1):
        self.fund = fund
        self.trading_duration_in_days = trading_duration_in_days
        self.lookup_days = lookup_days
    
        self.stock_data_local_dir = 'Latest Stock Data'
        
        self.lstm_length = 10
        self.train_val_test_ratios = [0.5, 0.2, 0.3]
        self.batch_size = 10
