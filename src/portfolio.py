
# from sklearn.impute import KNNImputer

import yfinance as yf
import pandas as pd

import datetime

import shutil
import os

from datetime import timedelta
from config import Config


class Portfolio:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.stock_volumes = dict()
    

    # Collect latest stock data of a given stock
    def fetch_stock_data(self, stock_name: str, date_today: str) -> pd.DataFrame:
        lookup_days = self.config.lookup_days
        
        start_date = date_today - timedelta(days=lookup_days)
        
        stock = yf.Ticker(stock_name)
        
        data = stock.history(start=start_date, end=date_today)
        
        data.reset_index(inplace=True)
        data['Symbol'] = stock_name
        
        return data
    
    
    # Get latest stock info
    def fetch_latest_stock_info(self, stock_name: str, date_today) -> pd.DataFrame:
        data = self.fetch_stock_data(stock_name, date_today)
        data = data[['Date', 'Close', 'Volume']]
        
        return data.iloc[-1]
        
    
    # KNN imputation with num_neighbors = 5
    def KNN_imputation(self, X) -> pd.Series:
        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        imputer.fit(X)
        
        return imputer.transform(X)
        
        
    # Check if a given stock is valid
    def is_valid_stock(self, stock_name: str, date_today) -> bool:
        try:
            data = self.fetch_stock_data(stock_name, date_today)
            
            if len(data) == 0:
                return False
            
            return True
        except:
            return False

    
    # Display a specific stock's latest close price given its name
    def display_current_stock_info(self, stock_name: str, date_today: str) -> None:
        stock_info = self.fetch_latest_stock_info(stock_name, date_today)
        
        print(f'Stock {stock_name}')
        print(stock_info)
        print()
        
    
    # Long a stock
    def long_stock(self, stock_name: str, date_today: str) -> bool:
        if not self.is_valid_stock(stock_name, date_today):
            print(f'{stock_name} is invalid.')
            return False
        
        self.display_current_stock_info(stock_name, date_today)
        
        latest_stock_price = self.fetch_latest_stock_info(stock_name, date_today)['Close']
        fund = self.config.fund
        
        # print(latest_stock_price)
        print(f'You have ${fund}')
        
        maximum_volumes_to_long = int(fund // latest_stock_price)
        
        volumes_to_long = input(f'Please enter the volumes you want to long (≤ {maximum_volumes_to_long}): ')

        while not volumes_to_long.isdigit():
            volumes_to_long = input('Invalid input. Please enter the volumes you want to long (≤ {maximum_volumes_to_long}): ')
        
        volumes_to_long = int(volumes_to_long)
        
        while latest_stock_price * volumes_to_long > fund:
            volumes_to_long = input(f"Spending exceeds your fund. Please enter a valid volumes you want to long (≤ {maximum_volumes_to_long}): ")
            
            while not volumes_to_long.isdigit():
                volumes_to_long = input('Invalid input. Please enter the volumes you want to long (≤ {maximum_volumes_to_long}): ')
            
            volumes_to_long = int(volumes_to_long)
        
        if volumes_to_long == 0:
            return
        
        self.config.fund -= latest_stock_price * volumes_to_long
        
        if stock_name not in self.stock_volumes:
            self.stock_volumes[stock_name] = 0
        
        self.stock_volumes[stock_name] += volumes_to_long
        self.display_portfolio(date_today)
        
        return True
        

    # Short a stock
    def short_stock(self, stock_name: str, date_today: str) -> bool:
        if stock_name not in self.stock_volumes:
            print(f'Your porfolio does not have {stock_name}')
            
            return False
        
        self.display_current_stock_info(stock_name, date_today)
        
        latest_stock_price = self.fetch_latest_stock_info(stock_name, date_today)['Close']
        fund = self.config.fund
        
        volumes_in_portfolio = self.stock_volumes[stock_name]
        
        volumes_to_short = input(f'Please enter the volume you want to short (≤ {volumes_in_portfolio}): ')

        while not volumes_to_short.isdigit():
            volumes_to_short = input('Invalid input. Please enter the volume you want to short (≤ {volumes_in_portfolio}): ')
        
        volumes_to_short = int(volumes_to_short)
        
        while volumes_to_short > volumes_in_portfolio:
            volumes_to_short = input(f"You do not have enough volume. Please enter a valid volume you want to short (≤ {volumes_in_portfolio}): ")
            
            while not volumes_to_short.isdigit():
                volumes_to_short = input('Invalid input. Please enter the volume you want to short (≤ {volumes_in_portfolio}): ')
            
            volumes_to_short = int(volumes_to_short)
                        
        self.stock_volumes[stock_name] -= volumes_to_short
        
        if self.stock_volumes[stock_name] == 0:
            del self.stock_volumes[stock_name]
        
        self.config.fund += latest_stock_price * volumes_to_short
        
        self.display_portfolio(date_today)
            
        return True
    
        
    # Display portfolio
    def display_portfolio(self, date_today) -> None:
        print(f'Displaying Portfolio ...\n')
        config, stock_volumes = self.config, self.stock_volumes
        
        print(f'Fund in your Portfolio:\n{config.fund}\n')
        
        print('Stocks in your portfolio: ')
        
        for stock_name in stock_volumes:
            print(f'Stock: {stock_name}')
            print(f'Volume in your Portfolio: {stock_volumes[stock_name]}')
            
            print(f'Current Market Info: ')
            self.display_current_stock_info(stock_name, date_today)
            
            print()
            
    
    # Save data of each stock in the portfolio
    def save_portfolio(self) -> None:
        stock_data_local_dir = self.config.stock_data_local_dir
        stock_volumes = self.stock_volumes
        
        # Remove existing folder storing stock data from a previous portfolio
        if os.path.isdir(stock_data_local_dir):
            shutil.rmtree(stock_data_local_dir)
        
        # Save each stock data
        os.mkdir(stock_data_local_dir)
        
        portfolio_info = list()
        
        portfolio_info.append(('Fund', self.config.fund))
        
        for stock_name in stock_volumes:
            portfolio_info.append((stock_name, stock_volumes[stock_name]))
        
        portfolio = pd.DataFrame(portfolio)
        portfolio.to_csv('portfolio.csv', index=False)
