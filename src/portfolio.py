
from sklearn.impute import KNNImputer

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
        self.stock_names = set()
             
    # Collect latest stock data of a given stock
    def fetch_stock_data(self, stock_name: str):
        lookup_days = self.config.lookup_days
        
        end_date = datetime.date.today()
        start_date = end_date - timedelta(days=lookup_days)
        
        stock = yf.Ticker(stock_name)
        
        data = stock.history(start=start_date, end=end_date)
        
        data.reset_index(inplace=True)
        data['Symbol'] = stock_name
        
        return data
    
    
    # KNN imputation with num_neighbors = 5
    def KNN_imputation(self, X) -> pd.Series:
        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        imputer.fit(X)
        
        return imputer.transform(X)
        
        
    # Check if a given stock is valid
    def is_valid_stock(self, stock_name: str):
        try:
            data = self.fetch_stock_data(stock_name)
            
            if len(data) == 0:
                return False
            
            return True
        except:
            return False
    
    
    # Add a given stock to the portfolio
    def add_stock(self, stock_name: str) -> bool:
        if not self.is_valid_stock(stock_name):
            print(f'{stock_name} is invalid.')
            return False
        
        if stock_name in self.stock_names:
            print(f"Stock '{stock_name}' is already in your porfolio")
            return True
            
        self.stock_names.add(stock_name)
        print(f"Stock '{stock_name}' added to your porfolio")
                
        return True
        

    # Delete a given stock from the portfolio
    def delete_stock(self, stock_name: str) -> bool:
        if stock_name not in self.stock_names:
            print(f'Your porfolio does not have {stock_name}')
            
            return False
            
        self.stock_names.remove(stock_name)
        print(f"Stock '{stock_name}' deleted from your porfolio")
        
        return True
    

    # Display portfolio
    def display_portfolio(self) -> None:
        print(f'Displaying Portfolio\n')
        config, stock_names = self.config, self.stock_names
        
        for stock_name in stock_names:
            print(f'Stock {stock_name}')
            stock_info = self.fetch_stock_data(stock_name)
            
            print(stock_info.dropna())
            print()

    
    # Save data of each stock in the portfolio
    def save_portfolio(self) -> None:
        stock_data_local_dir = self.config.stock_data_local_dir
        stock_names = self.stock_names
        
        # Remove existing folder storing stock data from a previous portfolio
        if os.path.isdir(stock_data_local_dir):
            shutil.rmtree(stock_data_local_dir)
        
        # Save each stock data
        os.mkdir(stock_data_local_dir)
        
        for stock_name in stock_names:
            stock_info = self.fetch_stock_data(stock_name)
            stock_info.to_csv(f'{stock_data_local_dir}/{stock_name}.csv', index=False)
            
    
