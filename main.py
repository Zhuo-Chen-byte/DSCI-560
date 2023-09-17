from src.portfolio import Portfolio as portfolio
from src.stock_price_predictor import StockPricePredictor

from config import Config as config

from datetime import timedelta

import datetime

import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    # Duration of this simulated trading
    end_trading_date = datetime.date.today()
    
    trading_duration_in_days = input('Please enter the duration of this simulated trading: ')
        
    while not trading_duration_in_days.isdigit():
        trading_duration_in_days = input('Invalid input. Please enter the duration of this simulated trading: ')
    
    trading_duration_in_days = int(trading_duration_in_days)
    date_today = end_trading_date - timedelta(days=trading_duration_in_days)
    
    # Initial fund of this simulated trading
    initial_fund = input('Please enter the money you want to put into this game: ')
    
    while not initial_fund.isdigit():
        initial_fund = input('Invalid input. Please enter the money you want to put into this game: ')
    
    initial_fund = int(initial_fund)
    
    c = config(trading_duration_in_days, initial_fund)
    p = portfolio(c)
    spp = StockPricePredictor()
    
    print(f'An empty portfolio of ${initial_fund} created. Simulation opens ...\n')
    
    
    while date_today <= end_trading_date:
        print(f'Today is {date_today}')
        
        # Ask the user to enter the date range to include stock prices
        lookup_days = input("Please enter the number of days to train LSTM which predicts tomorrow's stock price: ")
        
        while not lookup_days.isdigit():
            lookup_days = input("Invalid input. Please enter the number of days to train LSTM which predicts tomorrow's stock price: ")
    
        lookup_days = int(lookup_days)
        
        # Create a portfolio and start trading
        p.config.lookup_days = lookup_days
        
        continue_trading_or_not = 'Y'
    
        while not continue_trading_or_not.lower() == 'n':
            # Long stocks
            stock_name = input('Please enter the stock you want to long (N / n to exit): ')
            
            while not stock_name.lower() == 'n':
                stock_info = p.fetch_stock_data(stock_name, date_today)
                price_predicted = spp.train(stock_info)
                
                print(f"Tomorrow's closing price shall be {price_predicted}")
                
                p.long_stock(stock_name, date_today)
                stock_name = input('Please enter the stock you want to long (N / n to exit): ')
    
            # Short stocks
            stock_name = input('Please enter the stock you want to short (N / n to exit): ')
    
            while not stock_name.lower() == 'n':
                stock_info = p.fetch_stock_data(stock_name, date_today)
                price_predicted = spp.train(stock_info)
                
                print(f"Tomorrow's closing price shall be {price_predicted}")
                
                p.short_stock(stock_name, date_today)
                stock_name = input('Please enter the stock you want to short (N / n to exit): ')
        
            # Display the portfolio
            display_portfolio_or_not = input('Display your portfolio? (Y / n): ')
        
            if not display_portfolio_or_not.lower() == 'n':
                p.display_portfolio(date_today)
        
            # Ask the user whether continuing trading
            continue_trading_or_not = input('Continue trading (Y / n) ? ')
    
        print('\nTrading closed.\n')
    
        # Display the portfolio after closing the trade
        display_portfolio_or_not = input('Display portfolio? (Y / n): ')
    
        if not display_portfolio_or_not.lower() == 'n':
            p.display_portfolio(date_today)
        
        date_today += timedelta(days=1)
    
    
    print('Simulation ends.')
    
    print(f'You initial fund is ${initial_fund}')
    print(f'After trading, you have ${p.config.fund}')
    
    if p.config.fund > initial_fund:
        print(f'You make {p.config.fund - initial_fund}')
    else:
        print(f'You lose {initial_fund - p.config.fund}')
