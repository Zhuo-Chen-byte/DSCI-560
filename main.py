from src.portfolio import Portfolio as portfolio
from config import Config as config


if __name__ == '__main__':
    # Ask the user to enter the date range to include stock prices
    days_lookback = input('Please enter the number of days of stock price you want to check: ')
        
    while not days_lookback.isdigit():
        days_lookback = input('Invalid input. Please enter the number of days of stock price you want to check: ')
    
    days_lookback = int(days_lookback)
    
    
    # Create a portfolio and start trading
    print('An empty portfolio created. Trading opens ...\n')
    c = config(days_lookback)
    p = portfolio(c)
    
    continue_trading_or_not = 'Y'
    
    while not continue_trading_or_not.lower() == 'n':
        # Add stocks to the portfolio
        stock_name = input('Please enter the you want to add to your portfolio (N / n to exit): ')
    
        while not stock_name.lower() == 'n':
            p.add_stock(stock_name)
            stock_name = input('Please enter the you want to add to your portfolio (N / n to exit): ')
    
        # Delete stocks from the portfolio
        stock_name = input('Please enter the stock you want to delete from your portfolio (N / n to exit): ')
    
        while not stock_name.lower() == 'n':
            p.delete_stock(stock_name)
            stock_name = input('Please enter the stock you want to delete from your portfolio (N / n to exit): ')
        
        # Display the portfolio
        display_portfolio_or_not = input('Display your portfolio? (Y / n): ')
        
        if not display_portfolio_or_not.lower() == 'n':
            p.display_portfolio()
        
        # Ask the user whether continuing trading
        continue_trading_or_not = input('Continue trading (Y / n) ? ')
    
    print('\nTrading closed.\n')
    
    # Display the portfolio after closing the trade
    display_portfolio_or_not = input('Display portfolio? (Y / n): ')
    
    if not display_portfolio_or_not.lower() == 'n':
        p.display_portfolio()
    
    
    # Save the portfolio information
    p.save_portfolio()
