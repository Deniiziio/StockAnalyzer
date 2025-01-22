import yfinance as yf
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from StockAnalyzer import StockAnalyzer

def get_valid_ticker(ticker):
    """
    Attempt to find the correct ticker symbol format with error handling
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

    # Try direct verification first
    try:
        test_ticker = yf.Ticker(ticker)
        hist = test_ticker.history(period="1d")
        if not hist.empty:
            info = test_ticker.info
            print(f"Found valid ticker: {ticker}")
            print(f"Company Name: {info.get('longName', 'N/A')}")
            print(f"Exchange: {info.get('exchange', 'N/A')}")
            return ticker
    except:
        pass

    # Common exchange suffixes
    exchanges = [
        '',  # US stocks
        '.DE',  # German stocks
        '.F',  # Frankfurt
        '.L',  # London
        '.PA',  # Paris
        '.MI',  # Milan
        '.MC',  # Madrid
        '.AS',  # Amsterdam
        '.TO',  # Toronto
        '.HK'  # Hong Kong
    ]

    print(f"Searching for {ticker} across major exchanges...")

    for suffix in exchanges:
        try:
            test_ticker = f"{ticker}{suffix}"
            stock = yf.Ticker(test_ticker)

            # Try to get historical data (more reliable than info)
            hist = stock.history(period="1d")
            if not hist.empty:
                info = stock.info
                print(f"Found valid ticker: {test_ticker}")
                print(f"Company Name: {info.get('longName', 'N/A')}")
                print(f"Exchange: {info.get('exchange', 'N/A')}")
                return test_ticker

        except Exception as e:
            continue

    return None
def initialize_yfinance_session():
    """
    Initialize a yfinance session with proper headers
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    })
    return session
def main():
    # Initialize a session for yfinance
    session = initialize_yfinance_session()

    while True:
        # Get user input for stock ticker
        ticker = input("\nEnter stock ticker symbol (e.g., AAPL, MSFT, GOOGL) or 'quit' to exit: ").upper()

        if ticker.lower() == 'quit':
            break

        # Try to find the correct ticker format
        valid_ticker = get_valid_ticker(ticker)

        if valid_ticker is None:
            print(f"\nError: Could not find valid data for {ticker}")
            print("Please check if the ticker is correct and try again.")
            print("For international stocks, you might need to specify the exchange:")
            print("Examples:")
            print("- German stocks: RWE.DE, SAP.DE, BMW.DE")
            print("- UK stocks: BP.L, HSBA.L")
            print("- French stocks: AI.PA, OR.PA")
            continue

        # Create analyzer instance with the session
        print(f"\nInitializing analysis for {valid_ticker}...")

        try:
            stock = yf.Ticker(valid_ticker)
            stock._session = session  # Use our custom session
            analyzer = StockAnalyzer(valid_ticker, session=session)

            # Fetch data
            print("Fetching historical data...")
            data = analyzer.fetch_data()

            if data.empty:
                print(f"Error: No historical data available for {valid_ticker}")
                continue

            # Run analyses
            print("Analyzing fundamentals...")
            analyzer.analyze_fundamentals()

            print("Calculating technical indicators...")
            analyzer.calculate_technical_indicators()

            print("Calculating risk metrics...")
            analyzer.calculate_risk_metrics()

            # Generate and print report
            print("\nGenerating report...")
            report = analyzer.generate_report()
            print("\n" + "=" * 50)
            print(report)
            print("=" * 50)

            # Ask if user wants to see the technical analysis plot
            plot_choice = input("\nWould you like to see the technical analysis plot? (yes/no): ").lower().strip()
            if plot_choice == 'yes':
                print("\nGenerating technical analysis plot...")
                analyzer.plot_technical_analysis(show_plot=True)
            else:
                analyzer.plot_technical_analysis(show_plot=False)

            # Ask if user wants to analyze another stock
            choice = input("\nWould you like to analyze another stock? (yes/no): ").lower()
            if choice != 'yes':
                break

        except Exception as e:
            print(f"\nError accessing data for {valid_ticker}: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Check your internet connection")
            print("2. Try again in a few minutes (API rate limits)")
            print("3. Verify the ticker symbol on finance.yahoo.com")
            print("4. Try using a VPN if you're having regional access issues")
if __name__ == "__main__":
    print("Welcome to the Stock Analyzer!")
    print("This tool can analyze stocks from various international exchanges.")
    print("For international stocks, you can add exchange suffixes:")
    print("- German stocks: .DE (e.g., RWE.DE)")
    print("- UK stocks: .L (e.g., BP.L)")
    print("- French stocks: .PA (e.g., AI.PA)")
    main()