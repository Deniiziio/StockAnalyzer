import yfinance as yf
import requests
from typing import Optional
from StockAnalyzer import StockAnalyzer


def get_valid_ticker(ticker: str) -> Optional[str]:
    """
    Attempt to find the correct ticker symbol format with error handling
    """
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


def initialize_yfinance_session() -> requests.Session:
    """Initialize a yfinance session with proper headers"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,'
                  'image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    })
    return session


def display_troubleshooting_steps() -> None:
    """Display common troubleshooting steps for errors"""
    print("\nTroubleshooting steps:")
    print("1. Check your internet connection")
    print("2. Try again in a few minutes (API rate limits)")
    print("3. Verify the ticker symbol on finance.yahoo.com")
    print("4. Try using a VPN if you're having regional access issues")


def display_welcome_message() -> None:
    """Display welcome message and usage instructions"""
    print("Welcome to the Stock Analyzer!")
    print("This tool can analyze stocks from various international exchanges.")
    print("For international stocks, you can add exchange suffixes:")
    print("- German stocks: .DE (e.g., RWE.DE)")
    print("- UK stocks: .L (e.g., BP.L)")
    print("- French stocks: .PA (e.g., AI.PA)")


def main() -> None:
    """Main function to run the stock analyzer"""
    # Initialize a session for yfinance
    session = initialize_yfinance_session()

    while True:
        # Get user input for stock ticker
        ticker = input("\nEnter stock ticker symbol (e.g., AAPL, MSFT, GOOGL) "
                       "or 'quit' to exit: ").upper()

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
            continue

        # Create analyzer instance with the session
        print(f"\nInitializing analysis for {valid_ticker}...")

        try:
            analyzer = StockAnalyzer(valid_ticker, session=session)

            # Run core analyses
            print("Running comprehensive analysis...")
            analyzer.run_analysis()

            # Generate and print report
            print("\nGenerating report...")
            report = analyzer.generate_report()
            print("\n" + "=" * 50)
            print(report)
            print("=" * 50)

            # Handle visualization options
            while True:
                viz_choice = input("\nVisualization options:\n"
                                   "1. Technical analysis plot\n"
                                   "2. Continue to next analysis\n"
                                   "Choose an option (1-2): ").strip()

                if viz_choice == '1':
                    print("\nGenerating technical analysis plot...")
                    analyzer.plot_technical_analysis(show_plot=True)
                    break
                elif viz_choice == '2':
                    break
                else:
                    print("Invalid choice. Please select 1 or 2.")

            # Ask if user wants to analyze another stock
            continue_choice = input("\nWould you like to analyze another stock? (yes/no): ").lower()
            if continue_choice != 'yes':
                break

        except Exception as e:
            print(f"\nError accessing data for {valid_ticker}: {str(e)}")
            display_troubleshooting_steps()

            retry_choice = input("\nWould you like to try another stock? (yes/no): ").lower()
            if retry_choice != 'yes':
                break


def add_custom_analysis_module(analyzer: StockAnalyzer, module_class: type, module_name: str) -> bool:
    """
    Add a custom analysis module to the analyzer

    Parameters:
        analyzer (StockAnalyzer): The analyzer instance
        module_class (type): The class of the custom module
        module_name (str): Name for the new module

    Returns:
        bool: True if module was successfully added, False otherwise
    """
    try:
        module_instance = module_class()
        analyzer.register_analysis_module(module_name, module_instance)
        return True
    except Exception as e:
        print(f"Error adding custom module: {str(e)}")
        return False


if __name__ == "__main__":
    display_welcome_message()
    main()