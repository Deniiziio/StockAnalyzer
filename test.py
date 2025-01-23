# Import required modules
from StockAnalyzer import StockAnalyzer
import yfinance as yf


def main():
    try:
        # Initialize the analyzer with a stock symbol
        print("Initializing analyzer...")
        analyzer = StockAnalyzer("AAPL")

        # Fetch stock data
        print("Fetching stock data...")
        analyzer.fetch_data()

        # Run the analysis
        print("Running analysis...")
        analyzer.run_analysis()

        # Generate and print the report
        print("\nAnalysis Report:")
        print("=" * 50)
        report = analyzer.generate_report()
        print(report)

    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install yfinance pandas numpy matplotlib scikit-learn requests beautifulsoup4")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("\nDetailed error information:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()