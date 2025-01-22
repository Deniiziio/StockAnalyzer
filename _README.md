# Stock Analysis Tool

A comprehensive stock analysis tool that evaluates stocks based on fundamental, technical, and risk metrics to provide investment recommendations.

## Features

- Fundamental Analysis (P/E ratio, ROE, profit margins, etc.)
- Technical Analysis (Moving averages, RSI, MACD, Bollinger Bands)
- Risk Metrics (Volatility, Beta, Sharpe Ratio)
- News Sentiment Analysis
- Technical Charts (Price, RSI, MACD)
- Support for International Stocks

## Requirements

```bash
pip install yfinance pandas numpy requests beautifulsoup4 matplotlib scikit-learn
```

## Usage

1. Save both files (`main.py` and `StockAnalyzer.py`) in the same directory
2. Run the program:
```bash
python main.py
```
3. Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)
4. For international stocks, use exchange suffixes:
   - German stocks: .DE (e.g., RWE.DE)
   - UK stocks: .L (e.g., BP.L)
   - French stocks: .PA (e.g., AI.PA)

## Output

The tool provides:
1. Comprehensive analysis report
2. Overall score and recommendation
3. Fundamental metrics
4. Technical indicators
5. Risk analysis
6. Key strengths and concerns
7. Optional technical analysis charts

## Example

```bash
Enter stock ticker symbol: AAPL
```

## Notes

- Internet connection required for data fetching
- Some data might not be available for certain stocks
- Historical data used for calculations spans 1 year by default
