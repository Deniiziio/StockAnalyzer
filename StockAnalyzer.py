import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class StockAnalyzer:
    def __init__(self, ticker, session=None):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        if session:
            self.stock._session = session
        self.data = None
        self.fundamentals = None
        self.technical_signals = None
        self.risk_metrics = None
        
    def fetch_data(self, period="1y"):
        """Fetch historical data for the specified period"""
        self.data = self.stock.history(period=period)
        return self.data
    
    def analyze_fundamentals(self):
        """Analyze fundamental metrics"""
        try:
            info = self.stock.info
            self.fundamentals = {
                'Market Cap': info.get('marketCap'),
                'P/E Ratio': info.get('forwardPE'),
                'EPS': info.get('trailingEps'),
                'ROE': info.get('returnOnEquity'),
                'Profit Margin': info.get('profitMargins'),
                'Debt to Equity': info.get('debtToEquity'),
                'Current Ratio': info.get('currentRatio'),
                'Book Value': info.get('bookValue'),
                'Dividend Yield': info.get('dividendYield'),
                'Industry': info.get('industry')
            }
        except Exception as e:
            print(f"Error fetching fundamental data: {e}")
            self.fundamentals = None
        return self.fundamentals

    def calculate_technical_indicators(self):
        """Calculate technical indicators"""
        if self.data is None:
            self.fetch_data()
            
        df = self.data.copy()
        
        # Moving averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        self.technical_signals = {
            'Current Price': df['Close'].iloc[-1],
            'Price vs SMA50': df['Close'].iloc[-1] / df['SMA_50'].iloc[-1] - 1,
            'Price vs SMA200': df['Close'].iloc[-1] / df['SMA_200'].iloc[-1] - 1,
            'RSI': df['RSI'].iloc[-1],
            'MACD Signal': 'Buy' if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] else 'Sell',
            'BB Position': (df['Close'].iloc[-1] - df['BB_lower'].iloc[-1]) / (df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1])
        }
        
        return self.technical_signals

    def calculate_risk_metrics(self):
        """Calculate risk-related metrics"""
        try:
            if self.data is None:
                self.fetch_data()

            if self.data.empty:
                print("No data available for risk calculation")
                return None

            returns = self.data['Close'].pct_change().dropna()

            if len(returns) < 30:
                print("Insufficient data for risk calculation (need at least 30 days)")
                return None

            volatility = returns.std() * np.sqrt(252)
            max_drawdown = (self.data['Close'] / self.data['Close'].expanding().max() - 1).min()
            beta = self.calculate_beta()
            sharpe = self.calculate_sharpe_ratio(returns)
            var_95 = np.percentile(returns, 5)

            self.risk_metrics = {
                'Volatility': float(volatility),
                'Max Drawdown': float(max_drawdown),
                'Beta': float(beta),
                'Sharpe Ratio': float(sharpe),
                'Value at Risk (95%)': float(var_95)
            }

            # Handle any NaN values
            for key in self.risk_metrics:
                if np.isnan(self.risk_metrics[key]):
                    self.risk_metrics[key] = 1.0 if key == 'Beta' else 0.0

            return self.risk_metrics

        except Exception as e:
            print(f"Error calculating risk metrics: {str(e)}")
            self.risk_metrics = {
                'Volatility': 0.0,
                'Max Drawdown': 0.0,
                'Beta': 1.0,
                'Sharpe Ratio': 0.0,
                'Value at Risk (95%)': 0.0
            }
            return self.risk_metrics

    def format_risk_metrics(self):
        """Format risk metrics for display"""
        if not self.risk_metrics:
            return "Risk metrics not available"

        formatted_metrics = []
        for key, value in self.risk_metrics.items():
            if isinstance(value, (int, float)):
                if key in ['Volatility', 'Max Drawdown', 'Value at Risk (95%)']:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            formatted_metrics.append(f"{key}: {formatted_value}")

        return "\n".join(formatted_metrics)

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
        """Calculate Sharpe Ratio"""
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def calculate_beta(self):
        """Calculate beta relative to S&P 500"""
        try:
            # Get S&P 500 data for the same period as the stock data
            spy = yf.download('^GSPC',
                              start=self.data.index[0].strftime('%Y-%m-%d'),
                              end=self.data.index[-1].strftime('%Y-%m-%d'),
                              progress=False)['Close']

            # Align the data
            common_dates = self.data.index.intersection(spy.index)
            if len(common_dates) < 30:  # Need at least 30 days of data
                return 1.0  # Return market beta if not enough data

            stock_prices = self.data.loc[common_dates, 'Close']
            market_prices = spy.loc[common_dates]

            # Calculate returns
            stock_returns = stock_prices.pct_change().dropna()
            market_returns = market_prices.pct_change().dropna()

            # Calculate beta using covariance method
            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)

            if market_variance == 0:
                return 1.0  # Return market beta if variance is zero

            beta = covariance / market_variance
            return beta if not np.isnan(beta) else 1.0

        except Exception as e:
            print(f"Error calculating beta: {str(e)}")
            return 1.0  # Return market beta if calculation fails

    def get_news_sentiment(self):
        """Get recent news sentiment using a simple approach"""
        try:
            news = self.stock.news
            if not news:
                return None
            
            # Simple sentiment scoring based on title keywords
            positive_words = ['up', 'rise', 'gain', 'positive', 'growth', 'profit']
            negative_words = ['down', 'fall', 'drop', 'negative', 'loss', 'debt']
            
            sentiment_scores = []
            for article in news[:10]:  # Analyze last 10 news items
                title = article['title'].lower()
                score = sum(word in title for word in positive_words) - sum(word in title for word in negative_words)
                sentiment_scores.append(score)
            
            return {
                'Average Sentiment': np.mean(sentiment_scores),
                'Recent News Count': len(news),
                'Sentiment Trend': 'Positive' if np.mean(sentiment_scores) > 0 else 'Negative'
            }
        except:
            return None

    def generate_recommendation(self):
        """Generate a comprehensive analysis and recommendation"""
        if any(x is None for x in [self.fundamentals, self.technical_signals, self.risk_metrics]):
            print("Running all analyses first...")
            self.analyze_fundamentals()
            self.calculate_technical_indicators()
            self.calculate_risk_metrics()
        
        # Score different aspects (0-100)
        fundamental_score = self.score_fundamentals()
        technical_score = self.score_technical()
        risk_score = self.score_risk()
        
        # Weighted average (adjust weights based on preference)
        total_score = (
            fundamental_score * 0.4 +
            technical_score * 0.4 +
            risk_score * 0.2
        )
        
        recommendation = {
            'Total Score': total_score,
            'Fundamental Score': fundamental_score,
            'Technical Score': technical_score,
            'Risk Score': risk_score,
            'Recommendation': self.get_recommendation_text(total_score),
            'Key Strengths': self.get_key_points(True),
            'Key Concerns': self.get_key_points(False)
        }
        
        return recommendation

    def score_fundamentals(self):
        """Score fundamental metrics"""
        if not self.fundamentals:
            return 0
        
        score = 0
        max_score = 0
        
        metrics = {
            'P/E Ratio': {'weight': 2, 'ideal_range': (5, 25)},
            'ROE': {'weight': 2, 'min_good': 0.15},
            'Profit Margin': {'weight': 1.5, 'min_good': 0.1},
            'Current Ratio': {'weight': 1, 'ideal_range': (1.5, 3)},
            'Debt to Equity': {'weight': 1.5, 'max_good': 2}
        }
        
        for metric, criteria in metrics.items():
            value = self.fundamentals.get(metric)
            if value is None:
                continue
                
            max_score += criteria['weight'] * 100
            
            if 'ideal_range' in criteria:
                min_val, max_val = criteria['ideal_range']
                if min_val <= value <= max_val:
                    score += criteria['weight'] * 100
                elif value < min_val:
                    score += criteria['weight'] * (value / min_val * 100)
                else:
                    score += criteria['weight'] * (max_val / value * 100)
            
            elif 'min_good' in criteria:
                if value >= criteria['min_good']:
                    score += criteria['weight'] * 100
                else:
                    score += criteria['weight'] * (value / criteria['min_good'] * 100)
            
            elif 'max_good' in criteria:
                if value <= criteria['max_good']:
                    score += criteria['weight'] * 100
                else:
                    score += criteria['weight'] * (criteria['max_good'] / value * 100)
        
        return (score / max_score * 100) if max_score > 0 else 0

    def score_technical(self):
        """Score technical indicators"""
        if not self.technical_signals:
            return 0
        
        score = 0
        
        # Price vs Moving Averages
        if self.technical_signals['Price vs SMA50'] > 0:
            score += 20
        if self.technical_signals['Price vs SMA200'] > 0:
            score += 20
        
        # RSI
        rsi = self.technical_signals['RSI']
        if 40 <= rsi <= 60:
            score += 20
        elif 30 <= rsi <= 70:
            score += 10
        
        # MACD
        if self.technical_signals['MACD Signal'] == 'Buy':
            score += 20
        
        # Bollinger Bands
        bb_pos = self.technical_signals['BB Position']
        if 0.3 <= bb_pos <= 0.7:
            score += 20
        elif 0.1 <= bb_pos <= 0.9:
            score += 10
        
        return score

    def score_risk(self):
        """Score risk metrics"""
        if not self.risk_metrics:
            return 0
        
        score = 100
        
        # Volatility penalty
        vol = self.risk_metrics['Volatility']
        if vol > 0.4:  # High volatility
            score -= 30
        elif vol > 0.25:  # Moderate volatility
            score -= 15
        
        # Max Drawdown penalty
        drawdown = self.risk_metrics['Max Drawdown']
        if drawdown < -0.5:  # Severe drawdown
            score -= 30
        elif drawdown < -0.3:  # Moderate drawdown
            score -= 15
        
        # Beta penalty/bonus
        beta = self.risk_metrics['Beta']
        if beta > 1.5:  # High beta
            score -= 20
        elif beta < 0.8:  # Low beta
            score += 10
        
        # Sharpe Ratio bonus
        sharpe = self.risk_metrics['Sharpe Ratio']
        if sharpe > 1:
            score += 20
        elif sharpe > 0.5:
            score += 10
        
        return max(0, min(100, score))

    def get_recommendation_text(self, score):
        """Generate recommendation text based on score"""
        if score >= 80:
            return "Strong Buy - The stock shows excellent fundamentals, technical strength, and manageable risk."
        elif score >= 60:
            return "Buy - The stock shows good overall characteristics with some room for improvement."
        elif score >= 40:
            return "Hold - The stock shows mixed signals. Consider watching for improvements before investing."
        elif score >= 20:
            return "Sell - The stock shows significant weaknesses and might be risky."
        else:
            return "Strong Sell - The stock shows major red flags across multiple metrics."

    def get_key_points(self, strengths=True):
        """Identify key strengths or concerns"""
        points = []

        if self.fundamentals:
            # P/E Ratio analysis
            pe = self.fundamentals.get('P/E Ratio')
            if pe:
                if strengths and 5 <= pe <= 25:
                    points.append(f"Attractive P/E ratio of {pe:.2f}")
                elif not strengths and (pe < 5 or pe > 25):
                    points.append(
                        f"Concerning P/E ratio of {pe:.2f} - {'potentially overvalued' if pe > 25 else 'might indicate underlying issues'}")

            # ROE analysis
            roe = self.fundamentals.get('ROE')
            if roe:
                if strengths and roe >= 0.15:
                    points.append(f"Strong Return on Equity of {roe:.1%}")
                elif not strengths and roe < 0.10:
                    points.append(
                        f"Low Return on Equity of {roe:.1%} indicating inefficient use of shareholder capital")

            # Debt analysis
            debt = self.fundamentals.get('Debt to Equity')
            if debt:
                if strengths and debt < 0.5:
                    points.append(f"Healthy debt level with Debt/Equity ratio of {debt:.2f}")
                elif not strengths and debt > 1.5:
                    points.append(f"High debt level with Debt/Equity ratio of {debt:.2f} - potential financial risk")

            # Profit Margin analysis
            margin = self.fundamentals.get('Profit Margin')
            if margin:
                if strengths and margin > 0.15:
                    points.append(f"Strong profit margin of {margin:.1%}")
                elif not strengths and margin < 0.05:
                    points.append(f"Low profit margin of {margin:.1%} indicating potential operational inefficiency")

        if self.technical_signals:
            # MACD Signal analysis
            if strengths and self.technical_signals['MACD Signal'] == 'Buy':
                points.append("Positive MACD signal indicating upward momentum")
            elif not strengths and self.technical_signals['MACD Signal'] == 'Sell':
                points.append("Negative MACD signal suggesting downward pressure")

            # Moving Average analysis
            sma50 = self.technical_signals.get('Price vs SMA50')
            sma200 = self.technical_signals.get('Price vs SMA200')
            if sma50 and sma200:
                if strengths and sma50 > 0 and sma200 > 0:
                    points.append("Strong uptrend with price above both 50-day and 200-day moving averages")
                elif not strengths and sma50 < -0.05 and sma200 < -0.05:
                    points.append("Significant downtrend with price below major moving averages")

            # RSI analysis
            rsi = self.technical_signals.get('RSI')
            if rsi:
                if strengths and 40 <= rsi <= 60:
                    points.append(f"Balanced RSI at {rsi:.1f} indicating steady momentum")
                elif not strengths:
                    if rsi > 70:
                        points.append(f"Overbought conditions with RSI at {rsi:.1f}")
                    elif rsi < 30:
                        points.append(f"Oversold conditions with RSI at {rsi:.1f}")

        if self.risk_metrics:
            # Beta analysis
            beta = self.risk_metrics.get('Beta')
            if beta:
                if strengths and 0.8 <= beta <= 1.2:
                    points.append(f"Balanced market sensitivity (Beta: {beta:.2f})")
                elif not strengths and (beta > 1.5 or beta < 0.5):
                    points.append(
                        f"{'High' if beta > 1.5 else 'Low'} market sensitivity (Beta: {beta:.2f}) - {'increased volatility risk' if beta > 1.5 else 'limited market participation'}")

            # Volatility analysis
            vol = self.risk_metrics.get('Volatility')
            if vol:
                if strengths and vol < 0.20:
                    points.append(f"Low volatility of {vol:.1%} indicating price stability")
                elif not strengths and vol > 0.40:
                    points.append(f"High volatility of {vol:.1%} suggesting increased risk")

            # Maximum Drawdown analysis
            drawdown = self.risk_metrics.get('Max Drawdown')
            if drawdown:
                if not strengths and drawdown < -0.30:
                    points.append(f"Significant maximum drawdown of {drawdown:.1%} indicating historical price risk")

            # Sharpe Ratio analysis
            sharpe = self.risk_metrics.get('Sharpe Ratio')
            if sharpe:
                if strengths and sharpe > 1:
                    points.append(f"Strong risk-adjusted returns with Sharpe ratio of {sharpe:.2f}")
                elif not strengths and sharpe < 0.5:
                    points.append(f"Poor risk-adjusted returns with Sharpe ratio of {sharpe:.2f}")

        return sorted(points, key=lambda x: len(x), reverse=True)[:3]  # Return top 3 most detailed points

    def plot_technical_analysis(self, show_plot=False):
        """
        Create technical analysis visualization

        Parameters:
        show_plot (bool): Whether to display the plot (default: False)
        """
        if self.data is None:
            self.fetch_data()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Price and Moving Averages
        ax1.plot(self.data.index, self.data['Close'], label='Price')
        ax1.plot(self.data.index, self.data['Close'].rolling(window=50).mean(), label='50-day MA')
        ax1.plot(self.data.index, self.data['Close'].rolling(window=200).mean(), label='200-day MA')
        ax1.set_title(f'{self.ticker} Technical Analysis')
        ax1.set_ylabel('Price')
        ax1.legend()

        # RSI
        ax2.plot(self.data.index, self.data['Close'].rolling(window=14).apply(self.calculate_rsi))
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.set_ylabel('RSI')

        # MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        ax3.plot(self.data.index, macd, label='MACD')
        ax3.plot(self.data.index, signal, label='Signal')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_ylabel('MACD')
        ax3.legend()

        plt.tight_layout()

        if show_plot:
            plt.show()
        else:
            plt.close(fig)  # Close the figure if not showing it

    def calculate_rsi(self, data):
        """Helper function to calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_report(self):
        """Generate a comprehensive analysis report"""
        if self.data is None:
            self.fetch_data()

        # Run all analyses
        self.analyze_fundamentals()
        self.calculate_technical_indicators()
        self.calculate_risk_metrics()
        recommendation = self.generate_recommendation()
        news_sentiment = self.get_news_sentiment()

        report = []
        report.append(f"\nStock Analysis Report for {self.ticker}")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report.append("\n1. SUMMARY")
        report.append("-" * 50)
        report.append(f"Overall Score: {recommendation['Total Score']:.1f}/100")
        report.append(f"Recommendation: {recommendation['Recommendation']}")

        report.append("\n2. FUNDAMENTAL ANALYSIS")
        report.append("-" * 50)
        report.append(f"Score: {recommendation['Fundamental Score']:.1f}/100")

        if self.fundamentals:
            report.append("\nKey Metrics:")
            for metric, value in self.fundamentals.items():
                if value is not None:
                    if isinstance(value, float):
                        report.append(f"- {metric}: {value:.2f}")
                    else:
                        report.append(f"- {metric}: {value}")

        report.append("\n3. TECHNICAL ANALYSIS")
        report.append("-" * 50)
        report.append(f"Score: {recommendation['Technical Score']:.1f}/100")

        if self.technical_signals:
            report.append("\nCurrent Signals:")
            for signal, value in self.technical_signals.items():
                if isinstance(value, float):
                    report.append(f"- {signal}: {value:.2f}")
                else:
                    report.append(f"- {signal}: {value}")

        report.append("\n4. RISK ANALYSIS")
        report.append("-" * 50)
        report.append(f"Score: {recommendation['Risk Score']:.1f}/100")

        if self.risk_metrics:
            report.append("\nRisk Metrics:")
            report.append(self.format_risk_metrics())

        if news_sentiment:
            report.append("\n5. NEWS SENTIMENT")
            report.append("-" * 50)
            report.append(f"- Average Sentiment: {news_sentiment['Average Sentiment']:.2f}")
            report.append(f"- Recent News Count: {news_sentiment['Recent News Count']}")
            report.append(f"- Sentiment Trend: {news_sentiment['Sentiment Trend']}")

        report.append("\n6. KEY POINTS")
        report.append("-" * 50)

        if recommendation['Key Strengths']:
            report.append("\nStrengths:")
            for strength in recommendation['Key Strengths']:
                report.append(f"+ {strength}")

        if recommendation['Key Concerns']:
            report.append("\nConcerns:")
            for concern in recommendation['Key Concerns']:
                report.append(f"- {concern}")

        return "\n".join(report)

# Example usage
def analyze_stock(ticker):
    """
    Analyze a stock and generate a comprehensive report
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
    """
    analyzer = StockAnalyzer(ticker)
    
    try:
        # Fetch data
        print(f"Analyzing {ticker}...")
        analyzer.fetch_data()
        
        # Generate and print report
        report = analyzer.generate_report()
        print(report)
        
        # Create technical analysis plot
        analyzer.plot_technical_analysis()
        
        return analyzer
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None

if __name__ == "__main__":
    # Example: Analyze Apple stock
    ticker = "AAPL"
    analyzer = analyze_stock(ticker)