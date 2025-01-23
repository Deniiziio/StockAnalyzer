# StockAnalyzer.py
from abc import ABC, abstractmethod
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import requests
from typing import Dict, List, Optional, Any, Tuple


class AnalysisModule(ABC):
    """Base class for analysis modules"""

    @abstractmethod
    def analyze(self, data: pd.DataFrame, stock: yf.Ticker) -> Dict[str, Any]:
        """Perform analysis and return results"""
        pass

    @abstractmethod
    def score(self, results: Dict[str, Any]) -> float:
        """Calculate score based on analysis results"""
        pass

    @abstractmethod
    def get_key_points(self, results: Dict[str, Any], strengths: bool = True) -> List[str]:
        """Get key strengths or concerns based on analysis results"""
        pass


class FundamentalAnalysis(AnalysisModule):
    def analyze(self, data: pd.DataFrame, stock: yf.Ticker) -> Dict[str, Any]:
        try:
            info = stock.info
            return {
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
            print(f"Error in fundamental analysis: {e}")
            return {}

    def score(self, results: Dict[str, Any]) -> float:
        if not results:
            return 0

        metrics = {
            'P/E Ratio': {'weight': 2, 'ideal_range': (5, 25)},
            'ROE': {'weight': 2, 'min_good': 0.15},
            'Profit Margin': {'weight': 1.5, 'min_good': 0.1},
            'Current Ratio': {'weight': 1, 'ideal_range': (1.5, 3)},
            'Debt to Equity': {'weight': 1.5, 'max_good': 2}
        }

        return self._calculate_score(results, metrics)

    def _calculate_score(self, results: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]) -> float:
        score = 0
        max_score = 0

        for metric, criteria in metrics.items():
            value = results.get(metric)
            if value is None:
                continue

            max_score += criteria['weight'] * 100
            score += self._calculate_metric_score(value, criteria)

        return (score / max_score * 100) if max_score > 0 else 0

    def _calculate_metric_score(self, value: float, criteria: Dict[str, Any]) -> float:
        if 'ideal_range' in criteria:
            min_val, max_val = criteria['ideal_range']
            if min_val <= value <= max_val:
                return criteria['weight'] * 100
            return criteria['weight'] * (min(value / min_val, max_val / value) * 100)

        elif 'min_good' in criteria:
            return criteria['weight'] * min(value / criteria['min_good'], 1) * 100

        elif 'max_good' in criteria:
            return criteria['weight'] * min(criteria['max_good'] / value, 1) * 100

        return 0

    def get_key_points(self, results: Dict[str, Any], strengths: bool = True) -> List[str]:
        """Extract key insights from fundamental analysis results"""
        points = []

        if not results:
            return points

        # P/E Analysis
        pe_ratio = results.get('P/E Ratio')
        if pe_ratio is not None:
            if strengths and 5 <= pe_ratio <= 25:
                points.append(f"Attractive P/E ratio of {pe_ratio:.2f}")
            elif not strengths and (pe_ratio < 5 or pe_ratio > 25):
                points.append(
                    f"{'High' if pe_ratio > 25 else 'Low'} P/E ratio of {pe_ratio:.2f} indicating potential {'overvaluation' if pe_ratio > 25 else 'underlying issues'}")

        # ROE Analysis
        roe = results.get('ROE')
        if roe is not None:
            if strengths and roe >= 0.15:
                points.append(f"Strong Return on Equity of {roe:.1%}")
            elif not strengths and roe < 0.10:
                points.append(f"Low Return on Equity of {roe:.1%} suggesting inefficient capital use")

        # Profit Margin Analysis
        margin = results.get('Profit Margin')
        if margin is not None:
            if strengths and margin > 0.15:
                points.append(f"Healthy profit margin of {margin:.1%}")
            elif not strengths and margin < 0.05:
                points.append(f"Low profit margin of {margin:.1%} indicating potential efficiency issues")

        return sorted(points, key=len, reverse=True)


class TechnicalAnalysis(AnalysisModule):
    def analyze(self, data: pd.DataFrame, stock: yf.Ticker) -> Dict[str, Any]:
        if data.empty:
            return {}

        df = data.copy()
        self._calculate_indicators(df)

        return {
            'Current Price': df['Close'].iloc[-1],
            'Price vs SMA50': df['Close'].iloc[-1] / df['SMA_50'].iloc[-1] - 1,
            'Price vs SMA200': df['Close'].iloc[-1] / df['SMA_200'].iloc[-1] - 1,
            'RSI': df['RSI'].iloc[-1],
            'MACD Signal': 'Buy' if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] else 'Sell',
            'BB Position': (df['Close'].iloc[-1] - df['BB_lower'].iloc[-1]) /
                           (df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1])
        }

    def _calculate_indicators(self, df: pd.DataFrame) -> None:
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

    def score(self, results: Dict[str, Any]) -> float:
        if not results:
            return 0

        score = 0
        score_criteria = {
            'moving_averages': {'weight': 40, 'threshold': 0},
            'rsi': {'weight': 20, 'ranges': [(40, 60, 20), (30, 70, 10)]},
            'macd': {'weight': 20, 'buy_score': 20},
            'bollinger': {'weight': 20, 'ranges': [(0.3, 0.7, 20), (0.1, 0.9, 10)]}
        }

        return self._calculate_technical_score(results, score_criteria)

    def _calculate_technical_score(self, results: Dict[str, Any], criteria: Dict[str, Dict[str, Any]]) -> float:
        score = 0

        # Moving Averages
        if results['Price vs SMA50'] > criteria['moving_averages']['threshold']:
            score += criteria['moving_averages']['weight'] / 2
        if results['Price vs SMA200'] > criteria['moving_averages']['threshold']:
            score += criteria['moving_averages']['weight'] / 2

        # RSI
        rsi = results['RSI']
        for low, high, points in criteria['rsi']['ranges']:
            if low <= rsi <= high:
                score += points
                break

        # MACD
        if results['MACD Signal'] == 'Buy':
            score += criteria['macd']['buy_score']

        # Bollinger Bands
        bb_pos = results['BB Position']
        for low, high, points in criteria['bollinger']['ranges']:
            if low <= bb_pos <= high:
                score += points
                break

        return score

    def get_key_points(self, results: Dict[str, Any], strengths: bool = True) -> List[str]:
        """Extract key insights from technical analysis results"""
        points = []

        if not results:
            return points

        # Moving Average Analysis
        price_sma50 = results.get('Price vs SMA50')
        price_sma200 = results.get('Price vs SMA200')

        if price_sma50 is not None and price_sma200 is not None:
            if strengths and price_sma50 > 0 and price_sma200 > 0:
                points.append(
                    f"Strong uptrend: Price above both 50-day ({price_sma50:.1%}) and 200-day ({price_sma200:.1%}) moving averages")
            elif not strengths and price_sma50 < -0.05 and price_sma200 < -0.05:
                points.append(
                    f"Downtrend: Price below key moving averages (50-day: {price_sma50:.1%}, 200-day: {price_sma200:.1%})")

        # RSI Analysis
        rsi = results.get('RSI')
        if rsi is not None:
            if strengths and 40 <= rsi <= 60:
                points.append(f"Balanced RSI at {rsi:.1f} indicating steady momentum")
            elif not strengths:
                if rsi > 70:
                    points.append(f"Overbought conditions with RSI at {rsi:.1f}")
                elif rsi < 30:
                    points.append(f"Oversold conditions with RSI at {rsi:.1f}")

        # MACD Analysis
        macd_signal = results.get('MACD Signal')
        if macd_signal:
            if strengths and macd_signal == 'Buy':
                points.append("Positive MACD crossover suggesting upward momentum")
            elif not strengths and macd_signal == 'Sell':
                points.append("Negative MACD crossover indicating potential downward movement")

        return sorted(points, key=len, reverse=True)


class RiskAnalysis(AnalysisModule):
    def analyze(self, data: pd.DataFrame, stock: yf.Ticker) -> Dict[str, Any]:
        if data.empty:
            return {}

        returns = data['Close'].pct_change().dropna()

        try:
            beta = self._calculate_beta(stock, data)
            sharpe = self._calculate_sharpe_ratio(returns)

            return {
                'Volatility': float(returns.std() * np.sqrt(252)),
                'Max Drawdown': float((data['Close'] / data['Close'].expanding().max() - 1).min()),
                'Beta': float(beta),
                'Sharpe Ratio': float(sharpe),
                'Value at Risk (95%)': float(np.percentile(returns, 5))
            }
        except Exception as e:
            print(f"Error in risk analysis: {e}")
            return {}

    def score(self, results: Dict[str, Any]) -> float:
        if not results:
            return 0

        score = 100

        risk_criteria = {
            'Volatility': {'ranges': [(0.4, -30), (0.25, -15)]},
            'Max Drawdown': {'ranges': [(-0.5, -30), (-0.3, -15)]},
            'Beta': {'ranges': [(1.5, -20)], 'bonus': [(0.8, 10)]},
            'Sharpe Ratio': {'bonus': [(1, 20), (0.5, 10)]}
        }

        return self._calculate_risk_score(results, risk_criteria)

    def _calculate_risk_score(self, results: Dict[str, Any], criteria: Dict[str, Dict[str, Any]]) -> float:
        score = 100

        for metric, rules in criteria.items():
            value = results.get(metric)
            if value is None:
                continue

            if 'ranges' in rules:
                for threshold, penalty in rules['ranges']:
                    if (metric == 'Beta' and value > threshold) or (value < threshold):
                        score += penalty
                        break

            if 'bonus' in rules:
                for threshold, bonus in rules['bonus']:
                    if value > threshold:
                        score += bonus
                        break

        return max(0, min(100, score))

    def get_key_points(self, results: Dict[str, Any], strengths: bool = True) -> List[str]:
        """Extract key insights from risk analysis results"""
        points = []

        if not results:
            return points

        # Volatility Analysis
        volatility = results.get('Volatility')
        if volatility is not None:
            if strengths and volatility < 0.20:
                points.append(f"Low volatility of {volatility:.1%} indicating price stability")
            elif not strengths and volatility > 0.40:
                points.append(f"High volatility of {volatility:.1%} suggesting increased risk")

        # Beta Analysis
        beta = results.get('Beta')
        if beta is not None:
            if strengths and 0.8 <= beta <= 1.2:
                points.append(f"Balanced market sensitivity with beta of {beta:.2f}")
            elif not strengths and (beta > 1.5 or beta < 0.5):
                points.append(
                    f"{'High' if beta > 1.5 else 'Low'} market sensitivity (beta: {beta:.2f}) suggesting {'increased volatility' if beta > 1.5 else 'limited market participation'}")

        # Sharpe Ratio Analysis
        sharpe = results.get('Sharpe Ratio')
        if sharpe is not None:
            if strengths and sharpe > 1:
                points.append(f"Strong risk-adjusted returns with Sharpe ratio of {sharpe:.2f}")
            elif not strengths and sharpe < 0.5:
                points.append(f"Poor risk-adjusted returns with Sharpe ratio of {sharpe:.2f}")

        # Maximum Drawdown Analysis
        drawdown = results.get('Max Drawdown')
        if drawdown is not None:
            if strengths and drawdown > -0.20:
                points.append(f"Moderate maximum drawdown of {drawdown:.1%}")
            elif not strengths and drawdown < -0.30:
                points.append(f"Significant drawdown risk of {drawdown:.1%}")

        return sorted(points, key=len, reverse=True)

    def _calculate_beta(self, stock: yf.Ticker, data: pd.DataFrame) -> float:
        try:
            spy = yf.download('^GSPC',
                              start=data.index[0].strftime('%Y-%m-%d'),
                              end=data.index[-1].strftime('%Y-%m-%d'),
                              progress=False)['Close']

            common_dates = data.index.intersection(spy.index)
            if len(common_dates) < 30:
                return 1.0

            stock_returns = data.loc[common_dates, 'Close'].pct_change().dropna()
            market_returns = spy.loc[common_dates].pct_change().dropna()

            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)

            if market_variance == 0:
                return 1.0

            beta = covariance / market_variance
            return beta if not np.isnan(beta) else 1.0

        except Exception as e:
            print(f"Error calculating beta: {e}")
            return 1.0

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()


class NewsSentimentAnalysis(AnalysisModule):
    def analyze(self, data: pd.DataFrame, stock: yf.Ticker) -> Dict[str, Any]:
        try:
            news = stock.news
            if not news:
                return {}

            sentiment_scores = self._calculate_sentiment_scores(news[:10])

            return {
                'Average Sentiment': np.mean(sentiment_scores),
                'Recent News Count': len(news),
                'Sentiment Trend': 'Positive' if np.mean(sentiment_scores) > 0 else 'Negative'
            }
        except Exception as e:
            print(f"Error in news sentiment analysis: {e}")
            return {}

    def score(self, results: Dict[str, Any]) -> float:
        if not results:
            return 0

        base_score = 50
        if results.get('Average Sentiment', 0) > 0:
            base_score += 25
        if results.get('Recent News Count', 0) > 5:
            base_score += 25

        return base_score

    def get_key_points(self, results: Dict[str, Any], strengths: bool = True) -> List[str]:
        points = []
        if not results:
            return points

        avg_sentiment = results.get('Average Sentiment', 0)
        news_count = results.get('Recent News Count', 0)

        if strengths and avg_sentiment > 0:
            points.append(f"Positive news sentiment with score {avg_sentiment:.2f}")
        elif not strengths and avg_sentiment < 0:
            points.append(f"Negative news sentiment with score {avg_sentiment:.2f}")

        if strengths and news_count > 5:
            points.append(f"High news coverage with {news_count} recent articles")
        elif not strengths and news_count < 3:
            points.append(f"Limited news coverage with only {news_count} recent articles")

        return points

    def _calculate_sentiment_scores(self, news: List[Dict[str, Any]]) -> List[float]:
        positive_words = ['up', 'rise', 'gain', 'positive', 'growth', 'profit']
        negative_words = ['down', 'fall', 'drop', 'negative', 'loss', 'debt']

        sentiment_scores = []
        for article in news:
            title = article['title'].lower()
            score = sum(word in title for word in positive_words) - sum(word in title for word in negative_words)
            sentiment_scores.append(score)

        return sentiment_scores


class StockAnalyzer:
    def __init__(self, ticker: str, session: Optional[requests.Session] = None):
        """Initialize StockAnalyzer with configurable analysis modules"""
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        if session:
            self.stock._session = session
        self.data = None

        # Initialize analysis modules
        self._analysis_modules = {
            'fundamental': FundamentalAnalysis(),
            'technical': TechnicalAnalysis(),
            'risk': RiskAnalysis(),
            'news': NewsSentimentAnalysis()
        }

        # Store analysis results
        self._analysis_results = {}
        self._analysis_scores = {}

    def register_analysis_module(self, name: str, module: AnalysisModule) -> None:
        """
        Register a new analysis module

        Parameters:
            name (str): Unique identifier for the module
            module (AnalysisModule): Instance of an analysis module
        """
        if not isinstance(module, AnalysisModule):
            raise ValueError("Module must inherit from AnalysisModule")
        self._analysis_modules[name] = module

    def fetch_data(self, period: str = "1y") -> pd.DataFrame:
        """Fetch historical data for the specified period"""
        self.data = self.stock.history(period=period)
        return self.data

    def run_analysis(self, modules: Optional[List[str]] = None) -> None:
        """
        Run analysis using specified modules or all registered modules

        Parameters:
            modules (List[str], optional): List of module names to run
        """
        if self.data is None:
            self.fetch_data()

        # Determine which modules to run
        modules_to_run = modules if modules else self._analysis_modules.keys()

        # Run each module's analysis
        for module_name in modules_to_run:
            if module_name not in self._analysis_modules:
                print(f"Warning: Module '{module_name}' not found")
                continue

            try:
                module = self._analysis_modules[module_name]
                results = module.analyze(self.data, self.stock)
                score = module.score(results)

                self._analysis_results[module_name] = results
                self._analysis_scores[module_name] = score

            except Exception as e:
                print(f"Error running {module_name} analysis: {e}")
                self._analysis_results[module_name] = {}
                self._analysis_scores[module_name] = 0

    def get_overall_score(self) -> float:
        """Calculate weighted overall score"""
        if not self._analysis_scores:
            return 0

        weights = {
            'fundamental': 0.4,
            'technical': 0.3,
            'risk': 0.2,
            'news': 0.1
        }

        total_weight = 0
        weighted_score = 0

        for module_name, score in self._analysis_scores.items():
            weight = weights.get(module_name, 0.1)  # Default weight for custom modules
            weighted_score += score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0

    def get_recommendation(self) -> Dict[str, Any]:
        """Generate comprehensive analysis recommendation"""
        if not self._analysis_results:
            self.run_analysis()

        total_score = self.get_overall_score()

        # Gather key points from all modules
        strengths = []
        concerns = []
        for module_name, module in self._analysis_modules.items():
            results = self._analysis_results.get(module_name, {})
            strengths.extend(module.get_key_points(results, True))
            concerns.extend(module.get_key_points(results, False))

        return {
            'Total Score': total_score,
            'Module Scores': self._analysis_scores.copy(),
            'Recommendation': self._get_recommendation_text(total_score),
            'Key Strengths': sorted(strengths, key=len, reverse=True)[:3],
            'Key Concerns': sorted(concerns, key=len, reverse=True)[:3]
        }

    def _get_recommendation_text(self, score: float) -> str:
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

    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self._analysis_results:
            self.run_analysis()

        recommendation = self.get_recommendation()

        report = []
        report.append(f"\nStock Analysis Report for {self.ticker}")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Overall Summary
        report.append("\n1. SUMMARY")
        report.append("-" * 50)
        report.append(f"Overall Score: {recommendation['Total Score']:.1f}/100")
        report.append(f"Recommendation: {recommendation['Recommendation']}")

        # Module-specific results
        for module_name, results in self._analysis_results.items():
            report.append(f"\n{(module_name.upper())} ANALYSIS")
            report.append("-" * 50)
            report.append(f"Score: {self._analysis_scores[module_name]:.1f}/100")

            if results:
                report.append("\nKey Metrics:")
                self._append_metrics_to_report(report, results)

        # Key Points
        report.append("\nKEY POINTS")
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

    def _append_metrics_to_report(self, report: List[str], metrics: Dict[str, Any]) -> None:
        """Helper method to format metrics for the report"""
        for metric, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, float):
                if metric.endswith(('Ratio', 'Score')):
                    report.append(f"- {metric}: {value:.2f}")
                elif metric.endswith(('Percentage', 'Yield')):
                    report.append(f"- {metric}: {value:.2%}")
                else:
                    report.append(f"- {metric}: {value:.2f}")
            else:
                report.append(f"- {metric}: {value}")

    def plot_technical_analysis(self, show_plot: bool = False) -> None:
        """Create technical analysis visualization"""
        if self.data is None:
            self.fetch_data()

        technical_module = self._analysis_modules.get('technical')
        if not technical_module:
            print("Technical analysis module not found")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12),
                                            gridspec_kw={'height_ratios': [2, 1, 1]})

        # Price and Moving Averages
        ax1.plot(self.data.index, self.data['Close'], label='Price')
        ax1.plot(self.data.index, self.data['Close'].rolling(window=50).mean(),
                 label='50-day MA')
        ax1.plot(self.data.index, self.data['Close'].rolling(window=200).mean(),
                 label='200-day MA')
        ax1.set_title(f'{self.ticker} Technical Analysis')
        ax1.set_ylabel('Price')
        ax1.legend()

        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        ax2.plot(self.data.index, rsi)
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
            plt.close(fig)