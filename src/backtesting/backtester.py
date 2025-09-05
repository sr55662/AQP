# src/backtesting/advanced_backtester.py
# Comprehensive Backtesting & Walk Forward Analysis Engine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    initial_capital: float = 100000
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005   # 0.05% slippage
    margin_rate: float = 0.03  # 3% annual margin rate
    max_leverage: float = 1.0  # No leverage by default
    position_size_method: str = 'equal_weight'  # 'equal_weight', 'volatility_adjusted', 'custom'
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    benchmark: str = 'SPY'
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    trade_type: str  # 'long', 'short'
    pnl: float
    commission: float
    slippage: float
    holding_period: int  # days
    
@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    # Core metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Advanced metrics
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    skewness: float
    kurtosis: float
    winning_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration: float
    largest_win: float
    largest_loss: float
    
    # Portfolio metrics
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    positions: pd.DataFrame
    trades: List[Trade]
    
    # Benchmarking
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float
    
    # Time series data
    daily_returns: pd.Series
    monthly_returns: pd.Series
    rolling_sharpe: pd.Series
    rolling_volatility: pd.Series

class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for given data"""
        pass
    
    @abstractmethod
    def calculate_positions(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate position sizes based on signals"""
        pass

class AdvancedBacktester:
    """Professional-grade backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades = []
        self.equity_curve = pd.Series(dtype=float)
        self.positions = pd.DataFrame()
        self.benchmark_data = None
        
    def run_backtest(self, 
                    strategy: Strategy, 
                    data: Dict[str, pd.DataFrame],
                    start_date: str = None,
                    end_date: str = None) -> BacktestResults:
        """Execute comprehensive backtest"""
        
        logger.info("Starting backtest execution...")
        
        # Prepare data
        market_data = self._prepare_data(data, start_date, end_date)
        
        # Generate signals
        signals = strategy.generate_signals(market_data)
        
        # Calculate positions
        positions = strategy.calculate_positions(signals, market_data)
        
        # Execute simulation
        portfolio = self._simulate_trading(positions, market_data)
        
        # Calculate metrics
        results = self._calculate_metrics(portfolio, market_data)
        
        logger.info(f"Backtest completed. Sharpe ratio: {results.sharpe_ratio:.3f}")
        
        return results
    
    def walk_forward_analysis(self,
                             strategy: Strategy,
                             data: Dict[str, pd.DataFrame],
                             window_size: int = 252,  # 1 year
                             step_size: int = 63,     # quarterly
                             min_periods: int = 126) -> Dict[str, Any]:
        """Perform walk-forward analysis"""
        
        logger.info("Starting walk-forward analysis...")
        
        # Prepare full dataset
        full_data = self._prepare_data(data)
        
        # Calculate walk-forward windows
        windows = self._generate_walk_forward_windows(
            full_data.index, window_size, step_size, min_periods
        )
        
        results = []
        equity_curves = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}: {test_start} to {test_end}")
            
            # Training data
            train_data = {
                symbol: df.loc[train_start:train_end] 
                for symbol, df in data.items()
            }
            
            # Test data
            test_data = {
                symbol: df.loc[test_start:test_end] 
                for symbol, df in data.items()
            }
            
            # Run backtest on test period
            test_result = self.run_backtest(strategy, test_data)
            
            results.append({
                'window': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'sharpe_ratio': test_result.sharpe_ratio,
                'total_return': test_result.total_return,
                'max_drawdown': test_result.max_drawdown,
                'volatility': test_result.volatility,
                'winning_rate': test_result.winning_rate
            })
            
            equity_curves.append(test_result.equity_curve)
        
        # Combine results
        wf_results = pd.DataFrame(results)
        combined_equity = pd.concat(equity_curves)
        
        # Calculate walk-forward metrics
        wf_metrics = self._calculate_walk_forward_metrics(wf_results, combined_equity)
        
        return {
            'window_results': wf_results,
            'combined_equity_curve': combined_equity,
            'metrics': wf_metrics,
            'stability_score': self._calculate_stability_score(wf_results)
        }
    
    def monte_carlo_analysis(self,
                           strategy: Strategy,
                           data: Dict[str, pd.DataFrame],
                           n_simulations: int = 1000,
                           resample_method: str = 'bootstrap') -> Dict[str, Any]:
        """Perform Monte Carlo simulation analysis"""
        
        logger.info(f"Starting Monte Carlo analysis with {n_simulations} simulations...")
        
        results = []
        
        for i in range(n_simulations):
            if i % 100 == 0:
                logger.info(f"Simulation {i}/{n_simulations}")
            
            # Resample data
            if resample_method == 'bootstrap':
                resampled_data = self._bootstrap_resample(data)
            elif resample_method == 'block_bootstrap':
                resampled_data = self._block_bootstrap_resample(data, block_size=20)
            else:
                raise ValueError(f"Unknown resample method: {resample_method}")
            
            # Run backtest
            result = self.run_backtest(strategy, resampled_data)
            
            results.append({
                'simulation': i,
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'volatility': result.volatility
            })
        
        mc_df = pd.DataFrame(results)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric in ['sharpe_ratio', 'total_return', 'max_drawdown', 'volatility']:
            confidence_intervals[metric] = {
                '5th_percentile': mc_df[metric].quantile(0.05),
                '95th_percentile': mc_df[metric].quantile(0.95),
                'mean': mc_df[metric].mean(),
                'std': mc_df[metric].std()
            }
        
        return {
            'simulation_results': mc_df,
            'confidence_intervals': confidence_intervals,
            'probability_positive_sharpe': (mc_df['sharpe_ratio'] > 0).mean(),
            'probability_sharpe_gt_1': (mc_df['sharpe_ratio'] > 1.0).mean(),
            'probability_sharpe_gt_2': (mc_df['sharpe_ratio'] > 2.0).mean()
        }
    
    def _prepare_data(self, 
                     data: Dict[str, pd.DataFrame], 
                     start_date: str = None, 
                     end_date: str = None) -> pd.DataFrame:
        """Prepare and align market data"""
        
        # Combine all data into single DataFrame
        all_data = {}
        for symbol, df in data.items():
            for col in df.columns:
                all_data[f"{symbol}_{col}"] = df[col]
        
        market_data = pd.DataFrame(all_data)
        
        # Filter by date range
        if start_date:
            market_data = market_data[market_data.index >= pd.to_datetime(start_date)]
        if end_date:
            market_data = market_data[market_data.index <= pd.to_datetime(end_date)]
        
        # Forward fill missing data
        market_data = market_data.fillna(method='ffill').dropna()
        
        return market_data
    
    def _simulate_trading(self, 
                         positions: pd.DataFrame, 
                         market_data: pd.DataFrame) -> pd.DataFrame:
        """Simulate actual trading with realistic costs"""
        
        portfolio = pd.DataFrame(index=market_data.index)
        portfolio['cash'] = self.config.initial_capital
        portfolio['portfolio_value'] = self.config.initial_capital
        
        # Track positions and trades
        current_positions = {}
        
        for date in market_data.index:
            date_positions = positions.loc[date] if date in positions.index else pd.Series()
            
            # Calculate position changes
            position_changes = {}
            for symbol in date_positions.index:
                current_qty = current_positions.get(symbol, 0)
                target_qty = date_positions[symbol]
                position_changes[symbol] = target_qty - current_qty
            
            # Execute trades
            total_trade_cost = 0
            for symbol, qty_change in position_changes.items():
                if abs(qty_change) > 0.001:  # Only trade if significant change
                    
                    # Get market price
                    price_col = f"{symbol}_close"
                    if price_col in market_data.columns:
                        price = market_data.loc[date, price_col]
                        
                        # Calculate costs
                        trade_value = abs(qty_change * price)
                        commission = trade_value * self.config.commission
                        slippage = trade_value * self.config.slippage
                        
                        total_trade_cost += commission + slippage
                        
                        # Record trade
                        if qty_change > 0:
                            trade_type = 'long'
                        else:
                            trade_type = 'short'
                        
                        # Update position
                        current_positions[symbol] = current_positions.get(symbol, 0) + qty_change
            
            # Update cash position
            portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[-1] - total_trade_cost
            
            # Calculate portfolio value
            portfolio_value = portfolio.loc[date, 'cash']
            for symbol, qty in current_positions.items():
                price_col = f"{symbol}_close"
                if price_col in market_data.columns:
                    price = market_data.loc[date, price_col]
                    portfolio_value += qty * price
            
            portfolio.loc[date, 'portfolio_value'] = portfolio_value
        
        return portfolio
    
    def _calculate_metrics(self, 
                          portfolio: pd.DataFrame, 
                          market_data: pd.DataFrame) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        
        # Basic return calculations
        equity_curve = portfolio['portfolio_value']
        daily_returns = equity_curve.pct_change().dropna()
        
        # Core metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Annualized calculations
        trading_days = len(daily_returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown calculation
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_95 = daily_returns.quantile(0.05)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        
        # Higher moments
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurtosis()
        
        # Monthly returns for analysis
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Rolling metrics
        rolling_window = 63  # ~3 months
        rolling_sharpe = daily_returns.rolling(rolling_window).apply(
            lambda x: (x.mean() * 252 - self.config.risk_free_rate) / (x.std() * np.sqrt(252))
        )
        rolling_volatility = daily_returns.rolling(rolling_window).std() * np.sqrt(252)
        
        # Benchmark comparison (if available)
        benchmark_return = 0.0
        alpha = 0.0
        beta = 0.0
        information_ratio = 0.0
        tracking_error = 0.0
        
        if self.config.benchmark and f"{self.config.benchmark}_close" in market_data.columns:
            benchmark_prices = market_data[f"{self.config.benchmark}_close"]
            benchmark_returns = benchmark_prices.pct_change().dropna()
            benchmark_return = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1
            
            # Align returns
            aligned_portfolio = daily_returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_portfolio.index).dropna()
            
            if len(aligned_portfolio) > 0 and len(aligned_benchmark) > 0:
                # Calculate beta and alpha
                covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                portfolio_annual = aligned_portfolio.mean() * 252
                benchmark_annual = aligned_benchmark.mean() * 252
                alpha = portfolio_annual - (self.config.risk_free_rate + beta * (benchmark_annual - self.config.risk_free_rate))
                
                # Information ratio
                excess_returns = aligned_portfolio - aligned_benchmark
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            winning_rate=0.0,  # Would need trade-by-trade analysis
            profit_factor=0.0,  # Would need trade-by-trade analysis
            total_trades=len(self.trades),
            winning_trades=0,
            losing_trades=0,
            avg_trade_duration=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            equity_curve=equity_curve,
            drawdown_curve=drawdown,
            positions=pd.DataFrame(),
            trades=self.trades,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            rolling_sharpe=rolling_sharpe,
            rolling_volatility=rolling_volatility
        )
    
    def _generate_walk_forward_windows(self, 
                                     dates: pd.DatetimeIndex, 
                                     window_size: int, 
                                     step_size: int, 
                                     min_periods: int) -> List[Tuple]:
        """Generate walk-forward analysis windows"""
        
        windows = []
        start_idx = 0
        
        while start_idx + window_size + min_periods < len(dates):
            train_start = dates[start_idx]
            train_end = dates[start_idx + window_size - 1]
            test_start = dates[start_idx + window_size]
            test_end = dates[min(start_idx + window_size + step_size - 1, len(dates) - 1)]
            
            windows.append((train_start, train_end, test_start, test_end))
            start_idx += step_size
        
        return windows
    
    def _calculate_walk_forward_metrics(self, 
                                      results: pd.DataFrame, 
                                      equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate walk-forward specific metrics"""
        
        return {
            'avg_sharpe': results['sharpe_ratio'].mean(),
            'sharpe_stability': 1 - (results['sharpe_ratio'].std() / abs(results['sharpe_ratio'].mean())) if results['sharpe_ratio'].mean() != 0 else 0,
            'avg_return': results['total_return'].mean(),
            'avg_drawdown': results['max_drawdown'].mean(),
            'win_rate_periods': (results['total_return'] > 0).mean(),
            'consistency_score': (results['sharpe_ratio'] > 0).mean()
        }
    
    def _calculate_stability_score(self, results: pd.DataFrame) -> float:
        """Calculate overall strategy stability score"""
        
        # Factors contributing to stability
        sharpe_consistency = (results['sharpe_ratio'] > 0).mean()
        return_consistency = (results['total_return'] > 0).mean()
        drawdown_control = 1 - (results['max_drawdown'].mean() / -0.2)  # Normalize against -20% drawdown
        
        # Weighted average
        stability_score = (0.4 * sharpe_consistency + 
                          0.3 * return_consistency + 
                          0.3 * max(0, drawdown_control))
        
        return stability_score
    
    def _bootstrap_resample(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Bootstrap resample the data"""
        
        # Get first dataset to determine length
        first_key = list(data.keys())[0]
        n_samples = len(data[first_key])
        
        # Generate random indices with replacement
        random_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Resample all datasets with same indices
        resampled_data = {}
        for symbol, df in data.items():
            resampled_data[symbol] = df.iloc[random_indices].reset_index(drop=True)
        
        return resampled_data
    
    def _block_bootstrap_resample(self, 
                                 data: Dict[str, pd.DataFrame], 
                                 block_size: int = 20) -> Dict[str, pd.DataFrame]:
        """Block bootstrap resample to preserve time series structure"""
        
        first_key = list(data.keys())[0]
        n_samples = len(data[first_key])
        
        # Generate block starts
        n_blocks = n_samples // block_size
        block_starts = np.random.choice(n_samples - block_size + 1, size=n_blocks, replace=True)
        
        # Create resampled indices
        resampled_indices = []
        for start in block_starts:
            resampled_indices.extend(range(start, start + block_size))
        
        # Trim to original length
        resampled_indices = resampled_indices[:n_samples]
        
        # Resample all datasets
        resampled_data = {}
        for symbol, df in data.items():
            resampled_data[symbol] = df.iloc[resampled_indices].reset_index(drop=True)
        
        return resampled_data

# Example strategy implementation
class MomentumStrategy(Strategy):
    """Simple momentum strategy for testing"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals"""
        signals = pd.DataFrame(index=data.index)
        
        # Extract symbols
        symbols = list(set([col.split('_')[0] for col in data.columns if '_close' in col]))
        
        for symbol in symbols:
            price_col = f"{symbol}_close"
            if price_col in data.columns:
                prices = data[price_col]
                
                # Calculate momentum
                momentum = prices / prices.shift(self.lookback_period) - 1
                
                # Generate signals
                signals[f"{symbol}_signal"] = np.where(momentum > 0.02, 1, 
                                                      np.where(momentum < -0.02, -1, 0))
        
        return signals.fillna(0)
    
    def calculate_positions(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate equal-weight positions"""
        positions = pd.DataFrame(index=signals.index)
        
        # Extract symbols
        symbols = list(set([col.split('_')[0] for col in signals.columns if '_signal' in col]))
        
        for symbol in symbols:
            signal_col = f"{symbol}_signal"
            if signal_col in signals.columns:
                # Equal weight allocation
                positions[symbol] = signals[signal_col] / len(symbols)
        
        return positions.fillna(0)

# Example usage
if __name__ == "__main__":
    # Configuration
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        benchmark='SPY'
    )
    
    # Create backtester
    backtester = AdvancedBacktester(config)
    
    # Example data (replace with real data)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    example_data = {
        'AAPL': pd.DataFrame({
            'close': 100 * (1 + np.random.normal(0, 0.01, len(dates))).cumprod(),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates),
        'MSFT': pd.DataFrame({
            'close': 100 * (1 + np.random.normal(0, 0.01, len(dates))).cumprod(),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    }
    
    # Create strategy
    strategy = MomentumStrategy(lookback_period=20)
    
    # Run backtest
    results = backtester.run_backtest(strategy, example_data)
    
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Annualized Return: {results.annualized_return:.2%}")
    print(f"Volatility: {results.volatility:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    
    # Run walk-forward analysis
    wf_results = backtester.walk_forward_analysis(strategy, example_data)
    print(f"\nWalk Forward Avg Sharpe: {wf_results['metrics']['avg_sharpe']:.3f}")
    print(f"Stability Score: {wf_results['stability_score']:.3f}")
    
    # Run Monte Carlo analysis
    mc_results = backtester.monte_carlo_analysis(strategy, example_data, n_simulations=100)
    print(f"\nMonte Carlo Sharpe CI: [{mc_results['confidence_intervals']['sharpe_ratio']['5th_percentile']:.3f}, {mc_results['confidence_intervals']['sharpe_ratio']['95th_percentile']:.3f}]")
    print(f"Probability Sharpe > 2.0: {mc_results['probability_sharpe_gt_2']:.2%}")