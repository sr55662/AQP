"""
AQP Backtester & Walk Forward Tester
High-performance strategy testing with AWS scaling and comprehensive metrics
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from scipy import stats
import boto3
import pickle
import uuid
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None

@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: OrderSide
    pnl: float
    commission: float = 0.0
    trade_id: Optional[str] = None

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position]
    total_value: float
    total_pnl: float
    timestamp: datetime

@dataclass
class PerformanceMetrics:
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    var_95: float
    var_99: float
    beta: float
    alpha: float
    information_ratio: float
    trades_count: int
    avg_trade_duration: float

class Strategy:
    """Base strategy class that all strategies must inherit from"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on market data
        Returns: Series with 1 (buy), -1 (sell), 0 (hold)
        """
        raise NotImplementedError("Strategy must implement generate_signals method")
    
    def generate_orders(self, signals: pd.Series, data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        """Convert signals to actual orders"""
        orders = []
        for timestamp, signal in signals.items():
            if signal != 0:
                # Get current price from data
                current_price = data.loc[timestamp, 'close']
                symbol = data.iloc[0].name if hasattr(data.iloc[0], 'name') else 'UNKNOWN'
                
                # Simple position sizing (10% of portfolio)
                position_size = portfolio.total_value * 0.1 / current_price
                
                if signal > 0:  # Buy signal
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=position_size,
                        order_type=OrderType.MARKET,
                        timestamp=timestamp
                    ))
                elif signal < 0:  # Sell signal
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=position_size,
                        order_type=OrderType.MARKET,
                        timestamp=timestamp
                    ))
        return orders

class ExecutionEngine:
    """Handles order execution with realistic slippage and commission models"""
    
    def __init__(self, commission_rate: float = 0.001, slippage_bps: float = 2.0):
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000  # Convert basis points to decimal
        
    def execute_order(self, order: Order, market_data: pd.Series, portfolio: Portfolio) -> Tuple[Trade, Portfolio]:
        """Execute an order and update portfolio"""
        
        # Calculate execution price with slippage
        market_price = market_data['close']
        if order.side == OrderSide.BUY:
            execution_price = market_price * (1 + self.slippage_bps)
        else:
            execution_price = market_price * (1 - self.slippage_bps)
            
        # Calculate commission
        trade_value = order.quantity * execution_price
        commission = trade_value * self.commission_rate
        
        # Create trade
        trade = Trade(
            symbol=order.symbol,
            entry_time=order.timestamp,
            exit_time=order.timestamp,  # Will be updated for position closes
            entry_price=execution_price,
            exit_price=execution_price,
            quantity=order.quantity,
            side=order.side,
            pnl=0,  # Will be calculated when position is closed
            commission=commission,
            trade_id=str(uuid.uuid4())
        )
        
        # Update portfolio
        new_portfolio = self._update_portfolio(portfolio, order, execution_price, commission)
        
        return trade, new_portfolio
    
    def _update_portfolio(self, portfolio: Portfolio, order: Order, execution_price: float, commission: float) -> Portfolio:
        """Update portfolio after order execution"""
        
        new_positions = portfolio.positions.copy()
        new_cash = portfolio.cash
        
        trade_value = order.quantity * execution_price
        
        if order.side == OrderSide.BUY:
            # Buying shares
            new_cash -= (trade_value + commission)
            
            if order.symbol in new_positions:
                # Add to existing position
                existing_pos = new_positions[order.symbol]
                total_quantity = existing_pos.quantity + order.quantity
                total_cost = (existing_pos.quantity * existing_pos.avg_price) + trade_value
                avg_price = total_cost / total_quantity
                
                new_positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=total_quantity,
                    avg_price=avg_price,
                    market_value=total_quantity * execution_price,
                    unrealized_pnl=(execution_price - avg_price) * total_quantity,
                    realized_pnl=existing_pos.realized_pnl
                )
            else:
                # New position
                new_positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=execution_price,
                    market_value=trade_value,
                    unrealized_pnl=0,
                    realized_pnl=0
                )
                
        else:  # SELL
            # Selling shares
            new_cash += (trade_value - commission)
            
            if order.symbol in new_positions:
                existing_pos = new_positions[order.symbol]
                remaining_quantity = existing_pos.quantity - order.quantity
                
                if remaining_quantity > 0:
                    # Partial close
                    realized_pnl = (execution_price - existing_pos.avg_price) * order.quantity
                    new_positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=remaining_quantity,
                        avg_price=existing_pos.avg_price,
                        market_value=remaining_quantity * execution_price,
                        unrealized_pnl=(execution_price - existing_pos.avg_price) * remaining_quantity,
                        realized_pnl=existing_pos.realized_pnl + realized_pnl
                    )
                else:
                    # Full close
                    realized_pnl = (execution_price - existing_pos.avg_price) * existing_pos.quantity
                    del new_positions[order.symbol]
        
        # Calculate total portfolio value
        total_positions_value = sum(pos.market_value for pos in new_positions.values())
        total_value = new_cash + total_positions_value
        total_pnl = sum(pos.realized_pnl + pos.unrealized_pnl for pos in new_positions.values())
        
        return Portfolio(
            cash=new_cash,
            positions=new_positions,
            total_value=total_value,
            total_pnl=total_pnl,
            timestamp=order.timestamp
        )

class Backtester:
    """High-performance backtester with comprehensive analytics"""
    
    def __init__(self, initial_capital: float = 100000, commission_rate: float = 0.001, slippage_bps: float = 2.0):
        self.initial_capital = initial_capital
        self.execution_engine = ExecutionEngine(commission_rate, slippage_bps)
        self.logger = logging.getLogger('AQP-Backtester')
        
    def backtest_strategy(
        self, 
        strategy: Strategy, 
        data: Dict[str, pd.DataFrame], 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a single strategy
        """
        self.logger.info(f"Starting backtest for strategy: {strategy.name}")
        
        # Initialize portfolio
        portfolio = Portfolio(
            cash=self.initial_capital,
            positions={},
            total_value=self.initial_capital,
            total_pnl=0,
            timestamp=start_date or min(df.index[0] for df in data.values())
        )
        
        trades = []
        portfolio_history = []
        
        # Get all timestamps across all symbols
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index)
        all_timestamps = sorted(all_timestamps)
        
        if start_date:
            all_timestamps = [ts for ts in all_timestamps if ts >= start_date]
        if end_date:
            all_timestamps = [ts for ts in all_timestamps if ts <= end_date]
        
        # Run simulation
        for timestamp in all_timestamps:
            # Generate signals for each symbol
            all_orders = []
            
            for symbol, symbol_data in data.items():
                if timestamp in symbol_data.index:
                    # Get data up to current timestamp for signal generation
                    historical_data = symbol_data.loc[:timestamp]
                    
                    try:
                        signals = strategy.generate_signals(historical_data)
                        if len(signals) > 0 and not pd.isna(signals.iloc[-1]):
                            current_signal = signals.iloc[-1]
                            if current_signal != 0:
                                orders = strategy.generate_orders(
                                    pd.Series([current_signal], index=[timestamp]), 
                                    symbol_data, 
                                    portfolio
                                )
                                all_orders.extend(orders)
                    except Exception as e:
                        self.logger.warning(f"Error generating signals for {symbol} at {timestamp}: {e}")
            
            # Execute orders
            for order in all_orders:
                if order.symbol in data and timestamp in data[order.symbol].index:
                    market_data = data[order.symbol].loc[timestamp]
                    try:
                        trade, portfolio = self.execution_engine.execute_order(order, market_data, portfolio)
                        trades.append(trade)
                    except Exception as e:
                        self.logger.warning(f"Error executing order for {order.symbol} at {timestamp}: {e}")
            
            # Update portfolio timestamp and mark-to-market
            portfolio.timestamp = timestamp
            portfolio = self._mark_to_market(portfolio, data, timestamp)
            portfolio_history.append(portfolio)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio_history, trades)
        
        result = {
            'strategy_name': strategy.name,
            'parameters': strategy.parameters,
            'metrics': metrics,
            'trades': trades,
            'portfolio_history': portfolio_history,
            'final_portfolio': portfolio
        }
        
        self.logger.info(f"Backtest completed. Sharpe: {metrics.sharpe_ratio:.2f}, Return: {metrics.total_return:.2%}")
        return result
    
    def _mark_to_market(self, portfolio: Portfolio, data: Dict[str, pd.DataFrame], timestamp: datetime) -> Portfolio:
        """Update portfolio with current market values"""
        updated_positions = {}
        
        for symbol, position in portfolio.positions.items():
            if symbol in data and timestamp in data[symbol].index:
                current_price = data[symbol].loc[timestamp, 'close']
                market_value = position.quantity * current_price
                unrealized_pnl = (current_price - position.avg_price) * position.quantity
                
                updated_positions[symbol] = Position(
                    symbol=symbol,
                    quantity=position.quantity,
                    avg_price=position.avg_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=position.realized_pnl
                )
            else:
                updated_positions[symbol] = position
        
        total_positions_value = sum(pos.market_value for pos in updated_positions.values())
        total_value = portfolio.cash + total_positions_value
        total_pnl = sum(pos.realized_pnl + pos.unrealized_pnl for pos in updated_positions.values())
        
        return Portfolio(
            cash=portfolio.cash,
            positions=updated_positions,
            total_value=total_value,
            total_pnl=total_pnl,
            timestamp=timestamp
        )
    
    def _calculate_performance_metrics(self, portfolio_history: List[Portfolio], trades: List[Trade]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if not portfolio_history:
            return PerformanceMetrics(**{field: 0 for field in PerformanceMetrics.__dataclass_fields__})
        
        # Convert to DataFrame for easier analysis
        values = [p.total_value for p in portfolio_history]
        timestamps = [p.timestamp for p in portfolio_history]
        
        df = pd.DataFrame({'value': values, 'timestamp': timestamps})
        df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        df['returns'] = df['value'].pct_change()
        daily_returns = df['returns'].dropna()
        
        # Basic metrics
        total_return = (df['value'].iloc[-1] / df['value'].iloc[0]) - 1
        
        # Annualized return (assuming daily data)
        days = (df.index[-1] - df.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = daily_returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trading metrics
        profitable_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(profitable_trades) / len(trades) if trades else 0
        
        gross_profit = sum(t.pnl for t in profitable_trades)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR calculations
        var_95 = np.percentile(daily_returns, 5)
        var_99 = np.percentile(daily_returns, 1)
        
        # Beta and Alpha (vs SPY - would need benchmark data)
        beta = 1.0  # Placeholder
        alpha = annual_return - (risk_free_rate * 252 + beta * 0.1)  # Assuming 10% market return
        
        # Information ratio
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Trade duration
        trade_durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades if t.exit_time and t.entry_time]
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            var_99=var_99,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            trades_count=len(trades),
            avg_trade_duration=avg_trade_duration
        )

class WalkForwardOptimizer:
    """Walk Forward Analysis for robust strategy optimization"""
    
    def __init__(self, backtester: Backtester):
        self.backtester = backtester
        self.logger = logging.getLogger('AQP-WalkForward')
        
    def optimize_strategy(
        self, 
        strategy_class: type, 
        parameter_ranges: Dict[str, List], 
        data: Dict[str, pd.DataFrame],
        train_window: int = 252,  # Days
        test_window: int = 63,   # Days
        step_size: int = 21      # Days
    ) -> Dict[str, Any]:
        """
        Perform walk-forward optimization
        """
        self.logger.info(f"Starting walk-forward optimization for {strategy_class.__name__}")
        
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations(parameter_ranges)
        self.logger.info(f"Testing {len(parameter_combinations)} parameter combinations")
        
        # Get date range
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)
        
        walk_forward_results = []
        
        # Perform walk-forward analysis
        start_idx = 0
        while start_idx + train_window + test_window <= len(all_dates):
            train_start = all_dates[start_idx]
            train_end = all_dates[start_idx + train_window - 1]
            test_start = all_dates[start_idx + train_window]
            test_end = all_dates[start_idx + train_window + test_window - 1]
            
            self.logger.info(f"Training: {train_start.date()} to {train_end.date()}, Testing: {test_start.date()} to {test_end.date()}")
            
            # Optimize on training data
            best_params = self._optimize_parameters(
                strategy_class, 
                parameter_combinations, 
                data, 
                train_start, 
                train_end
            )
            
            # Test on out-of-sample data
            strategy = strategy_class(**best_params)
            test_result = self.backtester.backtest_strategy(data, strategy, test_start, test_end)
            
            walk_forward_results.append({
                'train_period': (train_start, train_end),
                'test_period': (test_start, test_end),
                'best_parameters': best_params,
                'test_performance': test_result['metrics']
            })
            
            start_idx += step_size
        
        # Aggregate results
        aggregated_metrics = self._aggregate_walk_forward_results(walk_forward_results)
        
        return {
            'strategy_class': strategy_class.__name__,
            'parameter_ranges': parameter_ranges,
            'walk_forward_results': walk_forward_results,
            'aggregated_metrics': aggregated_metrics
        }
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations"""
        import itertools
        
        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
            
        return combinations
    
    def _optimize_parameters(
        self, 
        strategy_class: type, 
        parameter_combinations: List[Dict], 
        data: Dict[str, pd.DataFrame], 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict:
        """Find best parameters for training period"""
        
        best_sharpe = -np.inf
        best_params = {}
        
        # Use multiprocessing for parameter optimization
        with ProcessPoolExecutor() as executor:
            futures = []
            
            for params in parameter_combinations:
                future = executor.submit(
                    self._test_parameter_combination,
                    strategy_class,
                    params,
                    data,
                    start_date,
                    end_date
                )
                futures.append((future, params))
            
            for future, params in futures:
                try:
                    result = future.result()
                    if result and result['metrics'].sharpe_ratio > best_sharpe:
                        best_sharpe = result['metrics'].sharpe_ratio
                        best_params = params
                except Exception as e:
                    self.logger.warning(f"Error testing parameters {params}: {e}")
        
        return best_params
    
    def _test_parameter_combination(
        self, 
        strategy_class: type, 
        params: Dict, 
        data: Dict[str, pd.DataFrame], 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict:
        """Test a single parameter combination"""
        try:
            strategy = strategy_class(**params)
            result = self.backtester.backtest_strategy(strategy, data, start_date, end_date)
            return result
        except Exception as e:
            return None
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> PerformanceMetrics:
        """Aggregate walk-forward test results"""
        
        if not results:
            return PerformanceMetrics(**{field: 0 for field in PerformanceMetrics.__dataclass_fields__})
        
        # Extract test performance metrics
        test_metrics = [r['test_performance'] for r in results]
        
        # Calculate aggregate metrics
        total_return = np.mean([m.total_return for m in test_metrics])
        annual_return = np.mean([m.annual_return for m in test_metrics])
        sharpe_ratio = np.mean([m.sharpe_ratio for m in test_metrics])
        sortino_ratio = np.mean([m.sortino_ratio for m in test_metrics])
        max_drawdown = np.mean([m.max_drawdown for m in test_metrics])
        win_rate = np.mean([m.win_rate for m in test_metrics])
        profit_factor = np.mean([m.profit_factor for m in test_metrics])
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=np.mean([m.calmar_ratio for m in test_metrics]),
            var_95=np.mean([m.var_95 for m in test_metrics]),
            var_99=np.mean([m.var_99 for m in test_metrics]),
            beta=np.mean([m.beta for m in test_metrics]),
            alpha=np.mean([m.alpha for m in test_metrics]),
            information_ratio=np.mean([m.information_ratio for m in test_metrics]),
            trades_count=sum([m.trades_count for m in test_metrics]),
            avg_trade_duration=np.mean([m.avg_trade_duration for m in test_metrics])
        )

class AWSScaledBacktester:
    """AWS-scaled backtesting for massive strategy optimization"""
    
    def __init__(self, s3_bucket: str, lambda_function: str):
        self.s3 = boto3.client('s3')
        self.lambda_client = boto3.client('lambda')
        self.s3_bucket = s3_bucket
        self.lambda_function = lambda_function
        
    async def run_massive_optimization(
        self, 
        strategy_classes: List[type], 
        parameter_ranges: Dict[str, Dict[str, List]], 
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Run optimization across multiple strategies using AWS Lambda"""
        
        # Upload data to S3
        data_key = f"backtest-data/{uuid.uuid4()}.pkl"
        data_bytes = pickle.dumps(data)
        self.s3.put_object(Bucket=self.s3_bucket, Key=data_key, Body=data_bytes)
        
        # Create optimization tasks
        tasks = []
        for strategy_class in strategy_classes:
            strategy_params = parameter_ranges.get(strategy_class.__name__, {})
            
            task = {
                'strategy_class_name': strategy_class.__name__,
                'strategy_module': strategy_class.__module__,
                'parameter_ranges': strategy_params,
                'data_s3_key': data_key,
                's3_bucket': self.s3_bucket
            }
            tasks.append(task)
        
        # Execute tasks in parallel using Lambda
        lambda_futures = []
        for task in tasks:
            future = self._invoke_lambda_async(task)
            lambda_futures.append(future)
        
        # Collect results
        optimization_results = await asyncio.gather(*lambda_futures, return_exceptions=True)
        
        # Process and rank results
        all_results = []
        for result in optimization_results:
            if isinstance(result, dict) and 'error' not in result:
                all_results.extend(result.get('strategy_results', []))
        
        # Rank by Sharpe ratio
        all_results.sort(key=lambda x: x['metrics'].sharpe_ratio, reverse=True)
        
        return {
            'top_strategies': all_results[:10],
            'all_results': all_results,
            'total_strategies_tested': len(all_results)
        }
    
    async def _invoke_lambda_async(self, task: Dict) -> Dict:
        """Invoke Lambda function asynchronously"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: self.lambda_client.invoke(
                    FunctionName=self.lambda_function,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(task)
                )
            )
        return json.loads(response['Payload'].read())


# Example strategy implementation
class MovingAverageCrossover(Strategy):
    """Simple moving average crossover strategy"""
    
    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__("MovingAverageCrossover", {"short_window": short_window, "long_window": long_window})
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on moving average crossover"""
        
        if len(data) < self.long_window:
            return pd.Series(dtype=float)
        
        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1   # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        
        # Only signal on crossovers
        signal_changes = signals.diff()
        signals = signals * (signal_changes != 0)
        
        return signals


# Example usage
if __name__ == "__main__":
    # Create sample data (in real usage, this would come from DataAggregator)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = {
        'AAPL': pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02),
            'high': 102 + np.cumsum(np.random.randn(len(dates)) * 0.02),
            'low': 98 + np.cumsum(np.random.randn(len(dates)) * 0.02),
            'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    }
    
    # Initialize backtester
    backtester = Backtester(initial_capital=100000)
    
    # Test strategy
    strategy = MovingAverageCrossover(short_window=10, long_window=30)
    result = backtester.backtest_strategy(strategy, sample_data)
    
    print(f"Strategy: {result['strategy_name']}")
    print(f"Total Return: {result['metrics'].total_return:.2%}")
    print(f"Sharpe Ratio: {result['metrics'].sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result['metrics'].max_drawdown:.2%}")
    print(f"Number of Trades: {result['metrics'].trades_count}")
    
    # Walk-forward optimization
    optimizer = WalkForwardOptimizer(backtester)
    parameter_ranges = {
        'short_window': [5, 10, 15, 20],
        'long_window': [20, 30, 40, 50]
    }
    
    wf_result = optimizer.optimize_strategy(
        MovingAverageCrossover,
        parameter_ranges,
        sample_data,
        train_window=252,
        test_window=63
    )
    
    print(f"\nWalk-Forward Results:")
    print(f"Average Sharpe: {wf_result['aggregated_metrics'].sharpe_ratio:.2f}")
    print(f"Average Return: {wf_result['aggregated_metrics'].annual_return:.2%}")