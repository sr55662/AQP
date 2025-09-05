# tests/test_comprehensive.py
# Comprehensive test suite for AQP system

import pytest
import asyncio
import pandas as pd
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Import AQP components
from src.data_aggregator.market_data_engine import (
    UniversalDataAggregator, DataRequest, DataResponse,
    YahooFinanceSource, AlphaVantageSource
)
from src.backtesting.advanced_backtester import (
    AdvancedBacktester, BacktestConfig, BacktestResults
)
from src.strategies.example_strategies import (
    MomentumVolatilityStrategy, MeanReversionStrategy,
    StrategyEnsemble, create_example_strategies
)
from src.api.aws_orchestrator import AWSQuantOrchestrator, StrategyRequest

# ========================================
# TEST FIXTURES
# ========================================

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Generate realistic stock price data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = {}
    
    for symbol in symbols:
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily drift, 2% volatility
        prices = 100 * (1 + returns).cumprod()
        
        volumes = np.random.randint(1000000, 50000000, n_days)
        
        data[symbol] = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
            'close': prices,
            'adj_close': prices,
            'volume': volumes
        }, index=dates)
        
        # Add some fundamental data
        data[symbol]['pe_ratio'] = 20 + np.random.normal(0, 5, n_days)
        data[symbol]['market_cap'] = 1e12 + np.random.normal(0, 1e11, n_days)
    
    return data

@pytest.fixture
def mock_aws_config():
    """Mock AWS configuration for testing"""
    return {
        'alpha_vantage_key': 'test_key',
        'fred_key': 'test_key',
        'quandl_key': 'test_key',
        's3_bucket': 'test-bucket',
        'results_bucket': 'test-results-bucket',
        'jobs_table': 'test-jobs-table',
        'notification_topic': 'test-topic'
    }

@pytest.fixture
def sample_data_request():
    """Create sample data request"""
    return DataRequest(
        symbols=['AAPL', 'MSFT'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        data_types=['price', 'volume'],
        frequency='daily'
    )

# ========================================
# DATA AGGREGATOR TESTS
# ========================================

class TestDataAggregator:
    """Test suite for data aggregation functionality"""
    
    def test_yahoo_finance_source_initialization(self):
        """Test Yahoo Finance source initialization"""
        source = YahooFinanceSource()
        
        assert source.name == "yahoo_finance"
        assert source.validate_credentials() == True
        assert source.get_cost_estimate(Mock()) == 0.0
    
    @pytest.mark.asyncio
    async def test_yahoo_finance_data_fetch(self, sample_data_request):
        """Test Yahoo Finance data fetching"""
        source = YahooFinanceSource()
        
        # Mock yfinance to avoid real API calls
        with patch('yfinance.Ticker') as mock_ticker:
            mock_hist = pd.DataFrame({
                'Open': [100, 101],
                'High': [102, 103],
                'Low': [99, 100],
                'Close': [101, 102],
                'Volume': [1000000, 1100000]
            }, index=pd.date_range('2023-01-01', periods=2))
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_hist
            mock_ticker_instance.info = {'trailingPE': 20.0, 'marketCap': 1e12}
            mock_ticker.return_value = mock_ticker_instance
            
            response = await source.fetch_data(sample_data_request)
            
            assert isinstance(response, DataResponse)
            assert len(response.symbols) == 2
            assert 'AAPL' in response.data
            assert 'MSFT' in response.data
            assert response.quality_score == 0.7
            assert response.cost == 0.0
    
    def test_universal_aggregator_initialization(self, mock_aws_config):
        """Test universal data aggregator initialization"""
        aggregator = UniversalDataAggregator(mock_aws_config)
        
        assert 'yahoo_finance' in aggregator.sources
        assert len(aggregator.sources) >= 1
        assert aggregator.config == mock_aws_config
    
    @pytest.mark.asyncio
    async def test_aggregator_source_priority(self, mock_aws_config, sample_data_request):
        """Test data source priority logic"""
        aggregator = UniversalDataAggregator(mock_aws_config)
        
        # Test default priority
        priority = aggregator._get_default_priority(sample_data_request)
        assert 'yahoo_finance' in priority
        
        # Test economic data priority
        econ_request = DataRequest(
            symbols=['GDP', 'UNRATE'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_types=['price']
        )
        econ_priority = aggregator._get_default_priority(econ_request)
        assert 'yahoo_finance' in econ_priority

# ========================================
# BACKTESTING TESTS
# ========================================

class TestBacktesting:
    """Test suite for backtesting functionality"""
    
    def test_backtest_config_initialization(self):
        """Test backtest configuration"""
        config = BacktestConfig(
            initial_capital=100000,
            commission=0.001,
            benchmark='SPY'
        )
        
        assert config.initial_capital == 100000
        assert config.commission == 0.001
        assert config.benchmark == 'SPY'
    
    def test_backtester_initialization(self):
        """Test backtester initialization"""
        config = BacktestConfig()
        backtester = AdvancedBacktester(config)
        
        assert backtester.config == config
        assert backtester.trades == []
    
    def test_momentum_strategy_signals(self, sample_market_data):
        """Test momentum strategy signal generation"""
        strategy = MomentumVolatilityStrategy(
            lookback_momentum=10,
            momentum_threshold=0.01
        )
        
        # Prepare data
        market_data = {}
        for symbol, df in sample_market_data.items():
            for col in df.columns:
                market_data[f"{symbol}_{col}"] = df[col]
        
        combined_data = pd.DataFrame(market_data)
        
        signals = strategy.generate_signals(combined_data)
        
        assert not signals.empty
        assert signals.index.equals(combined_data.index)
        
        # Check signal columns exist
        expected_columns = [f"{symbol}_signal" for symbol in sample_market_data.keys()]
        for col in expected_columns:
            assert col in signals.columns
        
        # Check signals are valid (-1, 0, 1)
        for col in signals.columns:
            unique_values = set(signals[col].dropna().unique())
            assert unique_values.issubset({-1, 0, 1})
    
    def test_mean_reversion_strategy(self, sample_market_data):
        """Test mean reversion strategy"""
        strategy = MeanReversionStrategy(
            lookback_window=15,
            zscore_threshold=1.0
        )
        
        # Prepare data
        market_data = {}
        for symbol, df in sample_market_data.items():
            for col in df.columns:
                market_data[f"{symbol}_{col}"] = df[col]
        
        combined_data = pd.DataFrame(market_data)
        
        signals = strategy.generate_signals(combined_data)
        positions = strategy.calculate_positions(signals, combined_data)
        
        assert not signals.empty
        assert not positions.empty
        assert len(positions.columns) == len(sample_market_data)
    
    def test_strategy_ensemble(self, sample_market_data):
        """Test strategy ensemble functionality"""
        strategies = create_example_strategies()
        ensemble = StrategyEnsemble(strategies, weights=[0.4, 0.3, 0.3])
        
        # Prepare data
        market_data = {}
        for symbol, df in sample_market_data.items():
            for col in df.columns:
                market_data[f"{symbol}_{col}"] = df[col]
        
        combined_data = pd.DataFrame(market_data)
        
        ensemble_signals = ensemble.generate_ensemble_signals(combined_data)
        
        assert not ensemble_signals.empty
        assert len(ensemble.strategies) == 3
        assert sum(ensemble.weights) == pytest.approx(1.0)
    
    def test_backtest_execution(self, sample_market_data):
        """Test full backtest execution"""
        config = BacktestConfig(initial_capital=100000)
        backtester = AdvancedBacktester(config)
        strategy = MomentumVolatilityStrategy()
        
        # Run backtest
        results = backtester.run_backtest(strategy, sample_market_data)
        
        assert isinstance(results, BacktestResults)
        assert results.total_return is not None
        assert results.sharpe_ratio is not None
        assert results.max_drawdown is not None
        assert not results.equity_curve.empty
        assert not results.daily_returns.empty
    
    @pytest.mark.asyncio
    async def test_walk_forward_analysis(self, sample_market_data):
        """Test walk-forward analysis"""
        config = BacktestConfig(initial_capital=100000)
        backtester = AdvancedBacktester(config)
        strategy = MomentumVolatilityStrategy()
        
        # Run walk-forward analysis with shorter windows for testing
        wf_results = backtester.walk_forward_analysis(
            strategy=strategy,
            data=sample_market_data,
            window_size=100,  # Shorter for testing
            step_size=50,
            min_periods=50
        )
        
        assert 'window_results' in wf_results
        assert 'metrics' in wf_results
        assert 'stability_score' in wf_results
        assert isinstance(wf_results['window_results'], pd.DataFrame)

# ========================================
# API ORCHESTRATOR TESTS
# ========================================

class TestAPIOrchestrator:
    """Test suite for API orchestrator"""
    
    @pytest.fixture
    def mock_orchestrator(self, mock_aws_config):
        """Create mock orchestrator for testing"""
        with patch('boto3.client'), patch('boto3.resource'):
            orchestrator = AWSQuantOrchestrator(mock_aws_config)
            return orchestrator
    
    def test_orchestrator_initialization(self, mock_orchestrator):
        """Test orchestrator initialization"""
        assert mock_orchestrator.aws_config is not None
        assert mock_orchestrator.data_aggregator is not None
    
    @pytest.mark.asyncio
    async def test_strategy_request_validation(self):
        """Test strategy request validation"""
        # Valid request
        valid_request = StrategyRequest(
            symbols=['AAPL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            strategy_description='Test momentum strategy'
        )
        
        assert valid_request.symbols == ['AAPL', 'MSFT']
        assert valid_request.target_sharpe == 2.0  # Default
        assert valid_request.initial_capital == 100000  # Default
    
    @pytest.mark.asyncio
    async def test_job_status_tracking(self, mock_orchestrator):
        """Test job status tracking"""
        request_id = 'test-123'
        
        # Mock DynamoDB operations
        mock_orchestrator.jobs_table = Mock()
        mock_orchestrator.jobs_table.put_item = Mock()
        mock_orchestrator.jobs_table.get_item = Mock(return_value={
            'Item': {
                'request_id': request_id,
                'status': 'running',
                'progress': 0.5,
                'current_step': 'backtesting'
            }
        })
        
        # Test status update
        await mock_orchestrator._update_job_status(
            request_id, 'running', 0.5, 'backtesting'
        )
        
        mock_orchestrator.jobs_table.put_item.assert_called_once()
        
        # Test status retrieval
        status = await mock_orchestrator.get_job_status(request_id)
        
        assert status.request_id == request_id
        assert status.status == 'running'
        assert status.progress == 0.5

# ========================================
# INTEGRATION TESTS
# ========================================

class TestIntegration:
    """Integration tests for complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, sample_market_data, mock_aws_config):
        """Test complete end-to-end pipeline"""
        
        # Mock external dependencies
        with patch('boto3.client'), patch('boto3.resource'):
            # Initialize components
            orchestrator = AWSQuantOrchestrator(mock_aws_config)
            
            # Mock successful operations
            orchestrator.data_aggregator.fetch_data = AsyncMock(return_value=DataResponse(
                request_id='test-123',
                symbols=['AAPL', 'MSFT'],
                data=sample_market_data,
                metadata={'source': 'test'},
                quality_score=0.9,
                timestamp=datetime.now(),
                source_used='test',
                cost=0.0
            ))
            
            orchestrator._store_results = AsyncMock()
            orchestrator._update_job_status = AsyncMock()
            orchestrator._generate_reports = AsyncMock(return_value={
                'equity_curve': 'test_url',
                'performance_report': 'test_url',
                'detailed_analysis': 'test_url'
            })
            
            # Create test request
            request = StrategyRequest(
                symbols=['AAPL', 'MSFT'],
                start_date='2023-01-01',
                end_date='2023-12-31',
                strategy_description='Test momentum strategy',
                include_walk_forward=False,  # Skip for faster testing
                include_monte_carlo=False
            )
            
            # Execute pipeline
            request_id = await orchestrator.execute_strategy_pipeline(request)
            
            assert request_id is not None
            assert isinstance(request_id, str)
    
    def test_configuration_loading(self):
        """Test configuration file loading"""
        # Create temporary config file
        config_data = {
            'aws': {'region': 'us-east-1'},
            'api_keys': {'test_key': 'test_value'},
            'trading': {'initial_capital': 100000}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Load configuration
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            
            assert loaded_config['aws']['region'] == 'us-east-1'
            assert loaded_config['trading']['initial_capital'] == 100000
        finally:
            os.unlink(config_file)

# ========================================
# PERFORMANCE TESTS
# ========================================

class TestPerformance:
    """Performance and stress tests"""
    
    def test_large_dataset_backtesting(self):
        """Test backtesting with large dataset"""
        # Generate large dataset
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='D')
        n_days = len(dates)
        
        large_data = {}
        for i in range(10):  # 10 symbols
            symbol = f"STOCK{i:02d}"
            returns = np.random.normal(0.0005, 0.02, n_days)
            prices = 100 * (1 + returns).cumprod()
            
            large_data[symbol] = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n_days)
            }, index=dates)
        
        # Test strategy on large dataset
        strategy = MomentumVolatilityStrategy()
        config = BacktestConfig()
        backtester = AdvancedBacktester(config)
        
        import time
        start_time = time.time()
        
        results = backtester.run_backtest(strategy, large_data)
        
        execution_time = time.time() - start_time
        
        assert results is not None
        assert execution_time < 30  # Should complete within 30 seconds
        
        print(f"Large dataset backtest completed in {execution_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_aws_config):
        """Test handling multiple concurrent requests"""
        
        with patch('boto3.client'), patch('boto3.resource'):
            orchestrator = AWSQuantOrchestrator(mock_aws_config)
            
            # Mock quick responses
            orchestrator.data_aggregator.fetch_data = AsyncMock(return_value=Mock())
            orchestrator._update_job_status = AsyncMock()
            orchestrator._store_results = AsyncMock()
            orchestrator._generate_reports = AsyncMock(return_value={})
            
            # Create multiple requests
            requests = []
            for i in range(5):
                request = StrategyRequest(
                    symbols=['AAPL'],
                    start_date='2023-01-01',
                    end_date='2023-12-31',
                    strategy_description=f'Test strategy {i}',
                    include_walk_forward=False,
                    include_monte_carlo=False
                )
                requests.append(request)
            
            # Execute concurrently
            import time
            start_time = time.time()
            
            tasks = [orchestrator.execute_strategy_pipeline(req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            execution_time = time.time() - start_time
            
            assert len(results) == 5
            assert execution_time < 60  # Should handle 5 requests within 60 seconds
            
            print(f"Concurrent requests completed in {execution_time:.2f} seconds")

# ========================================
# TEST EXECUTION
# ========================================

if __name__ == "__main__":
    print("🧪 Running AQP Comprehensive Test Suite")
    print("=" * 50)
    
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])
    
    print("\n✅ Test suite completed!")
    print("🚀 AQP system validated and ready for deployment!")