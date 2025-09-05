# A# src/ensemble/advanced_optimizer.py
# Advanced Ensemble Optimization Engine for Sharpe >2.0

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StrategyMetrics:
    """Individual strategy performance metrics"""
    name: str
    sharpe_ratio: float
    annual_return: float
    volatility: float
    max_drawdown: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    calmar_ratio: float
    sortino_ratio: float
    omega_ratio: float
    returns: np.ndarray
    
@dataclass
class EnsembleConfig:
    """Configuration for ensemble optimization"""
    target_sharpe: float = 2.0
    max_individual_weight: float = 0.4
    min_individual_weight: float = 0.05
    max_drawdown_constraint: float = 0.15
    var_95_constraint: float = 0.03
    rebalance_frequency: int = 21  # days
    lookback_period: int = 252  # trading days
    min_correlation_threshold: float = 0.7  # flag highly correlated strategies
    
class AdvancedEnsembleOptimizer:
    """
    Advanced ensemble optimization engine targeting Sharpe >2.0
    
    Features:
    - Dynamic weight allocation based on risk-adjusted performance
    - Correlation-aware optimization
    - Regime detection and adaptation
    - Real-time risk monitoring
    - Multi-objective optimization (Sharpe, drawdown, tail risk)
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self.optimal_weights: Optional[np.ndarray] = None
        self.ensemble_performance: Optional[Dict] = None
        self.regime_state: str = "normal"
        
    def add_strategy(self, strategy_metrics: StrategyMetrics) -> None:
        """Add a strategy to the ensemble"""
        self.strategy_metrics[strategy_metrics.name] = strategy_metrics
        print(f"Added strategy '{strategy_metrics.name}' with Sharpe {strategy_metrics.sharpe_ratio:.2f}")
        
    def calculate_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix with robust estimation"""
        if len(self.strategy_metrics) < 2:
            return np.array([[1.0]])
            
        returns_matrix = np.column_stack([
            metrics.returns for metrics in self.strategy_metrics.values()
        ])
        
        # Use Ledoit-Wolf shrinkage for robust correlation estimation
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns_matrix).covariance_
        
        # Convert to correlation matrix
        std_devs = np.sqrt(np.diag(cov_matrix))
        correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
        
    def detect_regime(self, market_data: pd.DataFrame) -> str:
        """Detect market regime for adaptive optimization"""
        # Calculate rolling volatility and trend
        returns = market_data['close'].pct_change().dropna()
        rolling_vol = returns.rolling(21).std() * np.sqrt(252)
        rolling_trend = returns.rolling(63).mean() * 252
        
        current_vol = rolling_vol.iloc[-1]
        current_trend = rolling_trend.iloc[-1]
        vol_percentile = (rolling_vol.iloc[-1] > rolling_vol.quantile(0.8))
        trend_positive = current_trend > 0.05
        
        if vol_percentile:
            regime = "high_volatility"
        elif trend_positive and not vol_percentile:
            regime = "bull_market"
        elif not trend_positive and not vol_percentile:
            regime = "bear_market"
        else:
            regime = "normal"
            
        self.regime_state = regime
        return regime
        
    def portfolio_sharpe_objective(self, weights: np.ndarray) -> float:
        """Objective function: minimize negative Sharpe ratio"""
        portfolio_return, portfolio_vol = self._calculate_portfolio_metrics(weights)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        return -sharpe  # Minimize negative Sharpe (maximize Sharpe)
        
    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float]:
        """Calculate portfolio return and volatility"""
        strategy_list = list(self.strategy_metrics.values())
        
        # Portfolio expected return
        portfolio_return = sum(w * s.annual_return for w, s in zip(weights, strategy_list))
        
        # Portfolio volatility using correlation matrix
        if self.correlation_matrix is not None:
            strategy_vols = np.array([s.volatility for s in strategy_list])
            portfolio_var = np.dot(weights, np.dot(self.correlation_matrix * np.outer(strategy_vols, strategy_vols), weights))
            portfolio_vol = np.sqrt(portfolio_var)
        else:
            # Fallback: assume zero correlation
            portfolio_vol = np.sqrt(sum((w * s.volatility) ** 2 for w, s in zip(weights, strategy_list)))
            
        return portfolio_return, portfolio_vol
        
    def portfolio_constraints(self, weights: np.ndarray) -> List[float]:
        """Calculate constraint violations"""
        violations = []
        
        # Calculate portfolio metrics
        strategy_list = list(self.strategy_metrics.values())
        
        # Max drawdown constraint
        # Approximate portfolio drawdown as weighted average (conservative)
        portfolio_drawdown = sum(w * s.max_drawdown for w, s in zip(weights, strategy_list))
        violations.append(self.config.max_drawdown_constraint - portfolio_drawdown)
        
        # VaR constraint (approximate)
        portfolio_var = sum(w * s.var_95 for w, s in zip(weights, strategy_list))
        violations.append(self.config.var_95_constraint - portfolio_var)
        
        return violations
        
    def optimize_ensemble(self) -> Dict[str, Any]:
        """
        Optimize ensemble weights for maximum Sharpe ratio
        Subject to risk constraints and position limits
        """
        if len(self.strategy_metrics) < 2:
            raise ValueError("Need at least 2 strategies for ensemble optimization")
            
        # Calculate correlation matrix
        self.calculate_correlation_matrix()
        
        n_strategies = len(self.strategy_metrics)
        strategy_names = list(self.strategy_metrics.keys())
        
        # Initial weights (equal allocation)
        x0 = np.ones(n_strategies) / n_strategies
        
        # Bounds: min/max weight per strategy
        bounds = [(self.config.min_individual_weight, self.config.max_individual_weight) 
                 for _ in range(n_strategies)]
        
        # Constraints
        constraints = [
            # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            # Risk constraints
            {'type': 'ineq', 'fun': lambda x: self.portfolio_constraints(x)}
        ]
        
        # Optimize
        result = minimize(
            self.portfolio_sharpe_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
            # Fallback to equal weights if optimization fails
            self.optimal_weights = np.ones(n_strategies) / n_strategies
        else:
            self.optimal_weights = result.x
            
        # Calculate ensemble performance
        portfolio_return, portfolio_vol = self._calculate_portfolio_metrics(self.optimal_weights)
        portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate additional metrics
        portfolio_drawdown = sum(w * s.max_drawdown for w, s in zip(self.optimal_weights, self.strategy_metrics.values()))
        portfolio_var = sum(w * s.var_95 for w, s in zip(self.optimal_weights, self.strategy_metrics.values()))
        
        self.ensemble_performance = {
            'sharpe_ratio': portfolio_sharpe,
            'annual_return': portfolio_return,
            'volatility': portfolio_vol,
            'max_drawdown': portfolio_drawdown,
            'var_95': portfolio_var,
            'weights': dict(zip(strategy_names, self.optimal_weights)),
            'correlation_matrix': self.correlation_matrix,
            'regime': self.regime_state,
            'target_achieved': portfolio_sharpe >= self.config.target_sharpe
        }
        
        return self.ensemble_performance
        
    def adaptive_rebalance(self, new_performance_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Adaptive rebalancing based on recent performance
        """
        # Update strategy metrics with new data
        for strategy_name, new_returns in new_performance_data.items():
            if strategy_name in self.strategy_metrics:
                # Update with recent performance (exponential weighting)
                old_metrics = self.strategy_metrics[strategy_name]
                
                # Calculate new metrics
                new_sharpe = np.mean(new_returns) / np.std(new_returns) * np.sqrt(252) if np.std(new_returns) > 0 else 0
                new_vol = np.std(new_returns) * np.sqrt(252)
                new_return = np.mean(new_returns) * 252
                
                # Exponential weighting (0.8 old, 0.2 new)
                alpha = 0.2
                updated_sharpe = (1 - alpha) * old_metrics.sharpe_ratio + alpha * new_sharpe
                updated_vol = (1 - alpha) * old_metrics.volatility + alpha * new_vol
                updated_return = (1 - alpha) * old_metrics.annual_return + alpha * new_return
                
                # Update the strategy metrics
                old_metrics.sharpe_ratio = updated_sharpe
                old_metrics.volatility = updated_vol
                old_metrics.annual_return = updated_return
                
        # Re-optimize with updated metrics
        return self.optimize_ensemble()
        
    def risk_monitor(self) -> Dict[str, Any]:
        """
        Real-time risk monitoring and alerts
        """
        if not self.ensemble_performance:
            return {'status': 'no_data', 'alerts': []}
            
        alerts = []
        
        # Check if ensemble is underperforming target
        current_sharpe = self.ensemble_performance['sharpe_ratio']
        if current_sharpe < self.config.target_sharpe * 0.8:  # 20% tolerance
            alerts.append({
                'type': 'performance',
                'severity': 'warning',
                'message': f"Ensemble Sharpe {current_sharpe:.2f} below target {self.config.target_sharpe}"
            })
            
        # Check correlation warnings
        if self.correlation_matrix is not None:
            max_correlation = np.max(self.correlation_matrix[np.triu_indices_from(self.correlation_matrix, k=1)])
            if max_correlation > self.config.min_correlation_threshold:
                alerts.append({
                    'type': 'correlation',
                    'severity': 'warning', 
                    'message': f"High correlation detected: {max_correlation:.2f}"
                })
                
        # Check risk constraints
        if self.ensemble_performance['max_drawdown'] > self.config.max_drawdown_constraint:
            alerts.append({
                'type': 'risk',
                'severity': 'critical',
                'message': f"Drawdown {self.ensemble_performance['max_drawdown']:.1%} exceeds limit"
            })
            
        return {
            'status': 'active',
            'current_sharpe': current_sharpe,
            'target_sharpe': self.config.target_sharpe,
            'alerts': alerts,
            'regime': self.regime_state,
            'last_rebalance': pd.Timestamp.now()
        }
        
    def generate_ensemble_report(self) -> str:
        """Generate comprehensive ensemble performance report"""
        if not self.ensemble_performance:
            return "No ensemble performance data available"
            
        report = f"""
🚀 ENSEMBLE OPTIMIZATION REPORT
==============================

🎯 TARGET ACHIEVEMENT
Target Sharpe: {self.config.target_sharpe:.2f}
Achieved Sharpe: {self.ensemble_performance['sharpe_ratio']:.2f}
Status: {'✅ TARGET ACHIEVED' if self.ensemble_performance['target_achieved'] else '📈 IN PROGRESS'}

📊 PORTFOLIO METRICS
Annual Return: {self.ensemble_performance['annual_return']:.1%}
Volatility: {self.ensemble_performance['volatility']:.1%}
Max Drawdown: {self.ensemble_performance['max_drawdown']:.1%}
VaR 95%: {self.ensemble_performance['var_95']:.1%}

⚖️ OPTIMAL WEIGHTS
"""
        for name, weight in self.ensemble_performance['weights'].items():
            individual_sharpe = self.strategy_metrics[name].sharpe_ratio
            report += f"{name}: {weight:.1%} (Sharpe: {individual_sharpe:.2f})\n"
            
        report += f"""
🔗 DIVERSIFICATION ANALYSIS
Regime: {self.regime_state}
Correlation Matrix Shape: {self.correlation_matrix.shape if self.correlation_matrix is not None else 'N/A'}

📈 ENSEMBLE ADVANTAGE
Individual Average Sharpe: {np.mean([s.sharpe_ratio for s in self.strategy_metrics.values()]):.2f}
Ensemble Sharpe: {self.ensemble_performance['sharpe_ratio']:.2f}
Diversification Ratio: {self.ensemble_performance['sharpe_ratio'] / np.mean([s.sharpe_ratio for s in self.strategy_metrics.values()]):.2f}x
"""
        return report

# Example usage and testing
if __name__ == "__main__":
    # Create sample strategy metrics for testing
    np.random.seed(42)
    
    # Generate correlated returns for realistic testing
    base_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), 252)
    
    strategies = []
    for i in range(4):
        # Add some noise and different risk characteristics
        noise = np.random.normal(0, 0.05/np.sqrt(252), 252)
        strategy_returns = base_returns + noise * (0.5 + i * 0.3)
        
        annual_ret = np.mean(strategy_returns) * 252
        vol = np.std(strategy_returns) * np.sqrt(252)
        sharpe = annual_ret / vol if vol > 0 else 0
        
        strategy = StrategyMetrics(
            name=f"Strategy_{i+1}",
            sharpe_ratio=sharpe,
            annual_return=annual_ret,
            volatility=vol,
            max_drawdown=np.random.uniform(0.08, 0.15),
            skewness=-0.5,
            kurtosis=4.0,
            var_95=0.02,
            cvar_95=0.03,
            calmar_ratio=sharpe * 0.8,
            sortino_ratio=sharpe * 1.2,
            omega_ratio=1.3,
            returns=strategy_returns
        )
        strategies.append(strategy)
    
    # Test ensemble optimization
    config = EnsembleConfig(target_sharpe=2.0)
    optimizer = AdvancedEnsembleOptimizer(config)
    
    # Add strategies
    for strategy in strategies:
        optimizer.add_strategy(strategy)
    
    # Optimize ensemble
    result = optimizer.optimize_ensemble()
    
    print(optimizer.generate_ensemble_report())
    print(f"\n🎯 Risk Monitor: {optimizer.risk_monitor()}")