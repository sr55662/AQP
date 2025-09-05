# src/monitoring/realtime_performance_monitor.py
# Real-time Performance Monitoring & Auto-Rebalancing System

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import our ensemble components
from ensemble.advanced_optimizer import AdvancedEnsembleOptimizer, StrategyMetrics, EnsembleConfig
from strategy_generation.llm_specialization_engine import LLMSpecializationEngine, GeneratedStrategy

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RebalanceReason(Enum):
    SCHEDULED = "scheduled_rebalance"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CORRELATION_INCREASE = "correlation_increase"
    RISK_BREACH = "risk_breach"
    REGIME_CHANGE = "regime_change"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    timestamp: datetime
    level: AlertLevel
    strategy_name: Optional[str]
    message: str
    metric: str
    current_value: float
    threshold_value: float
    recommended_action: str

@dataclass
class RebalanceEvent:
    """Rebalancing event record"""
    timestamp: datetime
    reason: RebalanceReason
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    expected_sharpe_before: float
    expected_sharpe_after: float
    triggered_by: str

@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring"""
    # Performance thresholds
    min_ensemble_sharpe: float = 1.8  # Alert if below this
    target_ensemble_sharpe: float = 2.0
    max_individual_drawdown: float = 0.12
    max_ensemble_drawdown: float = 0.08
    max_correlation_threshold: float = 0.6
    
    # Monitoring intervals
    performance_check_interval: int = 300  # seconds (5 minutes)
    rebalance_check_interval: int = 3600   # seconds (1 hour)
    daily_report_time: str = "09:30"       # Market open
    
    # Rebalancing triggers
    sharpe_degradation_threshold: float = 0.3  # Rebalance if Sharpe drops by this much
    correlation_increase_threshold: float = 0.2  # Rebalance if avg correlation increases by this
    min_rebalance_interval: int = 86400 * 3     # Minimum 3 days between rebalances
    
    # Risk controls
    emergency_stop_drawdown: float = 0.15      # Emergency stop if exceeded
    max_strategy_weight: float = 0.5           # Maximum allocation to any strategy
    min_strategies_required: int = 3           # Minimum strategies for ensemble

class RealTimePerformanceMonitor:
    """
    Real-time Performance Monitor & Auto-Rebalancer
    
    Continuously monitors ensemble performance and automatically rebalances
    to maintain Sharpe >2.0 target while managing risk
    """
    
    def __init__(self, 
                 ensemble_optimizer: AdvancedEnsembleOptimizer,
                 strategy_engine: LLMSpecializationEngine,
                 config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.ensemble_optimizer = ensemble_optimizer
        self.strategy_engine = strategy_engine
        
        # State tracking
        self.is_monitoring = False
        self.last_rebalance: Optional[datetime] = None
        self.performance_history: List[Dict] = []
        self.alerts: List[PerformanceAlert] = []
        self.rebalance_history: List[RebalanceEvent] = []
        
        # Current metrics cache
        self.current_metrics: Optional[Dict] = None
        self.baseline_performance: Optional[Dict] = None
        
        # Callbacks for external integration
        self.alert_callbacks: List[Callable] = []
        self.rebalance_callbacks: List[Callable] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def start_monitoring(self) -> None:
        """Start real-time monitoring loop"""
        if self.is_monitoring:
            self.logger.warning("Monitoring already active")
            return
            
        self.is_monitoring = True
        self.logger.info("Starting real-time performance monitoring")
        
        # Set baseline performance
        self.baseline_performance = await self.calculate_current_performance()
        
        # Start monitoring tasks
        await asyncio.gather(
            self.performance_monitoring_loop(),
            self.rebalancing_monitoring_loop(),
            self.daily_reporting_loop()
        )
        
    async def stop_monitoring(self) -> None:
        """Stop monitoring loops"""
        self.is_monitoring = False
        self.logger.info("Stopping real-time performance monitoring")
        
    async def performance_monitoring_loop(self) -> None:
        """Main performance monitoring loop"""
        while self.is_monitoring:
            try:
                # Calculate current performance
                current_perf = await self.calculate_current_performance()
                self.current_metrics = current_perf
                
                # Store performance history
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'performance': current_perf
                })
                
                # Limit history size
                if len(self.performance_history) > 1440:  # 24 hours at 1-minute intervals
                    self.performance_history = self.performance_history[-1440:]
                
                # Check for alerts
                await self.check_performance_alerts(current_perf)
                
                # Check emergency conditions
                await self.check_emergency_conditions(current_perf)
                
                await asyncio.sleep(self.config.performance_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
                
    async def rebalancing_monitoring_loop(self) -> None:
        """Rebalancing decision monitoring loop"""
        while self.is_monitoring:
            try:
                if await self.should_rebalance():
                    await self.execute_rebalance()
                    
                await asyncio.sleep(self.config.rebalance_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in rebalancing monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
                
    async def daily_reporting_loop(self) -> None:
        """Daily performance reporting loop"""
        while self.is_monitoring:
            try:
                now = datetime.now()
                report_time = datetime.strptime(self.config.daily_report_time, "%H:%M").time()
                
                if now.time() >= report_time:
                    await self.generate_daily_report()
                    # Wait until next day
                    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                    sleep_seconds = (tomorrow - now).total_seconds()
                    await asyncio.sleep(sleep_seconds)
                else:
                    # Wait until report time
                    today_report_time = now.replace(
                        hour=report_time.hour, 
                        minute=report_time.minute, 
                        second=0, 
                        microsecond=0
                    )
                    if today_report_time > now:
                        sleep_seconds = (today_report_time - now).total_seconds()
                    else:
                        tomorrow_report_time = today_report_time + timedelta(days=1)
                        sleep_seconds = (tomorrow_report_time - now).total_seconds()
                    
                    await asyncio.sleep(min(sleep_seconds, 3600))  # Check at least every hour
                    
            except Exception as e:
                self.logger.error(f"Error in daily reporting: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
                
    async def calculate_current_performance(self) -> Dict[str, Any]:
        """Calculate current ensemble performance metrics"""
        if not self.ensemble_optimizer.ensemble_performance:
            return {'status': 'no_data', 'timestamp': datetime.now()}
            
        # Get latest ensemble performance
        ensemble_perf = self.ensemble_optimizer.ensemble_performance.copy()
        
        # Add timestamp and additional metrics
        ensemble_perf.update({
            'timestamp': datetime.now(),
            'sharpe_vs_target': ensemble_perf['sharpe_ratio'] / self.config.target_ensemble_sharpe,
            'performance_status': self._get_performance_status(ensemble_perf['sharpe_ratio']),
            'risk_status': self._get_risk_status(ensemble_perf),
            'days_since_rebalance': self._days_since_last_rebalance()
        })
        
        return ensemble_perf
        
    def _get_performance_status(self, sharpe_ratio: float) -> str:
        """Get performance status based on Sharpe ratio"""
        if sharpe_ratio >= self.config.target_ensemble_sharpe:
            return "target_achieved"
        elif sharpe_ratio >= self.config.min_ensemble_sharpe:
            return "acceptable"
        elif sharpe_ratio >= self.config.min_ensemble_sharpe * 0.8:
            return "warning"
        else:
            return "critical"
            
    def _get_risk_status(self, performance: Dict) -> str:
        """Get risk status based on drawdown and VaR"""
        if performance['max_drawdown'] > self.config.max_ensemble_drawdown:
            return "high_risk"
        elif performance['var_95'] > 0.025:  # 2.5% daily VaR
            return "elevated_risk"
        else:
            return "normal_risk"
            
    def _days_since_last_rebalance(self) -> int:
        """Calculate days since last rebalance"""
        if not self.last_rebalance:
            return 0
        return (datetime.now() - self.last_rebalance).days
        
    async def check_performance_alerts(self, current_perf: Dict) -> None:
        """Check for performance-based alerts"""
        alerts = []
        
        # Sharpe ratio alerts
        current_sharpe = current_perf.get('sharpe_ratio', 0)
        if current_sharpe < self.config.min_ensemble_sharpe:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING if current_sharpe > self.config.min_ensemble_sharpe * 0.8 else AlertLevel.CRITICAL,
                strategy_name=None,
                message=f"Ensemble Sharpe ratio {current_sharpe:.2f} below minimum {self.config.min_ensemble_sharpe}",
                metric="ensemble_sharpe",
                current_value=current_sharpe,
                threshold_value=self.config.min_ensemble_sharpe,
                recommended_action="Consider rebalancing or regenerating strategies"
            ))
            
        # Drawdown alerts
        current_dd = current_perf.get('max_drawdown', 0)
        if current_dd > self.config.max_ensemble_drawdown:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                strategy_name=None,
                message=f"Ensemble drawdown {current_dd:.1%} exceeds threshold {self.config.max_ensemble_drawdown:.1%}",
                metric="max_drawdown",
                current_value=current_dd,
                threshold_value=self.config.max_ensemble_drawdown,
                recommended_action="Review risk management and consider defensive rebalancing"
            ))
            
        # Correlation alerts
        if hasattr(self.ensemble_optimizer, 'correlation_matrix') and self.ensemble_optimizer.correlation_matrix is not None:
            avg_correlation = np.mean(self.ensemble_optimizer.correlation_matrix[np.triu_indices_from(self.ensemble_optimizer.correlation_matrix, k=1)])
            if avg_correlation > self.config.max_correlation_threshold:
                alerts.append(PerformanceAlert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    strategy_name=None,
                    message=f"Average strategy correlation {avg_correlation:.2f} above threshold {self.config.max_correlation_threshold}",
                    metric="avg_correlation",
                    current_value=avg_correlation,
                    threshold_value=self.config.max_correlation_threshold,
                    recommended_action="Generate new uncorrelated strategies"
                ))
                
        # Process alerts
        for alert in alerts:
            await self.process_alert(alert)
            
    async def check_emergency_conditions(self, current_perf: Dict) -> None:
        """Check for emergency stop conditions"""
        current_dd = current_perf.get('max_drawdown', 0)
        
        if current_dd > self.config.emergency_stop_drawdown:
            emergency_alert = PerformanceAlert(
                timestamp=datetime.now(),
                level=AlertLevel.EMERGENCY,
                strategy_name=None,
                message=f"EMERGENCY: Drawdown {current_dd:.1%} exceeds emergency threshold {self.config.emergency_stop_drawdown:.1%}",
                metric="emergency_drawdown",
                current_value=current_dd,
                threshold_value=self.config.emergency_stop_drawdown,
                recommended_action="IMMEDIATE ACTION: Stop trading and review all positions"
            )
            
            await self.process_alert(emergency_alert)
            await self.trigger_emergency_rebalance()
            
    async def should_rebalance(self) -> bool:
        """Determine if rebalancing is needed"""
        if not self.current_metrics or not self.baseline_performance:
            return False
            
        # Check minimum interval
        if (self.last_rebalance and 
            (datetime.now() - self.last_rebalance).total_seconds() < self.config.min_rebalance_interval):
            return False
            
        current_sharpe = self.current_metrics.get('sharpe_ratio', 0)
        baseline_sharpe = self.baseline_performance.get('sharpe_ratio', 0)
        
        # Performance degradation check
        if baseline_sharpe > 0 and (baseline_sharpe - current_sharpe) > self.config.sharpe_degradation_threshold:
            self.logger.info(f"Rebalance triggered: Sharpe degradation {baseline_sharpe:.2f} -> {current_sharpe:.2f}")
            return True
            
        # Correlation increase check
        if hasattr(self.ensemble_optimizer, 'correlation_matrix') and self.ensemble_optimizer.correlation_matrix is not None:
            current_avg_corr = np.mean(self.ensemble_optimizer.correlation_matrix[np.triu_indices_from(self.ensemble_optimizer.correlation_matrix, k=1)])
            baseline_avg_corr = 0.3  # Assumed baseline
            
            if (current_avg_corr - baseline_avg_corr) > self.config.correlation_increase_threshold:
                self.logger.info(f"Rebalance triggered: Correlation increase {baseline_avg_corr:.2f} -> {current_avg_corr:.2f}")
                return True
                
        # Risk breach check
        if self.current_metrics.get('max_drawdown', 0) > self.config.max_ensemble_drawdown:
            self.logger.info("Rebalance triggered: Risk breach")
            return True
            
        return False
        
    async def execute_rebalance(self) -> None:
        """Execute portfolio rebalancing"""
        if not self.current_metrics:
            self.logger.error("Cannot rebalance: No current metrics available")
            return
            
        self.logger.info("Executing portfolio rebalance...")
        
        # Store old weights
        old_weights = self.ensemble_optimizer.optimal_weights.copy() if self.ensemble_optimizer.optimal_weights is not None else {}
        old_sharpe = self.current_metrics.get('sharpe_ratio', 0)
        
        # Re-optimize ensemble
        try:
            new_performance = self.ensemble_optimizer.optimize_ensemble()
            new_sharpe = new_performance.get('sharpe_ratio', 0)
            new_weights = new_performance.get('weights', {})
            
            # Record rebalance event
            rebalance_event = RebalanceEvent(
                timestamp=datetime.now(),
                reason=RebalanceReason.PERFORMANCE_DEGRADATION,  # Could be more sophisticated
                old_weights=old_weights,
                new_weights=new_weights,
                expected_sharpe_before=old_sharpe,
                expected_sharpe_after=new_sharpe,
                triggered_by="automated_monitor"
            )
            
            self.rebalance_history.append(rebalance_event)
            self.last_rebalance = datetime.now()
            
            # Update baseline
            self.baseline_performance = await self.calculate_current_performance()
            
            # Notify callbacks
            for callback in self.rebalance_callbacks:
                await callback(rebalance_event)
                
            self.logger.info(f"Rebalance complete: Sharpe {old_sharpe:.2f} -> {new_sharpe:.2f}")
            
        except Exception as e:
            self.logger.error(f"Rebalance failed: {e}")
            
    async def trigger_emergency_rebalance(self) -> None:
        """Trigger emergency rebalancing with defensive allocation"""
        self.logger.critical("Triggering emergency rebalance")
        
        # Move to defensive equal-weight allocation
        n_strategies = len(self.ensemble_optimizer.strategy_metrics)
        if n_strategies > 0:
            equal_weights = {name: 1.0/n_strategies for name in self.ensemble_optimizer.strategy_metrics.keys()}
            
            rebalance_event = RebalanceEvent(
                timestamp=datetime.now(),
                reason=RebalanceReason.EMERGENCY_STOP,
                old_weights=self.ensemble_optimizer.optimal_weights.copy() if self.ensemble_optimizer.optimal_weights is not None else {},
                new_weights=equal_weights,
                expected_sharpe_before=self.current_metrics.get('sharpe_ratio', 0),
                expected_sharpe_after=1.0,  # Conservative estimate
                triggered_by="emergency_stop"
            )
            
            self.rebalance_history.append(rebalance_event)
            self.last_rebalance = datetime.now()
            
            # Notify all callbacks immediately
            for callback in self.rebalance_callbacks:
                await callback(rebalance_event)
                
    async def process_alert(self, alert: PerformanceAlert) -> None:
        """Process and distribute alerts"""
        self.alerts.append(alert)
        
        # Limit alert history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
            
        # Log alert
        log_level = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.CRITICAL: self.logger.critical,
            AlertLevel.EMERGENCY: self.logger.critical
        }
        log_level[alert.level](f"Alert: {alert.message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            await callback(alert)
            
    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily performance report"""
        if not self.current_metrics:
            return {'error': 'No performance data available'}
            
        # Calculate daily statistics
        today_performance = []
        yesterday = datetime.now() - timedelta(days=1)
        
        for record in self.performance_history:
            if record['timestamp'] >= yesterday:
                today_performance.append(record['performance'])
                
        # Summary statistics
        report = {
            'date': datetime.now().date(),
            'current_performance': self.current_metrics,
            'target_achievement': {
                'current_sharpe': self.current_metrics.get('sharpe_ratio', 0),
                'target_sharpe': self.config.target_ensemble_sharpe,
                'achievement_ratio': self.current_metrics.get('sharpe_ratio', 0) / self.config.target_ensemble_sharpe,
                'status': 'ACHIEVED' if self.current_metrics.get('sharpe_ratio', 0) >= self.config.target_ensemble_sharpe else 'IN PROGRESS'
            },
            'daily_alerts': [alert for alert in self.alerts if alert.timestamp >= yesterday],
            'recent_rebalances': [rb for rb in self.rebalance_history if rb.timestamp >= yesterday],
            'monitoring_health': {
                'monitoring_active': self.is_monitoring,
                'data_points_collected': len(today_performance),
                'last_update': self.current_metrics.get('timestamp'),
                'days_since_rebalance': self._days_since_last_rebalance()
            }
        }
        
        self.logger.info(f"Daily Report Generated: Sharpe {report['target_achievement']['current_sharpe']:.2f} vs Target {report['target_achievement']['target_sharpe']:.2f}")
        
        return report
        
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
        
    def add_rebalance_callback(self, callback: Callable[[RebalanceEvent], None]) -> None:
        """Add callback for rebalance notifications"""
        self.rebalance_callbacks.append(callback)
        
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'is_monitoring': self.is_monitoring,
            'current_metrics': self.current_metrics,
            'alerts_today': len([a for a in self.alerts if a.timestamp.date() == datetime.now().date()]),
            'last_rebalance': self.last_rebalance,
            'days_since_rebalance': self._days_since_last_rebalance(),
            'performance_history_size': len(self.performance_history),
            'target_achievement': self.current_metrics.get('sharpe_ratio', 0) >= self.config.target_ensemble_sharpe if self.current_metrics else False
        }

# Example integration class showing how everything works together
class AQPMonitoringSystem:
    """
    Complete AQP Monitoring System integrating all components
    """
    
    def __init__(self):
        self.ensemble_optimizer = AdvancedEnsembleOptimizer()
        self.strategy_engine = LLMSpecializationEngine()
        self.performance_monitor = RealTimePerformanceMonitor(
            self.ensemble_optimizer,
            self.strategy_engine
        )
        
        # Setup alert handlers
        self.performance_monitor.add_alert_callback(self.handle_alert)
        self.performance_monitor.add_rebalance_callback(self.handle_rebalance)
        
    async def initialize_system(self) -> None:
        """Initialize the complete system"""
        print("🚀 Initializing AQP Monitoring System...")
        
        # 1. Generate diversified strategies
        strategies = await self.strategy_engine.generate_diversified_ensemble(target_ensemble_sharpe=2.0)
        print(f"Generated {len(strategies)} strategies")
        
        # 2. Convert to strategy metrics (placeholder - would use real backtesting)
        for strategy in strategies:
            # Mock strategy metrics - replace with real backtesting
            mock_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), 252)
            
            strategy_metrics = StrategyMetrics(
                name=strategy.name,
                sharpe_ratio=strategy.expected_sharpe,
                annual_return=np.mean(mock_returns) * 252,
                volatility=np.std(mock_returns) * np.sqrt(252),
                max_drawdown=0.08,
                skewness=-0.2,
                kurtosis=3.5,
                var_95=0.02,
                cvar_95=0.025,
                calmar_ratio=strategy.expected_sharpe * 0.8,
                sortino_ratio=strategy.expected_sharpe * 1.1,
                omega_ratio=1.2,
                returns=mock_returns
            )
            
            self.ensemble_optimizer.add_strategy(strategy_metrics)
            
        # 3. Optimize ensemble
        ensemble_result = self.ensemble_optimizer.optimize_ensemble()
        print(f"Ensemble optimized: Sharpe {ensemble_result['sharpe_ratio']:.2f}")
        
        # 4. Start monitoring
        print("Starting real-time monitoring...")
        return ensemble_result
        
    async def handle_alert(self, alert: PerformanceAlert) -> None:
        """Handle performance alerts"""
        print(f"🚨 ALERT [{alert.level.value.upper()}]: {alert.message}")
        
        # Could integrate with external alerting systems here
        # - Send to Slack/Discord
        # - Email notifications
        # - PagerDuty for critical alerts
        
    async def handle_rebalance(self, rebalance_event: RebalanceEvent) -> None:
        """Handle rebalancing events"""
        print(f"⚖️ REBALANCE: {rebalance_event.reason.value}")
        print(f"   Sharpe: {rebalance_event.expected_sharpe_before:.2f} -> {rebalance_event.expected_sharpe_after:.2f}")
        
        # Could integrate with trading systems here
        # - Execute trades via broker API
        # - Update position management systems
        # - Log to trade database

# Example usage
if __name__ == "__main__":
    async def run_monitoring_demo():
        """Demonstrate the complete monitoring system"""
        system = AQPMonitoringSystem()
        
        # Initialize system
        result = await system.initialize_system()
        
        print(f"\n🎯 ENSEMBLE PERFORMANCE")
        print(f"Target Sharpe: {system.performance_monitor.config.target_ensemble_sharpe}")
        print(f"Achieved Sharpe: {result['sharpe_ratio']:.2f}")
        print(f"Status: {'✅ TARGET ACHIEVED' if result['target_achieved'] else '📈 IN PROGRESS'}")
        
        # Start monitoring (run for demo period)
        print(f"\n📊 Starting monitoring for 30 seconds...")
        
        monitor_task = asyncio.create_task(system.performance_monitor.start_monitoring())
        
        # Let it run for 30 seconds
        await asyncio.sleep(30)
        
        # Stop monitoring
        await system.performance_monitor.stop_monitoring()
        
        # Generate final report
        status = system.performance_monitor.get_monitoring_status()
        print(f"\n📋 FINAL STATUS")
        print(f"Target Achievement: {'✅ YES' if status['target_achievement'] else '❌ NO'}")
        print(f"Alerts Generated: {status['alerts_today']}")
        print(f"Monitoring Active: {status['is_monitoring']}")
        
    # Run the demo
    asyncio.run(run_monitoring_demo())
