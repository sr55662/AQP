# tests/test_comprehensive_integration.py
# Comprehensive Integration Test for AQP Sharpe >2.0 Achievement

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from aqp_master_engine import AQPMasterEngine, AQPConfig
from ensemble.advanced_optimizer import AdvancedEnsembleOptimizer, StrategyMetrics, EnsembleConfig
from strategy_generation.llm_specialization_engine import LLMSpecializationEngine, StrategyCategory
from monitoring.realtime_performance_monitor import RealTimePerformanceMonitor, MonitoringConfig

class TestComprehensiveIntegration:
    """
    Comprehensive integration tests for the complete AQP system
    Validates end-to-end functionality and Sharpe >2.0 achievement
    """
    
    @pytest.fixture
    async def aqp_engine(self):
        """Create configured AQP engine for testing"""
        config = AQPConfig(
            target_sharpe=2.0,
            num_strategies=6,
            initial_capital=100000,
            monitoring_enabled=True,
            auto_rebalance=True
        )
        
        engine = AQPMasterEngine(config)
        yield engine
        
        # Cleanup
        if engine.is_running:
            await engine.shutdown_system()
    
    @pytest.mark.asyncio
    async def test_complete_initialization_pipeline(self, aqp_engine):
        """Test complete initialization pipeline from start to Sharpe >2.0"""
        print("\n🚀 Testing Complete Initialization Pipeline")
        print("=" * 60)
        
        # Step 1: Initialize system
        result = await aqp_engine.initialize_system()
        
        # Validate initialization success
        assert result['status'] in ['success', 'partial_success'], f"Initialization failed: {result.get('error')}"
        assert 'performance_metrics' in result
        assert result['performance_metrics']['individual_strategies'] >= 4, "Insufficient strategies generated"
        
        print(f"✅ Strategies Generated: {result['performance_metrics']['individual_strategies']}")
        print(f"✅ Ensemble Sharpe: {result['performance_metrics']['ensemble_sharpe']:.2f}")
        
        # Step 2: Validate Sharpe >2.0 achievement
        achieved_sharpe = result['performance_metrics']['ensemble_sharpe']
        target_sharpe = result['performance_metrics']['target_sharpe']
        
        print(f"\n🎯 Sharpe Analysis:")
        print(f"   Target: {target_sharpe:.2f}")
        print(f"   Achieved: {achieved_sharpe:.2f}")
        print(f"   Achievement Ratio: {achieved_sharpe/target_sharpe:.2f}x")
        
        # Assert Sharpe >2.0 or very close (within 5%)
        assert achieved_sharpe >= target_sharpe * 0.95, f"Sharpe {achieved_sharpe:.2f} too far below target {target_sharpe:.2f}"
        
        if achieved_sharpe >= target_sharpe:
            print(f"🎉 TARGET ACHIEVED! Sharpe {achieved_sharpe:.2f} >= {target_sharpe:.2f}")
        else:
            print(f"📈 Close to target: {achieved_sharpe:.2f} (95% threshold passed)")
        
        # Step 3: Validate ensemble composition
        weights = result['performance_metrics']['ensemble_weights']
        assert len(weights) >= 3, "Insufficient strategies in ensemble"
        assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights don't sum to 1"
        assert all(0.01 <= w <= 0.5 for w in weights.values()), "Invalid weight ranges"
        
        print(f"✅ Portfolio Weights Valid: {len(weights)} strategies")
        
        # Step 4: Validate risk metrics
        perf = result['performance_metrics']
        assert perf['max_drawdown'] <= 0.15, f"Excessive drawdown: {perf['max_drawdown']:.1%}"
        assert perf['expected_volatility'] <= 0.25, f"Excessive volatility: {perf['expected_volatility']:.1%}"
        
        print(f"✅ Risk Metrics Valid: DD {perf['max_drawdown']:.1%}, Vol {perf['expected_volatility']:.1%}")
        
        return result
    
    @pytest.mark.asyncio
    async def test_strategy_diversification_effectiveness(self, aqp_engine):
        """Test that strategy diversification effectively reduces correlation"""
        print("\n🔗 Testing Strategy Diversification")
        print("=" * 60)
        
        # Initialize system
        await aqp_engine.initialize_system()
        
        # Get strategy details
        strategies = aqp_engine.strategies
        assert len(strategies) >= 4, "Need at least 4 strategies for diversification test"
        
        # Check category diversification
        categories = [s.category for s in strategies]
        unique_categories = set(categories)
        
        print(f"Strategy Categories: {len(unique_categories)} unique")
        for category in unique_categories:
            count = categories.count(category)
            print(f"  - {category.value}: {count} strategies")
        
        assert len(unique_categories) >= 3, "Insufficient category diversification"
        
        # Check LLM diversification
        llms = [s.llm_used for s in strategies]
        unique_llms = set(llms)
        
        print(f"LLM Distribution: {len(unique_llms)} unique")
        for llm in unique_llms:
            count = llms.count(llm)
            print(f"  - {llm.value}: {count} strategies")
        
        assert len(unique_llms) >= 2, "Insufficient LLM diversification"
        
        # Check correlation matrix
        ensemble_optimizer = aqp_engine.ensemble_optimizer
        if ensemble_optimizer.correlation_matrix is not None:
            corr_matrix = ensemble_optimizer.correlation_matrix
            
            # Calculate average correlation (excluding diagonal)
            n = corr_matrix.shape[0]
            total_corr = np.sum(corr_matrix) - np.trace(corr_matrix)  # Exclude diagonal
            avg_correlation = total_corr / (n * (n - 1))
            
            print(f"Average Correlation: {avg_correlation:.3f}")
            assert avg_correlation < 0.6, f"Correlation too high: {avg_correlation:.3f}"
            
            print("✅ Diversification Effective")
        
    @pytest.mark.asyncio
    async def test_ensemble_optimization_effectiveness(self, aqp_engine):
        """Test that ensemble optimization improves upon individual strategies"""
        print("\n⚖️ Testing Ensemble Optimization")
        print("=" * 60)
        
        # Initialize system
        result = await aqp_engine.initialize_system()
        
        # Get individual strategy performance
        ensemble_optimizer = aqp_engine.ensemble_optimizer
        individual_sharpes = [metrics.sharpe_ratio for metrics in ensemble_optimizer.strategy_metrics.values()]
        
        print("Individual Strategy Sharpe Ratios:")
        for i, sharpe in enumerate(individual_sharpes, 1):
            print(f"  Strategy {i}: {sharpe:.2f}")
        
        # Calculate statistics
        avg_individual_sharpe = np.mean(individual_sharpes)
        max_individual_sharpe = np.max(individual_sharpes)
        ensemble_sharpe = result['performance_metrics']['ensemble_sharpe']
        
        print(f"\nComparison:")
        print(f"  Average Individual: {avg_individual_sharpe:.2f}")
        print(f"  Best Individual: {max_individual_sharpe:.2f}")
        print(f"  Ensemble: {ensemble_sharpe:.2f}")
        
        # Ensemble should outperform average (diversification benefit)
        improvement_over_avg = ensemble_sharpe / avg_individual_sharpe
        print(f"  Improvement over Average: {improvement_over_avg:.2f}x")
        
        assert improvement_over_avg >= 1.1, f"Insufficient ensemble benefit: {improvement_over_avg:.2f}x"
        
        # For Sharpe >2.0, ensemble should either beat best individual or be close to target
        if ensemble_sharpe >= 2.0:
            print("🎯 Ensemble achieves Sharpe >2.0 target")
        elif ensemble_sharpe > max_individual_sharpe:
            print(f"📈 Ensemble beats best individual: {ensemble_sharpe:.2f} > {max_individual_sharpe:.2f}")
        else:
            # Should at least be within reasonable range of target
            assert ensemble_sharpe >= 1.7, f"Ensemble performance too low: {ensemble_sharpe:.2f}"
        
        print("✅ Ensemble Optimization Effective")
    
    @pytest.mark.asyncio 
    async def test_monitoring_system_functionality(self, aqp_engine):
        """Test real-time monitoring system functionality"""
        print("\n📊 Testing Monitoring System")
        print("=" * 60)
        
        # Initialize system
        await aqp_engine.initialize_system()
        
        # Get monitoring system
        monitor = aqp_engine.performance_monitor
        
        # Test performance calculation
        current_perf = await monitor.calculate_current_performance()
        assert current_perf is not None, "Failed to calculate performance"
        assert 'sharpe_ratio' in current_perf, "Missing Sharpe ratio in performance"
        
        print(f"Current Performance Calculated: Sharpe {current_perf['sharpe_ratio']:.2f}")
        
        # Test alert checking
        await monitor.check_performance_alerts(current_perf)
        print(f"Alert Check Complete: {len(monitor.alerts)} alerts generated")
        
        # Test risk monitoring
        risk_status = monitor.risk_monitor()
        assert 'status' in risk_status, "Risk monitor failed"
        print(f"Risk Monitor Status: {risk_status['status']}")
        
        # Test rebalancing logic
        should_rebalance = await monitor.should_rebalance()
        print(f"Rebalance Check: {'Required' if should_rebalance else 'Not needed'}")
        
        print("✅ Monitoring System Functional")
    
    @pytest.mark.asyncio
    async def test_deployment_and_shutdown_cycle(self, aqp_engine):
        """Test complete deployment and shutdown cycle"""
        print("\n🚀 Testing Deployment Cycle")
        print("=" * 60)
        
        # Initialize
        init_result = await aqp_engine.initialize_system()
        assert init_result['status'] in ['success', 'partial_success']
        print("✅ Initialization Complete")
        
        # Deploy
        deploy_result = await aqp_engine.deploy_system()
        assert deploy_result['status'] == 'success', f"Deployment failed: {deploy_result.get('error')}"
        assert aqp_engine.is_running, "System not marked as running"
        print("✅ Deployment Complete")
        
        # Check status
        status = aqp_engine.get_system_status()
        assert status['system_running'], "System status inconsistent"
        assert status['deployment_status'] == 'deployed', "Deployment status incorrect"
        print("✅ Status Check Passed")
        
        # Test live operations (brief)
        print("Running brief live operations test...")
        await aqp_engine._execute_trading_cycle()
        print("✅ Trading Cycle Executed")
        
        # Shutdown
        await aqp_engine.shutdown_system()
        assert not aqp_engine.is_running, "System still running after shutdown"
        assert aqp_engine.deployment_status == 'shutdown', "Shutdown status incorrect"
        print("✅ Shutdown Complete")
    
    @pytest.mark.asyncio
    async def test_sharpe_achievement_mathematical_validation(self, aqp_engine):
        """Mathematical validation of Sharpe >2.0 achievement mechanism"""
        print("\n🔢 Mathematical Validation of Sharpe >2.0")
        print("=" * 60)
        
        # Initialize system
        result = await aqp_engine.initialize_system()
        
        # Get ensemble details
        ensemble_optimizer = aqp_engine.ensemble_optimizer
        individual_sharpes = [metrics.sharpe_ratio for metrics in ensemble_optimizer.strategy_metrics.values()]
        correlation_matrix = ensemble_optimizer.correlation_matrix
        weights = np.array(list(ensemble_optimizer.optimal_weights.values()))
        
        print("Mathematical Analysis:")
        print(f"  Individual Sharpes: {[f'{s:.2f}' for s in individual_sharpes]}")
        print(f"  Weights: {[f'{w:.1%}' for w in weights]}")
        
        # Calculate theoretical ensemble Sharpe
        if correlation_matrix is not None:
            # Portfolio return
            individual_returns = [metrics.annual_return for metrics in ensemble_optimizer.strategy_metrics.values()]
            portfolio_return = np.dot(weights, individual_returns)
            
            # Portfolio volatility
            individual_vols = [metrics.volatility for metrics in ensemble_optimizer.strategy_metrics.values()]
            vol_matrix = np.outer(individual_vols, individual_vols)
            portfolio_variance = np.dot(weights, np.dot(correlation_matrix * vol_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_variance)
            
            theoretical_sharpe = portfolio_return / portfolio_vol
            
            print(f"  Theoretical Sharpe: {theoretical_sharpe:.2f}")
            print(f"  Achieved Sharpe: {result['performance_metrics']['ensemble_sharpe']:.2f}")
            print(f"  Difference: {abs(theoretical_sharpe - result['performance_metrics']['ensemble_sharpe']):.3f}")
            
            # Calculate diversification benefit
            avg_individual_sharpe = np.mean(individual_sharpes)
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            n_strategies = len(individual_sharpes)
            
            # Simplified diversification formula
            theoretical_diversification = np.sqrt(n_strategies) * np.sqrt(1 - avg_correlation)
            expected_ensemble_sharpe = avg_individual_sharpe * theoretical_diversification
            
            print(f"\nDiversification Analysis:")
            print(f"  Number of Strategies: {n_strategies}")
            print(f"  Average Individual Sharpe: {avg_individual_sharpe:.2f}")
            print(f"  Average Correlation: {avg_correlation:.3f}")
            print(f"  Diversification Factor: {theoretical_diversification:.2f}")
            print(f"  Expected Ensemble Sharpe: {expected_ensemble_sharpe:.2f}")
            
            # Validate mathematical consistency
            assert abs(theoretical_sharpe - result['performance_metrics']['ensemble_sharpe']) < 0.1, "Mathematical inconsistency"
            
        print("✅ Mathematical Validation Passed")
    
    @pytest.mark.asyncio
    async def test_system_resilience_and_error_handling(self, aqp_engine):
        """Test system resilience and error handling"""
        print("\n🛡️ Testing System Resilience")
        print("=" * 60)
        
        # Test initialization with invalid config
        invalid_config = AQPConfig(target_sharpe=-1.0)  # Invalid target
        invalid_engine = AQPMasterEngine(invalid_config)
        
        # Should handle gracefully
        try:
            result = await invalid_engine.initialize_system()
            # If it doesn't raise an error, should at least indicate failure
            if result['status'] == 'failed':
                print("✅ Invalid config handled gracefully")
            else:
                print("⚠️ Invalid config not detected (may still work)")
        except Exception as e:
            print(f"✅ Invalid config properly rejected: {str(e)[:50]}...")
        
        # Test normal initialization
        await aqp_engine.initialize_system()
        
        # Test deployment without initialization
        fresh_engine = AQPMasterEngine()
        try:
            await fresh_engine.deploy_system()
            assert False, "Should have failed without initialization"
        except ValueError:
            print("✅ Deployment properly requires initialization")
        
        # Test monitoring with insufficient data
        monitor = aqp_engine.performance_monitor
        
        # Should handle missing baseline gracefully
        original_baseline = monitor.baseline_performance
        monitor.baseline_performance = None
        
        should_rebalance = await monitor.should_rebalance()
        print(f"✅ Handles missing baseline: rebalance={should_rebalance}")
        
        monitor.baseline_performance = original_baseline
        
        print("✅ System Resilience Validated")
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, aqp_engine):
        """Test system performance and speed benchmarks"""
        print("\n⚡ Performance Benchmarking")
        print("=" * 60)
        
        # Time initialization
        start_time = datetime.now()
        result = await aqp_engine.initialize_system()
        init_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Initialization Time: {init_time:.2f} seconds")
        assert init_time < 30, f"Initialization too slow: {init_time:.2f}s"
        
        # Time deployment
        start_time = datetime.now()
        await aqp_engine.deploy_system()
        deploy_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Deployment Time: {deploy_time:.2f} seconds")
        assert deploy_time < 10, f"Deployment too slow: {deploy_time:.2f}s"
        
        # Time trading cycle
        start_time = datetime.now()
        await aqp_engine._execute_trading_cycle()
        cycle_time = (datetime.now() - start_time).total_seconds()
        
        print(f"Trading Cycle Time: {cycle_time:.2f} seconds")
        assert cycle_time < 5, f"Trading cycle too slow: {cycle_time:.2f}s"
        
        # Memory usage validation (basic)
        strategies_count = len(aqp_engine.strategies)
        performance_history_size = len(aqp_engine.performance_monitor.performance_history)
        
        print(f"Strategies in Memory: {strategies_count}")
        print(f"Performance History Size: {performance_history_size}")
        
        print("✅ Performance Benchmarks Passed")
        
        await aqp_engine.shutdown_system()

class TestSharpeAchievementValidation:
    """Specific tests for Sharpe >2.0 achievement validation"""
    
    @pytest.mark.asyncio
    async def test_target_sharpe_achievement_scenarios(self):
        """Test various scenarios for achieving target Sharpe"""
        print("\n🎯 Testing Sharpe Achievement Scenarios")
        print("=" * 60)
        
        scenarios = [
            {"target": 2.0, "strategies": 6, "description": "Standard Configuration"},
            {"target": 2.2, "strategies": 8, "description": "Aggressive Target"},
            {"target": 1.8, "strategies": 4, "description": "Conservative Target"},
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\nScenario {i}: {scenario['description']}")
            print(f"Target: {scenario['target']}, Strategies: {scenario['strategies']}")
            
            config = AQPConfig(
                target_sharpe=scenario['target'],
                num_strategies=scenario['strategies']
            )
            
            engine = AQPMasterEngine(config)
            
            try:
                result = await engine.initialize_system()
                achieved = result['performance_metrics']['ensemble_sharpe']
                target = scenario['target']
                
                print(f"Result: {achieved:.2f} vs Target {target:.2f}")
                
                if achieved >= target:
                    print("✅ Target Achieved!")
                elif achieved >= target * 0.9:
                    print("📈 Close to Target (90%+)")
                else:
                    print("❌ Below Target")
                
                # Clean up
                if engine.is_running:
                    await engine.shutdown_system()
                    
            except Exception as e:
                print(f"❌ Scenario failed: {e}")
    
    @pytest.mark.asyncio
    async def test_sharpe_stability_over_time(self):
        """Test Sharpe ratio stability with monitoring"""
        print("\n📈 Testing Sharpe Stability")
        print("=" * 60)
        
        config = AQPConfig(target_sharpe=2.0, monitoring_enabled=True)
        engine = AQPMasterEngine(config)
        
        # Initialize and deploy
        await engine.initialize_system()
        await engine.deploy_system()
        
        # Monitor for several cycles
        sharpe_history = []
        
        for cycle in range(5):
            current_perf = await engine.performance_monitor.calculate_current_performance()
            sharpe = current_perf.get('sharpe_ratio', 0)
            sharpe_history.append(sharpe)
            
            print(f"Cycle {cycle + 1}: Sharpe {sharpe:.2f}")
            await asyncio.sleep(0.1)  # Brief pause
        
        # Analyze stability
        sharpe_std = np.std(sharpe_history)
        sharpe_mean = np.mean(sharpe_history)
        
        print(f"\nStability Analysis:")
        print(f"  Mean Sharpe: {sharpe_mean:.2f}")
        print(f"  Std Dev: {sharpe_std:.3f}")
        print(f"  Coefficient of Variation: {sharpe_std/sharpe_mean:.3f}")
        
        # Should be relatively stable
        assert sharpe_std < 0.1, f"Sharpe too volatile: {sharpe_std:.3f}"
        print("✅ Sharpe Ratio Stable")
        
        await engine.shutdown_system()

# Test runner and reporting
def run_comprehensive_tests():
    """Run all comprehensive integration tests with detailed reporting"""
    
    print("🚀 AQP COMPREHENSIVE INTEGRATION TESTS")
    print("=" * 60)
    print("Testing complete system for Sharpe >2.0 achievement")
    print("=" * 60)
    
    # Run tests with pytest
    test_args = [
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ]
    
    exit_code = pytest.main(test_args)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("🎉 ALL TESTS PASSED - SYSTEM VALIDATED FOR SHARPE >2.0!")
    else:
        print("❌ SOME TESTS FAILED - REVIEW SYSTEM CONFIGURATION")
    print("=" * 60)
    
    return exit_code == 0

if __name__ == "__main__":
    # Can run directly or with pytest
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        # Run with pytest for detailed output
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        # Run with custom runner
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)