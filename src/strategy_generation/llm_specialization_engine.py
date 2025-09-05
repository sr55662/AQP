#  src/strategy_generation/llm_specialization_engine.py
# LLM Strategy Specialization Engine for Maximum Diversification

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime

class StrategyCategory(Enum):
    """Strategy categories optimized for different LLMs"""
    SYSTEMATIC_RISK_MANAGED = "systematic_risk_managed"  # Claude specialty
    BEHAVIORAL_SENTIMENT = "behavioral_sentiment"        # GPT-4 specialty  
    MATHEMATICAL_ARBITRAGE = "mathematical_arbitrage"    # Gemini specialty
    CONTRARIAN_TAIL_RISK = "contrarian_tail_risk"       # Grok specialty
    MOMENTUM_TREND = "momentum_trend"                    # Shared
    MEAN_REVERSION = "mean_reversion"                    # Shared

class LLMModel(Enum):
    """Available LLM models with specializations"""
    CLAUDE = "claude-sonnet-4-20250514"
    GPT4 = "gpt-4-turbo"
    GEMINI = "gemini-pro"
    GROK = "grok-1"

@dataclass
class StrategyRequest:
    """Strategy generation request with LLM routing"""
    category: StrategyCategory
    target_sharpe: float
    description: str
    market_regime: str
    symbols: List[str]
    timeframe: str
    risk_tolerance: str
    preferred_llm: Optional[LLMModel] = None

@dataclass
class GeneratedStrategy:
    """Generated strategy with metadata"""
    name: str
    code: str
    description: str
    llm_used: LLMModel
    category: StrategyCategory
    expected_sharpe: float
    estimated_correlation: Dict[str, float]
    risk_characteristics: Dict[str, Any]
    backtest_ready: bool

class LLMSpecializationEngine:
    """
    LLM Strategy Specialization Engine
    
    Routes strategy generation to optimal LLM based on strategy type
    and uses specialized prompts to maximize diversification
    """
    
    def __init__(self):
        self.llm_specializations = self._initialize_specializations()
        self.strategy_templates = self._initialize_strategy_templates()
        self.generated_strategies: List[GeneratedStrategy] = []
        
    def _initialize_specializations(self) -> Dict[LLMModel, Dict]:
        """Initialize LLM specializations and capabilities"""
        return {
            LLMModel.CLAUDE: {
                "primary_categories": [
                    StrategyCategory.SYSTEMATIC_RISK_MANAGED,
                    StrategyCategory.MEAN_REVERSION
                ],
                "strengths": [
                    "Risk management implementation",
                    "Systematic approach",
                    "Position sizing algorithms",
                    "Drawdown control",
                    "Statistical analysis"
                ],
                "personality": "analytical_conservative",
                "target_sharpe_range": (1.2, 1.8),
                "preferred_timeframes": ["daily", "weekly"],
                "risk_profile": "moderate"
            },
            LLMModel.GPT4: {
                "primary_categories": [
                    StrategyCategory.BEHAVIORAL_SENTIMENT,
                    StrategyCategory.MOMENTUM_TREND
                ],
                "strengths": [
                    "Market sentiment analysis",
                    "Behavioral pattern recognition",
                    "Creative strategy development",
                    "News and event processing",
                    "Market psychology insights"
                ],
                "personality": "creative_intuitive",
                "target_sharpe_range": (1.0, 1.6),
                "preferred_timeframes": ["intraday", "daily"],
                "risk_profile": "moderate_aggressive"
            },
            LLMModel.GEMINI: {
                "primary_categories": [
                    StrategyCategory.MATHEMATICAL_ARBITRAGE,
                    StrategyCategory.SYSTEMATIC_RISK_MANAGED
                ],
                "strengths": [
                    "Mathematical modeling",
                    "Statistical arbitrage",
                    "Complex calculations",
                    "Quantitative analysis",
                    "Optimization algorithms"
                ],
                "personality": "mathematical_precise",
                "target_sharpe_range": (1.3, 2.0),
                "preferred_timeframes": ["intraday", "daily"],
                "risk_profile": "calculated_aggressive"
            },
            LLMModel.GROK: {
                "primary_categories": [
                    StrategyCategory.CONTRARIAN_TAIL_RISK,
                    StrategyCategory.BEHAVIORAL_SENTIMENT
                ],
                "strengths": [
                    "Contrarian analysis",
                    "Tail risk identification",
                    "Alternative perspectives",
                    "Edge case detection",
                    "Unconventional patterns"
                ],
                "personality": "contrarian_edgy",
                "target_sharpe_range": (0.8, 1.5),
                "preferred_timeframes": ["daily", "weekly"],
                "risk_profile": "high_conviction"
            }
        }
        
    def _initialize_strategy_templates(self) -> Dict[StrategyCategory, Dict[LLMModel, str]]:
        """Initialize specialized prompts for each LLM-category combination"""
        return {
            StrategyCategory.SYSTEMATIC_RISK_MANAGED: {
                LLMModel.CLAUDE: """
You are a systematic risk management expert. Create a trading strategy that prioritizes:

1. RISK-FIRST APPROACH: Start with risk management, then build returns
2. SYSTEMATIC IMPLEMENTATION: Rule-based, backtestable, no discretion
3. POSITION SIZING: Dynamic position sizing based on volatility and correlation
4. DRAWDOWN CONTROL: Maximum 8% drawdown with recovery mechanisms

Strategy Requirements:
- Target Sharpe: {target_sharpe}
- Symbols: {symbols}
- Market Regime: {market_regime}
- Description: {description}

Implement comprehensive risk controls:
- Stop losses with trailing mechanisms
- Position size based on Kelly criterion or risk parity
- Correlation-based exposure limits
- Volatility targeting (10-15% annual)

Return complete Python code with:
- Strategy class with clear entry/exit rules
- Risk management system
- Position sizing algorithm
- Performance metrics calculation
- Backtesting framework integration

Focus on consistency and risk-adjusted returns over absolute returns.
""",
                LLMModel.GEMINI: """
As a quantitative risk modeling expert, design a mathematically rigorous strategy:

MATHEMATICAL FRAMEWORK:
- Optimize risk-adjusted returns using modern portfolio theory
- Implement dynamic hedging based on Greeks or factor exposures
- Use statistical measures for entry/exit timing
- Apply advanced position sizing (Black-Litterman, risk budgeting)

Target: {target_sharpe} Sharpe ratio through {description}
Universe: {symbols} | Regime: {market_regime}

Required mathematical components:
1. Covariance matrix estimation with shrinkage
2. Expected returns using Bayesian updating
3. Optimal portfolio weights via quadratic programming
4. Risk attribution and decomposition
5. Monte Carlo simulation for stress testing

Provide Python implementation with:
- Mathematical derivations in comments
- Numerical optimization routines
- Statistical validation methods
- Performance attribution framework
"""
            },
            
            StrategyCategory.BEHAVIORAL_SENTIMENT: {
                LLMModel.GPT4: """
You are a behavioral finance and market sentiment expert. Design a strategy that exploits:

HUMAN BEHAVIORAL BIASES:
- Overreaction and underreaction patterns
- Herding behavior and contrarian opportunities  
- Anchoring bias in price discovery
- Momentum vs mean reversion in different timeframes

Market Sentiment Integration:
- News sentiment analysis and event processing
- Social media sentiment (when available)
- VIX and fear/greed indicators
- Earnings surprise and guidance reactions

Strategy Specification:
- Target: {target_sharpe} Sharpe via {description}
- Universe: {symbols}
- Market Context: {market_regime}

Implementation should include:
1. Sentiment scoring algorithms
2. Behavioral signal generation
3. Market microstructure considerations
4. Event-driven alpha capture
5. Regime-dependent strategy switching

Create Python code that:
- Processes alternative data sources
- Implements behavioral scoring systems
- Combines multiple sentiment signals
- Adapts to changing market conditions
- Includes realistic transaction costs
""",
                LLMModel.GROK: """
Think like a contrarian market participant who profits from crowd psychology mistakes.

CONTRARIAN PHILOSOPHY:
- When everyone is greedy, be fearful (and profit from it)
- Identify when "obvious" trades are too crowded
- Find value in despised assets and overlooked opportunities
- Exploit systematic biases in institutional decision-making

Develop an unconventional strategy for {description}:
Target: {target_sharpe} | Assets: {symbols} | Environment: {market_regime}

Your edge comes from:
1. Going against consensus when it's wrong
2. Identifying bubble formations and burst timing
3. Finding alpha in neglected corners of the market
4. Exploiting forced selling/buying from institutional constraints
5. Capitalizing on emotional extreme points

Build Python strategy that:
- Identifies overcrowded trades
- Measures sentiment extremes quantitatively
- Times contrarian entries with precision
- Manages the psychological difficulty of contrarian positions
- Incorporates "pain trade" identification

Be bold but systematic. Contrarian doesn't mean reckless.
"""
            },
            
            StrategyCategory.MATHEMATICAL_ARBITRAGE: {
                LLMModel.GEMINI: """
Design a mathematical arbitrage strategy exploiting statistical relationships:

STATISTICAL ARBITRAGE FRAMEWORK:
- Co-integration analysis for pairs/basket trading
- Mean reversion in spreads with statistical significance
- Factor model arbitrage (exposure to uncompensated risk)
- Cross-asset arbitrage opportunities

Mathematical Requirements:
- Johansen co-integration test implementation
- Ornstein-Uhlenbeck mean reversion modeling
- Half-life estimation for mean reversion
- Kalman filtering for dynamic relationships
- PCA for factor decomposition

Strategy Details:
Target: {target_sharpe} via {description}
Universe: {symbols} | Regime: {market_regime}

Implementation must include:
1. Statistical testing framework (ADF, Johansen, etc.)
2. Dynamic hedge ratio calculation
3. Entry/exit based on statistical significance
4. Risk management via statistical measures
5. Walk-forward parameter optimization

Deliver Python code with:
- Mathematical model implementation
- Statistical testing procedures
- Signal generation algorithms
- Risk management framework
- Performance measurement system

Focus on statistical rigor and mathematical precision.
""",
                LLMModel.CLAUDE: """
Create a systematic statistical arbitrage strategy with robust risk controls:

SYSTEMATIC ARBITRAGE APPROACH:
- Quantify relationships using regression analysis
- Implement mean reversion with confidence intervals
- Use correlation analysis for pair selection
- Apply systematic entry/exit rules

For {description} targeting {target_sharpe} Sharpe:
Assets: {symbols} | Market State: {market_regime}

Risk-managed implementation:
1. Statistical relationship validation
2. Position sizing based on signal strength
3. Stop-loss based on statistical significance
4. Portfolio-level risk budgeting
5. Correlation monitoring and adjustment

Python strategy requirements:
- Clear statistical methodology
- Systematic signal generation
- Comprehensive risk management
- Position sizing algorithms
- Performance tracking system

Emphasize robustness over complexity.
"""
            },
            
            StrategyCategory.CONTRARIAN_TAIL_RISK: {
                LLMModel.GROK: """
Design a tail-risk strategy that profits when others panic:

TAIL RISK EXPLOITATION:
- Identify when markets overshoot on fear
- Profit from forced liquidations and margin calls
- Capitalize on volatility spikes and dislocations
- Find opportunities in "impossible" scenarios

Strategy for {description} | Target: {target_sharpe}
Universe: {symbols} | Current Regime: {market_regime}

Your contrarian edge:
1. VIX spike monetization strategies
2. Credit spread blow-out opportunities  
3. Currency crisis alpha generation
4. Earnings/event shock capture
5. Correlation breakdown exploitation

Build Python implementation:
- Tail risk identification algorithms
- Volatility regime detection
- Crisis alpha capture mechanisms
- Risk management for extreme scenarios
- Performance measurement in tail events

Remember: Others' fear is your opportunity, but manage your own risk.
""",
                LLMModel.CLAUDE: """
Develop a systematic tail-risk management strategy:

SYSTEMATIC TAIL RISK APPROACH:
- Quantify tail risk using statistical measures
- Implement protective strategies during crisis
- Systematic identification of tail events
- Risk-managed approach to crisis alpha

Strategy specification:
Target: {target_sharpe} through {description}
Assets: {symbols} | Market Environment: {market_regime}

Systematic implementation:
1. VaR and CVaR monitoring systems
2. Tail risk indicators and early warnings
3. Systematic hedging mechanisms
4. Crisis alpha capture with size limits
5. Portfolio protection strategies

Python code should include:
- Tail risk measurement framework
- Systematic hedge ratio calculation
- Position sizing for tail events
- Risk budgeting for extreme scenarios
- Performance attribution system

Focus on systematic, risk-managed tail risk strategies.
"""
            }
        }
        
    def route_strategy_request(self, request: StrategyRequest) -> LLMModel:
        """Route strategy request to optimal LLM"""
        if request.preferred_llm:
            return request.preferred_llm
            
        # Find LLMs that specialize in this category
        specialized_llms = []
        for llm, specs in self.llm_specializations.items():
            if request.category in specs["primary_categories"]:
                specialized_llms.append(llm)
                
        if not specialized_llms:
            # Fallback to general assignment
            category_llm_map = {
                StrategyCategory.SYSTEMATIC_RISK_MANAGED: LLMModel.CLAUDE,
                StrategyCategory.BEHAVIORAL_SENTIMENT: LLMModel.GPT4,
                StrategyCategory.MATHEMATICAL_ARBITRAGE: LLMModel.GEMINI,
                StrategyCategory.CONTRARIAN_TAIL_RISK: LLMModel.GROK,
                StrategyCategory.MOMENTUM_TREND: LLMModel.GPT4,
                StrategyCategory.MEAN_REVERSION: LLMModel.CLAUDE
            }
            return category_llm_map.get(request.category, LLMModel.CLAUDE)
            
        # If multiple specialized LLMs, choose based on target Sharpe and risk tolerance
        if len(specialized_llms) == 1:
            return specialized_llms[0]
            
        # Multi-criteria selection
        best_llm = specialized_llms[0]
        best_score = 0
        
        for llm in specialized_llms:
            specs = self.llm_specializations[llm]
            score = 0
            
            # Sharpe range compatibility
            min_sharpe, max_sharpe = specs["target_sharpe_range"]
            if min_sharpe <= request.target_sharpe <= max_sharpe:
                score += 3
            elif abs(request.target_sharpe - (min_sharpe + max_sharpe) / 2) < 0.5:
                score += 1
                
            # Risk tolerance compatibility
            risk_compatibility = {
                ("low", "moderate"): 2,
                ("medium", "moderate_aggressive"): 2,
                ("high", "calculated_aggressive"): 2,
                ("high", "high_conviction"): 3
            }
            score += risk_compatibility.get((request.risk_tolerance, specs["risk_profile"]), 0)
            
            if score > best_score:
                best_score = score
                best_llm = llm
                
        return best_llm
        
    async def generate_strategy(self, request: StrategyRequest) -> GeneratedStrategy:
        """Generate strategy using specialized LLM"""
        selected_llm = self.route_strategy_request(request)
        
        # Get specialized prompt
        prompt_template = self.strategy_templates.get(request.category, {}).get(selected_llm)
        if not prompt_template:
            raise ValueError(f"No template found for {request.category} + {selected_llm}")
            
        # Format prompt with request details
        formatted_prompt = prompt_template.format(
            target_sharpe=request.target_sharpe,
            symbols=", ".join(request.symbols),
            market_regime=request.market_regime,
            description=request.description
        )
        
        # Call LLM API (placeholder - integrate with your LLM calling system)
        strategy_code = await self._call_llm_api(selected_llm, formatted_prompt)
        
        # Parse and validate generated strategy
        generated_strategy = GeneratedStrategy(
            name=f"{request.category.value}_{selected_llm.value}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            code=strategy_code,
            description=request.description,
            llm_used=selected_llm,
            category=request.category,
            expected_sharpe=request.target_sharpe,
            estimated_correlation=self._estimate_correlation(request.category, selected_llm),
            risk_characteristics=self._extract_risk_characteristics(request, selected_llm),
            backtest_ready=True
        )
        
        self.generated_strategies.append(generated_strategy)
        return generated_strategy
        
    async def _call_llm_api(self, llm: LLMModel, prompt: str) -> str:
        """Call LLM API - integrate with your existing system"""
        # Placeholder - replace with your actual LLM API calls
        # This should integrate with your existing AWS orchestrator
        
        api_endpoints = {
            LLMModel.CLAUDE: "https://api.anthropic.com/v1/messages",
            LLMModel.GPT4: "https://api.openai.com/v1/chat/completions",
            LLMModel.GEMINI: "https://api.google.com/gemini/v1/chat",
            LLMModel.GROK: "https://api.x.ai/v1/chat"
        }
        
        # Simulate API call for now
        await asyncio.sleep(1)  # Simulate network delay
        
        return f"""
# Generated by {llm.value}
# Strategy implementing {prompt[:100]}...

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class GeneratedStrategy:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.position = {{symbol: 0 for symbol in symbols}}
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        # {llm.value} specialized strategy implementation
        signals = {{}}
        
        # [Strategy-specific logic would go here]
        # This is a placeholder - actual implementation would be generated by LLM
        
        return signals
        
    def calculate_position_size(self, signal: float, volatility: float) -> float:
        # Risk-managed position sizing
        target_vol = 0.15  # 15% target volatility
        return signal * (target_vol / volatility) if volatility > 0 else 0
        
    def risk_management(self, current_positions: Dict, market_data: pd.DataFrame) -> Dict:
        # Strategy-specific risk controls
        return current_positions
"""
        
    def _estimate_correlation(self, category: StrategyCategory, llm: LLMModel) -> Dict[str, float]:
        """Estimate correlation with existing strategies"""
        correlation_matrix = {
            StrategyCategory.SYSTEMATIC_RISK_MANAGED: 0.3,
            StrategyCategory.BEHAVIORAL_SENTIMENT: 0.2,
            StrategyCategory.MATHEMATICAL_ARBITRAGE: 0.1,
            StrategyCategory.CONTRARIAN_TAIL_RISK: -0.1,
            StrategyCategory.MOMENTUM_TREND: 0.4,
            StrategyCategory.MEAN_REVERSION: 0.2
        }
        
        # Adjust based on LLM - different LLMs should produce less correlated strategies
        llm_adjustment = {
            LLMModel.CLAUDE: 0.0,
            LLMModel.GPT4: -0.1,
            LLMModel.GEMINI: -0.05,
            LLMModel.GROK: -0.15
        }
        
        base_correlation = correlation_matrix.get(category, 0.3)
        adjusted_correlation = max(-0.3, min(0.7, base_correlation + llm_adjustment.get(llm, 0)))
        
        return {"estimated_avg_correlation": adjusted_correlation}
        
    def _extract_risk_characteristics(self, request: StrategyRequest, llm: LLMModel) -> Dict[str, Any]:
        """Extract risk characteristics based on LLM specialization"""
        llm_specs = self.llm_specializations[llm]
        
        return {
            "target_volatility": 0.15,
            "max_drawdown_target": 0.10,
            "expected_sharpe_range": llm_specs["target_sharpe_range"],
            "risk_profile": llm_specs["risk_profile"],
            "preferred_timeframes": llm_specs["preferred_timeframes"],
            "strategy_type": request.category.value
        }
        
    async def generate_diversified_ensemble(self, target_ensemble_sharpe: float = 2.0) -> List[GeneratedStrategy]:
        """Generate a diversified ensemble of strategies targeting Sharpe >2.0"""
        
        # Strategy mix for maximum diversification
        strategy_requests = [
            StrategyRequest(
                category=StrategyCategory.SYSTEMATIC_RISK_MANAGED,
                target_sharpe=1.6,
                description="Volatility-adjusted momentum with risk parity",
                market_regime="normal",
                symbols=["SPY", "QQQ", "IWM"],
                timeframe="daily",
                risk_tolerance="medium",
                preferred_llm=LLMModel.CLAUDE
            ),
            StrategyRequest(
                category=StrategyCategory.BEHAVIORAL_SENTIMENT,
                target_sharpe=1.4,
                description="Earnings surprise and guidance sentiment",
                market_regime="normal", 
                symbols=["AAPL", "GOOGL", "MSFT", "TSLA"],
                timeframe="daily",
                risk_tolerance="medium",
                preferred_llm=LLMModel.GPT4
            ),
            StrategyRequest(
                category=StrategyCategory.MATHEMATICAL_ARBITRAGE,
                target_sharpe=1.7,
                description="ETF-underlying statistical arbitrage",
                market_regime="normal",
                symbols=["SPY", "XLF", "XLK", "XLE"],
                timeframe="intraday",
                risk_tolerance="high",
                preferred_llm=LLMModel.GEMINI
            ),
            StrategyRequest(
                category=StrategyCategory.CONTRARIAN_TAIL_RISK,
                target_sharpe=1.2,
                description="VIX spike and crisis alpha capture",
                market_regime="normal",
                symbols=["VXX", "TLT", "GLD", "SPY"],
                timeframe="daily",
                risk_tolerance="high",
                preferred_llm=LLMModel.GROK
            ),
            StrategyRequest(
                category=StrategyCategory.MEAN_REVERSION,
                target_sharpe=1.5,
                description="Cross-asset mean reversion with regime detection",
                market_regime="normal",
                symbols=["DXY", "TLT", "SPY", "GLD"],
                timeframe="weekly",
                risk_tolerance="low",
                preferred_llm=LLMModel.CLAUDE
            ),
            StrategyRequest(
                category=StrategyCategory.MOMENTUM_TREND,
                target_sharpe=1.3,
                description="Multi-timeframe momentum with sentiment overlay",
                market_regime="normal",
                symbols=["QQQ", "XLK", "ARKK", "SMH"],
                timeframe="daily",
                risk_tolerance="medium",
                preferred_llm=LLMModel.GPT4
            )
        ]
        
        # Generate all strategies concurrently
        strategies = await asyncio.gather(*[
            self.generate_strategy(request) for request in strategy_requests
        ])
        
        print(f"Generated {len(strategies)} specialized strategies:")
        for strategy in strategies:
            print(f"  - {strategy.name} | {strategy.llm_used.value} | Expected Sharpe: {strategy.expected_sharpe:.2f}")
            
        return strategies
        
    def analyze_ensemble_diversification(self) -> Dict[str, Any]:
        """Analyze diversification of generated ensemble"""
        if len(self.generated_strategies) < 2:
            return {"error": "Need at least 2 strategies for diversification analysis"}
            
        # Category diversification
        categories = [s.category for s in self.generated_strategies]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        
        # LLM diversification
        llms = [s.llm_used for s in self.generated_strategies]
        llm_counts = {llm: llms.count(llm) for llm in set(llms)}
        
        # Expected correlation analysis
        correlations = [s.estimated_correlation.get("estimated_avg_correlation", 0.3) 
                      for s in self.generated_strategies]
        avg_correlation = np.mean(correlations)
        
        # Expected ensemble Sharpe calculation
        individual_sharpes = [s.expected_sharpe for s in self.generated_strategies]
        avg_individual_sharpe = np.mean(individual_sharpes)
        n_strategies = len(self.generated_strategies)
        
        # Simplified ensemble Sharpe estimate
        diversification_benefit = np.sqrt(n_strategies) * np.sqrt(1 - avg_correlation)
        expected_ensemble_sharpe = avg_individual_sharpe * diversification_benefit
        
        return {
            "total_strategies": n_strategies,
            "category_distribution": category_counts,
            "llm_distribution": llm_counts,
            "avg_individual_sharpe": avg_individual_sharpe,
            "estimated_avg_correlation": avg_correlation,
            "diversification_ratio": diversification_benefit,
            "expected_ensemble_sharpe": expected_ensemble_sharpe,
            "target_achieved": expected_ensemble_sharpe >= 2.0,
            "individual_sharpe_range": (min(individual_sharpes), max(individual_sharpes))
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_specialization_engine():
        engine = LLMSpecializationEngine()
        
        # Generate diversified ensemble
        strategies = await engine.generate_diversified_ensemble(target_ensemble_sharpe=2.0)
        
        # Analyze diversification
        analysis = engine.analyze_ensemble_diversification()
        
        print("\n🎯 ENSEMBLE DIVERSIFICATION ANALYSIS")
        print("=" * 50)
        print(f"Total Strategies: {analysis['total_strategies']}")
        print(f"Average Individual Sharpe: {analysis['avg_individual_sharpe']:.2f}")
        print(f"Expected Ensemble Sharpe: {analysis['expected_ensemble_sharpe']:.2f}")
        print(f"Target Achieved (>2.0): {'✅ YES' if analysis['target_achieved'] else '❌ NO'}")
        print(f"\nCategory Distribution: {analysis['category_distribution']}")
        print(f"LLM Distribution: {analysis['llm_distribution']}")
        print(f"Estimated Correlation: {analysis['estimated_avg_correlation']:.2f}")
        print(f"Diversification Ratio: {analysis['diversification_ratio']:.2f}x")
        
    # Run the test
    asyncio.run(test_specialization_engine()) 