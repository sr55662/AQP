# src/api/aws_orchestrator.py
# AWS Lambda-based API for invoking the complete AQP pipeline
# Any role can invoke: Data → Strategy → Optimization → Backtesting → Report

import json
import boto3
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict

# FastAPI for REST API
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Our AQP components
from data_aggregator.market_data_engine import UniversalDataAggregator, DataRequest
from backtesting.advanced_backtester import AdvancedBacktester, BacktestConfig
from orchestrator.model_router import AgenticQuantPlatform

# AWS services
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ========================================
# REQUEST/RESPONSE MODELS
# ========================================

class StrategyRequest(BaseModel):
    """API request for strategy analysis"""
    
    # Data requirements
    symbols: List[str]
    start_date: str
    end_date: str
    data_sources: Optional[List[str]] = ['yahoo_finance', 'alpha_vantage']
    
    # Strategy parameters
    strategy_description: str
    target_sharpe: float = 2.0
    max_drawdown: float = 0.10
    
    # Backtesting configuration
    initial_capital: float = 100000
    commission: float = 0.001
    benchmark: str = 'SPY'
    
    # Analysis depth
    include_walk_forward: bool = True
    include_monte_carlo: bool = True
    monte_carlo_simulations: int = 1000
    
    # LLM routing preferences
    preferred_models: Optional[List[str]] = None
    
    # AWS settings
    priority: str = 'normal'  # 'low', 'normal', 'high'
    notify_email: Optional[str] = None

class StrategyResponse(BaseModel):
    """API response with complete analysis"""
    
    # Execution metadata
    request_id: str
    status: str
    timestamp: datetime
    execution_time_seconds: float
    
    # Strategy details
    strategy_code: str
    strategy_description: str
    model_ensemble_used: List[str]
    
    # Data summary
    data_quality_score: float
    data_source_used: str
    data_cost: float
    
    # Backtest results
    sharpe_ratio: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    volatility: float
    calmar_ratio: float
    
    # Advanced metrics
    walk_forward_stability: Optional[float]
    monte_carlo_confidence: Optional[Dict[str, float]]
    probability_target_achieved: float
    
    # Risk analysis
    var_95: float
    expected_shortfall: float
    tail_ratio: float
    
    # Visual assets
    equity_curve_url: str
    performance_report_url: str
    detailed_analysis_url: str
    
    # Deployment readiness
    deployment_ready: bool
    estimated_live_performance: Dict[str, float]
    
class JobStatus(BaseModel):
    """Job status tracking"""
    request_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float  # 0.0 to 1.0
    current_step: str
    estimated_completion: Optional[datetime]
    error_message: Optional[str]

# ========================================
# AWS ORCHESTRATOR
# ========================================

class AWSQuantOrchestrator:
    """Main orchestrator for AWS-deployed AQP system"""
    
    def __init__(self, aws_config: Dict[str, Any]):
        self.aws_config = aws_config
        
        # Initialize AWS services
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.sns_client = boto3.client('sns')
        self.lambda_client = boto3.client('lambda')
        
        # Initialize AQP components
        self.data_aggregator = UniversalDataAggregator(aws_config)
        self.backtester = None  # Will be initialized per request
        self.aqp_platform = AgenticQuantPlatform(aws_config.get('aqp_config', 'config.json'))
        
        # DynamoDB table for job tracking
        self.jobs_table = self.dynamodb.Table(aws_config.get('jobs_table', 'aqp-jobs'))
        
        # S3 bucket for results
        self.results_bucket = aws_config.get('results_bucket', 'aqp-results')
        
    async def execute_strategy_pipeline(self, request: StrategyRequest) -> str:
        """Execute the complete data-to-report pipeline"""
        
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Initialize job tracking
        await self._update_job_status(request_id, 'running', 0.0, 'Initializing pipeline')
        
        try:
            # Step 1: Data Aggregation (20% progress)
            logger.info(f"[{request_id}] Starting data aggregation")
            await self._update_job_status(request_id, 'running', 0.1, 'Fetching market data')
            
            data_request = DataRequest(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                data_types=['price', 'volume', 'fundamentals'],
                source_priority=request.data_sources
            )
            
            data_response = await self.data_aggregator.fetch_data(data_request)
            await self._update_job_status(request_id, 'running', 0.2, 'Data aggregation complete')
            
            # Step 2: Strategy Generation (40% progress)
            logger.info(f"[{request_id}] Generating strategy with LLM ensemble")
            await self._update_job_status(request_id, 'running', 0.25, 'Generating trading strategy')
            
            strategy_result = await self.aqp_platform.run_research_cycle(
                query=request.strategy_description,
                target_sharpe=request.target_sharpe,
                symbols=request.symbols,
                preferred_models=request.preferred_models or ['claude', 'grok', 'chatgpt']
            )
            
            await self._update_job_status(request_id, 'running', 0.4, 'Strategy generation complete')
            
            # Step 3: Backtesting (60% progress)
            logger.info(f"[{request_id}] Running comprehensive backtesting")
            await self._update_job_status(request_id, 'running', 0.45, 'Initializing backtesting')
            
            backtest_config = BacktestConfig(
                initial_capital=request.initial_capital,
                commission=request.commission,
                benchmark=request.benchmark
            )
            
            backtester = AdvancedBacktester(backtest_config)
            
            # Main backtest
            backtest_results = backtester.run_backtest(
                strategy=strategy_result['strategy_object'],
                data=data_response.data
            )
            
            await self._update_job_status(request_id, 'running', 0.55, 'Main backtesting complete')
            
            # Step 4: Advanced Analysis (80% progress)
            walk_forward_results = None
            monte_carlo_results = None
            
            if request.include_walk_forward:
                logger.info(f"[{request_id}] Running walk-forward analysis")
                await self._update_job_status(request_id, 'running', 0.6, 'Walk-forward analysis')
                
                walk_forward_results = backtester.walk_forward_analysis(
                    strategy=strategy_result['strategy_object'],
                    data=data_response.data
                )
                
                await self._update_job_status(request_id, 'running', 0.7, 'Walk-forward complete')
            
            if request.include_monte_carlo:
                logger.info(f"[{request_id}] Running Monte Carlo simulation")
                await self._update_job_status(request_id, 'running', 0.75, 'Monte Carlo simulation')
                
                monte_carlo_results = backtester.monte_carlo_analysis(
                    strategy=strategy_result['strategy_object'],
                    data=data_response.data,
                    n_simulations=request.monte_carlo_simulations
                )
                
                await self._update_job_status(request_id, 'running', 0.8, 'Monte Carlo complete')
            
            # Step 5: Report Generation (90% progress)
            logger.info(f"[{request_id}] Generating comprehensive report")
            await self._update_job_status(request_id, 'running', 0.85, 'Generating reports')
            
            report_urls = await self._generate_reports(
                request_id=request_id,
                backtest_results=backtest_results,
                walk_forward_results=walk_forward_results,
                monte_carlo_results=monte_carlo_results,
                strategy_result=strategy_result,
                data_response=data_response
            )
            
            await self._update_job_status(request_id, 'running', 0.95, 'Finalizing results')
            
            # Step 6: Prepare Response
            execution_time = (datetime.now() - start_time).total_seconds()
            
            response = StrategyResponse(
                request_id=request_id,
                status='completed',
                timestamp=datetime.now(),
                execution_time_seconds=execution_time,
                strategy_code=strategy_result['strategy_code'],
                strategy_description=strategy_result['description'],
                model_ensemble_used=strategy_result['models_used'],
                data_quality_score=data_response.quality_score,
                data_source_used=data_response.source_used,
                data_cost=data_response.cost,
                sharpe_ratio=backtest_results.sharpe_ratio,
                total_return=backtest_results.total_return,
                annualized_return=backtest_results.annualized_return,
                max_drawdown=backtest_results.max_drawdown,
                volatility=backtest_results.volatility,
                calmar_ratio=backtest_results.calmar_ratio,
                walk_forward_stability=walk_forward_results['stability_score'] if walk_forward_results else None,
                monte_carlo_confidence=monte_carlo_results['confidence_intervals']['sharpe_ratio'] if monte_carlo_results else None,
                probability_target_achieved=monte_carlo_results['probability_sharpe_gt_2'] if monte_carlo_results else 0.0,
                var_95=backtest_results.var_95,
                expected_shortfall=backtest_results.cvar_95,
                tail_ratio=0.0,  # Would calculate separately
                equity_curve_url=report_urls['equity_curve'],
                performance_report_url=report_urls['performance_report'],
                detailed_analysis_url=report_urls['detailed_analysis'],
                deployment_ready=backtest_results.sharpe_ratio >= request.target_sharpe,
                estimated_live_performance=self._estimate_live_performance(backtest_results)
            )
            
            # Store final results
            await self._store_results(request_id, response)
            await self._update_job_status(request_id, 'completed', 1.0, 'Pipeline completed successfully')
            
            # Send notification if requested
            if request.notify_email:
                await self._send_completion_notification(request.notify_email, response)
            
            logger.info(f"[{request_id}] Pipeline completed successfully in {execution_time:.1f}s")
            return request_id
            
        except Exception as e:
            logger.error(f"[{request_id}] Pipeline failed: {str(e)}")
            await self._update_job_status(request_id, 'failed', 0.0, f'Error: {str(e)}')
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
    
    async def get_job_status(self, request_id: str) -> JobStatus:
        """Get current job status"""
        try:
            response = self.jobs_table.get_item(Key={'request_id': request_id})
            
            if 'Item' not in response:
                raise HTTPException(status_code=404, detail="Job not found")
            
            item = response['Item']
            return JobStatus(
                request_id=request_id,
                status=item['status'],
                progress=float(item['progress']),
                current_step=item['current_step'],
                estimated_completion=datetime.fromisoformat(item['estimated_completion']) if item.get('estimated_completion') else None,
                error_message=item.get('error_message')
            )
            
        except ClientError as e:
            logger.error(f"DynamoDB error: {e}")
            raise HTTPException(status_code=500, detail="Database error")
    
    async def get_results(self, request_id: str) -> StrategyResponse:
        """Retrieve completed results"""
        try:
            # Get from S3
            key = f"results/{request_id}/response.json"
            response = self.s3_client.get_object(Bucket=self.results_bucket, Key=key)
            
            results_data = json.loads(response['Body'].read().decode('utf-8'))
            return StrategyResponse(**results_data)
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(status_code=404, detail="Results not found")
            else:
                logger.error(f"S3 error: {e}")
                raise HTTPException(status_code=500, detail="Storage error")
    
    async def _update_job_status(self, 
                               request_id: str, 
                               status: str, 
                               progress: float, 
                               current_step: str,
                               error_message: str = None):
        """Update job status in DynamoDB"""
        
        estimated_completion = None
        if status == 'running' and progress > 0:
            # Estimate completion time based on current progress
            elapsed = datetime.now() - datetime.fromisoformat(
                self.jobs_table.get_item(Key={'request_id': request_id}).get('Item', {}).get('created_at', datetime.now().isoformat())
            )
            total_estimated = elapsed / progress
            estimated_completion = (datetime.now() + (total_estimated - elapsed)).isoformat()
        
        try:
            self.jobs_table.put_item(
                Item={
                    'request_id': request_id,
                    'status': status,
                    'progress': progress,
                    'current_step': current_step,
                    'updated_at': datetime.now().isoformat(),
                    'estimated_completion': estimated_completion,
                    'error_message': error_message
                }
            )
        except ClientError as e:
            logger.error(f"Failed to update job status: {e}")
    
    async def _generate_reports(self, **kwargs) -> Dict[str, str]:
        """Generate and upload visual reports"""
        
        request_id = kwargs['request_id']
        backtest_results = kwargs['backtest_results']
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Generate equity curve plot
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            backtest_results.equity_curve.plot(title='Equity Curve')
            plt.ylabel('Portfolio Value')
            
            plt.subplot(2, 1, 2)
            backtest_results.drawdown_curve.plot(title='Drawdown', color='red')
            plt.ylabel('Drawdown %')
            plt.tight_layout()
            
            # Save to S3
            equity_curve_key = f"reports/{request_id}/equity_curve.png"
            plt.savefig(f"/tmp/equity_curve_{request_id}.png", dpi=150, bbox_inches='tight')
            
            self.s3_client.upload_file(
                f"/tmp/equity_curve_{request_id}.png",
                self.results_bucket,
                equity_curve_key
            )
            
            equity_curve_url = f"https://{self.results_bucket}.s3.amazonaws.com/{equity_curve_key}"
            
            # Generate performance report (HTML)
            performance_html = self._generate_performance_html(backtest_results, kwargs)
            performance_key = f"reports/{request_id}/performance_report.html"
            
            self.s3_client.put_object(
                Bucket=self.results_bucket,
                Key=performance_key,
                Body=performance_html,
                ContentType='text/html'
            )
            
            performance_url = f"https://{self.results_bucket}.s3.amazonaws.com/{performance_key}"
            
            # Generate detailed analysis (JSON)
            detailed_analysis = {
                'backtest_results': asdict(backtest_results),
                'walk_forward': kwargs.get('walk_forward_results'),
                'monte_carlo': kwargs.get('monte_carlo_results'),
                'strategy_details': kwargs.get('strategy_result')
            }
            
            detailed_key = f"reports/{request_id}/detailed_analysis.json"
            self.s3_client.put_object(
                Bucket=self.results_bucket,
                Key=detailed_key,
                Body=json.dumps(detailed_analysis, default=str, indent=2),
                ContentType='application/json'
            )
            
            detailed_url = f"https://{self.results_bucket}.s3.amazonaws.com/{detailed_key}"
            
            return {
                'equity_curve': equity_curve_url,
                'performance_report': performance_url,
                'detailed_analysis': detailed_url
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                'equity_curve': '',
                'performance_report': '',
                'detailed_analysis': ''
            }
    
    def _generate_performance_html(self, backtest_results, kwargs) -> str:
        """Generate HTML performance report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AQP Strategy Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 10px 0; }}
                .header {{ color: #2E86AB; font-size: 24px; margin-bottom: 20px; }}
                .section {{ margin: 30px 0; }}
                .good {{ color: green; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                .bad {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">AQP Strategy Performance Report</div>
            
            <div class="section">
                <h3>Core Performance Metrics</h3>
                <div class="metric">Sharpe Ratio: <span class="{'good' if backtest_results.sharpe_ratio >= 2.0 else 'warning' if backtest_results.sharpe_ratio >= 1.0 else 'bad'}">{backtest_results.sharpe_ratio:.3f}</span></div>
                <div class="metric">Total Return: {backtest_results.total_return:.2%}</div>
                <div class="metric">Annualized Return: {backtest_results.annualized_return:.2%}</div>
                <div class="metric">Volatility: {backtest_results.volatility:.2%}</div>
                <div class="metric">Max Drawdown: <span class="{'good' if backtest_results.max_drawdown > -0.1 else 'warning' if backtest_results.max_drawdown > -0.2 else 'bad'}">{backtest_results.max_drawdown:.2%}</span></div>
                <div class="metric">Calmar Ratio: {backtest_results.calmar_ratio:.3f}</div>
            </div>
            
            <div class="section">
                <h3>Risk Metrics</h3>
                <div class="metric">Value at Risk (95%): {backtest_results.var_95:.3f}</div>
                <div class="metric">Expected Shortfall: {backtest_results.cvar_95:.3f}</div>
                <div class="metric">Skewness: {backtest_results.skewness:.3f}</div>
                <div class="metric">Kurtosis: {backtest_results.kurtosis:.3f}</div>
            </div>
            
            <div class="section">
                <h3>Strategy Details</h3>
                <div class="metric">Models Used: {', '.join(kwargs.get('strategy_result', {}).get('models_used', []))}</div>
                <div class="metric">Data Source: {kwargs.get('data_response', {}).get('source_used', 'Unknown')}</div>
                <div class="metric">Data Quality: {kwargs.get('data_response', {}).get('quality_score', 0.0):.2f}</div>
            </div>
            
            <div class="section">
                <h3>Deployment Assessment</h3>
                <div class="metric">Target Achieved: <span class="{'good' if backtest_results.sharpe_ratio >= 2.0 else 'bad'}">{'Yes' if backtest_results.sharpe_ratio >= 2.0 else 'No'}</span></div>
                <div class="metric">Risk Controlled: <span class="{'good' if backtest_results.max_drawdown > -0.15 else 'bad'}">{'Yes' if backtest_results.max_drawdown > -0.15 else 'No'}</span></div>
                <div class="metric">Deployment Ready: <span class="{'good' if backtest_results.sharpe_ratio >= 2.0 and backtest_results.max_drawdown > -0.15 else 'bad'}">{'Yes' if backtest_results.sharpe_ratio >= 2.0 and backtest_results.max_drawdown > -0.15 else 'No'}</span></div>
            </div>
            
            <p><em>Generated by Agentic Quantitative Platform on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """
        
        return html
    
    async def _store_results(self, request_id: str, response: StrategyResponse):
        """Store final results in S3"""
        
        key = f"results/{request_id}/response.json"
        
        self.s3_client.put_object(
            Bucket=self.results_bucket,
            Key=key,
            Body=json.dumps(asdict(response), default=str, indent=2),
            ContentType='application/json'
        )
    
    def _estimate_live_performance(self, backtest_results) -> Dict[str, float]:
        """Estimate live trading performance adjustments"""
        
        # Apply realistic live trading degradation factors
        slippage_impact = 0.95  # 5% performance degradation from slippage
        capacity_impact = 0.98   # 2% degradation from capacity constraints
        regime_change_impact = 0.92  # 8% degradation from regime changes
        
        overall_impact = slippage_impact * capacity_impact * regime_change_impact
        
        return {
            'estimated_live_sharpe': backtest_results.sharpe_ratio * overall_impact,
            'estimated_live_return': backtest_results.annualized_return * overall_impact,
            'confidence_interval_lower': backtest_results.sharpe_ratio * overall_impact * 0.8,
            'confidence_interval_upper': backtest_results.sharpe_ratio * overall_impact * 1.1
        }
    
    async def _send_completion_notification(self, email: str, response: StrategyResponse):
        """Send completion notification via SNS"""
        
        message = f"""
        AQP Strategy Analysis Complete
        
        Request ID: {response.request_id}
        Strategy: {response.strategy_description}
        
        Results:
        - Sharpe Ratio: {response.sharpe_ratio:.3f}
        - Total Return: {response.total_return:.2%}
        - Max Drawdown: {response.max_drawdown:.2%}
        
        Target Achieved: {'Yes' if response.deployment_ready else 'No'}
        
        View full report: {response.performance_report_url}
        """
        
        try:
            self.sns_client.publish(
                TopicArn=self.aws_config.get('notification_topic'),
                Message=message,
                Subject='AQP Strategy Analysis Complete'
            )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

# ========================================
# FASTAPI APPLICATION
# ========================================

app = FastAPI(title="Agentic Quantitative Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator (would be configured via environment)
AWS_CONFIG = {
    'alpha_vantage_key': 'your_key',
    'fred_key': 'your_key',
    'quandl_key': 'your_key',
    's3_bucket': 'aqp-data-bucket',
    'results_bucket': 'aqp-results-bucket',
    'jobs_table': 'aqp-jobs',
    'notification_topic': 'arn:aws:sns:us-east-1:123456789:aqp-notifications'
}

orchestrator = AWSQuantOrchestrator(AWS_CONFIG)

@app.post("/strategy/analyze", response_model=dict)
async def analyze_strategy(request: StrategyRequest, background_tasks: BackgroundTasks):
    """Execute complete strategy analysis pipeline"""
    
    # Start pipeline in background
    request_id = await orchestrator.execute_strategy_pipeline(request)
    
    return {
        "request_id": request_id,
        "status": "initiated",
        "message": "Strategy analysis pipeline started",
        "estimated_completion_minutes": 15
    }

@app.get("/strategy/status/{request_id}", response_model=JobStatus)
async def get_strategy_status(request_id: str):
    """Get current status of strategy analysis"""
    return await orchestrator.get_job_status(request_id)

@app.get("/strategy/results/{request_id}", response_model=StrategyResponse)
async def get_strategy_results(request_id: str):
    """Get completed strategy analysis results"""
    return await orchestrator.get_results(request_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_data_sources": orchestrator.data_aggregator.get_available_sources()
    }

@app.get("/")
async def root():
    """API documentation"""
    return {
        "message": "Agentic Quantitative Platform API",
        "version": "1.0.0",
        "endpoints": {
            "POST /strategy/analyze": "Execute complete strategy analysis",
            "GET /strategy/status/{id}": "Get analysis status",
            "GET /strategy/results/{id}": "Get completed results",
            "GET /health": "System health check"
        },
        "target_performance": "Sharpe Ratio >= 2.0"
    }

# ========================================
# AWS LAMBDA HANDLER
# ========================================

def lambda_handler(event, context):
    """AWS Lambda handler for serverless deployment"""
    
    import awsgi
    return awsgi.response(app, event, context)

# ========================================
# EXAMPLE USAGE
# ========================================

if __name__ == "__main__":
    # For local development
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Example API call:
    """
    curl -X POST "http://localhost:8000/strategy/analyze" \
         -H "Content-Type: application/json" \
         -d '{
           "symbols": ["AAPL", "MSFT", "GOOGL"],
           "start_date": "2020-01-01",
           "end_date": "2024-01-01",
           "strategy_description": "Momentum strategy with volatility filtering",
           "target_sharpe": 2.0,
           "include_walk_forward": true,
           "include_monte_carlo": true,
           "notify_email": "trader@example.com"
         }'
    """