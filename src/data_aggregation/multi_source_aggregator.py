"""
AQP Data Aggregator - Multi-Source Market Data Pipeline
Handles real-time and historical data from multiple sources with caching and AWS scaling
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import redis
import boto3
from dataclasses import dataclass, asdict
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import alpaca_trade_api as tradeapi
from polygon import RESTClient
import finnhub
import quandl

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str
    adjusted_close: Optional[float] = None
    dividend: Optional[float] = None
    split_ratio: Optional[float] = None

@dataclass
class AlternativeData:
    """Alternative data structure"""
    symbol: str
    timestamp: datetime
    data_type: str  # sentiment, social, options_flow, etc.
    value: Union[float, str, Dict]
    source: str
    confidence: Optional[float] = None

class DataAggregator:
    """
    Multi-source data aggregator with intelligent routing, caching, and AWS scaling
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.setup_connections()
        self.setup_aws()
        self.data_sources = {}
        self.cache = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AQP-DataAggregator')
        
    def setup_connections(self):
        """Initialize all data source connections"""
        # Alpaca
        if 'alpaca_key' in self.config:
            self.data_sources['alpaca'] = tradeapi.REST(
                self.config['alpaca_key'],
                self.config['alpaca_secret'],
                base_url=self.config.get('alpaca_base_url', 'https://paper-api.alpaca.markets')
            )
            
        # Polygon
        if 'polygon_key' in self.config:
            self.data_sources['polygon'] = RESTClient(self.config['polygon_key'])
            
        # Finnhub
        if 'finnhub_key' in self.config:
            self.data_sources['finnhub'] = finnhub.Client(api_key=self.config['finnhub_key'])
            
        # Yahoo Finance (free)
        self.data_sources['yahoo'] = True
        
        # Quandl
        if 'quandl_key' in self.config:
            quandl.ApiConfig.api_key = self.config['quandl_key']
            self.data_sources['quandl'] = True
            
    def setup_aws(self):
        """Setup AWS services for scaling"""
        self.s3 = boto3.client('s3')
        self.timestream = boto3.client('timestream-write')
        self.lambda_client = boto3.client('lambda')
        
    async def get_market_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = '1Day',
        sources: List[str] = None
    ) -> Dict[str, List[MarketData]]:
        """
        Aggregate market data from multiple sources with intelligent failover
        """
        sources = sources or ['alpaca', 'polygon', 'yahoo']
        results = {}
        
        tasks = []
        for symbol in symbols:
            task = self._fetch_symbol_data(symbol, start_date, end_date, timeframe, sources)
            tasks.append(task)
            
        symbol_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, symbol in enumerate(symbols):
            if isinstance(symbol_data[i], Exception):
                self.logger.error(f"Failed to fetch data for {symbol}: {symbol_data[i]}")
                results[symbol] = []
            else:
                results[symbol] = symbol_data[i]
                
        return results
    
    async def _fetch_symbol_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime, 
        timeframe: str,
        sources: List[str]
    ) -> List[MarketData]:
        """Fetch data for a single symbol with source failover"""
        
        # Check cache first
        cache_key = f"market_data:{symbol}:{timeframe}:{start_date.date()}:{end_date.date()}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return [MarketData(**item) for item in json.loads(cached_data)]
        
        # Try each source in order
        for source in sources:
            try:
                data = await self._fetch_from_source(source, symbol, start_date, end_date, timeframe)
                if data:
                    # Cache successful result
                    cache_data = [asdict(item) for item in data]
                    self.cache.setex(cache_key, 3600, json.dumps(cache_data, default=str))
                    return data
            except Exception as e:
                self.logger.warning(f"Source {source} failed for {symbol}: {e}")
                continue
                
        self.logger.error(f"All sources failed for {symbol}")
        return []
    
    async def _fetch_from_source(
        self, 
        source: str, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime, 
        timeframe: str
    ) -> List[MarketData]:
        """Fetch from specific data source"""
        
        if source == 'alpaca' and 'alpaca' in self.data_sources:
            return await self._fetch_alpaca(symbol, start_date, end_date, timeframe)
        elif source == 'polygon' and 'polygon' in self.data_sources:
            return await self._fetch_polygon(symbol, start_date, end_date, timeframe)
        elif source == 'yahoo':
            return await self._fetch_yahoo(symbol, start_date, end_date, timeframe)
        else:
            raise ValueError(f"Unsupported source: {source}")
    
    async def _fetch_alpaca(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str) -> List[MarketData]:
        """Fetch from Alpaca"""
        alpaca = self.data_sources['alpaca']
        
        # Convert timeframe
        if timeframe == '1Day':
            tf = tradeapi.TimeFrame.Day
        elif timeframe == '1Hour':
            tf = tradeapi.TimeFrame.Hour
        elif timeframe == '1Min':
            tf = tradeapi.TimeFrame.Minute
        else:
            tf = tradeapi.TimeFrame.Day
            
        bars = alpaca.get_bars(symbol, tf, start=start_date, end=end_date).df
        
        data = []
        for timestamp, row in bars.iterrows():
            data.append(MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                source='alpaca'
            ))
        return data
    
    async def _fetch_polygon(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str) -> List[MarketData]:
        """Fetch from Polygon"""
        polygon = self.data_sources['polygon']
        
        # Convert timeframe for Polygon
        if timeframe == '1Day':
            multiplier, timespan = 1, 'day'
        elif timeframe == '1Hour':
            multiplier, timespan = 1, 'hour'
        elif timeframe == '1Min':
            multiplier, timespan = 1, 'minute'
        else:
            multiplier, timespan = 1, 'day'
            
        aggs = polygon.get_aggs(
            ticker=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d')
        )
        
        data = []
        for agg in aggs:
            data.append(MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(agg.timestamp / 1000),
                open=agg.open,
                high=agg.high,
                low=agg.low,
                close=agg.close,
                volume=agg.volume,
                source='polygon'
            ))
        return data
    
    async def _fetch_yahoo(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str) -> List[MarketData]:
        """Fetch from Yahoo Finance"""
        
        # Convert timeframe for yfinance
        if timeframe == '1Day':
            interval = '1d'
        elif timeframe == '1Hour':
            interval = '1h'
        elif timeframe == '1Min':
            interval = '1m'
        else:
            interval = '1d'
            
        ticker = yf.Ticker(symbol)
        hist = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval
        )
        
        data = []
        for timestamp, row in hist.iterrows():
            data.append(MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume'],
                source='yahoo',
                adjusted_close=row.get('Adj Close'),
                dividend=row.get('Dividends'),
                split_ratio=row.get('Stock Splits')
            ))
        return data
    
    async def get_alternative_data(
        self, 
        symbols: List[str], 
        data_types: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, List[AlternativeData]]:
        """
        Fetch alternative data (sentiment, social, options flow, etc.)
        """
        results = {}
        
        for symbol in symbols:
            symbol_alt_data = []
            
            for data_type in data_types:
                try:
                    if data_type == 'sentiment' and 'finnhub' in self.data_sources:
                        sentiment_data = await self._fetch_sentiment(symbol, start_date, end_date)
                        symbol_alt_data.extend(sentiment_data)
                    elif data_type == 'social':
                        social_data = await self._fetch_social_data(symbol, start_date, end_date)
                        symbol_alt_data.extend(social_data)
                    elif data_type == 'options_flow':
                        options_data = await self._fetch_options_flow(symbol, start_date, end_date)
                        symbol_alt_data.extend(options_data)
                        
                except Exception as e:
                    self.logger.error(f"Failed to fetch {data_type} for {symbol}: {e}")
                    
            results[symbol] = symbol_alt_data
            
        return results
    
    async def _fetch_sentiment(self, symbol: str, start_date: datetime, end_date: datetime) -> List[AlternativeData]:
        """Fetch sentiment data from Finnhub"""
        finnhub = self.data_sources['finnhub']
        
        sentiment_data = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                # Get news sentiment
                news = finnhub.company_news(symbol, 
                    _from=current_date.strftime('%Y-%m-%d'), 
                    to=current_date.strftime('%Y-%m-%d')
                )
                
                if news:
                    # Calculate average sentiment
                    sentiments = [article.get('sentiment', 0) for article in news if 'sentiment' in article]
                    if sentiments:
                        avg_sentiment = np.mean(sentiments)
                        sentiment_data.append(AlternativeData(
                            symbol=symbol,
                            timestamp=current_date,
                            data_type='sentiment',
                            value=avg_sentiment,
                            source='finnhub',
                            confidence=0.8
                        ))
                        
            except Exception as e:
                self.logger.warning(f"Failed to fetch sentiment for {symbol} on {current_date}: {e}")
                
            current_date += timedelta(days=1)
            
        return sentiment_data
    
    async def _fetch_social_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[AlternativeData]:
        """Fetch social media data"""
        # Placeholder for social data fetching
        # Would integrate with Reddit API, Twitter API, etc.
        return []
    
    async def _fetch_options_flow(self, symbol: str, start_date: datetime, end_date: datetime) -> List[AlternativeData]:
        """Fetch options flow data"""
        # Placeholder for options flow data
        # Would integrate with options data providers
        return []
    
    def store_data_aws(self, data: Dict[str, List[MarketData]], table_name: str = 'aqp-market-data'):
        """Store data in AWS TimeStream for scalable querying"""
        
        records = []
        for symbol, symbol_data in data.items():
            for point in symbol_data:
                record = {
                    'Time': str(int(point.timestamp.timestamp() * 1000)),
                    'TimeUnit': 'MILLISECONDS',
                    'Dimensions': [
                        {'Name': 'symbol', 'Value': symbol},
                        {'Name': 'source', 'Value': point.source}
                    ],
                    'MeasureName': 'market_data',
                    'MeasureValueType': 'MULTI',
                    'MeasureValues': [
                        {'Name': 'open', 'Value': str(point.open), 'Type': 'DOUBLE'},
                        {'Name': 'high', 'Value': str(point.high), 'Type': 'DOUBLE'},
                        {'Name': 'low', 'Value': str(point.low), 'Type': 'DOUBLE'},
                        {'Name': 'close', 'Value': str(point.close), 'Type': 'DOUBLE'},
                        {'Name': 'volume', 'Value': str(point.volume), 'Type': 'BIGINT'}
                    ]
                }
                records.append(record)
                
        # Write in batches
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                self.timestream.write_records(
                    DatabaseName='aqp-database',
                    TableName=table_name,
                    Records=batch
                )
            except Exception as e:
                self.logger.error(f"Failed to write batch to TimeStream: {e}")
    
    async def scale_with_lambda(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Scale data fetching using AWS Lambda for massive symbol lists"""
        
        # Split symbols into chunks for parallel Lambda execution
        chunk_size = 50
        symbol_chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        
        lambda_tasks = []
        for chunk in symbol_chunks:
            payload = {
                'symbols': chunk,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'config': self.config
            }
            
            task = self._invoke_lambda_async('aqp-data-fetcher', payload)
            lambda_tasks.append(task)
            
        # Wait for all Lambda functions to complete
        results = await asyncio.gather(*lambda_tasks, return_exceptions=True)
        
        # Aggregate results
        all_data = {}
        for result in results:
            if isinstance(result, dict):
                all_data.update(result)
                
        return all_data
    
    async def _invoke_lambda_async(self, function_name: str, payload: Dict):
        """Invoke Lambda function asynchronously"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: self.lambda_client.invoke(
                    FunctionName=function_name,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(payload)
                )
            )
            
        return json.loads(response['Payload'].read())


# Example usage and configuration
if __name__ == "__main__":
    config = {
        'alpaca_key': 'your_alpaca_key',
        'alpaca_secret': 'your_alpaca_secret',
        'polygon_key': 'your_polygon_key',
        'finnhub_key': 'your_finnhub_key',
        'quandl_key': 'your_quandl_key',
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    aggregator = DataAggregator(config)
    
    # Example: Fetch data for multiple symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    async def main():
        # Get market data
        market_data = await aggregator.get_market_data(symbols, start_date, end_date)
        
        # Get alternative data
        alt_data = await aggregator.get_alternative_data(
            symbols, 
            ['sentiment', 'social'], 
            start_date, 
            end_date
        )
        
        # Store in AWS for scalable querying
        aggregator.store_data_aws(market_data)
        
        print(f"Fetched data for {len(symbols)} symbols")
        for symbol, data in market_data.items():
            print(f"{symbol}: {len(data)} data points")
    
    asyncio.run(main())