"""
Alpha实盘交易系统配置文件
"""

# API配置
API_CONFIG = {
    'api_key': "w2bR9XU4g6eN8qT1jY0LzA7cD3fV5sK2rC1mF8hJ9pQ4uB6vW3oP5xI7lS0nM2tY",
    'secret_key': "p7LwX3gH1qV8yJ4bS0nK6tF2zU9mR5oC8dA1sI3vW7eN6lP4xT0jZ9fB2kY5hM",
    'base_url': "https://mock-api.roostoo.com"
}

# 交易对配置
TRADING_PAIRS = [
    "BTC/USD",
    "ETH/USD",
    "BNB/USD",
    "XRP/USD",
    "DOGE/USD",
    "SOL/USD",
    "ARB/USD",
]

# 策略参数
STRATEGY_CONFIG = {
    'min_data_points': 100,       # 开始交易前需要的最少数据点（30分钟）
    'rebalance_interval': 10,    # 再平衡间隔（分钟）
    'min_position_value': 10.0,  # 最小换仓变化区间
    'max_position_pct': 0.3,     # 单个资产最大持仓比例（30%）
}

# 数据管理配置
DATA_CONFIG = {
    'max_history': 1440,              # 最大保存历史数据条数（1440 = 24小时）
    'data_file': 'price_history.json', # 数据持久化文件
    'collection_interval': 60,         # 数据收集间隔（秒）
}

# 订单管理配置
ORDER_CONFIG = {
    'timeout_minutes': 5,    # 订单超时时间（分钟）
    'max_retry': 3,          # 最大重试次数
}

# Alpha因子配置
ALPHA_CONFIG = {
    'momentum_window': 20,      # 动量窗口（分钟）
    'reversal_window': 5,       # 反转窗口（分钟）
    'volatility_window': 20,    # 波动率窗口（分钟）
    
    # 因子权重
    'momentum_weight': 0.5,
    'reversal_weight': 0.3,
    'volatility_weight': 0.2,
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'file': 'trading.log',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}
