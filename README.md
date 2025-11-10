# Alpha实盘交易系统 - 完整使用指南

## 📁 文件结构

```
alpha_trading/
├── alpha_live_trading.py      # 主程序（核心交易引擎）
├── config.py                  # 配置文件
├── monitor.py                 # 监控脚本
├── AlphaOperation.py          # Alpha因子计算（目前暂无）
├── start_trading.py           # 快速启动脚本
└── data/
    ├── price_history.json     # 价格历史数据（自动生成）
    ├── performance.json       # 绩效记录（自动生成）
    └── trading.log           # 交易日志（自动生成）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install requests pandas numpy tabulate
```

### 2. 配置参数

编辑 `config.py`：

```python
# 修改交易对
TRADING_PAIRS = [
    "BTC/USD",
    "ETH/USD",
]

# 调整策略参数
STRATEGY_CONFIG = {
    'min_data_points': 30,       # 需要30分钟数据才开始交易
    'rebalance_interval': 60,    # 每60分钟再平衡一次
    'min_position_value': 10.0,  # 最小交易10美元
    'max_position_pct': 0.3,     # 单个资产最多30%
}
```

### 3. 启动交易系统

```bash
# 方法1: 直接运行
python alpha_live_trading.py

# 方法2: 使用启动脚本（推荐）
python start_trading.py
```

### 4. 监控系统（新开终端）

```bash
# 实时监控
python monitor.py monitor

# 查看绩效
python monitor.py performance
```

## 📊 系统运行流程

### 阶段1: 数据收集期（0-30分钟）

```
时间 00:00 → 收集第1个价格点
时间 01:00 → 收集第2个价格点
...
时间 29:00 → 收集第30个价格点 ✓ 数据充足，准备交易
```

**日志示例：**
```
[数据收集] 获取价格: {'BTC/USD': 50000.0, 'ETH/USD': 3000.0}
[等待数据] 数据收集中... {'BTC/USD': {'count': 15, ...}}
```

### 阶段2: 正常交易期（30分钟后）

```
时间 30:00 → 计算Alpha → 首次再平衡 → 执行交易
时间 31:00 → 收集数据 → 检查订单
时间 32:00 → 收集数据 → 检查订单
...
时间 90:00 → 计算Alpha → 第2次再平衡 → 执行交易
```

**再平衡日志示例：**
```
======================================================================
[再平衡] 开始执行...
[Alpha计算] 信号值: {'BTC/USD': 0.45, 'ETH/USD': -0.23}
[权重计算] 目标权重: {'BTC/USD': 1.0, 'ETH/USD': 0.0}
[再平衡] 组合价值: $10000.00
[再平衡] BTC/USD: BUY 0.15 (目标$10000 vs 当前$2500)
[再平衡] ETH/USD: SELL 2.0 (目标$0 vs 当前$6000)
[订单成功] abc12345
[再平衡] 完成
======================================================================
```

## 🔧 核心功能详解

### 1. 数据管理 (PriceDataManager)

**功能：**
- ✅ 每60秒自动收集价格数据
- ✅ 存储最近24小时数据（可配置）
- ✅ 自动保存到JSON文件
- ✅ 程序重启后自动加载历史数据

**数据文件格式：**
```json
{
  "BTC/USD": [
    {"timestamp": "2024-01-01T10:00:00", "price": 50000.0},
    {"timestamp": "2024-01-01T10:01:00", "price": 50050.0}
  ],
  "ETH/USD": [...]
}
```

### 2. 订单管理 (OrderManager)

**自动处理未成交订单：**

| 情况 | 处理方式 |
|------|---------|
| 5分钟未成交 | 取消订单 → 转市价单重新下单 |
| 部分成交超时 | 取消订单 → 剩余部分转市价单 |
| 市价单失败 | 最多重试3次 |
| 全部成交 | 自动从追踪列表移除 |

**订单状态追踪：**
```python
pending_orders = {
    'order_123': {
        'pair': 'BTC/USD',
        'side': 'BUY',
        'quantity': 0.1,
        'timestamp': datetime(2024,1,1,10,0,0),
        'retry_count': 0
    }
}
```

### 3. Alpha信号计算

**当前实现的示例策略：**
```python
# 组合3个因子
alpha = 0.5 * momentum_20min +      # 20分钟动量
        0.3 * reversal_5min +       # 5分钟反转
        0.2 * volatility_signal     # 波动率因子
```

**⚠️ 重要：你需要替换为自己的Alpha逻辑！**

编辑 `alpha_live_trading.py` 中的 `calculate_alpha_signals()` 方法：

```python
def calculate_alpha_signals(self, price_df: pd.DataFrame) -> pd.Series:
    # 导入你的因子计算函数
    from AlphaOperation import ts_skew, ts_pct, ts_delta
    import torch
    
    # 使用你自己的Alpha逻辑
    alpha_values = {}
    for pair in self.trading_pairs:
        prices = torch.tensor(price_df[pair].values, dtype=torch.float32)
        
        # 示例：使用偏度因子
        skewness = ts_skew(prices, d=20)[-1].item()
        momentum = ts_pct(prices)[-1].item()
        
        alpha_values[pair] = 0.6 * skewness + 0.4 * momentum
    
    return pd.Series(alpha_values)
```

### 4. 权重计算与再平衡

**逻辑：**
```python
# 1. 只保留正Alpha信号（做多策略）
positive_signals = alpha[alpha > 0]

# 2. 按信号强度分配权重
weights = positive_signals / positive_signals.sum()

# 3. 限制单个资产最大权重（风控）
weights = weights.clip(upper=0.3)  # 最多30%

# 4. 计算需要的交易
for each pair:
    target_value = portfolio_value × weight
    current_value = current_position × current_price
    
    if |target_value - current_value| > min_trade_value:
        execute_trade()
```

## 📈 监控面板说明

启动监控后，你会看到：

```
================================================================================
                        Alpha实盘交易系统监控面板
                        2024-01-01 10:30:00
================================================================================

【账户摘要】
  总资产: $10,250.50
  现金余额: $1,200.30
  盈亏: $250.50 (+2.50%)

【当前持仓】
╒═══════╤════════════╤═══════════╤═══════════╤══════════════╕
│ Coin  │ Quantity   │ Price     │ Value     │ Allocation   │
╞═══════╪════════════╪═══════════╪═══════════╪══════════════╡
│ BTC   │ 0.150000   │ $50,000   │ $7,500    │ 73.17%       │
│ ETH   │ 0.500000   │ $3,000    │ $1,500    │ 14.63%       │
╘═══════╧════════════╧═══════════╧═══════════╧══════════════╛

【未成交订单】
  无未成交订单

【实时价格】
╒═══════════╤═══════════╤═══════════╤═══════════╤═══════════╤═══════════╕
│ Pair      │ Last      │ Bid       │ Ask       │ Volume    │ Change%   │
╞═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
│ BTC/USD   │ $50,000   │ $49,995   │ $50,005   │ 1,250.50  │ +2.30%    │
│ ETH/USD   │ $3,000    │ $2,998    │ $3,002    │ 5,678.90  │ +1.80%    │
╘═══════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╛
```

## ⚠️ 重要注意事项

### 1. 数据收集期间

- ❌ 不会执行任何交易
- ✅ 持续收集价格数据
- ✅ 数据自动保存，程序重启后继续

### 2. 订单未成交处理

**场景1：限价单5分钟未成交**
```
订单ID: abc123 (限价单 BUY 0.1 BTC @ $48,000)
↓
5分钟后检测到未成交
↓
取消限价单
↓
转为市价单重新下单
```

**场景2：部分成交**
```
订单: BUY 1.0 BTC
已成交: 0.6 BTC
剩余: 0.4 BTC
↓
超时后取消原订单
↓
剩余0.4 BTC转市价单
```

### 3. 程序重启

```bash
# 第1次运行（冷启动）
python alpha_live_trading.py
# → 从0开始收集数据，需要30分钟

# 第2次运行（热启动）
python alpha_live_trading.py
# → 自动加载 price_history.json
# → 如果数据充足，立即开始交易
```

### 4. 风险控制建议

1. **小资金测试**
   ```python
   # 先用100美元测试
   initial_capital = 100
   min_position_value = 5.0  # 降低最小交易额
   ```

2. **单资产限制**
   ```python
   max_position_pct = 0.2  # 单个最多20%
   ```

3. **添加止损**
   ```python
   # 在 execute_rebalance() 中添加
   portfolio_value = self.get_portfolio_value()
   if portfolio_value < initial_value * 0.9:  # 亏损10%
       logger.critical("触发止损！")
       # 清仓逻辑
   ```

## 🐛 常见问题

### Q1: 程序启动后一直显示"等待数据"？

**A:** 这是正常的！需要收集30个数据点（30分钟）才会开始交易。

### Q2: 订单一直不成交怎么办？

**A:** 系统会自动处理：
- 5分钟后自动转市价单
- 最多重试3次
- 可以手动调整 `timeout_minutes` 参数

### Q3: 如何修改Alpha因子？

**A:** 编辑 `calculate_alpha_signals()` 方法，使用 `AlphaOperation.py` 中的函数。

### Q4: 如何查看历史交易记录？

**A:** 查看 `performance.json` 文件或运行：
```bash
python monitor.py performance
```

### Q5: 如何安全停止程序？

**A:** 按 `Ctrl+C`，系统会：
1. 自动保存所有数据
2. 不会取消未成交订单（下次启动继续处理）
3. 记录当前状态

## 📝 日志文件说明

### trading.log
```
2024-01-01 10:00:00 - INFO - [数据收集] 获取价格...
2024-01-01 10:30:00 - INFO - [再平衡] 开始执行...
2024-01-01 10:30:05 - INFO - [订单成功] abc123
2024-01-01 10:35:00 - WARNING - [订单超时] 订单xyz456超时
```

### price_history.json
存储所有历史价格数据

### performance.json
存储每次再平衡的绩效记录
