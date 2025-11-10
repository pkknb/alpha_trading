"""
实盘交易系统监控脚本
实时显示系统状态、持仓、收益等信息
"""

import sys
import time
from datetime import datetime
import pandas as pd
from tabulate import tabulate

# 假设主程序已导入
# from alpha_live_trading import RoostooAPIClient, AlphaLiveTrading, PriceDataManager


class TradingMonitor:
    """交易系统监控器"""
    
    def __init__(self, api_client, trading_pairs):
        self.api = api_client
        self.trading_pairs = trading_pairs
        self.initial_value = None
        
    def get_account_summary(self):
        """获取账户摘要"""
        balance = self.api.get_balance()
        
        if not balance or not balance.get('Success'):
            return None
        
        wallet = balance.get('Wallet', {})
        
        # 获取当前价格
        ticker = self.api.get_ticker()
        prices = {}
        if ticker and ticker.get('Success'):
            data = ticker.get('Data', {})
            for pair in self.trading_pairs:
                if pair in data:
                    prices[pair] = float(data[pair].get('LastPrice', 0))
        
        # 计算总价值
        usd_balance = float(wallet.get('USD', {}).get('Free', 0))
        total_value = usd_balance
        
        positions = []
        for pair in self.trading_pairs:
            coin = pair.split('/')[0]
            if coin in wallet:
                qty = float(wallet[coin].get('Free', 0)) + float(wallet[coin].get('Locked', 0))
                if qty > 0 and pair in prices:
                    value = qty * prices[pair]
                    total_value += value
                    positions.append({
                        'Coin': coin,
                        'Quantity': f"{qty:.6f}",
                        'Price': f"${prices[pair]:,.2f}",
                        'Value': f"${value:,.2f}",
                        'Allocation': f"{(value/total_value*100):.2f}%"
                    })
        
        if self.initial_value is None:
            self.initial_value = total_value
        
        pnl = total_value - self.initial_value
        pnl_pct = (pnl / self.initial_value * 100) if self.initial_value > 0 else 0
        
        return {
            'total_value': total_value,
            'usd_balance': usd_balance,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'positions': positions
        }
    
    def get_pending_orders(self):
        """获取所有未成交订单"""
        all_orders = []
        
        for pair in self.trading_pairs:
            result = self.api.query_order(pair=pair, pending_only=True)
            if result and result.get('Success'):
                orders = result.get('OrderMatched', [])
                for order in orders:
                    all_orders.append({
                        'OrderID': order.get('OrderID', '')[:8] + '...',
                        'Pair': pair,
                        'Side': order.get('Side', ''),
                        'Type': order.get('Type', ''),
                        'Quantity': f"{float(order.get('Quantity', 0)):.6f}",
                        'Price': f"${float(order.get('Price', 0)):,.2f}",
                        'Status': order.get('Status', ''),
                        'Filled': f"{float(order.get('ExecutedQty', 0)):.6f}"
                    })
        
        return all_orders
    
    def display_dashboard(self):
        """显示监控面板"""
        # 清屏
        print("\033[2J\033[H")  # ANSI清屏
        
        # 标题
        print("=" * 80)
        print(f"{'Alpha实盘交易系统监控面板':^80}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}")
        print("=" * 80)
        
        # 账户摘要
        summary = self.get_account_summary()
        if summary:
            print("\n【账户摘要】")
            print(f"  总资产: ${summary['total_value']:,.2f}")
            print(f"  现金余额: ${summary['usd_balance']:,.2f}")
            print(f"  盈亏: ${summary['pnl']:,.2f} ({summary['pnl_pct']:+.2f}%)")
            
            # 持仓明细
            if summary['positions']:
                print("\n【当前持仓】")
                print(tabulate(summary['positions'], headers='keys', tablefmt='grid'))
            else:
                print("\n【当前持仓】")
                print("  无持仓")
        else:
            print("\n【账户摘要】")
            print("  获取账户信息失败")
        
        # 未成交订单
        pending = self.get_pending_orders()
        if pending:
            print("\n【未成交订单】")
            print(tabulate(pending, headers='keys', tablefmt='grid'))
        else:
            print("\n【未成交订单】")
            print("  无未成交订单")
        
        # 实时价格
        print("\n【实时价格】")
        ticker = self.api.get_ticker()
        if ticker and ticker.get('Success'):
            data = ticker.get('Data', {})
            price_data = []
            for pair in self.trading_pairs:
                if pair in data:
                    info = data[pair]
                    price_data.append({
                        'Pair': pair,
                        'Last': f"${float(info.get('LastPrice', 0)):,.2f}",
                        'Bid': f"${float(info.get('BestBidPrice', 0)):,.2f}",
                        'Ask': f"${float(info.get('BestAskPrice', 0)):,.2f}",
                        'Volume': f"{float(info.get('Volume24H', 0)):,.2f}",
                        'Change%': f"{float(info.get('PriceChange24H', 0)):+.2f}%"
                    })
            
            if price_data:
                print(tabulate(price_data, headers='keys', tablefmt='grid'))
        
        print("\n" + "=" * 80)
        print("按 Ctrl+C 退出监控")
    
    def run_monitor(self, interval=10):
        """持续运行监控"""
        try:
            while True:
                self.display_dashboard()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n监控已停止")


"""
追踪和记录交易绩效
"""

import json
from datetime import datetime


class PerformanceTracker:
    """绩效追踪器"""
    
    def __init__(self, log_file='performance.json'):
        self.log_file = log_file
        self.records = []
        self.load_records()
    
    def load_records(self):
        """加载历史记录"""
        try:
            with open(self.log_file, 'r') as f:
                self.records = json.load(f)
        except FileNotFoundError:
            self.records = []
    
    def save_records(self):
        """保存记录"""
        with open(self.log_file, 'w') as f:
            json.dump(self.records, f, indent=2)
    
    def log_rebalance(self, portfolio_value, positions, trades):
        """记录再平衡事件"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'positions': positions,
            'trades': trades
        }
        self.records.append(record)
        self.save_records()
    
    def calculate_metrics(self):
        """计算绩效指标"""
        if len(self.records) < 2:
            return {}
        
        df = pd.DataFrame(self.records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # 计算收益率
        values = df['portfolio_value'].values
        returns = (values[1:] - values[:-1]) / values[:-1]
        
        metrics = {
            'total_return': (values[-1] - values[0]) / values[0],
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(values),
            'win_rate': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0,
            'num_rebalances': len(self.records)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, values):
        """计算最大回撤"""
        cummax = pd.Series(values).cummax()
        drawdown = (pd.Series(values) - cummax) / cummax
        return drawdown.min()
    
    def display_performance(self):
        """显示绩效报告"""
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("数据不足，无法计算绩效指标")
            return
        
        print("\n" + "=" * 50)
        print("绩效报告")
        print("=" * 50)
        print(f"总收益率: {metrics['total_return']*100:.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"胜率: {metrics['win_rate']*100:.2f}%")
        print(f"再平衡次数: {metrics['num_rebalances']}")
        print("=" * 50)


if __name__ == "__main__":
    import sys
    
    # 根据命令行参数选择功能
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "monitor":
            # 启动监控
            from alpha_live_trading import RoostooAPIClient
            from config import API_CONFIG, TRADING_PAIRS
            
            api = RoostooAPIClient(**API_CONFIG)
            monitor = TradingMonitor(api, TRADING_PAIRS)
            monitor.run_monitor(interval=10)
        
        elif command == "performance":
            # 显示绩效
            tracker = PerformanceTracker()
            tracker.display_performance()
        
        else:
            print("未知命令")
            print("用法:")
            print("  python monitor.py monitor      # 启动监控")
            print("  python monitor.py performance  # 显示绩效")
    else:
        print("用法:")
        print("  python monitor.py monitor      # 启动监控")
        print("  python monitor.py performance  # 显示绩效")