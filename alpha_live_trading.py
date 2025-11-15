import requests
import time
import hmac
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from collections import deque
import json
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RoostooAPIClient:
    """Roostoo API客户端"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://mock-api.roostoo.com"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        
    def _get_timestamp(self):
        """生成13位毫秒时间戳"""
        return str(int(time.time() * 1000))
    
    def _get_signed_headers(self, payload={}):
        """生成签名头"""
        payload['timestamp'] = self._get_timestamp()
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{key}={payload[key]}" for key in sorted_keys)
        
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            total_params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'RST-API-KEY': self.api_key,
            'MSG-SIGNATURE': signature
        }
        
        return headers, payload, total_params
    
    def get_balance(self) -> Optional[Dict]:
        """获取账户余额"""
        url = f"{self.base_url}/v3/balance"
        headers, payload, _ = self._get_signed_headers(payload={})
        
        try:
            logger.info(f"[DEBUG][get_balance] 请求: url={url}, headers={headers}, params={payload}")
            response = requests.get(url, headers=headers, params=payload, timeout=10)
            logger.info(f"[DEBUG][get_balance] 响应状态: {response.status_code}")
            logger.info(f"[DEBUG][get_balance] 响应文本: {response.text}")
            response.raise_for_status()
            data = response.json()
            logger.info(f"[DEBUG][get_balance] 解析后的 JSON: {data}")
            return data
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return None
    
    def get_ticker(self, pair: Optional[str] = None) -> Optional[Dict]:
        """获取行情数据"""
        url = f"{self.base_url}/v3/ticker"
        params = {'timestamp': self._get_timestamp()}
        if pair:
            params['pair'] = pair
            
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取行情失败: {e}")
            return None
    
    def place_order(self, pair: str, side: str, quantity: float, 
                   price: Optional[float] = None, order_type: str = "MARKET") -> Optional[Dict]:
        """下单"""
        url = f"{self.base_url}/v3/place_order"
        
        payload = {
            'pair': pair,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity)
        }
        if order_type.upper() == 'LIMIT' and price is not None:
            payload['price'] = str(price)
        
        headers, payload, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'

        logger.info(f"[DEBUG][place_order] 请求: url={url}")
        logger.info(f"[DEBUG][place_order] headers={headers}")
        logger.info(f"[DEBUG][place_order] body(total_params)={total_params}")
        
        try:
            response = requests.post(url, headers=headers, data=total_params, timeout=10)
            logger.info(f"[DEBUG][place_order] 响应状态: {response.status_code}")
            logger.info(f"[DEBUG][place_order] 响应文本: {response.text}")
            response.raise_for_status()
            data = response.json()
            logger.info(f"[DEBUG][place_order] 解析后的 JSON: {data}")
            return data
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return None
    
    def query_order(self, order_id: Optional[str] = None, pair: Optional[str] = None, 
                   pending_only: Optional[bool] = None) -> Optional[Dict]:
        """查询订单"""
        url = f"{self.base_url}/v3/query_order"
        
        payload = {}
        if order_id:
            payload['order_id'] = str(order_id)
        elif pair:
            payload['pair'] = pair
            if pending_only is not None:
                payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'
        
        headers, payload, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        try:
            response = requests.post(url, headers=headers, data=total_params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"查询订单失败: {e}")
            return None
    
    def cancel_order(self, order_id: Optional[str] = None, pair: Optional[str] = None) -> Optional[Dict]:
        """撤单"""
        url = f"{self.base_url}/v3/cancel_order"
        
        payload = {}
        if order_id:
            payload['order_id'] = str(order_id)
        elif pair:
            payload['pair'] = pair
        
        headers, payload, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        try:
            response = requests.post(url, headers=headers, data=total_params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"撤单失败: {e}")
            return None


class PriceDataManager:
    """价格数据管理器 - 每分钟收集并存储数据"""
    
    def __init__(self, api_client: RoostooAPIClient, trading_pairs: List[str], 
                 max_history: int = 1440, data_file: str = "price_history.json"):
        """
        初始化数据管理器
        
        Args:
            api_client: API客户端
            trading_pairs: 交易对列表
            max_history: 最大保存历史数据条数（默认1440=24小时）
            data_file: 数据持久化文件
        """
        self.api = api_client
        self.trading_pairs = trading_pairs
        self.max_history = max_history
        self.data_file = data_file
        
        # 使用deque存储价格历史 {pair: deque of (timestamp, price)}
        self.price_history = {pair: deque(maxlen=max_history) for pair in trading_pairs}
        
        # 加载历史数据
        self.load_history()
        
    def fetch_current_prices(self) -> Dict[str, float]:
        """获取当前价格"""
        prices = {}
        ticker_data = self.api.get_ticker()
        
        if ticker_data and ticker_data.get('Success'):
            data = ticker_data.get('Data', {})
            timestamp = datetime.now()
            
            for pair in self.trading_pairs:
                if pair in data:
                    price = float(data[pair].get('LastPrice', 0))
                    prices[pair] = price
                    
                    # 添加到历史记录
                    self.price_history[pair].append({
                        'timestamp': timestamp.isoformat(),
                        'price': price
                    })
            
            logger.info(f"[数据收集] 获取价格: {prices}")
        else:
            logger.warning("获取价格失败")
        
        return prices
    
    def get_price_dataframe(self) -> pd.DataFrame:
        """将价格历史转换为DataFrame"""
        if not self.is_ready():
            return pd.DataFrame()
        
        # 转换为DataFrame格式
        dfs = []
        for pair in self.trading_pairs:
            if len(self.price_history[pair]) > 0:
                df = pd.DataFrame(list(self.price_history[pair]))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df.columns = [pair]
                dfs.append(df)
        
        if dfs:
            result = pd.concat(dfs, axis=1)
            return result
        else:
            return pd.DataFrame()
    
    def is_ready(self, min_data_points: int = 30) -> bool:
        """检查是否有足够的数据进行Alpha计算"""
        for pair in self.trading_pairs:
            if len(self.price_history[pair]) < min_data_points:
                logger.info(f"[数据状态] {pair} 数据不足: {len(self.price_history[pair])}/{min_data_points}")
                return False
        return True
    
    def get_data_status(self) -> Dict:
        """获取数据状态"""
        status = {}
        for pair in self.trading_pairs:
            status[pair] = {
                'count': len(self.price_history[pair]),
                'latest': self.price_history[pair][-1] if self.price_history[pair] else None
            }
        return status
    
    def save_history(self):
        """保存历史数据到文件"""
        try:
            data = {pair: list(hist) for pair, hist in self.price_history.items()}
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"[数据保存] 已保存到 {self.data_file}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
    
    def load_history(self):
        """从文件加载历史数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                for pair in self.trading_pairs:
                    if pair in data:
                        self.price_history[pair] = deque(data[pair], maxlen=self.max_history)
                
                logger.info(f"[数据加载] 从 {self.data_file} 加载历史数据")
                logger.info(f"[数据状态] {self.get_data_status()}")
            except Exception as e:
                logger.error(f"加载数据失败: {e}")


class OrderManager:
    """订单管理器 - 处理未成交订单"""
    
    def __init__(self, api_client: RoostooAPIClient, timeout_minutes: int = 5):
        """
        初始化订单管理器
        
        Args:
            api_client: API客户端
            timeout_minutes: 订单超时时间（分钟）
        """
        self.api = api_client
        self.timeout_minutes = timeout_minutes
        self.pending_orders = {}  # {order_id: {pair, side, quantity, timestamp, retry_count}}
    
    def add_order(self, order_id: str, pair: str, side: str, quantity: float):
        """添加订单到追踪列表"""
        self.pending_orders[order_id] = {
            'pair': pair,
            'side': side,
            'quantity': quantity,
            'timestamp': datetime.now(),
            'retry_count': 0
        }
        logger.info(f"[订单追踪] 添加订单 {order_id}: {side} {quantity} {pair}")
    
    def check_and_handle_pending_orders(self):
        """检查并处理所有未成交订单"""
        if not self.pending_orders:
            return
        
        logger.info(f"[订单检查] 检查 {len(self.pending_orders)} 个订单...")
        
        completed_orders = []
        
        for order_id, order_info in self.pending_orders.items():
            # 查询订单状态
            result = self.api.query_order(order_id=order_id)
            
            if not result or not result.get('Success'):
                logger.warning(f"[订单检查] 查询订单 {order_id} 失败")
                continue
            
            orders = result.get('OrderMatched', [])
            if not orders:
                # 检查超时
                elapsed = (datetime.now() - order_info['timestamp']).total_seconds() / 60
                if elapsed > self.timeout_minutes:
                    logger.warning(f"[订单超时] 订单 {order_id} 超时 ({elapsed:.1f}分钟)")
                    self._handle_timeout_order(order_id, order_info)
                continue
            
            order = orders[0]
            status = order.get('Status', '')
            
            if status == 'FILLED':
                logger.info(f"[订单完成] 订单 {order_id} 已成交")
                completed_orders.append(order_id)
                
            elif status == 'PARTIALLY_FILLED':
                filled_qty = float(order.get('ExecutedQty', 0))
                total_qty = float(order.get('Quantity', 0))
                logger.info(f"[订单部分成交] 订单 {order_id}: {filled_qty}/{total_qty}")
                
                # 如果超时，处理剩余部分
                elapsed = (datetime.now() - order_info['timestamp']).total_seconds() / 60
                if elapsed > self.timeout_minutes:
                    self._handle_partial_fill(order_id, order_info, order)
                    
            elif status == 'CANCELED':
                logger.warning(f"[订单取消] 订单 {order_id} 已取消")
                completed_orders.append(order_id)
                
            elif status == 'PENDING':
                elapsed = (datetime.now() - order_info['timestamp']).total_seconds() / 60
                if elapsed > self.timeout_minutes:
                    self._handle_timeout_order(order_id, order_info)
        
        # 清理已完成订单
        for order_id in completed_orders:
            del self.pending_orders[order_id]
    
    def _handle_timeout_order(self, order_id: str, order_info: Dict):
        """处理超时订单"""
        logger.warning(f"[超时处理] 处理订单 {order_id}")
        
        # 取消订单并以市价重新下单
        cancel_result = self.api.cancel_order(order_id=order_id)
        
        if cancel_result and cancel_result.get('Success'):
            logger.info(f"[超时处理] 已取消订单 {order_id}")
            
            # 重新以市价单下单
            if order_info['retry_count'] < 3:  # 最多重试3次
                logger.info(f"[超时处理] 重新下市价单: {order_info['side']} {order_info['quantity']} {order_info['pair']}")
                
                new_order = self.api.place_order(
                    pair=order_info['pair'],
                    side=order_info['side'],
                    quantity=order_info['quantity'],
                    order_type='MARKET'
                )
                
                if new_order and new_order.get('Success'):
                    new_order_id = new_order.get('OrderDetail', {}).get('OrderID')
                    order_info['retry_count'] += 1
                    order_info['timestamp'] = datetime.now()
                    
                    # 更新追踪
                    del self.pending_orders[order_id]
                    self.pending_orders[new_order_id] = order_info
                    logger.info(f"[超时处理] 新订单ID: {new_order_id}")
            else:
                logger.error(f"[超时处理] 订单 {order_id} 重试次数过多，放弃")
                del self.pending_orders[order_id]
    
    def _handle_partial_fill(self, order_id: str, order_info: Dict, order_detail: Dict):
        """处理部分成交订单"""
        filled_qty = float(order_detail.get('ExecutedQty', 0))
        total_qty = float(order_detail.get('Quantity', 0))
        remaining_qty = total_qty - filled_qty
        
        logger.info(f"[部分成交] 订单 {order_id} 剩余 {remaining_qty}")
        
        # 取消原订单
        self.api.cancel_order(order_id=order_id)
        
        # 剩余部分以市价单执行
        if remaining_qty > 0:
            new_order = self.api.place_order(
                pair=order_info['pair'],
                side=order_info['side'],
                quantity=remaining_qty,
                order_type='MARKET'
            )
            
            if new_order and new_order.get('Success'):
                new_order_id = new_order.get('OrderDetail', {}).get('OrderID')
                logger.info(f"[部分成交] 剩余部分新订单: {new_order_id}")
                
                # 更新追踪
                del self.pending_orders[order_id]
                order_info['quantity'] = remaining_qty
                order_info['timestamp'] = datetime.now()
                self.pending_orders[new_order_id] = order_info
    
    def cancel_all_pending(self):
        """取消所有未成交订单"""
        logger.info(f"[强制撤单] 取消所有 {len(self.pending_orders)} 个订单")
        
        for order_id in list(self.pending_orders.keys()):
            self.api.cancel_order(order_id=order_id)
            time.sleep(0.2)
        
        self.pending_orders.clear()


class AlphaLiveTrading:
    """Alpha策略实盘交易系统"""

    def __init__(self, api_client: RoostooAPIClient, trading_pairs: List[str],
                 min_data_points: int = 30, rebalance_interval: int = 60,
                 min_position_value: float = 10.0, max_position_pct: float = 0.3):
        """
        初始化实盘交易系统
        
        Args:
            api_client: API客户端
            trading_pairs: 交易对列表
            min_data_points: 开始交易前需要的最少数据点
            rebalance_interval: 再平衡间隔（分钟）
            min_position_value: 最小换仓区间
            max_position_pct: 单个资产最大持仓比例
        """
        self.api = api_client
        self.trading_pairs = trading_pairs
        self.min_data_points = min_data_points
        self.rebalance_interval = rebalance_interval
        self.min_position_value = min_position_value
        self.max_position_pct = max_position_pct
        
        # 初始化子模块
        self.data_manager = PriceDataManager(api_client, trading_pairs)
        self.order_manager = OrderManager(api_client, timeout_minutes=5)
        
        # 状态
        self.last_rebalance_time = None
        self.current_positions = {}
        self.target_weights = {}
        
    def calculate_alpha_signals(self, price_df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha信号（这里是示例，你需要用自己的Alpha逻辑）
        
        Args:
            price_df: 价格历史数据 DataFrame
            
        Returns:
            Series: 每个交易对的Alpha值
        """
        # 动量+反转+波动率
        
        if len(price_df) >= 20:
            momentum_20 = price_df.pct_change(20).iloc[-1]
        else:
            momentum_20 = pd.Series(0, index=price_df.columns)
        
        if len(price_df) >= 5:
            reversal_5 = -price_df.pct_change(5).iloc[-1]
        else:
            reversal_5 = pd.Series(0, index=price_df.columns)
        
        if len(price_df) >= 20:
            volatility = price_df.pct_change().tail(20).std()
            vol_signal = -volatility 
        else:
            vol_signal = pd.Series(0, index=price_df.columns)
        
        # 组合因子
        alpha = (
            0.5 * momentum_20 +
            0.3 * reversal_5 +
            0.2 * vol_signal
        )
        
        # 标准化
        alpha = (alpha - alpha.mean()) / (alpha.std() + 1e-9)
        
        logger.info(f"[Alpha计算] 信号值: {alpha.to_dict()}")
        return alpha
    
    def get_current_positions(self) -> Dict[str, float]:
        """获取当前持仓"""
        positions = {}
        balance_data = self.api.get_balance()
        
        if balance_data and balance_data.get('Success'):
            wallet = balance_data.get('Wallet', {})
            
            for pair in self.trading_pairs:
                coin = pair.split('/')[0]
                if coin in wallet:
                    free = float(wallet[coin].get('Free', 0))
                    locked = float(wallet[coin].get('Locked', 0))
                    positions[pair] = free + locked
        
        self.current_positions = positions
        return positions
    
    def get_portfolio_value(self) -> float:
        """计算组合总价值"""
        balance_data = self.api.get_balance()
        
        if not balance_data or not balance_data.get('Success'):
            return 0.0
        
        wallet = balance_data.get('Wallet', {})
        total_value = float(wallet.get('USD', {}).get('Free', 0))
        
        current_prices = self.data_manager.fetch_current_prices()
        
        for pair in self.trading_pairs:
            coin = pair.split('/')[0]
            if coin in wallet and pair in current_prices:
                coin_amount = float(wallet[coin].get('Free', 0)) + float(wallet[coin].get('Locked', 0))
                total_value += coin_amount * current_prices[pair]
        
        return total_value
    
    def calculate_target_weights(self, alpha_signals: pd.Series) -> Dict[str, float]:
        """根据Alpha信号计算目标权重"""
        # 只保留正信号（做多策略）
        positive_signals = alpha_signals[alpha_signals > 0]
        
        if len(positive_signals) == 0:
            logger.warning("[权重计算] 无正信号，全部现金")
            return {pair: 0.0 for pair in self.trading_pairs}
        
        # 按信号强度分配权重
        weights = positive_signals / positive_signals.sum()
        
        # 限制单资产最大权重
        weights = weights.clip(upper=self.max_position_pct)
        
        # 重新归一化
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        target_weights = {pair: weights.get(pair, 0.0) for pair in self.trading_pairs}
        
        self.target_weights = target_weights
        logger.info(f"[权重计算] 目标权重: {target_weights}")
        return target_weights
    
    def execute_rebalance(self):
        """执行再平衡"""
        logger.info("=" * 70)
        logger.info("[再平衡] 开始执行...")
        
        # 检查并处理未成交订单
        self.order_manager.check_and_handle_pending_orders()
        
        # 获取价格数据
        price_df = self.data_manager.get_price_dataframe()
        if price_df.empty:
            logger.warning("[再平衡] 价格数据不足，跳过")
            return
        
        logger.info(f"[DEBUG] price_df.tail():\n{price_df.tail()}")

        # 计算Alpha信号
        alpha_signals = self.calculate_alpha_signals(price_df)
        logger.info(f"[DEBUG] Alpha 信号: {alpha_signals.to_dict()}")
        
        # 计算目标权重
        self.calculate_target_weights(alpha_signals)
        logger.info(f"[DEBUG] 目标权重: {self.target_weights}")
        
        # 获取当前状态
        positions = self.get_current_positions()
        logger.info(f"[DEBUG] 当前持仓(数量): {positions}")
        portfolio_value = self.get_portfolio_value()
        logger.info(f"[DEBUG] 组合总价值(USD): {portfolio_value}")

        current_prices = self.data_manager.fetch_current_prices()
        logger.info(f"[DEBUG] 当前价格: {current_prices}")
        
        # 计算并执行交易
        for pair in self.trading_pairs:
            target_weight = self.target_weights.get(pair, 0.0)
            target_value = portfolio_value * target_weight
            
            current_qty = self.current_positions.get(pair, 0.0)
            current_price = current_prices.get(pair, 0.0)
            current_value = current_qty * current_price
            
            value_diff = target_value - current_value

            logger.info(
                f"[DEBUG][{pair}] target_weight={target_weight:.4f}, "
                f"target_value={target_value:.4f}, current_qty={current_qty}, "
                f"current_price={current_price}, current_value={current_value:.4f}, "
                f"value_diff={value_diff:.4f}"
            )
            
            # 如果差异太小，跳过
            if abs(value_diff) < self.min_position_value:
                logger.info(
                    f"[再平衡][{pair}] value_diff={value_diff:.4f} "
                    f"< min_position_value={self.min_position_value}，跳过下单"
                )
                continue
            
            # 计算交易数量
            trade_qty = abs(value_diff) / current_price if current_price > 0 else 0
            trade_qty = round(trade_qty, 6)  # 保留6位小数
            logger.info(f"[DEBUG][{pair}] 计算得到 trade_qty={trade_qty}")
            
            if trade_qty == 0:
                logger.info(f"[再平衡][{pair}] trade_qty 为 0，跳过")
                continue
            
            action = 'BUY' if value_diff > 0 else 'SELL'
            logger.info(f"[再平衡] {pair}: {action} {trade_qty}")
            logger.info(
                f"[DEBUG][{pair}] 准备下单: pair={pair}, side={action}, "
                f"quantity={trade_qty}, type=MARKET"
            )

            # 下单
            order_result = self.api.place_order(
                pair=pair,
                side=action,
                quantity=trade_qty,
                order_type='MARKET'
            )
            
            logger.info(f"[DEBUG][{pair}] 下单返回: {order_result}")
            
            if order_result and order_result.get('Success'):
                order_id = order_result.get('OrderDetail', {}).get('OrderID')
                logger.info(f"[再平衡] 订单成功: {order_id}")
                
                # 添加到订单管理器
                self.order_manager.add_order(order_id, pair, action, trade_qty)
            else:
                logger.error(f"[再平衡] 订单失败: {order_result.get('ErrMsg') if order_result else 'Unknown'}")
            
            time.sleep(0.5)  # 避免API限流
        
        self.last_rebalance_time = datetime.now()
        logger.info("[再平衡] 完成")
        logger.info("=" * 70)
    
    def should_rebalance(self) -> bool:
        """判断是否应该再平衡"""
        if self.last_rebalance_time is None:
            return True
        
        elapsed_minutes = (datetime.now() - self.last_rebalance_time).total_seconds() / 60
        return elapsed_minutes >= self.rebalance_interval
    
    def run_forever(self, data_collection_interval: int = 60):
        """
        持续运行交易系统
        
        Args:
            data_collection_interval: 数据收集间隔（秒）
        """
        logger.info("=" * 70)
        logger.info("启动Alpha实盘交易系统")
        logger.info(f"交易对: {self.trading_pairs}")
        logger.info(f"数据收集间隔: {data_collection_interval}秒")
        logger.info(f"再平衡间隔: {self.rebalance_interval}分钟")
        logger.info(f"最少数据点: {self.min_data_points}")
        logger.info("=" * 70)
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                logger.info(f"\n[迭代 {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 收集价格数据
                self.data_manager.fetch_current_prices()
                
                # 检查数据是否充足
                if not self.data_manager.is_ready(self.min_data_points):
                    status = self.data_manager.get_data_status()
                    logger.info(f"[等待数据] 数据收集中... {status}")
                else:
                    # 检查是否需要再平衡
                    if self.should_rebalance():
                        self.execute_rebalance()
                    else:
                        elapsed = (datetime.now() - self.last_rebalance_time).total_seconds() / 60
                        logger.info(f"[等待再平衡] 距离下次再平衡还有 {self.rebalance_interval - elapsed:.1f} 分钟")
                    
                    # 定期检查未成交订单（每5分钟）
                    if iteration % 5 == 0:
                        self.order_manager.check_and_handle_pending_orders()
                
                # 定期保存数据（每10分钟）
                if iteration % 10 == 0:
                    self.data_manager.save_history()
                
                time.sleep(data_collection_interval)
                
        except KeyboardInterrupt:
            logger.info("\n收到中断信号，正在安全退出...")
            self.data_manager.save_history()
            logger.info("已保存数据，系统退出")
        except Exception as e:
            logger.error(f"系统错误: {e}", exc_info=True)
            self.data_manager.save_history()
            raise


if __name__ == "__main__":

    # API配置
    API_KEY = "w2bR9XU4g6eN8qT1jY0LzA7cD3fV5sK2rC1mF8hJ9pQ4uB6vW3oP5xI7lS0nM2tY"
    SECRET_KEY = "p7LwX3gH1qV8yJ4bS0nK6tF2zU9mR5oC8dA1sI3vW7eN6lP4xT0jZ9fB2kY5hM"
    api_client = RoostooAPIClient(API_KEY, SECRET_KEY)
    
    # 交易对
    trading_pairs = ["BTC/USD", "ETH/USD"]  # 可以添加更多交易对
    
    live_trading = AlphaLiveTrading(
        api_client=api_client,
        trading_pairs=trading_pairs,
        min_data_points=30,          # 至少30个数据点才开始交易
        rebalance_interval=60,       # 60分钟再平衡一次
        min_position_value=10.0,     # 最小交易10美元
        max_position_pct=0.3         # 单个资产最大30%
    )
    
    # 每60秒收集一次数据+运行策略
    live_trading.run_forever(data_collection_interval=60)
