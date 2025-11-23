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
                 max_history: int = 2000, data_file: str = "price_history.json",
                 rebalance_freq: str = "10min"):  # ← 新增
        """
        初始化数据管理器
        
        Args:
            api_client: API客户端
            trading_pairs: 交易对列表
            max_history: 最大保存历史数据条数（默认2000=33小时）
            data_file: 数据持久化文件
            rebalance_freq: 重采样频率（如 "10min"）
        """
        self.api = api_client
        self.trading_pairs = trading_pairs
        self.max_history = max_history
        self.data_file = data_file
        self.rebalance_freq = rebalance_freq
        
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
    
    def get_price_dataframe(self, apply_downsample: bool = True) -> pd.DataFrame:
        """
        将价格历史转换为DataFrame，并应用降采样
        """
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
        
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, axis=1)
        
        # ★ 应用降采样 - 与回测保持一致
        if apply_downsample and self.rebalance_freq != "1min":
            original_len = len(result)
            result = self._downsample_price(result)
            logger.info(
                f"[数据降采样] {self.rebalance_freq}: "
                f"{original_len}条 → {len(result)}条"
            )
        
        return result
    
    def _downsample_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对价格数据进行降采样
        与回测中的 price_downsample_mode="last" 保持一致（取最后价格=收盘价）
        """
        if self.rebalance_freq == "1min":
            return df
        
        # 使用 last 方法（取收盘价）- 与回测一致
        resampled = df.resample(self.rebalance_freq).last()
        
        # 移除全NaN行
        resampled = resampled.dropna(how='all')
        
        return resampled
    
    def is_ready(self, min_data_points: int = 30) -> bool:
        """检查是否有足够的数据进行Alpha计算"""
        for pair in self.trading_pairs:
            if len(self.price_history[pair]) < min_data_points:
                logger.info(
                    f"[数据状态] {pair} 数据不足: "
                    f"{len(self.price_history[pair])}/{min_data_points}"
                )
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
                 min_position_value: float = 10.0, max_position_pct: float = 0.3,
                 capital_usage_pct: float = 0.1,
                 rebalance_freq: str = "10min",          # ← 新增
                 commission_rate: float = 0.001,         # ← 新增
                 max_turnover_rate: float = 0.0001,      # ← 新增
                 turnover_threshold: float = 0.1):       # ← 新增
        """
        初始化实盘交易系统
        
        Args:
            api_client: API客户端
            trading_pairs: 交易对列表
            min_data_points: 开始交易前需要的最少原始数据点（分钟）
            rebalance_interval: 再平衡间隔（分钟）
            min_position_value: 最小持仓价值（USD）
            max_position_pct: 单个资产最大持仓比例
            capital_usage_pct: 资金使用比例
            rebalance_freq: 重采样频率（如 "10min"）- 与回测保持一致
            commission_rate: 手续费率（如 0.001 = 0.1%）
            max_turnover_rate: 最大换手率（如 0.0001 = 0.01%）
            turnover_threshold: 换手率阈值（如 0.1 = 10%）
        """
        self.api = api_client
        self.trading_pairs = trading_pairs
        self.min_data_points = min_data_points
        self.rebalance_interval = rebalance_interval
        self.min_position_value = min_position_value
        self.max_position_pct = max_position_pct
        self.capital_usage_pct = capital_usage_pct
        self.rebalance_freq = rebalance_freq              # ← 新增
        self.commission_rate = commission_rate            # ← 新增
        self.max_turnover_rate = max_turnover_rate        # ← 新增
        self.turnover_threshold = turnover_threshold      # ← 新增
        
        # 初始化子模块
        self.data_manager = PriceDataManager(
            api_client, 
            trading_pairs,
            rebalance_freq=rebalance_freq  # ← 传入降采样参数
        )
        self.order_manager = OrderManager(api_client, timeout_minutes=5)
        
        # 状态
        self.last_rebalance_time = None
        self.current_positions = {}
        self.target_weights = {}
        self.last_weights = {pair: 0.0 for pair in trading_pairs}
        self._initialize_last_weights()
    
    def _initialize_last_weights(self):
        """从当前持仓初始化last_weights"""
        positions = self.get_current_positions()
        portfolio_value = self.get_portfolio_value()
        
        if portfolio_value > 0:
            current_prices = self.data_manager.fetch_current_prices()
            for pair in self.trading_pairs:
                qty = positions.get(pair, 0)
                price = current_prices.get(pair, 0)
                value = qty * price
                self.last_weights[pair] = value / portfolio_value
            
            logger.info(f"[初始化] 从当前持仓计算的权重: {self.last_weights}")
        else:
            self.last_weights = {pair: 0.0 for pair in self.trading_pairs}
            logger.info("[初始化] 无持仓，权重初始化为0")
    def calculate_alpha_signals(self, price_df: pd.DataFrame) -> pd.Series:
        """
        计算Alpha信号
        """
        try:
            import torch
            import sys
            sys.path.append('.')
            import AlphaOperation as op
        except ImportError as e:
            logger.error(f"导入模块失败: {e}")
            return pd.Series(0, index=price_df.columns)
        
        logger.info(
            f"[Alpha计算] 输入数据 - 形状: {price_df.shape}, "
            f"时间范围: {price_df.index[0]} 到 {price_df.index[-1]}"
        )
        
        # 转换为torch tensor
        price_tensor = torch.tensor(price_df.values, dtype=torch.float32)
        
        # 计算log returns
        log_ret_tensor = op.log(op.div(price_tensor, op.ts_delay(price_tensor, 1)))
        
        # 应用alpha因子
        # 窗口参数基于降采样后的频率：
        # - ts_ewma(30): 30个10分钟 = 5小时
        # - ts_mean(15): 15个10分钟 = 2.5小时  
        # - ts_decay_linear(50): 50个10分钟 = 8.3小时
        alpha_tensor = -op.ts_decay_linear(
            (op.ts_ewma(log_ret_tensor, 30) + op.ts_mean(log_ret_tensor, 15)), 
            50
        )
        
        # 取最后一行作为当前信号
        alpha_values = alpha_tensor[-1].numpy()
        alpha_series = pd.Series(alpha_values, index=price_df.columns)
        
        # 处理NaN值
        alpha_series = alpha_series.fillna(0)
        
        # 检查异常值
        if alpha_series.isna().all():
            logger.warning("[Alpha计算] 所有信号为NaN，返回零信号")
            return pd.Series(0, index=price_df.columns)
        
        if np.isinf(alpha_series).any():
            logger.warning("[Alpha计算] 存在无穷大值，替换为0")
            alpha_series = alpha_series.replace([np.inf, -np.inf], 0)
        
        logger.info(f"[Alpha计算] 信号值: {alpha_series.to_dict()}")
        return alpha_series
    
    def get_current_positions(self) -> Dict[str, float]:
        """获取当前持仓"""
        positions = {}
        balance_data = self.api.get_balance()
        
        if balance_data and balance_data.get('Success'):
            wallet = balance_data.get('SpotWallet', {})
            
            for pair in self.trading_pairs:
                coin = pair.split('/')[0]
                if coin in wallet:
                    free = float(wallet[coin].get('Free', 0))
                    locked = float(wallet[coin].get('Locked', 0))
                    positions[pair] = free + locked
        
        self.current_positions = positions
        return positions
    
    def get_portfolio_value(self) -> float:
        """计算组合总价值（包括冻结资金）"""
        balance_data = self.api.get_balance()
        
        if not balance_data or not balance_data.get('Success'):
            return 0.0
        
        wallet = balance_data.get('SpotWallet', {})
        
        # ★ 同时考虑Free和Locked
        usd_free = float(wallet.get('USD', {}).get('Free', 0))
        usd_locked = float(wallet.get('USD', {}).get('Locked', 0))
        total_value = usd_free + usd_locked
        
        current_prices = self.data_manager.fetch_current_prices()
        
        for pair in self.trading_pairs:
            coin = pair.split('/')[0]
            if coin in wallet and pair in current_prices:
                coin_free = float(wallet[coin].get('Free', 0))
                coin_locked = float(wallet[coin].get('Locked', 0))
                coin_amount = coin_free + coin_locked  # ★ 包含Locked
                total_value += coin_amount * current_prices[pair]
        
        logger.info(f"[组合价值] Free=${usd_free:.2f}, Locked=${usd_locked:.2f}, Total=${total_value:.2f}")
        return total_value
    
    def calculate_target_weights(self, alpha_signals: pd.Series) -> Dict[str, float]:
        """
        根据Alpha信号计算目标权重，并应用换手率控制
        完全复刻回测逻辑
        """
        # ★ Step 1: 只保留正信号（做多策略）- 与回测的_normalize_long_only一致
        positive_signals = alpha_signals.clip(lower=0)
        
        if positive_signals.sum() == 0:
            logger.warning("[权重计算] 无正信号，全部现金")
            raw_weights = {pair: 0.0 for pair in self.trading_pairs}
        else:
            # 标准化到和为1
            weights = positive_signals / positive_signals.sum()
            
            # 限制单资产最大权重
            weights = weights.clip(upper=self.max_position_pct)
            
            # 重新归一化
            if weights.sum() > 0:
                weights = weights / weights.sum()
            
            raw_weights = {pair: weights.get(pair, 0.0) for pair in self.trading_pairs}
        
        logger.info(f"[权重计算] 原始目标权重: {raw_weights}")
        
        # ★ Step 2: 应用换手率控制 - 与回测的TurnoverControl保持一致
        # 计算权重变化（turnover）
        weight_changes = {
            pair: abs(raw_weights[pair] - self.last_weights[pair]) 
            for pair in self.trading_pairs
        }
        total_turnover = sum(weight_changes.values())
        
        logger.info(f"[换手率控制] 计算换手率: {total_turnover:.6f}")
        
        # threshold方法：如果turnover超过阈值，则限制变化
        if total_turnover > self.turnover_threshold:
            logger.warning(
                f"[换手率控制] 换手率 {total_turnover:.6f} 超过阈值 {self.turnover_threshold}"
            )
            
            # 限制到max_turnover_rate
            if total_turnover > self.max_turnover_rate:
                scale_factor = self.max_turnover_rate / total_turnover
                logger.warning(
                    f"[换手率控制] 缩减至 {self.max_turnover_rate:.6f}，"
                    f"缩放系数: {scale_factor:.4f}"
                )
                
                # 缩减权重变化
                final_weights = {}
                for pair in self.trading_pairs:
                    last_w = self.last_weights[pair]
                    target_w = raw_weights[pair]
                    final_weights[pair] = last_w + (target_w - last_w) * scale_factor
                
                self.target_weights = final_weights
            else:
                self.target_weights = raw_weights
        else:
            self.target_weights = raw_weights
        
        logger.info(f"[换手率控制] 最终目标权重: {self.target_weights}")
        
        # 更新last_weights为当前target_weights（在实际交易执行后）
        # 注意：这里先不更新，等execute_rebalance结束后再更新
        
        return self.target_weights
    
    def execute_rebalance(self):
        """执行再平衡"""
        logger.info("=" * 70)
        logger.info("[再平衡] 开始执行...")
        
        # 检查并处理未成交订单
        self.order_manager.check_and_handle_pending_orders()
        
        # 获取价格数据
        price_df = self.data_manager.get_price_dataframe(apply_downsample=True)
        if price_df.empty:
            logger.warning("[再平衡] 价格数据不足，跳过")
            return
        
        min_required = 50
        if len(price_df) < min_required:
            logger.warning(
                f"[再平衡] 降采样后数据不足 ({len(price_df)}/{min_required})，"
                f"需要至少 {min_required} 个 {self.rebalance_freq} 的数据点"
            )
            return
        
        logger.info(
            f"[数据状态] 降采样后数据: {len(price_df)} 个 {self.rebalance_freq} K线"
        )
        logger.info(f"[DEBUG] price_df.tail(3):\n{price_df.tail(3)}")
        
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

        tradable_value = portfolio_value * self.capital_usage_pct
        logger.info(f"[DEBUG] 用于交易的资金(USD): {tradable_value}")

        current_prices = self.data_manager.fetch_current_prices()
        logger.info(f"[DEBUG] 当前价格: {current_prices}")
        
        # 计算并执行交易

        executed_trades = []

        for pair in self.trading_pairs:
            target_weight = self.target_weights.get(pair, 0.0)
            target_value = tradable_value * target_weight
            
            current_qty = self.current_positions.get(pair, 0.0)
            current_price = current_prices.get(pair, 0.0)
            current_value = current_qty * current_price
            
            value_diff = target_value - current_value

            logger.info(
                f"[交易计划][{pair}] "
                f"目标权重={target_weight:.4f}, 目标价值=${target_value:.2f}, "
                f"当前数量={current_qty:.6f}, 当前价值=${current_value:.2f}, "
                f"价值差=${value_diff:.2f}"
            )
            
            # 如果差异太小，跳过
            if abs(value_diff) < self.min_position_value:
                logger.info(
                    f"[再平衡][{pair}] value_diff={value_diff:.4f} "
                    f"< min_position_value={self.min_position_value}，跳过下单"
                )
                continue

            estimated_commission = abs(value_diff) * self.commission_rate
            effective_value_change = abs(value_diff) - estimated_commission

            logger.info(
                f"[交易计划][{pair}] 价值差=${value_diff:.2f}, "
                f"预估手续费=${estimated_commission:.2f}, "
                f"净价值变化=${effective_value_change:.2f}"
            )

            # 用净价值变化判断是否交易
            if effective_value_change < self.min_position_value:
                logger.info(
                    f"[交易执行][{pair}] 扣除手续费后净价值变化 ${effective_value_change:.2f} "
                    f"< 最小阈值 ${self.min_position_value}，跳过"
                )
                continue
            
            # 计算交易数量
            trade_qty = abs(value_diff) / current_price if current_price > 0 else 0
            trade_qty = round(trade_qty, 2)  # 保留2位小数
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
                executed_trades.append((pair, action, trade_qty))
            else:
                error_msg = order_result.get('ErrMsg') if order_result else 'Unknown'
                logger.error(f"[交易执行][{pair}] ✗ 订单失败: {error_msg}")  
            
            time.sleep(0.5)  # 避免API限流

        self.last_weights = self.target_weights.copy()
        logger.info(f"[权重更新] 已更新上次权重: {self.last_weights}")

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

    try:
        import torch
        import AlphaOperation as op
        logger.info("✓ PyTorch 和 AlphaOperation 模块已加载")
        logger.info(f"  PyTorch版本: {torch.__version__}")
    except ImportError as e:
        logger.error(f"✗ 缺少必要模块: {e}")
        logger.error("  请确保已安装 PyTorch 并且 AlphaOperation.py 在当前目录")
        exit(1)


    # API配置
    API_KEY = "w2bR9XU4g6eN8qT1jY0LzA7cD3fV5sK2rC1mF8hJ9pQ4uB6vW3oP5xI7lS0nM2tY"
    SECRET_KEY = "p7LwX3gH1qV8yJ4bS0nK6tF2zU9mR5oC8dA1sI3vW7eN6lP4xT0jZ9fB2kY5hM"
    api_client = RoostooAPIClient(API_KEY, SECRET_KEY)
    
    # 交易对
    trading_pairs = ["BTC/USD","ETH/USD","BNB/USD","XRP/USD","DOGE/USD","SOL/USD","ARB/USD",] 
    
    live_trading = AlphaLiveTrading(
        api_client=api_client,
        trading_pairs=trading_pairs,
        
        min_data_points=200,
        
        # 再平衡间隔：10分钟检查一次
        rebalance_interval=10,
        
        # 位置管理
        min_position_value=10.0,      # 最小交易$10
        max_position_pct=0.3,         # 单资产最大30%
        capital_usage_pct=0.1,        # 使用10%资金
        
        # ★ 关键配置 - 与回测对齐
        rebalance_freq="10min",       # 降采样到10分钟
        commission_rate=0.001,        # 0.1% 手续费
        max_turnover_rate=0.0001,     # 0.01% 最大换手率
        turnover_threshold=0.1        # 10% 换手率阈值
    )
    
    # ★ Step 5: 启动信息
    logger.info("=" * 70)
    logger.info("🚀 Alpha实盘交易系统启动")
    logger.info("=" * 70)
    logger.info(f"📊 交易对: {trading_pairs}")
    logger.info(f"⏱️  数据收集: 每60秒一次（1分钟原始数据）")
    logger.info(f"📉 降采样: {live_trading.rebalance_freq}")
    logger.info(f"🔄 再平衡: 每{live_trading.rebalance_interval}分钟检查一次")
    logger.info(f"📏 最少数据: {live_trading.min_data_points}分钟原始数据")
    logger.info(f"💰 资金使用: {live_trading.capital_usage_pct*100}%")
    logger.info(f"🎯 换手率限制: {live_trading.max_turnover_rate*100}%")
    logger.info(f"⚠️  换手率阈值: {live_trading.turnover_threshold*100}%")
    logger.info("=" * 70)
    
    # ★ Step 6: 运行系统
    # 每60秒收集一次1分钟数据
    # 系统会自动降采样到10分钟，然后计算alpha
    live_trading.run_forever(data_collection_interval=60)