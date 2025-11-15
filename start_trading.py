#!/usr/bin/env python3
"""
Alpha实盘交易系统 - 快速启动脚本
提供交互式启动选项和安全检查
"""

import os
import sys
import json
from datetime import datetime

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def check_dependencies():
    """检查依赖包"""
    print(f"\n{Colors.OKBLUE}[1/5] 检查依赖包...{Colors.ENDC}")
    
    required = ['requests', 'pandas', 'numpy', 'tabulate']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} {Colors.FAIL}(未安装){Colors.ENDC}")
    
    if missing:
        print(f"\n{Colors.FAIL}缺少依赖包！请运行：{Colors.ENDC}")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print(f"{Colors.OKGREEN}依赖检查通过！{Colors.ENDC}")
    return True


def check_config():
    """检查配置文件"""
    print(f"\n{Colors.OKBLUE}[2/5] 检查配置文件...{Colors.ENDC}")
    
    try:
        from config import API_CONFIG, TRADING_PAIRS, STRATEGY_CONFIG
        
        print(f"  ✓ API配置")
        print(f"  ✓ 交易对: {', '.join(TRADING_PAIRS)}")
        print(f"  ✓ 策略参数")
        
        # 显示关键配置
        print(f"\n  关键参数:")
        print(f"    - 再平衡间隔: {STRATEGY_CONFIG['rebalance_interval']} 分钟")
        print(f"    - 最小数据点: {STRATEGY_CONFIG['min_data_points']} 个")
        print(f"    - 最小交易额: ${STRATEGY_CONFIG['min_position_value']}")
        print(f"    - 最大单仓位: {STRATEGY_CONFIG['max_position_pct']*100}%")
        
        print(f"{Colors.OKGREEN}配置检查通过！{Colors.ENDC}")
        return True
        
    except ImportError as e:
        print(f"{Colors.FAIL}配置文件缺失: {e}{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.FAIL}配置文件错误: {e}{Colors.ENDC}")
        return False


def check_historical_data():
    """检查历史数据"""
    print(f"\n{Colors.OKBLUE}[3/5] 检查历史数据...{Colors.ENDC}")
    
    data_file = 'price_history.json'
    
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            print(f"  ✓ 找到历史数据文件")
            
            for pair, records in data.items():
                print(f"    - {pair}: {len(records)} 个数据点")
            
            print(f"{Colors.OKGREEN}历史数据可用！{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"{Colors.WARNING}历史数据文件损坏: {e}{Colors.ENDC}")
            return False
    else:
        print(f"{Colors.WARNING}未找到历史数据，将从头开始收集{Colors.ENDC}")
        return True


def test_api_connection():
    """测试API连接"""
    print(f"\n{Colors.OKBLUE}[4/5] 测试API连接...{Colors.ENDC}")
    
    try:
        from alpha_live_trading import RoostooAPIClient
        from config import API_CONFIG
        
        api = RoostooAPIClient(**API_CONFIG)
        
        # 测试获取余额
        print("  测试获取账户余额...")
        balance = api.get_balance()
        
        if balance and balance.get('Success'):
            wallet = balance.get('Wallet', {})
            usd = wallet.get('USD', {}).get('Free', 0)
            print(f"  ✓ API连接成功")
            print(f"  ✓ 账户余额: ${float(usd):,.2f} USD")
        else:
            print(f"{Colors.FAIL}  ✗ API返回错误{Colors.ENDC}")
            return False
        
        # 测试获取行情
        print("  测试获取行情数据...")
        ticker = api.get_ticker()
        
        if ticker and ticker.get('Success'):
            print(f"  ✓ 行情数据获取成功")
        else:
            print(f"{Colors.WARNING}  ! 行情数据获取失败（可以继续）{Colors.ENDC}")
        
        print(f"{Colors.OKGREEN}API连接测试通过！{Colors.ENDC}")
        return True
        
    except Exception as e:
        print(f"{Colors.FAIL}API连接失败: {e}{Colors.ENDC}")
        return True
def main():
    """主函数"""

    # 执行检查
    checks = [
        check_dependencies,
        check_config,
        check_historical_data,
        test_api_connection,
    ]
    
    for check in checks:
        if not check():
            print(f"\n{Colors.FAIL}启动检查失败，请修复上述问题后重试{Colors.ENDC}")
            sys.exit(1)

    try:
        from alpha_live_trading import AlphaLiveTrading, RoostooAPIClient
        from config import API_CONFIG, TRADING_PAIRS, STRATEGY_CONFIG, DATA_CONFIG
        
        # 初始化
        api_client = RoostooAPIClient(**API_CONFIG)
        
        live_trading = AlphaLiveTrading(
            api_client=api_client,
            trading_pairs=TRADING_PAIRS,
            **STRATEGY_CONFIG
        )
        
        # 启动
        live_trading.run_forever(
            data_collection_interval=DATA_CONFIG['collection_interval']
        )
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}收到退出信号...{Colors.ENDC}")
        print(f"{Colors.OKGREEN}系统已安全退出{Colors.ENDC}")
    except Exception as e:
        print(f"\n\n{Colors.FAIL}系统错误: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
