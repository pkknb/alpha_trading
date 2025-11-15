import requests
import time
import hmac
import hashlib

BASE_URL = "https://mock-api.roostoo.com"
API_KEY = "w2bR9XU4g6eN8qT1jY0LzA7cD3fV5sK2rC1mF8hJ9pQ4uB6vW3oP5xI7lS0nM2tY"
SECRET_KEY = "p7LwX3gH1qV8yJ4bS0nK6tF2zU9mR5oC8dA1sI3vW7eN6lP4xT0jZ9fB2kY5hM"
# --------------------------------------------------------
# 加密API签名
def _get_timestamp():
    """Returns a 13-digit millisecond timestamp as a string."""
    return str(int(time.time() * 1000))

def _get_signed_headers(payload={}):
    """
    Creates a signature for a given payload (dict) and returns
    the correct headers for a SIGNED (RCL_TopLevelCheck) request.
    """
    # 1. Add timestamp to the payload
    payload['timestamp'] = _get_timestamp()
    
    # 2. Sort keys and create the totalParams string
    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{key}={payload[key]}" for key in sorted_keys)
    
    # 3. Create HMAC-SHA256 signature
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        total_params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # 4. Create headers
    headers = {
        'RST-API-KEY': API_KEY,
        'MSG-SIGNATURE': signature
    }
    
    return headers, payload, total_params

# --- Now we can define functions for each API call ---
# 账户资金
def get_balance():
    """Gets account balance. (Auth: RCL_TopLevelCheck)"""
    url = f"{BASE_URL}/v3/balance"
    
    # 1. Get signed headers and the payload (which now includes timestamp)
    # For a GET request with no params, the payload is just the timestamp
    headers, payload, total_params_string = _get_signed_headers(payload={})
    
    try:
        # 2. Send the request
        # In a GET request, the payload is sent as 'params'
        response = requests.get(url, headers=headers, params=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting balance: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None

if __name__ == "__main__":
    print("--- Getting Balance ---")
    balance = get_balance()
    if balance and balance.get('Success'):
        print(f"USD Free: {balance.get('Wallet', {}).get('USD', {}).get('Free')}")
    elif balance:
        print(f"Error: {balance.get('ErrMsg')}")