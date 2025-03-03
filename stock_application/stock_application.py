import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ここにメール送信関数を追加
def send_email(subject, body, to_email="kyogoliver@gmail.com"):
    from_email = "kyogoliver@gmail.com"  
    from_password = "zrgc awbb mryu xppp" 

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, from_password)
            server.sendmail(from_email, to_email, msg.as_string())
        print("メールが送信されました。")
    except Exception as e:
        print(f"メール送信エラー: {e}")

def fetch_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    if data.empty:
        raise ValueError(f"No data found for {ticker} with period {period}")
    return data

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(data):
    data["MA_5"] = data["Close"].rolling(window=5).mean()
    data["MA_20"] = data["Close"].rolling(window=20).mean()
    data["Volatility"] = data["Close"].rolling(window=20).std()
    data["RSI"] = compute_rsi(data["Close"], 14)
    data["Momentum"] = data["Close"].diff(10)
    data["Volume_Change"] = data["Volume"].pct_change()
    return data.dropna()

def build_and_evaluate_model(X, y):
    if len(X) < 5:
        print("Insufficient data for training.")
        return None
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.5]
    }
    
    model = XGBRegressor(random_state=42)
    cv_splits = min(5, len(X))
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_splits, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def decide_trading_action(rsi, volatility, ma_5, ma_20, momentum):
    if rsi < 25 and ma_5 > ma_20 and momentum > 0:
        action = "買い"
        stock_count = np.random.randint(1, 4)
    elif rsi > 80 and ma_5 < ma_20 and momentum < 0:
        action = "売り"
        stock_count = np.random.randint(1, 4)
    else:
        action = "保留"
        stock_count = 0
    
    if action != "保留" and volatility > 2:
        stock_count = max(1, stock_count)

    return action, stock_count

def main():
    ticker = "8267.T"
    print(f"==== イオン ({ticker}) の株価予測 ====")
    data = fetch_stock_data(ticker)
    data = add_features(data)
    
    X = data[["MA_5", "MA_20", "Volatility", "RSI", "Momentum", "Volume_Change"]]
    y = data["Close"].shift(-1).dropna()
    X = X[:-1]
    
    model = build_and_evaluate_model(X, y)
    if model is None:
        return
    
    latest_data = data.tail(1)
    rsi = latest_data["RSI"].values[-1]
    volatility = latest_data["Volatility"].values[-1]
    ma_5 = latest_data["MA_5"].values[-1]
    ma_20 = latest_data["MA_20"].values[-1]
    momentum = latest_data["Momentum"].values[-1]
    latest_price = latest_data["Close"].values[-1]
    latest_date = latest_data.index[-1].strftime('%Y-%m-%d')
    
    action, stock_count = decide_trading_action(rsi, volatility, ma_5, ma_20, momentum)
    
    result = f"""
    最新の株価 ({latest_date}): {latest_price:.2f}円
    最新のRSI: {rsi:.2f}
    最新のボラティリティ: {volatility:.2f}
    最新の短期MA: {ma_5:.2f}, 長期MA: {ma_20:.2f}
    最新のモメンタム: {momentum:.2f}
    推奨アクション: {action}
    推奨株数: {stock_count}株
    データ更新日: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    print(result)
    
    # メール送信
    send_email("イオン株価予測結果", result)

if __name__ == "__main__":
    main()








