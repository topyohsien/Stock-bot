import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Telegram 設定
BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "8321686170:AAGqyD9TSLBtw3B5XV0Z9r8ShMEgiYidv-E"
)
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "您的_CHAT_ID")


def send_telegram_msg(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    # Telegram 單則訊息字數限制 4096 字，若太長拆段發送
    if len(message) > 4000:
        chunks = [message[i : i + 4000] for i in range(0, len(message), 4000)]
        for chunk in chunks:
            payload = {
                "chat_id": CHAT_ID,
                "text": chunk,
                "parse_mode": "HTML",
            }
            requests.post(url, json=payload)
    else:
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"Telegram 發送失敗: {e}")


def run_strategy(target_input):
    # 自動處理代碼格式
    if target_input.startswith("^"):
        stock_id = target_input
        ticker = target_input
        stock_name = "台灣加權指數"
    elif target_input == "2330":
        stock_id = target_input
        ticker = f"{target_input}.TW"
        stock_name = "台積電"
    else:
        stock_id = target_input
        ticker = (
            f"{target_input}.TW"
            if not target_input.endswith(".TW")
            else target_input
        )
        stock_name = target_input

    # 抓取歷史資料
    df = yf.download(
        ticker, start="2018-01-01", auto_adjust=False, interval="1d"
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"]).reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    # 技術指標計算
    df["Returns"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Night_Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(
        1
    )

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    df["ATR_Pct"] = df["ATR14"] / df["Close"]

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["OSC"] = (ema12 - ema26) - (ema12 - ema26).ewm(
        span=9, adjust=False
    ).mean()

    df["STD20"] = df["Close"].rolling(20).std()
    df["BB_Width"] = (4 * df["STD20"]) / df["MA20"]

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))
    df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()

    df["Retail_Overheat"] = (df["RSI"] / 100) * df["Vol_Ratio"]
    df["Retail_Panic"] = ((100 - df["RSI"]) / 100) * (
        df["Returns"].clip(upper=0).abs()
    )
    df["Target"] = (df["Close"].shift(-5) > df["Close"] * 1.01).astype(int)
    df_clean = df.dropna().copy().reset_index(drop=True)

    # 風控參數
    if stock_id == "^TWII":
        take_profit, stop_loss, max_hold_days = 0.03, -0.02, 7
    elif stock_id == "2330":
        take_profit, stop_loss, max_hold_days = 0.04, -0.025, 7
    else:
        take_profit, stop_loss, max_hold_days = 0.03, -0.02, 7

    # 模型訓練
    train_mask = (df_clean["Date"] >= "2018-01-01") & (
        df_clean["Date"] < "2024-01-01"
    )
    test_mask = df_clean["Date"] >= "2024-01-01"

    train_df = df_clean[train_mask].copy()
    test_df = df_clean[test_mask].copy().reset_index(drop=True)

    features = [
        "Returns",
        "Night_Gap",
        "BB_Width",
        "Vol_Ratio",
        "RSI",
        "OSC",
        "Retail_Overheat",
        "Retail_Panic",
    ]
    model = RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )
    model.fit(train_df[features], train_df["Target"])

    probs = model.predict_proba(test_df[features])[:, 1]
    buy_signal = probs > 0.50

    # 模擬交易與紀錄完整歷程
    position, buy_price, buy_date, hold_days = False, 0, "", 0
    trades, trades_pct = [], []
    trade_logs = []  # 儲存歷史交易明細

    for i in range(1, len(test_df)):
        date_str = test_df.loc[i, "Date"].strftime("%Y-%m-%d")
        today_open = float(test_df.loc[i, "Open"])
        yesterday_signal = buy_signal[i - 1]

        if not position and yesterday_signal:
            position, buy_price, buy_date, hold_days = (
                True,
                today_open,
                date_str,
                0,
            )
        elif position:
            hold_days += 1
            open_pnl_pct = (today_open - buy_price) / buy_price

            if (
                open_pnl_pct >= take_profit
                or open_pnl_pct <= stop_loss
                or hold_days >= max_hold_days
            ):
                profit = today_open - buy_price
                trades.append(profit)
                trades_pct.append(open_pnl_pct)

                # 判定平倉原因
                if open_pnl_pct >= take_profit:
                    reason = "🎯 停利"
                elif open_pnl_pct <= stop_loss:
                    reason = "✂️ 停損"
                else:
                    reason = "⏰ 到期"

                log_entry = f"• {buy_date} 買進 ({buy_price:.1f}) ➔ {date_str} 出場 ({today_open:.1f}) | 損益: {open_pnl_pct*100:+5.2f}% ({reason})"
                trade_logs.append(log_entry)

                if yesterday_signal:
                    buy_price, buy_date, hold_days = today_open, date_str, 0
                else:
                    position = False

    # 組立 Telegram 訊息
    last_row = test_df.iloc[-1]
    last_obs_date = last_row["Date"].strftime("%Y-%m-%d")
    last_close = float(last_row["Close"])
    last_signal = buy_signal[-1]

    msg = f"📌 <b>【{stock_name} ({stock_id}) 歷史回測與最新狀態】</b>\n\n"

    # 1. 歷史交易紀錄區塊
    msg += "📜 <b>2024 年至今交易明細：</b>\n"
    if trade_logs:
        recent_logs = trade_logs[-15:]
        msg += "\n".join(recent_logs) + "\n"
        if len(trade_logs) > 15:
            msg += f"<i>(已隱藏早期 {len(trade_logs)-15} 筆交易紀錄)</i>\n"
    else:
        msg += "尚無完成交易紀錄\n"

    # 2. 總體統計績效
    total_trades = len(trades)
    win_rate = (
        (sum(1 for t in trades if t > 0) / total_trades * 100)
        if total_trades > 0
        else 0
    )
    total_return = sum(trades_pct) * 100 if trades_pct else 0

    msg += f"\n📊 <b>回測績效總覽：</b>\n"
    msg += f"• 總交易次數：{total_trades} 次\n"
    msg += f"• 勝率：{win_rate:.1f}%\n"
    msg += f"• 累計報酬率：{total_return:+6.2f}%\n"

    msg += "\n-----------------------------------\n"
    msg += f"📅 資料日期：{last_obs_date}\n"
    msg += f"📈 最新收盤價：{last_close:.1f}\n"

    # 3. 當前最新狀態
    if position:
        unrealized_pnl = last_close - buy_price
        unrealized_pnl_pct = (unrealized_pnl / buy_price) * 100
        tp_price = buy_price * (1 + take_profit)
        sl_price = buy_price * (1 + stop_loss)

        msg += f"🚨 <b>當前狀態：持倉中</b>\n"
        msg += f"• 建倉日期：{buy_date}\n"
        msg += f"• 買進成本：{buy_price:.1f}\n"
        msg += f"• 持有天數：{hold_days} / {max_hold_days} 天\n"
        msg += f"• 未實現損益：{unrealized_pnl:+7.1f} ({unrealized_pnl_pct:+5.2f}%)\n"
        msg += f"• 🎯 停利價：{tp_price:.1f}\n"
        msg += f"• ✂️ 停損價：{sl_price:.1f}\n"
        msg += f"💡 建議：繼續持倉。\n"
    elif last_signal:
        msg += f"🛒 <b>當前狀態：發出買進訊號！</b>\n"
        msg += f"💡 建議：明日 09:00 開盤價建立倉位。\n"
    else:
        msg += f"💤 <b>當前狀態：觀望中</b>\n"
        msg += f"💡 建議：保持觀望，暫不進場。\n"

    return msg


if __name__ == "__main__":
    targets = ["2330", "^TWII"]

    for t in targets:
        report = run_strategy(t)
        send_telegram_msg(report)
