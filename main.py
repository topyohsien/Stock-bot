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
            payload = {"chat_id": CHAT_ID, "text": chunk}
            requests.post(url, json=payload)
    else:
        payload = {"chat_id": CHAT_ID, "text": message}
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

    # 模擬交易（追蹤單次完整交易點數與階段歷程）
    position = False
    buy_price = 0
    buy_date = ""
    hold_days = 0

    original_buy_price = 0  # 紀錄最初進場價（用於計算真實單筆交易點數）
    original_buy_date = ""

    trade_logs = []
    stage_realized_profits = []  # 階段損益列表
    full_trade_pnl_list = []  # 真實完全出場單筆損益點數

    for i in range(1, len(test_df)):
        date_str = test_df.loc[i, "Date"].strftime("%Y-%m-%d")
        today_open = float(test_df.loc[i, "Open"])
        yesterday_signal = buy_signal[i - 1]

        if not position and yesterday_signal:
            position = True
            buy_price = today_open
            buy_date = date_str
            original_buy_price = today_open
            original_buy_date = date_str
            hold_days = 0
        elif position:
            hold_days += 1
            open_pnl_pct = (today_open - buy_price) / buy_price
            pnl_val = today_open - buy_price

            # 判斷是否觸發階段平倉條件
            is_tp = open_pnl_pct >= take_profit
            is_sl = open_pnl_pct <= stop_loss
            is_time_out = hold_days >= max_hold_days

            if is_tp or is_sl or is_time_out:
                stage_realized_profits.append(pnl_val)
                trade_idx = len(trade_logs) + 1

                if yesterday_signal:
                    # 精確拆分三種續抱情境
                    if is_tp:
                        reason = "停利重置續抱 🔄"
                    elif is_sl:
                        reason = "停損重置續抱 🔄"
                    else:
                        reason = "滿天數重置續抱 🔄"

                    if original_buy_date != buy_date:
                        log_str = f"第 {trade_idx:02d} 筆 | 原進場: {original_buy_date} (${buy_price:.1f}) -> 今日重置: {date_str} (${today_open:.1f}) | 階段損益: {pnl_val:+7.1f} ({open_pnl_pct*100:+5.1f}%) | {reason}"
                    else:
                        log_str = f"第 {trade_idx:02d} 筆 | 原進場: {buy_date} (${buy_price:.1f}) -> 今日重置: {date_str} (${today_open:.1f}) | 階段損益: {pnl_val:+7.1f} ({open_pnl_pct*100:+5.1f}%) | {reason}"

                    trade_logs.append(log_str)

                    # 重置成本與天數，但保留最初 original_buy_price
                    buy_price = today_open
                    buy_date = date_str
                    hold_days = 0
                else:
                    # 完全平倉出場：計算從最初進場到現在的真實單筆總損益
                    full_trade_pnl = today_open - original_buy_price
                    full_trade_pnl_list.append(full_trade_pnl)

                    if is_tp:
                        reason = "停利平倉 🎉"
                    elif is_sl:
                        reason = "停損平倉 ✂️"
                    else:
                        reason = "滿天數平倉 ⏱️"

                    log_str = f"第 {trade_idx:02d} 筆 | 進場: {buy_date} (${buy_price:.1f}) -> 出場: {date_str} (${today_open:.1f}) | 損益: {pnl_val:+7.1f} ({open_pnl_pct*100:+5.1f}%) | {reason}"
                    trade_logs.append(log_str)
                    position = False

    # 組立傳統純文字輸出報告
    last_row = test_df.iloc[-1]
    last_obs_date = last_row["Date"].strftime("%Y-%m-%d")
    last_close = float(last_row["Close"])

    train_start = df_clean[train_mask]["Date"].iloc[0].strftime("%Y-%m-%d")
    train_end = df_clean[train_mask]["Date"].iloc[-1].strftime("%Y-%m-%d")
    test_start = test_df["Date"].iloc[0].strftime("%Y-%m-%d")
    avg_atr_pct = test_df["ATR_Pct"].mean() * 100

    output = []
    output.append(
        "=================================================="
    )
    output.append(f"🔍 評估標的：{stock_name} ({stock_id})")
    output.append(f"📊 平均日波動度 (ATR比率)：{avg_atr_pct:.2f}%")
    output.append(
        f"⚙️ 匹配風控參數：停利 {take_profit*100:+.1f}% | 停損 {stop_loss*100:+.1f}% | 最長持有 {max_hold_days} 天"
    )
    output.append(
        "==================================================\n"
    )

    output.append(
        f"📅 方案 A 訓練區間：{train_start} ~ {train_end}"
    )
    output.append(
        f"🎯 近期盲測區間：{test_start} ~ {last_obs_date}"
    )
    output.append(
        "===========================================================================\n"
    )

    output.append(
        f"=== {stock_name} ({stock_id}) 歷史『已平倉』交易明細 (2024 起至今) ==="
    )
    output.append(
        "---------------------------------------------------------------------------"
    )
    for log in trade_logs:
        output.append(log)
    output.append(
        "==========================================================================="
    )

    # 階段階段與勝率統計
    total_closed = len(stage_realized_profits)
    win_trades = sum(1 for p in stage_realized_profits if p > 0)
    loss_trades = sum(1 for p in stage_realized_profits if p < 0)
    win_rate = (win_trades / total_closed * 100) if total_closed > 0 else 0
    total_pnl = sum(stage_realized_profits)

    # 計算真實完整單筆交易的最大獲利與最大虧損
    if full_trade_pnl_list:
        max_single_win = max(full_trade_pnl_list)
        max_single_loss = min(full_trade_pnl_list)
    else:
        max_single_win = 0.0
        max_single_loss = 0.0

    output.append(
        f"📊 2024 至今已平倉交易筆數: {total_closed} 筆"
    )
    output.append(
        f"🏆 獲利筆數: {win_trades} 筆 | 虧損筆數: {loss_trades} 筆"
    )
    output.append(f"🎯 歷史已實現勝率: {win_rate:.2f}%")
    output.append(
        f"📈 累積已實現獲利: {total_pnl:+0.1f} 元/點"
    )
    output.append(
        f"🚀 單次最大獲利點數 (完全出場): {max_single_win:+0.1f} 元/點"
    )
    output.append(
        f"💥 單次最大虧損點數 (完全出場): {max_single_loss:+0.1f} 元/點\n"
    )

    output.append("★" * 42)
    output.append(
        f"📌 【{stock_name} 當前最新交易狀態】 (資料更新至：{last_obs_date})"
    )
    output.append(f"📈 最新收盤價：{last_close:.1f}")
    output.append("-" * 50)

    if position:
        unrealized_pnl = last_close - buy_price
        unrealized_pnl_pct = (unrealized_pnl / buy_price) * 100
        tp_price = buy_price * (1 + take_profit)
        sl_price = buy_price * (1 + stop_loss)

        output.append("🚨 【狀態：持倉中 / 尚未平倉】")
        output.append(
            f"  • 最近一次建立/重置日期：{buy_date}"
        )
        output.append(
            f"  • 最新基準成本 (開盤價)：{buy_price:.1f}"
        )
        output.append(
            f"  • 當前段落已持有天數：{hold_days} / {max_hold_days} 天"
        )
        output.append(
            f"  • 當前未實現損益：{unrealized_pnl:+0.1f} ({unrealized_pnl_pct:+0.2f}%)"
        )
        output.append(
            f"  • 🎯 目標停利價：{tp_price:.1f} ({take_profit*100:+.1f}%)"
        )
        output.append(
            f"  • ✂️ 防守停損價：{sl_price:.1f} ({stop_loss*100:+.1f}%)"
        )
        output.append(
            "  💡 操作建議：持倉繼續有效，明晨 08:30 觀察是否觸發新目標價或滿天數。"
        )
    else:
        output.append("💤 【狀態：空倉觀望】")
        output.append("  💡 操作建議：目前無持倉，保持觀望。")

    output.append("★" * 42)

    return "\n".join(output)


if __name__ == "__main__":
    targets = ["2330", "^TWII"]

    for t in targets:
        report = run_strategy(t)
        print(report)
        print("\n\n")
        send_telegram_msg(report)
