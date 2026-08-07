import datetime
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# Telegram 設定
BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "8321686170:AAGqyD9TSLBtw3B5XV0Z9r8ShMEgiYidv-E"
)
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "您的_CHAT_ID")


# ==============================================================================
# 1. 特徵工程輔助函式
# ==============================================================================
def get_before_holiday_feature(df):
    """判斷今天是否為「連假/長假前最後一個交易日」"""
    twse_custom_holidays = {
        "2024-02-06",
        "2024-02-07",
        "2024-02-08",
        "2024-02-09",
        "2024-02-12",
        "2024-02-13",
        "2024-02-14",
        "2024-02-28",
        "2024-04-04",
        "2024-04-05",
        "2024-05-01",
        "2024-06-10",
        "2024-09-17",
        "2024-10-10",
        "2025-01-27",
        "2025-01-28",
        "2025-01-29",
        "2025-01-30",
        "2025-01-31",
        "2025-02-28",
        "2025-04-03",
        "2025-04-04",
        "2025-05-01",
        "2025-05-30",
        "2025-10-06",
        "2025-10-10",
        "2026-01-01",
        "2026-02-16",
        "2026-02-17",
        "2026-02-18",
        "2026-02-19",
        "2026-02-20",
        "2026-02-27",
        "2026-04-03",
        "2026-04-06",
        "2026-05-01",
        "2026-06-19",
        "2026-09-25",
        "2026-10-09",
        "2026-10-12",
    }

    def is_market_closed(dt):
        date_str = dt.strftime("%Y-%m-%d")
        is_weekend = dt.weekday() >= 5
        is_holiday = date_str in twse_custom_holidays
        return is_weekend or is_holiday

    is_before_holiday_list = []
    for current_date in df["Date"]:
        curr_dt = current_date.to_pydatetime()
        consecutive_closed_days = 0
        for next_day_offset in range(1, 6):
            check_date = curr_dt + datetime.timedelta(days=next_day_offset)
            if is_market_closed(check_date):
                consecutive_closed_days += 1
            else:
                break
        is_before_holiday_list.append(1 if consecutive_closed_days >= 3 else 0)

    return is_before_holiday_list


def get_txf_settlement_date(year, month):
    """計算台指期結算日 (當月第 3 個星期三)"""
    first_day = datetime.date(year, month, 1)
    first_wednesday = first_day + datetime.timedelta(
        days=(2 - first_day.weekday()) % 7
    )
    return first_wednesday + datetime.timedelta(days=14)


def get_next_settlement_date(ref_date=None):
    """計算未來最近台指期結算日"""
    if ref_date is None:
        ref_date = datetime.date.today()
    elif isinstance(ref_date, (pd.Timestamp, datetime.datetime)):
        ref_date = ref_date.date()

    settlement_date = get_txf_settlement_date(ref_date.year, ref_date.month)
    if ref_date > settlement_date:
        if ref_date.month == 12:
            settlement_date = get_txf_settlement_date(ref_date.year + 1, 1)
        else:
            settlement_date = get_txf_settlement_date(
                ref_date.year, ref_date.month + 1
            )
    return settlement_date


def add_settlement_features(df):
    days_to_settlement, is_settlement_day, is_before_settlement = [], [], []
    for dt in df["Date"]:
        curr_date = dt.date()
        settlement_date = get_next_settlement_date(curr_date)
        days_diff = (settlement_date - curr_date).days
        days_to_settlement.append(days_diff)
        is_settlement_day.append(1 if days_diff == 0 else 0)
        is_before_settlement.append(1 if 1 <= days_diff <= 2 else 0)

    df["Days_To_Settlement"] = days_to_settlement
    df["Is_Settlement_Day"] = is_settlement_day
    df["Is_Before_Settlement"] = is_before_settlement
    return df


def add_advanced_behavior_features(df):
    day_of_week, is_quarter_end, is_tsmc_earnings_window = [], [], []
    is_ex_dividend_season, is_red_envelope_season = [], []

    for dt in df["Date"]:
        day_of_week.append(dt.weekday())
        is_quarter_end.append(
            1 if ((dt.month in [3, 6, 9, 12]) and (dt.day >= 24)) else 0
        )
        is_tsmc_earnings_window.append(
            1 if ((dt.month in [1, 4, 7, 10]) and (12 <= dt.day <= 20)) else 0
        )
        is_ex_dividend_season.append(1 if dt.month in [6, 7, 8] else 0)
        is_red_envelope_season.append(
            1 if dt.month in [11, 12, 1, 2] else 0
        )

    df["Day_Of_Week"] = day_of_week
    df["Is_Quarter_End"] = is_quarter_end
    df["Is_TSMC_Earnings_Window"] = is_tsmc_earnings_window
    df["Is_Ex_Dividend_Season"] = is_ex_dividend_season
    df["Is_Red_Envelope_Season"] = is_red_envelope_season
    return df


def fetch_twd_exchange_rate(start_date="2018-01-01"):
    twd_df = yf.download(
        "TWD=X", start=start_date, auto_adjust=False, interval="1d"
    )
    if isinstance(twd_df.columns, pd.MultiIndex):
        twd_df.columns = twd_df.columns.get_level_values(0)
    twd_df = twd_df.dropna(subset=["Close"]).reset_index()
    twd_df["Date"] = pd.to_datetime(twd_df["Date"])
    twd_df["TWD_Return"] = twd_df["Close"].pct_change()
    return twd_df[["Date", "TWD_Return"]]


# ==============================================================================
# 2. Telegram 發送模組
# ==============================================================================
def send_telegram_msg(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    if len(message) > 4000:
        chunks = [message[i : i + 4000] for i in range(0, len(message), 4000)]
        for chunk in chunks:
            requests.post(url, json={"chat_id": CHAT_ID, "text": chunk})
    else:
        try:
            requests.post(url, json={"chat_id": CHAT_ID, "text": message})
        except Exception as e:
            print(f"Telegram 發送失敗: {e}")


# ==============================================================================
# 3. 主策略運算邏輯
# ==============================================================================
def run_strategy(target_input, twd_fx_df=None):
    if target_input.startswith("^"):
        stock_id, ticker, stock_name = (
            target_input,
            target_input,
            "台灣加權指數",
        )
    elif target_input == "2330":
        stock_id, ticker, stock_name = (
            target_input,
            f"{target_input}.TW",
            "台積電",
        )
    else:
        stock_id = target_input
        ticker = (
            f"{target_input}.TW"
            if not target_input.endswith(".TW")
            else target_input
        )
        stock_name = target_input

    df = yf.download(
        ticker, start="2018-01-01", auto_adjust=False, interval="1d"
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"]).reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    # 技術指標
    df["Returns"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Night_Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(
        1
    )

    tr = (
        pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
    )
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

    # 特徵工程
    df["Is_Before_Long_Holiday"] = get_before_holiday_feature(df)
    df = add_settlement_features(df)
    df = add_advanced_behavior_features(df)

    if twd_fx_df is not None:
        df = pd.merge(df, twd_fx_df, on="Date", how="left")
        df["TWD_Return"] = df["TWD_Return"].ffill().fillna(0)
    else:
        df["TWD_Return"] = 0

    df["Target"] = (df["Close"].shift(-5) > df["Close"] * 1.01).astype(int)
    df_clean = df.dropna().copy().reset_index(drop=True)

    take_profit, stop_loss, max_hold_days = (
        (0.04, -0.025, 7) if stock_id == "2330" else (0.03, -0.02, 7)
    )

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
        "Is_Before_Long_Holiday",
        "Days_To_Settlement",
        "Is_Settlement_Day",
        "Is_Before_Settlement",
        "Day_Of_Week",
        "Is_Quarter_End",
        "Is_TSMC_Earnings_Window",
        "Is_Ex_Dividend_Season",
        "Is_Red_Envelope_Season",
        "TWD_Return",
    ]

    model = RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )
    model.fit(train_df[features], train_df["Target"])

    probs = model.predict_proba(test_df[features])[:, 1]
    buy_signal = probs > 0.50

    # 回測模擬
    position = False
    buy_price, buy_date, hold_days = 0, "", 0
    original_buy_price, original_buy_date = 0, ""

    trade_logs = []
    stage_realized_profits, full_trade_pnl_list = [], []

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

            is_tp = open_pnl_pct >= take_profit
            is_sl = open_pnl_pct <= stop_loss
            is_time_out = hold_days >= max_hold_days

            if is_tp or is_sl or is_time_out:
                stage_realized_profits.append(pnl_val)
                trade_idx = len(trade_logs) + 1

                if yesterday_signal:
                    reason = (
                        "停利重置續抱 🔄"
                        if is_tp
                        else ("停損重置續抱 🔄" if is_sl else "滿天數重置續抱 🔄")
                    )
                    entry_date_display = (
                        original_buy_date
                        if original_buy_date != buy_date
                        else buy_date
                    )
                    trade_logs.append(
                        f"第 {trade_idx:02d} 筆 | 原進場: {entry_date_display} (${buy_price:.1f}) -> 重置: {date_str} (${today_open:.1f}) | 損益: {pnl_val:+7.1f} ({open_pnl_pct*100:+5.1f}%) | {reason}"
                    )
                    buy_price, buy_date, hold_days = today_open, date_str, 0
                else:
                    full_trade_pnl_list.append(today_open - original_buy_price)
                    reason = (
                        "停利平倉 🎉"
                        if is_tp
                        else ("停損平倉 ✂️" if is_sl else "滿天數平倉 ⏱️")
                    )
                    trade_logs.append(
                        f"第 {trade_idx:02d} 筆 | 進場: {buy_date} (${buy_price:.1f}) -> 出場: {date_str} (${today_open:.1f}) | 損益: {pnl_val:+7.1f} ({open_pnl_pct*100:+5.1f}%) | {reason}"
                    )
                    position = False

    # 產出報告
    last_row = test_df.iloc[-1]
    last_obs_date = last_row["Date"].strftime("%Y-%m-%d")
    last_close = float(last_row["Close"])
    latest_signal = buy_signal[-1]

    output = []
    output.append("==================================================")
    output.append(f"🔍 評估標的：{stock_name} ({stock_id})")
    output.append(f"📊 平均日波動度 (ATR)：{test_df['ATR_Pct'].mean()*100:.2f}%")
    output.append(
        f"⚙️ 風控參數：停利 {take_profit*100:+.1f}% | 停損 {stop_loss*100:+.1f}% | 持有 {max_hold_days} 天"
    )
    output.append("==================================================\n")

    output.append(
        f"=== {stock_name} ({stock_id}) 歷史『已平倉』明細 (2024 起) ==="
    )
    for log in trade_logs:
        output.append(log)
    output.append("--------------------------------------------------")

    total_closed = len(stage_realized_profits)
    win_trades = sum(1 for p in stage_realized_profits if p > 0)
    loss_trades = sum(1 for p in stage_realized_profits if p < 0)
    win_rate = (win_trades / total_closed * 100) if total_closed > 0 else 0
    total_pnl = sum(stage_realized_profits)

    # 計算單次完全出場的最大獲利與最大虧損
    if full_trade_pnl_list:
        max_single_win = max(full_trade_pnl_list)
        max_single_loss = min(full_trade_pnl_list)
    else:
        max_single_win = 0.0
        max_single_loss = 0.0

    output.append(f"📊 2024 至今已平倉交易筆數: {total_closed} 筆")
    output.append(f"🏆 獲利筆數: {win_trades} 筆 | 虧損筆數: {loss_trades} 筆")
    output.append(f"🎯 歷史已實現勝率: {win_rate:.2f}%")
    output.append(f"📈 累積已實現獲利: {total_pnl:+0.1f} 元/點")
    output.append(f"🚀 單次最大獲利點數 (完全出場): {max_single_win:+0.1f} 元/點")
    output.append(f"💥 單次最大虧損點數 (完全出場): {max_single_loss:+0.1f} 元/點\n")

    output.append("★" * 42)
    output.append(
        f"📌 【{stock_name} 當前最新交易狀態】 (資料更新至：{last_obs_date})"
    )
    output.append(f"📈 最新收盤價：{last_close:.1f}")

    # 判斷明確操作動作 (續抱 / 平倉 / 買進 / 觀望)
    if position:
        unrealized_pnl = last_close - buy_price
        unrealized_pnl_pct = (unrealized_pnl / buy_price) * 100
        output.append("🚨 【狀態：持倉中 / 尚未平倉】")
        output.append(f"  • 持有成本：{buy_price:.1f} (天數: {hold_days}/{max_hold_days})")
        output.append(
            f"  • 未實現損益：{unrealized_pnl:+0.1f} ({unrealized_pnl_pct:+0.2f}%)"
        )
        action_summary = "🟢 建議【繼續持倉】(觀察明日是否觸發停利/停損/滿天數)"
    else:
        if latest_signal:
            output.append("🚨 【狀態：觸發買進訊號】")
            action_summary = "🚀 建議【明晨開盤買進】"
        else:
            output.append("💤 【狀態：空倉觀望】")
            action_summary = "💤 建議【繼續空倉觀望】"

    output.append(f"  👉 操作方向：{action_summary}")

    next_settlement = get_next_settlement_date(last_row["Date"])
    days_to_settlement = (next_settlement - last_row["Date"].date()).days
    output.append(
        f"🗓️ 下次台指期結算日：{next_settlement.strftime('%Y-%m-%d')} (剩 {days_to_settlement} 天)"
    )
    output.append("★" * 42)

    summary_info = {
        "stock_name": stock_name,
        "stock_id": stock_id,
        "action": action_summary,
        "last_date": last_obs_date,
    }

    return "\n".join(output), summary_info


# ==============================================================================
# 4. 主執行階段
# ==============================================================================
if __name__ == "__main__":
    print("⏳ 正在取得美元兌台幣匯率歷史資料...")
    twd_fx_df = fetch_twd_exchange_rate(start_date="2018-01-01")

    targets = ["2330", "^TWII"]
    reports = []
    summary_list = []

    for t in targets:
        report_text, summary = run_strategy(t, twd_fx_df=twd_fx_df)
        reports.append(report_text)
        summary_list.append(summary)

    # 1. 印出完整詳細報告
    full_report = "\n\n".join(reports)
    print(full_report)

    # 2. 於【最底部】建立極度顯眼的「操作重點速查 Block」
    upcoming_settlement = get_next_settlement_date(datetime.date.today())

    bottom_summary = []
    bottom_summary.append("\n" + "=" * 60)
    bottom_summary.append("⚡【今日 / 明日 全標的操作重點速查卡】⚡")
    bottom_summary.append("=" * 60)

    for item in summary_list:
        bottom_summary.append(
            f"🔹 {item['stock_name']} ({item['stock_id']}) [{item['last_date']}]："
        )
        bottom_summary.append(f"   ➔ {item['action']}")

    bottom_summary.append("-" * 60)
    bottom_summary.append(
        f"🗓️ 未來最近台指期結算日為：{upcoming_settlement.strftime('%Y-%m-%d')}"
    )
    bottom_summary.append("=" * 60)

    bottom_summary_str = "\n".join(bottom_summary)

    # 終端機最後印出重點摘要
    print(bottom_summary_str)

    # 發送 Telegram (詳細報告 + 底部重點摘要)
    send_telegram_msg(full_report + "\n\n" + bottom_summary_str)
