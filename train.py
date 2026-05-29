import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
import config
import data_manager as dm
from zigzag_persistence import zigzag_persistence_scores

def normalize_scores(score_dict):
    scores = np.array(list(score_dict.values()))
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.0 for k in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    return {ticker: float(norm[i]) for i, ticker in enumerate(score_dict.keys())}

def rolling_walkforward_backtest(returns_df, window_days, top_n=3):
    n = len(returns_df)
    sum_returns = {}
    count = {}
    for t in range(window_days, n - 1):
        window = returns_df.iloc[t - window_days : t]
        next_day = returns_df.iloc[t]
        scores = zigzag_persistence_scores(window, max_dim=config.MAX_DIMENSION, num_steps=config.NUM_FILTRATION_STEPS)
        norm = normalize_scores(scores)
        sorted_etfs = sorted(norm.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [ticker for ticker, _ in sorted_etfs[:top_n]]
        for ticker in top_etfs:
            ret = next_day[ticker]
            sum_returns[ticker] = sum_returns.get(ticker, 0.0) + ret
            count[ticker] = count.get(ticker, 0) + 1
    avg_returns = {ticker: sum_returns[ticker]/count[ticker] for ticker in sum_returns}
    return avg_returns

def run_for_window(returns, window_days):
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    raw_scores = zigzag_persistence_scores(ret_window, max_dim=config.MAX_DIMENSION, num_steps=config.NUM_FILTRATION_STEPS)
    norm_scores = normalize_scores(raw_scores)
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": t, "zigzag_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:config.TOP_N]]
    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores
    }

def main():
    print("Loading master data...")
    dm.load_master_data()
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print("  No data -> skipping")
            continue
        all_window_results = []
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, w)
            if out:
                all_window_results.append(out)
            else:
                print(f"    Failed for window {w}")
        for wdata in all_window_results:
            w = wdata['window']
            print(f"  Backtest for window {w}...")
            backtest_etf_avg = rolling_walkforward_backtest(returns, w, top_n=config.TOP_N)
            wdata['backtest_per_etf_avg_return'] = backtest_etf_avg
        best_avg = -np.inf
        best_window = None
        best_data = None
        for wdata in all_window_results:
            bt_vals = list(wdata['backtest_per_etf_avg_return'].values())
            if bt_vals:
                avg_bt = np.mean(bt_vals)
                if avg_bt > best_avg:
                    best_avg = avg_bt
                    best_window = wdata['window']
                    best_data = wdata
        results["universes"][uni_name] = {
            "best_window_by_backtest": best_window,
            "best_window_data": best_data,
            "all_windows": all_window_results
        }
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/zigzag_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
