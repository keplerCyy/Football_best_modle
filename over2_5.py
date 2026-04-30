import pandas as pd
import numpy as np
import os
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 核心配置：回归“共识”与“联动”
# ==========================================
FILES = [
    'E0_2324.CSV', 'E0_2425.CSV', 'E0_2526.CSV',
    'SP1_2324.csv', 'SP1_2425.csv', 'SP1_2526.csv',
    'F1_2324.CSV', 'F1_2425.CSV', 'F1_2526.CSV'
]

def run_v42_fusion_backtest(prob_threshold=0.60):
    all_results = []
    
    for file in FILES:
        if not os.path.exists(file): continue
        df = pd.read_csv(file)
        df.columns = [c.upper() for c in df.columns]
        df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
        df = df.sort_values('DATE').dropna(subset=['FTHG', 'FTAG', 'B365H', 'B365>2.5'])

        # 1. 预训练一个 1X2 模型作为环境参考 (使用前 100 场热身)
        # 2. 滚动回测
        for i in range(100, len(df)):
            history = df.iloc[:i]
            current = df.iloc[i]
            
            # --- A. 泊松引擎 (进球数分布) ---
            home_t, away_t = current['HOMETEAM'], current['AWAYTEAM']
            h_attack = history[history['HOMETEAM'] == home_t]['FTHG'].tail(10).mean()
            a_defend = history[history['AWAYTEAM'] == away_t]['FTHG'].tail(10).mean()
            a_attack = history[history['AWAYTEAM'] == away_t]['FTAG'].tail(10).mean()
            h_defend = history[history['HOMETEAM'] == home_t]['FTAG'].tail(10).mean()
            l_avg = (history['FTHG'].mean() + history['FTAG'].mean())
            
            if pd.isna([h_attack, a_defend]).any(): continue
            
            lambda_h = (h_attack * a_defend) / (l_avg / 2)
            lambda_a = (a_attack * h_defend) / (l_avg / 2)
            p_over = 1 - sum([poisson.pmf(k, lambda_h + lambda_a) for k in range(3)])

            # --- B. RF 引擎 (胜平负辅助) ---
            # 逻辑：如果 RF 模型认为这场比赛【平局概率 < 20%】，则大球置信度提升
            train_chunk = history.tail(200)
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            X_train = pd.DataFrame({'PH': 1/train_chunk['B365H'], 'PD': 1/train_chunk['B365D'], 'PA': 1/train_chunk['B365A']})
            y_train = train_chunk['FTR'].map({'H': 0, 'D': 1, 'A': 2})
            rf.fit(X_train, y_train)
            
            curr_feat = pd.DataFrame([[1/current['B365H'], 1/current['B365D'], 1/current['B365A']]], columns=['PH', 'PD', 'PA'])
            probs_1x2 = rf.predict_proba(curr_feat)[0]
            prob_draw = probs_1x2[1] # 平局概率

            # --- C. 融合策略 (Fusion Logic) ---
            # 只有当：泊松算大球概率高 且 1X2 模型认为不容易出平局(D) 时，才是真大球
            is_over_trigger = (p_over > prob_threshold) and (prob_draw < 0.25)
            # 只有当：泊松算小球概率高 且 1X2 模型认为平局倾向明显时，才是真小球
            is_under_trigger = ((1-p_over) > prob_threshold) and (prob_draw > 0.35)

            actual_over = 1 if (current['FTHG'] + current['FTAG']) > 2.5 else 0
            
            all_results.append({
                'is_over_trigger': is_over_trigger,
                'is_under_trigger': is_under_trigger,
                'actual': actual_over
            })

    res_df = pd.DataFrame(all_results)
    print(f"\n" + "="*60)
    print(f"📊 Sniper V42.0 双核联动回测 (1X2 辅助过滤)")
    print("="*60)
    
    o_subset = res_df[res_df['is_over_trigger']]
    u_subset = res_df[res_df['is_under_trigger']]
    
    if not o_subset.empty:
        print(f"[大球(O) 联动版] 触发: {len(o_subset):<5} | 胜率: {o_subset['actual'].mean():.2%}")
    if not u_subset.empty:
        print(f"[小球(U) 联动版] 触发: {len(u_subset):<5} | 胜率: {(1-u_subset['actual']).mean():.2%}")

if __name__ == "__main__":
    run_v42_fusion_backtest(prob_threshold=0.60)