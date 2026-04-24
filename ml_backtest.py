import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 核心基本面与市场特征引擎
# ==========================================

def calculate_elo_engine(df, k_factor=35, home_adv=75):
    teams = list(set(df['HomeTeam']).union(set(df['AwayTeam'])))
    elo_dict = {team: 1500 for team in teams}
    elo_diffs = []
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        current_diff = (elo_dict[home] + home_adv) - elo_dict[away]
        elo_diffs.append(current_diff)
        expected_home = 1 / (1 + 10 ** (-current_diff / 400))
        actual_score = 1.0 if row['FTHG'] > row['FTAG'] else (0.5 if row['FTHG'] == row['FTAG'] else 0.0)
        margin = abs(row['FTHG'] - row['FTAG'])
        g_multiplier = 1.0 if margin < 2 else (1.5 if margin == 2 else (11 + margin) / 8)
        shift = k_factor * g_multiplier * (actual_score - expected_home)
        elo_dict[home] += shift
        elo_dict[away] -= shift
    return elo_diffs

def add_momentum_engine(df):
    df = df.copy()
    df['h_pts'] = df['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    df['a_pts'] = df['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
    df['h_form'] = df.groupby('HomeTeam')['h_pts'].transform(lambda x: x.rolling(5, closed='left').mean()).fillna(1.0)
    df['a_form'] = df.groupby('AwayTeam')['a_pts'].transform(lambda x: x.rolling(5, closed='left').mean()).fillna(1.0)
    df['Form_Diff'] = df['h_form'] - df['a_form']
    return df

def add_xg_efficiency_engine(df):
    df = df.copy()
    df['h_scored'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(10, closed='left').mean()).fillna(1.5)
    df['h_conceded'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.rolling(10, closed='left').mean()).fillna(1.5)
    df['a_scored'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(10, closed='left').mean()).fillna(1.5)
    df['a_conceded'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.rolling(10, closed='left').mean()).fillna(1.5)
    df['Home_xG'] = (df['h_scored'] + df['a_conceded']) / 2
    df['Away_xG'] = (df['a_scored'] + df['h_conceded']) / 2
    df['xG_Diff'] = df['Home_xG'] - df['Away_xG']
    return df

def add_market_probabilities(df):
    """
    提取庄家对胜平负的隐含概率作为最强先验特征
    """
    df = df.copy()
    odds_h = 'B365H' if 'B365H' in df.columns else 'BbAvAHH'
    odds_d = 'B365D' if 'B365D' in df.columns else 'BbAvD'
    odds_a = 'B365A' if 'B365A' in df.columns else 'BbAvA'
    
    if odds_h in df.columns and odds_d in df.columns and odds_a in df.columns:
        df['Market_Margin'] = (1/df[odds_h] + 1/df[odds_d] + 1/df[odds_a])
        df['Prob_H_Bookie'] = (1/df[odds_h]) / df['Market_Margin']
        df['Prob_D_Bookie'] = (1/df[odds_d]) / df['Market_Margin']
        df['Prob_A_Bookie'] = (1/df[odds_a]) / df['Market_Margin']
    else:
        df['Prob_H_Bookie'] = 0.33
        df['Prob_D_Bookie'] = 0.33
        df['Prob_A_Bookie'] = 0.33
    return df

# ==========================================
# 2. 滚动回测主流程 (三分类预测)
# ==========================================

def run_v28_pure_prediction_workflow(file_list):
    print(f"--- Sniper V28.0: 英超 (E0) 全域胜平负预测引擎 ---")
    
    all_df = []
    for f in file_list:
        if os.path.exists(f):
            temp_df = pd.read_csv(f)
            temp_df['Date'] = pd.to_datetime(temp_df['Date'], dayfirst=True, errors='coerce')
            all_df.append(temp_df)
            
    if not all_df:
        print("错误：未找到有效数据文件。")
        return

    df = pd.concat(all_df).sort_values('Date').reset_index(drop=True).copy()
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])

    df['Elo_Diff'] = calculate_elo_engine(df)
    df = add_momentum_engine(df)
    df = add_xg_efficiency_engine(df)
    df = add_market_probabilities(df)
    
    # 目标变量映射为数字：Away=0, Draw=1, Home=2
    label_map = {'A': 0, 'D': 1, 'H': 2}
    reverse_map = {0: 'A', 1: 'D', 2: 'H'}
    df['target'] = df['FTR'].map(label_map)
    
    features = ['Elo_Diff', 'Form_Diff', 'xG_Diff', 'Prob_H_Bookie', 'Prob_D_Bookie', 'Prob_A_Bookie']
    df = df.dropna(subset=features + ['target']).reset_index(drop=True)

    train_size, step, predictions = 250, 20, []
    
    for start in range(train_size, len(df), step):
        end = min(start + step, len(df))
        
        # 使用原生 RandomForest 进行多分类预测
        model = RandomForestClassifier(n_estimators=1000, max_depth=6, random_state=42)
        model.fit(df.iloc[:start][features], df.iloc[:start]['target'])
        
        chunk = df.iloc[start:end].copy()
        # 获取 [A, D, H] 的概率数组
        probs = model.predict_proba(chunk[features])
        
        # 记录模型最高置信度的选项及概率
        chunk['pred_class_num'] = np.argmax(probs, axis=1)
        chunk['pred_class'] = chunk['pred_class_num'].map(reverse_map)
        chunk['pred_prob'] = np.max(probs, axis=1)
        
        # 记录具体每项的概率
        chunk['model_prob_A'] = probs[:, 0]
        chunk['model_prob_D'] = probs[:, 1]
        chunk['model_prob_H'] = probs[:, 2]
        
        predictions.append(chunk)

    final_df = pd.concat(predictions)
    
    # 触发条件：为了覆盖足够多的场次，我们只设置一个极低的置信度过滤
    # （例如预测概率大于 0.40 即可触发，平局由于概率天然低，可降低要求）
    def is_trigger(row):
        prob = row['pred_prob']
        pred = row['pred_class']
        if pred == 'D':
            return prob > 0.28  # 平局的概率极少超过35%，降低门槛以覆盖平局
        else:
            return prob > 0.45  # 胜负稍微要求一点置信度
            
    final_df['is_trigger'] = final_df.apply(is_trigger, axis=1)
    hits = final_df[final_df['is_trigger']].copy()
    
    hits['is_correct'] = (hits['pred_class'] == hits['FTR']).astype(int)

    print("\n" + "="*50)
    print(f"   V28.0 核心指标 (忽略 ROI，专注预测准确率)")
    print("="*50)
    
    if not hits.empty:
        wr = hits['is_correct'].mean()
        
        print(f"总回测场次: {len(final_df)} 场")
        print(f"实际触发场次: {len(hits)} 场 (占比 {len(hits)/len(final_df):.2%})")
        print(f"全局预测准确率: {wr:.2%}")
        
        print("\n--- 细分赛果覆盖与准确率 ---")
        for outcome in ['H', 'D', 'A']:
            subset = hits[hits['pred_class'] == outcome]
            if not subset.empty:
                acc = subset['is_correct'].mean()
                print(f"预测 [{outcome}] 触发: {len(subset)} 场 | 准确率: {acc:.2%}")
            else:
                print(f"预测 [{outcome}] 触发: 0 场")
                
        recent_hits = hits.tail(100)
        print(f"\n最近 100 场触发命中数: {recent_hits['is_correct'].sum()} / {len(recent_hits)}")
    else:
        print("未触发信号。")

if __name__ == "__main__":
    data_files = ['SP12324.csv', 'ESP12425.csv', 'SP1.csv','E0_2324.CSV','E0_2425.CSV','E0_2526.CSV']
    run_v28_pure_prediction_workflow(data_files)