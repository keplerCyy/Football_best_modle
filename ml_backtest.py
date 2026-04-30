import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

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
    df = df.copy()
    # 优先使用 B365 赔率，若无则使用平均赔率
    odds_h = 'B365H' if 'B365H' in df.columns else 'BbAvH'
    odds_d = 'B365D' if 'B365D' in df.columns else 'BbAvD'
    odds_a = 'B365A' if 'B365A' in df.columns else 'BbAvA'
    
    if odds_h in df.columns and odds_d in df.columns and odds_a in df.columns:
        df['Market_Margin'] = (1/df[odds_h] + 1/df[odds_d] + 1/df[odds_a])
        df['Prob_H_Bookie'] = (1/df[odds_h]) / df['Market_Margin']
        df['Prob_D_Bookie'] = (1/df[odds_d]) / df['Market_Margin']
        df['Prob_A_Bookie'] = (1/df[odds_a]) / df['Market_Margin']
    else:
        df['Prob_H_Bookie'], df['Prob_D_Bookie'], df['Prob_A_Bookie'] = 0.33, 0.33, 0.33
    return df

# ==========================================
# 2. Sniper V40.0 置信度引擎
# ==========================================

def calculate_confidence_score(row):
    """
    多维置信度算法：融合绝对概率、优势差与熵隙
    """
    model_prob = row['pred_prob']
    pred_class = row['pred_class']
    
    # 1. 计算优势差 (Edge): 模型概率与庄家隐含概率的差值
    market_prob = row[f'Prob_{pred_class}_Bookie']
    edge = model_prob - market_prob
    
    # 2. 计算熵隙 (Entropy Gap): 第一预测与第二预测的概率差
    all_probs = sorted([row['model_prob_H'], row['model_prob_D'], row['model_prob_A']], reverse=True)
    prob_gap = all_probs[0] - all_probs[1]
    
    # 3. 综合加权评分 (Standardized 0-100)
    # 权重: 绝对概率(30%) + 优势差(50%) + 熵隙(20%)
    # Edge 是核心，代表了“超额获胜的可能性”
    edge_norm = np.clip((edge + 0.1) / 0.3, 0, 1) # 将 -0.1~0.2 映射到 0~1
    gap_norm = np.clip(prob_gap / 0.4, 0, 1)
    
    score = (model_prob * 0.3 + edge_norm * 0.5 + gap_norm * 0.2) * 100
    return round(score, 2)

# ==========================================
# 3. 滚动回测与评估流程
# ==========================================

def run_v40_confidence_workflow(file_list):
    print(f"--- Sniper V40.0: 多维置信度量化引擎 ---")
    
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

    # 特征生成
    df['Elo_Diff'] = calculate_elo_engine(df)
    df = add_momentum_engine(df)
    df = add_xg_efficiency_engine(df)
    df = add_market_probabilities(df)
    
    # 目标变量映射
    label_map = {'A': 0, 'D': 1, 'H': 2}
    reverse_map = {0: 'A', 1: 'D', 2: 'H'}
    df['target'] = df['FTR'].map(label_map)
    
    features = ['Elo_Diff', 'Form_Diff', 'xG_Diff', 'Prob_H_Bookie', 'Prob_D_Bookie', 'Prob_A_Bookie']
    df = df.dropna(subset=features + ['target']).reset_index(drop=True)

    train_size, step, predictions = 300, 25, []
    
    print(f"开始滚动回测: 总样本 {len(df)} 场...")
    
    for start in range(train_size, len(df), step):
        end = min(start + step, len(df))
        
        # 训练模型
        model = RandomForestClassifier(n_estimators=1000, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(df.iloc[:start][features], df.iloc[:start]['target'])
        
        chunk = df.iloc[start:end].copy()
        probs = model.predict_proba(chunk[features])
        
        # 记录每种预测的概率
        chunk['model_prob_A'] = probs[:, 0]
        chunk['model_prob_D'] = probs[:, 1]
        chunk['model_prob_H'] = probs[:, 2]
        
        # 核心预测结果
        chunk['pred_class_num'] = np.argmax(probs, axis=1)
        chunk['pred_class'] = chunk['pred_class_num'].map(reverse_map)
        chunk['pred_prob'] = np.max(probs, axis=1)
        
        # 计算置信度评分
        chunk['confidence_score'] = chunk.apply(calculate_confidence_score, axis=1)
        predictions.append(chunk)

    final_df = pd.concat(predictions)
    final_df['is_correct'] = (final_df['pred_class'] == final_df['FTR']).astype(int)

    # ==========================================
    # 4. 结果分级展示 (PM 报告风格)
    # ==========================================
    print("\n" + "="*60)
    print(f"   Sniper V40.0 核心回测报告 (基于置信度分级)")
    print("="*60)
    
    # 定义评级
    bins = [0, 50, 70, 85, 100]
    labels = ['C级 (噪音)', 'B级 (观察)', 'A级 (稳健)', 'S级 (核心)']
    final_df['Grade'] = pd.cut(final_df['confidence_score'], bins=bins, labels=labels)

    summary = final_df.groupby('Grade').agg({
        'is_correct': ['count', 'mean']
    }).reset_index()
    
    summary.columns = ['评级', '场次数', '准确率']
    
    for _, row in summary.iterrows():
        print(f"{row['评级']}: 场次={int(row['场次数']):<4} | 准确率={row['准确率']:.2%}")

    print("-" * 60)
    
    # 针对高置信度场次的细分
    s_hits = final_df[final_df['Grade'].isin(['S级 (核心)', 'A级 (稳健)'])]
    if not s_hits.empty:
        print(f"【高置信度触发】(Score > 70)")
        print(f"总触发场次: {len(s_hits)} | 综合胜率: {s_hits['is_correct'].mean():.2%}")
        for res in ['H', 'D', 'A']:
            res_df = s_hits[s_hits['pred_class'] == res]
            if not res_df.empty:
                print(f"  - 预测 {res} 胜率: {res_df['is_correct'].mean():.2%}")

    print("\n[最新 5 场高置信度信号预测]")
    print(final_df[final_df['confidence_score'] > 65][['Date', 'HomeTeam', 'AwayTeam', 'pred_class', 'confidence_score', 'FTR']].tail(5))

if __name__ == "__main__":
    # 请确保这些文件在同一目录下
    data_files = ['SP1_2324.csv', 'SP1_2425.csv', 'SP1_2526.csv','E0_2324.CSV','E0_2425.CSV','E0_2526.CSV','F1_2324.CSV','F1_2425.CSV','F1_2526.CSV']
    run_v40_confidence_workflow(data_files)