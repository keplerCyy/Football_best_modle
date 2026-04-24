import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ==========================================
# 0. 页面配置与 UI 样式 (解决渲染问题的关键)
# ==========================================
st.set_page_config(page_title="Sniper AI 预测终端 V28.0", page_icon="🎯", layout="wide")

# 这里定义 CSS，确保卡片样式生效
st.markdown("""
<style>
    .prediction-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #e6e9ef;
        border-left: 8px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .safe-home { border-left-color: #28a745; }
    .safe-away { border-left-color: #17a2b8; }
    .warning { border-left-color: #ffc107; }
    .prob-container {
        display: flex;
        gap: 30px;
        margin: 15px 0;
        background: #f8f9fa;
        padding: 12px 20px;
        border-radius: 8px;
    }
    .prob-item { display: flex; flex-direction: column; }
    .prob-label { font-size: 0.85em; color: #666; margin-bottom: 4px; }
    .prob-value { font-family: 'Segoe UI', sans-serif; font-weight: 800; font-size: 1.3em; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. AI 核心引擎
# ==========================================
@st.cache_data(show_spinner=False)
def process_features(df):
    def calculate_elo_engine(df, k_factor=35, home_adv=75):
        teams = list(set(df['HomeTeam']).union(set(df['AwayTeam'])))
        elo_dict = {team: 1500 for team in teams}
        elo_diffs = []
        for _, row in df.iterrows():
            home, away = row['HomeTeam'], row['AwayTeam']
            current_diff = (elo_dict[home] + home_adv) - elo_dict[away]
            elo_diffs.append(current_diff)
            if pd.isna(row['FTHG']) or pd.isna(row['FTAG']): continue 
            actual_score = 1.0 if row['FTHG'] > row['FTAG'] else (0.5 if row['FTHG'] == row['FTAG'] else 0.0)
            margin = abs(row['FTHG'] - row['FTAG'])
            g_multiplier = 1.0 if margin < 2 else (1.5 if margin == 2 else (11 + margin) / 8)
            expected = 1 / (1 + 10 ** (-current_diff / 400))
            shift = k_factor * g_multiplier * (actual_score - expected)
            elo_dict[home] += shift
            elo_dict[away] -= shift
        return elo_diffs

    df = df.copy()
    df['Elo_Diff'] = calculate_elo_engine(df)
    
    # 动能引擎
    df['h_pts'] = df['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    df['h_form'] = df.groupby('HomeTeam')['h_pts'].transform(lambda x: x.rolling(5, closed='left').mean()).fillna(1.0)
    df['a_pts'] = df['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
    df['a_form'] = df.groupby('AwayTeam')['a_pts'].transform(lambda x: x.rolling(5, closed='left').mean()).fillna(1.0)
    df['Form_Diff'] = df['h_form'] - df['a_form']
    
    # xG 引擎
    df['h_scored'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(10, closed='left').mean()).fillna(1.5)
    df['h_conceded'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.rolling(10, closed='left').mean()).fillna(1.5)
    df['a_scored'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(10, closed='left').mean()).fillna(1.5)
    df['a_conceded'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.rolling(10, closed='left').mean()).fillna(1.5)
    df['xG_Diff'] = ((df['h_scored'] + df['a_conceded'])/2) - ((df['a_scored'] + df['h_conceded'])/2)
    
    # 市场先验
    h_col = 'B365H' if 'B365H' in df.columns else ('BbAvH' if 'BbAvH' in df.columns else None)
    d_col = 'B365D' if 'B365D' in df.columns else ('BbAvD' if 'BbAvD' in df.columns else None)
    a_col = 'B365A' if 'B365A' in df.columns else ('BbAvA' if 'BbAvA' in df.columns else None)
    
    if h_col and d_col and a_col:
        margin = (1/df[h_col] + 1/df[d_col] + 1/df[a_col])
        df['Prob_H_Bookie'] = (1/df[h_col]) / margin
        df['Prob_D_Bookie'] = (1/df[d_col]) / margin
        df['Prob_A_Bookie'] = (1/df[a_col]) / margin
    else:
        df['Prob_H_Bookie'], df['Prob_D_Bookie'], df['Prob_A_Bookie'] = 0.33, 0.33, 0.33
    
    return df

@st.cache_resource(show_spinner=False)
def train_model(df):
    features = ['Elo_Diff', 'Form_Diff', 'xG_Diff', 'Prob_H_Bookie', 'Prob_D_Bookie', 'Prob_A_Bookie']
    train_df = df.dropna(subset=['FTR'] + features).copy()
    label_map = {'A': 0, 'D': 1, 'H': 2}
    model = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42)
    model.fit(train_df[features], train_df['FTR'].map(label_map))
    return model, features

# ==========================================
# 2. 侧边栏交互与逻辑
# ==========================================
with st.sidebar:
    st.header("⚙️ Sniper 控制台")
    
    # 联赛多选
    league_options = {"英超 (EPL)": "E0", "西甲 (La Liga)": "SP1"}
    selected_names = st.multiselect("选择分析联赛", options=list(league_options.keys()), default=list(league_options.keys()))
    selected_codes = [league_options[name] for name in selected_names]
    
    # 日期筛选
    today = datetime.now()
    date_range = st.date_input("日期区间", [today.replace(month=1, day=1), today])
    
    # 指定文件列表
    target_files = ['SP12324.csv', 'ESP12425.csv', 'SP1.csv', 'E0_2324.CSV', 'E0_2425.CSV', 'E0_2526.CSV']
    
    st.divider()
    run_btn = st.button("🚀 启动 AI 预测", use_container_width=True)

# ==========================================
# 3. 主程序渲染
# ==========================================
st.title("🎯 Sniper AI 预测终端")

if run_btn:
    all_data = []
    with st.spinner("读取历史数据中..."):
        for f in target_files:
            if not os.path.exists(f): continue
            
            # 自动分类逻辑
            f_up = f.upper()
            is_epl = "E0" in f_up
            is_laliga = "SP1" in f_up or "ESP" in f_up
            
            if (is_epl and "E0" in selected_codes) or (is_laliga and "SP1" in selected_codes):
                tmp = pd.read_csv(f)
                tmp['Date'] = pd.to_datetime(tmp['Date'], dayfirst=True, errors='coerce')
                tmp['League_Tag'] = "英超" if is_epl else "西甲"
                all_data.append(tmp)
    
    if not all_data:
        st.error("❌ 未找到对应的 CSV 文件，请检查本地目录。")
    else:
        full_df = pd.concat(all_data).sort_values('Date').reset_index(drop=True)
        
        with st.spinner("AI 特征引擎计算中..."):
            processed_df = process_features(full_df)
            model, features = train_model(processed_df)
            
            # 日期过滤
            if len(date_range) == 2:
                start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                display_df = processed_df[(processed_df['Date'] >= start_dt) & (processed_df['Date'] <= end_dt)].copy()
            else:
                display_df = processed_df.tail(10)

        if display_df.empty:
            st.warning("所选日期范围内暂无比赛。")
        else:
            # 预测概率
            probs = model.predict_proba(display_df[features])
            display_df['p_A'], display_df['p_D'], display_df['p_H'] = probs[:, 0], probs[:, 1], probs[:, 2]
            
            st.success(f"✅ 分析完成：已定位 {len(display_df)} 场核心赛事")

            # 遍历渲染 (解决 HTML 缩进问题)
            for _, row in display_df.sort_values('Date', ascending=False).iterrows():
                ph, pd_, pa = row['p_H'], row['p_D'], row['p_A']
                
                # 决策逻辑
                if ph > 0.55:
                    cls, badge, reco = "safe-home", "🟢 主场稳胆", "主队胜算极大，博胜首选"
                elif pa > 0.50:
                    cls, badge, reco = "safe-away", "🔵 降维打击", "客队优势明显，建议博负"
                else:
                    cls, badge, reco = "warning", "🟡 防冷预警", "局势不明，建议防平或跳过"

                # 构建 HTML 字符串 (左对齐，防止触发 Markdown 代码块)
                card_html = f"""
<div class="prediction-card {cls}">
    <div style="display: flex; justify-content: space-between;">
        <span style="font-weight: bold; color: #333;">{row['League_Tag']} | {row['Date'].strftime('%Y-%m-%d')}</span>
        <span style="font-size: 0.8em; color: #999;">Sniper V28.0</span>
    </div>
    <h3 style="margin: 12px 0;">{row['HomeTeam']} <span style="color:#bbb; font-weight:normal;">vs</span> {row['AwayTeam']}</h3>
    <div class="prob-container">
        <div class="prob-item">
            <span class="prob-label">主胜 (H)</span>
            <span class="prob-value" style="color: #28a745;">{ph:.1%}</span>
        </div>
        <div class="prob-item">
            <span class="prob-label">平局 (D)</span>
            <span class="prob-value" style="color: #6c757d;">{pd_:.1%}</span>
        </div>
        <div class="prob-item">
            <span class="prob-label">客胜 (A)</span>
            <span class="prob-value" style="color: #17a2b8;">{pa:.1%}</span>
        </div>
    </div>
    <div style="border-top: 1px dashed #ddd; padding-top: 10px; margin-top: 5px; font-size: 0.95em;">
        <b>AI 决策：</b> <span>{badge} —— {reco}</span>
    </div>
</div>"""
                st.markdown(card_html, unsafe_allow_html=True)
else:
    st.info("👈 请在左侧选择联赛和日期区间，点击按钮开始分析。")