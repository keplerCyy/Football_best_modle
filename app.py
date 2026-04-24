import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 0. UI 样式配置
# ==========================================
st.set_page_config(page_title="Sniper AI V29.5", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .prediction-card {
        background: white; border-radius: 12px; padding: 20px;
        margin-bottom: 20px; border-left: 8px solid #ddd;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-val { font-size: 1.5em; font-weight: 800; color: #222; }
    .status-tag { padding: 3px 12px; border-radius: 15px; font-weight: bold; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心类：SniperFetcher (完全整合你的解析逻辑)
# ==========================================
class SniperFetcher:
    def __init__(self, api_key):
        self.base_url = "https://sofascore.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "sofascore.p.rapidapi.com",
        }

    def get_matches(self, category_id, target_tournament_id, date_str):
        """完全参照你提供的扫描逻辑"""
        endpoint = f"{self.base_url}/tournaments/get-scheduled-events"
        params = {"categoryId": str(category_id), "date": date_str}
        try:
            res = requests.get(endpoint, headers=self.headers, params=params, timeout=10)
            events = res.json().get("events", [])
            results = []
            for ev in events:
                ut_id = str(ev.get("tournament", {}).get("uniqueTournament", {}).get("id", ""))
                if ut_id != str(target_tournament_id): continue
                
                results.append({
                    "match_id": ev.get("id"),
                    "home_team": ev.get("homeTeam", {}).get("name"),
                    "away_team": ev.get("awayTeam", {}).get("name"),
                    "kickoff": pd.to_datetime(ev.get("startTimestamp"), unit="s").strftime("%H:%M"),
                    "status": ev.get("status", {}).get("type")
                })
            return pd.DataFrame(results)
        except: return pd.DataFrame()

    def get_real_odds(self, match_id):
        """
        🚀 直接搬运你代码中的“究极解析逻辑”
        确保处理多级嵌套、分数赔率转换等所有边缘情况
        """
        url = f"{self.base_url}/matches/get-all-odds"
        try:
            res = requests.get(url, headers=self.headers, params={"matchId": match_id}, timeout=12)
            data = res.json()
            markets = data.get('markets', [])
            if not markets and data.get('providers'):
                markets = data.get('providers')[0].get('markets', [])

            for m in markets:
                m_name = str(m.get('marketName', '')).lower()
                f_name = str(m.get('filterName', '')).lower()
                
                # 寻找 1X2 市场
                is_1x2 = any(k in m_name or k in f_name for k in ["1x2", "3-way", "result", "full time", "winner"])
                is_not_half = all(k not in m_name and k not in f_name for k in ["half", "quarter"])

                if is_1x2 and is_not_half:
                    choices = m.get('choices', [])
                    if len(choices) >= 3:
                        res_dict = {"home": None, "draw": None, "away": None}
                        for c in choices:
                            name = str(c.get('name', '')).upper()
                            val = c.get('decimalValue')
                            # 关键：分数赔率转换逻辑 (n/d + 1)
                            if not val and c.get('fractionalValue') and '/' in str(c.get('fractionalValue')):
                                n, d = str(c.get('fractionalValue')).split('/')
                                val = (float(n) / float(d)) + 1.0
                            
                            if val:
                                if name in ['1', 'HOME']: res_dict["home"] = round(float(val), 2)
                                elif name in ['X', 'DRAW']: res_dict["draw"] = round(float(val), 2)
                                elif name in ['2', 'AWAY']: res_dict["away"] = round(float(val), 2)
                        
                        if all(v is not None for v in res_dict.values()): 
                            return res_dict
            return None
        except: return None

# ==========================================
# 2. 自动化模型训练 (基于 B365 隐含概率)
# ==========================================
@st.cache_resource
def train_sniper_model():
    data_files = ['SP12324.csv', 'ESP12425.csv', 'SP1.csv', 'E0_2324.CSV', 'E0_2425.CSV', 'E0_2526.CSV']
    all_data = []
    for f in data_files:
        if os.path.exists(f):
            try:
                tmp = pd.read_csv(f)
                tmp.columns = [c.upper() for c in tmp.columns]
                all_data.append(tmp)
            except: pass
    if not all_data: return None
    df = pd.concat(all_data, ignore_index=True).dropna(subset=['B365H', 'B365D', 'B365A', 'FTR'])
    # 特征工程：隐含概率
    X = pd.DataFrame({'PH': 1/df['B365H'], 'PD': 1/df['B365D'], 'PA': 1/df['B365A']})
    y = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    model.fit(X, y)
    return model

# ==========================================
# 3. 主界面逻辑
# ==========================================
def main():
    st.title("🎯 Sniper AI 预测终端 (V29.5)")
    
    DEFAULT_KEY = "1c77b88820mshbd12fc151d1a3b1p161770jsn1e5a287b9915"
    fetcher = SniperFetcher(DEFAULT_KEY)
    model = train_sniper_model()

    if "match_list" not in st.session_state: st.session_state.match_list = pd.DataFrame()

    # --- 侧边栏 ---
    st.sidebar.header("🕹️ 控制台")
    league_choice = st.sidebar.selectbox("选择联赛", ["英超 (Premier League)", "西甲 (La Liga)"])
    date_input = st.sidebar.date_input("分析日期", datetime.now())
    
    league_map = {"英超 (Premier League)": (1, 17), "西甲 (La Liga)": (32, 8)}
    
    if st.sidebar.button("🔍 扫描赛事"):
        cat_id, tour_id = league_map[league_choice]
        with st.spinner("正在同步赛程..."):
            df = fetcher.get_matches(cat_id, tour_id, date_input.strftime('%Y-%m-%d'))
            if not df.empty:
                df.insert(0, "分析", False)
                st.session_state.match_list = df
            else: st.sidebar.warning("该日期暂无赛事")

    # --- 阶段 1：勾选列表 ---
    if not st.session_state.match_list.empty:
        st.subheader("📋 待处理赛程单")
        edited_df = st.data_editor(
            st.session_state.match_list,
            column_config={"分析": st.column_config.CheckboxColumn("选择", default=False), "match_id": None},
            disabled=["home_team", "away_team", "kickoff", "status"],
            hide_index=True, use_container_width=True, key="main_editor"
        )
        
        selected = edited_df[edited_df["分析"] == True]
        
        if not selected.empty:
            if st.button(f"🚀 执行量化深度分析 ({len(selected)} 场)", type="primary"):
                st.divider()
                for _, row in selected.iterrows():
                    render_report(row, fetcher, model)

def render_report(row, fetcher, model):
    with st.spinner(f"正在深度解析 {row['home_team']}..."):
        odds = fetcher.get_real_odds(row['match_id'])
        
        if not odds:
            st.error(f"❌ {row['home_team']} vs {row['away_team']}: 赔率解析失败 (API 未返回有效 1X2 市场)")
            return

        # AI 推理
        feat = pd.DataFrame([[1/odds['home'], 1/odds['draw'], 1/odds['away']]], columns=['PH', 'PD', 'PA'])
        probs = model.predict_proba(feat)[0]
        hp, dp, ap = probs[0], probs[1], probs[2]

        # 标签逻辑
        if hp > 0.58: label, color, icon = "主场稳胆", "#28a745", "🟢"
        elif ap > 0.48: label, color, icon = "降维打击", "#007bff", "🔵"
        else: label, color, icon = "均衡博弈", "#ffc107", "🟡"

        # 渲染卡片
        st.markdown(f"""
        <div class="prediction-card" style="border-left-color: {color};">
            <div style="display: flex; justify-content: space-between;">
                <span style="color:#666;">{row['kickoff']} | {row['status']}</span>
                <span class="status-tag" style="background:{color}22; color:{color};">{icon} {label}</span>
            </div>
            <h3 style="text-align:center; margin:15px 0;">{row['home_team']} <small>vs</small> {row['away_team']}</h3>
            <div style="display: flex; justify-content: space-around; background:#f9f9f9; padding:15px; border-radius:8px;">
                <div style="text-align:center;"><div style="font-size:0.8em; color:#666;">主胜 (H)</div><div class="metric-val">{hp:.1%}</div></div>
                <div style="text-align:center;"><div style="font-size:0.8em; color:#666;">平局 (D)</div><div class="metric-val">{dp:.1%}</div></div>
                <div style="text-align:center;"><div style="font-size:0.8em; color:#666;">客胜 (A)</div><div class="metric-val">{ap:.1%}</div></div>
            </div>
            <div style="margin-top:10px; font-size:0.85em; color:#999; text-align:right;">
                参考赔率: {odds['home']} / {odds['draw']} / {odds['away']}
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()