import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 1. 페이지 설정
st.set_page_config(page_title="Pure-Play Quant System", layout="wide")
st.title("🚀 실전 투자자용 초정밀 퓨어 테마 타겟팅 대시보드")

# [핵심 변경] 상용 ETF가 아닌, 실전 투자용 진짜 동종 주식군(Peer Group)으로 재구축
THEME_MAP = {
    "🌌 순수 우주항공/위성 스타트업": {
        "etf": "RKLB",  # 테마의 기준점으로 쓸 대장격 종목 (로켓랩)
        "peers": ["ASTS", "RKLB", "PL", "RDW", "BKSY"] # ASTS, 로켓랩, 플래닛랩스, 레드와이어, 블랙스카이
    },
    "🧠 AI 소프트웨어 / 퓨어 LLM 생태계": {
        "etf": "PLTR",
        "peers": ["PLTR", "AI", "SOUN", "BBAI", "PATH"] # 팔란티어, 씨쓰리에이아이, 사운드하운드 등
    },
    "🔬 차세대 반도체 장비 핵심 독점주": {
        "etf": "ASML",
        "peers": ["ASML", "AMAT", "LRCX", "KLAC", "TSM"] # 슈퍼 을(乙) 장비 생태계
    },
    "🚗 자율주행 및 로보틱스 혁신주": {
        "etf": "TSLA",
        "peers": ["TSLA", "MBLY", "NXPI", "VNE", "RBRK"]
    }
}

# 2. 사이드바 - 설정
st.sidebar.header("⚙️ 퓨어 퀀트 설정")
ticker_input = st.sidebar.text_input("분석할 주식 티커 입력:", value="ASTS").upper().strip()
theme_selected = st.sidebar.selectbox("정밀 타겟팅 비교 테마군 선택:", options=list(THEME_MAP.keys()))

theme_info = THEME_MAP[theme_selected]
sector_base = theme_info["etf"]
peers = theme_info["peers"]

# 데이터 다운로드 엔진 (캐싱)
@st.cache_data(ttl=3600)
def fetch_pure_data(main_ticker, base_ticker, peer_tickers):
    all_tickers = list(set(["^GSPC", base_ticker, main_ticker] + peer_tickers))
    df_close = pd.DataFrame()
    for t in all_tickers:
        try:
            t_data = yf.download(t, period="5y", auto_adjust=True)
            if not t_data.empty:
                df_close[t] = t_data['Close'].iloc[:, 0] if isinstance(t_data.columns, pd.MultiIndex) else t_data['Close']
        except:
            pass
    return df_close.dropna()

@st.cache_data(ttl=86400)
def fetch_diluted_eps(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        financials = t_obj.financials
        if financials.empty:
            return None
        
        # 최악의 물타기까지 방어하는 Diluted EPS 및 매출 트렌드 추출
        rev_key = [k for k in financials.index if 'Revenue' in k or 'Total Revenue' in k]
        eps_key = [k for k in financials.index if 'Diluted EPS' in k or 'Diluted' in k]
        
        df_fin = pd.DataFrame()
        if rev_key: df_fin['Revenue'] = financials.loc[rev_key[0]]
        if eps_key: df_fin['EPS'] = financials.loc[eps_key[0]]
        
        df_fin = df_fin.sort_index(ascending=True)
        df_fin.index = pd.to_datetime(df_fin.index).year
        return df_fin
    except:
        return None

try:
    with st.spinner("퓨어 플레이어 자산군 바스켓 동시 연산 중..."):
        df_price = fetch_pure_data(ticker_input, sector_base, peers)
        df_fin = fetch_diluted_eps(ticker_input)
        
    if ticker_input not in df_price.columns:
        st.error(f"'{ticker_input}' 데이터를 가져오지 못했습니다. 올바른 티커인지 확인해 주세요.")
    else:
        df_returns = df_price.pct_change().dropna() * 100
        df_cum_returns = (df_price / df_price.iloc[0] - 1) * 100
        
        X_sp = df_returns[["^GSPC"]].values
        y_stock = df_returns[ticker_input].values
        
        model_sp = LinearRegression().fit(X_sp, y_stock)
        residuals = y_stock - model_sp.predict(X_sp)
        cum_residuals = np.cumsum(residuals)
        res_std = np.std(cum_residuals)
        
        df_analysis = df_returns.copy()
        df_analysis['cum_residual'] = cum_residuals
        
        # 퀀트 시그널 추출 (-1.5 시그마 저평가 타점)
        buy_signal_threshold = -1.5 * res_std
        df_analysis['signal'] = df_analysis['cum_residual'] <= buy_signal_threshold
        df_analysis['signal_start'] = df_analysis['signal'] & (~df_analysis['signal'].shift(1).fillna(False))
        signal_dates = df_analysis[df_analysis['signal_start']].index
        
        win_60, ret_60 = [], []
        for d in signal_dates:
            idx = df_price.index.get_loc(d)
            if idx + 60 < len(df_price):
                r60 = (df_price.iloc[idx+60][ticker_input] / df_price.iloc[idx][ticker_input] - 1) * 100
                ret_60.append(r60)
                win_60.append(r60 > 0)

        # --- 레이아웃 출력 ---
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.subheader("🎯 통계적 저평가 진입 시 승률 백테스팅")
            if len(win_60) > 0:
                st.metric(label="3달 후 (60거래일) 상승 확률", value=f"{np.mean(win_60)*100:.1f}%")
                st.caption(f"지난 5년간 총 {len(signal_dates)}번의 -1.5σ 진입 타점이 있었습니다. 평균 수익률: {np.mean(ret_60):+.2f}%")
            else:
                st.info("신호 발생 이력이 부족합니다.")
                
        with col_metric2:
            st.subheader(f"🛡️ {ticker_input} 'Diluted EPS' 및 매출 트렌드")
            if df_fin is not None and not df_fin.empty:
                st.caption("희석 주당순이익(Diluted EPS) 선이 꺾이지 않고 버티거나 올라가는 중인지 확인하세요.")
                fig_fin = go.Figure()
                fig_fin.add_trace(go.Bar(x=df_fin.index, y=df_fin['Revenue']/1e6 if 'Revenue' in df_fin.columns else [0], name="매출 ($M)", marker_color='rgba(99, 110, 250, 0.5)'))
                fig_fin.add_trace(go.Scatter(x=df_fin.index, y=df_fin['EPS'] if 'EPS' in df_fin.columns else [0], mode='lines+markers', name="Diluted EPS ($)", line=dict(color='crimson', width=3)))
                fig_fin.update_layout(template="plotly_white", height=180, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                fig_fin.update_xaxes(type='category')
                st.plotly_chart(fig_fin, use_container_width=True)

        st.markdown("---")
        
        st.subheader("📈 S&P 500 대비 이격도 및 매수 진입 타점 (🔮)")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_returns.index, y=cum_residuals, mode='lines', name='추세 이격도(잔차 누적)', line=dict(color='blue', width=2)))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[0]*len(df_returns), mode='lines', name='평균 추세', line=dict(color='black', dash='dash')))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[2*res_std]*len(df_returns), mode='lines', name='과열 임계선', line=dict(color='red', dash='dot')))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[buy_signal_threshold]*len(df_returns), mode='lines', name='진입선 (-1.5σ)', line=dict(color='green', dash='dot')))
        fig1.add_trace(go.Scatter(x=signal_dates, y=df_analysis.loc[signal_dates, 'cum_residual'], mode='markers', name='진입 타점', marker=dict(color='gold', size=12, symbol='star', line=dict(color='black', width=1))))
        fig1.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")

        st.subheader(f"🎯 퓨어 플레이어 주식군 동행 성과 비교 ({theme_selected})")
        st.markdown("대형 방산주를 제외한 **순수 퓨어 기업들 사이에서 내 종목이 보이는 상대적 강도**를 추적합니다.")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[ticker_input], mode='lines', name=f"★ 내 종목: {ticker_input}", line=dict(width=4, color='red')))
        for peer in peers:
            if peer in df_cum_returns.columns and peer != ticker_input:
                fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[peer], mode='lines', name=peer, line=dict(width=1.5), opacity=0.5))
        fig2.update_layout(template="plotly_white", height=400, xaxis_title="날짜", yaxis_title="누적 수익률 (%)", margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"엔진 렌더링 에러: {e}")
