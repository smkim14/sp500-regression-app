import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 1. 페이지 설정
st.set_page_config(page_title="Hyper-Focused Stock Analyzer", layout="wide")
st.title("🚀 초세분화 테마별 주도주 비교 및 시장 추종 강도 분석 대시보드")

# 2. 세분화된 하이테크/전부 테마 맵핑 테이블 (대표 ETF 및 핵심 주도주 5개)
THEME_MAP = {
    "SOXX (반도체)": {"etf": "SOXX", "peers": ["NVDA", "AVGO", "AMD", "QCOM", "TXN"]},
    "AI-N-ROBO (인공지능/로봇)": {"etf": "BOTZ", "peers": ["NVDA", "ISRG", "ASML", "KEYENCE", "PATH"]},
    "ARKX (우주항공/디펜스)": {"etf": "ARKX", "peers": ["LMT", "RTX", "NOC", "LHX", "AMZN"]},
    "CLOU (클라우드/SaaS)": {"etf": "CLOU", "peers": ["MSFT", "ORCL", "NOW", "CRM", "WDAY"]},
    "LIT (2차전지/전기차)": {"etf": "LIT", "peers": ["TSLA", "BYDDF", "ALB", "SQM", "Panasonic"]},
    "IBB (바이오테크)": {"etf": "IBB", "peers": ["VRTX", "REGN", "AMGN", "GILD", "BIIB"]},
    "TAN (친환경/태양광)": {"etf": "TAN", "peers": ["FSLR", "ENPH", "SEDG", "RUN", "CSIQ"]}
}

# 3. 사이드바 - 설정 영역
st.sidebar.header("⚙️ 분석 설정")
ticker_input = st.sidebar.text_input("분석할 주식 티커 입력:", value="NVDA").upper().strip()

# 세분화 테마 선택
theme_selected = st.sidebar.selectbox("비교할 전문 테마 세부 섹터:", options=list(THEME_MAP.keys()))
theme_info = THEME_MAP[theme_selected]
sector_etf = theme_info["etf"]
peers = theme_info["peers"]

# 데이터 다운로드 함수 (캐싱 적용)
@st.cache_data(ttl=3600)
def fetch_financial_data(main_ticker, etf_ticker, peer_tickers):
    all_tickers = list(set(["^GSPC", etf_ticker, main_ticker] + peer_tickers))
    
    # yfinance 안정적 분할 다운로드 후 결합
    df_close = pd.DataFrame()
    for t in all_tickers:
        try:
            t_data = yf.download(t, period="5y", auto_adjust=True)
            if not t_data.empty:
                # 다중 인덱스 컬럼 방어 코드
                if isinstance(t_data.columns, pd.MultiIndex):
                    df_close[t] = t_data['Close'].iloc[:, 0]
                else:
                    df_close[t] = t_data['Close']
        except Exception:
            pass
    return df_close.dropna()

try:
    with st.spinner("야후 파이낸스에서 하이테크 테마 및 시장 데이터를 가져오는 중..."):
        df_price = fetch_financial_data(ticker_input, sector_etf, peers)
        
    if ticker_input not in df_price.columns:
        st.error(f"'{ticker_input}' 데이터를 찾을 수 없습니다. 미국 증시 티커인지 확인해주세요.")
    else:
        # --- [기본 데이터 가공] ---
        df_returns = df_price.pct_change().dropna() * 100
        df_cum_returns = (df_price / df_price.iloc[0] - 1) * 100
        
        # --- [1. S&P 500 추세 추종 강도 수치화 레이아웃] ---
        st.subheader("📊 1. S&P 500 평균 추세 추종 강도 및 민감도 분석")
        
        X_sp = df_returns[["^GSPC"]].values
        y_stock = df_returns[ticker_input].values
        
        # 전체 5년 선형회귀
        model_sp = LinearRegression().fit(X_sp, y_stock)
        overall_beta = model_sp.coef_[0]
        overall_r2 = model_sp.score(X_sp, y_stock)
        
        # 수치 대시보드 카드 배치
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="5개년 평균 추종도 (R-squared)", value=f"{overall_r2:.4f}")
            st.caption("1에 가까울수록 S&P 500과 똑같이 움직이며, 0에 가까울수록 시장과 무관하게 독자적으로 움직입니다.")
        with col2:
            st.metric(label="5개년 평균 민감도 (Beta)", value=f"{overall_beta:.4f}")
            st.caption("시장 변동 대비 배율입니다. 1보다 크면 시장보다 화끈하게 움직이고, 1보다 작으면 둔하게 움직입니다.")
        with col3:
            # 추종 강도 해석용 지표
            if overall_r2 >= 0.6:
                status = "🔄 시장 초동행형 (지수 영향 극대)"
            elif overall_r2 >= 0.3:
                status = "⚖️ 시장 동행 분산형 (독자 모멘텀 보유)"
            else:
                status = "🚀 개별 모멘텀 독립형 (디커플링 성향)"
            st.metric(label="종목 성향 진단", value=status)
            st.caption("R-squared 기준으로 분석한 이 종목의 주가 성격입니다.")
            
        # --- [60일 Rolling Beta 시계열 그래프 추가] ---
        # 시간에 따른 추종 강도 변화 추적
        rolling_window = 60
        cov_matrix = df_returns[ticker_input].rolling(rolling_window).cov(df_returns["^GSPC"])
        var_market = df_returns["^GSPC"].rolling(rolling_window).var()
        rolling_beta = (cov_matrix / var_market).dropna()
        
        st.markdown("#### 🔄 시간에 따른 시장 민감도(Rolling 60-Day Beta) 흐름")
        st.caption("특정 시점에 이 종목이 시장 대비 얼마나 민감해졌는지 추적합니다. 선이 위로 솟구칠 때 시장 주도력이 강해졌음을 의미합니다.")
        
        fig_beta = go.Figure()
        fig_beta.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta.values, mode='lines', name='60일 유동 베타', line=dict(color='#EF553B', width=2)))
        fig_beta.add_trace(go.Scatter(x=rolling_beta.index, y=[1.0]*len(rolling_beta), mode='lines', name='시장 평균선 (Beta=1.0)', line=dict(color='gray', dash='dash')))
        fig_beta.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_beta, use_container_width=True)

        st.markdown("---")

        # --- [2. 회귀선 기준 고/저평가 타임라인] ---
        st.subheader(f"📈 2. S&P 500 회귀선 기준 이격도 타임라인 ({ticker_input})")
        st.markdown("> 잔차가 **하단 초록 밴드(-2σ)**에 진입할 때가 시장 대비 펀더멘탈 낙폭 과대로 인한 **통계적 진입 찬스**입니다.")
        
        residuals = y_stock - model_sp.predict(X_sp)
        cum_residuals = np.cumsum(residuals)
        res_std = np.std(cum_residuals)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_returns.index, y=cum_residuals, mode='lines', name='추세 이격도(잔차 누적)', line=dict(color='blue', width=2)))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[0]*len(df_returns), mode='lines', name='평균 추세', line=dict(color='black', dash='dash')))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[2*res_std]*len(df_returns), mode='lines', name='과열 고평가 (+2σ)', line=dict(color='red', dash='dot')))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[-2*res_std]*len(df_returns), mode='lines', name='과매도 저평가 (-2σ)', line=dict(color='green', dash='dot')))
        fig1.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")

        # --- [3. 세분화 테마 ETF 및 TOP 5 주도주 비교] ---
        st.subheader(f"🎯 3. {theme_selected} 생태계 내 대장주들과의 5개년 누적 성과 비교")
        st.markdown(f"선택한 테마의 벤치마크 ETF인 **{sector_etf}** 및 해당 분야 시가총액 최상위 5개 기업군과 내 종목의 성장 속도를 정밀 비교합니다.")
        
        fig2 = go.Figure()
        # 내 종목 강조 (두꺼운 빨간색)
        fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[ticker_input], mode='lines', name=f"★ 내 종목: {ticker_input}", line=dict(width=4, color='red')))
        # 테마 ETF 강조 (두꺼운 검은 점선)
        if sector_etf in df_cum_returns.columns:
            fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[sector_etf], mode='lines', name=f"🧬 테마 평균: {sector_etf}", line=dict(width=3, color='black', dash='dash')))
        
        # 경쟁 피어 그룹 배치
        for peer in peers:
            if peer in df_cum_returns.columns and peer != ticker_input:
                fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[peer], mode='lines', name=peer, line=dict(width=1.5), opacity=0.5))
                
        fig2.update_layout(template="plotly_white", height=500, xaxis_title="날짜", yaxis_title="누적 수익률 (%)", margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"데이터 렌더링 중 에러가 발생했습니다: {e}")
