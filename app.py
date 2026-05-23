import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# 1. 페이지 및 섹터 데이터 설정
st.set_page_config(page_title="Advanced Stock Analysis", layout="wide")
st.title("📊 S&P 500 & 섹터 대비 시계열 고/저평가 및 주도주 비교 대시보드")

# 미국 11대 섹터 매핑 테이블 (대표 ETF 및 상위 주도주 5개)
SECTOR_MAP = {
    "XLK": {"name": "Technology (기술)", "peers": ["MSFT", "AAPL", "NVDA", "AVGO", "ORCL"]},
    "XLY": {"name": "Consumer Discretionary (임의소비재)", "peers": ["AMZN", "TSLA", "HD", "NKE", "MCD"]},
    "XLC": {"name": "Communication Services (통신)", "peers": ["META", "GOOGL", "NFLX", "TMUS", "DIS"]},
    "XLF": {"name": "Financials (금융)", "peers": ["BRK-B", "JPM", "V", "MA", "BAC"]},
    "XLV": {"name": "Health Care (헬스케어)", "peers": ["LLY", "UNH", "JNJ", "MRK", "ABBV"]},
    "XLP": {"name": "Consumer Staples (필수소비재)", "peers": ["PG", "COST", "WMT", "KO", "PEP"]},
    "XLE": {"name": "Energy (에너지)", "peers": ["XOM", "CVX", "COP", "SLB", "EOG"]},
    "XLI": {"name": "Industrials (산업재)", "peers": ["GE", "CAT", "UNP", "HON", "RTX"]},
    "XLB": {"name": "Materials (소재)", "peers": ["LIN", "APD", "SHW", "FCX", "NEM"]},
    "XLRE": {"name": "Real Estate (부동산)", "peers": ["PLD", "AMT", "EQIX", "WELL", "CCI"]},
    "XLU": {"name": "Utilities (유틸리티)", "peers": ["NEE", "SO", "DUK", "CEG", "AEP"]}
}

# 티커별 섹터 자동 매칭용 가이드 (사용자 편의용 기본 매칭)
TICKER_TO_SECTOR = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "ORCL": "XLK",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY",
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC",
    "JPM": "XLF", "BRK-B": "XLF", "LLY": "XLV", "UNH": "XLV", "WMT": "XLP", "PG": "XLP",
    "XOM": "XLE", "CVX": "XLE", "GE": "XLI", "CAT": "XLI"
}

# 2. 사이드바 입력 설정
st.sidebar.header("⚙️ 분석 설정")
ticker_input = st.sidebar.text_input("분석할 주식 티커:", value="AAPL").upper().strip()

# 해당 티커의 섹터 추정 및 선택
default_sector = TICKER_TO_SECTOR.get(ticker_input, "XLK")
sector_key = st.sidebar.selectbox(
    "종목의 해당 섹터 ETF를 선택하세요:", 
    options=list(SECTOR_MAP.keys()), 
    index=list(SECTOR_MAP.keys()).index(default_sector),
    format_func=lambda x: f"{x} - {SECTOR_MAP[x]['name']}"
)

sector_info = SECTOR_MAP[sector_key]
peers = sector_info["peers"]

# 데이터 다운로드 함수 (캐싱 처리)
@st.cache_data(ttl=3600)
def fetch_all_data(main_ticker, sector_etf, peer_list):
    all_tickers = ["^GSPC", sector_etf, main_ticker] + peer_list
    raw_data = yf.download(all_tickers, period="5y", auto_adjust=True)
    
    # 구조 평탄화 및 Close 가격만 추출
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)
    
    # 다운로드된 데이터에서 Close 컬럼 가려내기 (yfinance 버전에 따라 일괄 다운 시 MultiIndex 처리 분기)
    # 안전하게 각 티커별 Close를 쪼개어 결합
    df_close = pd.DataFrame()
    for t in all_tickers:
        try:
            t_data = yf.download(t, period="5y", auto_adjust=True)
            if isinstance(t_data.columns, pd.MultiIndex):
                df_close[t] = t_data['Close'].iloc[:, 0] if t_data['Close'].shape[1] > 1 else t_data['Close']
            else:
                df_close[t] = t_data['Close']
        except:
            pass
            
    return df_close.dropna()

try:
    with st.spinner("미국 증시 데이터 및 섹터 주도주 데이터를 동시 로드 중..."):
        df_price = fetch_all_data(ticker_input, sector_key, peers)
        
    if ticker_input not in df_price.columns:
        st.error(f"'{ticker_input}' 데이터를 불러오지 못했습니다. 티커를 확인해 주세요.")
    else:
        # --- [데이터 가공] ---
        # 1. 일별 수익률 계산 (회귀분석용)
        df_returns = df_price.pct_change().dropna() * 100
        
        # 2. 누적 수익률 계산 (시계열 비교용)
        df_cum_returns = (df_price / df_price.iloc[0] - 1) * 100
        
        # --- [1번 기능: S&P 500 대비 시계열 고/저평가 밴드] ---
        # 수익률 기반 리니어 리그레션 실행
        X_sp = df_returns[["^GSPC"]].values
        y_stock = df_returns[ticker_input].values
        
        model_sp = LinearRegression()
        model_sp.fit(X_sp, y_stock)
        
        # 일별 '예측 수익률'과 '실제 수익률'의 차이 = 잔차(Residual)
        predicted_returns = model_sp.predict(X_sp)
        residuals = y_stock - predicted_returns
        
        # 잔차의 누적값(Cumulative Residuals)을 구해 시계열적인 고/저평가 트렌드 산출
        cum_residuals = np.cumsum(residuals)
        res_std = np.std(cum_residuals)
        
        # 시계열 차트 그리기 1
        st.subheader(f"1. S&P 500 대비 고평가 / 저평가 시계열 추세 ({ticker_input})")
        st.markdown("> **차트 해석:** 중앙선(0)은 S&P 500과 완벽하게 발맞추어 간 평균 추세입니다. 선이 **상단 빨간 밴드(+2σ)**에 근접하면 시장 대비 과열(고평가) 상태이므로 조심해야 하며, **하단 초록 밴드(-2σ)**에 도달하면 시장 대비 과매도(저평가) 상태로 **매력적인 진입 타이밍**으로 볼 수 있습니다.")
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_returns.index, y=cum_residuals, mode='lines', name='S&P 500 대비 이격도(잔차 누적)', line=dict(color='blue', width=2)))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[0]*len(df_returns), mode='lines', name='평균 추세선', line=dict(color='black', dash='dash')))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[2*res_std]*len(df_returns), mode='lines', name='고평가 임계선 (+2σ)', line=dict(color='red', dash='dot')))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[-2*res_std]*len(df_returns), mode='lines', name='저평가 기회선 (-2σ)', line=dict(color='green', dash='dot')))
        
        fig1.update_layout(template="plotly_white", height=400, xaxis_title="날짜", yaxis_title="상대적 과열/과매도 강도", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig1, use_container_width=True)
        
        
        # --- [2번 기능: 섹터 ETF 대비 상대 강도 및 추세선] ---
        st.subheader(f"2. {sector_info['name']} 섹터 ETF({sector_key}) 대비 누적 성과 비교")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[ticker_input], mode='lines', name=f"{ticker_input} 누적수익률", line=dict(width=2.5, color='#636EFA')))
        fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[sector_key], mode='lines', name=f"{sector_key} (섹터평균) 누적수익률", line=dict(width=2, color='#AB63FA', dash='dash')))
        
        fig2.update_layout(template="plotly_white", height=400, xaxis_title="날짜", yaxis_title="누적 수익률 (%)", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)
        
        
        # --- [3번 기능: 섹터 내 상위 주도주 5개와 비교] ---
        st.subheader(f"3. 섹터 내 TOP 5 주도주 vs {ticker_input} 5개년 성과 비교")
        st.markdown("선택한 종목이 섹터 대장주들(시가총액 상위) 사이에서 얼마나 강한 포지션을 잡고 있는지 확인합니다.")
        
        fig3 = go.Figure()
        # 내 종목 강조
        fig3.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[ticker_input], mode='lines', name=f"★ {ticker_input}", line=dict(width=4, color='red')))
        
        # 경쟁 주도주들 추가
        for peer in peers:
            if peer in df_cum_returns.columns and peer != ticker_input:
                fig3.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[peer], mode='lines', name=peer, line=dict(width=1.5), opacity=0.6))
                
        fig3.update_layout(template="plotly_white", height=500, xaxis_title="날짜", yaxis_title="누적 수익률 (%)", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

except Exception as e:
    st.error(f"대시보드 생성 중 오류 발생: {e}")
