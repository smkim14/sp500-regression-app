import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 1. 페이지 설정
st.set_page_config(page_title="S&P 500 vs Stocks Regression", layout="wide")
st.title("📈 S&P 500 대비 개별 종목 회귀 분석 (최근 5개년)")
st.markdown("S&P 500(`^GSPC`) 수익률을 X축, 선택한 종목의 수익률을 Y축으로 설정하여 Linear Regression 분석을 수행합니다.")

# 2. 사이드바 - 티커 선택 및 데이터 설정
st.sidebar.header("⚙️ 설정")
ticker_input = st.sidebar.text_input("분석할 주식 티커를 입력하세요:", value="AAPL").upper().strip()

# 데이터 캐싱 (속도 및 중복 요청 방지)
@st.cache_data(ttl=3600)
def load_data(ticker):
    # 최신 yfinance 버전에 대응하기 위해 각각 다운로드 후 결합하는 방식 채택
    # auto_adjust=True 일 경우 'Close' 컬럼에 수정종가(Adjusted Close)가 자동 반영됩니다.
    sp500 = yf.download("^GSPC", period="5y", auto_adjust=True)
    stock = yf.download(ticker, period="5y", auto_adjust=True)
    
    if sp500.empty or stock.empty:
        return pd.DataFrame()
        
    # 복잡해진 2중 컬럼(Multi-Index) 구조를 1차원으로 단순화
    sp500.columns = sp500.columns.get_level_values(0)
    stock.columns = stock.columns.get_level_values(0)
    
    # 두 데이터셋의 종가(Close)만 추출하여 하나의 데이터프레임으로 병합
    df = pd.DataFrame({
        "^GSPC": sp500["Close"],
        ticker: stock["Close"]
    }).dropna()
    
    # 일별 수익률(Percentage Change) 계산 (%)
    returns = df.pct_change().dropna() * 100
    return returns

try:
    with st.spinner("야후 파이낸스에서 5개년 데이터를 불러오는 중..."):
        df_returns = load_data(ticker_input)
    
    if df_returns.empty:
        st.error(f"티커 '{ticker_input}'의 데이터를 가져오지 못했습니다. 올바른 티커(예: AAPL, TSLA, NVDA)인지 확인해주세요.")
    else:
        x_col = "^GSPC"
        y_col = ticker_input

        # 3. 선형 회귀 분석 데이터 준비
        X = df_returns[[x_col]].values  # S&P 500 수익률
        y = df_returns[y_col].values    # 개별 종목 수익률
        
        # 회귀 모델 훈련
        model = LinearRegression()
        model.fit(X, y)
        
        # 회귀 직선 상의 예측값 계산
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_range)
        
        # 회귀 지표 계산
        beta = model.coef_[0]
        alpha = model.intercept_
        r_squared = model.score(X, y)
        
        # 4. 주요 지표(Metric) 시각화
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Beta (시장 민감도)", value=f"{beta:.4f}")
        col2.metric(label="Alpha (초과 수익률)", value=f"{alpha:.4f}%")
        col3.metric(label="R-squared (모델 설명력)", value=f"{r_squared:.4f}")
        
        # 5. Plotly를 활용한 인터랙티브 그래프 생성
        fig = go.Figure()
        
        # 일별 수익률 산점도 (Scatter Plot)
        fig.add_trace(go.Scatter(
            x=df_returns[x_col],
            y=df_returns[y_col],
            mode='markers',
            name='일별 수익률 매핑',
            marker=dict(color='rgba(99, 110, 250, 0.4)', size=6),
            hovertemplate="S&P 500: %{x:.2f}%<br>"+f"{ticker_input}: "+"%{y:.2f}%<extra></extra>"
        ))
        
        # 추세선 (Regression Line)
        fig.add_trace(go.Scatter(
            x=X_range.flatten(),
            y=y_pred,
            mode='lines',
            name=f'회귀 직선 (y = {beta:.3f}x + {alpha:.3f})',
            line=dict(color='red', width=3)
        ))
        
        # 레이아웃 스타일 설정
        fig.update_layout(
            xaxis_title="S&P 500 일별 수익률 (%)",
            yaxis_title=f"{ticker_input} 일별 수익률 (%)",
            hovermode="closest",
            height=650,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        # 대시보드에 그래프 출력
        st.plotly_chart(fig, use_container_width=True)
        
except Exception as e:
    st.error(f"프로그램 실행 중 오류가 발생했습니다: {e}")
