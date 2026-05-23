import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# 1. 페이지 설정 및 하이테크 테마 정의
st.set_page_config(page_title="Quant Trading System", layout="wide")
st.title("🛡️ 실전 퀀트 투자자용 대시보드 (승률 백테스팅 & 실적 검증 시스템)")

THEME_MAP = {
    "SOXX (반도체)": {"etf": "SOXX", "peers": ["NVDA", "AVGO", "AMD", "QCOM", "TXN"]},
    "AI-N-ROBO (인공지능/로봇)": {"etf": "BOTZ", "peers": ["NVDA", "ISRG", "ASML", "KEYENCE", "PATH"]},
    "ARKX (우주항공/디펜스)": {"etf": "ARKX", "peers": ["LMT", "RTX", "NOC", "LHX", "AMZN"]},
    "CLOU (클라우드/SaaS)": {"etf": "CLOU", "peers": ["MSFT", "ORCL", "NOW", "CRM", "WDAY"]},
    "LIT (2차전지/전기차)": {"etf": "LIT", "peers": ["TSLA", "BYDDF", "ALB", "SQM", "Panasonic"]},
    "IBB (바이오테크)": {"etf": "IBB", "peers": ["VRTX", "REGN", "AMGN", "GILD", "BIIB"]}
}

# 2. 사이드바 - 설정
st.sidebar.header("⚙️ 실전 퀀트 매매 설정")
ticker_input = st.sidebar.text_input("분석할 주식 티커 입력:", value="NVDA").upper().strip()
theme_selected = st.sidebar.selectbox("비교할 전문 테마 세부 섹터:", options=list(THEME_MAP.keys()))

theme_info = THEME_MAP[theme_selected]
sector_etf = theme_info["etf"]
peers = theme_info["peers"]

# [데이터 엔진] 주가 데이터 및 재무 데이터 로드 (캐싱)
@st.cache_data(ttl=3600)
def fetch_quant_data(main_ticker, etf_ticker, peer_tickers):
    all_tickers = list(set(["^GSPC", etf_ticker, main_ticker] + peer_tickers))
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
def fetch_financial_trends(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        # 연간 재무제표 로드
        financials = t_obj.financials
        if financials.empty:
            return None
        
        # 필요한 항목 추출 (Diluted EPS 및 Total Revenue)
        # yfinance 항목 명칭 방어 코드 처리
        rev_key = [k for k in financials.index if 'Revenue' in k or 'Total Revenue' in k]
        eps_key = [k for k in financials.index if 'EPS' in k or 'Diluted' in k]
        
        df_fin = pd.DataFrame()
        if rev_key: df_fin['Revenue'] = financials.loc[rev_key[0]]
        if eps_key: df_fin['EPS'] = financials.loc[eps_key[0]]
        
        # 날짜 내림차순 정렬을 오름차순으로 변경
        df_fin = df_fin.sort_index(ascending=True)
        df_fin.index = pd.to_datetime(df_fin.index).year
        return df_fin
    except:
        return None

try:
    with st.spinner("시장이격도 연산 및 과거 5년 백테스팅 엔진 구동 중..."):
        df_price = fetch_quant_data(ticker_input, sector_etf, peers)
        df_fin = fetch_financial_trends(ticker_input)
        
    if ticker_input not in df_price.columns:
        st.error(f"'{ticker_input}' 데이터를 가져오지 못했습니다. 미국 주식 티커인지 확인해 주세요.")
    else:
        # 기초 가공
        df_returns = df_price.pct_change().dropna() * 100
        df_cum_returns = (df_price / df_price.iloc[0] - 1) * 100
        
        X_sp = df_returns[["^GSPC"]].values
        y_stock = df_returns[ticker_input].values
        
        # 선형회귀 및 잔차 계산
        model_sp = LinearRegression().fit(X_sp, y_stock)
        residuals = y_stock - model_sp.predict(X_sp)
        cum_residuals = np.cumsum(residuals)
        res_std = np.std(cum_residuals)
        
        # 데이터프레임에 잔차 누적값 합산 및 시계열 정렬
        df_analysis = df_returns.copy()
        df_analysis['cum_residual'] = cum_residuals
        df_analysis['price'] = df_price.loc[df_analysis.index, ticker_input]
        
        # --- [실전 백테스팅 오라클 엔진 알고리즘] ---
        # 저평가 영역 기준선 정의 (-1.5 시그마 이하 진입 시점 포착)
        buy_signal_threshold = -1.5 * res_std
        
        # 과매도 상태 진입한 날짜 마킹
        df_analysis['signal'] = df_analysis['cum_residual'] <= buy_signal_threshold
        
        # 신호가 연속으로 뜰 때 첫 진입 시점만 솎아내기
        df_analysis['signal_start'] = df_analysis['signal'] & (~df_analysis['signal'].shift(1).fillna(False))
        signal_dates = df_analysis[df_analysis['signal_start']].index
        
        # 이후 수익률 추적 연산
        win_20, win_60 = [], []
        ret_20, ret_60 = [], []
        
        for d in signal_dates:
            idx = df_price.index.get_loc(d)
            # 20거래일 뒤 데이터가 존재하는지 검사
            if idx + 20 < len(df_price):
                r20 = (df_price.iloc[idx+20][ticker_input] / df_price.iloc[idx][ticker_input] - 1) * 100
                ret_20.append(r20)
                win_20.append(r20 > 0)
            # 60거래일 뒤 데이터가 존재하는지 검사
            if idx + 60 < len(df_price):
                r60 = (df_price.iloc[idx+60][ticker_input] / df_price.iloc[idx][ticker_input] - 1) * 100
                ret_60.append(r60)
                win_60.append(r60 > 0)
                
        # --- 화면 레이아웃 배치 ---
        
        # 🟢 SECTION 1: 백테스팅 스코어보드
        st.subheader("🎯 1. 통계적 저평가(-1.5σ) 구간 매수 시 과거 승률 백테스팅")
        st.markdown(f"**과거 5년간 시장 대비 과매도 기회 신호 포착 횟수:** `{len(signal_dates)}회` (일시적 쏠림이 아닌 첫 진입일 기준)")
        
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        if len(win_20) > 0:
            b_col1.metric(label="진입 1달 후 (20일) 승률", value=f"{np.mean(win_20)*100:.1f}%")
            b_col2.metric(label="1달 후 평균 수익률", value=f"{np.mean(ret_20):+.2f}%")
        else:
            b_col1.metric(label="진입 1달 후 승률", value="데이터 부족")
            
        if len(win_60) > 0:
            b_col3.metric(label="진입 3달 후 (60일) 승률", value=f"{np.mean(win_60)*100:.1f}%", delta="추천 타임프레임")
            b_col4.metric(label="3달 후 평균 수익률", value=f"{np.mean(ret_60):+.2f}%")
        else:
            b_col3.metric(label="진입 3달 후 승률", value="데이터 부족")
            
        st.markdown("---")
        
        # 🟢 SECTION 2: 밸류트랩 방지 펀더멘탈 실적 차트
        st.subheader(f"🛡️ 2. {ticker_input} 실적 트렌드 검증 (밸류 트랩 방지 시스템)")
        if df_fin is not None and not df_fin.empty:
            st.markdown("주가가 아무리 통계적 저평가 영역이어도 **아래의 연간 EPS(주당순이익)와 매출액 추세**가 무너지고 있다면 진입을 유보해야 합니다.")
            
            fig_fin = make_subplots(specs=[[{"secondary_y": True}]])
            fig_fin.add_trace(go.Bar(x=df_fin.index, y=df_fin['Revenue']/1e9 if 'Revenue' in df_fin.columns else [0], name="연간 매출액 ($B)", marker_color='rgba(100, 150, 250, 0.6)'), secondary_y=False)
            fig_fin.add_trace(go.Scatter(x=df_fin.index, y=df_fin['EPS'] if 'EPS' in df_fin.columns else [0], mode='lines+markers', name="Diluted EPS ($)", line=dict(color='orange', width=3)), secondary_y=True)
            
            fig_fin.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=10, b=10), hovermode="x unified")
            fig_fin.update_xaxes(type='category')
            st.plotly_chart(fig_fin, use_container_width=True)
        else:
            st.warning("이 티커의 최신 재무제표 펀더멘탈 데이터를 야후 파이낸스에서 연동하지 못했습니다.")
            
        st.markdown("---")

        # 🟢 SECTION 3: 시계열 잔차 및 진입 신호 매핑 그래프
        st.subheader("📈 3. S&P 500 대비 이격도 및 퀀트 진입 신호 위치 타임라인")
        
        fig1 = go.Figure()
        # 잔차선
        fig1.add_trace(go.Scatter(x=df_returns.index, y=cum_residuals, mode='lines', name='추세 이격도(잔차 누적)', line=dict(color='blue', width=2)))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[0]*len(df_returns), mode='lines', name='평균 추세', line=dict(color='black', dash='dash')))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[2*res_std]*len(df_returns), mode='lines', name='과열 고평가 (+2σ)', line=dict(color='red', dash='dot')))
        fig1.add_trace(go.Scatter(x=df_returns.index, y=[buy_signal_threshold]*len(df_returns), mode='lines', name='백테스팅 진입선 (-1.5σ)', line=dict(color='green', dash='dot')))
        
        # 그래프에 실제 백테스팅 매수 신호가 발생했던 포인트 도장 찍기
        fig1.add_trace(go.Scatter(
            x=signal_dates, y=df_analysis.loc[signal_dates, 'cum_residual'],
            mode='markers', name='🔮 과거 진입 신호 발생 포인터',
            marker=dict(color='gold', size=11, symbol='star', line=dict(color='black', width=1))
        ))
        
        fig1.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")

        # 🟢 SECTION 4: 테마 내 5개년 누적 성과 비교
        st.subheader(f"🎯 4. {theme_selected} 생태계 내 주도주 5개년 상대 누적 성과")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[ticker_input], mode='lines', name=f"★ 내 종목: {ticker_input}", line=dict(width=4, color='red')))
        if sector_etf in df_cum_returns.columns:
            fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[sector_etf], mode='lines', name=f"🧬 테마 벤치마크: {sector_etf}", line=dict(width=2.5, color='black', dash='dash')))
        
        for peer in peers:
            if peer in df_cum_returns.columns and peer != ticker_input:
                fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[peer], mode='lines', name=peer, line=dict(width=1.5), opacity=0.4))
                
        fig2.update_layout(template="plotly_white", height=450, xaxis_title="날짜", yaxis_title="누적 수익률 (%)", margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"퀀트 엔진 실행 도중 예외가 발생했습니다: {e}")
