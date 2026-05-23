import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 1. 페이지 설정
st.set_page_config(page_title="Custom Pure-Play Quant System", layout="wide")
st.title("🎯 실전 투자자용 다이렉트 매칭 (Direct-Indexing) 퀀트 대시보드")
st.markdown("상용 ETF의 왜곡을 제거하기 위해, 내가 분석할 종목과 완벽히 동종인 라이벌 주식군을 직접 입력하여 정밀 비교합니다.")

# 2. 사이드바 - 투자자 맞춤형 완전 자유 입력 설정
st.sidebar.header("⚙️ 유동적 그룹핑 설정")

# (1) 분석 중심 종목
main_ticker = st.sidebar.text_input("1. 기준이 될 중심 종목 티커:", value="ASTS").upper().strip()

# (2) 비교 대상 라이벌 종목들 (자유 입력)
default_peers = "ASTS, RKLB, PL, RDW, BKSY" if main_ticker in ["ASTS", "RKLB", "PL", "RDW", "BKSY"] else f"{main_ticker}, AAPL, MSFT, NVDA"
peers_input = st.sidebar.text_area(
    "2. 비교할 라이벌 종목 티커들을 입력하세요 (콤마 또는 공백 구분):", 
    value=default_peers
)

# 입력받은 문자열을 파싱하여 깔끔한 티커 리스트로 가공
peer_list = [t.strip().upper() for t in peers_input.replace(",", " ").split() if t.strip()]

# 중복 제거 및 기준 종목 포함 보장
if main_ticker not in peer_list and main_ticker:
    peer_list.insert(0, main_ticker)

# 데이터 다운로드 및 병합 엔진 (캐싱)
@st.cache_data(ttl=3600)
def fetch_custom_group_data(target_ticker, tickers_to_compare):
    # S&P 500 지수 항시 포함
    all_tickers = list(set(["^GSPC", target_ticker] + tickers_to_compare))
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
def fetch_diluted_eps_dynamic(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        financials = t_obj.financials
        if financials.empty:
            return None
        
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

# 가동 및 예외 방어
if not main_ticker:
    st.info("왼쪽 사이드바에 기준 종목 티커를 입력해 주세요.")
else:
    try:
        with st.spinner("사용자 정의 자산 바스켓 동시 연산 및 퀀트 백테스팅 중..."):
            df_price = fetch_custom_group_data(main_ticker, peer_list)
            df_fin = fetch_diluted_eps_dynamic(main_ticker)
            
        if main_ticker not in df_price.columns:
            st.error(f"기준 종목 '{main_ticker}'의 데이터를 가져오지 못했습니다. 미국 거래소 상장 티커인지 확인해 주세요.")
        else:
            # 기초 수익률 가공
            df_returns = df_price.pct_change().dropna() * 100
            df_cum_returns = (df_price / df_price.iloc[0] - 1) * 100
            
            X_sp = df_returns[["^GSPC"]].values
            y_stock = df_returns[main_ticker].values
            
            # 선형회귀 및 이격도(잔차) 도출
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
                    r60 = (df_price.iloc[idx+60][main_ticker] / df_price.iloc[idx][main_ticker] - 1) * 100
                    ret_60.append(r60)
                    win_60.append(r60 > 0)

            # --- 상단 레이아웃: 검증 지표 스코어보드 ---
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.subheader("🔮 통계적 저평가 타점 승률 백테스팅")
                if len(win_60) > 0:
                    st.metric(label="3달 후 (60거래일) 상승 확률", value=f"{np.mean(win_60)*100:.1f}%")
                    st.caption(f"지난 5년간 S&P 500 대비 극단적 소외 구간(-1.5σ) 진입 횟수: {len(signal_dates)}회 (평균 수익률: {np.mean(ret_60):+.2f}%)")
                else:
                    st.info("최근 5년간 규정된 저평가 임계선 터치 이력이 없거나 백테스팅 데이터가 부족합니다.")
                    
            with col_metric2:
                st.subheader(f"🛡️ {main_ticker} Diluted EPS & 매출 트렌드")
                if df_fin is not None and not df_fin.empty:
                    fig_fin = go.Figure()
                    fig_fin.add_trace(go.Bar(x=df_fin.index, y=df_fin['Revenue']/1e6 if 'Revenue' in df_fin.columns else [0], name="매출 ($M)", marker_color='rgba(99, 110, 250, 0.5)'))
                    fig_fin.add_trace(go.Scatter(x=df_fin.index, y=df_fin['EPS'] if 'EPS' in df_fin.columns else [0], mode='lines+markers', name="Diluted EPS ($)", line=dict(color='crimson', width=3)))
                    fig_fin.update_layout(template="plotly_white", height=180, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                    fig_fin.update_xaxes(type='category')
                    st.plotly_chart(fig_fin, use_container_width=True)
                else:
                    st.warning("재무제표 데이터를 연동할 수 없습니다.")

            st.markdown("---")
            
            # --- 중단 레이아웃: 진입 신호 추세선 ---
            st.subheader(f"📈 S&P 500 대비 이격도 타임라인 및 매수 진입 타점 (🔮)")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_returns.index, y=cum_residuals, mode='lines', name='추세 이격도(잔차 누적)', line=dict(color='blue', width=2)))
            fig1.add_trace(go.Scatter(x=df_returns.index, y=[0]*len(df_returns), mode='lines', name='평균 추세', line=dict(color='black', dash='dash')))
            fig1.add_trace(go.Scatter(x=df_returns.index, y=[2*res_std]*len(df_returns), mode='lines', name='과열 임계선', line=dict(color='red', dash='dot')))
            fig1.add_trace(go.Scatter(x=df_returns.index, y=[buy_signal_threshold]*len(df_returns), mode='lines', name='진입선 (-1.5σ)', line=dict(color='green', dash='dot')))
            fig1.add_trace(go.Scatter(x=signal_dates, y=df_analysis.loc[signal_dates, 'cum_residual'], mode='markers', name='진입 타점', marker=dict(color='gold', size=12, symbol='star', line=dict(color='black', width=1))))
            fig1.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("---")

            # --- 하단 레이아웃: 커스텀 바스켓 성과 비교 ---
            st.subheader("🎯 내가 직접 구성한 동종 주식군(Peer Group) 상대 성과 비교")
            st.markdown(f"현재 비교 중인 바스켓: `{', '.join([t for t in peer_list if t in df_cum_returns.columns])}`")
            
            fig2 = go.Figure()
            # 기준 종목은 두꺼운 빨간색선으로 고정 강조
            fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[main_ticker], mode='lines', name=f"★ 기준: {main_ticker}", line=dict(width=4, color='red')))
            
            # 투자자가 직접 입력한 나머지 라이벌 종목들 플롯
            for peer in peer_list:
                if peer in df_cum_returns.columns and peer != main_ticker:
                    fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[peer], mode='lines', name=peer, line=dict(width=1.8), opacity=0.7))
            
            fig2.update_layout(template="plotly_white", height=450, xaxis_title="날짜", yaxis_title="누적 수익률 (%)", margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"자산군 연산 도중 에러가 발생했습니다: {e}")
