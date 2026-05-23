import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 1. 페이지 설정
st.set_page_config(page_title="AI-Powered Intelligent Quant", layout="wide")
st.title("🧠 AI 자율 그룹핑 및 종합 멀티팩터 퀀트 투자 시스템")
st.markdown("티커를 입력하면 AI 룰 엔진이 숨겨진 '초정밀 퓨어 테마 주식군'을 자동으로 분류하여 바스켓을 빌드하고, 회사의 종합 재무 체력을 다각도로 검증합니다.")

# 2. AI 기반 초세분화 매핑 가이드 (테마 자율 매칭 엔진)
# 투자자가 티커를 치면 AI가 해당 기업의 진짜 성격을 분석하여 최적의 비교군 4개를 강제 타겟팅합니다.
def ai_get_pure_peer_group(ticker):
    ticker = ticker.upper().strip()
    
    # 🌌 우주항공/스타트업/위성통신 세부 생태계
    if ticker in ["ASTS", "RKLB", "PL", "RDW", "BKSY", "LLAP", "TSLA"]:
        if ticker in ["ASTS", "RKLB", "PL", "RDW", "BKSY"]:
            return ["ASTS", "RKLB", "PL", "RDW", "BKSY"], "🌌 저궤도 위성통신 및 순수 우주 스타트업 생태계"
            
    # 🧠 AI 가속기 / 빅테크 반도체 하드웨어 독점주
    if ticker in ["NVDA", "AMD", "AVGO", "SMCI", "INTC", "ARM"]:
        return ["NVDA", "AMD", "AVGO", "SMCI", "ARM"], "🧠 AI 반도체 가속기 및 독점 인프라 밸류체인"
        
    # 🔬 반도체 미세공정 '슈퍼 을(乙)' 초정밀 장비 생태계
    if ticker in ["ASML", "AMAT", "LRCX", "KLAC", "TSM"]:
        return ["ASML", "AMAT", "LRCX", "KLAC", "TSM"], "🔬 파운드리 및 반도체 노광/식각 초정밀 핵심 장비주"
        
    # 💻 AI 거대언어모델(LLM) 및 엔터프라이즈 소프트웨어 플레이어
    if ticker in ["PLTR", "MSFT", "GOOGL", "META", "AI", "SOUN"]:
        return ["PLTR", "MSFT", "GOOGL", "META", "AI"], "💻 빅테크 LLM 및 인공지능 엔터프라이즈 소프트웨어 생태계"
        
    # 🚗 자율주행, 전동화 및 로보틱스 모빌리티 혁신주
    if ticker in ["TSLA", "MBLY", "RIVN", "LCID", "QS"]:
        return ["TSLA", "MBLY", "RIVN", "QS", "RBRK"], "🚗 자율주행 인공지능 및 차세대 모빌리티 생태계"
        
    # 🧬 비만치료제 및 혁신 바이오 플랫폼 대장주
    if ticker in ["LLY", "NVO", "VRTX", "REGN", "AMGN"]:
        return ["LLY", "NVO", "VRTX", "REGN", "AMGN"], "🧬 글로벌 혁신 바이오테크 및 가속 성장 신약 생태계"

    # 만약 사전에 매핑되지 않은 완전 새로운 티커일 경우, 자산 스케일이 비슷한 대표 대형주들을 기본 임시 그룹핑하여 방어
    return [ticker, "AAPL", "MSFT", "NVDA", "GOOGL"], "🌐 일반 대형 기술주 카테고리 (AI 자동 분류 중)"

# 3. 사이드바 설정
st.sidebar.header("⚙️ 스마트 분석 엔진")
ticker_input = st.sidebar.text_input("1. 분석할 주식 티커를 입력하세요:", value="ASTS").upper().strip()

# AI 엔진 가동 -> 피어 그룹 리스트와 테마 진단명 자동 추출
peer_list, theme_diagnosis = ai_get_pure_peer_group(ticker_input)

st.sidebar.markdown(f"**🤖 AI 진단 섹터 분류:**\n`{theme_diagnosis}`")
st.sidebar.markdown(f"**📊 자율 매칭된 라이벌 종목:**\n`{', '.join(peer_list)}`")

# 데이터 다운로드 엔진 (주가용)
@st.cache_data(ttl=3600)
def fetch_quant_data(target_ticker, tickers_to_compare):
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

# 데이터 다운로드 엔진 (입체적 팩터 재무 지표용)
@st.cache_data(ttl=86400)
def fetch_comprehensive_financials(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        financials = t_obj.financials
        cashflow = t_obj.cashflow
        
        # 1. 실시간 배수 및 재무 구조 팩터 추출
        per = info.get('trailingPE', np.nan)
        debt_to_equity = info.get('debtToEquity', np.nan) # 부채비율 (%)
        
        # 2. 잉여현금흐름(Free Cash Flow) 추출 (가장 최신 연도 기준)
        fcf = info.get('freeCashflow', np.nan)
        if np.isnan(fcf) and not cashflow.empty:
            # 직접 연산 방어 코드: 영업현금흐름 - 자본지출(CapEx)
            ocf_key = [k for k in cashflow.index if 'Operating' in k or 'Cash Flow From Operating Activities' in k]
            capex_key = [k for k in cashflow.index if 'Capital Expenditures' in k or 'Investments In Property Plant And Equipment' in k]
            if ocf_key and capex_key:
                fcf = cashflow.loc[ocf_key[0]].iloc[0] + cashflow.loc[capex_key[0]].iloc[0] # CapEx는 보통 음수값으로 들어옴
                
        # 3. 차트용 연간 매출 및 Diluted EPS 추세 가공
        df_fin = pd.DataFrame()
        if not financials.empty:
            rev_key = [k for k in financials.index if 'Revenue' in k or 'Total Revenue' in k]
            eps_key = [k for k in financials.index if 'Diluted EPS' in k or 'Diluted' in k]
            if rev_key: df_fin['Revenue'] = financials.loc[rev_key[0]]
            if eps_key: df_fin['EPS'] = financials.loc[eps_key[0]]
            df_fin = df_fin.sort_index(ascending=True)
            df_fin.index = pd.to_datetime(df_fin.index).year
            
        return {"per": per, "debt_equity": debt_to_equity, "fcf": fcf, "df_trend": df_fin}
    except:
        return {"per": np.nan, "debt_equity": np.nan, "fcf": np.nan, "df_trend": pd.DataFrame()}

if main_ticker := ticker_input:
    try:
        with st.spinner("AI가 동종 주식 바스켓을 구성하고 종합 재무 데이터를 스캐닝하는 중..."):
            df_price = fetch_quant_data(main_ticker, peer_list)
            fin_data = fetch_comprehensive_financials(main_ticker)
            
        if main_ticker not in df_price.columns:
            st.error(f"'{main_ticker}' 데이터를 가져오지 못했습니다. 올바른 티커인지 확인해 주세요.")
        else:
            # 기본 연산
            df_returns = df_price.pct_change().dropna() * 100
            df_cum_returns = (df_price / df_price.iloc[0] - 1) * 100
            
            X_sp = df_returns[["^GSPC"]].values
            y_stock = df_returns[main_ticker].values
            
            model_sp = LinearRegression().fit(X_sp, y_stock)
            residuals = y_stock - model_sp.predict(X_sp)
            cum_residuals = np.cumsum(residuals)
            res_std = np.std(cum_residuals)
            
            df_analysis = df_returns.copy()
            df_analysis['cum_residual'] = cum_residuals
            
            # 승률 백테스팅 연산
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

            # --- 🟢 SECTION 1: 종합 컴퍼니 분석 레이더 카드 배치 ---
            st.subheader(f"🛡️ 1. {main_ticker} 종합 재무 건강도 및 벨류에이션 검증 판정")
            
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            
            # 카드 1: PER
            per_v = fin_data["per"]
            f_col1.metric(label="현재 가치 평가 (Trailing PER)", value=f"{per_v:.2f}배" if not np.isnan(per_v) else "적자(N/A)")
            f_col1.caption("업계 평균 및 과거 자사 배수 대비 위치를 체크하세요.")
            
            # 카드 2: 부채비율
            de_v = fin_data["debt_equity"]
            f_col2.metric(label="재무적 맷집 (부채비율)", value=f"{de_v:.2f}%" if not np.isnan(de_v) else "N/A", 
                          delta="⚠️ 고위험" if de_v > 150 else "✅ 안정구간", delta_color="inverse")
            f_col2.caption("100% 미만이면 금리 변동에 강한 생존 체력을 가졌음을 뜻합니다.")
            
            # 카드 3: 잉여현금흐름
            fcf_v = fin_data["fcf"]
            if not np.isnan(fcf_v):
                fcf_display = f"${fcf_v/1e6:.1f}M" if abs(fcf_v) < 1e9 else f"${fcf_v/1e9:.2f}B"
                f_col3.metric(label="진짜 금고 현금 (Free Cash Flow)", value=fcf_display, 
                              delta="정상 흑자" if fcf_v > 0 else "현금 잠식")
            else:
                f_col3.metric(label="진짜 금고 현금 (Free Cash Flow)", value="N/A")
            f_col3.caption("영업으로 번 돈에서 인프라 투자를 끝내고 남은 알짜 현금입니다.")
            
            # 카드 4: 통계적 승률 연동
            if len(win_60) > 0:
                f_col4.metric(label="저평가 타점 진입 시 3달 승률", value=f"{np.mean(win_60)*100:.1f}%")
                f_col4.caption(f"과거 5년간 -1.5σ 도달 후 복귀 성공 확률 (총 {len(signal_dates)}회 신호)")
            else:
                f_col4.metric(label="저평가 타점 진입 시 3달 승률", value="신호 없음")
                f_col4.caption("과거 5년간 극단적 소외 구간에 들어간 적이 없는 종목입니다.")

            st.markdown("---")

            # --- 🟢 SECTION 2: 두 가지 시각화 차트 레이아웃 배치 ---
            chart_col1, chart_col2 = st.columns([6, 4])
            
            with chart_col1:
                st.subheader("📈 S&P 500 대비 이격도 타임라인 및 퀀트 타점 (🔮)")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df_returns.index, y=cum_residuals, mode='lines', name='추세 이격도', line=dict(color='blue', width=2)))
                fig1.add_trace(go.Scatter(x=df_returns.index, y=[0]*len(df_returns), mode='lines', name='평균 추세', line=dict(color='black', dash='dash')))
                fig1.add_trace(go.Scatter(x=df_returns.index, y=[buy_signal_threshold]*len(df_returns), mode='lines', name='진입선', line=dict(color='green', dash='dot')))
                fig1.add_trace(go.Scatter(x=signal_dates, y=df_analysis.loc[signal_dates, 'cum_residual'], mode='markers', name='매수 신호', marker=dict(color='gold', size=12, symbol='star', line=dict(color='black', width=1))))
                fig1.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig1, use_container_width=True)
                
            with chart_col2:
                st.subheader("📊 연간 매출액 및 Diluted EPS 성장 궤도")
                df_fin = fin_data["df_trend"]
                if df_fin is not None and not df_fin.empty:
                    fig_fin = go.Figure()
                    fig_fin.add_trace(go.Bar(x=df_fin.index, y=df_fin['Revenue']/1e6 if 'Revenue' in df_fin.columns else [0], name="매출 ($M)", marker_color='rgba(99, 110, 250, 0.5)'))
                    fig_fin.add_trace(go.Scatter(x=df_fin.index, y=df_fin['EPS'] if 'EPS' in df_fin.columns else [0], mode='lines+markers', name="Diluted EPS ($)", line=dict(color='crimson', width=3)))
                    fig_fin.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                    fig_fin.update_xaxes(type='category')
                    st.plotly_chart(fig_fin, use_container_width=True)
                else:
                    st.warning("실적 시계열 데이터를 불러오지 못했습니다.")

            st.markdown("---")

            # --- 🟢 SECTION 3: AI 자율 바스켓 상대 성과 비교 ---
            st.subheader(f"🎯 AI 자율 빌드 동종 자산군 바스켓 성과 비교")
            st.markdown(f"**분류 그룹 벤치마크:** `{theme_diagnosis}`")
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[main_ticker], mode='lines', name=f"★ {main_ticker} (기준)", line=dict(width=4, color='red')))
            
            for peer in peer_list:
                if peer in df_cum_returns.columns and peer != main_ticker:
                    fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[peer], mode='lines', name=peer, line=dict(width=1.8), opacity=0.6))
                    
            fig2.update_layout(template="plotly_white", height=450, xaxis_title="날짜", yaxis_title="누적 수익률 (%)", margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"대시보드 구동 중 오류가 발생했습니다: {e}")
