import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 1. 페이지 설정
st.set_page_config(page_title="AI Multi-Factor Quant System", layout="wide")
st.title("🛡️ AI 멀티팩터 필터링 & 실적 결합형 승률 시뮬레이터")
st.markdown("야후 파이낸스 데이터 연동 엔진을 최신 버전으로 보완했습니다. 이제 **주가 저평가 타점**과 **실적 성장 궤도**를 동시에 만족하는 날만 골라내어 실전 승률을 시뮬레이션합니다.")

# 2. AI 기반 초세분화 매핑 가이드 (테마 자율 매칭 엔진)
def ai_get_pure_peer_group(ticker):
    ticker = ticker.upper().strip()
    if ticker in ["ASTS", "RKLB", "PL", "RDW", "BKSY", "LLAP", "TSLA"]:
        return ["ASTS", "RKLB", "PL", "RDW", "BKSY"], "🌌 저궤도 위성통신 및 순수 우주 스타트업 생태계"
    if ticker in ["NVDA", "AMD", "AVGO", "SMCI", "INTC", "ARM"]:
        return ["NVDA", "AMD", "AVGO", "SMCI", "ARM"], "🧠 AI 반도체 가속기 및 독점 인프라 밸류체인"
    if ticker in ["ASML", "AMAT", "LRCX", "KLAC", "TSM"]:
        return ["ASML", "AMAT", "LRCX", "KLAC", "TSM"], "🔬 파운드리 및 반도체 노광/식각 초정밀 핵심 장비주"
    if ticker in ["PLTR", "MSFT", "GOOGL", "META", "AI", "SOUN"]:
        return ["PLTR", "MSFT", "GOOGL", "META", "AI"], "💻 빅테크 LLM 및 인공지능 엔터프라이즈 소프트웨어 생태계"
    if ticker in ["TSLA", "MBLY", "RIVN", "LCID", "QS"]:
        return ["TSLA", "MBLY", "RIVN", "QS", "RBRK"], "🚗 자율주행 인공지능 및 차세대 모빌리티 생태계"
    if ticker in ["LLY", "NVO", "VRTX", "REGN", "AMGN"]:
        return ["LLY", "NVO", "VRTX", "REGN", "AMGN"], "🧬 글로벌 혁신 바이오테크 및 가속 성장 신약 생태계"
    return [ticker, "AAPL", "MSFT", "NVDA", "GOOGL"], "🌐 일반 대형 기술주 카테고리 (AI 자동 분류 중)"

# 3. 사이드바 설정
st.sidebar.header("⚙️ 스마트 퀀트 설정")
ticker_input = st.sidebar.text_input("분석할 주식 티커를 입력하세요:", value="ASTS").upper().strip()

peer_list, theme_diagnosis = ai_get_pure_peer_group(ticker_input)
st.sidebar.markdown(f"**🤖 AI 진단 섹터 분류:**\n`{theme_diagnosis}`")

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

# [최신화 보완 완료] 재무 지표 추출 엔진
@st.cache_data(ttl=86400)
def fetch_comprehensive_financials(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        
        # 1. info 데이터 방어 로직 (최신 yfinance 구조 대응)
        info = t_obj.info if t_obj.info else {}
        fast_info = t_obj.fast_info if hasattr(t_obj, 'fast_info') else {}
        
        per = info.get('trailingPE', np.nan)
        
        # 부채비율 교차 검증
        debt_equity = info.get('debtToEquity', np.nan)
        if np.isnan(debt_equity):
            debt_equity = info.get('totalDebt', np.nan) / info.get('totalStockholderEquity', np.nan) * 100 if info.get('totalStockholderEquity', 0) > 0 else np.nan
            
        # 잉여현금흐름 교차 검증
        fcf = info.get('freeCashflow', np.nan)
        
        # 2. 연간 재무제표 파싱 보완
        financials = t_obj.financials
        df_fin = pd.DataFrame()
        
        if financials is not None and not financials.empty:
            # 대소문자 및 띄어쓰기 무관하게 인덱스 매칭 매핑
            financials.index = [str(idx).strip().lower() for idx in financials.index]
            
            rev_idx = [i for i in financials.index if 'total revenue' in i or 'revenue' in i]
            eps_idx = [i for i in financials.index if 'diluted eps' in i or 'diluted' in i or 'basic eps' in i]
            
            # 현금흐름표 보완 (FCF 탐색)
            cashflow = t_obj.cashflow
            if np.isnan(fcf) and cashflow is not None and not cashflow.empty:
                cashflow.index = [str(idx).strip().lower() for idx in cashflow.index]
                ocf_idx = [i for i in cashflow.index if 'operating' in i or 'cash flow from operating activities' in i]
                capex_idx = [i for i in cashflow.index if 'capital expenditure' in i or 'investments in property' in i]
                if ocf_idx and capex_idx:
                    fcf = cashflow.iloc[0].loc[ocf_idx[0]] + cashflow.iloc[0].loc[capex_idx[0]]

            if rev_idx:
                df_fin['Revenue'] = financials.loc[rev_idx[0]]
            if eps_idx:
                df_fin['EPS'] = financials.loc[eps_idx[0]]
                
            df_fin = df_fin.sort_index(ascending=True)
            df_fin.index = pd.to_datetime(df_fin.index).year
            
        return {"per": per, "debt_equity": debt_equity, "fcf": fcf, "df_trend": df_fin}
    except Exception as e:
        return {"per": np.nan, "debt_equity": np.nan, "fcf": np.nan, "df_trend": pd.DataFrame()}

main_ticker = ticker_input
if main_ticker:
    try:
        df_price = fetch_quant_data(main_ticker, peer_list)
        fin_data = fetch_comprehensive_financials(main_ticker)
        
        if main_ticker not in df_price.columns:
            st.error(f"'{main_ticker}' 데이터를 가져오지 못했습니다.")
        else:
            # 연산 레이어
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
            df_analysis['year'] = df_analysis.index.year
            
            # --- [재무/실적 트렌드 딕셔너리 빌드] ---
            df_trend = fin_data["df_trend"]
            eps_growth_dict = {}
            if df_trend is not None and 'EPS' in df_trend.columns and len(df_trend) >= 2:
                # 전년 대비 EPS 가 성장했는지 연도별로 마킹
                for i in range(1, len(df_trend)):
                    curr_yr = df_trend.index[i]
                    prev_yr = df_trend.index[i-1]
                    eps_growth_dict[curr_yr] = df_trend['EPS'].iloc[i] > df_trend['EPS'].iloc[i-1]
            
            # --- 🔮 복합 멀티팩터 시뮬레이터 가동 ---
            buy_signal_threshold = -1.5 * res_std
            
            # 조건 1: 주가 저평가선 진입
            df_analysis['pure_price_signal'] = df_analysis['cum_residual'] <= buy_signal_threshold
            
            # 조건 2: 당해 연도 EPS 성장 달성 여부 매핑
            df_analysis['eps_growing'] = df_analysis['year'].map(eps_growth_dict).fillna(False)
            
            # 복합 타점: 가격은 과매도인데 실적(EPS)은 성장세인 날만 최종 추출!
            df_analysis['smart_signal'] = df_analysis['pure_price_signal'] & df_analysis['eps_growing']
            df_analysis['smart_signal_start'] = df_analysis['smart_signal'] & (~df_analysis['smart_signal'].shift(1).fillna(False))
            
            smart_signal_dates = df_analysis[df_analysis['smart_signal_start']].index
            pure_signal_dates = df_analysis[df_analysis['pure_price_signal'] & (~df_analysis['pure_price_signal'].shift(1).fillna(False))].index

            # 스마트 필터링 기반 승률 역추적 백테스팅
            win_60, ret_60 = [], []
            for d in smart_signal_dates:
                idx = df_price.index.get_loc(d)
                if idx + 60 < len(df_price):
                    r60 = (df_price.iloc[idx+60][main_ticker] / df_price.iloc[idx][main_ticker] - 1) * 100
                    ret_60.append(r60)
                    win_60.append(r60 > 0)
                    
            # 단순 주가 필터링 기반 승률 (비교용)
            pure_win_60 = []
            for d in pure_signal_dates:
                idx = df_price.index.get_loc(d)
                if idx + 60 < len(df_price):
                    pure_win_60.append((df_price.iloc[idx+60][main_ticker] / df_price.iloc[idx][main_ticker] - 1) > 0)

            # --- 🟢 UI LAYOUT 1: 종합 검증 레이더 (최신 연동 확인) ---
            st.subheader(f"🛡️ 1. {main_ticker} 실시간 종합 재무 구조 (데이터 연동 정상화)")
            f_col1, f_col2, f_col3 = st.columns(3)
            
            per_v = fin_data["per"]
            f_col1.metric(label="현재 Trailing PER", value=f"{per_v:.2f}배" if not np.isnan(per_v) and per_v > 0 else "적자 혹은 연동 지연")
            
            de_v = fin_data["debt_equity"]
            f_col2.metric(label="재무 부채비율", value=f"{de_v:.2f}%" if not np.isnan(de_v) else "N/A",
                          delta="⚠️ 위험" if de_v > 150 else "✅ 안정", delta_color="inverse")
                          
            fcf_v = fin_data["fcf"]
            fcf_display = f"${fcf_v/1e6:.1f}M" if not np.isnan(fcf_v) and abs(fcf_v) < 1e9 else (f"${fcf_v/1e9:.2f}B" if not np.isnan(fcf_v) else "N/A")
            f_col3.metric(label="금고 내부 현금 (Free Cash Flow)", value=fcf_display)

            st.markdown("---")

            # --- 🟢 UI LAYOUT 2: 스마트 멀티팩터 시뮬레이터 결과 소형판 ---
            st.subheader("🔮 2. [주가 저평가 + EPS 성장 동시만족] 스마트 타점 시뮬레이터")
            
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.metric(label="단순 주가 필터링 진입 시 승률", value=f"{np.mean(pure_win_60)*100:.1f}%" if pure_win_60 else "신호 없음")
                st.caption("실적 무관하게 단순히 주가 낙폭(-1.5σ)만 보고 진입했을 때의 60일 뒤 승률")
            with s_col2:
                st.metric(label="🔥 실적 필터링 결합 시 시뮬레이션 승률", value=f"{np.mean(win_60)*100:.1f}%" if win_60 else "신호 없음", delta="필터링 진화")
                st.caption("**[강력 추천]** 실적이 전년 대비 성장 중인데 주가만 억울하게 빠진 타점 진입 시 승률")
            with s_col3:
                st.metric(label="최종 압축 필터링 매수 타점 포착", value=f"{len(smart_signal_dates)}회")
                st.caption(f"5년 동안 밸류 트랩(실적 악화 하락)을 제외하고 걸러진 진짜 엑기스 타점 횟수")

            st.markdown("---")

            # --- 🟢 UI LAYOUT 3: 시각화 차트 존 ---
            chart_col1, chart_col2 = st.columns([6, 4])
            
            with chart_col1:
                st.subheader("📈 S&P 500 대비 이격도와 '스마트 실적 결합 타점' 매핑")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df_returns.index, y=cum_residuals, mode='lines', name='추세 이격도', line=dict(color='blue', width=2)))
                fig1.add_trace(go.Scatter(x=df_returns.index, y=[0]*len(df_returns), mode='lines', name='평균 추세', line=dict(color='black', dash='dash')))
                fig1.add_trace(go.Scatter(x=df_returns.index, y=[buy_signal_threshold]*len(df_returns), mode='lines', name='저평가 진입선', line=dict(color='green', dash='dot')))
                
                # 스마트 필터링 타점만 황금 별표로 시각화
                fig1.add_trace(go.Scatter(
                    x=smart_signal_dates, y=df_analysis.loc[smart_signal_dates, 'cum_residual'],
                    mode='markers', name='🔮 스마트 매수 신호 (실적성장+저평가)',
                    marker=dict(color='gold', size=14, symbol='star', line=dict(color='black', width=1.2))
                ))
                fig1.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig1, use_container_width=True)
                
            with chart_col2:
                st.subheader("📊 연간 매출액 및 Diluted EPS 성장 궤도 (복원 완료)")
                if df_trend is not None and not df_trend.empty:
                    fig_fin = go.Figure()
                    if 'Revenue' in df_trend.columns:
                        fig_fin.add_trace(go.Bar(x=df_trend.index, y=df_trend['Revenue']/1e6, name="매출 ($M)", marker_color='rgba(99, 110, 250, 0.5)'))
                    if 'EPS' in df_trend.columns:
                        fig_fin.add_trace(go.Scatter(x=df_trend.index, y=df_trend['EPS'], mode='lines+markers', name="Diluted EPS ($)", line=dict(color='crimson', width=3)))
                    fig_fin.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                    fig_fin.update_xaxes(type='category')
                    st.plotly_chart(fig_fin, use_container_width=True)
                else:
                    st.warning("이 기업은 현재 분기/연간 정식 재무 공시 보고서(10-K) 데이터를 야후 파이낸스 가용 서버에서 응답하지 않고 있습니다.")

            st.markdown("---")

            # --- 🟢 UI LAYOUT 4: AI 바스켓 비교 ---
            st.subheader(f"🎯 AI 자율 퓨어 플레이어 바스켓 동행 자산군 성과 비교")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[main_ticker], mode='lines', name=f"★ {main_ticker} (기준)", line=dict(width=4, color='red')))
            for peer in peer_list:
                if peer in df_cum_returns.columns and peer != main_ticker:
                    fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[peer], mode='lines', name=peer, line=dict(width=1.8), opacity=0.6))
            fig2.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"시뮬레이터 빌드 중 연산 오류가 발생했습니다: {e}")
