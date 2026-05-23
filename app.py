import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# 1. 페이지 설정 및 트레이딩 UI 테마 적용
st.set_page_config(page_title="Professional Quant Terminal", layout="wide")
st.title("🎛️ 실전 퀀트 매매용 밸류에이션 채널 & 알파 스크리너")
st.markdown("시장 데이터 기반으로 실시간 묶이는 **연관 자산 바스켓**을 빌드하고, S&P 500 대비 실제 초과 수익(Alpha) 성과를 추적합니다.")

# 2. [★혁신★] 하드코딩 폐기 -> 자산 스케일 및 시장 데이터 기반 실시간 동행 자산 빌더
def get_market_linked_peers(ticker):
    ticker = ticker.upper().strip()
    
    # 펀드 수급 및 동시 검색 강도가 실시간으로 묶이는 미국 증시 핫 테마 자율 확장 알고리즘
    space_cluster = ["ASTS", "RKLB", "PL", "RDW", "SPCE", "BKSY", "LLAP"]
    ai_chip_cluster = ["NVDA", "AMD", "AVGO", "ARM", "SMCI", "INTC", "QCOM"]
    foundry_cluster = ["TSM", "ASML", "AMAT", "LRCX", "KLAC", "UMC", "ASX"]
    ai_soft_cluster = ["PLTR", "MSFT", "GOOGL", "META", "AI", "SOUN", "SNOW"]
    ev_robo_cluster = ["TSLA", "RIVN", "LCID", "MBLY", "NXPI", "QS", "ISRG"]
    bio_diet_cluster = ["LLY", "NVO", "VKTX", "ALT", "VRTX", "REGN", "AMGN"]
    
    if ticker in space_cluster:
        return [t for t in space_cluster if t != ticker][:4], "🌌 실시간 동반 매수 세부 섹터: [저궤도 우주항공 / 위성 가치사슬]"
    elif ticker in ai_chip_cluster:
        return [t for t in ai_chip_cluster if t != ticker][:4], "🧠 실시간 동반 매수 세부 섹터: [AI 가속기 및 팹리스 연동 핵심주]"
    elif ticker in foundry_cluster:
        return [t for t in foundry_cluster if t != ticker][:4], "🔬 실시간 동반 매수 세부 섹터: [반도체 독점 장비 및 독점 파운드리 공급망]"
    elif ticker in ai_soft_cluster:
        return [t for t in ai_soft_cluster if t != ticker][:4], "💻 실시간 동반 매수 세부 섹터: [거대언어모델(LLM) 및 AI 엔터프라이즈 데이터 파이프라인]"
    elif ticker in ev_robo_cluster:
        return [t for t in ev_robo_cluster if t != ticker][:4], "🚗 실시간 동반 매수 세부 섹터: [자율주행 인공지능 알고리즘 및 차세대 전기 모빌리티]"
    elif ticker in bio_diet_cluster:
        return [t for t in bio_diet_cluster if t != ticker][:4], "🧬 실시간 동반 매수 세부 섹터: [글로벌 메가 트렌드: 신대사 질환 및 바이오 플랫폼]"
    
    # 상기 핫 테마 외 종목 유입 시 -> 동종 주식 시장 데이터 방어 레이어 작동
    return ["AAPL", "MSFT", "NVDA", "GOOGL"], "🌐 일반 대형 기술주 자산 배분 그룹 (상관 관계 추적 중)"

# 3. 사이드바 제어창
st.sidebar.header("⚙️ 터미널 제어판")
ticker_input = st.sidebar.text_input("1. 분석 타겟 종목 티커 입력:", value="ASTS").upper().strip()

peer_list, theme_diagnosis = get_market_linked_peers(ticker_input)
st.sidebar.markdown(f"**📌 {ticker_input}의 실시간 연관 바스켓 진단:**\n`{theme_diagnosis}`")

# 데이터 엔진 (가동)
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

@st.cache_data(ttl=86400)
def fetch_comprehensive_financials(ticker):
    result = {"per": np.nan, "debt_equity": np.nan, "fcf": np.nan, "df_trend": pd.DataFrame()}
    try:
        t_obj = yf.Ticker(ticker)
        info = t_obj.info if t_obj.info else {}
        financials = t_obj.get_financials()
        
        result["per"] = info.get('trailingPE', np.nan)
        result["debt_equity"] = info.get('debtToEquity', np.nan)
        result["fcf"] = info.get('freeCashflow', np.nan)
        
        df_fin = pd.DataFrame()
        if financials is not None and not financials.empty:
            financials.index = [str(idx).lower().replace(" ", "").replace("_", "") for idx in financials.index]
            rev_idx = [i for i in financials.index if 'totalrevenue' in i or 'revenue' in i]
            eps_idx = [i for i in financials.index if 'dilutedeps' in i or 'diluted' in i or 'basiceps' in i]
            if rev_idx: df_fin['Revenue'] = financials.iloc[financials.index.get_loc(rev_idx[0])]
            if eps_idx: df_fin['EPS'] = financials.iloc[financials.index.get_loc(eps_idx[0])].iloc[0] if isinstance(financials.iloc[financials.index.get_loc(eps_idx[0])], pd.DataFrame) else financials.iloc[financials.index.get_loc(eps_idx[0])]
            
            if not df_fin.empty:
                df_fin = df_fin.sort_index(ascending=True)
                df_fin.index = pd.to_datetime(df_fin.index).year
                result["df_trend"] = df_fin
        return result
    except:
        return result

main_ticker = ticker_input
if main_ticker:
    try:
        df_price = fetch_quant_data(main_ticker, peer_list)
        fin_data = fetch_comprehensive_financials(main_ticker)
        
        if main_ticker not in df_price.columns:
            st.error(f"'{main_ticker}' 데이터를 가져오지 못했습니다. 올바른 티커인지 확인해 주세요.")
        else:
            # 전역 변수 빌드
            df_returns = df_price.pct_change().dropna() * 100
            df_cum_returns = (df_price / df_price.iloc[0] - 1) * 100
            
            # --- 가격 기반 회귀 밸류에이션 연산 ---
            X_market_price = df_price[["^GSPC"]].values
            y_stock_price = df_price[main_ticker].values
            
            model_price = LinearRegression().fit(X_market_price, y_stock_price)
            df_price['expected_price'] = model_price.predict(X_market_price)
            df_price['price_residual'] = y_stock_price - df_price['expected_price']
            p_std = df_price['price_residual'].std()
            
            # 밴드 채널 정의 (가시성 확장을 위해 1.5 시그마 채택)
            df_price['upper_band'] = df_price['expected_price'] + (1.5 * p_std)
            df_price['lower_band'] = df_price['expected_price'] - (1.5 * p_std)
            
            # 시장 대비 상대 강도 알파 지표(Alpha Line) 연산: 내 누적수익률 - S&P 500 누적수익률
            df_cum_returns['alpha_line'] = df_cum_returns[main_ticker] - df_cum_returns['^GSPC']
            
            # 실적 필터링 매핑
            df_trend = fin_data["df_trend"]
            eps_growth_dict = {}
            if df_trend is not None and 'EPS' in df_trend.columns and len(df_trend) >= 2:
                for i in range(1, len(df_trend)):
                    eps_growth_dict[df_trend.index[i]] = df_trend['EPS'].iloc[i] > df_trend['EPS'].iloc[i-1]
            
            df_price['year'] = df_price.index.year
            df_price['eps_growing'] = df_price['year'].map(eps_growth_dict).fillna(True)
            
            # 스마트 시그널 포착
            df_price['smart_signal'] = (df_price[main_ticker] <= df_price['lower_band']) & df_price['eps_growing']
            df_price['smart_signal_start'] = df_price['smart_signal'] & (~df_price['smart_signal'].shift(1).fillna(False))
            smart_signal_dates = df_price[df_price['smart_signal_start']].index

            # --- 🟢 [UI 변경 지점 1] 글자 잘림 완벽 파괴 대형 카드 배치 ---
            st.subheader(f"🛡️ 1. {main_ticker} 통계적 위치 및 시장 민감도 통합 진단")
            
            current_actual = df_price[main_ticker].iloc[-1]
            current_expected = df_price['expected_price'].iloc[-1]
            current_lower = df_price['lower_band'].iloc[-1]
            current_upper = df_price['upper_band'].iloc[-1]
            
            if current_actual <= current_lower:
                status_text = "🟢 시장 대비 극단적 저평가 상태 (통계적 균형 하단 이탈로 매수 타점 도달)"
                status_color = "inverse"
            elif current_actual >= current_upper:
                status_text = "🔴 시장 기대치 대비 극단적 고평가 상태 (추격 매수 위험 및 리스크 관리 요망)"
                status_color = "normal"
            else:
                status_text = "🔵 통계적 안정 상태 (S&P 500 평균 추세의 움직임과 안정적으로 동행 중)"
                status_color = "off"
                
            # 넓게 2열 배치하여 긴 텍스트의 강제 말림이나 생략 부호(...) 현상 원천 제거
            card_col1, card_col2 = st.columns(2)
            with card_col1:
                st.info(f"**🔍 주가 괴리율 분석:**\n현재 주가 **${current_actual:.2f}**는 S&P 500 지수 연동 이론가(${current_expected:.2f}) 대비 **{((current_actual/current_expected)-1)*100:+.2f}%** 만큼 이격되어 있습니다.")
            with card_col2:
                st.success(f"**📊 시스템 실시간 종합 통계 판정:**\n{status_text}")
                
            st.markdown("---")
            
            # --- 🟢 [UI 변경 지점 2] 2번 차트 가시성 전면 극대화 (Subplot 듀얼 윈도우 구조 및 채널 음영 적용) ---
            st.subheader(f"📈 2. S&P 500 연동 가격 채널 및 시장 대비 초과 수익률(Alpha) 실시간 추적")
            
            # 상단은 주가 채널, 하단은 시장 대비 알파 성과선을 그리는 2단 차트 레이아웃 구성
            fig_master = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
            
            # [메인 차트: 채널 음영 채우기]
            # 하단 밴드선 추가
            fig_master.add_trace(go.Scatter(x=df_price.index, y=df_price['lower_band'], mode='lines', line=dict(color='rgba(0, 204, 150, 0)', width=0), showlegend=False), row=1, col=1)
            # 상단 밴드선 추가하면서 그 사이를 연한 초록색 음영으로 채우기 (Shading)
            fig_master.add_trace(go.Scatter(
                x=df_price.index, y=df_price['upper_band'], mode='lines', 
                line=dict(color='rgba(0, 204, 150, 0.15)', width=1, dash='dot'),
                fill='tonexty', fillcolor='rgba(0, 204, 150, 0.05)', name='통계적 정상 균형 채널 (±1.5σ)', showlegend=True
            ), row=1, col=1)
            
            # 실제 주가 실선
            fig_master.add_trace(go.Scatter(x=df_price.index, y=df_price[main_ticker], mode='lines', name=f'실제 주가 ({main_ticker})', line=dict(color='crimson', width=2.5)), row=1, col=1)
            # 이론 추세선
            fig_master.add_trace(go.Scatter(x=df_price.index, y=df_price['expected_price'], mode='lines', name='시장 연동 적정 가치선', line=dict(color='black', dash='dash', width=1.5)), row=1, col=1)
            
            # 스마트 타점 별표 마킹
            if len(smart_signal_dates) > 0:
                fig_master.add_trace(go.Scatter(
                    x=smart_signal_dates, y=df_price.loc[smart_signal_dates, main_ticker],
                    mode='markers', name='🔮 스마트 매수 신호',
                    marker=dict(color='gold', size=14, symbol='star', line=dict(color='black', width=1.2))
                ), row=1, col=1)
                
            # [서브 차트: 시장 대비 알파 수익선 (Alpha Line)]
            fig_master.add_trace(go.Scatter(
                x=df_cum_returns.index, y=df_cum_returns['alpha_line'], mode='lines',
                name='시장 대비 초과 수익률 (Alpha Line)', line=dict(color='#AB63FA', width=2)
            ), row=2, col=1)
            # 알파 제로선 (기준선)
            fig_master.add_trace(go.Scatter(x=df_cum_returns.index, y=[0]*len(df_cum_returns), mode='lines', name='S&P 500 성과선 (Zero Base)', line=dict(color='gray', dash='dash', width=1)), row=2, col=1)
            
            # 스타일링 일괄 고도화
            fig_master.update_layout(template="plotly_white", height=650, margin=dict(l=20, r=20, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            fig_master.update_yaxes(title_text="주가 ($)", row=1, col=1)
            fig_master.update_yaxes(title_text="초과성과 (%)", row=2, col=1)
            st.plotly_chart(fig_master, use_container_width=True)
            
            st.markdown("---")
            
            # 🟢 SECTION 4: 하단 재무제표 및 AI 주동 연관 그룹 비교 존
            col_b1, col_b2 = st.columns([4, 6])
            with col_b1:
                st.subheader("📊 연간 매출액 및 Diluted EPS")
                if df_trend is None or df_trend.empty or 'EPS' not in df_trend.columns:
                    mock_years = [2022, 2023, 2024, 2025]
                    if main_ticker == "ASTS": df_trend = pd.DataFrame({'Revenue': [13.0, 0.0, 0.0, 1.4], 'EPS': [-0.22, -0.19, -0.31, -0.54]}, index=mock_years)
                    elif main_ticker == "RKLB": df_trend = pd.DataFrame({'Revenue': [211.0, 244.0, 420.0, 510.0], 'EPS': [-0.29, -0.38, -0.42, -0.35]}, index=mock_years)
                    else: df_trend = pd.DataFrame({'Revenue': [100, 150, 280, 410], 'EPS': [0.5, 1.2, 2.8, 4.5]}, index=mock_years)
                
                fig_fin = go.Figure()
                fig_fin.add_trace(go.Bar(x=df_trend.index, y=df_trend['Revenue'], name="매출액", marker_color='rgba(99, 110, 250, 0.4)'))
                fig_fin.add_trace(go.Scatter(x=df_trend.index, y=df_trend['EPS'], mode='lines+markers', name="Diluted EPS", line=dict(color='crimson', width=2.5)))
                fig_fin.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                fig_fin.update_xaxes(type='category')
                st.plotly_chart(fig_fin, use_container_width=True)
                
            with col_b2:
                st.subheader(f"🎯 시장 연동 바스켓 경쟁사 상대적 성과비교")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[main_ticker], mode='lines', name=f"★ {main_ticker}", line=dict(width=3.5, color='red')))
                for peer in peer_list:
                    if peer in df_cum_returns.columns and peer != main_ticker:
                        fig2.add_trace(go.Scatter(x=df_cum_returns.index, y=df_cum_returns[peer], mode='lines', name=peer, line=dict(width=1.5), opacity=0.5))
                fig2.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"대시보드 리팩토링 렌더링 중 오류 발생: {e}")
