import dash
from dash import dcc, html, Input, Output
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

# 데이터 로드 함수
def load_data(tickers):
    df = yf.download(tickers, start="2000-01-01", progress=False)["Close"]
    df.index = pd.to_datetime(df.index)
    return df

# 주요 지표 설정
main_indices = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI"
}
# M7 기업: 한글 이름 -> 티커
m7_tickers = {
    "애플": "AAPL",
    "마이크로소프트": "MSFT",
    "구글": "GOOGL",
    "아마존": "AMZN",
    "메타": "META",
    "테슬라": "TSLA",
    "엔비디아": "NVDA"
}
# 국가별 주요 지수
country_indices = {
    "Euro Stoxx 600": "^STOXX",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Hang Seng": "^HSI",
    "Shanghai Composite": "000001.SS",
    "대만 지수": "^TWII"
}

# 데이터 불러오기
df_main = load_data(list(main_indices.values()))
df_m7 = load_data(list(m7_tickers.values()))
df_countries = load_data(list(country_indices.values()))

# 전체 기간 범위
dates = df_main.index.union(df_m7.index).union(df_countries.index)

# Dash 앱 초기화
app = dash.Dash(__name__)

# 레이아웃 정의
app.layout = html.Div([
    html.H1("Multi-Index Dashboard", style={'textAlign':'center'}),

    # 메인 지표: 2열 배치
    html.Div([
        html.Div(dcc.Graph(id=f"{name.replace(' ','_')}_chart"),
                 style={'width':'49%','display':'inline-block','padding':'5px'})
        for name in main_indices
    ], style={'textAlign':'center'}),

    # M7 기업 주가 차트 (2열 배치로 한 칸 차지)
    html.Div([
        html.Div(dcc.Graph(id='m7_chart'), style={'width':'49%','display':'inline-block','padding':'5px'}),
        html.Div(html.Div(), style={'width':'49%','display':'inline-block'})
    ], style={'textAlign':'center'}),

    # 국가별 주요 지수: 2열 배치
    html.Div([
        html.Div(dcc.Graph(id=f"{name.replace(' ','_')}_ctry"),
                 style={'width':'49%','display':'inline-block','padding':'5px'})
        for name in country_indices
    ], style={'textAlign':'center'}),

    # 정규화 섹션: 기간선택 + 비교대상 드롭다운
    html.H2("정규화 지표 비교", style={'textAlign':'center','marginTop':'40px'}),
    html.Div([
        html.Div([
            html.Label("기간 선택:"),
            dcc.DatePickerRange(id='norm_range',
                                min_date_allowed=dates.min(),
                                max_date_allowed=dates.max(),
                                start_date=dates.min(),
                                end_date=dates.max()),
            html.Label("비교 대상 선택:"),
            dcc.Dropdown(id='norm_select', multi=True,
                         options=[{'label':name,'value':ticker} for d in (main_indices, m7_tickers, country_indices) for name,ticker in d.items()],
                         value=[main_indices['S&P 500']] + list(m7_tickers.values()))
        ], style={'display':'flex','gap':'20px','justifyContent':'center','marginBottom':'20px'}),
        html.Div([
            html.Div(dcc.Graph(id='norm_line'), style={'width':'67%','display':'inline-block'}),
            html.Div(dcc.Graph(id='norm_bar'), style={'width':'32%','display':'inline-block','verticalAlign':'top'})
        ])
    ], style={'maxWidth':'1200px','margin':'0 auto'}),

    # 두번째 정규화: 국가별 지수 비교
    html.H2("국가별 지수 정규화 비교", style={'textAlign':'center','marginTop':'40px'}),
    html.Div([
        html.Div(dcc.Graph(id='norm_line_countries'), style={'width':'67%','display':'inline-block'}),
        html.Div(dcc.Graph(id='norm_bar_countries'), style={'width':'32%','display':'inline-block','verticalAlign':'top'})
    ], style={'maxWidth':'1200px','margin':'0 auto', 'paddingTop':'20px'})
], style={'padding':'20px'})

# 공통 Figure 생성 함수 (rangeslider 제거)
def make_figure(series, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=title))
    last_date = series.index.max(); last_val = series.iloc[-1]
    fig.update_layout(title=title,
                      xaxis=dict(range=[series.index.min(), series.index.max()], tickformat='%Y-%m-%d'),
                      yaxis_title='Price', template='plotly_white')
    fig.add_annotation(x=last_date, y=last_val,
                       text=f"Last: {last_val:.2f} on {last_date.date()}",
                       showarrow=True, arrowhead=1)
    return fig

# 메인 지표 콜백
for name, ticker in main_indices.items():
    @app.callback(Output(f"{name.replace(' ','_')}_chart", 'figure'),
                  Input('norm_range','start_date'), Input('norm_range','end_date'))
    def update_main(start_date, end_date, ticker=ticker, title=name):
        s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
        series = load_data([ticker])[ticker].loc[s:e].dropna()
        return make_figure(series, title)

# M7 기업 콜백 (한글 이름)
@app.callback(Output('m7_chart','figure'),
              Input('norm_range','start_date'), Input('norm_range','end_date'))
def update_m7(start_date, end_date):
    s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
    df = df_m7.loc[s:e].dropna(how='all')
    fig = go.Figure()
    ticker_to_label = {ticker:label for label,ticker in m7_tickers.items()}
    for ticker in df.columns:
        label = ticker_to_label.get(ticker, ticker)
        fig.add_trace(go.Scatter(x=df.index, y=df[ticker], mode='lines', name=label))
    last_date = df.index.max(); last_vals = df.iloc[-1]
    for ticker, val in last_vals.nlargest(3).items():
        fig.add_annotation(x=last_date, y=val,
                           text=f"{ticker_to_label.get(ticker,ticker)}: {val:.2f}",
                           showarrow=False, yshift=10)
    fig.update_layout(title='M7 기업',
                      xaxis=dict(range=[s,e], tickformat='%Y-%m-%d'),
                      template='plotly_white')
    return fig

# 국가별 지수 콜백
for name, ticker in country_indices.items():
    @app.callback(Output(f"{name.replace(' ','_')}_ctry", 'figure'),
                  Input('norm_range','start_date'), Input('norm_range','end_date'))
    def update_country(start_date, end_date, ticker=ticker, title=name):
        s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
        series = load_data([ticker])[ticker].loc[s:e].dropna()
        return make_figure(series, title)

# 정규화 일반 콜백
@app.callback(Output('norm_line','figure'), Output('norm_bar','figure'),
              Input('norm_range','start_date'), Input('norm_range','end_date'), Input('norm_select','value'))
def update_normalized(start_date, end_date, tickers):
    s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
    if not tickers:
        return go.Figure(), go.Figure()
    df = load_data(tickers).loc[s:e].dropna(how='all')
    base = df.iloc[0]
    norm = df.div(base).mul(100)
    # 선형 차트
    line = go.Figure()
    for col in norm.columns:
        label = col
        if col in m7_tickers.values(): label = {v:k for k,v in m7_tickers.items()}[col]
        line.add_trace(go.Scatter(x=norm.index, y=norm[col], mode='lines', name=label))
    line.update_layout(title=f"Normalized (Base=100): {s.date()}~{e.date()}",
                       xaxis=dict(range=[s,e], tickformat='%Y-%m-%d'), template='plotly_white')
    # 막대 차트
    pct = (norm.iloc[-1] - 100)
    bar = go.Figure()
    for idx, val in pct.items():
        label = idx
        if idx in m7_tickers.values(): label = {v:k for k,v in m7_tickers.items()}[idx]
        bar.add_trace(go.Bar(x=[val], y=[label], orientation='h'))
    bar.update_layout(title="Change (%)", template='plotly_white', yaxis=dict(autorange='reversed'))
    return line, bar

# 국가별 정규화 콜백
@app.callback(Output('norm_line_countries','figure'), Output('norm_bar_countries','figure'),
              Input('norm_range','start_date'), Input('norm_range','end_date'))
def update_normalized_countries(start_date, end_date):
    s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
    df = df_countries.loc[s:e].dropna(how='all')
    base = df.iloc[0]
    norm = df.div(base).mul(100)
    line = go.Figure()
    for col in norm.columns:
        line.add_trace(go.Scatter(x=norm.index, y=norm[col], mode='lines', name=col))
    line.update_layout(title=f"국가별 지수 Normalized (Base=100): {s.date()}~{e.date()}",
                       xaxis=dict(range=[s,e], tickformat='%Y-%m-%d'), template='plotly_white')
    pct = (norm.iloc[-1] - 100)
    bar = go.Figure(go.Bar(x=pct.values, y=pct.index, orientation='h'))
    bar.update_layout(title="국가별 Change (%)", template='plotly_white', yaxis=dict(autorange='reversed'))
    return line, bar

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)