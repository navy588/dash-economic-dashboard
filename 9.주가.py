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

# 지표 설정 (기업 주가 페이지)
main_indices = {
    "SP500": "^GSPC",
    "Nasdaq": "^IXIC",
    "DowJones": "^DJI"
}
title_map = {"SP500": "S&P 500", "Nasdaq": "Nasdaq", "DowJones": "Dow Jones"}

m7_tickers = {
    "애플": "AAPL",
    "마이크로소프트": "MSFT",
    "구글": "GOOGL",
    "아마존": "AMZN",
    "메타": "META",
    "테슬라": "TSLA",
    "엔비디아": "NVDA"
}

# 데이터 불러오기
df_main = load_data(list(main_indices.values()))
df_m7 = load_data(list(m7_tickers.values()))
dates = df_main.index.union(df_m7.index)

# 카드 스타일 헬퍼
def card(children):
    return html.Div(children, style={
        'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '5px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin':'5px', 'width':'48%'
    })

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("주가 (기업)", style={'textAlign':'center'}),

    # 주요 지표 2열
    html.Div([
        card([dcc.Graph(id=f"{idx}_chart"), html.P(id=f"{idx}_caption")])
        for idx in main_indices
    ], style={'display':'flex','flexWrap':'wrap','justifyContent':'space-between'}),

    # M7 기업 카드
    html.Div([
        card([dcc.Graph(id='m7_chart'), html.P(id='m7_caption')])
    ], style={'display':'flex','justifyContent':'space-between'}),

    # M7 정규화 섹션
    html.H2("M7 기업 정규화 지표 비교", style={'textAlign':'center','marginTop':'40px'}),
    html.Div([
        html.Div([
            html.Label("기간 선택:"),
            dcc.DatePickerRange(
                id='m7_norm_range',
                min_date_allowed=dates.min(),
                max_date_allowed=dates.max(),
                start_date=dates.min(),
                end_date=dates.max()
            ),
            html.Label("비교 대상:"),
            dcc.Dropdown(
                id='m7_norm_select', multi=True,
                options=[{'label':k,'value':v} for k,v in m7_tickers.items()],
                value=list(m7_tickers.values())
            )
        ], style={'display':'flex','flexWrap':'nowrap','gap':'10px','justifyContent':'center','alignItems':'center','marginBottom':'20px'}),
        html.Div([
            card([dcc.Graph(id='m7_norm_line'), html.P(id='m7_norm_caption')]),
            card([dcc.Graph(id='m7_norm_bar'), html.P(id='m7_bar_caption')])
        ], style={'display':'flex','justifyContent':'space-between'})
    ], style={'maxWidth':'1200px','margin':'0 auto'})
], style={'backgroundColor':'white','padding':'20px'})

# 콜백: SP500
@app.callback(
    Output('SP500_chart', 'figure'),
    Output('SP500_caption', 'children'),
    Input('m7_norm_range','start_date'), Input('m7_norm_range','end_date')
)
def update_sp500(start, end):
    s,e = pd.to_datetime(start), pd.to_datetime(end)
    series = load_data([main_indices['SP500']])['^GSPC'].loc[s:e].dropna()
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode='lines', hovertemplate='%{y:.1f}'))
    last = series.iloc[-1]; date = series.index.max()
    fig.update_layout(title=title_map['SP500'], xaxis={'tickformat':'%Y-%m-%d'}, yaxis_title='Price', template='plotly_white')
    return fig, f"최신: {date.date()} / {last:.1f}"

# 콜백: Nasdaq
@app.callback(
    Output('Nasdaq_chart', 'figure'), Output('Nasdaq_caption', 'children'),
    Input('m7_norm_range','start_date'), Input('m7_norm_range','end_date')
)
def update_nasdaq(start, end):
    s,e = pd.to_datetime(start), pd.to_datetime(end)
    series = load_data([main_indices['Nasdaq']])['^IXIC'].loc[s:e].dropna()
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode='lines', hovertemplate='%{y:.1f}'))
    last = series.iloc[-1]; date = series.index.max()
    fig.update_layout(title=title_map['Nasdaq'], xaxis={'tickformat':'%Y-%m-%d'}, yaxis_title='Price', template='plotly_white')
    return fig, f"최신: {date.date()} / {last:.1f}"

# 콜백: DowJones
@app.callback(
    Output('DowJones_chart', 'figure'), Output('DowJones_caption', 'children'),
    Input('m7_norm_range','start_date'), Input('m7_norm_range','end_date')
)
def update_dow(start, end):
    s,e = pd.to_datetime(start), pd.to_datetime(end)
    series = load_data([main_indices['DowJones']])['^DJI'].loc[s:e].dropna()
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode='lines', hovertemplate='%{y:.1f}'))
    last = series.iloc[-1]; date = series.index.max()
    fig.update_layout(title=title_map['DowJones'], xaxis={'tickformat':'%Y-%m-%d'}, yaxis_title='Price', template='plotly_white')
    return fig, f"최신: {date.date()} / {last:.1f}"

# 콜백: M7 통합
@app.callback(
    Output('m7_chart','figure'), Output('m7_caption','children'),
    Input('m7_norm_range','start_date'), Input('m7_norm_range','end_date')
)
def update_m7(start, end):
    s,e = pd.to_datetime(start), pd.to_datetime(end)
    df = df_m7.loc[s:e].dropna(how='all')
    fig = go.Figure()
    for label,ticker in m7_tickers.items():
        fig.add_trace(go.Scatter(x=df.index, y=df[ticker], mode='lines', name=label, hovertemplate='%{y:.1f}'))
    last = df.iloc[-1]; date = df.index.max()
    cap = ' / '.join([f"{label}: {last[ticker]:.1f}" for label,ticker in m7_tickers.items()])
    fig.update_layout(title='M7 기업', xaxis={'tickformat':'%Y-%m-%d'}, yaxis_title='Price', template='plotly_white')
    return fig, f"최신: {date.date()} / {cap}"

# 콜백: M7 정규화
@app.callback(
    Output('m7_norm_line','figure'), Output('m7_norm_bar','figure'), Output('m7_norm_caption','children'),
    Input('m7_norm_range','start_date'), Input('m7_norm_range','end_date'), Input('m7_norm_select','value')
)
def update_m7_norm(start, end, tickers):
    s,e = pd.to_datetime(start), pd.to_datetime(end)
    df = load_data(tickers).loc[s:e].dropna(how='all')
    base = df.iloc[0]; norm = df.div(base).mul(100)
    line = go.Figure(); bar = go.Figure()
    captions = []
    for label,ticker in m7_tickers.items():
        if ticker in norm.columns:
            val = norm[ticker].iloc[-1]
            line.add_trace(go.Scatter(x=norm.index, y=norm[ticker], mode='lines', name=label, hovertemplate='%{y:.1f}'))
            bar.add_trace(go.Bar(x=[val-100], y=[label], orientation='h', hovertemplate='%{x:.1f}%'))
            captions.append(f"{label}: {val-100:.1f}%")
    date = norm.index.max()
    cap = f"최신: {date.date()} / " + ' / '.join(captions)
    line.update_layout(title="M7 정규화 선형", xaxis={'tickformat':'%Y-%m-%d'}, template='plotly_white')
    bar.update_layout(title="M7 정규화 변화율", yaxis={'autorange':'reversed'}, template='plotly_white')
    return line, bar, cap

if __name__=='__main__':
    app.run(debug=True)