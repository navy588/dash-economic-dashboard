import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

# ── 통화 코드 정의 & 데이터 로드
currency_symbols = {
    '원': 'KRW=X',
    '달러 인덱스': 'DX-Y.NYB',
    '유로': 'EURUSD=X',
    '엔': 'JPY=X',
    '위안': 'CNY=X',
    '파운드': 'GBPUSD=X',
    '캐나다달러': 'CAD=X',
    '스웨덴 크로나': 'SEK=X',
    '스위스 프랑': 'CHF=X'
}

def load_currency():
    df = pd.DataFrame()
    for name, sym in currency_symbols.items():
        df[name] = yf.download(sym, start='2000-01-01', progress=False)['Close']
    df.index = pd.to_datetime(df.index)
    return df

# 데이터 준비
data = load_currency()
latest = data.dropna(how='all').index.max()
# 슬라이더 마크 설정: 1월만, 연도 중복 방지
marks = {}
seen = set()
for i, dt in enumerate(data.index):
    if dt.month == 1 and dt.year not in seen:
        marks[i] = dt.strftime('%Y')
        seen.add(dt.year)

# Dash 앱 초기화
app = dash.Dash(__name__)

# 레이아웃 정의
grid_style = {'display':'grid', 'gridTemplateColumns':'1fr 1fr', 'gap':'20px', 'padding':'20px'}
app.layout = html.Div([
    html.H2('환율', style={'textAlign':'center'}),
    dcc.RangeSlider(id='currency-slider', min=0, max=len(data.index)-1,
                    value=[0, len(data.index)-1], marks=marks, allowCross=False),
    html.Div(id='currency-charts', style=grid_style),
    # 정규화 섹션
    html.Div([
        html.H3('환율 정규화: 기간 및 통화 선택', style={'textAlign':'center', 'marginTop':'40px'}),
        html.Div([
            dcc.Dropdown(id='norm-currencies',
                         options=[{'label':k,'value':k} for k in currency_symbols],
                         value=list(currency_symbols), multi=True,
                         style={'width':'400px','display':'inline-block','marginRight':'20px'}),
            dcc.DatePickerRange(id='norm-range',
                                min_date_allowed=data.index.min(),
                                max_date_allowed=latest,
                                start_date=data.index.min(),
                                end_date=latest,
                                display_format='YYYY-MM-DD',
                                style={'display':'inline-block'})
        ], style={'textAlign':'center', 'marginBottom':'20px'}),
        html.Div([
            dcc.Graph(id='norm-line', config={'displayModeBar':False}),
            dcc.Graph(id='norm-bar', config={'displayModeBar':False})
        ], style={'display':'grid', 'gridTemplateColumns':'2fr 1fr', 'gap':'20px'})
    ], style={'maxWidth':'1000px', 'margin':'0 auto', 'padding':'20px'})
], style={'maxWidth':'1200px', 'margin':'0 auto', 'padding':'20px'})

# 메인 차트 콜백
@app.callback(Output('currency-charts', 'children'), Input('currency-slider', 'value'))
def update_currency(range_idx):
    s_idx, _ = range_idx
    window = data.iloc[s_idx:]
    start, end = window.index[0], latest
    comps = []
    box_style = {'backgroundColor':'white', 'padding':'10px', 'borderRadius':'5px'}
    for name in currency_symbols:
        ser = window[name].dropna()
        last_date = end if end in ser.index else ser.index.max()
        last_val = ser.loc[last_date]
        fig = go.Figure(go.Scatter(x=ser.index, y=ser.values, mode='lines',
                                   hovertemplate='%{x|%Y-%m-%d} %{y:.4f}', name=name))
        fig.update_layout(title=name, xaxis={'range':[start, end], 'tickformat':'%Y-%m-%d'},
                          template='plotly_white', yaxis={})
        caption = f"전일 종가  |   {last_date.date()} {name} {last_val:.2f}"
        comps.append(html.Div([dcc.Graph(figure=fig), html.P(caption)], style=box_style))
    return comps

# 정규화 차트 콜백 (범례 연동 via restyleData)
@app.callback(
    Output('norm-line', 'figure'),
    Output('norm-bar', 'figure'),
    Input('norm-range', 'start_date'),
    Input('norm-range', 'end_date'),
    Input('norm-currencies', 'value'),
    Input('norm-line', 'restyleData')
)
def update_normalized(start_date, end_date, currencies, restyle):
    # 기준일, 종료일
    s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
    # determine visible series from restyleData
    if restyle and isinstance(restyle, list):
        # restyleData structure: [changes, traceIndices]
        # get current figure from callback context via prevProps? fallback to all
        # simpler: keep all visible by value
        visible = currencies
    else:
        visible = currencies
    line_fig = go.Figure(); pct = {}
    for cur in currencies:
        if cur in visible:
            series = data[cur].dropna()
            base = series.asof(s)
            sub = series[(series.index>=s)&(series.index<=e)]
            norm = sub/base*100
            line_fig.add_trace(go.Scatter(
                x=norm.index, y=norm.values, mode='lines', name=cur,
                hovertemplate='%{x|%Y-%m-%d} %{y:.2f}'
            ))
            pct[cur] = norm.iloc[-1] - 100
    line_fig.update_layout(
        title=f"정규화 선형: {s.date()} ~ {e.date()}",
        xaxis={'range':[s,e], 'tickformat':'%Y-%m-%d'},
        yaxis={'title':'Index (Base=100)'}, template='plotly_white'
    )
    # 막대그래프
    bar_fig = go.Figure()
    for cur, val in pct.items():
        if cur in visible:
            bar_fig.add_trace(go.Bar(
                x=[val], y=[cur], orientation='h',
                hovertemplate=f"%{{x:.1f}}%"
            ))
    bar_fig.update_layout(
        title=f"{s.date()} 대비 {e.date()} 변화율 (%)", template='plotly_white',
        xaxis={'title':'% Change'}, yaxis={'autorange':'reversed'}, showlegend=False
    )
    return line_fig, bar_fig

# 실행
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
