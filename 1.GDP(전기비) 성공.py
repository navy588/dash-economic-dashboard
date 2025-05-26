import requests
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
from requests.adapters import HTTPAdapter

# --- BEA API 설정 ---
BASE_URL    = "https://apps.bea.gov/api/data"
BEA_API_KEY = "1A43CED6-707A-4E61-B475-A31AAB37AD01"
Q_MAP       = {'Q1':'-03-31','Q2':'-06-30','Q3':'-09-30','Q4':'-12-31'}

# HTTP 세션 설정
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=3))

# 데이터 로드 함수
def fetch_bea(table):
    params = {
        'UserID': BEA_API_KEY,
        'method': 'GetData',
        'datasetname': 'NIPA',
        'TableName': table,
        'Frequency': 'Q',
        'Year': ','.join(str(y) for y in range(2000, 2026)),
        'ResultFormat': 'JSON'
    }
    r = session.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get('BEAAPI', {}).get('Results', {}).get('Data', [])
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['TimePeriod'].str[:4] + df['TimePeriod'].str[4:].map(Q_MAP))
    df['Value'] = pd.to_numeric(df['DataValue'].str.replace(',', ''), errors='coerce')
    df['LineDescription'] = df['LineDescription'].astype(str)
    return df[['Date','LineDescription','Value']]

# BEA 데이터 불러오기
growth_df = fetch_bea('T10101')
df_contrib_raw = fetch_bea('T10102')

# 분석 대상 시리즈 및 라벨
series = [
    'Gross domestic product',
    'Personal consumption expenditures',
    'Gross private domestic investment',
    'Government consumption expenditures and gross investment',
    'Exports', 'Imports'
]
labels = {
    'Gross domestic product':'경제성장률',
    'Personal consumption expenditures':'소비',
    'Gross private domestic investment':'투자',
    'Government consumption expenditures and gross investment':'정부지출',
    'Exports':'수출',
    'Imports':'수입'
}

# Pivot 테이블 생성
df_pct = (
    growth_df[growth_df['LineDescription'].isin(series)]
    .pivot(index='Date', columns='LineDescription', values='Value')
    .sort_index()
)

df_contrib = (
    df_contrib_raw[df_contrib_raw['LineDescription'].isin(series)]
    .pivot(index='Date', columns='LineDescription', values='Value')
    .sort_index()
)

# Dash 앱 초기화
app = dash.Dash(__name__)

# 레이아웃 정의
app.layout = html.Div([
    html.H2('경제성장률(GDP, 전기비 연율)', style={'textAlign':'center'}),
    dcc.RangeSlider(
        id='date-slider',
        min=0, max=len(df_pct)-1,
        value=[len(df_pct)-20, len(df_pct)-1],
        marks={i: str(d.year) for i,d in enumerate(df_pct.index) if d.month==12 and d.year%5==0},
        allowCross=False
    ),
    dcc.Dropdown(
        id='contrib-dropdown',
        options=[{'label':f"{d.year} Q{(d.month-1)//3+1}", 'value':i} for i,d in enumerate(df_pct.index)],
        value=len(df_pct)-1,
        clearable=False,
        style={'width':'200px','margin':'20px auto'}
    ),
    html.Div(id='charts', style={
        'display':'grid',
        'gridTemplateColumns':'repeat(2, 1fr)',
        'gridAutoRows':'minmax(300px, auto)',
        'gap':'20px',
        'marginTop':'20px',
        'paddingBottom':'60px',
        'overflow':'visible'
    })
], style={'padding':'20px'})

# 콜백 정의
@app.callback(
    Output('charts','children'),
    [Input('date-slider','value'), Input('contrib-dropdown','value')]
)
def update_charts(date_range, contrib_idx):
    start, end = date_range
    dates = df_pct.index[start:end+1]
    latest = df_pct.index[contrib_idx]
    qnum = (latest.month-1)//3 + 1
    cards = []
    colors = px.colors.qualitative.Plotly

    # 경제성장률 차트
    vals = df_pct['Gross domestic product'].loc[dates]
    fig1 = go.Figure(go.Scatter(x=dates, y=vals, mode='lines', line_color=colors[0]))
    fig1.update_layout(template='plotly_white', title='경제성장률', margin={'t':50,'b':100})
    fig1.add_annotation(text=f"전기비, 연율, %, 최신: {latest.year} Q{qnum}, 값: {vals.loc[latest]:.2f}%",
                        xref='paper', yref='paper', x=0, y=-0.2, showarrow=False, font={'size':10})
    cards.append(html.Div(dcc.Graph(figure=fig1), style={'position':'relative'}))

    # 기여도 차트
    vals2 = df_contrib.loc[latest, series]
    fig2 = go.Figure(go.Bar(x=[labels[s] for s in series], y=vals2.values, marker_color=colors))
    fig2.update_layout(template='plotly_white', title='기여도', margin={'t':50,'b':100})
    fig2.add_annotation(text=f"전기비, 연율, %, 최신: {latest.year} Q{qnum}, GDP: {vals2['Gross domestic product']:.2f}%pt",
                        xref='paper', yref='paper', x=0, y=-0.2, showarrow=False, font={'size':10})
    cards.append(html.Div(dcc.Graph(figure=fig2), style={'position':'relative'}))

    # 소비/투자/... 차트
    for i,key in enumerate(series[1:], start=1):
        vals3 = df_pct[key].loc[dates]
        fig = go.Figure(go.Scatter(x=dates, y=vals3, mode='lines', line_color=colors[i]))
        fig.update_layout(template='plotly_white', title=labels[key], margin={'t':50,'b':100})
        fig.add_annotation(text=f"전기비, 연율, %, 최신: {latest.year} Q{qnum}, 값: {df_pct.loc[latest,key]:.2f}%",
                           xref='paper', yref='paper', x=0, y=-0.2, showarrow=False, font={'size':10})
        cards.append(html.Div(dcc.Graph(figure=fig), style={'position':'relative'}))

    return cards

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=False)
