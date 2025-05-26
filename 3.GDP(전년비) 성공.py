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
Q_MAP = {'Q1':'-03-31','Q2':'-06-30','Q3':'-09-30','Q4':'-12-31'}

# HTTP 세션 설정
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=3))

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

# 전년비 데이터 로드
yoy = fetch_bea('T10111')

# 관심 시리즈 및 라벨
targets = [
    'Gross domestic product',
    'Personal consumption expenditures',
    'Gross private domestic investment',
    'Government consumption expenditures and gross investment',
    'Exports',
    'Imports'
]
labels = {
    'Gross domestic product':'GDP',
    'Personal consumption expenditures':'소비',
    'Gross private domestic investment':'투자',
    'Government consumption expenditures and gross investment':'정부지출',
    'Exports':'수출',
    'Imports':'수입'
}

# Pivot 전년비 데이터
yoy_df = (
    yoy[yoy['LineDescription'].isin(targets)]
       .groupby(['Date','LineDescription'], as_index=False)['Value'].last()
       .pivot(index='Date', columns='LineDescription', values='Value')
       .sort_index()
)

# Dash 앱 초기화
app = dash.Dash(__name__)

# 앱 레이아웃 정의
app.layout = html.Div([
    html.H2('경제성장률(GDP, 전년비)', style={'textAlign':'center'}),
    dcc.RangeSlider(
        id='yoy-slider',
        min=0, max=len(yoy_df)-1,
        value=[len(yoy_df)-8, len(yoy_df)-1],
        marks={i: str(d.year) for i, d in enumerate(yoy_df.index) if d.month == 12 and d.year % 5 == 0},
        allowCross=False
    ),
    html.Div(id='graphs', style={
        'display': 'grid',
        'gridTemplateColumns': '1fr 1fr',
        'gap': '20px',
        'marginTop': '20px',
        'paddingBottom':'120px'
    }),
], style={'padding':'20px','maxWidth':'1000px','margin':'0 auto'})

# 전년비 콜백 정의
@app.callback(
    Output('graphs', 'children'),
    Input('yoy-slider', 'value')
)
def update_yoy(rng):
    s, e = rng
    dates = yoy_df.index[s:e+1]
    cards = []
    colors = px.colors.qualitative.Plotly
    for i, key in enumerate(targets):
        vals = yoy_df[key].loc[dates]
        fig = go.Figure(go.Scatter(
            x=dates, y=vals, mode='lines', line_color=colors[i],
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f}%'  # 월일까지 표시
        ))
        fig.update_layout(
            template='plotly_white',
            title=labels[key],
            margin={'t':50,'b':100},
            xaxis=dict(tickformat='%Y')
        )
        ld = dates[-1].strftime('%Y-%m-%d')
        lv = vals.iloc[-1]
        fig.add_annotation(
            text=f"전년비, %, 최신: {ld}, 값: {lv:.2f}%",
            xref='paper', yref='paper', x=0, y=-0.25,
            showarrow=False, font={'size':10}
        )
        cards.append(html.Div(
            dcc.Graph(figure=fig),
            style={'border':'1px solid #ddd','padding':'10px','borderRadius':'5px'}
        ))
    return cards

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=False)