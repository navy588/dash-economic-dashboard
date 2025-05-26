import dash
from dash import dcc, html, Input, Output
import pandas as pd
from fredapi import Fred
import plotly.graph_objs as go
import plotly.express as px

# FRED Nowcast 데이터 로드
fred = Fred(api_key="fb40b5238c2c5ff6b281706029681f15")
now_codes = {'Atlanta':'GDPNOW', 'St. Louis':'STLENI'}
now_df = pd.concat({k: fred.get_series(v) for k, v in now_codes.items()}, axis=1)
now_df.index = pd.to_datetime(now_df.index.to_period('Q').to_timestamp(how='end'))

# Dash 앱 초기화
app = dash.Dash(__name__)

# hover용 문자열과 연도 마킹값 미리 생성
dates_all = now_df.index
dates_str_all = dates_all.strftime('%Y-%m-%d')
years = dates_all.year
year_marks = {i: str(years[i]) for i in range(len(dates_all)) if dates_str_all[i].endswith('-06-30')}

# 레이아웃: 단일 탭 (GDP Nowcast)
app.layout = html.Div([
    html.H2('경제성장률 전망 (GDP Nowcast, 전기비 연율)', style={'textAlign':'center'}),
    dcc.RangeSlider(
        id='now-slider',
        min=0,
        max=len(dates_all)-1,
        value=[len(dates_all)-8, len(dates_all)-1],
        marks=year_marks,
        allowCross=False
    ),
    html.Div(id='nowcast-graphs', style={
        'display':'grid',
        'gridTemplateColumns':'1fr 1fr',
        'gap':'20px',
        'marginTop':'20px'
    })
], style={'padding':'20px','maxWidth':'1000px','margin':'0 auto'})

# Nowcast 그래프 콜백
@app.callback(
    Output('nowcast-graphs', 'children'),
    Input('now-slider', 'value')
)
def update_nowcast(rng):
    s, e = rng
    dates = dates_all[s:e+1]
    dates_str = dates.strftime('%Y-%m-%d')  # hover용 문자열
    dates_minus = (dates - pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # 하루 전 문자열
    cards = []
    colors = px.colors.qualitative.Plotly
    for i, region in enumerate(now_codes.keys()):
        vals = now_df[region].loc[dates]
        fig = go.Figure(go.Scatter(
            x=dates_str,
            y=vals,
            mode='lines',
            line_color=colors[i],
            name=f'{region} Fed Nowcast',
            customdata=dates_minus,
            hovertemplate='%{customdata}<br>%{y:.2f}%'
        ))
        fig.update_layout(
            template='plotly_white',
            title=f'{region} Fed Nowcast',
            xaxis=dict(tickformat='%Y'),
            margin={'t':50, 'b':100}
        )
        latest_date = dates_minus[-1]
        latest_val = vals.iloc[-1]
        fig.add_annotation(
            text=f"최신값: {latest_date}, {latest_val:.2f}%",
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