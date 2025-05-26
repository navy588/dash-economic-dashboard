import dash
from dash import dcc, html, Input, Output
import pandas as pd
from fredapi import Fred
import plotly.graph_objs as go

# FRED API 키
fred = Fred(api_key="fb40b5238c2c5ff6b281706029681f15")

# 시리즈 코드 정의
series_codes = {
    '기준금리': 'DFEDTARU',
    '3개월': 'DGS3MO',
    '2년': 'DGS2',
    '5년': 'DGS5',
    '10년': 'DGS10',
    '30년': 'DGS30',
    '10Y-3M': 'T10Y3M',
    '10Y-2Y': 'T10Y2Y',
    '30Y 모기지': 'MORTGAGE30US'
}

# 데이터 로드 및 2000년 이후 필터링
data = pd.concat({name: fred.get_series(code) for name, code in series_codes.items()}, axis=1)
data.index = pd.to_datetime(data.index)
data = data.loc['2000-01-01':]

# 전체 최신일
latest_overall = data.dropna(how='all').index.max()

# 슬라이더 마크: 2000년부터 5년 단위, 중복 제거
marks = {}
for year in range(2000, latest_overall.year+1, 5):
    idx = data.index.get_loc(data.index[data.index.year == year][0])
    marks[idx] = str(year)

# Dash 앱 초기화
app = dash.Dash(__name__)

# 차트 생성 함수
def make_time_chart(title, traces, x_start):
    fig = go.Figure(data=traces)
    fig.update_layout(
        title={'text': title, 'x': 0.5},
        xaxis={
            'range': [x_start, latest_overall],
            'tickformat': '%Y-%m',
            'showgrid': True
        },
        yaxis={
            'title': '%',
            'showgrid': True
        },
        template='plotly_white',
        hovermode='x unified',
        margin={'t': 50, 'b': 30}
    )
    return fig

# 레이아웃 정의
app.layout = html.Div([
    html.H2('금리 대시보드', style={'textAlign': 'center', 'margin': '20px 0'}),
    dcc.RangeSlider(
        id='rate-slider', min=0, max=len(data.index)-1,
        value=[0, len(data.index)-1], marks=marks,
        allowCross=False, tooltip={'placement':'bottom'}
    ),
    html.Div(id='charts-container', style={'display': 'grid','gridTemplateColumns': '1fr 1fr','gap': '20px','padding': '20px'})
], style={'maxWidth': '1200px','margin': '0 auto','padding': '20px'})

# 콜백 정의: 슬라이더는 시작만 제어, 끝은 항상 최신으로 설정
@app.callback(Output('charts-container','children'), Input('rate-slider','value'))
def update_charts(slider_range):
    start_idx = slider_range[0]
    # 시작일만 반영, 끝은 최신 데이터
    window = data.iloc[start_idx:]
    start_date = window.index[0]
    comps = []

    # 1. 기준금리
    s0 = window['기준금리']
    trace0 = go.Scatter(x=s0.index, y=s0, mode='lines', connectgaps=True,
                        name='기준금리', hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%')
    fig1 = make_time_chart('기준금리', [trace0], start_date)
    last0 = s0.dropna().index.max()
    comps.append(html.Div([
        dcc.Graph(figure=fig1, config={'displayModeBar': False}),
        html.P(f"최신: {last0.date()} | {s0.loc[last0]:.2f}%", style={'fontSize': '12px','color': '#555'})
    ], style={'backgroundColor':'white','borderRadius':'8px','padding':'15px','boxShadow':'0 2px 4px rgba(0,0,0,0.1)'}) )

    # 2. 국채 금리
    bonds = ['3개월','2년','5년','10년','30년']
    traces1 = []
    last_dates = []
    for col in bonds:
        s = window[col]
        traces1.append(go.Scatter(x=s.index, y=s, mode='lines', connectgaps=True,
                                  name=col, hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%'))
        last_dates.append(s.dropna().index.max())
    common_last = min(last_dates)
    fig2 = make_time_chart('국채 금리', traces1, start_date)
    vals1 = ' | '.join([f"{col} {data.at[common_last,col]:.2f}%" for col in bonds])
    comps.append(html.Div([
        dcc.Graph(figure=fig2, config={'displayModeBar':False}),
        html.P(f"최신: {common_last.date()} | {vals1}", style={'fontSize':'12px','color':'#555'})
    ], style={'backgroundColor':'white','borderRadius':'8px','padding':'15px','boxShadow':'0 2px 4px rgba(0,0,0,0.1)'}) )

    # 3. 스프레드
    spd = ['10Y-3M','10Y-2Y']
    traces2 = []
    spd_dates = []
    for col in spd:
        s = window[col]
        traces2.append(go.Scatter(x=s.index, y=s, mode='lines', connectgaps=True,
                                  name=col, hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%'))
        spd_dates.append(s.dropna().index.max())
    common_spd = min(spd_dates)
    fig3 = make_time_chart('장단기 금리 스프레드', traces2, start_date)
    vals2 = ' | '.join([f"{col} {data.at[common_spd,col]:.2f}%" for col in spd])
    comps.append(html.Div([
        dcc.Graph(figure=fig3, config={'displayModeBar':False}),
        html.P(f"최신: {common_spd.date()} | {vals2}", style={'fontSize':'12px','color':'#555'})
    ], style={'backgroundColor':'white','borderRadius':'8px','padding':'15px','boxShadow':'0 2px 4px rgba(0,0,0,0.1)'}) )

    # 4. 30Y 모기지 금리
    s3 = window['30Y 모기지']
    trace3 = go.Scatter(x=s3.index, y=s3, mode='lines', connectgaps=True,
                        name='30Y 모기지', hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%')
    fig4 = make_time_chart('30년 모기지 금리', [trace3], start_date)
    last3 = s3.dropna().index.max()
    comps.append(html.Div([
        dcc.Graph(figure=fig4, config={'displayModeBar':False}),
        html.P(f"최신: {last3.date()} | {s3.loc[last3]:.2f}%", style={'fontSize':'12px','color':'#555'})
    ], style={'backgroundColor':'white','borderRadius':'8px','padding':'15px','boxShadow':'0 2px 4px rgba(0,0,0,0.1)'}) )

    return comps

if __name__=='__main__':
    app.run(debug=False)
