import dash
from dash import dcc, html, Input, Output
import pandas as pd
from fredapi import Fred
import plotly.graph_objs as go

# ── FRED API 키 설정 및 Dash 앱 초기화
fred = Fred(api_key="fb40b5238c2c5ff6b281706029681f15")
app = dash.Dash(__name__)

# ── 가져올 시리즈 코드 정의
series_codes = {
    '실업률': 'UNRATE',
    '경제활동참가율': 'CIVPART',
    '고용률': 'EMRATIO',
    '비농업고용자수': 'PAYEMS',
    '구인 건수': 'JTSJOL',
    '자발적 이직률': 'JTSQUR',
    '평균 시급': 'CES0500000003'
}

# ── 데이터 로드 및 전처리
def load_data():
    df = pd.concat({name: fred.get_series(code) for name, code in series_codes.items()}, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.loc['2000-01-01':]
    df['전월비 비농업고용자수(천명)'] = df['비농업고용자수'].diff()
    df['평균 시급 증감률'] = df['평균 시급'].pct_change(fill_method=None) * 100
    return df

# ── 데이터 준비
data = load_data()
latest = data.dropna(how='all').index.max()
marks = {i: str(dt.year) for i, dt in enumerate(data.index) if dt.month==1 and dt.year%5==0}

# ── 레이아웃
app.layout = html.Div([
    html.H2('노동시장', style={'textAlign':'center'}),
    dcc.RangeSlider(id='labour-slider', min=0, max=len(data.index)-1,
                    value=[0, len(data.index)-1], marks=marks, allowCross=False,
                    tooltip={'placement':'bottom'}),
    html.Div(id='charts-container', style={'display':'grid',
        'gridTemplateColumns':'1fr 1fr','gap':'20px','padding':'20px'})
], style={'maxWidth':'1200px','margin':'0 auto','padding':'20px'})

# ── 콜백: 시작만 슬라이더 적용, 끝은 항상 최신값 유지
@app.callback(Output('charts-container','children'), Input('labour-slider','value'))
def update_charts(slider_range):
    start_idx, _ = slider_range
    window = data.iloc[start_idx:]
    start = window.index[0]
    end = latest
    comps = []
    box = {'backgroundColor':'white','padding':'10px','borderRadius':'5px'}

    # 1. 실업률
    u = window['실업률'].dropna()
    last = u.index.max(); val = u.loc[last]
    fig1 = go.Figure([go.Scatter(x=u.index, y=u, mode='lines')])
    fig1.update_layout(title='실업률', xaxis={'range':[start,end],'tickformat':'%Y-%m'}, template='plotly_white')
    comps.append(html.Div([dcc.Graph(figure=fig1, config={'displayModeBar':False}),
        html.P(f"단위: %  |   {last.date()} 실업률 {val:.2f}%")], style=box))

    # 2. 경제활동참가율 & 고용률
    part = window['경제활동참가율'].dropna(); emp = window['고용률'].dropna()
    last_p, val_p = part.index.max(), part.iloc[-1]
    last_e, val_e = emp.index.max(), emp.iloc[-1]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=part.index, y=part, name='참가율', yaxis='y1'))
    fig2.add_trace(go.Scatter(x=emp.index, y=emp, name='고용률', yaxis='y2'))
    fig2.update_layout(title='경제활동참가율 및 고용률', xaxis={'range':[start,end],'tickformat':'%Y-%m'},
                       yaxis=dict(title='% (참가율)', side='left'), yaxis2=dict(title='% (고용률)', overlaying='y', side='right'),
                       template='plotly_white')
    comps.append(html.Div([dcc.Graph(figure=fig2, config={'displayModeBar':False}),
        html.P(f"단위: %  |   {max(last_p,last_e).date()} 참가율 {val_p:.2f}%, 고용률 {val_e:.2f}%")], style=box))

    # 3. 비농업고용자수 변화
    c = window['전월비 비농업고용자수(천명)'].dropna()
    last_c, val_c = c.index.max(), c.iloc[-1]
    fig3 = go.Figure([go.Scatter(x=c.index, y=c, mode='lines')])
    fig3.update_layout(title='비농업고용자수 변화', xaxis={'range':[start,end],'tickformat':'%Y-%m'}, yaxis={'title':'천명'}, template='plotly_white')
    comps.append(html.Div([dcc.Graph(figure=fig3, config={'displayModeBar':False}),
        html.P(f"단위: 전월비, 천명  |   {last_c.date()} {val_c:.0f}천명")], style=box))

    # 4. 구인 건수 및 이직률
    j = window['구인 건수'].dropna(); q = window['자발적 이직률'].dropna()
    last_j, val_j = j.index.max(), j.iloc[-1]
    last_q, val_q = q.index.max(), q.iloc[-1]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=j.index, y=j, mode='lines', name='구인 건수', yaxis='y1'))
    fig4.add_trace(go.Scatter(x=q.index, y=q, mode='lines', name='이직률', yaxis='y2'))
    fig4.update_layout(title='구인 건수 및 이직률', xaxis={'range':[start,end],'tickformat':'%Y-%m'},
                       yaxis=dict(title='건수', side='left'), yaxis2=dict(title='%', overlaying='y', side='right'), template='plotly_white')
    comps.append(html.Div([dcc.Graph(figure=fig4, config={'displayModeBar':False}),
        html.P(f"단위: 건수 / %  |   {max(last_j,last_q).date()} 구인 {val_j:.0f}건, 이직률 {val_q:.2f}%")], style=box))

    # 5. 평균 시급 증감률
    pct = window['평균 시급 증감률'].dropna(); lvl = window['평균 시급'].dropna()
    last_lvl, val_lvl = lvl.index.max(), lvl.iloc[-1]
    last_pct, val_pct = pct.index.max(), pct.iloc[-1]
    fig5 = go.Figure([go.Scatter(x=pct.index, y=pct, mode='lines')])
    fig5.update_layout(title='평균 시급 증감률', xaxis={'range':[start,end],'tickformat':'%Y-%m'}, yaxis={'title':'%'}, template='plotly_white')
    comps.append(html.Div([dcc.Graph(figure=fig5, config={'displayModeBar':False}),
        html.P(f"단위: % / 달러  |   {last_lvl.date()} 평균시급 {val_lvl:.2f}달러, 증감률 {val_pct:.2f}%")], style=box))

    return comps

# ── 실행
if __name__=='__main__':
    app.run(debug=True, use_reloader=False)