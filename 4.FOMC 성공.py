import dash
from dash import dcc, html, Input, Output
import pandas as pd
from fredapi import Fred
import plotly.graph_objs as go
import plotly.express as px

# FRED API 키
fred = Fred(api_key="fb40b5238c2c5ff6b281706029681f15")

# 전망 시리즈 및 장기 변수
proj_codes = {
    'Federal funds rate': ('FEDTARMD', 'FEDTARMDLR'),
    'GDP growth': ('GDPC1MD', 'GDPC1MDLR'),
    'PCE inflation': ('PCECTPIMD', 'PCECTPIMDLR'),
    'Unemployment rate': ('UNRATEMD', 'UNRATEMDLR')
}

# 챠트 제목 및 각주 매핑
title_map = {
    'Federal funds rate': '기준금리 (Federal funds rate)',
    'GDP growth': '경제성장률',
    'PCE inflation': '인플레이션 (PCE)',
    'Unemployment rate': '실업률'
}
footnote_map = {
    'Federal funds rate': '※ 장기 목표 수준 금리',
    'GDP growth': '※ 장기 잠재 성장률 추정치',
    'PCE inflation': '※ FOMC가 장기적으로 이중 목표와 가장 일치한다고 판단하는 인플레이션율',
    'Unemployment rate': '※ 장기 정상 실업률 추정치'
}

# 연도별 전망(2025-2027)과 장기(LR)
years = ['2025', '2026', '2027', '장기']
data = {}
for name, (code_md, code_lr) in proj_codes.items():
    ts = fred.get_series(code_md)
    ts_lr = fred.get_series(code_lr)
    vals = []
    for y in [2025, 2026, 2027]:
        yearly = ts[ts.index.year == y]
        vals.append(yearly.iloc[-1] if len(yearly) else None)
    vals.append(ts_lr.iloc[-1] if len(ts_lr) else None)
    data[name] = vals

# DataFrame 생성
df = pd.DataFrame(data, index=years).round(2)

# 최대값 기준 상한 설정 (20% 여유)
ymax = df.max().max() * 1.2

# Dash 앱 초기화
app = dash.Dash(__name__)

# 스타일 설정: Qualitative palette for distinct colors
palette = px.colors.qualitative.Plotly

# 레이아웃
app.layout = html.Div([
    html.H2('FOMC 전망', style={'textAlign':'center', 'marginBottom':'5px'}),
    html.P(
        '※ 장기 전망은 정책 입안자가 추가적인 충격이 없고 적절한 통화 정책 하에서 경제가 시간이 지남에 따라 수렴할 것으로 예상되는 전망치',
        style={'fontSize':'12px','fontStyle':'italic','color':'#666','marginTop':'0','marginBottom':'20px', 'textAlign':'center'}
    ),
    html.Div([
        html.Div([
            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Bar(
                            x=years,
                            y=df[col],
                            text=df[col],
                            textposition='inside',
                            textfont=dict(color='white'),
                            marker=dict(color=palette[i], line=dict(color='#333', width=1.5)),
                            hovertemplate=f'%{{x}}<br>{{y:.2f}}%'
                        )
                    ],
                    layout=go.Layout(
                        title={'text': title_map[col], 'x':0.5, 'xanchor':'center', 'font':{'size':20}},
                        template='simple_white',
                        yaxis=dict(title='%', range=[0, ymax], gridcolor='#ddd', gridwidth=1),
                        xaxis=dict(showgrid=False, tickfont={'size':14}),
                        plot_bgcolor='white',
                        margin={'t':70,'b':50,'l':50,'r':20}
                    )
                ), config={'displayModeBar': False}
            ),
            html.P(footnote_map[col], style={'fontSize':'12px','color':'#666','marginTop':'10px', 'textAlign':'center'})
        ], style={
            'backgroundColor':'white',
            'borderRadius':'10px',
            'boxShadow':'0 4px 8px rgba(0,0,0,0.1)',
            'padding':'20px'
        })
        for i, col in enumerate(df.columns)
    ], style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':'30px'})
], style={'padding':'40px','maxWidth':'1000px','margin':'0 auto', 'backgroundColor':'#f0f2f5'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=False)