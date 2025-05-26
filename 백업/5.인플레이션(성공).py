import dash
from dash import dcc, html, Input, Output
import pandas as pd
from fredapi import Fred
import plotly.graph_objs as go

# FRED API 키
fred = Fred(api_key="fb40b5238c2c5ff6b281706029681f15")

# 시리즈 코드 정의
series_codes = {
    'CPI': 'CPIAUCSL',
    'Core CPI': 'CPILFESL',
    'PPI': 'PPIACO',
    'PCE': 'PCEPI',
    'Core PCE': 'PCEPILFE',
    'Expectations 1Y': 'EXPINF1YR',
    'Expectations 2Y': 'EXPINF2YR',
    'Expectations 3Y': 'EXPINF3YR',
    'Expectations 5Y': 'EXPINF5YR',
    'Expectations 10Y': 'EXPINF10YR',
    'Expectations 30Y': 'EXPINF30YR'
}
# 케이스실러 주택 가격 지수
case_code = 'CSUSHPINSA'

# 데이터 로드 및 전처리
data = pd.concat({name: fred.get_series(code) for name, code in series_codes.items()}, axis=1)
data.index = pd.to_datetime(data.index)
data = data.loc['2000-01-01':]

# 케이스실러 시리즈 로드
case_series = fred.get_series(case_code)
case_series.index = pd.to_datetime(case_series.index)
case_series = case_series.loc['2000-01-01':]

# 전년비 계산 함수
def calc_yoy(series):
    return series.pct_change(12, fill_method=None) * 100

# Dash 앱 초기화
app = dash.Dash(__name__)

# 차트 생성 함수
def make_time_chart(title, traces, x_range, ytitle='%'):
    fig = go.Figure(data=traces)
    fig.update_layout(
        title={'text': title, 'x':0.5},
        xaxis={'range': x_range, 'tickformat':'%Y-%m', 'showgrid':True},
        yaxis={'title':ytitle},
        template='plotly_white',
        hovermode='x unified',
        margin={'t':50,'b':30}
    )
    return fig

# 레이아웃 정의
app.layout = html.Div([
    html.H2('인플레이션', style={'textAlign':'center', 'margin':'20px 0'}),
    dcc.RangeSlider(
        id='inflation-slider',
        min=0,
        max=len(data.index)-1,
        value=[0, len(data.index)-1],
        marks={i: str(date.year) for i, date in enumerate(data.index) if date.month==1 and date.year%5==0},
        allowCross=False,
        tooltip={'placement':'bottom'}
    ),
    html.Div(id='charts-container', style={
        'display':'grid',
        'gridTemplateColumns':'1fr 1fr',
        'gap':'20px',
        'padding':'20px'
    })
], style={'maxWidth':'1200px','margin':'0 auto','padding':'20px'})

# 콜백: 슬라이더에 따라 차트 업데이트
@app.callback(
    Output('charts-container', 'children'),
    Input('inflation-slider', 'value')
)
def update_charts(range_idx):
    start, end = range_idx
    idx = data.index[start:end+1]

    # 1. 소비자물가지수
    cpi_y = calc_yoy(data['CPI']).reindex(idx)
    core_cpi_y = calc_yoy(data['Core CPI']).reindex(idx)
    ppi_y = calc_yoy(data['PPI']).reindex(idx)
    fig1 = make_time_chart('소비자물가지수', [
        go.Scatter(x=idx, y=cpi_y, mode='lines', name='CPI', hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%'),
        go.Scatter(x=idx, y=core_cpi_y, mode='lines', name='Core CPI', hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%'),
        go.Scatter(x=idx, y=ppi_y, mode='lines', name='PPI', hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%')
    ], x_range=[idx.min(), idx.max()])
    last_date1 = calc_yoy(data['CPI']).last_valid_index()
    foot1 = html.P(
        f"전년비(%) | 최신: {last_date1.strftime('%Y-%m-%d')} | CPI {cpi_y.loc[last_date1]:.2f}% | Core CPI {core_cpi_y.loc[last_date1]:.2f}% | PPI {ppi_y.loc[last_date1]:.2f}%",
        style={'fontSize':'12px','color':'#555'}
    )

    # 2. 개인소비지출 물가지수
    pce_y = calc_yoy(data['PCE']).reindex(idx)
    core_pce_y = calc_yoy(data['Core PCE']).reindex(idx)
    fig2 = make_time_chart('개인소비지출 물가지수', [
        go.Scatter(x=idx, y=pce_y, mode='lines', name='PCE', hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%'),
        go.Scatter(x=idx, y=core_pce_y, mode='lines', name='Core PCE', hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%')
    ], x_range=[idx.min(), idx.max()])
    last_date2 = calc_yoy(data['PCE']).last_valid_index()
    foot2 = html.P(
        f"전년비(%) | 최신: {last_date2.strftime('%Y-%m-%d')} | PCE {pce_y.loc[last_date2]:.2f}% | Core PCE {core_pce_y.loc[last_date2]:.2f}%",
        style={'fontSize':'12px','color':'#555'}
    )

    # 3. 기대 인플레이션
    exp_df = data[[col for col in data.columns if 'Expectations' in col]].reindex(idx)
    traces_exp = [
        go.Scatter(x=idx, y=exp_df[col], mode='lines', name=col.split()[-1], hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%')
        for col in exp_df.columns
    ]
    y_min = exp_df.min().min()*0.9
    y_max = exp_df.max().max()*1.1
    fig3 = go.Figure(data=traces_exp)
    fig3.update_layout(
        title={'text':'기대 인플레이션', 'x':0.5},
        xaxis={'range':[idx.min(), idx.max()], 'tickformat':'%Y-%m','showgrid':True},
        yaxis={'title':'%', 'range':[y_min, y_max]},
        template='plotly_white',
        hovermode='x unified',
        margin={'t':50,'b':30}
    )
    last_date3 = calc_yoy(data['Expectations 2Y']).last_valid_index()
    foot3 = html.P(
        f"전년비(%) | 최신: {last_date3.strftime('%Y-%m-%d')} | 2Y {exp_df.loc[last_date3,'Expectations 2Y']:.2f}% | 10Y {exp_df.loc[last_date3,'Expectations 10Y']:.2f}%",
        style={'fontSize':'12px','color':'#555'}
    )

    # 4. 케이스실러 주택 가격 지수
    case_y = calc_yoy(case_series).reindex(idx)
    fig4 = make_time_chart('케이스실러 주택 가격 지수', [
        go.Scatter(x=idx, y=case_y, mode='lines', name='CS HPI', hovertemplate='%{x|%Y-%m-%d} %{y:.2f}%')
    ], x_range=[idx.min(), idx.max()])
    last_case_date = calc_yoy(case_series).last_valid_index()
    foot4 = html.P(
        f"전년비(%) | 최신: {last_case_date.strftime('%Y-%m-%d')} | CS HPI {case_y.loc[last_case_date]:.2f}%",
        style={'fontSize':'12px','color':'#555'}
    )

    # 컴포넌트 배열
    comps = []
    for fig, foot in [(fig1, foot1), (fig2, foot2), (fig3, foot3), (fig4, foot4)]:
        comps.append(
            html.Div([
                dcc.Graph(figure=fig, config={'displayModeBar':False}),
                foot
            ], style={
                'backgroundColor':'white',
                'borderRadius':'8px',
                'padding':'15px',
                'boxShadow':'0 2px 4px rgba(0,0,0,0.1)'
            })
        )
    return comps

if __name__ == '__main__':
    app.run(debug=False)