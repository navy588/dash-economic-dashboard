# File: app.py
# ---------------------------------------------------------------------------------
import requests
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
from fredapi import Fred
from requests.adapters import HTTPAdapter

# API 설정
BASE_URL     = "https://apps.bea.gov/api/data"
BEA_API_KEY  = "1A43CED6-707A-4E61-B475-A31AAB37AD01"
FRED_API_KEY = "fb40b5238c2c5ff6b281706029681f15"
Q_MAP        = {'Q1':'-03-31','Q2':'-06-30','Q3':'-09-30','Q4':'-12-31'}

# HTTP 세션 with 재시도
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=3))

def init_data():
    def fetch_bea(table):
        params = {
            'UserID': BEA_API_KEY,
            'method': 'GetData',
            'datasetname': 'NIPA',
            'TableName': table,
            'Frequency': 'Q',
            'Year': ','.join(str(y) for y in range(2000,2026)),
            'ResultFormat': 'JSON'
        }
        r = session.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get('BEAAPI',{}).get('Results',{}).get('Data',[])
        df = pd.DataFrame(data)
        if df.empty: 
            return df
        df['Date'] = pd.to_datetime(
            df['TimePeriod'].str[:4] + df['TimePeriod'].str[4:].map(Q_MAP)
        )
        df['LineDescription'] = df['LineDescription'].astype(str)
        df['Value'] = pd.to_numeric(df['DataValue'].str.replace(',',''), errors='coerce')
        return df[['Date','LineDescription','Value']]

    raw_pct     = fetch_bea('T10101')
    raw_contrib = fetch_bea('T10102')

    fred  = Fred(api_key=FRED_API_KEY)
    codes = {
        'CPI':'CPIAUCSL','PCEPI':'PCEPI','WTI':'DCOILWTICO',
        'HH':'DHHNGSP','Rate':'DFEDTARU','Spread':'T10Y2Y',
        'UNRATE':'UNRATE','CIVPART':'CIVPART','PAYEMS':'PAYEMS','HPI':'CSUSHPINSA'
    }
    dfs = {}
    for name, code in codes.items():
        s = fred.get_series(code)
        df = pd.DataFrame(s, columns=[name])
        df.index = pd.to_datetime(df.index)
        dfs[name] = df

    # 추가 계산
    dfs['CPI']['YoY']     = dfs['CPI']['CPI'].pct_change(12)*100
    dfs['PCEPI']['YoY']   = dfs['PCEPI']['PCEPI'].pct_change(12)*100
    dfs['PAYEMS']['Diff'] = dfs['PAYEMS']['PAYEMS'].diff()
    for k in ['CPI','PCEPI','PAYEMS']:
        col = 'YoY' if k!='PAYEMS' else 'Diff'
        dfs[k].dropna(subset=[col], inplace=True)

    return raw_pct, raw_contrib, dfs

raw_pct, raw_contrib, dfs = init_data()

# 개별 프레임 분리
df_cpi, df_pcepi, df_wti, df_hh, df_rate, df_spread, df_unrate, df_civ, df_payems, df_hpi = (
    dfs['CPI'], dfs['PCEPI'], dfs['WTI'], dfs['HH'],
    dfs['Rate'], dfs['Spread'], dfs['UNRATE'],
    dfs['CIVPART'], dfs['PAYEMS'], dfs['HPI']
)
# File: app.py
# ---------------------------------------------------------------------------------
import requests
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
from fredapi import Fred
from requests.adapters import HTTPAdapter

# API 설정
BASE_URL     = "https://apps.bea.gov/api/data"
BEA_API_KEY  = "1A43CED6-707A-4E61-B475-A31AAB37AD01"
FRED_API_KEY = "fb40b5238c2c5ff6b281706029681f15"
Q_MAP        = {'Q1':'-03-31','Q2':'-06-30','Q3':'-09-30','Q4':'-12-31'}

# HTTP 세션 with 재시도
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=3))

def init_data():
    def fetch_bea(table):
        params = {
            'UserID': BEA_API_KEY,
            'method': 'GetData',
            'datasetname': 'NIPA',
            'TableName': table,
            'Frequency': 'Q',
            'Year': ','.join(str(y) for y in range(2000,2026)),
            'ResultFormat': 'JSON'
        }
        r = session.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get('BEAAPI',{}).get('Results',{}).get('Data',[])
        df = pd.DataFrame(data)
        if df.empty: 
            return df
        df['Date'] = pd.to_datetime(
            df['TimePeriod'].str[:4] + df['TimePeriod'].str[4:].map(Q_MAP)
        )
        df['LineDescription'] = df['LineDescription'].astype(str)
        df['Value'] = pd.to_numeric(df['DataValue'].str.replace(',',''), errors='coerce')
        return df[['Date','LineDescription','Value']]

    raw_pct     = fetch_bea('T10101')
    raw_contrib = fetch_bea('T10102')

    fred  = Fred(api_key=FRED_API_KEY)
    codes = {
        'CPI':'CPIAUCSL','PCEPI':'PCEPI','WTI':'DCOILWTICO',
        'HH':'DHHNGSP','Rate':'DFEDTARU','Spread':'T10Y2Y',
        'UNRATE':'UNRATE','CIVPART':'CIVPART','PAYEMS':'PAYEMS','HPI':'CSUSHPINSA'
    }
    dfs = {}
    for name, code in codes.items():
        s = fred.get_series(code)
        df = pd.DataFrame(s, columns=[name])
        df.index = pd.to_datetime(df.index)
        dfs[name] = df

    # 추가 계산
    dfs['CPI']['YoY']     = dfs['CPI']['CPI'].pct_change(12)*100
    dfs['PCEPI']['YoY']   = dfs['PCEPI']['PCEPI'].pct_change(12)*100
    dfs['PAYEMS']['Diff'] = dfs['PAYEMS']['PAYEMS'].diff()
    for k in ['CPI','PCEPI','PAYEMS']:
        col = 'YoY' if k!='PAYEMS' else 'Diff'
        dfs[k].dropna(subset=[col], inplace=True)

    return raw_pct, raw_contrib, dfs

raw_pct, raw_contrib, dfs = init_data()

# 개별 프레임 분리
df_cpi, df_pcepi, df_wti, df_hh, df_rate, df_spread, df_unrate, df_civ, df_payems, df_hpi = (
    dfs['CPI'], dfs['PCEPI'], dfs['WTI'], dfs['HH'],
    dfs['Rate'], dfs['Spread'], dfs['UNRATE'],
    dfs['CIVPART'], dfs['PAYEMS'], dfs['HPI']
)
# Part 2: BEA 전처리, 슬라이더 설정, Dash 앱 초기화

# BEA pivot
series_order = [
    'Gross domestic product','Personal consumption expenditures',
    'Gross private domestic investment','Government consumption expenditures and gross investment',
    'Exports','Imports'
]
display_labels = {
    'Gross domestic product':'GDP','Personal consumption expenditures':'소비',
    'Gross private domestic investment':'투자',
    'Government consumption expenditures and gross investment':'정부지출',
    'Exports':'수출','Imports':'수입'
}
palette = px.colors.qualitative.Plotly
colors  = {s:palette[i%len(palette)] for i,s in enumerate(series_order)}

def prepare_bea(df):
    if df.empty: return df
    sub = df[df['LineDescription'].isin(series_order)]
    grp = sub.groupby(['Date','LineDescription'], as_index=False)['Value'].last()
    grp['LineDescription'] = pd.Categorical(
        grp['LineDescription'], categories=series_order, ordered=True
    )
    return grp.pivot(index='Date', columns='LineDescription', values='Value').sort_index()

DF_PCT     = prepare_bea(raw_pct)
DF_CONTRIB = prepare_bea(raw_contrib)

# 슬라이더 연도 설정 & 기본값
years = {
    'cpi':  sorted({d.year for d in df_cpi.index}),
    'comm': sorted({d.year for d in df_wti.index}),
    'lab':  sorted({d.year for d in df_unrate.index}),
    'hpi':  sorted({d.year for d in df_hpi.index})
}
marks = lambda yrs: {y:str(y) for y in yrs if y%5==0}
def idx_default(idx, year=2021):
    return next((i for i,d in enumerate(idx) if d.year>=year), 0)

defaults = {
    'gdp': next(i for i,d in enumerate(DF_PCT.index) if d.year>=2021),
    'inf': idx_default(df_cpi.index),
    'comm': idx_default(df_wti.index),
    'lab':  idx_default(df_unrate.index),
    'hpi':  idx_default(df_hpi.index)
}

# Dash 앱 초기화
app    = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# 레이아웃 함수들 (2열 그리드)
def _gdp_layout():
    return html.Div([
        html.H2('GDP Dashboard', style={'textAlign':'center'}),
        dcc.RangeSlider(
            id='date-slider',
            min=0, max=len(DF_PCT)-1,
            value=[defaults['gdp'], len(DF_PCT)-1],
            marks={i:d.strftime('%Y') for i,d in enumerate(DF_PCT.index) if d.month==12},
            allowCross=False
        ),
        html.Div(id='gdp-container', className='flex-grid')
    ])

def _infl_layout():
    return html.Div([
        html.H2('Inflation YoY Change', style={'textAlign':'center'}),
        dcc.RangeSlider(
            id='inf-slider',
            min=years['cpi'][0], max=years['cpi'][-1],
            value=[years['cpi'][0], years['cpi'][-1]],
            marks=marks(years['cpi']), allowCross=False
        ),
        html.Div([
            html.Div(dcc.Graph(id='cpi-graph'), className='graph-card'),
            html.Div(dcc.Graph(id='pcepi-graph'), className='graph-card')
        ], className='flex-grid')
    ])

def _comm_layout():
    return html.Div([
        html.H2('Commodity Prices', style={'textAlign':'center'}),
        dcc.RangeSlider(
            id='comm-slider',
            min=years['comm'][0], max=years['comm'][-1],
            value=[years['comm'][0], years['comm'][-1]],
            marks=marks(years['comm']), allowCross=False
        ),
        html.Div([
            html.Div(dcc.Graph(id='wti-graph'), className='graph-card'),
            html.Div(dcc.Graph(id='hh-graph'), className='graph-card')
        ], className='flex-grid')
    ])

def _int_layout():
    return html.Div([
        html.H2('Interest Rate Indicators', style={'textAlign':'center'}),
        dcc.RangeSlider(
            id='int-slider',
            min=years['cpi'][0], max=years['cpi'][-1],
            value=[years['cpi'][0], years['cpi'][-1]],
            marks=marks(years['cpi']), allowCross=False
        ),
        html.Div([
            html.Div(dcc.Graph(id='rate-graph'), className='graph-card'),
            html.Div(dcc.Graph(id='spread-graph'), className='graph-card')
        ], className='flex-grid')
    ])

def _lab_layout():
    return html.Div([
        html.H2('Labor Market', style={'textAlign':'center'}),
        dcc.RangeSlider(
            id='lab-slider',
            min=years['lab'][0], max=years['lab'][-1],
            value=[years['lab'][0], years['lab'][-1]],
            marks=marks(years['lab']), allowCross=False
        ),
        html.Div([
            html.Div(dcc.Graph(id='unrate-graph'), className='graph-card'),
            html.Div(dcc.Graph(id='civpart-graph'), className='graph-card'),
            html.Div(dcc.Graph(id='payems-graph'), className='graph-card')
        ], className='flex-grid')
    ])

def _house_layout():
    return html.Div([
        html.H2('Housing Prices', style={'textAlign':'center'}),
        dcc.RangeSlider(
            id='house-slider',
            min=years['hpi'][0], max=years['hpi'][-1],
            value=[years['hpi'][0], years['hpi'][-1]],
            marks=marks(years['hpi']), allowCross=False
        ),
        html.Div([
            html.Div(dcc.Graph(id='hpi-graph'), className='graph-card')
        ], className='flex-grid')
    ])

# 앱 레이아웃 정의
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Button('☰', id='btn_toggle', n_clicks=0),
    html.Div([
        html.H3('Menu', style={'margin':'20px 0'}),
        dcc.Link('GDP','/gdp',style={'display':'block','padding':'8px'}),
        dcc.Link('Inflation','/inflation',style={'display':'block','padding':'8px'}),
        dcc.Link('Commodity','/commodity',style={'display':'block','padding':'8px'}),
        dcc.Link('Interest','/interest',style={'display':'block','padding':'8px'}),
        dcc.Link('Labor','/labor',style={'display':'block','padding':'8px'}),
        dcc.Link('House','/house',style={'display':'block','padding':'8px'})
    ], id='sidebar'),
    html.Div(id='page-content')
])
# Part 3: Callbacks & Server 실행

@app.callback(Output('page-content','children'),
              Input('url','pathname'))
def render_page(path):
    maps = {
        '/gdp':       _gdp_layout,
        '/inflation': _infl_layout,
        '/commodity': _comm_layout,
        '/interest':  _int_layout,
        '/labor':     _lab_layout,
        '/house':     _house_layout
    }
    return maps.get(path, _gdp_layout)()

@app.callback(Output('gdp-container','children'),
              [Input('url','pathname'), Input('date-slider','value')])
def update_gdp(path, slider):
    if path not in ['/', '/gdp']:
        raise PreventUpdate
    start, end = slider
    dates = DF_PCT.index[start:end+1]
    cards = []
    # GDP 차트
    val = DF_PCT['Gross domestic product'].iloc[end]
    fig = go.Figure(go.Scatter(
        x=dates, y=DF_PCT['Gross domestic product'].loc[dates],
        mode='lines', line_color=colors['Gross domestic product']
    ))
    fig.update_layout(
        template='plotly_white',
        title=f"GDP — {dates[-1].year%100}.{dates[-1].month:02d}: {val:.2f}%",
        xaxis_tickformat="'%y.%m", yaxis_title='%', margin={'t':50,'b':40}
    )
    cards.append(html.Div(dcc.Graph(figure=fig), className='graph-card'))

    # 기여도 차트
    opts = [
        {'label':f"{d.year} Q{(d.month-1)//3+1}", 'value':i}
        for i,d in enumerate(DF_CONTRIB.index)
    ]
    cards.append(html.Div([
        dcc.Dropdown(id='contrib-dd', options=opts, value=len(DF_CONTRIB)-1, style={'width':'120px'}),
        dcc.Graph(id='contrib-graph')
    ], className='graph-card'))

    # 나머지 시리즈들
    for s in series_order[1:]:
        ser = DF_PCT[s].loc[dates]
        fig2 = go.Figure(go.Scatter(x=dates, y=ser, mode='lines', line_color=colors[s]))
        fig2.update_layout(
            template='plotly_white',
            title=f"{display_labels[s]} — {ser.index[-1].year%100}.{ser.index[-1].month:02d}: {ser.iloc[-1]:.2f}%",
            xaxis_tickformat="'%y.%m", yaxis_title='%', margin={'t':50,'b':40}
        )
        cards.append(html.Div(dcc.Graph(figure=fig2), className='graph-card'))
    return cards

@app.callback(Output('contrib-graph','figure'),
              [Input('url','pathname'), Input('contrib-dd','value')])
def update_contrib(path, idx):
    if path not in ['/', '/gdp']:
        raise PreventUpdate
    date = DF_CONTRIB.index[idx]
    vals = DF_CONTRIB.iloc[idx]
    fig = go.Figure(go.Bar(
        x=[display_labels[s] for s in series_order],
        y=[vals[s] for s in series_order],
        marker_color=[colors[s] for s in series_order]
    ))
    fig.update_layout(
        template='plotly_white',
        title=f"Contributions — {date.year} Q{(date.month-1)//3+1}",
        yaxis_title='%', margin={'t':50,'b':40}
    )
    return fig

@app.callback([Output('cpi-graph','figure'), Output('pcepi-graph','figure')],
              Input('inf-slider','value'))
def update_inflation(rng):
    s,e = rng
    cdf = df_cpi[(df_cpi.index.year>=s)&(df_cpi.index.year<=e)]
    pdf = df_pcepi[(df_pcepi.index.year>=s)&(df_pcepi.index.year<=e)]
    fig1 = go.Figure(go.Scatter(x=cdf.index, y=cdf['YoY'], mode='lines'))
    fig2 = go.Figure(go.Scatter(x=pdf.index, y=pdf['YoY'], mode='lines'))
    fig1.update_layout(template='plotly_white', title='CPI YoY')
    fig2.update_layout(template='plotly_white', title='PCEPI YoY')
    return fig1, fig2

@app.callback([Output('wti-graph','figure'), Output('hh-graph','figure')],
              Input('comm-slider','value'))
def update_commodity(rng):
    s,e = rng
    wdf = df_wti[(df_wti.index.year>=s)&(df_wti.index.year<=e)]
    hdf = df_hh[(df_hh.index.year>=s)&(df_hh.index.year<=e)]
    fig1 = go.Figure(go.Scatter(x=wdf.index, y=wdf['WTI'], mode='lines'))
    fig2 = go.Figure(go.Scatter(x=hdf.index, y=hdf['HH'], mode='lines'))
    fig1.update_layout(template='plotly_white', title='WTI Oil Price')
    fig2.update_layout(template='plotly_white', title='Henry Hub Gas Price')
    return fig1, fig2

@app.callback([Output('rate-graph','figure'), Output('spread-graph','figure')],
              Input('int-slider','value'))
def update_interest(rng):
    s,e  = rng
    rdf  = df_rate[(df_rate.index.year>=s)&(df_rate.index.year<=e)]
    sdf  = df_spread[(df_spread.index.year>=s)&(df_spread.index.year<=e)]
    fig1 = go.Figure(go.Scatter(x=rdf.index, y=rdf['Rate'], mode='lines'))
    fig2 = go.Figure(go.Scatter(x=sdf.index, y=sdf['Spread'], mode='lines'))
    fig1.update_layout(template='plotly_white', title='Federal Funds Rate')
    fig2.update_layout(template='plotly_white', title='10Y-2Y Treasury Spread')
    return fig1, fig2

@app.callback([Output('unrate-graph','figure'), Output('civpart-graph','figure'), Output('payems-graph','figure')],
              Input('lab-slider','value'))
def update_labor(rng):
    s,e = rng
    ur  = df_unrate[(df_unrate.index.year>=s)&(df_unrate.index.year<=e)]
    cp  = df_civ[(df_civ.index.year>=s)&(df_civ.index.year<=e)]
    pe  = df_payems[(df_payems.index.year>=s)&(df_payems.index.year<=e)]
    fig1 = go.Figure(go.Scatter(x=ur.index, y=ur['UNRATE'], mode='lines'))
    fig2 = go.Figure(go.Scatter(x=cp.index, y=cp['CIVPART'], mode='lines'))
    fig3 = go.Figure(go.Scatter(x=pe.index, y=pe['Diff'], mode='lines'))
    fig1.update_layout(template='plotly_white', title='Unemployment Rate')
    fig2.update_layout(template='plotly_white', title='Civilian Participation')
    fig3.update_layout(template='plotly_white', title='Nonfarm Payroll Change')
    return fig1, fig2, fig3

@app.callback(Output('hpi-graph','figure'),
              Input('house-slider','value'))
def update_house(rng):
    s,e  = rng
    hdf  = df_hpi[(df_hpi.index.year>=s)&(df_hpi.index.year<=e)]
    fig  = go.Figure(go.Scatter(x=hdf.index, y=hdf['HPI'], mode='lines'))
    fig.update_layout(template='plotly_white', title='Case-Shiller House Price Index')
    return fig

@app.callback([Output('sidebar','style'), Output('page-content','style')],
              Input('btn_toggle','n_clicks'),
              State('sidebar','style'), State('page-content','style'))
def toggle_sidebar(n, s, c):
    if n and s.get('transform')!='translateX(-100%)':
        s['transform']      = 'translateX(-100%)'
        c['margin-left']    = '0'
    else:
        s['transform']      = 'translateX(0)'
        c['margin-left']    = '250px'
    return s, c

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)
