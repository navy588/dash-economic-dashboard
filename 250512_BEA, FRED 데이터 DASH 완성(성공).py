import requests
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
from fredapi import Fred
from requests.adapters import HTTPAdapter
from requests.exceptions import ChunkedEncodingError, RequestException

# -- 설정 ----------------------------------------------------------------------
BASE_URL     = "https://apps.bea.gov/api/data"
BEA_API_KEY  = "1A43CED6-707A-4E61-B475-A31AAB37AD01"
FRED_API_KEY = "fb40b5238c2c5ff6b281706029681f15"
Q_MAP        = {'Q1':'-03-31', 'Q2':'-06-30', 'Q3':'-09-30', 'Q4':'-12-31'}

# -- HTTP 세션 with 재시도 -----------------------------------------------------
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=3))

# -- 데이터 초기 로딩 ----------------------------------------------------------
def init_data():
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
        data = r.json()['BEAAPI']['Results']['Data']
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(
            df['TimePeriod'].str[:4] + df['TimePeriod'].str[4:].map(Q_MAP)
        )
        df['LineDescription'] = df['LineDescription'].astype(str)
        df['Value'] = pd.to_numeric(df['DataValue'].str.replace(',', ''), errors='coerce')
        return df[['Date','LineDescription','Value']]

    raw_pct     = fetch_bea('T10101')
    raw_contrib = fetch_bea('T10102')

    fred      = Fred(api_key=FRED_API_KEY)
    df_cpi    = pd.DataFrame(fred.get_series('CPIAUCSL'), columns=['CPI'])
    df_pcepi  = pd.DataFrame(fred.get_series('PCEPI'),  columns=['PCEPI'])
    df_wti    = pd.DataFrame(fred.get_series('DCOILWTICO'), columns=['WTI'])
    df_hh     = pd.DataFrame(fred.get_series('DHHNGSP'),    columns=['HH'])
    df_rate   = pd.DataFrame(fred.get_series('DFEDTARU'),   columns=['Rate'])
    df_spread = pd.DataFrame(fred.get_series('T10Y2Y'),     columns=['Spread'])
    df_unrate = pd.DataFrame(fred.get_series('UNRATE'),     columns=['UNRATE'])
    df_civ    = pd.DataFrame(fred.get_series('CIVPART'),    columns=['CIVPART'])
    df_payems = pd.DataFrame(fred.get_series('PAYEMS'),     columns=['PAYEMS'])
    df_hpi    = pd.DataFrame(fred.get_series('CSUSHPINSA'), columns=['HPI'])

    for df in [df_cpi, df_pcepi, df_wti, df_hh, df_rate,
               df_spread, df_unrate, df_civ, df_payems, df_hpi]:
        df.index = pd.to_datetime(df.index)

    # YoY, diff 계산
    df_cpi['CPI_YoY']     = df_cpi['CPI'].pct_change(12)*100
    df_pcepi['PCEPI_YoY'] = df_pcepi['PCEPI'].pct_change(12)*100
    df_cpi.dropna(subset=['CPI_YoY'], inplace=True)
    df_pcepi.dropna(subset=['PCEPI_YoY'], inplace=True)
    df_payems['PAY_DIFF'] = df_payems['PAYEMS'].diff()
    df_payems.dropna(subset=['PAY_DIFF'], inplace=True)
    for df in [df_wti, df_hh, df_rate, df_spread]:
        df.dropna(inplace=True)

    return (raw_pct, raw_contrib,
            df_cpi, df_pcepi,
            df_wti, df_hh,
            df_rate, df_spread,
            df_unrate, df_civ,
            df_payems, df_hpi)

(raw_pct, raw_contrib,
 df_cpi, df_pcepi,
 df_wti, df_hh,
 df_rate, df_spread,
 df_unrate, df_civ,
 df_payems, df_hpi) = init_data()

series_order = [
    'Gross domestic product',
    'Personal consumption expenditures',
    'Gross private domestic investment',
    'Government consumption expenditures and gross investment',
    'Exports','Imports'
]
display_labels = {
    'Gross domestic product': 'GDP',
    'Personal consumption expenditures': '소비',
    'Gross private domestic investment': '투자',
    'Government consumption expenditures and gross investment': '정부지출',
    'Exports': '수출', 'Imports': '수입'
}
palette = px.colors.qualitative.Plotly
colors = {s: palette[i%len(palette)] for i,s in enumerate(series_order)}

def prepare_bea(raw):
    df = raw[raw['LineDescription'].isin(series_order)].copy()
    df = df.groupby(['Date','LineDescription'], as_index=False)['Value'].last()
    df['LineDescription'] = pd.Categorical(df['LineDescription'], categories=series_order, ordered=True)
    return df.pivot(index='Date', columns='LineDescription', values='Value').sort_index()

df_pct     = prepare_bea(raw_pct)
df_contrib = prepare_bea(raw_contrib)

# 슬라이더 마크
marks = lambda years: {y: str(y) for y in years if y%5==0}

# RangeSlider 설정
years_cpi   = sorted({d.year for d in df_cpi.index if d.year>=2000})
years_comm  = sorted({d.year for d in df_wti.index if d.year>=2000})
years_lab   = sorted({d.year for d in df_unrate.index if d.year>=2000})
years_hpi   = sorted({d.year for d in df_hpi.index if d.year>=2000})

# 기본 시작
default = lambda yrs: 2021 if 2021 in yrs else yrs[0]

defaults = {
    'gdp': next(i for i,d in enumerate(df_pct.index) if d.year>=2021),
    'inf': default(years_cpi),
    'comm': default(years_comm),
    'lab': default(years_lab),
    'house': default(years_hpi)
}

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# 레이아웃
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Button('☰', id='btn_toggle', n_clicks=0,
                style={'position':'absolute','top':10,'left':10,'fontSize':24,'zIndex':2}),
    html.Div([
        html.H3('Menu', style={'textAlign':'center'}),
        *[dcc.Link(name, href=path, style={'display':'block','padding':10}) for name,path in [
            ('GDP','/gdp'),('Inflation','/inflation'),('Commodity Price','/commodity'),
            ('Interest Rate','/interest'),('Labor Market','/labor'),('House','/house')
        ]]
    ], id='sidebar', style={'width':'250px','paddingTop':'50px','position':'fixed','height':'100%','backgroundColor':'#f0f0f0'}),
    html.Div(id='page-content', style={'marginLeft':'250px','padding':20})
])

# 페이지 렌더링
@app.callback(Output('page-content','children'), Input('url','pathname'))
def render_page(path):
    if path in ['/', '/gdp']:
        return html.Div([
            html.H2('GDP Dashboard', style={'textAlign':'center'}),
            dcc.RangeSlider( id='date-slider', min=0, max=len(df_pct)-1,
                value=[defaults['gdp'], len(df_pct)-1],
                marks={i:d.strftime('%Y') for i,d in enumerate(df_pct.index) if d.month==12},
                allowCross=False, updatemode='mouseup'),
            html.Div(id='gdp-container', style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':20,'marginTop':20})
        ])
    if path=='/inflation':
        return html.Div([
            html.H2('Inflation: YoY Change', style={'textAlign':'center'}),
            dcc.RangeSlider(id='inf-slider', min=years_cpi[0], max=years_cpi[-1],
                value=[defaults['inf'], years_cpi[-1]],
                marks=marks(years_cpi), allowCross=False, updatemode='mouseup'),
            html.Div([ dcc.Graph(id='cpi-graph'), dcc.Graph(id='pcepi-graph') ],
                style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':20,'marginTop':20})
        ])
    if path=='/commodity':
        return html.Div([
            html.H2('Commodity Prices', style={'textAlign':'center'}),
            dcc.RangeSlider(id='comm-slider', min=years_comm[0], max=years_comm[-1],
                value=[defaults['comm'], years_comm[-1]],
                marks=marks(years_comm), allowCross=False, updatemode='mouseup'),
            html.Div([dcc.Graph(id='wti-graph'), dcc.Graph(id='hh-graph')],
                style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':20,'marginTop':20})
        ])
    if path=='/interest':
        return html.Div([
            html.H2('Interest Rate Indicators', style={'textAlign':'center'}),
            dcc.RangeSlider(id='int-slider', min=years_cpi[0], max=years_cpi[-1],
                value=[defaults['inf'], years_cpi[-1]],
                marks=marks(years_cpi), allowCross=False, updatemode='mouseup'),
            html.Div([dcc.Graph(id='rate-graph'), dcc.Graph(id='spread-graph')],
                style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':20,'marginTop':20})
        ])
    if path=='/labor':
        return html.Div([
            html.H2('Labor Market', style={'textAlign':'center'}),
            dcc.RangeSlider(id='lab-slider', min=years_lab[0], max=years_lab[-1],
                value=[defaults['lab'], years_lab[-1]],
                marks=marks(years_lab), allowCross=False, updatemode='mouseup'),
            html.Div([dcc.Graph(id='unrate-graph'), dcc.Graph(id='civpart-graph'), dcc.Graph(id='payems-graph')],
                style={'display':'grid','gridTemplateColumns':'1fr 1fr 1fr','gap':20,'marginTop':20})
        ])
    if path=='/house':
        return html.Div([
            html.H2('Case-Shiller House Price Index', style={'textAlign':'center'}),
            dcc.RangeSlider(id='house-slider', min=years_hpi[0], max=years_hpi[-1],
                value=[defaults['house'], years_hpi[-1]], marks=marks(years_hpi), allowCross=False, updatemode='mouseup'),
            dcc.Graph(id='hpi-graph', style={'marginTop':20})
        ])
    return render_page('/gdp')

# GDP 업데이트 콜백
@app.callback(Output('gdp-container','children'), [Input('url','pathname'), Input('date-slider','value')])
def update_gdp(path, slider):
    if path not in ['/', '/gdp']: raise PreventUpdate
    start,end = slider
    dates = df_pct.index[start:end+1]
    cards=[]
    # GDP
    ts,val=dates[-1], df_pct['Gross domestic product'].loc[dates[-1]]
    fig=go.Figure(go.Scatter(x=dates,y=df_pct['Gross domestic product'].loc[dates], mode='lines', line_color=colors['Gross domestic product'], hovertemplate='%{x|%Y.%m}: %{y:.2f}%'))
    fig.update_layout(template='plotly_white', title=f"GDP — {ts.year%100}.{ts.month}: {val:.2f}%", xaxis_tickformat="'%y.%m", yaxis_title='%', margin={'t':50,'b':40})
    cards.append(html.Div(dcc.Graph(figure=fig), style={'padding':10}))
    # Contributions dropdown
    opts=[{'label':f"{d.year} Q{((d.month-1)//3)+1}", 'value':i} for i,d in enumerate(df_contrib.index)]
    cards.append(html.Div([html.Div([html.H4('Contributions', style={'display':'inline-block'}), dcc.Dropdown(id='contrib-dd', options=opts, value=len(df_contrib)-1, style={'width':'150px','float':'right'})]), dcc.Graph(id='contrib-graph')], style={'padding':10}))
    # Others
    for s in series_order[1:]:
        ser=df_pct[s].loc[dates]
        ts2,val2=ser.index[-1],ser.iloc[-1]
        fig2=go.Figure(go.Scatter(x=dates,y=ser,mode='lines',line_color=colors[s],hovertemplate='%{x|%Y.%m}: %{y:.2f}%'))
        fig2.update_layout(template='plotly_white', title=f"{display_labels[s]} — {ts2.year%100}.{ts2.month}: {val2:.2f}%", xaxis_tickformat="'%y.%m", yaxis_title='%', margin={'t':50,'b':40})
        cards.append(html.Div(dcc.Graph(figure=fig2), style={'padding':10}))
    return cards

# 기여도 콜백
@app.callback(Output('contrib-graph','figure'), [Input('url','pathname'), Input('contrib-dd','value')])
def update_contrib(path, idx):
    if path not in ['/', '/gdp']: raise PreventUpdate
    date,quarter=df_contrib.index[idx],((df_contrib.index[idx].month-1)//3+1)
    vals=df_contrib.iloc[idx]
    fig=go.Figure(go.Bar(x=[display_labels[s] for s in series_order],y=[vals[s] for s in series_order], marker_color=[colors[s] for s in series_order], hovertemplate='%{x}: %{y:.2f}%'))
    fig.update_layout(template='plotly_white', title=f"Contributions — {date.year} Q{quarter}", yaxis_title='%', margin={'t':50,'b':40})
    return fig

# Inflation 콜백
@app.callback([Output('cpi-graph','figure'), Output('pcepi-graph','figure')], [Input('url','pathname'), Input('inf-slider','value')])
def update_inf(path, yr):
    if path!='/inflation': raise PreventUpdate
    s,e=yr
    cdf=df_cpi[(df_cpi.index.year>=s)&(df_cpi.index.year<=e)]; pdf=df_pcepi[(df_pcepi.index.year>=s)&(df_pcepi.index.year<=e)]
    ts_c,val_c=cdf.index[-1],cdf['CPI_YoY'].iloc[-1]
    fig_c=go.Figure(go.Scatter(x=cdf.index,y=cdf['CPI_YoY'],mode='lines', hovertemplate='%{x|%Y.%m}: %{y:.2f}%'))
    fig_c.update_layout(template='plotly_white', title=f"CPI YoY — {ts_c.year%100}.{ts_c.month}: {val_c:.2f}%", xaxis_tickformat="'%y.%m", yaxis_title='%', margin={'t':50,'b':40})
    ts_p,val_p=pdf.index[-1],pdf['PCEPI_YoY'].iloc[-1]
    fig_p=go.Figure(go.Scatter(x=pdf.index,y=pdf['PCEPI_YoY'],mode='lines', hovertemplate='%{x|%Y.%m}: %{y:.2f}%'))
    fig_p.update_layout(template='plotly_white', title=f"PCEPI YoY — {ts_p.year%100}.{ts_p.month}: {val_p:.2f}%", xaxis_tickformat="'%y.%m", yaxis_title='%', margin={'t':50,'b':40})
    return fig_c, fig_p

# Commodity 콜백
@app.callback([Output('wti-graph','figure'), Output('hh-graph','figure')], [Input('url','pathname'), Input('comm-slider','value')])
def update_comm(path, yr):
    if path!='/commodity': raise PreventUpdate
    s,e=yr
    wdf=df_wti[(df_wti.index.year>=s)&(df_wti.index.year<=e)]; hdf=df_hh[(df_hh.index.year>=s)&(df_hh.index.year<=e)]
    ts_w,val_w=wdf.index[-1],wdf['WTI'].iloc[-1]; ts_h,val_h=hdf.index[-1],hdf['HH'].iloc[-1]
    fig_w=go.Figure(go.Scatter(x=wdf.index,y=wdf['WTI'],mode='lines', hovertemplate='%{x|%Y.%m}: %{y:.2f} USD/bbl'))
    fig_w.update_layout(template='plotly_white', title=f"WTI Oil Price — {ts_w.year%100}.{ts_w.month}: {val_w:.2f} USD/bbl", xaxis_tickformat="'%y.%m", yaxis_title='USD/barrel', margin={'t':50,'b':40})
    fig_h=go.Figure(go.Scatter(x=hdf.index,y=hdf['HH'],mode='lines', hovertemplate='%{x|%Y.%m}: %{y:.2f} USD/MMbtu'))
    fig_h.update_layout(template='plotly_white', title=f"Henry Hub Gas Price — {ts_h.year%100}.{ts_h.month}: {val_h:.2f} USD/MMbtu", xaxis_tickformat="'%y.%m", yaxis_title='USD/MMbtu', margin={'t':50,'b':40})
    return fig_w, fig_h

# Interest 콜백
@app.callback([Output('rate-graph','figure'), Output('spread-graph','figure')], [Input('url','pathname'), Input('int-slider','value')])
def update_int(path, yr):
    if path!='/interest': raise PreventUpdate
    s,e=yr
    rdf=df_rate[(df_rate.index.year>=s)&(df_rate.index.year<=e)]; sdf=df_spread[(df_spread.index.year>=s)&(df_spread.index.year<=e)]
    ts_r,val_r=rdf.index[-1],rdf['Rate'].iloc[-1]; ts_s,val_s=sdf.index[-1],sdf['Spread'].iloc[-1]
    fig_r=go.Figure(go.Scatter(x=rdf.index,y=rdf['Rate'],mode='lines', hovertemplate='%{x|%Y.%m}: %{y:.2f}%'))
    fig_r.update_layout(template='plotly_white', title=f"Federal Funds Rate — {ts_r.year%100}.{ts_r.month}: {val_r:.2f}%", xaxis_tickformat="'%y.%m", yaxis_title='%', margin={'t':50,'b':40})
    fig_s=go.Figure(go.Scatter(x=sdf.index,y=sdf['Spread'],mode='lines', hovertemplate='%{x|%Y.%m}: %{y:.2f}%'))
    fig_s.update_layout(template='plotly_white', title=f"10Y-2Y Treasury Spread — {ts_s.year%100}.{ts_s.month}: {val_s:.2f}%", xaxis_tickformat="'%y.%m", yaxis_title='%', margin={'t':50,'b':40})
    return fig_r, fig_s

# Labor Market 콜백
@app.callback([Output('unrate-graph','figure'), Output('civpart-graph','figure'), Output('payems-graph','figure')], [Input('url','pathname'), Input('lab-slider','value')])
def update_labor(path, yr):
    if path!='/labor': raise PreventUpdate
    s,e=yr
    ur=df_unrate[(df_unrate.index.year>=s)&(df_unrate.index.year<=e)]; cp=df_civ[(df_civ.index.year>=s)&(df_civ.index.year<=e)]; pe=df_payems[(df_payems.index.year>=s)&(df_payems.index.year<=e)]
    ts_u,val_u=ur.index[-1],ur['UNRATE'].iloc[-1]
    fig_u=go.Figure(go.Scatter(x=ur.index,y=ur['UNRATE'],mode='lines', hovertemplate='%{x|%Y.%m}: %{y:.2f}%'))
    fig_u.update_layout(template='plotly_white', title=f"Unemployment Rate — {ts_u.year%100}.{ts_u.month}: {val_u:.2f}%", xaxis_tickformat="'%y.%m", yaxis_title='%', margin={'t':50,'b':40})
    ts_c2,val_c2=cp.index[-1],cp['CIVPART'].iloc[-1]
    fig_c2=go.Figure(go.Scatter(x=cp.index,y=cp['CIVPART'],mode='lines', hovertemplate='%{x|%Y.%m}: %{y:.2f}%'))
    fig_c2.update_layout(template='plotly_white', title=f"Labor Participation — {ts_c2.year%100}.{ts_c2.month}: {val_c2:.2f}%", xaxis_tickformat="'%y.%m", yaxis_title='%', margin={'t':50,'b':40})
    ts_p2,val_p2=pe.index[-1],pe['PAY_DIFF'].iloc[-1]
    fig_p2=go.Figure(go.Scatter(x=pe.index,y=pe['PAY_DIFF']/1000,mode='lines', hovertemplate='%{x|%Y.%m}: %{y:.2f}k'))
    fig_p2.update_layout(template='plotly_white', title=f"Nonfarm Payroll Change — {ts_p2.year%100}.{ts_p2.month}: {val_p2/1000:.2f}k", xaxis_tickformat="'%y.%m", yaxis_title='Thousands', margin={'t':50,'b':40})
    return fig_u, fig_c2, fig_p2

# House Price Index 콜백
@app.callback(Output('hpi-graph','figure'), [Input('url','pathname'), Input('house-slider','value')])
def update_house(path, yr):
    if path!='/house': raise PreventUpdate
    s,e=yr
    hp=df_hpi[(df_hpi.index.year>=s)&(df_hpi.index.year<=e)]
    ts_h,val_h=hp.index[-1],hp['HPI'].iloc[-1]
    fig_h=go.Figure(go.Scatter(x=hp.index,y=hp['HPI'],mode='lines', hovertemplate='%{x|%Y.%m}: %{y:.2f}'))
    fig_h.update_layout(template='plotly_white', title=f"Case-Shiller Index — {ts_h.year%100}.{ts_h.month}: {val_h:.2f}", xaxis_tickformat="'%y.%m", yaxis_title='Index (2000=100)', margin={'t':50,'b':40})
    return fig_h

# 사이드바 토글
@app.callback([Output('sidebar','style'), Output('page-content','style')], Input('btn_toggle','n_clicks'), [State('sidebar','style'), State('page-content','style')])
def toggle_sidebar(n, s, c):
    if n and s.get('width')=='250px': s['width']='0'; c['marginLeft']='0'
    else: s['width']='250px'; c['marginLeft']='250px'
    return s, c

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False, use_reloader=False)