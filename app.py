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
import pickle

# -- 설정 ----------------------------------------------------------------------
BASE_URL     = "https://apps.bea.gov/api/data"
BEA_API_KEY  = "1A43CED6-707A-4E61-B475-A31AAB37AD01"
FRED_API_KEY = "fb40b5238c2c5ff6b281706029681f15"
Q_MAP        = {'Q1':'-03-31', 'Q2':'-06-30', 'Q3':'-09-30', 'Q4':'-12-31'}

# -- HTTP 세션 with 재시도 -----------------------------------------------------
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=3))

# -- 데이터 초기 로딩 함수 -----------------------------------------------------
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

# -- 캐시 로드 or init_data() 호출 --------------------------------------------------
try:
    with open("latest_data.pkl", "rb") as f:
        (raw_pct, raw_contrib,
         df_cpi, df_pcepi,
         df_wti, df_hh,
         df_rate, df_spread,
         df_unrate, df_civ,
         df_payems, df_hpi) = pickle.load(f)
    print("✅ 캐시 데이터 로드 완료")
except FileNotFoundError:
    (raw_pct, raw_contrib,
     df_cpi, df_pcepi,
     df_wti, df_hh,
     df_rate, df_spread,
     df_unrate, df_civ,
     df_payems, df_hpi) = init_data()
    print("⚠️ 캐시 없음 — init_data()로 초기 로드")

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

# 슬라이더 마크
marks = lambda years: {y: str(y) for y in years if y%5==0}

# RangeSlider 설정
# ... (기존 코드 계속 유지) ...

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False, use_reloader=False)
