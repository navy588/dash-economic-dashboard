import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
from fredapi import Fred

# ─── FRED API 설정 ───
FRED_API_KEY = "fb40b5238c2c5ff6b281706029681f15"
fred = Fred(api_key=FRED_API_KEY)

# ─── 데이터 로드 & YoY 계산 ───
cpi    = fred.get_series("CPIAUCSL").pct_change(12) * 100
pcepi  = fred.get_series("PCEPI").pct_change(12) * 100
hpi    = fred.get_series("CSUSHPINSA").pct_change(12) * 100

exp_terms = {
    "1년":  "EXPINF1YR",
    "2년":  "EXPINF2YR",
    "3년":  "EXPINF3YR",
    "5년":  "EXPINF5YR",
    "10년": "EXPINF10YR",
    "30년": "EXPINF30YR"
}
exp_df = pd.concat(
    {label: fred.get_series(code) for label, code in exp_terms.items()},
    axis=1
)

# ─── 2000년 1월 이후로 필터링 ───
df = pd.DataFrame({
    "CPI_YoY":   cpi,
    "PCEPI_YoY": pcepi,
    "HPI_YoY":   hpi
}).join(exp_df)
df = df[df.index >= "2000-01-01"].dropna(subset=["CPI_YoY", "PCEPI_YoY", "HPI_YoY"])

# ─── 슬라이더용 인덱스 & 5년 단위 마크(1월만) ───
idx = df.index
marks = {
    i: d.strftime("%Y")
    for i, d in enumerate(idx)
    if (d.year % 5 == 0 and d.month == 1)
}

# ─── Dash 앱 설정 ───
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("인플레이션", style={'textAlign':'center','marginBottom':'20px'}),
    dcc.RangeSlider(
        id="infl-slider",
        min=0, max=len(idx)-1,
        value=[0, len(idx)-1],
        marks=marks,
        allowCross=False
    ),
    html.Div([
        html.Div(dcc.Graph(id="cpi-graph"), style={'width':'48%','display':'inline-block','padding':'0 1%'}),
        html.Div(dcc.Graph(id="pce-graph"), style={'width':'48%','display':'inline-block','padding':'0 1%'})
    ], style={'marginTop':'20px'}),
    html.Div([
        html.Div(dcc.Graph(id="exp-graph"), style={'width':'48%','display':'inline-block','padding':'0 1%'}),
        html.Div(dcc.Graph(id="hpi-graph"), style={'width':'48%','display':'inline-block','padding':'0 1%'})
    ], style={'marginTop':'20px'})
], style={'maxWidth':'900px','margin':'0 auto','fontFamily':'Malgun Gothic'})

@app.callback(
    [
      Output("cpi-graph","figure"),
      Output("pce-graph","figure"),
      Output("exp-graph","figure"),
      Output("hpi-graph","figure")
    ],
    Input("infl-slider","value")
)
def update_inflation(rng):
    s, e = rng
    sub = df.iloc[s:e+1]

    # 소비자물가지수(CPI)
    fig_cpi = go.Figure(go.Scatter(x=sub.index, y=sub["CPI_YoY"], mode="lines"))
    fig_cpi.update_layout(template="plotly_white", title="소비자물가지수(CPI)")
    # 최신값
    ld = sub["CPI_YoY"].dropna().index[-1].strftime("%Y-%m-%d")
    lv = sub["CPI_YoY"].dropna().iloc[-1]
    fig_cpi.add_annotation(
        text=f"전년비, % | {ld} | {lv:.2f}",
        xref="paper", yref="paper",
        x=0, y=-0.15, showarrow=False,
        font=dict(size=10, color="#666")
    )

    # 개인소비지출(PCE) 물가지수
    fig_pce = go.Figure(go.Scatter(x=sub.index, y=sub["PCEPI_YoY"], mode="lines"))
    fig_pce.update_layout(template="plotly_white", title="개인소비지출(PCE) 물가지수")
    ld = sub["PCEPI_YoY"].dropna().index[-1].strftime("%Y-%m-%d")
    lv = sub["PCEPI_YoY"].dropna().iloc[-1]
    fig_pce.add_annotation(
        text=f"전년비, % | {ld} | {lv:.2f}",
        xref="paper", yref="paper",
        x=0, y=-0.15, showarrow=False,
        font=dict(size=10, color="#666")
    )

    # 기대인플레이션(미시간)
    fig_exp = go.Figure()
    for term in exp_terms:
        fig_exp.add_trace(go.Scatter(
            x=sub.index, y=sub[term], mode="lines", name=term
        ))
    fig_exp.update_layout(template="plotly_white", title="기대인플레이션(미시간)")
    # 최신값은 '1년' 기준
    ld = sub["1년"].dropna().index[-1].strftime("%Y-%m-%d")
    lv = sub["1년"].dropna().iloc[-1]
    fig_exp.add_annotation(
        text=f"% | {ld} | {lv:.2f}",
        xref="paper", yref="paper",
        x=0, y=-0.15, showarrow=False,
        font=dict(size=10, color="#666")
    )

    # 케이스-실러 주택가격지수
    fig_hpi = go.Figure(go.Scatter(x=sub.index, y=sub["HPI_YoY"], mode="lines"))
    fig_hpi.update_layout(template="plotly_white", title="케이스-실러 주택가격지수")
    ld = sub["HPI_YoY"].dropna().index[-1].strftime("%Y-%m-%d")
    lv = sub["HPI_YoY"].dropna().iloc[-1]
    fig_hpi.add_annotation(
        text=f"전년비, % | {ld} | {lv:.2f}",
        xref="paper", yref="paper",
        x=0, y=-0.15, showarrow=False,
        font=dict(size=10, color="#666")
    )

    return fig_cpi, fig_pce, fig_exp, fig_hpi

if __name__ == "__main__":
    app.run(debug=False)
