import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(page_title="🎬 영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")
st.write("영화의 흥행·상영 특성을 바탕으로 비슷한 영화끼리 세 유형으로 묶습니다.")

FEATURE_LABELS = {
    "log_first_scrn": "스크린 수 (상용로그)",
    "log_total_audi": "누적 관객 (상용로그)",
    "days_in_top10": "10위권 일수",
    "longrun_index": "롱런 지수"
}
FEATURES = list(FEATURE_LABELS.keys())


@st.cache_data
def load_movies():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "first_scrn", "first_week_audi",
        "total_audi", "days_in_top10"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing))

    for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["longrun_index"] = np.where(
        df["first_week_audi"] > 0,
        df["total_audi"] / df["first_week_audi"],
        np.nan
    )
    df["longrun_index"] = df["longrun_index"].clip(upper=20)

    df["log_first_scrn"] = np.where(
        df["first_scrn"] > 0, np.log10(df["first_scrn"]), np.nan
    )
    df["log_total_audi"] = np.where(
        df["total_audi"] > 0, np.log10(df["total_audi"]), np.nan
    )
    return df


try:
    raw = load_movies()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.info("인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.stop()

total_count = len(raw)

valid = raw[
    raw["first_week_audi"].notna()
    & (raw["first_week_audi"] > 0)
    & raw[FEATURES].notna().all(axis=1)
    & (raw["first_scrn"] > 0)
    & (raw["total_audi"] > 0)
].copy()

st.caption(f"전체 영화 {total_count:,}편 · 분석에 묶은 영화 {len(valid):,}편")

if len(valid) < 3:
    st.error("세 묶음을 만들 데이터가 3편 미만입니다.")
    st.stop()

st.subheader("1. 묶는 데 사용할 속성 선택")
selected_features = st.multiselect(
    "분석에 사용할 속성을 두 개 이상 고르세요.",
    options=FEATURES,
    default=FEATURES,
    format_func=lambda x: FEATURE_LABELS[x]
)

if len(selected_features) < 2:
    st.warning("군집 분석을 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(valid[selected_features].astype(float))

model = KMeans(n_clusters=3, random_state=42, n_init=10)
valid["군집번호"] = model.fit_predict(X_scaled)

cluster_order = (
    valid.groupby("군집번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_labels = {
    cluster_id: ["㉮", "㉯", "㉰"][i]
    for i, cluster_id in enumerate(cluster_order)
}
valid["영화 유형"] = valid["군집번호"].map(cluster_labels)
valid["유형"] = valid["영화 유형"]

cluster_colors = {
    "㉮": "#e76f51",
    "㉯": "#2a9d8f",
    "㉰": "#457b9d"
}

st.subheader("2. 2차원 산점도")
c1, c2 = st.columns(2)

with c1:
    x_feature = st.selectbox(
        "가로축 속성",
        selected_features,
        index=0,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_x"
    )

with c2:
    y_feature = st.selectbox(
        "세로축 속성",
        selected_features,
        index=1,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_y"
    )

fig2 = px.scatter(
    valid,
    x=x_feature,
    y=y_feature,
    color="유형",
    color_discrete_map=cluster_colors,
    hover_name="movieNm",
    hover_data={
        "영화 유형": True,
        "first_scrn": ":,",
        "total_audi": ":,",
        "days_in_top10": True
    },
    labels={**FEATURE_LABELS, "유형": "영화 유형"},
    title="영화 유형별 2차원 분포"
)
fig2.update_traces(marker=dict(size=8, opacity=0.78))
fig2.update_layout(legend_title_text="영화 유형")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("선택한 속성이 3개보다 적어 3차원 산점도를 표시할 수 없습니다. 위에서 속성을 3개 이상 선택해 주세요.")
else:
    c3, c4, c5 = st.columns(3)

    with c3:
        x3 = st.selectbox(
            "X축 속성", selected_features, index=0,
            format_func=lambda x: FEATURE_LABELS[x], key="x3"
        )
    with c4:
        y3 = st.selectbox(
            "Y축 속성", selected_features, index=1,
            format_func=lambda x: FEATURE_LABELS[x], key="y3"
        )
    with c5:
        z3 = st.selectbox(
            "Z축 속성", selected_features, index=2,
            format_func=lambda x: FEATURE_LABELS[x], key="z3"
        )

    fig3 = px.scatter_3d(
        valid,
        x=x3, y=y3, z=z3,
        color="유형",
        color_discrete_map=cluster_colors,
        hover_name="movieNm",
        hover_data={
            "영화 유형": True,
            "first_scrn": ":,",
            "total_audi": ":,",
            "days_in_top10": True
        },
        labels={**FEATURE_LABELS, "유형": "영화 유형"},
        title="영화 유형별 3차원 분포"
    )
    fig3.update_traces(marker=dict(size=3, opacity=0.8))
    fig3.update_layout(legend_title_text="영화 유형")
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. 유형별 편수와 속성 평균")
summary = valid.groupby("영화 유형").agg(
    편수=("movieNm", "size"),
    스크린수_평균=("first_scrn", "mean"),
    누적관객_평균=("total_audi", "mean"),
    십위권일수_평균=("days_in_top10", "mean"),
    롱런지수_평균=("longrun_index", "mean")
).reindex(["㉮", "㉯", "㉰"]).reset_index()

summary = summary.rename(columns={
    "스크린수_평균": "스크린 수 평균",
    "누적관객_평균": "누적 관객 평균",
    "십위권일수_평균": "10위권 일수 평균",
    "롱런지수_평균": "롱런 지수 평균"
})

st.dataframe(
    summary.style.format({
        "편수": "{:,.0f}",
        "스크린 수 평균": "{:,.1f}",
        "누적 관객 평균": "{:,.0f}",
        "10위권 일수 평균": "{:.2f}",
        "롱런 지수 평균": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.subheader("5. 유형별 누적 관객 상위 영화 5편")
for label in ["㉮", "㉯", "㉰"]:
    group = valid[valid["영화 유형"] == label].sort_values(
        "total_audi", ascending=False
    ).head(5)

    with st.expander(f"{label} 유형 — 누적 관객 상위 {len(group)}편", expanded=True):
        if group.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            top = group[
                ["movieNm", "total_audi", "first_week_audi",
                 "first_scrn", "days_in_top10", "longrun_index"]
            ].copy()

            top = top.rename(columns={
                "movieNm": "영화 제목",
                "total_audi": "누적 관객",
                "first_week_audi": "첫 주 관객",
                "first_scrn": "스크린 수",
                "days_in_top10": "10위권 일수",
                "longrun_index": "롱런 지수"
            })

            st.dataframe(
                top.style.format({
                    "누적 관객": "{:,.0f}",
                    "첫 주 관객": "{:,.0f}",
                    "스크린 수": "{:,.0f}",
                    "10위권 일수": "{:,.0f}",
                    "롱런 지수": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

st.caption(
    "선택한 속성을 StandardScaler로 표준화하고 "
    "KMeans(n_clusters=3, random_state=42, n_init=10)로 군집화했습니다. "
    "스크린 수와 누적 관객은 log10 변환했으며, 롱런 지수는 "
    "누적 관객 ÷ 첫 주 관객으로 계산하고 20을 상한으로 적용했습니다."
)
