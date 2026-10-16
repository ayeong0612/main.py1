
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

st.set_page_config(
    page_title="🎬 영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 흥행과 상영 특성을 분석해 "
    "비슷한 영화끼리 묶어 봅니다."
)

FEATURE_LABELS = {
    "log_first_scrn": "스크린 수 (상용로그)",
    "log_total_audi": "누적 관객 (상용로그)",
    "days_in_top10": "10위권 일수",
    "longrun_index": "롱런 지수"
}

FEATURES = list(FEATURE_LABELS.keys())

CLUSTER_SYMBOLS = [
    "㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"
]

CLUSTER_COLORS = [
    "#e76f51",
    "#2a9d8f",
    "#457b9d",
    "#e9c46a",
    "#9b5de5",
    "#f15bb5",
    "#6c757d"
]


@st.cache_data
def load_movies():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "movieCd",
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "필수 열이 없습니다: " + ", ".join(missing)
        )

    numeric_cols = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 롱런 지수 = 누적 관객 수 / 첫 주 관객 수
    # 첫 주 관객 수가 0 이하이면 계산하지 않음
    df["longrun_index"] = np.nan

    valid_first_week = df["first_week_audi"] > 0

    df.loc[valid_first_week, "longrun_index"] = (
        df.loc[valid_first_week, "total_audi"]
        / df.loc[valid_first_week, "first_week_audi"]
    )

    # 롱런 지수의 최댓값은 20
    df["longrun_index"] = df["longrun_index"].clip(upper=20)

    # 스크린 수와 누적 관객 수에 상용로그 적용
    df["log_first_scrn"] = np.nan
    df["log_total_audi"] = np.nan

    positive_screens = df["first_scrn"] > 0
    positive_audience = df["total_audi"] > 0

    df.loc[positive_screens, "log_first_scrn"] = np.log10(
        df.loc[positive_screens, "first_scrn"]
    )

    df.loc[positive_audience, "log_total_audi"] = np.log10(
        df.loc[positive_audience, "total_audi"]
    )

    return df


try:
    raw = load_movies()

except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.info("인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.stop()


total_count = len(raw)

# 분석에 사용할 모든 속성이 유효한 영화만 남김
valid = raw[
    raw["first_week_audi"].notna()
    & (raw["first_week_audi"] > 0)
    & raw[FEATURES].notna().all(axis=1)
    & (raw["first_scrn"] > 0)
    & (raw["total_audi"] > 0)
].copy()

st.caption(
    f"전체 영화 {total_count:,}편"
    f"  ·  분석 대상 영화 {len(valid):,}편"
)

if len(valid) < 7:
    st.error(
        "묶음 수 1~7개를 비교하려면 "
        "분석 가능한 영화가 최소 7편 필요합니다."
    )
    st.stop()


# --------------------------------------------------
# 1. 묶음 수와 분석 속성 선택
# --------------------------------------------------

st.subheader("1. 분석 조건 설정")

col_k, col_features = st.columns([1, 3])

with col_k:
    cluster_count = st.selectbox(
        "묶음 수",
        options=list(range(2, 8)),
        index=1
    )

with col_features:
    selected_features = st.multiselect(
        "묶음 분석에 사용할 속성 (2개 이상)",
        options=FEATURES,
        default=FEATURES,
        format_func=lambda x: FEATURE_LABELS[x]
    )

if len(selected_features) < 2:
    st.warning("분석 속성을 두 개 이상 선택해 주세요.")
    st.stop()

if cluster_count >= len(valid):
    st.error("묶음 수는 분석 대상 영화 수보다 작아야 합니다.")
    st.stop()


# 선택한 속성만 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    valid[selected_features].astype(float)
)


# --------------------------------------------------
# 2. 묶음 수별 WCSS 계산
# --------------------------------------------------

st.subheader("2. 적절한 묶음 수 살펴보기")

st.write(
    "묶음 수마다 각 영화와 해당 묶음 중심 사이의 "
    "제곱거리를 모두 더한 값(WCSS)을 계산합니다. "
    "값이 작을수록 각 묶음 안의 점들이 중심에 가깝습니다."
)

wcss_rows = []
max_k = min(7, len(valid) - 1)

for k in range(1, max_k + 1):
    elbow_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    elbow_model.fit(X_scaled)

    wcss_rows.append({
        "묶음 수": k,
        "WCSS": elbow_model.inertia_
    })

wcss_df = pd.DataFrame(wcss_rows)

fig_elbow = px.line(
    wcss_df,
    x="묶음 수",
    y="WCSS",
    markers=True,
    title="묶음 수에 따른 WCSS 변화"
)

fig_elbow.update_traces(
    line=dict(width=3),
    marker=dict(size=9)
)

fig_elbow.update_layout(
    xaxis=dict(
        tickmode="linear",
        dtick=1,
        title="묶음 수"
    ),
    yaxis_title="군집 내 제곱거리 합 (WCSS)",
    hovermode="x unified"
)

# 현재 선택한 묶음 수에 세로선 표시
fig_elbow.add_vline(
    x=cluster_count,
    line_dash="dash",
    line_width=2,
    line_color="red",
    annotation_text=f"현재 선택: {cluster_count}개",
    annotation_position="top right"
)

st.plotly_chart(fig_elbow, use_container_width=True)

# 직전 묶음 수와 비교
wcss_df["직전 값 대비 감소량"] = (
    wcss_df["WCSS"].shift(1) - wcss_df["WCSS"]
)

wcss_display = wcss_df.copy()

wcss_display["WCSS"] = wcss_display["WCSS"].map(
    lambda x: f"{x:,.2f}"
)

wcss_display["직전 값 대비 감소량"] = (
    wcss_display["직전 값 대비 감소량"].map(
        lambda x: "" if pd.isna(x) else f"{x:,.2f}"
    )
)

st.dataframe(
    wcss_display,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "WCSS는 묶음 수가 늘어날수록 대체로 감소합니다. "
    "그래프가 급격히 꺾인 뒤 완만해지는 지점은 "
    "묶음 수를 검토할 때 참고할 수 있습니다."
)


# --------------------------------------------------
# 3. 선택한 묶음 수로 최종 군집 분석
# --------------------------------------------------

model = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=10
)

valid["군집번호"] = model.fit_predict(X_scaled)


# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰... 배정
cluster_order = (
    valid.groupby("군집번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_labels = {
    cluster_id: CLUSTER_SYMBOLS[i]
    for i, cluster_id in enumerate(cluster_order)
}

valid["영화 유형"] = valid["군집번호"].map(cluster_labels)
valid["유형"] = valid["영화 유형"]

active_symbols = [
    cluster_labels[cluster_id]
    for cluster_id in cluster_order
]

color_map = {
    symbol: CLUSTER_COLORS[i]
    for i, symbol in enumerate(active_symbols)
}


# --------------------------------------------------
# 4. 실루엣 점수
# --------------------------------------------------

st.subheader("3. 현재 묶음 수의 실루엣 점수")

try:
    silhouette = silhouette_score(
        X_scaled,
        valid["군집번호"]
    )

    st.write(
        f"현재 묶음 수: **{cluster_count}개**"
        f"  ·  실루엣 점수: **{silhouette:.3f}**"
    )

    st.caption(
        "실루엣 점수는 -1에서 1 사이입니다. "
        "1에 가까울수록 같은 묶음 안의 영화는 서로 비슷하고, "
        "다른 묶음과는 잘 구분되는 경향이 있습니다. "
        "0 근처는 묶음 경계가 겹칠 수 있고, "
        "음수는 일부 영화가 다른 묶음에 더 가까울 수 있음을 뜻합니다."
    )

except Exception as e:
    st.warning(f"실루엣 점수를 계산하지 못했습니다: {e}")


# --------------------------------------------------
# 5. 2차원 산점도
# --------------------------------------------------

st.subheader("4. 2차원 산점도")

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
    y_default = min(1, len(selected_features) - 1)

    y_feature = st.selectbox(
        "세로축 속성",
        selected_features,
        index=y_default,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_y"
    )

fig2 = px.scatter(
    valid,
    x=x_feature,
    y=y_feature,
    color="유형",
    color_discrete_map=color_map,
    category_orders={"유형": active_symbols},
    hover_name="movieNm",
    hover_data={
        "영화 유형": True,
        "first_scrn": ":,",
        "total_audi": ":,",
        "days_in_top10": True,
        "longrun_index": ":.2f"
    },
    labels={
        **FEATURE_LABELS,
        "유형": "영화 유형",
        "movieNm": "영화 제목"
    },
    title=f"영화 유형별 2차원 분포 ({cluster_count}개 묶음)"
)

fig2.update_traces(
    marker=dict(size=8, opacity=0.78)
)

fig2.update_layout(legend_title_text="영화 유형")

st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# 6. 3차원 산점도
# --------------------------------------------------

st.subheader("5. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 분석 속성을 "
        "3개 이상 선택해 주세요."
    )

else:
    c3, c4, c5 = st.columns(3)

    with c3:
        x3 = st.selectbox(
            "X축 속성",
            selected_features,
            index=0,
            format_func=lambda x: FEATURE_LABELS[x],
            key="x3"
        )

    with c4:
        y3 = st.selectbox(
            "Y축 속성",
            selected_features,
            index=1,
            format_func=lambda x: FEATURE_LABELS[x],
            key="y3"
        )

    with c5:
        z3 = st.selectbox(
            "Z축 속성",
            selected_features,
            index=2,
            format_func=lambda x: FEATURE_LABELS[x],
            key="z3"
        )

    fig3 = px.scatter_3d(
        valid,
        x=x3,
        y=y3,
        z=z3,
        color="유형",
        color_discrete_map=color_map,
        category_orders={"유형": active_symbols},
        hover_name="movieNm",
        hover_data={
            "영화 유형": True,
            "first_scrn": ":,",
            "total_audi": ":,",
            "days_in_top10": True,
            "longrun_index": ":.2f"
        },
        labels={
            **FEATURE_LABELS,
            "유형": "영화 유형",
            "movieNm": "영화 제목"
        },
        title=f"영화 유형별 3차원 분포 ({cluster_count}개 묶음)"
    )

    fig3.update_traces(
        marker=dict(size=3, opacity=0.8)
    )

    fig3.update_layout(
        legend_title_text="영화 유형"
    )

    st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------
# 7. 유형별 편수와 속성 평균
# --------------------------------------------------

st.subheader("6. 유형별 편수와 속성 평균")

summary = valid.groupby("영화 유형").agg(
    편수=("movieNm", "size"),
    스크린수_평균=("first_scrn", "mean"),
    누적관객_평균=("total_audi", "mean"),
    십위권일수_평균=("days_in_top10", "mean"),
    롱런지수_평균=("longrun_index", "mean")
)

summary = summary.reindex(active_symbols).reset_index()

summary = summary.rename(columns={
    "영화 유형": "영화 유형",
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


# --------------------------------------------------
# 8. 유형별 누적 관객 상위 영화 5편
# --------------------------------------------------

st.subheader("7. 유형별 누적 관객 상위 영화 5편")

for label in active_symbols:
    group = (
        valid[valid["영화 유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with st.expander(
        f"{label} 유형 — 누적 관객 상위 {len(group)}편",
        expanded=True
    ):
        if group.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")

        else:
            top = group[
                [
                    "movieNm",
                    "total_audi",
                    "first_week_audi",
                    "first_scrn",
                    "days_in_top10",
                    "longrun_index"
                ]
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


# --------------------------------------------------
# 안내
# --------------------------------------------------

st.caption(
    "분석 방법: 선택한 속성을 StandardScaler로 표준화한 뒤 "
    f"KMeans(n_clusters={cluster_count}, random_state=42, n_init=10)로 "
    "군집화했습니다. 스크린 수와 누적 관객은 log10 변환했고, "
    "롱런 지수는 누적 관객 ÷ 첫 주 관객으로 계산해 "
    "최댓값을 20으로 제한했습니다. "
    "유형 기호는 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 순으로 배정됩니다."
)

