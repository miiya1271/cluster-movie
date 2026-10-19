import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 설정 (브라우저 탭 제목 및 아이콘)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

# 헤더 아이콘 및 앱 제목
st.title("🎬 영화 유형 나누기")

# 2. 데이터 불러오기 및 기본 처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

try:
    raw_df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 데 실패했습니다: {e}")
    st.stop()

total_raw_count = len(raw_df)

# 속성 생성을 위한 정제 (값 누락 및 first_week_audi가 0인 영화 제외)
df = raw_df.dropna(subset=["first_scrn", "total_audi", "days_in_top10", "first_week_audi", "movieNm"]).copy()
df = df[(df["first_week_audi"] > 0) & (df["first_scrn"] > 0) & (df["total_audi"] > 0)].copy()

# 파생 속성 계산
df["log_first_scrn"] = np.log10(df["first_scrn"])
df["log_total_audi"] = np.log10(df["total_audi"])
df["days_in_top10_feat"] = df["days_in_top10"]
df["long_run_idx"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)

filtered_count = len(df)

# 전체 편수와 묶은 편수를 한 줄로 표시
st.markdown(f"**전체 영화:** {total_raw_count:,}편 &nbsp;|&nbsp; **분석 대상 영화:** {filtered_count:,}편")
st.write("---")

# 3. 군집화 속성 선택 UI
feature_dict = {
    "스크린 수 (상용로그)": "log_first_scrn",
    "누적 관객 수 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10_feat",
    "롱런 지수": "long_run_idx"
}
feature_names = list(feature_dict.keys())

selected_feature_names = st.multiselect(
    "📌 군집화에 사용할 속성을 선택하세요 (2개 이상 선택 가능):",
    options=feature_names,
    default=feature_names
)

if len(selected_feature_names) < 2:
    st.warning("⚠️ 군집화를 수행하려면 속성을 최소 2개 이상 선택해야 합니다.")
    st.stop()

selected_feature_cols = [feature_dict[name] for name in selected_feature_names]

# 4. 표준화 및 K-평균 군집화 (k=3, 난수 고정)
X = df[selected_feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["temp_cluster"] = kmeans.fit_predict(X_scaled)

# 묶음 번호 재할당 (누적 관객 평균이 큰 순서대로 ㉮, ㉯, ㉰)
cluster_audi_means = df.groupby("temp_cluster")["total_audi"].mean()
sorted_clusters = cluster_audi_means.sort_values(ascending=False).index

label_map = {
    sorted_clusters[0]: "㉮",
    sorted_clusters[1]: "㉯",
    sorted_clusters[2]: "㉰"
}
df["cluster"] = df["temp_cluster"].map(label_map)
cluster_order = ["㉮", "㉯", "㉰"]

# 5. 2차원 산점도
st.subheader("📊 2차원 산점도")
col2d_x, col2d_y = st.columns(2)
with col2d_x:
    x_2d_name = st.selectbox("2D 가로축 속성", feature_names, index=0, key="2d_x")
with col2d_y:
    y_2d_name = st.selectbox("2D 세로축 속성", feature_names, index=1, key="2d_y")

fig_2d = px.scatter(
    df,
    x=feature_dict[x_2d_name],
    y=feature_dict[y_2d_name],
    color="cluster",
    category_orders={"cluster": cluster_order},
    hover_name="movieNm",
    labels={
        feature_dict[x_2d_name]: x_2d_name,
        feature_dict[y_2d_name]: y_2d_name,
        "cluster": "묶음"
    },
    title=f"{x_2d_name} vs {y_2d_name}"
)
st.plotly_chart(fig_2d, use_container_width=True)

# 6. 3차원 산점도
st.subheader("🧊 3차원 산점도")
if len(selected_feature_names) < 3:
    st.info("💡 3차원 산점도를 확인하려면 군집화 속성을 3개 이상 선택해 주세요.")
else:
    col3d_x, col3d_y, col3d_z = st.columns(3)
    with col3d_x:
        x_3d_name = st.selectbox("3D X축 속성", selected_feature_names, index=0, key="3d_x")
    with col3d_y:
        y_3d_name = st.selectbox("3D Y축 속성", selected_feature_names, index=1, key="3d_y")
    with col3d_z:
        z_3d_name = st.selectbox("3D Z축 속성", selected_feature_names, index=2, key="3d_z")

    fig_3d = px.scatter_3d(
        df,
        x=feature_dict[x_3d_name],
        y=feature_dict[y_3d_name],
        z=feature_dict[z_3d_name],
        color="cluster",
        category_orders={"cluster": cluster_order},
        hover_name="movieNm",
        labels={
            feature_dict[x_3d_name]: x_3d_name,
            feature_dict[y_3d_name]: y_3d_name,
            feature_dict[z_3d_name]: z_3d_name,
            "cluster": "묶음"
        },
        title=f"3D 산점도 ({x_3d_name} × {y_3d_name} × {z_3d_name})"
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

st.write("---")

# 7. 묶음별 요약 통계 표 (원래 단위 평균)
st.subheader("📋 묶음별 편수 및 속성 평균 (원래 단위)")

summary_df = df.groupby("cluster").agg(
    편수=("movieNm", "count"),
    스크린수_평균=("first_scrn", "mean"),
    누적관객수_평균=("total_audi", "mean"),
    _10위권일수_평균=("days_in_top10", "mean"),
    롱런지수_평균=("long_run_idx", "mean")
).reindex(cluster_order)

summary_df.columns = ["편수", "스크린 수", "누적 관객 수", "10위권 일수", "롱런 지수"]

st.dataframe(
    summary_df.style.format({
        "편수": "{:,.0f}",
        "스크린 수": "{:,.1f}",
        "누적 관객 수": "{:,.0f}",
        "10위권 일수": "{:,.1f}",
        "롱런 지수": "{:,.2f}"
    }),
    use_container_width=True
)

# 8. 묶음별 누적 관객 수 TOP 5 영화 목록
st.subheader("🏆 묶음별 누적 관객 수 TOP 5 영화")
top_cols = st.columns(3)

for idx, c_label in enumerate(cluster_order):
    with top_cols[idx]:
        st.markdown(f"#### 묶음 {c_label}")
        cluster_movies = (
            df[df["cluster"] == c_label]
            .sort_values(by="total_audi", ascending=False)
            .head(5)
        )
        for rank, (_, row) in enumerate(cluster_movies.iterrows(), start=1):
            st.write(f"**{rank}. {row['movieNm']}** ({row['total_audi']:,}명)")
