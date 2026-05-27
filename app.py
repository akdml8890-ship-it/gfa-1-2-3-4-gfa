import io
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


APP_TITLE = "GFA 소재 성과 진단 & 제작 자동화 툴"

REQUIRED_COLUMNS = ["비용"]
RECOMMENDED_DIMENSION_COLUMNS = ["날짜", "캠페인명", "광고그룹명", "소재명"]
OPTIONAL_PERFORMANCE_COLUMNS = ["전환수", "전환매출"]
PERFORMANCE_SIGNAL_COLUMNS = ["노출수", "클릭수", "전환수", "전환매출", "ROAS"]

BASE_NUMERIC_COLUMNS = ["노출수", "클릭수", "비용", "전환수", "전환매출"]
METRIC_COLUMNS = ["CTR", "CPC", "CVR", "CPA", "ROAS"]

COLUMN_ALIASES = {
    "날짜": ["일자", "기간", "일", "기준일", "보고일", "집행일"],
    "캠페인명": ["캠페인", "캠페인 명", "캠페인 이름", "캠페인명"],
    "광고그룹명": ["광고그룹", "광고 그룹", "광고그룹 명", "광고 그룹명", "광고그룹 이름", "광고 그룹 이름"],
    "소재명": [
        "소재",
        "소재 명",
        "소재 이름",
        "광고소재",
        "광고 소재",
        "광고소재명",
        "광고 소재명",
        "광고소재 이름",
        "광고 소재 이름",
        "애셋",
        "애셋명",
        "애셋 이름",
        "에셋",
        "에셋명",
        "에셋 이름",
    ],
    "노출수": ["노출", "노출 수", "노출수(회)", "노출(회)", "총노출수", "총 노출수", "impression", "impressions"],
    "클릭수": ["클릭", "클릭 수", "클릭수(회)", "클릭(회)", "총클릭수", "총 클릭수", "click", "clicks"],
    "비용": [
        "광고비",
        "광고 비용",
        "비용(VAT포함)",
        "비용(VAT 포함)",
        "소진액",
        "소진 비용",
        "집행금액",
        "집행 금액",
        "총비용",
        "총 비용",
        "cost",
    ],
    "전환수": [
        "구매완료 수",
        "구매완료수",
        "구매 완료 수",
        "구매 완료수",
        "전환",
        "전환 수",
        "전환수(회)",
        "전환(회)",
        "전환 건수",
        "총전환수",
        "총 전환수",
        "conversion",
        "conversions",
    ],
    "전환매출": [
        "구매완료 전환매출액",
        "구매완료전환매출액",
        "구매 완료 전환매출액",
        "구매 완료 전환 매출액",
        "전환 매출",
        "전환매출액",
        "전환 매출액",
        "매출",
        "매출액",
        "총전환매출",
        "총 전환매출",
        "구매금액",
        "구매 금액",
        "conversionrevenue",
    ],
    "CTR": ["클릭률", "클릭율", "ctr(%)"],
    "CPC": ["평균클릭비용", "평균 클릭 비용", "평균 CPC", "cpc"],
    "CVR": ["전환율", "전환률", "cvr(%)"],
    "CPA": ["전환당비용", "전환당 비용", "cpa"],
    "ROAS": ["roas(%)", "광고수익률", "구매완료 광고수익률(%)", "구매완료광고수익률"],
    "소재 이미지 URL": ["이미지 URL", "소재 이미지", "광고 소재 이미지 URL"],
    "랜딩 URL": ["랜딩URL", "연결 URL", "연결URL", "랜딩 페이지", "랜딩페이지"],
    "상품명": ["상품", "제품명", "제품"],
    "타겟명": ["타겟", "오디언스", "오디언스명", "기기", "디바이스", "device"],
    "지면명": ["지면", "게재 위치", "게재위치", "매체 그룹", "매체그룹", "위치명"],
    "소재 유형": ["소재유형", "광고 소재 유형", "광고소재유형", "형식"],
    "카피 문구": ["카피", "문구", "광고 문구", "소재 문구"],
    "혜택 문구": ["혜택", "혜택문구"],
    "프로모션명": ["프로모션", "프로모션 명"],
}

MATCH_EXCLUDE_KEYWORDS = {
    "노출수": ["노출률", "노출율", "노출당"],
    "클릭수": ["클릭률", "클릭율", "ctr", "클릭당", "평균클릭"],
    "비용": ["cpc", "cpa", "전환당", "클릭당", "평균"],
    "전환수": ["전환률", "전환율", "cvr", "전환당", "매출", "매출액", "금액", "수익률", "roas"],
    "전환매출": ["roas", "수익률"],
}

OPTIONAL_COLUMNS = [
    "소재 이미지 URL",
    "랜딩 URL",
    "상품명",
    "타겟명",
    "지면명",
    "소재 유형",
    "카피 문구",
    "혜택 문구",
    "프로모션명",
]

TEXT_COLUMNS_FOR_INSIGHT = ["카피 문구", "혜택 문구", "프로모션명", "상품명", "타겟명", "소재 유형"]

MIN_IMPRESSIONS = 1000
MIN_CLICKS = 20

RISKY_REPLACEMENTS = {
    "치료": "관리",
    "완치": "꾸준한 관리",
    "예방": "일상 관리",
    "혈압이 내려간다": "건강 관리가 필요할 때",
    "콜레스테롤이 개선된다": "건강 지표를 챙기고 싶을 때",
    "관절 통증이 사라진다": "움직임이 신경 쓰이는 순간",
    "무조건 효과": "부담 없이 시작",
    "100% 효과": "꾸준히 챙기는 습관",
}


st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .small-muted { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def sanitize_ad_text(text: str) -> str:
    safe = str(text)
    for risky, replacement in RISKY_REPLACEMENTS.items():
        safe = safe.replace(risky, replacement)
    return safe


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def normalize_column_key(value: str) -> str:
    text = str(value).replace("\ufeff", "").replace("\u200b", "").strip().lower()
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", text)


def make_unique_columns(columns) -> list[str]:
    seen = {}
    unique_columns = []
    for idx, col in enumerate(columns):
        name = str(col).strip()
        if name == "" or name.lower() in ["nan", "none"]:
            name = f"빈 컬럼 {idx + 1}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique_columns.append(name if count == 0 else f"{name}_{count + 1}")
    return unique_columns


def is_alias_match(canonical: str, column_key: str, candidate_key: str) -> bool:
    if not column_key or not candidate_key:
        return False

    exclude_keywords = [normalize_column_key(keyword) for keyword in MATCH_EXCLUDE_KEYWORDS.get(canonical, [])]
    if any(keyword and keyword in column_key for keyword in exclude_keywords):
        return False

    if column_key == candidate_key:
        return True

    if column_key.startswith(candidate_key) or candidate_key in column_key:
        return len(candidate_key) >= 2

    return False


def detect_column_mapping(df: pd.DataFrame) -> dict[str, str]:
    normalized_columns = {}
    for col in df.columns:
        normalized_columns.setdefault(normalize_column_key(col), col)

    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        candidates = [canonical] + aliases
        for candidate in candidates:
            candidate_key = normalize_column_key(candidate)
            matched_col = normalized_columns.get(candidate_key)
            if matched_col is not None:
                mapping[canonical] = matched_col
                break
            for column_key, original_col in normalized_columns.items():
                if is_alias_match(canonical, column_key, candidate_key):
                    mapping[canonical] = original_col
                    break
            if canonical in mapping:
                break
    return mapping


def header_score(values) -> int:
    temp_columns = make_unique_columns(values)
    temp_df = pd.DataFrame(columns=temp_columns)
    mapping = detect_column_mapping(temp_df)
    score = sum(3 for col in REQUIRED_COLUMNS if col in mapping)
    score += sum(1 for col in RECOMMENDED_DIMENSION_COLUMNS + OPTIONAL_PERFORMANCE_COLUMNS if col in mapping)
    return score


def promote_detected_header(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    search_limit = min(len(df), 30)
    best_row = None
    best_score = header_score(df.columns)

    for row_idx in range(search_limit):
        score = header_score(df.iloc[row_idx].tolist())
        if score > best_score:
            best_score = score
            best_row = row_idx

    if best_row is None or best_score < 3:
        return df

    promoted = df.iloc[best_row + 1 :].copy()
    promoted.columns = make_unique_columns(df.iloc[best_row].tolist())
    promoted = promoted.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return promoted.reset_index(drop=True)


def apply_column_aliases(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    cleaned = clean_column_names(df)
    mapping = detect_column_mapping(cleaned)
    normalized = cleaned.copy()

    for canonical, source_col in mapping.items():
        if canonical not in normalized.columns:
            normalized[canonical] = cleaned[source_col]

    return normalized, mapping


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
        .str.replace(",", "", regex=False)
        .str.replace("₩", "", regex=False)
        .str.replace("원", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def safe_ratio(numerator, denominator, multiplier: float = 1.0) -> np.ndarray:
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0).astype(float)
    denominator = pd.to_numeric(denominator, errors="coerce").fillna(0).astype(float)
    result = np.zeros(len(numerator), dtype=float)
    denom_values = denominator.to_numpy()
    np.divide(
        numerator.to_numpy(),
        denom_values,
        out=result,
        where=denom_values != 0,
    )
    return result * multiplier


def safe_scalar_ratio(numerator: float, denominator: float, multiplier: float = 1.0) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator) * multiplier


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".csv"):
        last_error = None
        for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
            try:
                normal_df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, on_bad_lines="skip")
                raw_df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, header=None, on_bad_lines="skip")
                promoted_df = promote_detected_header(raw_df)
                if header_score(promoted_df.columns) > header_score(normal_df.columns):
                    return promoted_df
                return promote_detected_header(normal_df)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(f"CSV 인코딩을 읽을 수 없습니다. 파일 저장 형식을 확인해 주시기 바랍니다. ({last_error})")

    if file_name.endswith(".xlsx"):
        normal_df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        raw_df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", header=None)
        promoted_df = promote_detected_header(raw_df)
        if header_score(promoted_df.columns) > header_score(normal_df.columns):
            return promoted_df
        return promote_detected_header(normal_df)

    raise ValueError("CSV 또는 XLSX 파일만 업로드할 수 있습니다.")


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def build_missing_column_message(missing_columns: list[str]) -> str:
    guide_parts = []
    for column in missing_columns:
        aliases = ", ".join(COLUMN_ALIASES.get(column, [])[:6])
        guide_parts.append(f"{column}({aliases})" if aliases else column)
    return "필수 성과 컬럼을 찾지 못했습니다. 확인이 필요한 컬럼은 다음과 같습니다: " + " / ".join(guide_parts)


def first_non_empty(series: pd.Series) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[~values.isin(["", "nan", "None"])]
    return values.iloc[0] if not values.empty else ""


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["CTR"] = safe_ratio(result["클릭수"], result["노출수"], 100)
    result["CPC"] = safe_ratio(result["비용"], result["클릭수"])
    result["CVR"] = safe_ratio(result["전환수"], result["클릭수"], 100)
    result["CPA"] = safe_ratio(result["비용"], result["전환수"])
    result["ROAS"] = safe_ratio(result["전환매출"], result["비용"], 100)
    return result


def prepare_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    prepared = clean_column_names(df)
    for col in BASE_NUMERIC_COLUMNS:
        if col not in prepared.columns:
            prepared[col] = 0

    for col in BASE_NUMERIC_COLUMNS:
        prepared[col] = clean_numeric_series(prepared[col]).fillna(0)

    for col in METRIC_COLUMNS:
        if col in prepared.columns:
            prepared[col] = clean_numeric_series(prepared[col])

    default_values = {
        "날짜": "전체",
        "캠페인명": "캠페인 미제공",
        "광고그룹명": "광고그룹 미제공",
        "소재명": "소재명 미제공",
    }
    for col, default_value in default_values.items():
        if col not in prepared.columns:
            if col == "소재명":
                if "캠페인명" in prepared.columns and "타겟명" in prepared.columns:
                    prepared[col] = prepared["캠페인명"].astype(str) + " / " + prepared["타겟명"].astype(str)
                elif "캠페인명" in prepared.columns:
                    prepared[col] = prepared["캠페인명"].astype(str)
                else:
                    prepared[col] = [f"성과 항목 {idx + 1}" for idx in range(len(prepared))]
            else:
                prepared[col] = default_value

    for col in RECOMMENDED_DIMENSION_COLUMNS:
        prepared[col] = (
            prepared[col]
            .fillna(default_values.get(col, "미입력"))
            .astype(str)
            .str.strip()
            .replace({"": default_values.get(col, "미입력"), "nan": default_values.get(col, "미입력"), "None": default_values.get(col, "미입력")})
        )

    for col in OPTIONAL_COLUMNS:
        if col in prepared.columns:
            prepared[col] = prepared[col].fillna("").astype(str).str.strip()

    return calculate_metrics(prepared)


def aggregate_creatives(df: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["소재명", "캠페인명", "광고그룹명"]
    agg_dict = {col: "sum" for col in BASE_NUMERIC_COLUMNS}

    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            agg_dict[col] = first_non_empty

    grouped = df.groupby(group_keys, as_index=False, dropna=False).agg(agg_dict)
    return calculate_metrics(grouped)


def build_summary(df: pd.DataFrame) -> dict:
    total_impressions = float(df["노출수"].sum())
    total_clicks = float(df["클릭수"].sum())
    total_cost = float(df["비용"].sum())
    total_conversions = float(df["전환수"].sum())
    total_revenue = float(df["전환매출"].sum())

    return {
        "총 비용": total_cost,
        "총 매출": total_revenue,
        "전체 ROAS": safe_scalar_ratio(total_revenue, total_cost, 100),
        "전체 노출수": total_impressions,
        "전체 클릭수": total_clicks,
        "전체 CTR": safe_scalar_ratio(total_clicks, total_impressions, 100),
        "전체 전환수": total_conversions,
        "전체 CVR": safe_scalar_ratio(total_conversions, total_clicks, 100),
        "평균 CPC": safe_scalar_ratio(total_cost, total_clicks),
        "평균 CPA": safe_scalar_ratio(total_cost, total_conversions),
    }


def diagnose_row(row: pd.Series, benchmarks: dict) -> tuple[str, str, str]:
    avg_roas = benchmarks["전체 ROAS"]
    avg_ctr = benchmarks["전체 CTR"]
    has_delivery_data = benchmarks["전체 노출수"] > 0 and benchmarks["전체 클릭수"] > 0

    if not has_delivery_data:
        if row["비용"] <= 0:
            return (
                "판단 보류",
                "비용 또는 전환 데이터를 추가 확보한 뒤 판단합니다. 단기 테스트 상태로 유지합니다.",
                "노출수와 클릭수 데이터가 없는 리포트이며 비용도 없어 현재 기준으로는 성과 판단이 어렵습니다.",
            )

        if row["ROAS"] > avg_roas and row["전환수"] >= 1:
            return (
                "확대 추천",
                "예산 확대를 검토합니다. 동일 캠페인과 타겟 조건에서 유사 소재를 추가 테스트합니다. 고효율 조건에 우선 배치합니다.",
                f"노출수와 클릭수 데이터는 없지만 ROAS가 {row['ROAS']:.1f}%로 전체 평균 {avg_roas:.1f}%보다 높고 전환도 발생했습니다.",
            )

        if row["ROAS"] >= avg_roas and (row["전환수"] >= 1 or row["전환매출"] > 0):
            return (
                "유지",
                "현재 운영을 유지합니다. 소재 단위 리포트를 추가 확보한 뒤 카피와 비주얼 개선 테스트를 진행합니다.",
                "노출수와 클릭수 데이터는 없지만 ROAS가 전체 평균 이상으로 확인됩니다.",
            )

        if row["전환수"] >= 1 or row["전환매출"] > 0:
            return (
                "개선 필요",
                "전환은 발생하지만 효율이 낮습니다. 혜택, 상품 매칭, 예산 배분을 점검합니다.",
                "전환은 발생했지만 ROAS가 평균보다 낮아 효율 개선이 필요합니다.",
            )

        return (
            "교체 필요",
            "비용 사용 대비 전환 성과가 부족합니다. 캠페인, 타겟, 소재 방향을 재점검합니다.",
            "비용은 사용됐지만 전환 성과가 확인되지 않아 교체 또는 예산 축소 검토가 필요합니다.",
        )

    if row["노출수"] < MIN_IMPRESSIONS or row["클릭수"] < MIN_CLICKS:
        return (
            "판단 보류",
            "데이터를 추가 확보한 뒤 판단합니다. 단기 테스트 상태로 유지합니다.",
            f"노출 {row['노출수']:,.0f}회, 클릭 {row['클릭수']:,.0f}회로 판단 기준보다 모수가 적습니다.",
        )

    if row["ROAS"] > avg_roas and row["전환수"] >= 1 and row["CTR"] >= avg_ctr:
        return (
            "확대 추천",
            "예산 확대를 검토합니다. 동일 소구의 유사 소재를 추가 제작합니다. 고효율 타겟에 우선 배치합니다.",
            f"ROAS {row['ROAS']:.1f}%로 전체 {avg_roas:.1f}%보다 높고 CTR도 평균 이상입니다.",
        )

    if row["ROAS"] >= avg_roas:
        return (
            "유지",
            "현재 운영을 유지합니다. 카피 또는 썸네일 일부 개선 테스트를 진행합니다.",
            "ROAS는 전체 평균 이상이지만 CTR 또는 전환 모수는 추가 보강 여지가 있습니다.",
        )

    if row["CTR"] >= avg_ctr and row["ROAS"] < avg_roas:
        return (
            "개선 필요",
            "클릭 유도력은 있으나 전환 효율이 낮습니다. 랜딩, 혜택, 상품 매칭, 가격 소구를 점검합니다. 후킹 카피는 유지하고 구매 설득 요소를 보강합니다.",
            "CTR은 평균 이상이지만 ROAS가 낮아 구매 설득 요소 점검이 필요합니다.",
        )

    if row["CTR"] < avg_ctr and row["ROAS"] < avg_roas:
        return (
            "교체 필요",
            "소재 교체를 우선 검토합니다. 카피, 비주얼, 혜택 표현을 전면 재기획합니다.",
            "CTR과 ROAS가 모두 평균보다 낮아 소재 교체 우선순위가 높습니다.",
        )

    return (
        "유지",
        "현재 운영을 유지합니다. 추가 데이터를 확인합니다.",
        "성과가 평균권에 있어 운영을 유지하면서 추가 데이터를 확인하는 것이 적절합니다.",
    )


def add_diagnosis(creative_df: pd.DataFrame, benchmarks: dict) -> pd.DataFrame:
    diagnosed = creative_df.copy()
    diagnosis = diagnosed.apply(lambda row: diagnose_row(row, benchmarks), axis=1)
    diagnosed["진단 등급"] = [item[0] for item in diagnosis]
    diagnosed["운영 액션"] = [item[1] for item in diagnosis]
    diagnosed["진단 코멘트"] = [item[2] for item in diagnosis]
    diagnosed["문제 유형"] = diagnosed.apply(lambda row: classify_problem(row, benchmarks), axis=1)
    return diagnosed


def classify_problem(row: pd.Series, benchmarks: dict) -> str:
    has_delivery_data = benchmarks["전체 노출수"] > 0 and benchmarks["전체 클릭수"] > 0

    if not has_delivery_data:
        if row["비용"] <= 0:
            return "데이터 부족"
        if row["전환수"] == 0 and row["전환매출"] == 0:
            return "과소진 대비 저효율"
        if row["ROAS"] < benchmarks["전체 ROAS"]:
            return "전환 설득 부족"
        return "타겟-소재 매칭 약함"

    if row["노출수"] < MIN_IMPRESSIONS or row["클릭수"] < MIN_CLICKS:
        return "데이터 부족"

    if row["비용"] > 0 and row["전환수"] == 0 and row["클릭수"] >= MIN_CLICKS:
        return "과소진 대비 저효율"

    if row["CTR"] < benchmarks["전체 CTR"]:
        if row["ROAS"] < benchmarks["전체 ROAS"]:
            return "클릭 유도 부족"
        return "타겟-소재 매칭 약함"

    if row["CTR"] >= benchmarks["전체 CTR"] and row["ROAS"] < benchmarks["전체 ROAS"]:
        benefit_text = " ".join(str(row.get(col, "")) for col in ["혜택 문구", "프로모션명", "카피 문구"])
        if not any(keyword in benefit_text for keyword in ["혜택", "할인", "특가", "쿠폰", "증정", "구성"]):
            return "혜택 소구 약함"
        return "전환 설득 부족"

    return "타겟-소재 매칭 약함"


def format_currency(value: float) -> str:
    return f"{value:,.0f}원"


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:,.2f}%"


def make_metric_cards(summary: dict) -> None:
    rows = [
        [("총 비용", format_currency(summary["총 비용"])), ("총 매출", format_currency(summary["총 매출"])), ("전체 ROAS", format_percent(summary["전체 ROAS"])), ("전체 노출수", format_number(summary["전체 노출수"])), ("전체 클릭수", format_number(summary["전체 클릭수"]))],
        [("전체 CTR", format_percent(summary["전체 CTR"])), ("전체 전환수", format_number(summary["전체 전환수"])), ("전체 CVR", format_percent(summary["전체 CVR"])), ("평균 CPC", format_currency(summary["평균 CPC"])), ("평균 CPA", format_currency(summary["평균 CPA"]))],
    ]

    for metric_row in rows:
        columns = st.columns(5)
        for col, (label, value) in zip(columns, metric_row):
            col.metric(label, value)


def top_values(df: pd.DataFrame, col: str, default: str = "") -> str:
    if col not in df.columns:
        return default
    values = df[col].dropna().astype(str).str.strip()
    values = values[~values.isin(["", "nan", "None", "미입력"])]
    if values.empty:
        return default
    return values.value_counts().index[0]


def combined_text(df: pd.DataFrame, columns: list[str]) -> str:
    parts = []
    for col in columns:
        if col in df.columns:
            parts.extend(df[col].dropna().astype(str).tolist())
    return " ".join(parts)


def has_creative_text(df: pd.DataFrame) -> bool:
    text_columns = [col for col in ["카피 문구", "혜택 문구", "프로모션명", "상품명", "소재 유형"] if col in df.columns]
    if not text_columns:
        return False
    text = combined_text(df, text_columns).strip()
    return bool(text and text not in ["미입력", "nan", "None"])


def common_keywords(df: pd.DataFrame, limit: int = 5) -> list[str]:
    text = combined_text(df, TEXT_COLUMNS_FOR_INSIGHT)
    tokens = re.findall(r"[가-힣A-Za-z0-9%]+", text)
    stopwords = {
        "그리고",
        "하지만",
        "입니다",
        "합니다",
        "있는",
        "없는",
        "으로",
        "에서",
        "에게",
        "까지",
        "부터",
        "대표",
        "상품",
        "서비스",
        "미입력",
    }
    filtered = [token for token in tokens if len(token) >= 2 and token not in stopwords]
    return [word for word, _ in Counter(filtered).most_common(limit)]


def infer_primary_appeal(df: pd.DataFrame) -> str:
    text = combined_text(df, TEXT_COLUMNS_FOR_INSIGHT)
    if not text.strip():
        return "전환 효율 중심"
    groups = {
        "할인/혜택": ["할인", "특가", "쿠폰", "혜택", "증정", "무료", "첫구매", "구성"],
        "문제 인식": ["고민", "걱정", "필요", "부담", "관리", "챙겨", "습관", "불편"],
        "후기/공감": ["후기", "리뷰", "공감", "요즘", "다들", "선택"],
        "긴급성/마감": ["마감", "오늘", "지금", "한정", "기간", "놓치"],
        "CRM 리마인드": ["다시", "재구매", "장바구니", "회원", "기존"],
    }
    scores = {name: sum(text.count(keyword) for keyword in keywords) for name, keywords in groups.items()}
    best_name, best_score = max(scores.items(), key=lambda item: item[1])
    return best_name if best_score > 0 else "전환 효율 중심"


def generate_high_insights(creative_df: pd.DataFrame, benchmarks: dict) -> list[str]:
    if creative_df.empty:
        return ["분석할 소재 데이터가 충분하지 않습니다."]

    top_roas = creative_df.sort_values("ROAS", ascending=False).head(5)
    top_ctr = creative_df.sort_values("CTR", ascending=False).head(5)
    appeal = infer_primary_appeal(top_roas)
    keywords = common_keywords(top_roas)
    target = top_values(top_roas, "타겟명")
    creative_type = top_values(top_roas, "소재 유형")
    placement = top_values(top_roas, "지면명")

    if has_creative_text(top_roas):
        insights = [f"ROAS 상위 소재에서는 {appeal} 소구가 상대적으로 강하게 확인됩니다."]
    else:
        insights = ["ROAS 상위 항목은 캠페인, 타겟, 기기 조합에서 전환 효율이 상대적으로 높게 나타납니다."]

    if has_creative_text(top_roas) and keywords:
        insights.append(f"고효율 소재 카피에는 {', '.join(keywords[:3])} 등의 표현이 자주 포함됩니다.")
    if target:
        insights.append(f"{target} 타겟에서 전환 효율이 상대적으로 높게 확인됩니다.")
    if creative_type:
        insights.append(f"{creative_type} 유형 소재가 상위권에 반복적으로 포함됩니다.")
    if placement:
        insights.append(f"{placement} 지면에서는 현재 고효율 소구를 우선 확장하는 것이 적절합니다.")

    if benchmarks["전체 노출수"] == 0 and benchmarks["전체 클릭수"] == 0:
        insights.append("상위 항목은 비용 대비 전환매출이 높아 예산 배분 우선순위를 검토할 수 있습니다.")
    elif top_ctr["CTR"].mean() > benchmarks["전체 CTR"] and top_ctr["ROAS"].mean() < benchmarks["전체 ROAS"]:
        insights.append("CTR 상위 소재 중 일부는 후킹은 강하지만 구매 설득 요소가 부족해 ROAS 보강이 필요합니다.")
    else:
        insights.append("상위 소재는 클릭 유도 문구와 구매 명분이 함께 제시될 때 반응이 더 안정적으로 나타납니다.")

    return insights[:6]


def generate_low_insights(problem_df: pd.DataFrame) -> list[str]:
    if problem_df.empty:
        return ["하위 소재를 분석할 데이터가 충분하지 않습니다."]

    counts = problem_df["문제 유형"].value_counts()
    insights = []
    for problem_type, count in counts.items():
        insights.append(f"{problem_type} 유형이 {count}개 항목에서 확인됩니다.")

    if "클릭 유도 부족" in counts.index:
        insights.append("CTR 하위 소재는 첫 화면에서 소구가 바로 읽히도록 카피와 비주얼 우선순위를 단순화해야 합니다.")
    if "전환 설득 부족" in counts.index or "혜택 소구 약함" in counts.index:
        insights.append("클릭은 발생하지만 ROAS가 낮은 소재는 혜택, 구성, 랜딩 메시지의 일관성을 보강해야 합니다.")
    if "과소진 대비 저효율" in counts.index:
        insights.append("비용이 사용됐지만 전환이 부족한 소재는 교체 또는 예산 축소를 우선 검토합니다.")
    if "데이터 부족" in counts.index:
        insights.append("모수가 부족한 소재는 단기 테스트를 유지하되 판단 기준 도달 후 재분류합니다.")

    return insights[:6]


def get_context_values(creative_df: pd.DataFrame) -> dict:
    top_roas = creative_df.sort_values("ROAS", ascending=False).head(5)
    return {
        "product": top_values(top_roas, "상품명", "업로드 파일 내 상품명 미제공"),
        "target": top_values(top_roas, "타겟명", "핵심 전환 가능성이 높은 타겟"),
        "appeal": infer_primary_appeal(top_roas),
        "creative_type": top_values(top_roas, "소재 유형", "이미지형/배너형"),
        "keywords": common_keywords(top_roas, 3),
    }


def product_copy_subject(product: str) -> str:
    if product in ["", "업로드 파일 내 상품명 미제공", "미입력", "nan", "None"]:
        return "일상 관리"
    return product


def generate_creative_directions(creative_df: pd.DataFrame) -> pd.DataFrame:
    context = get_context_values(creative_df)
    product = product_copy_subject(context["product"])
    target = context["target"]
    appeal = context["appeal"]
    keyword_text = ", ".join(context["keywords"]) if context["keywords"] else "관리, 혜택, 시작"

    rows = [
        {
            "소재 방향명": "관리 필요 순간 공감형",
            "타겟": target,
            "핵심 소구": "일상 속 관리 필요성을 자연스럽게 환기합니다.",
            "메인 카피": f"{product}, 챙겨야 할 때가 있습니다",
            "서브 카피": "부담 없이 시작하는 데일리 관리 루틴입니다.",
            "디자인 방향": "상황 컷과 짧은 문장을 크게 배치하고, 제품 또는 혜택 정보는 하단에 정리합니다.",
            "강조해야 할 문구": "꾸준한 관리와 부담 없는 시작을 강조합니다.",
            "피해야 할 요소": "과도한 효능 단정, 복잡한 정보 나열, 지나친 자극 문구는 피합니다.",
            "기대 효과": "문제 인식 기반 클릭률 상승과 신규 유입 확대가 기대됩니다.",
        },
        {
            "소재 방향명": "혜택 명확 제안형",
            "타겟": target,
            "핵심 소구": f"상위 항목에서 확인된 {appeal} 흐름을 혜택 중심으로 확장합니다.",
            "메인 카피": "지금 시작하기 좋은 혜택",
            "서브 카피": f"{product} 구성과 혜택을 한눈에 확인할 수 있습니다.",
            "디자인 방향": "혜택 문구를 최상단에 고정하고 가격과 구성 정보는 대비가 큰 영역에 배치합니다.",
            "강조해야 할 문구": "오늘의 혜택과 부담 없는 시작을 강조합니다.",
            "피해야 할 요소": "혜택 정보가 작게 보이거나 조건이 불명확한 표현은 피합니다.",
            "기대 효과": "클릭 이후 구매를 검토할 명분이 강화됩니다.",
        },
        {
            "소재 방향명": "후기 공감 확장형",
            "타겟": "구매를 고민 중인 비교 탐색층",
            "핵심 소구": "직접적인 단정보다 공감형 메시지로 신뢰감을 형성합니다.",
            "메인 카피": "꾸준한 관리부터 시작합니다",
            "서브 카피": f"{keyword_text} 포인트를 중심으로 선택 이유를 제시합니다.",
            "디자인 방향": "짧은 후기형 문장, 말풍선형 정보 구조, 제품 이미지를 균형 있게 배치합니다.",
            "강조해야 할 문구": "선택 이유와 꾸준한 습관을 강조합니다.",
            "피해야 할 요소": "검증되지 않은 수치 표현과 과한 전후 비교는 피합니다.",
            "기대 효과": "신뢰 보강을 통한 CVR 개선이 기대됩니다.",
        },
        {
            "소재 방향명": "CRM 리마인드형",
            "타겟": "방문 또는 클릭 이력이 있는 리타겟 모수",
            "핵심 소구": "고민 중인 사용자의 재방문을 유도합니다.",
            "메인 카피": "고민 중이라면 혜택부터 확인합니다",
            "서브 카피": "놓치기 아쉬운 구성으로 다시 제안합니다.",
            "디자인 방향": "제품, 혜택, CTA를 3단 구조로 단순하게 구성하고 시선 흐름을 버튼 방향으로 유도합니다.",
            "강조해야 할 문구": "다시 확인할 혜택과 유지되는 구성을 강조합니다.",
            "피해야 할 요소": "압박감이 큰 문구와 정보가 과도하게 많은 배너는 피합니다.",
            "기대 효과": "기존 관심자의 전환 회수율 개선이 기대됩니다.",
        },
        {
            "소재 방향명": "신규 유입 입문형",
            "타겟": "브랜드 또는 제품을 처음 접하는 신규 타겟",
            "핵심 소구": "첫 구매 장벽을 낮출 수 있도록 쉽게 설명합니다.",
            "메인 카피": "처음이라면 가볍게 시작합니다",
            "서브 카피": f"{product}의 핵심 포인트를 짧고 명확하게 소개합니다.",
            "디자인 방향": "제품명, 핵심 혜택, 사용 상황이 한 화면에서 읽히도록 구성합니다.",
            "강조해야 할 문구": "첫 시작과 간편한 선택을 강조합니다.",
            "피해야 할 요소": "전문 용어가 많거나 작은 글씨 중심인 구성은 피합니다.",
            "기대 효과": "신규 유입 CTR 개선과 테스트 모수 확대가 기대됩니다.",
        },
    ]

    return pd.DataFrame([{key: sanitize_ad_text(value) for key, value in row.items()} for row in rows])


def generate_copy_list(creative_df: pd.DataFrame) -> pd.DataFrame:
    context = get_context_values(creative_df)
    product = product_copy_subject(context["product"])

    copy_map = {
        "문제 인식형": [
            f"{product}, 챙겨야 할 때가 있습니다",
            "바쁜 일상 속 관리가 필요한 순간입니다",
            "요즘 컨디션 관리가 신경 쓰이는 때입니다",
            "미루던 관리를 오늘부터 가볍게 시작합니다",
            "매일 챙기는 습관이 필요한 순간입니다",
            "나를 위한 관리 루틴을 시작합니다",
            "놓치기 쉬운 일상 관리 포인트입니다",
            "꾸준히 챙기고 싶은 분에게 맞습니다",
            "관리의 시작은 작은 선택부터입니다",
            "오늘의 나를 위한 간편한 챙김입니다",
        ],
        "할인/혜택형": [
            "지금 시작하기 좋은 혜택입니다",
            "부담을 낮춘 특별 구성입니다",
            "첫 시작을 위한 혜택입니다",
            "오늘의 구성을 한눈에 비교합니다",
            "놓치기 아쉬운 혜택 모음입니다",
            "가볍게 시작하는 실속 구성입니다",
            "혜택은 크게, 부담은 가볍게 구성됩니다",
            "지금 확인하기 좋은 구성입니다",
            "인기 구성 혜택을 제안합니다",
            "구매 전 꼭 확인할 혜택입니다",
        ],
        "후기/공감형": [
            "꾸준한 관리부터 시작합니다",
            "선택 이유가 분명한 데일리 루틴입니다",
            "요즘 많이 찾는 관리 습관입니다",
            "처음 시작하는 분도 부담이 적습니다",
            "꾸준히 챙기는 분들의 선택입니다",
            "나에게 맞는 루틴을 찾는 과정입니다",
            "후기로 확인하는 선택 포인트입니다",
            "고민 끝에 고른 이유가 있습니다",
            "일상에 자연스럽게 더하는 습관입니다",
            "공감되는 관리 고민을 쉽게 시작합니다",
        ],
        "긴급성/마감형": [
            "혜택 마감 전 확인이 필요합니다",
            "오늘 구성은 지금 확인됩니다",
            "기간 한정 혜택이 제공됩니다",
            "이번 혜택은 오늘 확인이 필요합니다",
            "준비된 수량이 소진되기 전입니다",
            "지금 확인하기 좋은 구성입니다",
            "고민 중이라면 마감 전 확인이 필요합니다",
            "오늘의 혜택을 빠르게 확인합니다",
            "이번 주 추천 구성입니다",
            "늦기 전에 혜택부터 확인합니다",
        ],
        "CRM 리타겟용": [
            "아직 고민 중이라면 다시 확인합니다",
            "장바구니 속 혜택을 확인합니다",
            "전에 본 구성도 지금 확인 가능합니다",
            "관심 있던 혜택을 다시 제안합니다",
            "망설였다면 이번 구성을 확인합니다",
            "다시 찾은 분들을 위한 혜택입니다",
            "놓친 혜택이 있는지 확인합니다",
            "구매 전 마지막 체크 포인트입니다",
            "관심 상품 혜택을 이어서 확인합니다",
            "다시 보면 더 쉬운 선택입니다",
        ],
        "신규 유입용": [
            "처음이라면 가볍게 시작합니다",
            f"{product}, 핵심만 쉽게 확인됩니다",
            "처음 만나는 데일리 관리 루틴입니다",
            "입문자를 위한 쉬운 선택입니다",
            "복잡한 고민 없이 간편하게 시작합니다",
            "오늘부터 시작하는 관리 습관입니다",
            "처음 시작할 때 필요한 구성입니다",
            "나에게 맞는 선택을 찾습니다",
            "부담 없는 첫 시작을 도와드립니다",
            "핵심 혜택부터 천천히 확인합니다",
        ],
    }

    rows = []
    for copy_type, copies in copy_map.items():
        for copy in copies:
            rows.append({"카피 유형": copy_type, "카피": sanitize_ad_text(copy)})
    return pd.DataFrame(rows)


def generate_designer_brief(
    creative_df: pd.DataFrame,
    directions_df: pd.DataFrame,
    high_insights: list[str],
) -> str:
    context = get_context_values(creative_df)
    product = context["product"]
    target = context["target"]
    top_direction = directions_df.iloc[0]
    required_phrases = " / ".join(directions_df["강조해야 할 문구"].head(3).tolist())

    brief = f"""
[디자이너 전달용 제작 요청서]

제작 목적
GFA 성과 분석 결과를 바탕으로 효율이 높은 소구는 확장하고, 효율이 낮은 항목은 신규 방향으로 교체하기 위한 제작 요청입니다.

운영 매체
네이버 GFA

타겟
{target}

제품/브랜드
{product}

핵심 메시지
{top_direction['핵심 소구']}

필수 포함 문구
{required_phrases}

디자인 톤앤매너
명확하고 직관적인 정보 구조를 우선합니다. 메인 카피는 첫 화면에서 바로 읽히도록 크게 배치합니다. 혜택, 구성, CTA는 하단 또는 우측에 정리합니다.

참고할 고효율 소재 특징
{chr(10).join(f"- {insight}" for insight in high_insights)}

제작 사이즈
1080x1080, 1200x628, 1200x1200, 1080x1920 기준으로 베리에이션을 준비합니다. 실제 집행 규격은 운영 계정 기준으로 최종 확인합니다.

베리에이션 방향
1. 문제 인식형 카피 중심
2. 혜택/구성 강조형
3. 후기/공감형
4. CRM 리마인드형
5. 신규 유입 입문형

주의사항
의학적 단정, 질병 관련 직접 표현, 과도한 효능 보장, 검증되지 않은 수치 표현은 배제합니다. 건강기능식품 소재로도 활용 가능하도록 '꾸준한 관리', '섭취 습관', '일상 속 관리'처럼 완곡한 표현을 사용합니다.
""".strip()
    return sanitize_ad_text(brief)


def generate_report_comment(
    summary: dict,
    creative_df: pd.DataFrame,
    high_insights: list[str],
    low_insights: list[str],
    directions_df: pd.DataFrame,
) -> str:
    context = get_context_values(creative_df)
    appeal = context["appeal"]
    top_direction = directions_df.iloc[0]["소재 방향명"]
    replace_count = int((creative_df["진단 등급"] == "교체 필요").sum())
    hold_count = int((creative_df["진단 등급"] == "판단 보류").sum())
    has_delivery_data = summary["전체 노출수"] > 0 and summary["전체 클릭수"] > 0
    has_text_data = has_creative_text(creative_df)
    if has_delivery_data:
        performance_sentence = (
            f"이번 GFA 성과는 총 비용 {format_currency(summary['총 비용'])}, 총 매출 {format_currency(summary['총 매출'])}, "
            f"전체 ROAS {format_percent(summary['전체 ROAS'])}로 확인됩니다. "
            f"전체 CTR은 {format_percent(summary['전체 CTR'])}, CVR은 {format_percent(summary['전체 CVR'])}입니다."
        )
    else:
        performance_sentence = (
            f"이번 GFA 성과는 총 비용 {format_currency(summary['총 비용'])}, 총 매출 {format_currency(summary['총 매출'])}, "
            f"전체 ROAS {format_percent(summary['전체 ROAS'])}로 확인됩니다. "
            "업로드 파일에 노출수와 클릭수 컬럼이 없어 CTR, CPC, CVR 지표는 참고에서 제외하고 비용, 전환수, 전환매출, ROAS 중심으로 해석합니다."
        )
    if has_text_data:
        high_feature_sentence = f"성과 상위 항목에서는 {appeal} 소구가 상대적으로 긍정적인 반응을 보입니다."
    else:
        high_feature_sentence = "성과 상위 항목은 캠페인, 타겟, 기기 조합에서 전환 효율이 상대적으로 높게 나타납니다."
    supporting_high_insight = high_insights[1] if len(high_insights) > 1 else "상위 항목을 중심으로 추가 확장이 가능합니다."
    if has_delivery_data:
        operation_sentence = (
            "ROAS와 CTR이 함께 좋은 항목은 예산 확대와 유사 베리에이션 제작을 검토합니다. "
            "CTR은 높지만 ROAS가 낮은 항목은 랜딩 메시지와 혜택 표현을 보강합니다. "
            "CTR과 ROAS가 모두 낮은 항목은 우선 교체 대상으로 관리합니다."
        )
    else:
        operation_sentence = (
            "ROAS와 전환매출이 높은 항목은 예산 확대와 유사 베리에이션 제작을 검토합니다. "
            "비용은 사용됐지만 전환 성과가 낮은 항목은 타겟, 캠페인 구조, 혜택 메시지를 우선 점검합니다. "
            "전환 성과가 확인되지 않는 항목은 교체 또는 예산 축소 대상으로 관리합니다."
        )

    report = f"""
[광고주 보고용 코멘트]

전체 성과 요약
{performance_sentence}

고효율 소재 특징
{high_feature_sentence} {supporting_high_insight}

저효율 소재 원인
저효율 항목에서는 {low_insights[0] if low_insights else '클릭 또는 전환 설득 요소가 부족한 흐름이 확인됩니다.'} 교체 필요 항목은 {replace_count}개이며, 데이터 추가 확인이 필요한 항목은 {hold_count}개입니다.

향후 운영 방향
{operation_sentence}

신규 소재 제작 방향
신규 소재는 '{top_direction}' 방향을 우선 제작합니다. 혜택 명확 제안형과 CRM 리마인드형을 함께 테스트해 효율 개선을 시도합니다.
""".strip()
    return sanitize_ad_text(report)


def generate_kakao_summary(creative_df: pd.DataFrame, directions_df: pd.DataFrame) -> str:
    context = get_context_values(creative_df)
    appeal = context["appeal"]
    direction = directions_df.iloc[0]["소재 방향명"]
    return sanitize_ad_text(
        f"""안녕하세요 팀장님:)
GFA 소재 성과 확인 결과, {appeal} 소구 소재의 반응이 상대적으로 우수하게 확인되었습니다.
해당 방향은 유지하되, 저효율 소재는 교체하고 {direction} 중심의 신규 베리에이션을 추가 제작하겠습니다."""
    )


def make_problem_table(creative_df: pd.DataFrame) -> pd.DataFrame:
    low_roas = creative_df.sort_values("ROAS", ascending=True).head(5)
    low_ctr = creative_df.sort_values("CTR", ascending=True).head(5)
    problem_df = (
        pd.concat([low_roas, low_ctr], ignore_index=True)
        .drop_duplicates(subset=["소재명", "캠페인명", "광고그룹명"])
        .copy()
    )
    return problem_df


def make_bar_chart(df: pd.DataFrame, metric: str, title: str, ascending: bool = False) -> go.Figure:
    chart_df = df.sort_values(metric, ascending=ascending).head(10)
    fig = px.bar(
        chart_df,
        x=metric,
        y="소재명",
        color="진단 등급",
        orientation="h",
        hover_data=["캠페인명", "광고그룹명", "CTR", "CVR", "ROAS", "비용"],
        title=title,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def make_grade_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["진단 등급"].value_counts().reset_index()
    counts.columns = ["진단 등급", "소재 수"]
    fig = px.bar(counts, x="진단 등급", y="소재 수", color="진단 등급", text="소재 수")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def make_scatter(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="진단 등급",
        size="클릭수",
        hover_name="소재명",
        hover_data=["캠페인명", "광고그룹명", "비용", "전환수", "전환매출", "ROAS"],
        title=title,
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def make_campaign_chart(raw_df: pd.DataFrame) -> go.Figure:
    campaign_df = raw_df.groupby("캠페인명", as_index=False).agg(
        {"비용": "sum", "전환매출": "sum", "노출수": "sum", "클릭수": "sum", "전환수": "sum"}
    )
    campaign_df["ROAS"] = safe_ratio(campaign_df["전환매출"], campaign_df["비용"], 100)
    campaign_df = campaign_df.sort_values("비용", ascending=False).head(15)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=campaign_df["캠페인명"], y=campaign_df["비용"], name="비용"), secondary_y=False)
    fig.add_trace(go.Bar(x=campaign_df["캠페인명"], y=campaign_df["전환매출"], name="매출"), secondary_y=False)
    fig.add_trace(go.Scatter(x=campaign_df["캠페인명"], y=campaign_df["ROAS"], name="ROAS", mode="lines+markers"), secondary_y=True)
    fig.update_layout(title="캠페인별 비용/매출/ROAS 비교", barmode="group", height=460, margin=dict(l=10, r=10, t=60, b=90))
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(title_text="비용/매출", secondary_y=False)
    fig.update_yaxes(title_text="ROAS(%)", secondary_y=True)
    return fig


def dataframe_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def text_download(text: str) -> bytes:
    return text.encode("utf-8-sig")


def display_top_table_pair(creative_df: pd.DataFrame) -> None:
    left, right = st.columns(2)
    columns = ["소재명", "캠페인명", "광고그룹명", "비용", "CTR", "CVR", "ROAS", "진단 등급"]
    with left:
        st.markdown("**ROAS 상위 5개 소재**")
        st.dataframe(creative_df.sort_values("ROAS", ascending=False).head(5)[columns], use_container_width=True, hide_index=True)
    with right:
        st.markdown("**CTR 상위 5개 소재**")
        st.dataframe(creative_df.sort_values("CTR", ascending=False).head(5)[columns], use_container_width=True, hide_index=True)


def display_low_table_pair(creative_df: pd.DataFrame) -> None:
    left, right = st.columns(2)
    columns = ["소재명", "캠페인명", "광고그룹명", "비용", "CTR", "CVR", "ROAS", "진단 등급", "문제 유형"]
    with left:
        st.markdown("**ROAS 하위 5개 소재**")
        st.dataframe(creative_df.sort_values("ROAS", ascending=True).head(5)[columns], use_container_width=True, hide_index=True)
    with right:
        st.markdown("**CTR 하위 5개 소재**")
        st.dataframe(creative_df.sort_values("CTR", ascending=True).head(5)[columns], use_container_width=True, hide_index=True)


def render_download_buttons(
    diagnosis_df: pd.DataFrame,
    directions_df: pd.DataFrame,
    copy_df: pd.DataFrame,
    designer_brief: str,
    report_comment: str,
) -> None:
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "소재별 진단 결과 CSV",
        data=dataframe_download(diagnosis_df),
        file_name="gfa_creative_diagnosis.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col2.download_button(
        "신규 소재 제작 방향 CSV",
        data=dataframe_download(directions_df),
        file_name="gfa_new_creative_directions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col3.download_button(
        "신규 카피 리스트 CSV",
        data=dataframe_download(copy_df),
        file_name="gfa_new_copy_list.csv",
        mime="text/csv",
        use_container_width=True,
    )

    col4, col5 = st.columns(2)
    col4.download_button(
        "디자이너 제작 요청서 TXT",
        data=text_download(designer_brief),
        file_name="gfa_designer_brief.txt",
        mime="text/plain",
        use_container_width=True,
    )
    col5.download_button(
        "광고주 보고용 코멘트 TXT",
        data=text_download(report_comment),
        file_name="gfa_client_report_comment.txt",
        mime="text/plain",
        use_container_width=True,
    )


st.title(APP_TITLE)
st.caption("네이버 GFA 소재 성과 데이터를 업로드하면 진단, 인사이트, 신규 제작 방향, 카피, 요청서까지 한 번에 정리합니다.")

st.header("1. 파일 업로드 영역")
uploaded_file = st.file_uploader("CSV 또는 XLSX 파일 업로드", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info(
        "GFA 권장 리포트 설정은 분석 단위 '광고 소재', 기간 단위 '일' 또는 '전체'입니다. "
        "게재 위치와 오디언스는 '전체' 기준을 권장합니다. "
        "필수 성과 컬럼은 노출수, 클릭수, 비용이며 전환수/전환매출은 있으면 ROAS·CPA 진단에 활용합니다."
    )
    st.stop()

try:
    raw_df = read_uploaded_file(uploaded_file)
    raw_df, column_mapping = apply_column_aliases(raw_df)
except Exception as exc:
    st.error(f"파일을 읽는 중 문제가 발생했습니다: {exc}")
    st.stop()

missing_columns = validate_required_columns(raw_df)
if missing_columns:
    st.error(build_missing_column_message(missing_columns))
    st.stop()

missing_optional_columns = [
    col for col in RECOMMENDED_DIMENSION_COLUMNS + OPTIONAL_PERFORMANCE_COLUMNS if col not in raw_df.columns
]
if missing_optional_columns:
    st.caption(
        "GFA 리포트에서 확인되지 않은 선택 컬럼은 자동 보정했습니다: "
        + ", ".join(missing_optional_columns)
    )

with st.expander("인식된 GFA 리포트 컬럼 매핑", expanded=False):
    if column_mapping:
        mapping_df = pd.DataFrame(
            [{"앱 기준 컬럼": canonical, "업로드 파일 컬럼": source} for canonical, source in column_mapping.items()]
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
    else:
        st.write("자동 매핑된 컬럼이 없습니다. 업로드 파일의 컬럼명을 확인해 주시기 바랍니다.")

prepared_df = prepare_raw_data(raw_df)

if prepared_df["노출수"].sum() == 0 and prepared_df["클릭수"].sum() == 0:
    st.warning(
        "이 파일에는 노출수/클릭수 컬럼이 없어 CTR, CPC, CVR 기반 진단은 제한됩니다. "
        "현재 파일은 비용, 전환수, 전환매출, ROAS 중심으로 분석합니다."
    )

if "소재명" not in column_mapping:
    st.warning(
        "이 파일에는 소재명 컬럼이 없어 캠페인명과 기기/타겟 정보를 조합해 분석 항목명을 만들었습니다. "
        "정확한 소재별 분석이 필요하면 GFA 리포트의 분석 단위를 '광고 소재'로 설정하는 것을 권장합니다."
    )

creative_df = aggregate_creatives(prepared_df)
summary = build_summary(prepared_df)
diagnosis_df = add_diagnosis(creative_df, summary)

diagnosis_columns = [
    "소재명",
    "캠페인명",
    "광고그룹명",
    "비용",
    "노출수",
    "클릭수",
    "CTR",
    "CPC",
    "전환수",
    "CVR",
    "CPA",
    "전환매출",
    "ROAS",
    "진단 등급",
    "운영 액션",
    "진단 코멘트",
]
diagnosis_export_df = diagnosis_df[diagnosis_columns + ["문제 유형"]].copy()

top_roas_5 = diagnosis_df.sort_values("ROAS", ascending=False).head(5)
top_ctr_5 = diagnosis_df.sort_values("CTR", ascending=False).head(5)
problem_df = make_problem_table(diagnosis_df)
high_insights = generate_high_insights(diagnosis_df, summary)
low_insights = generate_low_insights(problem_df)
directions_df = generate_creative_directions(diagnosis_df)
copy_df = generate_copy_list(diagnosis_df)
designer_brief = generate_designer_brief(diagnosis_df, directions_df, high_insights)
report_comment = generate_report_comment(summary, diagnosis_df, high_insights, low_insights, directions_df)
kakao_summary = generate_kakao_summary(diagnosis_df, directions_df)

st.header("2. 데이터 미리보기")
st.dataframe(prepared_df.head(50), use_container_width=True, hide_index=True)

st.header("3. 전체 성과 요약 카드")
make_metric_cards(summary)

st.header("4. 소재별 진단 테이블")
st.dataframe(diagnosis_export_df, use_container_width=True, hide_index=True)

st.header("5. 소재 진단 등급별 분포 차트")
st.plotly_chart(make_grade_distribution_chart(diagnosis_df), use_container_width=True)

st.header("6. ROAS 상위/하위 소재 차트")
roas_tab_1, roas_tab_2 = st.tabs(["ROAS 상위 10", "ROAS 하위 10"])
with roas_tab_1:
    st.plotly_chart(make_bar_chart(diagnosis_df, "ROAS", "ROAS 상위 10개 소재", ascending=False), use_container_width=True)
with roas_tab_2:
    st.plotly_chart(make_bar_chart(diagnosis_df, "ROAS", "ROAS 하위 10개 소재", ascending=True), use_container_width=True)

st.header("7. CTR 상위/하위 소재 차트")
ctr_tab_1, ctr_tab_2 = st.tabs(["CTR 상위 10", "CTR 하위 10"])
with ctr_tab_1:
    st.plotly_chart(make_bar_chart(diagnosis_df, "CTR", "CTR 상위 10개 소재", ascending=False), use_container_width=True)
with ctr_tab_2:
    st.plotly_chart(make_bar_chart(diagnosis_df, "CTR", "CTR 하위 10개 소재", ascending=True), use_container_width=True)

scatter_col_1, scatter_col_2 = st.columns(2)
with scatter_col_1:
    st.plotly_chart(make_scatter(diagnosis_df, "비용", "ROAS", "비용 대비 ROAS 산점도"), use_container_width=True)
with scatter_col_2:
    st.plotly_chart(make_scatter(diagnosis_df, "CTR", "CVR", "CTR 대비 CVR 산점도"), use_container_width=True)

st.plotly_chart(make_campaign_chart(prepared_df), use_container_width=True)

st.header("8. 고효율 소재 인사이트")
display_top_table_pair(diagnosis_df)
for insight in high_insights:
    st.markdown(f"- {insight}")

st.header("9. 저효율 소재 문제 분석")
display_low_table_pair(diagnosis_df)
problem_columns = ["소재명", "캠페인명", "광고그룹명", "비용", "CTR", "CVR", "ROAS", "진단 등급", "문제 유형", "진단 코멘트"]
st.dataframe(problem_df[problem_columns], use_container_width=True, hide_index=True)
for insight in low_insights:
    st.markdown(f"- {insight}")

st.header("10. 신규 소재 제작 방향")
st.dataframe(directions_df, use_container_width=True, hide_index=True)

st.header("11. 신규 카피 제안")
copy_tabs = st.tabs(list(copy_df["카피 유형"].unique()))
for tab, copy_type in zip(copy_tabs, copy_df["카피 유형"].unique()):
    with tab:
        st.dataframe(copy_df[copy_df["카피 유형"] == copy_type], use_container_width=True, hide_index=True)

st.header("12. 디자이너 제작 요청서")
st.text_area("제작 요청서", designer_brief, height=420)

st.header("13. 광고주 보고용 코멘트")
st.text_area("보고 코멘트", report_comment, height=360)

st.header("14. 카톡 공유용 요약")
st.text_area("카톡 요약", kakao_summary, height=160)

st.header("15. 결과 다운로드 버튼")
render_download_buttons(diagnosis_export_df, directions_df, copy_df, designer_brief, report_comment)
