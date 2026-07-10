import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ==========================================
# 0. 文件路径配置
# ==========================================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

MODEL_PATH = BASE_DIR / "HGB.pkl"
DATA_PATH = BASE_DIR / "BFRC.xlsx"
SHEET_NAME = "抗折"
TARGET_COLUMN = "抗折强度/Mpa"

FALLBACK_FEATURE_COLUMNS = [
    "水泥 kg/m³",
    "粉煤灰 kg/m³",
    "硅灰 kg/m³",
    "粗骨料 kg/m³",
    "细骨料 kg/m³",
    "水 kg/m³",
    "减水剂 kg/m³",
    "纤维直径/mm",
    "纤维长度/mm",
    "纤维掺量/%"
]

# ==========================================
# 1. 多语言配置
# ==========================================
TRANSLATIONS = {
    "cn": {
        "page_title": "BFRC 抗折强度预测系统",
        "main_title": "🧱 BFRC 抗折强度智能预测系统",
        "sub_title": "基于已训练好的 **HGBoost 回归模型**，预测玄武岩纤维混凝土抗折强度。",
        "load_fail": "⚠️ 模型加载失败：",
        "data_fail": "⚠️ 数据集读取失败，已使用内置特征顺序：",
        "sec1_title": "### 1. 胶凝材料参数",
        "sec2_title": "### 2. 骨料、水与外加剂参数",
        "sec3_title": "### 3. 玄武岩纤维参数",
        "cement": "水泥 (kg/m³)",
        "fly_ash": "粉煤灰 (kg/m³)",
        "silica_fume": "硅灰 (kg/m³)",
        "coarse_agg": "粗骨料 (kg/m³)",
        "fine_agg": "细骨料 (kg/m³)",
        "water": "水 (kg/m³)",
        "water_reducer": "减水剂 (kg/m³)",
        "fiber_diameter": "纤维直径 (mm)",
        "fiber_length": "纤维长度 (mm)",
        "fiber_content": "纤维掺量 (%)",
        "btn_predict": "🔍 开始预测",
        "res_title": "预测结果：抗折强度",
        "res_unit": "MPa",
        "btn_download": "📥 导出结果",
        "input_summary": "输入参数",
        "feature_order": "模型输入特征顺序",
        "prediction_error": "预测失败："
    },
    "en": {
        "page_title": "BFRC Flexural Strength Prediction System",
        "main_title": "🧱 BFRC Flexural Strength Prediction System",
        "sub_title": "Prediction of basalt fiber reinforced concrete flexural strength using the trained **HGBoost regressor**.",
        "load_fail": "⚠️ Model loading failed: ",
        "data_fail": "⚠️ Failed to load dataset. Built-in feature order is used: ",
        "sec1_title": "### 1. Binder Parameters",
        "sec2_title": "### 2. Aggregates, Water and Admixture",
        "sec3_title": "### 3. Basalt Fiber Parameters",
        "cement": "Cement (kg/m³)",
        "fly_ash": "Fly ash (kg/m³)",
        "silica_fume": "Silica fume (kg/m³)",
        "coarse_agg": "Coarse aggregate (kg/m³)",
        "fine_agg": "Fine aggregate (kg/m³)",
        "water": "Water (kg/m³)",
        "water_reducer": "Water reducer (kg/m³)",
        "fiber_diameter": "Fiber diameter (mm)",
        "fiber_length": "Fiber length (mm)",
        "fiber_content": "Fiber content (%)",
        "btn_predict": "🔍 Run Prediction",
        "res_title": "Prediction Result: Flexural Strength",
        "res_unit": "MPa",
        "btn_download": "📥 Download Results",
        "input_summary": "Input Parameters",
        "feature_order": "Model Input Feature Order",
        "prediction_error": "Prediction failed: "
    }
}

FEATURE_LABEL_KEYS = {
    "水泥 kg/m³": "cement",
    "粉煤灰 kg/m³": "fly_ash",
    "硅灰 kg/m³": "silica_fume",
    "粗骨料 kg/m³": "coarse_agg",
    "细骨料 kg/m³": "fine_agg",
    "水 kg/m³": "water",
    "减水剂 kg/m³": "water_reducer",
    "纤维直径/mm": "fiber_diameter",
    "纤维长度/mm": "fiber_length",
    "纤维掺量/%": "fiber_content"
}

INPUT_CONFIG = {
    "水泥 kg/m³": {"step": 1.0, "format": "%.3f"},
    "粉煤灰 kg/m³": {"step": 1.0, "format": "%.3f"},
    "硅灰 kg/m³": {"step": 1.0, "format": "%.3f"},
    "粗骨料 kg/m³": {"step": 1.0, "format": "%.3f"},
    "细骨料 kg/m³": {"step": 1.0, "format": "%.3f"},
    "水 kg/m³": {"step": 1.0, "format": "%.3f"},
    "减水剂 kg/m³": {"step": 0.1, "format": "%.3f"},
    "纤维直径/mm": {"step": 0.001, "format": "%.4f"},
    "纤维长度/mm": {"step": 1.0, "format": "%.3f"},
    "纤维掺量/%": {"step": 0.01, "format": "%.3f"}
}

# ==========================================
# 2. 页面基本配置
# ==========================================
st.set_page_config(
    page_title="BFRC Flexural Strength Prediction",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 3. CSS 样式
# ==========================================
st.markdown("""
<style>
/* 常规文字字体：Arial 优先，中文使用系统中文字体兜底 */
html, body, .stApp, .stMarkdown, .stText, .stCaption,
label, p, input, textarea, button, select,
[data-testid="stWidgetLabel"],
[data-testid="stMarkdownContainer"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    font-family: Arial, "Microsoft YaHei", "SimHei", sans-serif !important;
    font-weight: 700 !important;
}

/* 恢复 Streamlit / Material 图标字体，防止 keyboard_arrow_right 等图标文字外露 */
span[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] *,
.material-icons,
.material-icons-outlined,
.material-icons-round,
.material-symbols-outlined,
.material-symbols-rounded,
.material-symbols-sharp {
    font-family: "Material Symbols Rounded", "Material Icons", "Material Icons Round", sans-serif !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 1.25rem !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: "liga" !important;
    -webkit-font-smoothing: antialiased !important;
    font-feature-settings: "liga" !important;
}

/* 主页面背景与宽度 */
.stApp {
    max-width: 860px;
    margin: auto;
    background-color: #eef6ff;
    padding: 1rem 1.5rem 2rem 1.5rem;
}

/* 主标题 */
h1 {
    color: #1565c0;
    font-weight: 900 !important;
    font-size: 1.55rem !important;
    line-height: 1.35;
    font-family: Arial, "Microsoft YaHei", "SimHei", sans-serif !important;
}

/* 三级标题 */
.stMarkdown h3 {
    color: #0d47a1;
    border-bottom: 2px solid #90caf9;
    padding-bottom: 0.3rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    font-family: Arial, "Microsoft YaHei", "SimHei", sans-serif !important;
}

/* 表单边框 */
[data-testid="stForm"] {
    border: 1px solid #b7c7d8;
    border-radius: 8px;
    padding: 1rem;
}

/* 数字输入框 */
[data-testid="stNumberInput"] input {
    font-family: Arial, sans-serif !important;
    font-weight: 700 !important;
}

/* 预测按钮 */
.stButton > button {
    background-color: #2e7d32;
    color: white;
    font-weight: bold !important;
    font-size: 1.08rem;
    padding: 0.65rem 2rem;
    border-radius: 8px;
    border: none;
    width: 100%;
    margin-top: 1rem;
    transition: all 0.3s;
    font-family: Arial, "Microsoft YaHei", "SimHei", sans-serif !important;
}

.stButton > button:hover {
    background-color: #1b5e20;
    color: white;
}

/* 下载按钮 */
.stDownloadButton > button {
    background-color: #1565c0;
    color: white;
    font-weight: bold !important;
    border-radius: 8px;
    border: none;
    width: 100%;
    margin-top: 0.8rem;
    font-family: Arial, "Microsoft YaHei", "SimHei", sans-serif !important;
}

.stDownloadButton > button:hover {
    background-color: #0d47a1;
    color: white;
}

/* 结果框 */
.result-box {
    background-color: #e8f5e9;
    border: 1px solid #c8e6c9;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    margin-top: 1.5rem;
    font-family: Arial, "Microsoft YaHei", "SimHei", sans-serif !important;
}

.result-label {
    color: #455a64;
    font-size: 1.1rem;
    font-weight: bold;
}

.result-value {
    font-size: 2.15rem;
    font-weight: 900 !important;
    color: #2e7d32;
}

.result-unit {
    font-size: 1rem;
    color: #666;
}

/* 语言切换区域：强制“中文”和“English”在同一排 */
div[data-testid="stRadio"] {
    min-width: 190px !important;
}

div[data-testid="stRadio"] > label {
    min-height: 0 !important;
    height: 0 !important;
    visibility: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    justify-content: flex-end !important;
    align-items: center !important;
    gap: 0.75rem !important;
    width: 100% !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label {
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
    width: auto !important;
    min-width: max-content !important;
    margin: 0 !important;
    padding: 0 !important;
    white-space: nowrap !important;
    font-family: Arial, "Microsoft YaHei", "SimHei", sans-serif !important;
    font-weight: 700 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label p {
    white-space: nowrap !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}

/* Expander 标题与图标间距，避免底部文字重叠 */
[data-testid="stExpander"] summary {
    min-height: 2.2rem !important;
    align-items: center !important;
}

[data-testid="stExpander"] summary p {
    margin: 0 !important;
    line-height: 1.3 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. 数据与模型工具函数
# ==========================================
@st.cache_data(show_spinner=False)
def load_dataset_info(data_path: Path):
    """
    读取 1BFRC.xlsx 的“抗折”工作表：
    1. 自动定位包含“抗折强度/Mpa”的表头行；
    2. 提取目标列之前的输入特征；
    3. 计算输入特征中位数作为 GUI 默认值。
    """
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{data_path.name}")

    raw_df = pd.read_excel(data_path, sheet_name=SHEET_NAME, header=None)

    header_row_index = None
    for idx in range(len(raw_df)):
        row_values = raw_df.iloc[idx].astype(str).str.strip().tolist()
        if TARGET_COLUMN in row_values:
            header_row_index = idx
            break

    if header_row_index is None:
        raise ValueError(f"未在工作表“{SHEET_NAME}”中找到目标列：{TARGET_COLUMN}")

    raw_headers = raw_df.iloc[header_row_index].tolist()

    target_index = None
    for i, value in enumerate(raw_headers):
        if str(value).strip() == TARGET_COLUMN:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"无法定位目标列：{TARGET_COLUMN}")

    feature_columns = []
    for value in raw_headers[1:target_index]:
        if pd.notna(value) and str(value).strip():
            feature_columns.append(str(value).strip())

    if len(feature_columns) == 0:
        raise ValueError("未读取到有效输入特征列。")

    data_df = raw_df.iloc[header_row_index + 1:].copy()

    columns = []
    for i, value in enumerate(raw_headers):
        if pd.isna(value) or str(value).strip() == "":
            columns.append(f"Unnamed_{i}")
        else:
            columns.append(str(value).strip())

    data_df.columns = columns

    numeric_df = data_df[feature_columns + [TARGET_COLUMN]].apply(
        pd.to_numeric,
        errors="coerce"
    )
    numeric_df = numeric_df.dropna(subset=feature_columns, how="all")

    defaults = numeric_df[feature_columns].median(numeric_only=True).to_dict()

    fallback_defaults = {
        "水泥 kg/m³": 400.0,
        "粉煤灰 kg/m³": 0.0,
        "硅灰 kg/m³": 0.0,
        "粗骨料 kg/m³": 1135.0,
        "细骨料 kg/m³": 671.0,
        "水 kg/m³": 170.0,
        "减水剂 kg/m³": 3.36,
        "纤维直径/mm": 0.015,
        "纤维长度/mm": 17.0,
        "纤维掺量/%": 0.10
    }

    for col in feature_columns:
        if col not in defaults or pd.isna(defaults[col]):
            defaults[col] = fallback_defaults.get(col, 0.0)

    return {
        "feature_columns": feature_columns,
        "defaults": defaults,
        "data_shape": numeric_df.shape
    }


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    """加载训练好的 XGBoost 模型。"""
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件：{model_path.name}")
    return joblib.load(model_path)


class BFRCXGBModel:
    """BFRC 抗折强度 XGBoost 预测模型封装。"""

    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = list(feature_columns)

        model_n_features = getattr(model, "n_features_in_", None)
        if model_n_features is not None and int(model_n_features) != len(self.feature_columns):
            raise ValueError(
                f"模型需要 {model_n_features} 个输入特征，但当前提供了 {len(self.feature_columns)} 个特征。"
            )

        self.model_feature_names = self._get_model_feature_names()

    def _get_model_feature_names(self):
        """尽可能获取模型保存的特征名；如果没有，则使用 Excel 中的固定顺序。"""
        if hasattr(self.model, "feature_names_in_"):
            return [str(x).strip() for x in self.model.feature_names_in_]

        if hasattr(self.model, "get_booster"):
            booster = self.model.get_booster()
            if getattr(booster, "feature_names", None):
                return [str(x).strip() for x in booster.feature_names]

        return self.feature_columns

    def predict(self, input_df: pd.DataFrame) -> float:
        """
        执行预测：
        1. 保持输入特征顺序与训练数据一致；
        2. 如果模型没有保存特征名，则使用 numpy 数组输入，避免 XGBoost 特征名不匹配。
        """
        input_map = {str(col).strip().lower(): col for col in input_df.columns}
        final_df = pd.DataFrame()

        for req_col in self.feature_columns:
            key = str(req_col).strip().lower()
            if key in input_map:
                final_df[req_col] = pd.to_numeric(input_df[input_map[key]], errors="coerce")
            else:
                final_df[req_col] = 0.0

        final_df = final_df.fillna(0.0).astype(float)

        if self.model_feature_names == self.feature_columns:
            prediction = self.model.predict(final_df.to_numpy(dtype=float))[0]
        else:
            model_df = final_df[self.model_feature_names]
            prediction = self.model.predict(model_df)[0]

        return float(prediction)


def make_input(label, default_value, feature_name):
    """统一创建数字输入框。"""
    cfg = INPUT_CONFIG.get(feature_name, {"step": 1.0, "format": "%.3f"})

    return st.number_input(
        label,
        min_value=0.0,
        value=float(default_value),
        step=float(cfg["step"]),
        format=cfg["format"]
    )


# ==========================================
# 5. 语言选择与标题
# ==========================================
col_blank, col_lang = st.columns([2.2, 1.8])

with col_lang:
    lang_option = st.radio(
        "语言选择",
        ["中文", "English"],
        index=0,
        horizontal=True,
        key="language_switch",
        label_visibility="collapsed"
    )

current_lang = "cn" if lang_option == "中文" else "en"
t = TRANSLATIONS[current_lang]

st.title(t["main_title"])
st.markdown(t["sub_title"])

# ==========================================
# 6. 加载数据集与模型
# ==========================================
data_status = True
model_status = True
data_msg = ""
model_msg = ""

try:
    data_info = load_dataset_info(DATA_PATH)
    feature_columns = data_info["feature_columns"]
    defaults = data_info["defaults"]
except Exception as e:
    data_status = False
    data_msg = str(e)
    feature_columns = FALLBACK_FEATURE_COLUMNS
    defaults = {
        "水泥 kg/m³": 400.0,
        "粉煤灰 kg/m³": 0.0,
        "硅灰 kg/m³": 0.0,
        "粗骨料 kg/m³": 1135.0,
        "细骨料 kg/m³": 671.0,
        "水 kg/m³": 170.0,
        "减水剂 kg/m³": 3.36,
        "纤维直径/mm": 0.015,
        "纤维长度/mm": 17.0,
        "纤维掺量/%": 0.10
    }

try:
    xgb_model = load_model(MODEL_PATH)
    predictor = BFRCXGBModel(xgb_model, feature_columns)
except Exception as e:
    model_status = False
    model_msg = str(e)
    predictor = None

if not data_status:
    st.warning(f"{t['data_fail']}{data_msg}")

if not model_status:
    st.error(f"{t['load_fail']}{model_msg}")

# ==========================================
# 7. GUI 输入表单
# ==========================================
with st.form("prediction_form"):
    st.markdown(t["sec1_title"])

    c1, c2, c3 = st.columns(3)

    with c1:
        cement = make_input(
            t[FEATURE_LABEL_KEYS["水泥 kg/m³"]],
            defaults["水泥 kg/m³"],
            "水泥 kg/m³"
        )

    with c2:
        fly_ash = make_input(
            t[FEATURE_LABEL_KEYS["粉煤灰 kg/m³"]],
            defaults["粉煤灰 kg/m³"],
            "粉煤灰 kg/m³"
        )

    with c3:
        silica_fume = make_input(
            t[FEATURE_LABEL_KEYS["硅灰 kg/m³"]],
            defaults["硅灰 kg/m³"],
            "硅灰 kg/m³"
        )

    st.markdown(t["sec2_title"])

    c4, c5, c6, c7 = st.columns(4)

    with c4:
        coarse_agg = make_input(
            t[FEATURE_LABEL_KEYS["粗骨料 kg/m³"]],
            defaults["粗骨料 kg/m³"],
            "粗骨料 kg/m³"
        )

    with c5:
        fine_agg = make_input(
            t[FEATURE_LABEL_KEYS["细骨料 kg/m³"]],
            defaults["细骨料 kg/m³"],
            "细骨料 kg/m³"
        )

    with c6:
        water = make_input(
            t[FEATURE_LABEL_KEYS["水 kg/m³"]],
            defaults["水 kg/m³"],
            "水 kg/m³"
        )

    with c7:
        water_reducer = make_input(
            t[FEATURE_LABEL_KEYS["减水剂 kg/m³"]],
            defaults["减水剂 kg/m³"],
            "减水剂 kg/m³"
        )

    st.markdown(t["sec3_title"])

    c8, c9, c10 = st.columns(3)

    with c8:
        fiber_diameter = make_input(
            t[FEATURE_LABEL_KEYS["纤维直径/mm"]],
            defaults["纤维直径/mm"],
            "纤维直径/mm"
        )

    with c9:
        fiber_length = make_input(
            t[FEATURE_LABEL_KEYS["纤维长度/mm"]],
            defaults["纤维长度/mm"],
            "纤维长度/mm"
        )

    with c10:
        fiber_content = make_input(
            t[FEATURE_LABEL_KEYS["纤维掺量/%"]],
            defaults["纤维掺量/%"],
            "纤维掺量/%"
        )

    submit_btn = st.form_submit_button(t["btn_predict"])

# ==========================================
# 8. 预测与结果导出
# ==========================================
if submit_btn:
    if not model_status or predictor is None:
        st.error(f"{t['load_fail']}{model_msg}")
    else:
        input_values = {
            "水泥 kg/m³": cement,
            "粉煤灰 kg/m³": fly_ash,
            "硅灰 kg/m³": silica_fume,
            "粗骨料 kg/m³": coarse_agg,
            "细骨料 kg/m³": fine_agg,
            "水 kg/m³": water,
            "减水剂 kg/m³": water_reducer,
            "纤维直径/mm": fiber_diameter,
            "纤维长度/mm": fiber_length,
            "纤维掺量/%": fiber_content
        }

        input_df = pd.DataFrame([
            {col: input_values[col] for col in feature_columns}
        ])

        try:
            prediction = predictor.predict(input_df)

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">{t['res_title']}</div>
                <div class="result-value">
                    {prediction:.4f}
                    <span class="result-unit">{t['res_unit']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            export_df = input_df.copy()
            export_df[TARGET_COLUMN + "_预测值"] = prediction
            export_df["Prediction_Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            export_df["Model_File"] = MODEL_PATH.name

            csv = export_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label=t["btn_download"],
                data=csv,
                file_name="BFRC_XGB_flexural_strength_prediction.csv",
                mime="text/csv"
            )

            with st.expander(t["input_summary"]):
                st.dataframe(export_df, use_container_width=True)

            with st.expander(t["feature_order"]):
                feature_order_df = pd.DataFrame({
                    "Order": list(range(1, len(feature_columns) + 1)),
                    "Feature": feature_columns
                })
                st.dataframe(feature_order_df, use_container_width=True)

        except Exception as e:
            st.error(f"{t['prediction_error']}{e}")
