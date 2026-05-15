import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mini Power BI + AutoML",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Mini Power BI + AutoML Platform")
st.markdown("Upload data, perform automated EDA, and train machine learning models.")

# ---------------- HELPER FUNCTIONS ----------------
def get_api_data(endpoint, method="GET", params=None, files=None):
    """Generic helper to call FastAPI backend."""
    try:
        if method == "GET":
            response = requests.get(
                f"{API_URL}{endpoint}",
                params=params,
                timeout=120
            )
        else:
            response = requests.post(
                f"{API_URL}{endpoint}",
                params=params,
                files=files,
                timeout=120
            )

        if response.ok:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to backend.\n\n"
            "Please start the FastAPI server first:\n"
            "`python -m uvicorn main:app --reload`"
        )
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        return None


def plot_missing_values(data):
    """Missing values bar chart."""
    missing_df = pd.DataFrame({
        "Column": list(data["missing"].keys()),
        "Missing": list(data["missing"].values()),
        "Missing %": list(data["missing_percent"].values())
    })

    if missing_df["Missing"].sum() == 0:
        st.success("✅ No missing values detected.")
        return

    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing",
        hover_data=["Missing %"],
        title="Missing Values by Column"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_matrix(corr_data):
    """Correlation heatmap."""
    if not corr_data:
        st.info("Not enough numeric columns to compute correlations.")
        return

    corr_df = pd.DataFrame(corr_data)

    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_outliers(outliers):
    """Outlier counts."""
    if not outliers:
        st.info("No numeric columns available.")
        return

    outlier_df = pd.DataFrame({
        "Column": list(outliers.keys()),
        "Outliers": list(outliers.values())
    })

    fig = px.bar(
        outlier_df,
        x="Column",
        y="Outliers",
        title="Outlier Count (IQR Method)"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_skewness(skewness):
    """Skewness chart."""
    if not skewness:
        st.info("No numeric columns available.")
        return

    skew_df = pd.DataFrame({
        "Column": list(skewness.keys()),
        "Skewness": list(skewness.values())
    })

    fig = px.bar(
        skew_df,
        x="Column",
        y="Skewness",
        title="Feature Skewness"
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Module",
    ["Upload", "Dashboard", "AutoML"]
)

# ===================== UPLOAD =====================
if menu == "Upload":
    st.header("📤 Upload CSV Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        st.info(
            f"Selected File: **{uploaded_file.name}** "
            f"({uploaded_file.size / 1024:.2f} KB)"
        )

        if st.button("Upload to API", type="primary"):
            result = get_api_data(
                "/upload",
                method="POST",
                files={"file": uploaded_file}
            )

            if result:
                st.success("✅ File uploaded successfully!")
                st.json(result)

# ===================== DASHBOARD =====================
elif menu == "Dashboard":
    st.header("📈 Automated Exploratory Data Analysis (EDA)")

    data = get_api_data("/summary")

    if data:
        if "error" in data:
            st.warning(data["error"])
        else:
            # ---------- KPI CARDS ----------
            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Rows", f"{data['rows']:,}")
            col2.metric("Columns", data["columns_count"])
            col3.metric("Missing Values", f"{data['total_missing_values']:,}")
            col4.metric("Duplicates", f"{data['duplicate_rows']:,}")
            col5.metric("Completeness", f"{data['completeness_percent']}%")

            st.metric("Memory Usage (MB)", data["memory_usage_mb"])

            # ---------- COLUMN OVERVIEW ----------
            st.subheader("📋 Column Overview")

            df_columns = pd.DataFrame({
                "Column": data["columns"],
                "Data Type": [data["dtypes"][c] for c in data["columns"]],
                "Missing": [data["missing"][c] for c in data["columns"]],
                "Missing %": [
                    data["missing_percent"][c] for c in data["columns"]
                ],
                "Unique Values": [
                    data["unique_counts"][c] for c in data["columns"]
                ]
            })

            st.dataframe(df_columns, use_container_width=True)

            # ---------- COLUMN TYPE SUMMARY ----------
            st.subheader("🧾 Column Type Summary")
            c1, c2, c3 = st.columns(3)

            c1.write("**Numeric Columns**")
            c1.write(data["numeric_columns"] or ["None"])

            c2.write("**Categorical Columns**")
            c2.write(data["categorical_columns"] or ["None"])

            c3.write("**Datetime Columns**")
            c3.write(data["datetime_columns"] or ["None"])

            # ---------- MISSING VALUES ----------
            st.subheader("🧩 Missing Value Analysis")
            plot_missing_values(data)

            # ---------- NUMERIC SUMMARY ----------
            if data["numeric_summary"]:
                st.subheader("📊 Numeric Statistics")
                numeric_df = pd.DataFrame(data["numeric_summary"])
                st.dataframe(
                    numeric_df.round(4),
                    use_container_width=True
                )

            # ---------- CORRELATION ----------
            st.subheader("🔗 Correlation Matrix")
            plot_correlation_matrix(data["correlation_matrix"])

            # ---------- OUTLIERS ----------
            st.subheader("🚨 Outlier Detection")
            plot_outliers(data["outliers"])

            # ---------- SKEWNESS ----------
            st.subheader("📐 Skewness Analysis")
            plot_skewness(data["skewness"])

            # ---------- CATEGORICAL SUMMARY ----------
            if data["categorical_summary"]:
                st.subheader("🏷️ Top Categories")

                for col_name, values in data["categorical_summary"].items():
                    st.markdown(f"### {col_name}")

                    cat_df = pd.DataFrame({
                        "Category": list(values.keys()),
                        "Count": list(values.values())
                    })

                    fig = px.bar(
                        cat_df,
                        x="Category",
                        y="Count",
                        title=f"Top Values in {col_name}"
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

# ===================== AUTOML =====================
elif menu == "AutoML":
    st.header("🤖 Automated Machine Learning")

    summary = get_api_data("/summary")

    if summary:
        if "error" in summary:
            st.warning(summary["error"])
        else:
            # ---------- MODEL CONFIGURATION ----------
            st.subheader("⚙️ Model Configuration")

            col1, col2 = st.columns(2)

            with col1:
                target = st.selectbox(
                    "🎯 Select Dependent Variable (Target)",
                    summary["columns"]
                )

            with col2:
                feature_candidates = [
                    col for col in summary["columns"]
                    if col != target
                ]

                selected_features = st.multiselect(
                    "📌 Select Independent Variables (Features)",
                    feature_candidates,
                    default=feature_candidates
                )

            if not selected_features:
                st.warning(
                    "Please select at least one independent variable."
                )
            else:
                # ---------- MODEL INFORMATION ----------
                with st.expander("🧠 Models to be Evaluated", expanded=False):
                    st.markdown(
                        """
                        **Classification Models**
                        - Logistic Regression
                        - Random Forest Classifier
                        - Decision Tree Classifier
                        - Gradient Boosting Classifier

                        **Regression Models**
                        - Linear Regression
                        - Random Forest Regressor
                        - Decision Tree Regressor
                        - Gradient Boosting Regressor

                        The system automatically detects whether your target
                        is categorical or numeric, trains all relevant models,
                        compares their performance, and selects the best model.
                        """
                    )

                # ---------- TRAIN BUTTON ----------
                if st.button("🚀 Run AutoML", type="primary"):
                    with st.spinner(
                        "Training multiple machine learning models..."
                    ):
                        result = get_api_data(
                            "/train",
                            method="POST",
                            params={
                                "target": target,
                                "features": ",".join(selected_features)
                            }
                        )

                    if result:
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success(
                                "✅ AutoML completed successfully!"
                            )

                            # ---------- TOP KPI CARDS ----------
                            metric_label = result.get(
                                "metric_name",
                                "score"
                            ).replace("_", " ").title()

                            best_score = result.get(
                                "best_score",
                                result.get("accuracy",
                                           result.get("r2_score", 0))
                            )

                            c1, c2, c3, c4 = st.columns(4)

                            c1.metric(
                                "🏆 Best Model",
                                result.get(
                                    "best_model",
                                    "Unknown"
                                )
                            )

                            c2.metric(
                                "📈 Problem Type",
                                result.get(
                                    "problem_type",
                                    "Unknown"
                                ).title()
                            )

                            c3.metric(
                                f"🎯 Best {metric_label}",
                                f"{best_score:.4f}"
                                if isinstance(
                                    best_score,
                                    (int, float)
                                )
                                else best_score
                            )

                            c4.metric(
                                "📊 Features Used",
                                len(selected_features)
                            )

                            # ---------- DATA SPLIT INFO ----------
                            c5, c6 = st.columns(2)
                            c5.metric(
                                "Train Rows",
                                result.get("train_rows", 0)
                            )
                            c6.metric(
                                "Test Rows",
                                result.get("test_rows", 0)
                            )

                            # ---------- MODEL RECOMMENDATION ----------
                            st.subheader("💡 Recommended Model")

                            st.info(
                                f"""
                                **{result.get('best_model', 'Unknown')}**
                                was automatically selected because it achieved
                                the highest
                                **{metric_label.lower()}**
                                of
                                **{best_score:.4f}**.
                                """
                            )

                            # ---------- MODEL COMPARISON ----------
                            all_models = result.get("all_models", {})

                            if all_models:
                                st.subheader(
                                    "📊 Model Performance Comparison"
                                )

                                comparison_df = pd.DataFrame({
                                    "Model": list(all_models.keys()),
                                    metric_label: list(
                                        all_models.values()
                                    )
                                }).sort_values(
                                    metric_label,
                                    ascending=False
                                )

                                st.dataframe(
                                    comparison_df,
                                    use_container_width=True
                                )

                                fig = px.bar(
                                    comparison_df,
                                    x="Model",
                                    y=metric_label,
                                    title=(
                                        f"Best Model Comparison "
                                        f"({metric_label})"
                                    )
                                )

                                st.plotly_chart(
                                    fig,
                                    use_container_width=True
                                )

                            # ---------- FEATURE LIST ----------
                            st.subheader("🧾 Selected Features")
                            st.write(selected_features)

                            # ---------- RAW JSON ----------
                            with st.expander(
                                "🔍 Detailed API Response"
                            ):
                                st.json(result)