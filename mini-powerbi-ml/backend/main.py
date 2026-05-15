from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)


app = FastAPI(title="Mini Power BI + AutoML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_STORE = {"df": None}

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))
    DATA_STORE["df"] = df
    return {
        "filename": file.filename,
        "rows": int(df.shape[0]),
        "columns": list(df.columns)
    }

# Replace your existing /summary endpoint with this enhanced EDA version

@app.get("/summary")
def summary():
    df = DATA_STORE["df"]

    if df is None:
        return {"error": "No dataset uploaded"}

    # Basic Information
    rows, cols = df.shape
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing = df.isnull().sum().to_dict()
    missing_percent = ((df.isnull().sum() / len(df)) * 100).round(2).to_dict()

    # Column Types
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    datetime_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()

    # Duplicate Rows
    duplicate_rows = int(df.duplicated().sum())

    # Memory Usage
    memory_usage_mb = round(
        df.memory_usage(deep=True).sum() / (1024 * 1024), 4
    )

    # Numeric Summary
    numeric_summary = {}
    if numeric_cols:
        numeric_summary = (
            df[numeric_cols]
            .describe()
            .fillna(0)
            .round(4)
            .to_dict()
        )

    # Correlation Matrix
    correlation_matrix = {}
    if len(numeric_cols) >= 2:
        correlation_matrix = (
            df[numeric_cols]
            .corr()
            .fillna(0)
            .round(4)
            .to_dict()
        )

    # Outlier Detection using IQR
    outliers = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        count = int(
            ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        )

        outliers[col] = count

    # Skewness
    skewness = {}
    if numeric_cols:
        skewness = df[numeric_cols].skew().fillna(0).round(4).to_dict()

    # Unique Values per Column
    unique_counts = df.nunique().to_dict()

    # Categorical Value Counts (Top 10)
    categorical_summary = {}
    for col in categorical_cols:
        categorical_summary[col] = (
            df[col]
            .astype(str)
            .value_counts()
            .head(10)
            .to_dict()
        )

    # Data Quality Metrics
    total_cells = rows * cols
    total_missing = int(df.isnull().sum().sum())
    completeness_percent = round(
        ((total_cells - total_missing) / total_cells) * 100, 2
    ) if total_cells > 0 else 0

    return {
        # Dataset Overview
        "shape": [rows, cols],
        "rows": rows,
        "columns_count": cols,
        "columns": list(df.columns),

        # Data Types
        "dtypes": dtypes,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,

        # Missing Values
        "missing": missing,
        "missing_percent": missing_percent,
        "total_missing_values": total_missing,

        # Data Quality
        "duplicate_rows": duplicate_rows,
        "completeness_percent": completeness_percent,
        "memory_usage_mb": memory_usage_mb,

        # Column Statistics
        "unique_counts": unique_counts,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,

        # Advanced EDA
        "correlation_matrix": correlation_matrix,
        "outliers": outliers,
        "skewness": skewness
    }

@app.post("/train")
def train(target: str, features: str = None):
    df = DATA_STORE["df"]

    if df is None:
        return {"error": "No dataset uploaded"}

    if target not in df.columns:
        return {"error": f"Target '{target}' not found"}

    # ---------------- FEATURE SELECTION ----------------
    if features:
        feature_list = [
            col.strip()
            for col in features.split(",")
            if col.strip()
        ]
    else:
        feature_list = [col for col in df.columns if col != target]

    if not feature_list:
        return {"error": "No features selected"}

    missing_features = [
        col for col in feature_list if col not in df.columns
    ]
    if missing_features:
        return {
            "error": (
                "These features were not found: "
                + ", ".join(missing_features)
            )
        }

    # ---------------- PREPARE DATA ----------------
    selected_columns = feature_list + [target]
    work = df[selected_columns].dropna(subset=[target]).copy()

    if len(work) < 10:
        return {
            "error": (
                "Not enough rows to train a model. "
                "At least 10 non-null target rows are required."
            )
        }

    X = work[feature_list]
    y = work[target]

    # ---------------- DETECT PROBLEM TYPE ----------------
    is_classification = (
        y.dtype == "object"
        or str(y.dtype).startswith("category")
        or y.nunique() <= 20
    )

    # ---------------- PREPROCESSING ----------------
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_cols = [
        col for col in X.columns
        if col not in categorical_cols
    ]

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_cols
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]),
                categorical_cols
            )
        )

    if not transformers:
        return {"error": "No valid feature columns found"}

    preprocessor = ColumnTransformer(transformers=transformers)

    # ---------------- TRAIN TEST SPLIT ----------------
    stratify = y if is_classification and y.nunique() > 1 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    # ---------------- DEFINE MODELS ----------------
    if is_classification:
        models = {
            "Linear Regression": LinearRegression(),
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Decision Tree Classifier": DecisionTreeClassifier(
                random_state=42
            ),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),
            "Gradient Boosting Classifier": (
                GradientBoostingClassifier(random_state=42)
            ),
        }

        metric_name = "accuracy"

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(
                random_state=42
            ),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),
            "Gradient Boosting Regressor": (
                GradientBoostingRegressor(random_state=42)
            ),
        }

        metric_name = "r2_score"

    # ---------------- TRAIN ALL MODELS ----------------
    all_models = {}
    best_model_name = None
    best_score = float("-inf")
    best_pipeline = None

    for model_name, model in models.items():
        try:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            if is_classification:
                score = accuracy_score(y_test, predictions)
            else:
                score = r2_score(y_test, predictions)

            score = round(float(score), 4)
            all_models[model_name] = score

            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_pipeline = pipeline

        except Exception as e:
            # Skip models that fail on specific datasets
            all_models[model_name] = f"Failed: {str(e)}"

    # ---------------- FINAL VALIDATION ----------------
    if best_model_name is None:
        return {
            "error": (
                "All models failed to train. "
                "Please review your dataset."
            )
        }

    # Optional: store the best model for future prediction use
    DATA_STORE["model"] = best_pipeline
    DATA_STORE["features"] = feature_list
    DATA_STORE["target"] = target

    # ---------------- RESPONSE ----------------
    return {
        "best_model": best_model_name,
        "problem_type": (
            "classification"
            if is_classification
            else "regression"
        ),
        "best_score": round(float(best_score), 4),
        "metric_name": metric_name,
        "all_models": all_models,
        "selected_features": feature_list,
        "target": target,
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    }

