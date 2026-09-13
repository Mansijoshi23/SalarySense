import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = ["age", "years_experience", "previous_salary", "certifications", "is_managerial"]
CATEGORICAL_FEATURES = [
    "gender",
    "education_level",
    "job_title",
    "industry",
    "location",
    "employment_type",
    "company_size",
    "work_mode",
]
SKILL_COLUMN = "skills"
ALL_SKILLS = [
    "Python", "Java", "C++", "SQL", "Machine Learning", "Data Analysis",
    "Cloud Computing", "Project Management", "Communication", "Leadership",
    "Excel", "JavaScript",
]
TARGET_COLUMN = "salary"

def split_skills(raw_value):
    if pd.isna(raw_value) or not str(raw_value).strip():
        return []
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]

def add_skill_columns(dataframe):
    dataframe = dataframe.copy()
    parsed_skills = dataframe[SKILL_COLUMN].apply(split_skills)
    for skill in ALL_SKILLS:
        column_name = f"skill_{skill.lower().replace(' ', '_').replace('+', 'p')}"
        dataframe[column_name] = parsed_skills.apply(lambda skills: 1 if skill in skills else 0)
    dataframe["skill_count"] = parsed_skills.apply(len)
    return dataframe

def get_engineered_feature_names():
    return [f"skill_{skill.lower().replace(' ', '_').replace('+', 'p')}" for skill in ALL_SKILLS] + ["skill_count"]

def clean_raw_dataframe(dataframe):
    dataframe = dataframe.copy()
    dataframe = dataframe.drop_duplicates()
    dataframe = dataframe[dataframe[TARGET_COLUMN].notna()]
    dataframe = dataframe[dataframe[TARGET_COLUMN] > 0]
    dataframe = dataframe[(dataframe["age"] >= 18) & (dataframe["age"] <= 75)]
    dataframe = dataframe[dataframe["years_experience"].isna() | (dataframe["years_experience"] >= 0)]
    dataframe = dataframe[dataframe["previous_salary"].isna() | (dataframe["previous_salary"] >= 0)]
    for column in CATEGORICAL_FEATURES:
        dataframe[column] = dataframe[column].astype(str).str.strip()
    lower_bound = dataframe[TARGET_COLUMN].quantile(0.005)
    upper_bound = dataframe[TARGET_COLUMN].quantile(0.995)
    dataframe = dataframe[(dataframe[TARGET_COLUMN] >= lower_bound) & (dataframe[TARGET_COLUMN] <= upper_bound)]
    return dataframe.reset_index(drop=True)

def build_feature_columns():
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + get_engineered_feature_names()

def build_preprocessing_pipeline():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    engineered_columns = get_engineered_feature_names()
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ("engineered", SimpleImputer(strategy="constant", fill_value=0), engineered_columns),
    ])
    return preprocessor

def prepare_features_and_target(dataframe):
    dataframe = clean_raw_dataframe(dataframe)
    dataframe = add_skill_columns(dataframe)
    feature_columns = build_feature_columns()
    features = dataframe[feature_columns]
    target = dataframe[TARGET_COLUMN]
    return features, target

def build_input_row(form_data):
    row = {
        "age": float(form_data.get("age")),
        "gender": form_data.get("gender"),
        "education_level": form_data.get("education_level"),
        "job_title": form_data.get("job_title"),
        "industry": form_data.get("industry"),
        "years_experience": float(form_data.get("years_experience")),
        "location": form_data.get("location"),
        "employment_type": form_data.get("employment_type"),
        "company_size": form_data.get("company_size"),
        "work_mode": form_data.get("work_mode"),
        "previous_salary": float(form_data.get("previous_salary")),
        "skills": ", ".join(form_data.get("skills", [])),
        "certifications": int(form_data.get("certifications", 0)),
        "is_managerial": 1 if form_data.get("is_managerial") in ("1", "true", "on", True, 1) else 0,
    }
    dataframe = pd.DataFrame([row])
    dataframe = add_skill_columns(dataframe)
    feature_columns = build_feature_columns()
    return dataframe[feature_columns]
