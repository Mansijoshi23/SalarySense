import json
import os
import re
from functools import wraps

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from utils import db
from utils.preprocessing import ALL_SKILLS, build_input_row, clean_raw_dataframe
from utils.recommendations import build_career_suggestions

app = Flask(__name__)
app.secret_key = os.environ.get("SALARYSENSE_SECRET_KEY", "salarysense-development-secret-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "salary_prediction_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "employee_salary.csv")

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

EDUCATION_OPTIONS = ["High School", "Diploma", "Bachelor's", "Master's", "PhD"]
EMPLOYMENT_TYPE_OPTIONS = ["Full-time", "Part-time", "Contract", "Freelance"]
WORK_MODE_OPTIONS = ["On-site", "Hybrid", "Remote"]
GENDER_OPTIONS = ["Male", "Female", "Other"]

model_pipeline = None
model_metrics = {}
job_title_options = []
industry_options = []
location_options = []
company_size_options = []
job_title_averages = {}
overall_average_salary = 0
best_model_rmse = 12000
best_model_r2 = 0.9

def load_model_assets():
    global model_pipeline, model_metrics
    global job_title_options, industry_options, location_options, company_size_options
    global job_title_averages, overall_average_salary, best_model_rmse, best_model_r2

    model_pipeline = joblib.load(MODEL_PATH)

    with open(METRICS_PATH) as metrics_file:
        model_metrics = json.load(metrics_file)

    best_model_name = model_metrics.get("best_model")
    best_metrics = model_metrics.get("models", {}).get(best_model_name, {})
    best_model_rmse = best_metrics.get("rmse", 12000)
    best_model_r2 = best_metrics.get("r2", 0.9)

    raw_dataframe = pd.read_csv(DATA_PATH)
    cleaned_dataframe = clean_raw_dataframe(raw_dataframe)

    job_title_options = sorted(cleaned_dataframe["job_title"].unique().tolist())
    industry_options = sorted(cleaned_dataframe["industry"].unique().tolist())
    location_options = sorted(cleaned_dataframe["location"].unique().tolist())
    company_size_options = sorted(cleaned_dataframe["company_size"].unique().tolist())

    job_title_averages = cleaned_dataframe.groupby("job_title")["salary"].mean().to_dict()
    overall_average_salary = float(cleaned_dataframe["salary"].mean())

def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)
    return wrapped_view

def get_current_user():
    if "user_id" not in session:
        return None
    return db.get_user_by_id(session["user_id"])

@app.context_processor
def inject_current_user():
    return {"current_user": get_current_user()}

def confidence_label(r2_score):
    if r2_score >= 0.9:
        return "High"
    if r2_score >= 0.75:
        return "Moderate"
    return "Indicative"

def validate_registration(name, email, password, confirm_password):
    errors = []
    if not name or len(name.strip()) < 2:
        errors.append("Please enter your full name.")
    if not email or not EMAIL_PATTERN.match(email):
        errors.append("Please enter a valid email address.")
    if not password or len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    elif not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        errors.append("Password must contain both letters and numbers.")
    if password != confirm_password:
        errors.append("Passwords do not match.")
    return errors

def validate_prediction_form(form):
    errors = {}
    try:
        age = int(form.get("age", ""))
        if age < 18 or age > 75:
            errors["age"] = "Age must be between 18 and 75."
    except ValueError:
        errors["age"] = "Please enter a valid age."

    try:
        experience = float(form.get("years_experience", ""))
        if experience < 0 or experience > 50:
            errors["years_experience"] = "Experience must be between 0 and 50 years."
    except ValueError:
        errors["years_experience"] = "Please enter valid years of experience."

    try:
        previous_salary = float(form.get("previous_salary", ""))
        if previous_salary < 0:
            errors["previous_salary"] = "Previous salary cannot be negative."
    except ValueError:
        errors["previous_salary"] = "Please enter a valid previous salary."

    required_fields = [
        "gender", "education_level", "job_title", "industry",
        "location", "employment_type", "company_size", "work_mode",
    ]
    for field in required_fields:
        if not form.get(field):
            errors[field] = "This field is required."

    return errors

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html", metrics=model_metrics)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = validate_registration(name, email, password, confirm_password)

        if not errors and db.get_user_by_email(email):
            errors.append("An account with this email already exists.")

        if errors:
            for error in errors:
                flash(error, "error")
            return render_template("register.html", name=name, email=email)

        password_hash = generate_password_hash(password)
        db.create_user(name, email, password_hash)
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", name="", email="")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember_me = request.form.get("remember_me") == "on"

        user = db.get_user_by_email(email)

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return render_template("login.html", email=email)

        session["user_id"] = user["id"]
        session.permanent = remember_me
        flash(f"Welcome back, {user['name']}.", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html", email="")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    user = get_current_user()
    predictions = db.get_predictions_for_user(user["id"])
    prediction_count = len(predictions)
    average_predicted_salary = (
        sum(row["predicted_salary"] for row in predictions) / prediction_count
        if prediction_count > 0 else 0
    )
    latest_prediction = predictions[0] if predictions else None
    profile_fields_completed = sum([bool(user["name"]), bool(user["email"])])
    profile_completion = int((profile_fields_completed / 2) * 100)

    return render_template(
        "dashboard.html",
        prediction_count=prediction_count,
        average_predicted_salary=average_predicted_salary,
        latest_prediction=latest_prediction,
        recent_predictions=predictions[:5],
        profile_completion=profile_completion,
    )

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        form = request.form
        selected_skills = request.form.getlist("skills")
        errors = validate_prediction_form(form)

        if errors:
            for message in errors.values():
                flash(message, "error")
            return render_template(
                "predict.html",
                job_titles=job_title_options,
                industries=industry_options,
                locations=location_options,
                company_sizes=company_size_options,
                education_levels=EDUCATION_OPTIONS,
                employment_types=EMPLOYMENT_TYPE_OPTIONS,
                work_modes=WORK_MODE_OPTIONS,
                genders=GENDER_OPTIONS,
                all_skills=ALL_SKILLS,
                form_data=form,
                selected_skills=selected_skills,
            )

        payload = {
            "age": int(form.get("age")),
            "gender": form.get("gender"),
            "education_level": form.get("education_level"),
            "job_title": form.get("job_title"),
            "industry": form.get("industry"),
            "years_experience": float(form.get("years_experience")),
            "location": form.get("location"),
            "employment_type": form.get("employment_type"),
            "company_size": form.get("company_size"),
            "work_mode": form.get("work_mode"),
            "previous_salary": float(form.get("previous_salary")),
            "skills": selected_skills,
            "certifications": int(form.get("certifications", 0)),
            "is_managerial": 1 if form.get("is_managerial") == "on" else 0,
        }

        input_row = build_input_row(payload)
        predicted_salary = float(model_pipeline.predict(input_row)[0])
        predicted_salary = max(predicted_salary, 0)

        salary_min = max(0, predicted_salary - best_model_rmse)
        salary_max = predicted_salary + best_model_rmse

        prediction_id = db.insert_prediction(
            session["user_id"], payload, predicted_salary, salary_min, salary_max
        )

        return redirect(url_for("view_result", prediction_id=prediction_id))

    return render_template(
        "predict.html",
        job_titles=job_title_options,
        industries=industry_options,
        locations=location_options,
        company_sizes=company_size_options,
        education_levels=EDUCATION_OPTIONS,
        employment_types=EMPLOYMENT_TYPE_OPTIONS,
        work_modes=WORK_MODE_OPTIONS,
        genders=GENDER_OPTIONS,
        all_skills=ALL_SKILLS,
        form_data={},
        selected_skills=[],
    )

@app.route("/result/<int:prediction_id>")
@login_required
def view_result(prediction_id):
    prediction = db.get_prediction_by_id(prediction_id, session["user_id"])
    if prediction is None:
        flash("Prediction not found.", "error")
        return redirect(url_for("history"))

    predicted_salary = prediction["predicted_salary"]
    job_average = job_title_averages.get(prediction["job_title"], overall_average_salary)
    difference_percentage = (
        ((predicted_salary - job_average) / job_average) * 100 if job_average else 0
    )

    top_factors = list(model_metrics.get("feature_importance", {}).items())[:5]

    payload_for_suggestions = {
        "skills": [skill.strip() for skill in (prediction["skills"] or "").split(",") if skill.strip()],
        "years_experience": prediction["years_experience"],
        "education_level": prediction["education_level"],
        "is_managerial": prediction["is_managerial"],
    }
    suggestions = build_career_suggestions(payload_for_suggestions)

    return render_template(
        "result.html",
        prediction=prediction,
        monthly_salary=predicted_salary / 12,
        top_factors=top_factors,
        job_average=job_average,
        difference_percentage=difference_percentage,
        suggestions=suggestions,
        confidence=confidence_label(best_model_r2),
        rmse=best_model_rmse,
        skills_list=payload_for_suggestions["skills"],
    )

@app.route("/history")
@login_required
def history():
    predictions = db.get_predictions_for_user(session["user_id"])
    return render_template("history.html", predictions=predictions)

@app.route("/history/delete/<int:prediction_id>", methods=["POST"])
@login_required
def delete_history_item(prediction_id):
    db.delete_prediction(prediction_id, session["user_id"])
    flash("Prediction record deleted.", "success")
    return redirect(url_for("history"))

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = get_current_user()

    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "update_profile":
            name = request.form.get("name", "").strip()
            if len(name) < 2:
                flash("Please enter a valid name.", "error")
            else:
                db.update_user_name(user["id"], name)
                flash("Profile updated successfully.", "success")
            return redirect(url_for("profile"))

        if form_type == "change_password":
            current_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirm_new_password = request.form.get("confirm_new_password", "")

            if not check_password_hash(user["password_hash"], current_password):
                flash("Current password is incorrect.", "error")
            elif len(new_password) < 8 or not re.search(r"[0-9]", new_password) or not re.search(r"[A-Za-z]", new_password):
                flash("New password must be at least 8 characters and include letters and numbers.", "error")
            elif new_password != confirm_new_password:
                flash("New passwords do not match.", "error")
            else:
                db.update_user_password(user["id"], generate_password_hash(new_password))
                flash("Password changed successfully.", "success")
            return redirect(url_for("profile"))

    predictions = db.get_predictions_for_user(user["id"])
    return render_template("profile.html", user=user, prediction_count=len(predictions))

@app.errorhandler(404)
def handle_not_found(error):
    return render_template("error.html", code=404, message="Page not found."), 404

@app.errorhandler(500)
def handle_server_error(error):
    return render_template("error.html", code=500, message="Something went wrong on our end."), 500

db.init_db()
load_model_assets()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
