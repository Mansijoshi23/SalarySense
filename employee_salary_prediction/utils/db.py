import os
import sqlite3
from datetime import datetime

DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database.db")

def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection

def init_db():
    connection = get_connection()
    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            education_level TEXT NOT NULL,
            job_title TEXT NOT NULL,
            industry TEXT NOT NULL,
            years_experience REAL NOT NULL,
            location TEXT NOT NULL,
            employment_type TEXT NOT NULL,
            company_size TEXT NOT NULL,
            work_mode TEXT NOT NULL,
            previous_salary REAL NOT NULL,
            skills TEXT,
            certifications INTEGER NOT NULL,
            is_managerial INTEGER NOT NULL,
            predicted_salary REAL NOT NULL,
            salary_min REAL NOT NULL,
            salary_max REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    connection.commit()
    connection.close()

def create_user(name, email, password_hash):
    connection = get_connection()
    connection.execute(
        "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (name, email, password_hash, datetime.utcnow().isoformat()),
    )
    connection.commit()
    connection.close()

def get_user_by_email(email):
    connection = get_connection()
    row = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    connection.close()
    return row

def get_user_by_id(user_id):
    connection = get_connection()
    row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    connection.close()
    return row

def update_user_name(user_id, name):
    connection = get_connection()
    connection.execute("UPDATE users SET name = ? WHERE id = ?", (name, user_id))
    connection.commit()
    connection.close()

def update_user_password(user_id, password_hash):
    connection = get_connection()
    connection.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
    connection.commit()
    connection.close()

def insert_prediction(user_id, payload, predicted_salary, salary_min, salary_max):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO predictions (
            user_id, age, gender, education_level, job_title, industry, years_experience,
            location, employment_type, company_size, work_mode, previous_salary, skills,
            certifications, is_managerial, predicted_salary, salary_min, salary_max, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            payload["age"],
            payload["gender"],
            payload["education_level"],
            payload["job_title"],
            payload["industry"],
            payload["years_experience"],
            payload["location"],
            payload["employment_type"],
            payload["company_size"],
            payload["work_mode"],
            payload["previous_salary"],
            ", ".join(payload["skills"]),
            payload["certifications"],
            payload["is_managerial"],
            predicted_salary,
            salary_min,
            salary_max,
            datetime.utcnow().isoformat(),
        ),
    )
    connection.commit()
    prediction_id = connection.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    connection.close()
    return prediction_id

def get_predictions_for_user(user_id):
    connection = get_connection()
    rows = connection.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
    ).fetchall()
    connection.close()
    return rows

def get_prediction_by_id(prediction_id, user_id):
    connection = get_connection()
    row = connection.execute(
        "SELECT * FROM predictions WHERE id = ? AND user_id = ?", (prediction_id, user_id)
    ).fetchone()
    connection.close()
    return row

def delete_prediction(prediction_id, user_id):
    connection = get_connection()
    connection.execute("DELETE FROM predictions WHERE id = ? AND user_id = ?", (prediction_id, user_id))
    connection.commit()
    connection.close()
