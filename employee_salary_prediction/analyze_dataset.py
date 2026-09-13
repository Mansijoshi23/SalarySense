import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

DATA_PATH = os.path.join("data", "employee_salary.csv")
OUTPUT_DIR = os.path.join("static", "images", "charts")

def load_clean_dataframe():
    dataframe = pd.read_csv(DATA_PATH)
    dataframe = dataframe.dropna(subset=["salary"])
    return dataframe

def save_salary_distribution(dataframe):
    plt.figure(figsize=(8, 5))
    sns.histplot(dataframe["salary"], bins=40, color="#10233F")
    plt.title("Salary Distribution")
    plt.xlabel("Annual Salary")
    plt.ylabel("Number of Employees")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "salary_distribution.png"), dpi=150)
    plt.close()

def save_salary_vs_experience(dataframe):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=dataframe, x="years_experience", y="salary", alpha=0.35, color="#3E5C76")
    plt.title("Salary vs Years of Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Annual Salary")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "salary_vs_experience.png"), dpi=150)
    plt.close()

def save_salary_by_education(dataframe):
    order = dataframe.groupby("education_level")["salary"].median().sort_values().index
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=dataframe, x="education_level", y="salary", order=order, color="#C08A28")
    plt.title("Salary by Education Level")
    plt.xlabel("Education Level")
    plt.ylabel("Annual Salary")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "salary_by_education.png"), dpi=150)
    plt.close()

def save_top_job_titles(dataframe):
    top_titles = dataframe.groupby("job_title")["salary"].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=top_titles.values, y=top_titles.index, color="#10233F")
    plt.title("Top 10 Job Titles by Average Salary")
    plt.xlabel("Average Annual Salary")
    plt.ylabel("Job Title")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_job_titles.png"), dpi=150)
    plt.close()

def save_salary_by_gender(dataframe):
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=dataframe, x="gender", y="salary", color="#3E5C76")
    plt.title("Salary by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Annual Salary")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "salary_by_gender.png"), dpi=150)
    plt.close()

def save_experience_distribution(dataframe):
    plt.figure(figsize=(8, 5))
    sns.histplot(dataframe["years_experience"].dropna(), bins=30, color="#C08A28")
    plt.title("Years of Experience Distribution")
    plt.xlabel("Years of Experience")
    plt.ylabel("Number of Employees")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "experience_distribution.png"), dpi=150)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataframe = load_clean_dataframe()
    save_salary_distribution(dataframe)
    save_salary_vs_experience(dataframe)
    save_salary_by_education(dataframe)
    save_top_job_titles(dataframe)
    save_salary_by_gender(dataframe)
    save_experience_distribution(dataframe)
    print(f"Saved charts to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
