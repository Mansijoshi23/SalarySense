import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUM_RECORDS = 6000

np.random.seed(RANDOM_SEED)

JOB_TITLES = {
    "Software Engineer": 900000,
    "Senior Software Engineer": 1600000,
    "Data Scientist": 1300000,
    "Data Analyst": 650000,
    "Product Manager": 1800000,
    "Project Manager": 1100000,
    "Business Analyst": 750000,
    "Marketing Manager": 1050000,
    "Sales Executive": 550000,
    "Sales Manager": 1200000,
    "HR Manager": 1000000,
    "HR Executive": 450000,
    "Financial Analyst": 800000,
    "Accountant": 550000,
    "Operations Manager": 1100000,
    "Customer Support Executive": 350000,
    "Graphic Designer": 500000,
    "UX Designer": 950000,
    "DevOps Engineer": 1500000,
    "QA Engineer": 700000,
    "Content Writer": 450000,
    "Research Scientist": 1400000,
    "Administrative Assistant": 320000,
    "Executive Director": 3500000,
}

INDUSTRIES = {
    "Information Technology": 1.18,
    "Finance": 1.15,
    "Healthcare": 1.05,
    "Manufacturing": 0.95,
    "Retail": 0.88,
    "Education": 0.82,
    "Consulting": 1.10,
    "Telecommunications": 1.02,
    "Government": 0.90,
    "Media": 0.94,
}

LOCATIONS = {
    "Bangalore": 1.12,
    "Mumbai": 1.15,
    "Delhi NCR": 1.10,
    "Hyderabad": 1.05,
    "Pune": 1.00,
    "Chennai": 0.98,
    "Kolkata": 0.90,
    "Ahmedabad": 0.85,
    "Jaipur": 0.82,
    "Remote (India)": 0.95,
}

EDUCATION_LEVELS = {
    "High School": 0.75,
    "Diploma": 0.85,
    "Bachelor's": 1.00,
    "Master's": 1.20,
    "PhD": 1.38,
}

EMPLOYMENT_TYPES = {
    "Full-time": 1.00,
    "Part-time": 0.55,
    "Contract": 0.92,
    "Freelance": 0.85,
}

COMPANY_SIZES = {
    "Startup (1-50)": 0.88,
    "Small (51-200)": 0.95,
    "Medium (201-1000)": 1.05,
    "Large (1000-5000)": 1.15,
    "Enterprise (5000+)": 1.25,
}

WORK_MODES = {
    "On-site": 1.00,
    "Hybrid": 1.03,
    "Remote": 1.02,
}

ALL_SKILLS = [
    "Python", "Java", "C++", "SQL", "Machine Learning", "Data Analysis",
    "Cloud Computing", "Project Management", "Communication", "Leadership",
    "Excel", "JavaScript",
]

SKILL_PREMIUM = {
    "Python": 0.06,
    "Java": 0.04,
    "C++": 0.04,
    "SQL": 0.03,
    "Machine Learning": 0.09,
    "Data Analysis": 0.05,
    "Cloud Computing": 0.07,
    "Project Management": 0.05,
    "Communication": 0.02,
    "Leadership": 0.06,
    "Excel": 0.01,
    "JavaScript": 0.04,
}

GENDERS = ["Male", "Female", "Other"]

def sample_skills():
    count = np.random.randint(2, 6)
    chosen = np.random.choice(ALL_SKILLS, size=count, replace=False)
    return list(chosen)

def build_record():
    job_title = np.random.choice(list(JOB_TITLES.keys()))
    base_salary = JOB_TITLES[job_title]

    industry = np.random.choice(list(INDUSTRIES.keys()))
    location = np.random.choice(list(LOCATIONS.keys()))
    education = np.random.choice(list(EDUCATION_LEVELS.keys()), p=[0.08, 0.12, 0.45, 0.28, 0.07])
    employment_type = np.random.choice(list(EMPLOYMENT_TYPES.keys()), p=[0.82, 0.06, 0.08, 0.04])
    company_size = np.random.choice(list(COMPANY_SIZES.keys()))
    work_mode = np.random.choice(list(WORK_MODES.keys()))
    gender = np.random.choice(GENDERS, p=[0.56, 0.42, 0.02])
    is_managerial = 1 if "Manager" in job_title or "Director" in job_title else np.random.choice([0, 1], p=[0.85, 0.15])

    experience = max(0, np.random.gamma(shape=2.2, scale=3.2))
    experience = round(min(experience, 35), 1)
    age = int(np.clip(22 + experience + np.random.normal(3, 4), 21, 65))

    skills = sample_skills()
    certifications = np.random.randint(0, 4)

    experience_multiplier = 1 + (experience * 0.028)
    education_multiplier = EDUCATION_LEVELS[education]
    industry_multiplier = INDUSTRIES[industry]
    location_multiplier = LOCATIONS[location]
    employment_multiplier = EMPLOYMENT_TYPES[employment_type]
    company_multiplier = COMPANY_SIZES[company_size]
    work_mode_multiplier = WORK_MODES[work_mode]
    managerial_multiplier = 1.18 if is_managerial else 1.0
    certification_multiplier = 1 + (certifications * 0.02)
    skill_multiplier = 1 + sum(SKILL_PREMIUM.get(skill, 0) for skill in skills) * 0.5

    salary = (
        base_salary
        * experience_multiplier
        * education_multiplier
        * industry_multiplier
        * location_multiplier
        * employment_multiplier
        * company_multiplier
        * work_mode_multiplier
        * managerial_multiplier
        * certification_multiplier
        * skill_multiplier
    )

    noise = np.random.normal(1.0, 0.07)
    salary = max(18000, salary * noise)

    previous_experience = max(0, experience - np.random.uniform(0.5, 2.5))
    previous_experience_multiplier = 1 + (previous_experience * 0.028)
    previous_noise = np.random.normal(0.88, 0.16)
    previous_salary = (
        base_salary
        * previous_experience_multiplier
        * education_multiplier
        * industry_multiplier
        * location_multiplier
        * employment_multiplier
        * managerial_multiplier
        * previous_noise
    )
    previous_salary = max(15000, min(previous_salary, salary * 0.99))

    return {
        "age": age,
        "gender": gender,
        "education_level": education,
        "job_title": job_title,
        "industry": industry,
        "years_experience": experience,
        "location": location,
        "employment_type": employment_type,
        "company_size": company_size,
        "work_mode": work_mode,
        "previous_salary": round(previous_salary, 2),
        "skills": ", ".join(skills),
        "certifications": certifications,
        "is_managerial": is_managerial,
        "salary": round(salary, 2),
    }

def main():
    records = [build_record() for _ in range(NUM_RECORDS)]
    dataframe = pd.DataFrame(records)

    duplicate_rows = dataframe.sample(frac=0.01, random_state=RANDOM_SEED)
    dataframe = pd.concat([dataframe, duplicate_rows], ignore_index=True)

    missing_indices = dataframe.sample(frac=0.015, random_state=RANDOM_SEED + 1).index
    dataframe.loc[missing_indices, "previous_salary"] = np.nan

    missing_indices_2 = dataframe.sample(frac=0.01, random_state=RANDOM_SEED + 2).index
    dataframe.loc[missing_indices_2, "years_experience"] = np.nan

    dataframe.to_csv("employee_salary.csv", index=False)
    print(f"Generated {len(dataframe)} records at employee_salary.csv")

if __name__ == "__main__":
    main()
