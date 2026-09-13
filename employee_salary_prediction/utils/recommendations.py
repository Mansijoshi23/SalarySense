HIGH_VALUE_SKILLS = ["Machine Learning", "Cloud Computing", "Python", "Leadership", "Data Analysis"]

EDUCATION_RANK = {
    "High School": 1,
    "Diploma": 2,
    "Bachelor's": 3,
    "Master's": 4,
    "PhD": 5,
}

def build_skill_suggestions(current_skills):
    missing_high_value = [skill for skill in HIGH_VALUE_SKILLS if skill not in current_skills]
    suggestions = []
    for skill in missing_high_value[:3]:
        suggestions.append(f"Adding {skill} to your skill set is associated with higher predicted salaries in this dataset.")
    if not missing_high_value:
        suggestions.append("Your current skill set already covers the strongest salary-associated skills in this dataset.")
    return suggestions

def build_experience_observation(years_experience):
    if years_experience < 2:
        return "You are early in your career. Predicted salary typically rises quickly during the first five years of experience."
    if years_experience < 5:
        return "You are in a phase where experience tends to add predicted salary steadily year over year."
    if years_experience < 10:
        return "At this experience level, specialization and leadership responsibility tend to matter more than additional years alone."
    return "At this experience level, predicted salary growth is more closely tied to seniority, scope, and managerial responsibility than years alone."

def build_education_observation(education_level):
    rank = EDUCATION_RANK.get(education_level, 3)
    if rank <= 2:
        return "Pursuing a bachelor's degree or equivalent professional certification is associated with a meaningful salary increase in this dataset."
    if rank == 3:
        return "A postgraduate qualification such as a master's degree shows a further salary association in this dataset, particularly in specialized roles."
    return "Your education level is already associated with the top salary bracket in this dataset."

def build_managerial_observation(is_managerial, years_experience):
    if is_managerial:
        return "Managerial responsibility is associated with a noticeable salary premium in this dataset."
    if years_experience >= 6:
        return "With your experience level, moving into a managerial or team-lead role could be associated with a further salary increase."
    return None

def build_career_suggestions(payload):
    suggestions = []
    suggestions.extend(build_skill_suggestions(payload.get("skills", [])))
    suggestions.append(build_experience_observation(payload.get("years_experience", 0)))
    suggestions.append(build_education_observation(payload.get("education_level", "Bachelor's")))
    managerial_note = build_managerial_observation(payload.get("is_managerial", 0), payload.get("years_experience", 0))
    if managerial_note:
        suggestions.append(managerial_note)
    return suggestions
