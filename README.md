# CV Analyzer API

An AI-powered CV Analyzer that extracts information from your CV and provides personalized career recommendations, including courses, certifications, job links, and career advice. Supports PDF, DOCX, TXT, and image files (PNG/JPG).

---

## Features

- Extract text from CVs in multiple formats (PDF, DOCX, TXT, PNG, JPG)
- Analyze education, skills, and experience
- Recommend relevant courses and certifications
- Suggest job opportunities with direct links
- Provide career advice and master's program recommendations
- Chat functionality for follow-up questions about your CV

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/shumail-hassan-khan/Job_recommender.git
cd Job_recommender

pip install -r requirements.txt

cd app
uvicorn main:app --reload --port 7777
