"""
CV Analyzer Agent - Analyzes CVs and provides career recommendations
Supports: PDF, DOCX, TXT, PNG (with OCR)
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from fastapi.responses import HTMLResponse

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.db.sqlite import SqliteDb


# Add database for session storage
db = SqliteDb(db_file="tmp/cv_analyzer.db")

# ============== Structured Output Models ==============

class Course(BaseModel):
    name: str = Field(..., description="Course name")
    platform: str = Field(..., description="Platform (Udemy, Coursera, etc.)")
    url: str = Field(..., description="Direct link to the course")

class Certification(BaseModel):
    name: str = Field(..., description="Certification name")
    provider: str = Field(..., description="Certification provider")
    url: str = Field(..., description="Link to certification info")

class JobRecommendation(BaseModel):
    title: str = Field(..., description="Job title")
    platform: str = Field(..., description="Job platform (LinkedIn, Indeed, etc.)")
    url: str = Field(..., description="Direct job search link")

class MasterProgram(BaseModel):
    name: str = Field(..., description="Master's program name")
    field: str = Field(..., description="Field of study")
    description: str = Field(..., description="Brief description")

class CVAnalysisResult(BaseModel):
    candidate_name: str = Field(..., description="Name of the candidate")
    current_education: str = Field(..., description="Current education status")
    education_status: str = Field(..., description="'ongoing' or 'completed'")
    graduation_year: Optional[str] = Field(None, description="Expected/actual graduation year")
    field_of_study: str = Field(..., description="Field of study (e.g., Computer Science)")
    
    skills: List[str] = Field(default_factory=list, description="List of skills from CV")
    experience_summary: str = Field(..., description="Brief summary of experience")
    
    recommended_courses: List[Course] = Field(default_factory=list, description="Recommended courses")
    recommended_certifications: List[Certification] = Field(default_factory=list, description="Recommended certifications")
    recommended_jobs: List[JobRecommendation] = Field(default_factory=list, description="Job recommendations with links")
    recommended_masters: List[MasterProgram] = Field(default_factory=list, description="Master's programs if applicable")
    
    additional_resources: List[str] = Field(default_factory=list, description="Additional learning platforms/resources")
    career_advice: str = Field(..., description="Personalized career advice")

class CVChatRequest(BaseModel):
    session_id: str
    user_message: str


# ============== Text Extraction Functions ==============

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    import pypdf
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    from docx import Document
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_image(file_path: str) -> str:
    """Extract text from image using OCR."""
    import pytesseract
    from PIL import Image
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)

def extract_text(file_path: str) -> str:
    """Extract text based on file extension."""
    ext = Path(file_path).suffix.lower()
    extractors = {
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
        ".txt": extract_text_from_txt,
        ".png": extract_text_from_image,
        ".jpg": extract_text_from_image,
        ".jpeg": extract_text_from_image,
    }
    if ext not in extractors:
        raise ValueError(f"Unsupported file type: {ext}")
    return extractors[ext](file_path)


# ============== CV Analyzer Agent ==============

cv_analyzer = Agent(
    name="CV Career Advisor",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    description="Expert CV analyzer and career advisor",
    db=db,  
    add_history_to_context=True,  
    num_history_runs=5,  
    instructions=f"""
    Today's date is: {datetime.now().strftime("%B %d, %Y")}
    
    You are an expert CV analyzer and career advisor. Analyze the CV content provided and return structured recommendations.
    
    RULES:
    1. CHECK EDUCATION STATUS:
       - Compare graduation year to today's date
       - If graduation year <= current year: education_status = "completed"
       - If graduation year > current year: education_status = "ongoing"

    2. CHECK EDUCATION STATUS:
       - If Bachelor's is ongoing OR graduation year is 2027 or later: Focus on courses & certifications
       - If Bachelor's is completed: Recommend Master's programs
    
    3. PROVIDE REAL, WORKING LINKS:
       - LinkedIn Jobs: https://www.linkedin.com/jobs/search/?keywords={{encoded_field}}
       - Indeed: https://www.indeed.com/jobs?q={{encoded_field}}
       - Google Jobs: https://www.google.com/search?q={{encoded_field}}+jobs&ibp=htl;jobs
       - Udemy: https://www.udemy.com/courses/search/?q={{encoded_topic}}
       - Coursera: https://www.coursera.org/search?query={{encoded_topic}}
    
    4. FOR CYBER SECURITY CANDIDATES:
       - TryHackMe: https://tryhackme.com
       - HackTheBox: https://www.hackthebox.com
       - CompTIA: https://www.comptia.org/certifications/security
       - SANS: https://www.sans.org/cyber-security-courses/
    
    5. Search the web for current, relevant courses and job opportunities.
    
    6. All URLs must be properly formatted and clickable.

    7. CERTIFICATIONS ARE MANDATORY:
        - Always return at least 3 certifications relevant to the candidate's skills
        - Certifications must be industry-recognized
        - Do NOT leave recommended_certifications empty
    """,
    output_schema=CVAnalysisResult,
)


# ============== FastAPI App ==============

app = FastAPI(
    title="CV Analyzer API",
    description="Upload your CV and get personalized career recommendations",
    version="1.0.0"
)


import uuid

# Store CV text temporarily in memory (for the session)
cv_storage: dict = {}

@app.post("/analyze-cv")
async def analyze_cv(
    file: UploadFile = File(..., description="CV file (PDF, DOCX, TXT, PNG)")
):
    """
    Upload a CV file and get structured career recommendations.
    
    Supported formats: .pdf, .docx, .txt, .png, .jpg, .jpeg
    """
    # Generate a unique session_id for this CV
    session_id = str(uuid.uuid4())
    
    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix.lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Extract text from CV
        cv_text = extract_text(tmp_path)
        
        # Store CV text for this session
        cv_storage[session_id] = cv_text
        
        # Create prompt with CV content
        prompt = f"""
        Analyze this CV and provide comprehensive career recommendations:
        
        --- CV CONTENT ---
        {cv_text}
        --- END CV ---
        
        Provide:
        1. Education status analysis (ongoing vs completed, graduation year)
        2. Relevant courses with direct Udemy/Coursera links
        3. Certifications appropriate for their field
        4. Job search links for LinkedIn, Indeed, and Google Jobs
        5. Master's program recommendations if education is completed
        6. Personalized career advice
        
        Use web search to find current, relevant opportunities.
        """
        
        response = cv_analyzer.run(prompt, stream=False, session_id=session_id)
        
        # response.content is already a Pydantic model, convert to dict
        if isinstance(response.content, CVAnalysisResult):
            return {
                "session_id": session_id,
                **response.content.model_dump()
            }
        else:
            return {"session_id": session_id, "analysis": response.content}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@app.post("/chat-cv")
async def chat_about_cv(request: CVChatRequest):
    """
    Chat with the CV advisor about the analyzed CV.
    Use the session_id returned from /analyze-cv
    """
    try:
        session_id = request.session_id
        user_message = request.user_message

        # Get stored CV text for context
        cv_text = cv_storage.get(session_id, "")
        if not cv_text:
            return JSONResponse(
                status_code=400,
                content={"error": "Session expired or invalid session_id"}
            )

        prompt = f"""
        CV CONTENT:
        {cv_text}

        USER QUESTION:
        {user_message}

        Respond clearly, professionally, and with actionable advice.
        """

        response = cv_analyzer.run(prompt, stream=False, session_id=session_id)

        # Return the reply as string (or structured if needed)
        return {"reply": response.content}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "CV Analyzer API"}

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7777, reload=True)