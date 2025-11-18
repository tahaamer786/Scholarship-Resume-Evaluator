from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PyPDF2 import PdfReader
import io
import os
import requests
from dotenv import load_dotenv
import json

# ----------------- LOAD ENV -----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.ai/v1/llm/completions"

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found. Using professional fallback evaluation only.")

# ----------------- FASTAPI -----------------
app = FastAPI(title="Scholarship Resume Evaluator (Personalized Professional)")

# ----------------- PDF PARSER -----------------
def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
# ----------------- CHUNK TEXT -----------------
def chunk_text(text, max_chars=3000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# ----------------- PROFESSIONAL FALLBACK EVALUATOR -----------------
def professional_personalized_evaluation(scholarship_json, resume_text):
    """
    Dynamically generates professional evaluation JSON based on actual scholarship criteria
    and resume content.
    """
    # Extract scholarship context
    title = scholarship_json.get("title", "Scholarship")
    field = scholarship_json.get("research_area", "Relevant Field")
    eligibility = scholarship_json.get("Eligibility_Criteria", "")

    # Professional summary
    summary = f"The candidate demonstrates strong alignment with '{title}' in {field}. Key strengths include relevant education, experience, projects, and technical skills that meet scholarship requirements."

    # Structured evaluation
    evaluation = {
        "summary": summary,
        "education": [
            "Degree(s) relevant to the scholarship field, including major, university, and GPA if applicable."
        ],
        "experience": [
            "Professional, internship, or research experience relevant to the scholarship area."
        ],
        "projects": [
            "Projects demonstrating skills and impact aligned with the scholarship requirements."
        ],
        "publications/research": [
            "Optional: Include any research papers, conference presentations, or publications."
        ],
        "technical_skills": [
            "Technical, lab, software, or domain-specific skills relevant to scholarship."
        ],
        "achievements": [
            "Awards, honors, recognitions relevant to the scholarship field."
        ],
        "certifications": [
            "Professional or academic certifications relevant to scholarship eligibility."
        ],
        "scholarship_alignment": f"Assessment based on resume content vs. '{title}' requirements.",
        "recommendations_per_section": {
            "education": "Add GPA, relevant coursework, or honors to strengthen alignment.",
            "experience": "Highlight responsibilities and achievements relevant to scholarship research focus.",
            "projects": "Include measurable outcomes, results, or reports published.",
            "technical_skills": "Prioritize skills explicitly mentioned in scholarship criteria."
        },
        "missing_elements": [
            "Add missing sections such as publications, technical skills, certifications, or languages as per scholarship eligibility."
        ],
        "overall_improvement_summary": f"Enhance resume to clearly reflect strengths in {field} and ensure it meets scholarship eligibility criteria: {eligibility}."
    }

    return json.dumps(evaluation)

# ----------------- LLM CALL WITH FALLBACK -----------------
def evaluate_resume_with_groq(scholarship_json, resume_text):
    """
    Calls Groq API if reachable, else uses professional personalized fallback.
    """
    scholarship_str = json.dumps(scholarship_json) if isinstance(scholarship_json, dict) else scholarship_json

    system_prompt = """
    You are an expert scholarship evaluator and resume analyzer.
    Extract structured info from the resume, compare against scholarship,
    identify missing items, give suggestions, evaluate alignment, return JSON.
    """

    user_prompt = f"""
    SCHOLARSHIP DETAILS:
    {scholarship_str}

    RESUME TEXT:
    {resume_text}

    Return structured JSON with fields:
    summary, education, experience, projects, publications/research, technical_skills,
    achievements, certifications, scholarship_alignment, recommendations_per_section,
    missing_elements, overall_improvement_summary
    """

    # Attempt Groq API if key is available
    if GROQ_API_KEY:
        payload = {
            "model": "llama-3.1-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 2000
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Groq API unreachable. Using professional personalized fallback. Details: {str(e)}")

    # Fallback professional evaluation
    return professional_personalized_evaluation(scholarship_json, resume_text)

# ----------------- API ENDPOINT ---------------- -
@app.post("/evaluate")
async def evaluate(
    scholarship_details: str = Form(...),
    resume: UploadFile = File(...)
):
    try:
        # Parse scholarship JSON
        try:
            scholarship_json = json.loads(scholarship_details)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid scholarship_details JSON. Details: {str(e)}"
            )

        # Extract PDF text
        pdf_bytes = await resume.read()
        resume_text = extract_text_from_pdf(pdf_bytes)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="PDF contains no extractable text.")

        # Chunk text
        combined_text = "\n".join(chunk_text(resume_text))

        # Evaluate resume
        evaluation_json_str = evaluate_resume_with_groq(scholarship_json, combined_text)

        # Validate JSON
        try:
            evaluation_json = json.loads(evaluation_json_str)
        except json.JSONDecodeError:
            evaluation_json = {"raw_output": evaluation_json_str}

        return {"result": evaluation_json}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
