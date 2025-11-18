# Resume_Analyzer

**app_fallback.py** is a FastAPI-based application that evaluates PDF resumes against scholarship criteria.  

- Extracts text from uploaded PDF resumes using `PyPDF2`.  
- Provides structured evaluation including education, experience, projects, technical skills, and overall scholarship alignment.  
- Uses Groq LLM API if available; otherwise, falls back to a professional, personalized evaluator.  
- Returns JSON with recommendations per section and missing elements.  
- Easy to run locally with `uvicorn` and environment variable setup for Groq API.  
- API endpoint: **POST `/evaluate`** with form fields `scholarship_details` (JSON) and `resume` (PDF).  
- Perfect for automating scholarship resume assessments and generating actionable improvement suggestions.  
- File name: **`app_fallback.py`** – main application code.  
