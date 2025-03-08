# main.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
    )
# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Set your API key as an environment variable
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        response = model.generate_content(query.text)
        cleaned_response = response.text.replace("\n", "").replace("*", "") #Clean the response
        return {"response": cleaned_response} #return cleaned response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8383)
