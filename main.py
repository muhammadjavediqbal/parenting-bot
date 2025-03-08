from utils import get_response, extract_docx_data, data_insert
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import status
import uvicorn
from fastapi import FastAPI, UploadFile
from langchain_openai import OpenAIEmbeddings
import os
from pydantic import BaseModel

load_dotenv()
app=FastAPI()
embed_fn = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=os.getenv("OPENAI_API_KEY"))

origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
    )

class InsertResponse(BaseModel):
    success: bool
    message: str

@app.get("/")
def start_app():
    return JSONResponse(content={"message":"Welcome to Parenting bot!"})


@app.post("/upload-docx", response_model=InsertResponse)
async def upload_docx(file: UploadFile = File(...), index_name: str = ""):
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="File type must be .docx")

    try:
        # Read the content of the uploaded DOCX file
        content = await file.read()
        # Extract data from the DOCX file
        documents = extract_docx_data(content)

        if not documents:
            raise HTTPException(status_code=500, detail="Failed to extract documents from the file.")

        # Insert the extracted data into the Pinecone vector database
        embeddings = embed_fn # Define your embeddings model
        result = data_insert(documents, embeddings, index_name)

        if isinstance(result, Exception):
            raise HTTPException(status_code=500, detail=str(result))

        return JSONResponse(content={"success": True, "message": "Data inserted successfully."})

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.get("/parenting-bot")
def parenting_bot(query:str):
    try:
        response=get_response(query)
        return JSONResponse(content={"message": "Response Generated Successfully!", 
                                     "data":response  
                                    }, status_code=status.HTTP_200_OK)
    except Exception as ex:
        return JSONResponse(content={"message": f"An error occurred: {str(ex)}", 
                                    }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8383)