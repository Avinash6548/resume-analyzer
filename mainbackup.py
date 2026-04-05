from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

app = FastAPI()

class PromptRequest(BaseModel):
    query: str

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.post("/ask-ai")
def ask_ai(data: PromptRequest):
    try:
        response = client.chat.completions.create( 
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": data.query}],
            max_tokens=200
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

@app.get("/first-api")
def first_api(name: str):
    return {"message": f"Hello {name}, I am the first API!"}