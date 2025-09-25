import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pdfplumber
from llama_cpp import Llama

# Initialize LLaMA Q4_0 with proper context window
llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_0.gguf",
    n_ctx=4096  # explicit context length
)

MAX_RESPONSE_TOKENS = 50
CONTEXT_WINDOW = 4096
MAX_INPUT_TOKENS = CONTEXT_WINDOW - MAX_RESPONSE_TOKENS

app = FastAPI()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, max_input_tokens: int = MAX_INPUT_TOKENS):
    chunk_size = max_input_tokens * 4  # approx chars per token
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

def is_invoice(text: str) -> str:
    results = []
    for chunk in chunk_text(text):
        prompt = f"Is this document an invoice? Answer 'Yes' or 'No':\n{chunk}"
        response = llm(prompt, max_tokens=MAX_RESPONSE_TOKENS)
        results.append(response['choices'][0]['text'].strip())

    yes_count = sum(1 for r in results if r.lower().startswith("yes"))
    no_count = sum(1 for r in results if r.lower().startswith("no"))
    return "Yes" if yes_count >= no_count else "No"

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)
        if not text.strip():
            return JSONResponse({"error": "No text extracted from PDF"}, status_code=400)

        vote = is_invoice(text)
        return JSONResponse({"result": vote})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
