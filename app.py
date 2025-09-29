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

app = FastAPI()

def extract_pages_from_pdf(file_bytes: bytes):
    """Return a list of page texts."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = []
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return pages

def is_invoice(pages: list) -> str:
    """Check each page individually and do majority vote."""
    results = []
    for page_text in pages:
        prompt = f"""
You are a document classifier. Determine whether the following document page is an **invoice**.
Answer strictly 'Yes' for invoices, 'No' for any other type of document (e.g., CV, resume, report, letter).

Page text:
{page_text}
"""
        response = llm(prompt, max_tokens=MAX_RESPONSE_TOKENS)
        results.append(response['choices'][0]['text'].strip())

    # Majority vote
    yes_count = sum(1 for r in results if r.lower().startswith("yes"))
    no_count = sum(1 for r in results if r.lower().startswith("no"))
    return "yes" if yes_count >= no_count else "no"

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        pages = extract_pages_from_pdf(file_bytes)
        if not pages:
            return JSONResponse({"error": "No text extracted from PDF"}, status_code=400)

        vote = is_invoice(pages)
        return JSONResponse({"result": vote})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
