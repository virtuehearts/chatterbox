import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from chatterbox.tts import ChatterboxTTS

PERSONALITY_DIR = "webapp/personalities"

app = FastAPI()
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = None

@app.on_event("startup")
async def load_model():
    global model
    os.makedirs(PERSONALITY_DIR, exist_ok=True)
    model = ChatterboxTTS.from_pretrained(DEVICE)

@app.get("/")
async def index():
    with open("webapp/static/index.html", "r", encoding="utf8") as f:
        return HTMLResponse(content=f.read())


@app.get("/personalities")
async def list_personalities():
    names = [os.path.splitext(p)[0] for p in os.listdir(PERSONALITY_DIR) if p.endswith(".wav")]
    return {"personalities": names}


@app.post("/personalities")
async def add_personality(name: str = Form(...), voice: UploadFile = File(...)):
    path = os.path.join(PERSONALITY_DIR, f"{name}.wav")
    with open(path, "wb") as f:
        content = await voice.read()
        f.write(content)
    return JSONResponse({"status": "ok"})

@app.post("/generate")
async def generate(text: str = Form(...), personality: str = Form(None), voice: UploadFile = File(None)):
    voice_path = None
    if personality:
        path = os.path.join(PERSONALITY_DIR, f"{personality}.wav")
        if os.path.exists(path):
            voice_path = path
    if voice is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        content = await voice.read()
        tmp.write(content)
        tmp.flush()
        voice_path = tmp.name
    if not voice_path:
        return JSONResponse({"error": "No voice sample provided"}, status_code=400)
    wav = model.generate(text, audio_prompt_path=voice_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
        import torchaudio as ta
        ta.save(out.name, wav, model.sr)
        return FileResponse(out.name, media_type="audio/wav", filename="output.wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.app:app", host="0.0.0.0", port=8000)
