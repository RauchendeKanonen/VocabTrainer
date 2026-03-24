import io
import os
import uuid
import torch
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BitsAndBytesConfig
from kugelaudio_open import KugelAudioForConditionalGenerationInference, KugelAudioProcessor

app = FastAPI()

# ---- Load model ONCE at startup ----
skip = [
    "model.acoustic_tokenizer",
    "model.semantic_tokenizer",
    "model.acoustic_connector",
    "model.semantic_connector",
    "model.prediction_head",
]

bnb8 = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=skip,
)

model = KugelAudioForConditionalGenerationInference.from_pretrained(
    "kugelaudio/kugelaudio-0-open",
    quantization_config=bnb8,
    dtype=torch.float16,
    low_cpu_mem_usage=False,
).eval()

# Reduce memory spikes
if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "config"):
    model.model.language_model.config.use_cache = False

processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")

OUT_DIR = "tts_out"
os.makedirs(OUT_DIR, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    cfg_scale: float = 3.0
    voice_prompt: str | None = None  # optional path on server, e.g. "voices/ref.wav"
    return_wav_bytes: bool = False   # if True: return base64 wav instead of filename
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 2048
    do_sample: bool = False
    trim_start_s: float = 0.0        # seconds to trim from beginning
    trim_end_s: float = 0.0          # seconds to trim from end
    

@app.post("/tts")
def tts(req: TTSRequest):
    # Keep inputs on CPU; model has mixed placement managed internally
    if req.voice_prompt:
        inputs = processor(text=req.text, voice_prompt=req.voice_prompt, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=req.text, return_tensors="pt")

    print(f"cfg_scale {req.cfg_scale}")
    print(f"temperature {req.temperature}")
    print(f"top_p {req.top_p}")
    print(f"repetition_penalty {req.repetition_penalty}")
    print(f"max_new_tokens {req.max_new_tokens}")
    print(f"do_sample {req.do_sample}")
    print(f"trim_start_s {req.trim_start_s}")
    print(f"trim_end_s {req.trim_end_s}")
    print(req.text)
    
    with torch.inference_mode():
        #out = model.generate(**inputs, cfg_scale=req.cfg_scale)
        out = model.generate(
            **inputs,
            cfg_scale=req.cfg_scale,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            max_new_tokens=req.max_new_tokens, 
            do_sample=req.do_sample
        )
    
    wav = out.speech_outputs[0].detach().float().cpu().numpy()

    # --- Trim begin/end (seconds -> samples) ---
    sr = 24000
    trim_start_s = max(0.0, float(req.trim_start_s))
    trim_end_s = max(0.0, float(req.trim_end_s))
    start_n = int(trim_start_s * sr)
    end_n = int(trim_end_s * sr)

    n = int(wav.shape[0])
    if start_n >= n:
        # If trimming removes everything, return empty (or you could raise an HTTP error)
        wav = wav[:0]
    else:
        if end_n > 0:
            # Cap end trim so we don't go negative
            end_n = min(end_n, max(0, n - start_n))
            wav = wav[start_n : n - end_n]
        else:
            wav = wav[start_n:]


    
    # Write a normal PCM WAV
    fn = f"{uuid.uuid4().hex}.wav"
    path = os.path.join(OUT_DIR, fn)
    sf.write(path, wav, sr, subtype="PCM_16")

    return {"ok": True, "wav_path": path}
