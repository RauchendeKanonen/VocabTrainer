import argparse
import requests

def main():
    parser = argparse.ArgumentParser(description="Simple TTS client")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="CFG scale (default: 3.0)")
    parser.add_argument("--url", default="http://127.0.0.1:8000/tts", help="TTS server URL")
    parser.add_argument("--voice-prompt", default=None,
                    help="Path to reference voice WAV on server, e.g. voices/anna.wav")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="repetition penalty")
    parser.add_argument("--top-p", type=float, default=1.0, help="sampling control up to 1.0, the higher the more freedom")
    parser.add_argument("--temperature", type=float, default=1.0, help="0.8-1.0")
    parser.add_argument("--do-sample", type=bool, default=False, help="for temperature, top-p")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="maximum tokens to generate")
    parser.add_argument("--trim-start", type=float, default=0.0,
                        help="Trim this many seconds from the beginning of generated audio (default: 0.0)")
    parser.add_argument("--trim-end", type=float, default=0.0,
                        help="Trim this many seconds from the end of generated audio (default: 0.0)")
                        
    args = parser.parse_args()

    payload = {
        "text":                 args.text,
        "cfg_scale":            args.cfg_scale,
        "voice_prompt":         args.voice_prompt,
        "top_p":                args.top_p,
        "temperature":          args.temperature,
        "repetition_penalty":   args.repetition_penalty,
        "do_sample":            args.do_sample,
        "max_new_tokens":       args.max_new_tokens,
        "trim_start_s":           args.trim_start,
        "trim_end_s":             args.trim_end
        
    }

    r = requests.post(args.url, json=payload, timeout=600)
    r.raise_for_status()
    print(r.json())

if __name__ == "__main__":
    main()
