import logging
import gc
import torch
from pathlib import Path

log = logging.getLogger("dub_pipeline.qwen_fallback")

def download_qwen(path: Path):
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    log.info(f"Downloading Qwen model from {url} to {path} ... This may take a few minutes (approx. 2.0 GB).")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = f"\rDownloading: {percent:.1f}% ({readsofar / 1e6:.1f} MB / {totalsize / 1e6:.1f} MB)"
                print(s, end="", flush=True)
            else:
                print(f"\rDownloading: {readsofar / 1e6:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, str(path), reporthook)
        print("\nDownload complete!")
    except Exception as e:
        log.error(f"Failed to download Qwen model: {e}")
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        raise

class QwenFallbackTranslator:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.llm = None

    def load(self):
        """Lazy loader to prevent loading model prematurely into VRAM."""
        if self.llm is not None:
            return
        
        path = Path(self.model_path)
        if not path.exists():
            print(f"\n[WARNING] Local Qwen model not found at: {path.absolute()}")
            choice = input("Would you like to download Qwen2.5-3B-Instruct-Q4_K_M.gguf (approx. 2.0 GB) from Hugging Face now? [y/N]: ").strip().lower()
            if choice == 'y':
                download_qwen(path)
            else:
                raise FileNotFoundError(f"Local Qwen GGUF model file not found at: {path.absolute()}")

        log.info(f"Loading Qwen-3B fallback model from {path.name} …")
        try:
            from llama_cpp import Llama
            n_gpu = -1 if self.device == "cuda" else 0
            self.llm = Llama(
                model_path=str(path),
                n_ctx=4096,
                n_gpu_layers=n_gpu,
                verbose=False
            )
            log.info("Qwen-3B loaded successfully")
        except Exception as e:
            log.error(f"Failed to load local Qwen model: {e}")
            raise

    def unload(self):
        """Strict VRAM lifecycle release."""
        if self.llm is not None:
            log.info("Unloading local Qwen model and freeing VRAM …")
            del self.llm
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("VRAM cleared post Qwen unloading")

    def _generate(self, prompt: str) -> str:
        self.load()
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a professional educational localization translator and refiner. Follow the specified output format strictly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                top_p=0.9,
                repeat_penalty=1.1,
                max_tokens=256
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.error(f"Inference failed: {e}")
            raise

    def translate(self, text: str, source_lang: str = "ru", target_lang: str = "en") -> str:
        prompt = (
            f"Translate this Russian technical lecture excerpt into natural spoken English.\n\n"
            f"Russian: {text}\n\n"
            f"Rules:\n"
            f"- preserve technical meaning\n"
            f"- preserve programming terminology\n"
            f"- sound like a university lecturer\n"
            f"- avoid unnecessary filler words\n"
            f"- do not explain concepts\n"
            f"- output ONLY translated English text with no quotes, explanations, or labels."
        )
        return self._generate(prompt)

    def rewrite_for_timing(self, text: str, duration_budget: float) -> str:
        prompt = (
            f"Rewrite this translated lecture segment into concise spoken English to fit in a {duration_budget:.2f} second speaking window.\n\n"
            f"Literal English: {text}\n\n"
            f"Rules:\n"
            f"- preserve all technical meaning\n"
            f"- preserve terminology exactly\n"
            f"- reduce verbosity\n"
            f"- keep natural lecture flow\n"
            f"- optimize for short spoken duration\n"
            f"- output ONLY the refined English text with no quotes or extra commentary."
        )
        return self._generate(prompt)
