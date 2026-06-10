import logging
import gc
import torch
from pathlib import Path

log = logging.getLogger("dub_pipeline.translategemma_fallback")

def download_translategemma(path: Path):
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/DevQuasar/google.translategemma-4b-it-GGUF/resolve/main/google.translategemma-4b-it.Q4_K_M.gguf"
    log.info(f"Downloading TranslateGemma model from {url} to {path} ... This may take a few minutes (approx. 2.5 GB).")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = f"\rDownloading TranslateGemma: {percent:.1f}% ({readsofar / 1e6:.1f} MB / {totalsize / 1e6:.1f} MB)"
                print(s, end="", flush=True)
            else:
                print(f"\rDownloading TranslateGemma: {readsofar / 1e6:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, str(path), reporthook)
        print("\nDownload complete!")
    except Exception as e:
        log.error(f"Failed to download TranslateGemma model: {e}")
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        raise

class TranslateGemmaTranslator:
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
            print(f"\n[WARNING] Local TranslateGemma model not found at: {path.absolute()}")
            choice = input("Would you like to download google.translategemma-4b-it.Q4_K_M.gguf (approx. 2.5 GB) from Hugging Face now? [y/N]: ").strip().lower()
            if choice == 'y':
                download_translategemma(path)
            else:
                raise FileNotFoundError(f"Local TranslateGemma GGUF model file not found at: {path.absolute()}")

        log.info(f"Loading TranslateGemma-4B fallback model from {path.name} …")
        try:
            from llama_cpp import Llama
            n_gpu = -1 if self.device == "cuda" else 0
            self.llm = Llama(
                model_path=str(path),
                n_ctx=4096,
                n_gpu_layers=n_gpu,
                verbose=False
            )
            log.info("TranslateGemma-4B loaded successfully")
        except Exception as e:
            log.error(f"Failed to load local TranslateGemma model: {e}")
            raise

    def unload(self):
        """Strict VRAM lifecycle release."""
        if self.llm is not None:
            log.info("Unloading local TranslateGemma model and freeing VRAM …")
            del self.llm
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("VRAM cleared post TranslateGemma unloading")

    def _generate(self, prompt: str) -> str:
        self.load()
        try:
            # We use direct completion for precise formatting or chat template
            response = self.llm(
                prompt=prompt,
                temperature=0.2,
                top_p=0.9,
                repeat_penalty=1.1,
                max_tokens=256
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            log.error(f"TranslateGemma inference failed: {e}")
            raise

    def translate(self, text: str, source_lang: str = "ru", target_lang: str = "en") -> str:
        prompt = (
            f"<start_of_turn>user\n"
            f"You are a professional Russian (ru) to English (en) translator. "
            f"Translate the following Russian lecture text into natural spoken English.\n"
            f"Rules:\n"
            f"- Preserve all technical meaning and C++ programming terms.\n"
            f"- Output ONLY the English translation with no quotes, commentary, introduction, or notes.\n\n"
            f"Russian: {text}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        return self._generate(prompt)

    def rewrite_for_timing(self, text: str, duration_budget: float) -> str:
        prompt = (
            f"<start_of_turn>user\n"
            f"You are a professional English editor. Rewrite the following English text to be more concise "
            f"so it fits within a speaking window of {duration_budget:.2f} seconds.\n"
            f"Rules:\n"
            f"- Keep the C++ programming concepts and terms exact.\n"
            f"- Make it shorter and easier to say quickly.\n"
            f"- Output ONLY the refined English text with no quotes, explanations, or notes.\n\n"
            f"Literal English: {text}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        return self._generate(prompt)
