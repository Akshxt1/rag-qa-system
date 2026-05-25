from typing import List, Tuple, Dict, Optional
from loguru import logger


SYSTEM_PROMPT = """You are an expert document Q&A assistant. Your job is to answer questions \
accurately using ONLY the provided context passages.

Rules:
1. Base your answer exclusively on the provided context — never use outside knowledge.
2. If the answer is not present in the context, respond with:
   "I could not find the answer in the provided documents."
3. Cite the source number (e.g. [Source 1]) when referencing specific passages.
4. Be concise and precise. Avoid padding or unnecessary caveats.
5. If multiple sources support the answer, synthesize them cohesively."""

QUERY_TEMPLATE = """\
CONTEXT:
{context}

---
QUESTION: {question}

ANSWER:"""


class GeminiGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.1-flash-lite",
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ):
        import google.generativeai as genai
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        genai.configure(api_key=api_key)
        self._api_key = api_key
        self.model_name = model_name
        self._gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": 0.9,
        }
        self._model = self._build_model(model_name)
        logger.info(f"Gemini generator initialized: {model_name}")

    def _build_model(self, model_name: str):
        import google.generativeai as genai
        return genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
        )

    def set_model(self, model_name: str):
        """Switch to a different Gemini model."""
        if model_name != self.model_name:
            self.model_name = model_name
            self._model = self._build_model(model_name)
            logger.info(f"Switched to model: {model_name}")

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Tuple[Dict, float]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        import re
        import time
        import google.generativeai as genai
        from google.api_core.exceptions import ResourceExhausted

        if not retrieved_chunks:
            return "I could not find any relevant passages in the documents."

        context = self._format_context(retrieved_chunks)
        prompt = QUERY_TEMPLATE.format(context=context, question=question)

        # Build per-call config (override defaults if provided)
        call_config = dict(self._gen_config)
        if temperature is not None:
            call_config["temperature"] = temperature
        if max_tokens is not None:
            call_config["max_output_tokens"] = max_tokens

        for attempt in range(3):
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(**call_config),
                )
                return response.text.strip()

            except ResourceExhausted as e:
                match = re.search(r"seconds: (\d+)", str(e))
                wait = int(match.group(1)) + 5 if match else 60
                logger.warning(f"Rate limit. Waiting {wait}s (attempt {attempt + 1}/3)...")
                time.sleep(wait)

            except Exception as e:
                raise e

        return "⚠️ Rate limit reached after 3 attempts. Please wait a moment and try again."

    @staticmethod
    def _format_context(chunks: List[Tuple[Dict, float]]) -> str:
        parts = []
        for i, (chunk, score) in enumerate(chunks, start=1):
            source_name = chunk.get("filename", chunk.get("source", "unknown"))
            chunk_idx = chunk.get("chunk_idx", 0)
            total = chunk.get("total_chunks", "?")
            parts.append(
                f"[Source {i}: {source_name} (chunk {chunk_idx + 1}/{total}, "
                f"relevance={score:.3f})]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)