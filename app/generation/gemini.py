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
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ):
        from google import genai

        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")

        self._client = genai.Client(api_key=api_key)
        self._api_key = api_key
        self.model_name = model_name
        self._default_temperature = temperature
        self._default_max_tokens = max_output_tokens
        logger.info(f"Gemini generator initialized: {model_name}")

    def set_model(self, model_name: str):
        if model_name != self.model_name:
            self.model_name = model_name
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
        from google import genai
        from google.genai import types

        if not retrieved_chunks:
            return "I could not find any relevant passages in the documents."

        context = self._format_context(retrieved_chunks)
        prompt = QUERY_TEMPLATE.format(context=context, question=question)

        temp = temperature if temperature is not None else self._default_temperature
        tokens = max_tokens if max_tokens is not None else self._default_max_tokens

        for attempt in range(3):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=temp,
                        max_output_tokens=tokens,
                        top_p=0.9,
                    ),
                )
                return response.text.strip()

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower() or "resource" in error_str.lower():
                    match = re.search(r"seconds: (\d+)", error_str)
                    wait = int(match.group(1)) + 5 if match else 60
                    logger.warning(f"Rate limit. Waiting {wait}s (attempt {attempt + 1}/3)...")
                    time.sleep(wait)
                else:
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