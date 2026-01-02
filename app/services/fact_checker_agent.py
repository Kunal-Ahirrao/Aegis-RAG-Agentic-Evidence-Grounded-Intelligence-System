"""
FactCheckerAgent

A lightweight verification agent that reviews the final answer
against the retrieved context and flags possible hallucinations
or unsupported claims.
"""
from typing import List, Dict, Any
import json
import re

import google.generativeai as genai
import openai

from app.core.config import settings


class FactCheckerAgent:
    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self.provider = (settings.LLM_PROVIDER or "gemini").lower()
        self.model_name = model_name

        if self.provider == "gemini":
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not found in settings. Please check your configuration.")
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self._gemini_model = genai.GenerativeModel(model_name)
            print(f"[FactChecker] Gemini model '{model_name}' initialized.")
        elif self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set but LLM_PROVIDER='openai'.")
            openai.api_key = settings.OPENAI_API_KEY
            if self.model_name == "gemini-2.5-flash":
                self.model_name = "gpt-4.1-mini"
            print(f"[FactChecker] OpenAI model '{self.model_name}' initialized.")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER for FactCheckerAgent: {self.provider}")

    async def _generate_text(self, prompt: str) -> str:
        if self.provider == "gemini":
            response = await self._gemini_model.generate_content_async(prompt)
            return response.text
        elif self.provider == "openai":
            resp = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message["content"]
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def review_answer(
        self,
        question: str,
        answer: str,
        context_chunks: List[str],
    ) -> Dict[str, Any]:
        """
        Returns a small JSON-style dict with:
        - verdict: 'supported', 'partially_supported', or 'unsupported'
        - risk_score: float 0-1
        - notes: explanation
        """
        context_preview = "\n\n".join(context_chunks[:5])
        prompt = f"""You are a strict fact-checking assistant.

You are given:
- A user question
- A proposed answer
- A set of context excerpts retrieved from a document

Task:
1) Determine whether the answer is directly supported by the context.
2) If only parts are supported, mark it as partially supported.
3) If the answer goes beyond or contradicts the context, mark as unsupported.
4) Be conservative: if unsure, lean toward 'partially_supported' or 'unsupported'.

Return ONLY a JSON object with keys:
- "verdict": one of ["supported", "partially_supported", "unsupported"]
- "risk_score": a float between 0 and 1 where higher means more hallucination risk
- "notes": short explanation (1-3 sentences)

QUESTION:
{question}

ANSWER:
{answer}

CONTEXT:
{context_preview}
"""

        raw = await self._generate_text(prompt)

        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in fact-checker response.")
            json_str = raw[start : end + 1]
            json_str = re.sub(r"^```(?:json)?\s*", "", json_str)
            json_str = re.sub(r"\s*```$", "", json_str)
            parsed = json.loads(json_str)
            # Minimal sanity checks
            if not isinstance(parsed, dict):
                raise ValueError("Parsed fact-check result is not a dict.")
            return parsed
        except Exception as e:
            print(f"[FactChecker] Failed to parse JSON response: {e}. Raw:\n{raw}")
            # Fallback: return a conservative default
            return {
                "verdict": "partially_supported",
                "risk_score": 0.7,
                "notes": "Could not parse structured fact-check output; treating as medium risk.",
            }
