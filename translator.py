import os
import re

from groq import Groq


class GroqTranslator:
    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def translate_to_hindi(self, text: str) -> str:
        if not text or not text.strip() or self.client is None:
            return ""

        translated = self._translate_once(text, strict=False)
        if self._is_valid_translation(text, translated):
            return translated

        translated = self._translate_once(text, strict=True)
        return translated if self._is_valid_translation(text, translated) else ""

    def _translate_once(self, text: str, strict: bool) -> str:
        system_prompt = (
            "You are a translation engine.\n"
            "Translate English to Hindi.\n"
            "\n"
            "Rules:\n"
            "- NEVER answer questions.\n"
            "- NEVER provide explanations.\n"
            "- NEVER act as an assistant.\n"
            "- NEVER infer intent.\n"
            "- NEVER continue conversations.\n"
            "- NEVER summarize.\n"
            "- NEVER paraphrase.\n"
            "- NEVER improve the meaning.\n"
            "- ONLY translate the exact input text.\n"
            "- Preserve meaning exactly.\n"
            "- Preserve named entities.\n"
            "- Preserve dates, times, numbers and abbreviations.\n"
            "- Preserve sentence intent.\n"
            "- Output Hindi only.\n"
            "- Output translation only.\n"
            "- No markdown.\n"
            "- No quotes.\n"
            "- No extra words.\n"
            "\n"
            "Few-shot examples:\n"
            "Input: What are you doing?\n"
            "Output: तुम क्या कर रहे हो?\n\n"
            "Input: Can we move the meeting to 5 PM tomorrow?\n"
            "Output: क्या हम मीटिंग को कल शाम 5 बजे कर सकते हैं?\n\n"
            "Input: Google Meet starts at 5 PM.\n"
            "Output: Google Meet शाम 5 बजे शुरू होता है।\n"
        )

        user_prompt = text
        if strict:
            user_prompt = (
                "Translate the following text exactly and only translate it.\n"
                "Do not answer, explain, summarize, or add anything.\n"
                f"Input: {text}"
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()

    def _is_valid_translation(self, source_text: str, translated_text: str) -> bool:
        if not translated_text:
            return False

        if self._looks_like_chat_response(translated_text):
            return False

        if self._suspicious_first_person(source_text, translated_text):
            return False

        if self._adds_new_information(source_text, translated_text):
            return False

        return True

    def _looks_like_chat_response(self, translated_text: str) -> bool:
        patterns = (
            "अनुवाद कर रहा",
            "अनुवाद कर रही",
            "मैं अंग्रेजी से",
            "मैं आपके लिए",
            "आपके लिए अंग्रेजी",
            "मैं आपके",
            "मैं यह",
            "मैं ",
        )
        if any(pattern in translated_text for pattern in patterns):
            return True
        return translated_text.startswith("मैं ") and not translated_text.startswith("मैं क्या")

    def _suspicious_first_person(self, source_text: str, translated_text: str) -> bool:
        source_mentions_first_person = re.search(r"\b(I|i|I'm|I am|we are|we can|we'll|I'll|we)\b", source_text) is not None
        source_mentions_first_person = source_mentions_first_person or re.search(r"\b(I'm|I am|I can|we can|we are|I'll|I'll be)\b", source_text) is not None
        suspicious_phrase = "मैं अंग्रेजी से" in translated_text or "अनुवाद कर" in translated_text
        return suspicious_phrase and not source_mentions_first_person

    def _adds_new_information(self, source_text: str, translated_text: str) -> bool:
        source_words = set(re.findall(r"[\w']+", source_text.lower()))
        translated_words = set(re.findall(r"[\w']+", translated_text.lower()))
        if len(translated_text.split()) > max(2, len(source_text.split()) + 6):
            return True
        if "doing" in source_words and "कर" not in translated_text:
            return True
        if "meeting" in source_words and "मीटिंग" not in translated_text and "बैठक" not in translated_text:
            return True
        if source_words and "google" in source_words and "google" not in translated_words:
            return True
        return False
