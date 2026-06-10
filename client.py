import queue
import threading
import time
import re
from difflib import SequenceMatcher

from asr_client import ASRClient
from translator import GroqTranslator
from tts import TTSManager


TRANSLATION_QUEUE = queue.Queue()
SHUTDOWN = threading.Event()
LOCK = threading.Lock()
LATEST_TEXT = ""
LATEST_TS = 0.0
LATEST_COMPLETED = False
CURRENT_PARTIAL_TEXT = ""
LAST_FINAL_TEXT = ""
LAST_TRANSLATED_TEXT = ""
LAST_TRANSLATED_AT = 0.0
LAST_TRANSLATION_TIME = 0.0
CURRENT_UTTERANCE_ID = 0
LAST_FINAL_UTTERANCE_ID = -1
LAST_SEEN_TEXT = ""
LAST_TTS_FINISHED_AT = 0.0
ASR_PAUSE_UNTIL = 0.0
tts_manager = None


def is_duplicate_final(text):
    if not text:
        return False

    candidates = [LAST_FINAL_TEXT, LAST_TRANSLATED_TEXT]
    for previous in candidates:
        if not previous:
            continue
        similarity = SequenceMatcher(None, text.lower(), previous.lower()).ratio()
        if (
            text == previous
            or text.startswith(previous)
            or previous.startswith(text)
            or similarity > 0.90
        ):
            return True
    return False


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_similar_utterance(current_text):
    normalized_current = normalize_text(current_text)
    if not normalized_current or not LAST_TRANSLATED_TEXT:
        return False

    normalized_last = normalize_text(LAST_TRANSLATED_TEXT)
    similarity = SequenceMatcher(None, normalized_current, normalized_last).ratio()
    if similarity > 0.80:
        return True

    if LAST_TRANSLATION_TIME and time.time() - LAST_TRANSLATION_TIME < 10.0:
        if normalized_current.startswith(normalized_last) or normalized_last.startswith(normalized_current):
            return True

    return False


def translate_worker(translator, tts):
    global LAST_TTS_FINISHED_AT, ASR_PAUSE_UNTIL
    while not SHUTDOWN.is_set():
        text = TRANSLATION_QUEUE.get()
        try:
            if text is None:
                return
            hindi = translator.translate_to_hindi(text)
            if hindi.strip():
                print(f"[Output] {hindi}")
                tts.speak(hindi)
                with LOCK:
                    LAST_TTS_FINISHED_AT = time.time()
                    ASR_PAUSE_UNTIL = LAST_TTS_FINISHED_AT + 2.5
        finally:
            TRANSLATION_QUEUE.task_done()


def stabilizer_worker():
    global LAST_TRANSLATED_TEXT, LAST_TRANSLATED_AT, LAST_FINAL_TEXT, LAST_FINAL_UTTERANCE_ID, LAST_TRANSLATION_TIME
    global CURRENT_PARTIAL_TEXT, LATEST_TS, LATEST_COMPLETED
    while not SHUTDOWN.is_set():
        time.sleep(0.1)
        with LOCK:
            text = CURRENT_PARTIAL_TEXT.strip()
            ts = LATEST_TS
            completed = LATEST_COMPLETED
            last_translated = LAST_TRANSLATED_TEXT
            utterance_id = CURRENT_UTTERANCE_ID
            asr_pause_until = ASR_PAUSE_UNTIL

        if not text or tts_manager.is_playing:
            continue
        if asr_pause_until and time.time() < asr_pause_until:
            continue

        if completed:
            if is_duplicate_final(text):
                continue
            if is_similar_utterance(text):
                continue
            with LOCK:
                if utterance_id == LAST_FINAL_UTTERANCE_ID:
                    continue
                LAST_FINAL_TEXT = text
                LAST_FINAL_UTTERANCE_ID = utterance_id
                LAST_TRANSLATED_TEXT = text
                LAST_TRANSLATED_AT = time.time()
                LAST_TRANSLATION_TIME = LAST_TRANSLATED_AT
            TRANSLATION_QUEUE.put(text)
            continue

        # Fallback only when completed=True never arrives.
        if time.time() - ts < 3.0:
            continue
        if len(text) <= 20 or text == last_translated:
            continue

        if is_duplicate_final(text):
            continue
        if is_similar_utterance(text):
            continue

        with LOCK:
            if text == LAST_TRANSLATED_TEXT:
                continue
            LAST_TRANSLATED_TEXT = text
            LAST_TRANSLATED_AT = time.time()
            LAST_TRANSLATION_TIME = LAST_TRANSLATED_AT

        TRANSLATION_QUEUE.put(text)


def on_transcript(text, segments):
    global LATEST_TEXT, LATEST_TS, LATEST_COMPLETED, CURRENT_PARTIAL_TEXT, LAST_SEEN_TEXT, CURRENT_UTTERANCE_ID
    if not segments:
        return

    completed = any(seg.get("completed", False) for seg in segments)
    if text == LAST_SEEN_TEXT:
        return

    with LOCK:
        LATEST_TEXT = text
        LATEST_TS = time.time()
        LATEST_COMPLETED = completed
        if ASR_PAUSE_UNTIL and time.time() < ASR_PAUSE_UNTIL:
            LAST_SEEN_TEXT = text
            return

    LAST_SEEN_TEXT = text
    print(f"[Input] {text}")
    with LOCK:
        CURRENT_PARTIAL_TEXT = text
        if completed:
            CURRENT_UTTERANCE_ID += 1
        # keep current partial available for fallback; final segments are handled by stabilizer_worker


def main():
    translator = GroqTranslator()
    global tts_manager
    tts_manager = TTSManager()

    print("[Cue] Start speaking in English now.")

    threading.Thread(target=translate_worker, args=(translator, tts_manager), daemon=True).start()
    threading.Thread(target=stabilizer_worker, daemon=True).start()

    client = ASRClient()
    client.connect(on_transcript)

    try:
        client.start_microphone()
        client.stream_microphone()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        SHUTDOWN.set()
        TRANSLATION_QUEUE.put_nowait(None)
        try:
            client.close()
        except Exception:
            pass
        try:
            tts_manager.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
