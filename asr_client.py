import json
import threading

import pyaudio
import websocket


class ASRClient:
    def __init__(
        self,
        server_host="localhost",
        server_port=9090,
        model="small",
        language="en",
        should_send_audio=None,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.model = model
        self.language = language
        self.should_send_audio = should_send_audio
        self.ws = None
        self.audio_stream = None
        self.audio = pyaudio.PyAudio()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.last_seen_text = ""

    def connect(self, on_transcript):
        url = f"ws://{self.server_host}:{self.server_port}"

        self.ws = websocket.WebSocketApp(
            url,
            on_open=lambda ws: self._on_open(ws),
            on_message=lambda ws, message: self._on_message(ws, message, on_transcript),
            on_error=lambda _ws, error: print(f"[ASR] WebSocket error: {error}"),
            on_close=lambda _ws, *_: self.stop_event.set(),
        )

        threading.Thread(target=self.ws.run_forever, daemon=True).start()
        self.ready_event.wait()

    def _on_open(self, ws):
        ws.send(
            json.dumps(
                {
                    "uid": "submission-client",
                    "language": self.language,
                    "model": self.model,
                    "task": "transcribe",
                    "use_vad": True,
                    "send_last_n_segments": 10,
                }
            )
        )
        self.ready_event.set()

    def _on_message(self, _ws, message, on_transcript):
        data = json.loads(message)
        if data.get("message") == "SERVER_READY":
            print("[ASR] Server ready")
            return
        segments = data.get("segments") or []
        if not segments:
            return
        text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
        if text and text != self.last_seen_text:
            self.last_seen_text = text
            on_transcript(text, segments)

    def start_microphone(self):
        self.audio_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096,
        )

    def stream_microphone(self):
        while not self.stop_event.is_set():
            try:
                data = self.audio_stream.read(4096, exception_on_overflow=False)
                if self.ws:
                    self.ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception:
                break

    def close(self):
        self.stop_event.set()
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception:
                pass
        try:
            self.audio.terminate()
        except Exception:
            pass
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
