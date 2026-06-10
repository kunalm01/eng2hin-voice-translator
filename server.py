import argparse
import json
import socket
import threading
import time
import base64
from hashlib import sha1

import numpy as np
from faster_whisper import WhisperModel


MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _recv_exact(conn, size):
    buf = bytearray()
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _send_frame(conn, payload, opcode=1):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    first = 0x80 | (opcode & 0x0F)
    length = len(payload)
    if length < 126:
        header = bytes([first, length])
    elif length < 65536:
        header = bytes([first, 126]) + length.to_bytes(2, "big")
    else:
        header = bytes([first, 127]) + length.to_bytes(8, "big")
    conn.sendall(header + payload)


def _read_frame(conn):
    first_two = _recv_exact(conn, 2)
    fin = first_two[0] & 0x80
    opcode = first_two[0] & 0x0F
    masked = first_two[1] & 0x80
    length = first_two[1] & 0x7F
    if length == 126:
        length = int.from_bytes(_recv_exact(conn, 2), "big")
    elif length == 127:
        length = int.from_bytes(_recv_exact(conn, 8), "big")
    mask = _recv_exact(conn, 4) if masked else b"\x00\x00\x00\x00"
    payload = bytearray(_recv_exact(conn, length))
    if masked:
        for i in range(length):
            payload[i] ^= mask[i % 4]
    return fin, opcode, bytes(payload)


class SimpleASRServer:
    def __init__(self, host="0.0.0.0", port=9090, model_name="small", language="en"):
        self.host = host
        self.port = port
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.language = language

    def serve_forever(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            print(f"[ASR] Server listening on {self.host}:{self.port}")
            while True:
                conn, _ = server.accept()
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn):
        try:
            self._handshake(conn)
            _send_frame(conn, json.dumps({"message": "SERVER_READY"}))
            config = self._read_config(conn)
            print(f"[ASR] Client connected: {config.get('uid', 'unknown')}")
            self._stream_transcription(conn)
        except Exception as e:
            print(f"[ASR] Connection ended: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _handshake(self, conn):
        request = b""
        while b"\r\n\r\n" not in request:
            chunk = conn.recv(1024)
            if not chunk:
                raise ConnectionError("bad handshake")
            request += chunk
        headers = {}
        for line in request.decode("utf-8", errors="ignore").split("\r\n")[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        key = headers["sec-websocket-key"]
        accept = base64.b64encode(sha1((key + MAGIC).encode()).digest()).decode()
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        conn.sendall(response.encode("utf-8"))

    def _read_config(self, conn):
        while True:
            _, opcode, payload = _read_frame(conn)
            if opcode == 1:
                return json.loads(payload.decode("utf-8"))

    def _stream_transcription(self, conn):
        audio_buffer = bytearray()
        speech_started_at = None
        last_voice_at = None
        silence_frames = 0
        while True:
            _, opcode, payload = _read_frame(conn)
            if opcode == 8:
                return
            if opcode != 2:
                continue
            audio_buffer.extend(payload)
            now = time.time()
            pcm = np.frombuffer(payload, dtype=np.int16)
            is_voice = bool(np.abs(pcm).mean() > 250)

            if is_voice:
                if speech_started_at is None:
                    speech_started_at = now
                last_voice_at = now
                silence_frames = 0
            else:
                silence_frames += 1

            if speech_started_at is None:
                continue

            if len(audio_buffer) < 16000 * 2 * 2:
                continue

            if last_voice_at and now - last_voice_at < 0.8:
                continue

            if silence_frames < 4:
                continue

            text = self._transcribe(bytes(audio_buffer))
            if text:
                completed = bool(now - last_voice_at >= 0.8)
                _send_frame(
                    conn,
                    json.dumps(
                        {
                            "uid": "voice-translator-server",
                            "segments": [
                                {
                                    "start": f"{speech_started_at:.3f}",
                                    "end": f"{now:.3f}",
                                    "text": text,
                                    "completed": completed,
                                }
                            ],
                        }
                    ),
                )

            audio_buffer.clear()
            speech_started_at = None
            last_voice_at = None
            silence_frames = 0

    def _transcribe(self, audio_bytes):
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


def main():
    parser = argparse.ArgumentParser(description="Standalone WhisperLive-style ASR server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9090, type=int)
    parser.add_argument("--model", default="small")
    args = parser.parse_args()
    SimpleASRServer(args.host, args.port, args.model).serve_forever()


if __name__ == "__main__":
    main()
