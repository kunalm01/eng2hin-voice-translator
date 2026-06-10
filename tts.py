import asyncio
import os
import subprocess
import sys
import tempfile
import time


class TTSManager:
    def __init__(self, voice="hi-IN-MadhurNeural"):
        self.voice = voice
        self.is_playing = False
        self._process = None

    def speak(self, text: str):
        if not text or not text.strip():
            return

        path = os.path.join(tempfile.gettempdir(), f"eng2hin_hindi_{int(time.time() * 1000)}.mp3")
        try:
            self.is_playing = True
            asyncio.run(self._generate_audio(text, path))
            self._play(path)
        finally:
            self.is_playing = False
            print("[Cue] Speak now.")
            self._process = None
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    async def _generate_audio(self, text: str, path: str):
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(path)

    def _play(self, path: str):
        try:
            if sys.platform == "darwin":
                self._process = subprocess.Popen(["afplay", path])
                self._process.wait()
            elif sys.platform.startswith("linux"):
                for cmd in (["paplay", path], ["aplay", path], ["ffplay", "-nodisp", "-autoexit", path]):
                    try:
                        self._process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        self._process.wait()
                        return
                    except Exception:
                        continue
            elif sys.platform == "win32":
                import winsound

                winsound.PlaySound(path, winsound.SND_FILENAME)
        finally:
            self._process = None

    def stop(self):
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except Exception:
                pass
        self._process = None
        self.is_playing = False
