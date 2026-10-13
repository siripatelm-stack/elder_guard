import os
import tempfile


class Transcriber:
    def __init__(self, size="base"):
        self.size = size
        self._model = None

    def transcribe_bytes(self, data, suffix=".wav"):
        if self._model is None:
            from faster_whisper import WhisperModel      # loaded only when voice is used
            self._model = WhisperModel(self.size, device="cpu", compute_type="int8")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            segments, info = self._model.transcribe(path)
            text = " ".join(s.text.strip() for s in segments)
        finally:
            os.remove(path)
        return text, info.language
