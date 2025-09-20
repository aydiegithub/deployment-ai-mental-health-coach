import whisper

class SpeechToText:
    def __init__(self, model_name: str = "base"):
        """
        We initialise the whisper model.
        """
        self.model = whisper.load_model(model_name)
        
    def transcribe(self, audio_path: str) -> str:
        """ 
        Transcribe an audio file to plain text.
        """
        result = self.model.transcribe(audio_path)
        return result["text"]