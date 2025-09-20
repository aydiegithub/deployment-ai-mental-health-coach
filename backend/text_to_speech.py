import requests
import base64
import os
import dotenv

dotenv.load_dotenv()

class MurfTTSClient:
    """
    Murf Text-to-Speech client.
    """

    BASE_URL = "https://api.murf.ai/v1/speech/generate"

    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("MURF_API_KEY")
        if not api_key or not isinstance(api_key, str):
            raise ValueError("API key must be provided and must be a string.")
        self.api_key = api_key

    def generate_speech(
        self,
        text: str,
        voice_id: str,
        style: str = None,
        encode_as_base64: bool = True,
        format: str = "MP3",
        sample_rate: int = 44100,
        channel_type: str = "MONO",
        rate: float = 0.0,
        pitch: float = 0.0,
        variation: int = 1,
        pronunciation_dict: dict = None,
    ) -> dict:
        """
        Generate speech from text.
        """

        # Validate minimal required
        if not text or not isinstance(text, str):
            return {"success": False, "error": "Text is required and must be a string."}
        if not voice_id or not isinstance(voice_id, str):
            return {"success": False, "error": "voice_id is required and must be a string."}

        # Validate optional parameters against Murf limits
        if rate is not None and not (-50 <= rate <= 50):
            return {"success": False, "error": "rate must be between -50 and +50."}
        if pitch is not None and not (-50 <= pitch <= 50):
            return {"success": False, "error": "pitch must be between -50 and +50."}
        if sample_rate not in {8000, 24000, 44100, 48000}:
            return {"success": False, "error": f"sample_rate {sample_rate} is invalid."}
        if channel_type.upper() not in {"MONO", "STEREO"}:
            return {"success": False, "error": "channel_type must be MONO or STEREO."}

        payload = {
            "text": text,
            "voiceId": voice_id,
            "format": format,
            "encodeAsBase64": encode_as_base64,
            "sampleRate": sample_rate,
            "channelType": channel_type.upper(),
            "rate": rate,
            "pitch": pitch,
            "variation": variation,
        }
        if style:
            payload["style"] = style
        if pronunciation_dict:
            payload["pronunciationDictionary"] = pronunciation_dict

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

        try:
            resp = requests.post(self.BASE_URL, json=payload, headers=headers, timeout=30)
        except requests.RequestException as e:
            print(f"Network or request exception: {e}")
            return {
                "success": False,
                "error": "Network or request exception",
                "details": str(e)
            }

        print(f"DEBUG Murf API status: {resp.status_code}")
        print(f"DEBUG Murf API response: {resp.text}")

        if resp.status_code != 200:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            return {
                "success": False,
                "status_code": resp.status_code,
                "error": "Murf API error",
                "body": body
            }

        result = resp.json()
        return {
            "success": True,
            "audio_file": result.get("audioFile"),
            "encoded_audio": result.get("encodedAudio"),
            "audio_length_seconds": result.get("audioLengthInSeconds"),
            "warning": result.get("warning", None),
        }

    def save_audio(
        self,
        encoded_audio: str,
        folder: str = "audios",
        filename: str = "ai_response.mp3"
    ) -> str:
        """
        Saves base64 encoded audio to a file in the specified folder, overwriting if exists.

        Returns the full path to the saved file.
        """
        if not encoded_audio:
            raise ValueError("encoded_audio must not be empty.")

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        with open(path, "wb") as f:
            f.write(base64.b64decode(encoded_audio))
        print(f"Audio saved successfully at {path}")
        return path