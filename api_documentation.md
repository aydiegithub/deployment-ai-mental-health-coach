# Mental Health AI Coach API Documentation

## Endpoint

`POST http://localhost:5000/chat`

## Request

Send a JSON object with:

| Field         | Type   | Required | Description                                                  |
| ------------- | ------ | -------- | ------------------------------------------------------------ |
| user_message  | string | Yes      | For text: the user's message.<br>For audio: file path of audio stored in frontend folder. |
| dtype         | string | Yes      | `"message"` for text, `"audio"` for audio file               |

**Examples**

### For Text Message

```json
{
  "user_message": "I feel stressed and anxious.",
  "dtype": "message"
}
```

### For Audio Message

```json
{
  "user_message": "audios/my_audio_file.mp3",
  "dtype": "audio"
}
```
*Note: The file path must point to a valid audio file on the server (same folder as frontend code).*

---

## Response

### For Text Message (`dtype: "message"`)

```json
{
  "content": "AI therapist response text.",
  "type": "message"
}
```

### For Audio Message (`dtype: "audio"`)

```json
{
  "content": "AI therapist response text.",
  "audio_filepath": "audios/ai_response.mp3",
  "transcribed_text": "Transcribed text from user's audio.",
  "type": "audio"
}
```

---

## Error Examples

```json
{ "error": "Missing JSON body" }
{ "error": "Invalid dtype, must be 'audio' or 'message'" }
{ "error": "Missing or empty user_message" }
{ "error": "Audio file not found: audios/my_audio_file.mp3" }
{ "error": "AI response generation failed: ..." }
{ "error": "Audio generation failed: ..." }
```

---

## Frontend Integration Example (JavaScript / Fetch)

```javascript
// For text message
fetch("http://localhost:5000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    user_message: "I feel stressed and anxious.",
    dtype: "message"
  })
}).then(response => response.json())
  .then(data => console.log(data));

// For audio message (file path must exist on backend)
fetch("http://localhost:5000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    user_message: "audios/my_audio_file.mp3",
    dtype: "audio"
  })
}).then(response => response.json())
  .then(data => console.log(data));
```

---

## Notes For Frontend Team

- **Audio files** must be uploaded to and accessible in the `audios/` directory of the backend before calling the API with their file path.
- The backend returns the AI therapist's response as text for both text and audio inputs.
- For audio, backend also returns the transcribed text and the generated AI audio file path.

If you have any questions about API parameters or error handling, ask the backend team!