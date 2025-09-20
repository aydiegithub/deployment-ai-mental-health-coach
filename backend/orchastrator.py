from backend.system_instruction import get_advanced_therapist_instruction, get_therapeutic_techniques
from backend.conversation import GeminiChatSession

class Orchestrator:
    def __init__(self):
        self.instruction = get_advanced_therapist_instruction()
        self.techniques = get_therapeutic_techniques()
        self.session = GeminiChatSession(self.instruction, self.techniques)

    def start_session(self, user_messages: list) -> dict:
        return self.session.run_chat(user_messages)