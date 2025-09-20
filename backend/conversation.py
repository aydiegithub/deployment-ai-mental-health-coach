from typing import List, Dict, Optional
from backend.system_instruction import SystemInstruction, TherapeuticTechnique
from backend.gemini_client import get_gemini_chat_completion

class GeminiChatSession:
    def __init__(self, instruction: SystemInstruction, techniques: List[TherapeuticTechnique]):
        self.instruction = instruction
        self.techniques = techniques
        self.chat_history: List[Dict] = []
        system_msg = (
            f"{self.instruction.role}\n"
            f"{self.instruction.core_principles}\n"
            f"{self.instruction.therapeutic_approach}\n"
            f"{self.instruction.communication_style}\n"
            f"{self.instruction.intervention_strategies}\n"
            f"{self.instruction.ethical_boundaries}\n"
            f"{self.instruction.crisis_management}\n"
        )
        self.chat_history.append({"role": "model", "parts": [{"text": system_msg}]})

    def get_phase_intro(self) -> str:
        return f"{self.instruction.core_principles}\n{self.instruction.assessment_framework}"

    def add_user_message(self, user_message: str) -> Optional[str]:
        user_message = user_message.strip()
        if not user_message:
            return None
        self.chat_history.append({"role": "user", "parts": [{"text": user_message}]})
        # Safety check
        for trigger in self.instruction.safety_protocols.trigger_words:
            if trigger in user_message.lower():
                safety_msg = self.instruction.safety_protocols.response_template
                self.chat_history.append({"role": "model", "parts": [{"text": safety_msg}]})
                return safety_msg
        return None

    def generate_solution(self) -> str:
        return get_gemini_chat_completion(self.chat_history)

    def run_chat(self, user_messages: List[str]) -> dict:
        phase_intro = self.get_phase_intro()
        safety_warnings = []
        for user_message in user_messages:
            warning = self.add_user_message(user_message)
            if warning:
                safety_warnings.append(warning)
        solution = self.generate_solution()
        return {
            "phase_intro": phase_intro,
            "safety_warnings": safety_warnings,
            "solution": solution
        }