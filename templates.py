import os

# Define the structure (no extra root folder)
project_structure = {
    "backend": {
        "app.py": "# Entry point for backend server\n",
        "requirements.txt": "# Add your backend dependencies here\n",
        ".env.example": "# Example environment variables\nSECRET_KEY=your_secret_key\n",
        "README.md": "# Backend README\n\nThis directory contains the backend of the AI Mental Health Coach project."
    },
    "frontend": {
        "public": {},  # empty folder
        "src": {
            "components": {
                "VoiceChatPanel.jsx": "// Voice-based chat component\n",
                "TextChatPanel.jsx": "// Text-based chat component\n",
                "ConversationHistory.jsx": "// Display chat history\n",
                "MicButton.jsx": "// Microphone button component\n",
            },
            "App.jsx": "// Main React app\n",
            "index.js": "// React entry point\n"
        },
        "package.json": '{\n  "name": "frontend",\n  "version": "1.0.0"\n}\n',
        "README.md": "# Frontend README\n\nThis directory contains the frontend of the AI Mental Health Coach project."
    },
    "diagrams": {
        "data_flow.png": None,  # placeholder file
        "context_flow.png": None
    },
    "README.md": "# AI Mental Health Coach\n\nThis is the root README for the project."
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):  
            # Create directory
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)  
        else:
            # Create file
            with open(path, "w", encoding="utf-8") as f:
                if content is not None:
                    f.write(content)

if __name__ == "__main__":
    base_dir = os.getcwd()  # current working directory
    create_structure(base_dir, project_structure)
    print("✅ Project structure created successfully in:", base_dir)