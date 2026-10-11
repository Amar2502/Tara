# Tara - Groq & LangChain Conversational Assistant

A clean, modular conversational assistant built with **LangChain (LCEL)**, **ChatGroq**, and **Rich**, designed to be easy to understand and learn from.

---

## Project Structure

```
Tara/
│
├── config.py         # 1. Reads environment variables from .env
├── llm.py            # 2. Initializes ChatGroq model using LangChain
├── assistant.py      # 3. Core logic: Prompt + LangChain chain + Memory
├── main.py           # 4. Entry point: Rich terminal chat loop & live Markdown
│
├── .env              # Secret API keys & configs
├── requirements.txt  # Dependencies
└── .gitignore        # Git ignore rules
```

---

## How It Works (Step-by-Step)

1. **[config.py](file:///p:/Tara/config.py)**: Loads your `GROQ_API_KEY`, `GROQ_MODEL`, and `GROQ_TEMPERATURE` from `.env`.
2. **[llm.py](file:///p:/Tara/llm.py)**: Sets up `ChatGroq(model=..., streaming=True)` using the configuration.
3. **[assistant.py](file:///p:/Tara/assistant.py)**:
   - Sets up conversation memory list (`self.history`).
   - Uses `ChatPromptTemplate` with `MessagesPlaceholder` to keep past messages.
   - Combines them using LangChain Expression Language (LCEL):
     `chain = prompt | llm | StrOutputParser()`
   - Streams answers and updates memory after every turn.
4. **[main.py](file:///p:/Tara/main.py)**:
   - Takes input using `console.input()`.
   - Streams responses in real-time with **Rich Markdown** syntax highlighting.
   - Handles commands like `/clear`, `/status`, `/help`, and `exit`.

---

## Configuration (`.env`)

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.7
```

---

## Running the Assistant

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

### Commands inside chat:
- `/clear` or `/reset`: Clears conversation history.
- `/status`: Shows active model and message count.
- `/help`: Displays available commands.
- `exit` or `quit`: Exits the session.
