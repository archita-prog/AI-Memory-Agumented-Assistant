import json
import os
import sys
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    from groq import Groq
except ImportError:
    print("Missing required Python package 'groq'.")
    print("Please create and activate a virtual environment and install dependencies:")
    print("  python -m venv .venv")
    print("  .venv\\Scripts\\Activate.ps1   (PowerShell)")
    print("  pip install -r requirements.txt")
    sys.exit(1)
# .venv\Scripts\python.exe app.py - to run the app
# .venv\Scripts\Activate.ps1 - for getting in venv
try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing required Python package 'python-dotenv'.")
    print("Install it inside your environment with: pip install python-dotenv")
    sys.exit(1)

load_dotenv()
client = Groq(api_key = os.getenv("GROQ_API_KEY"))

MEMORY_FILE = "memory.json"
# session flag to control whether past memories are included in prompts
recall_enabled = True
EMOTION_FILE = "emotion.json"

# simple mood lexicons for heuristic updates
POSITIVE_WORDS = ["happy", "great", "awesome", "good", "fantastic", "love", "yay", "thanks", "thank", "wonderful", "beautiful", "excellent"]
NEGATIVE_WORDS = ["sad", "bad", "angry", "hate", "upset", "terrible", "no", "don't", "dont", "angst", "awful", "horrible", "disgusting"]
NEUTRAL_WORDS = ["ok", "okay", "meh", "fine", "alright", "so-so"]

# helper to ensure memory file exists
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump([], f)

# ensure emotion file exists
if not os.path.exists(EMOTION_FILE):
    with open(EMOTION_FILE, "w") as f:
        json.dump({"valence": 0.0, "history": []}, f)

def load_memory():
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def store_memory(user_input):
    memory = load_memory()

    embedding = embedding_model.encode(user_input).tolist()

    memory_entry = {
        "text": user_input,
        "embedding": embedding,
        "timestamp": str(datetime.now())
    }

    memory.append(memory_entry)
    save_memory(memory)
    

def load_emotion():
    with open(EMOTION_FILE, "r") as f:
        return json.load(f)

def save_emotion(emotion):
    with open(EMOTION_FILE, "w") as f:
        json.dump(emotion, f, indent=2)

def valence_to_label(valence):
    if valence >= 0.6:
        return "ecstatic", "🤩"
    if valence >= 0.3:
        return "happy", "😊"
    if valence >= 0.0:
        return "content", "🙃"
    if valence > -0.2:
        return "neutral", "😐"
    if valence > -0.4:
        return "melancholy", "😔"
    if valence > -0.6:
        return "sad", "😢"
    return "upset", "😠"

def update_emotion_from_text(emotion, text):
    text_l = text.lower()
    pos = sum(text_l.count(w) for w in POSITIVE_WORDS)
    neg = sum(text_l.count(w) for w in NEGATIVE_WORDS)
    neu = sum(text_l.count(w) for w in NEUTRAL_WORDS)
    delta = (pos - neg) * 0.1 + neu * 0.02
    val = emotion.get("valence", 0.0) + delta
    val = max(-1.0, min(1.0, val))
    # decay emotion slightly toward neutral over time
    val *= 0.92
    emotion["valence"] = val
    return emotion
def retrieve_relevant_memories(query, top_k=3):

    memory = load_memory()

    if len(memory)==0:
        return ""

    query_embedding = embedding_model.encode(query)

    similarities = []

    for m in memory:

        score = cosine_similarity(
            [query_embedding],
            [m["embedding"]]
        )[0][0]

        similarities.append((score,m["text"]))

    similarities.sort(reverse=True)

    top_memories = [m[1] for m in similarities[:top_k]]

    return "\n".join(top_memories)


def generate_response(user_input, past_memory):
    # include AI mood into the prompt so the model can reflect it in tone
    emotion = load_emotion()
    valence = emotion.get("valence", 0.0)
    mood_label, mood_emoji = valence_to_label(valence)
    prompt = f"""
You are a personal AI assistant with memory and feelings.
AI mood: {mood_label} ({mood_emoji})  # valence={valence}
User's past memories:
{past_memory}

Current message:
{user_input}

When you respond, subtly reflect the AI's mood in tone (e.g., upbeat when happy, gentle when sad), while remaining helpful and accurate.
Respond in a helpful and personalized way.
    """
    chat_completion = client.chat.completions.create(
        messages = [{"role": "user", "content": prompt}],
        model = "llama-3.1-8b-instant"
    )
    return chat_completion.choices[0].message.content

print("Cognitive AI Memory Assistant with Emotions")
print("Type 'exit' to stop.")
print("Commands: /disable_recall, /enable_recall, /clear_memory, /mood, /reset_mood\n")

while True:
    user_input = input("You:")
    if user_input.lower() == "exit":
        break

    cmd = user_input.strip().lower()
    if cmd in ("/disable_recall", "disable recall", "disable_recall"):
        recall_enabled = False
        print("AI: Memory recall disabled for this session.")
        continue
    if cmd in ("/enable_recall", "enable recall", "enable_recall"):
        recall_enabled = True
        print("AI: Memory recall enabled for this session.")
        continue
    if cmd in ("/clear_memory", "clear memory", "clear memories"):
        save_memory([])
        print("AI: All stored memories cleared.")
        continue
    if cmd in ("/mood", "mood", "check mood", "/check_mood"):
        emotion = load_emotion()
        mood_label, mood_emoji = valence_to_label(emotion.get("valence", 0.0))
        valence = emotion.get("valence", 0.0)
        print(f"AI {mood_emoji} ({mood_label}): I'm feeling {mood_label}. (Emotional depth: {valence:.2f})")
        continue
    if cmd in ("/reset_mood", "reset mood", "reset emotion"):
        save_emotion({"valence": 0.0, "history": []})
        print("AI 😐 (neutral): My emotions have been reset to neutral. Starting fresh!")
        continue

    # store non-command inputs
    store_memory(user_input)

    # load and update emotion based on the latest user input, then save
    emotion = load_emotion()
    emotion = update_emotion_from_text(emotion, user_input)
    save_emotion(emotion)

    past_memory = retrieve_relevant_memories(user_input)
    reply = generate_response(user_input, past_memory)
    mood_label, mood_emoji = valence_to_label(emotion.get("valence", 0.0))
    print(f"\nAI {mood_emoji} ({mood_label}):", reply)
    print()