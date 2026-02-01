📝 AI-Powered Journaling Companion

A privacy-first journaling companion that uses emotionally accurate, deterministic NLP to support reflection, encourage consistent journaling, and surface meaningful patterns over time.

This project prioritizes emotional accuracy, trust, and insight over time rather than generic sentiment or overconfident AI responses.


✨ Key Features

Emotion-aware reflections (without labeling the user)

Dynamic, empathetic prompts adapted to emotional context

Gentle engagement reinforcement (no streaks or gamification)

Pattern aggregation across recent entries

Reflection summaries that help users notice recurring themes

Fully local, deterministic NLP (no LLMs for emotional classification)


🧠 Design Philosophy (Quick Overview)

Understanding before expression: emotional inference guides tone, not labels

Reflection over correction: no advice, diagnosis, or prescriptions

Insight over time: patterns matter more than individual entries

Privacy by design: no external APIs for emotional understanding


🛠️ Tech Stack

Language: Python

Framework: FastAPI

NLP: TextBlob + rule-based logic

AI Models: Deterministic NLP only (no LLMs for emotion)

Interface: REST API with interactive Swagger UI

Storage: Lightweight in-memory persistence


🚀 Getting Started (Run Locally)

1️⃣ Clone the repository
git clone <your-repo-url> // https://github.com/aditis99/AI-powered-Journaling-Companion.git

cd <your-repo-name> // AI Journaling Companion

2️⃣ Create a virtual environment (recommended)
python -m venv venv

source venv/bin/activate   # Mac/Linux

venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Start the server
python main.py


The server will run at:

http://localhost:8000


📘 Interactive Demo (Recommended for Judges)

Open the interactive API documentation:

http://localhost:8000/docs


This provides a live demo interface to submit journal entries and observe system behavior.


▶️ How to Use the Demo (Step-by-Step)

Use the POST /entries endpoint.

Each request requires a simple JSON body:

{
  "content": "Your journal entry text here"
}


🧪 Suggested Prompts to Test Core Features

These prompts are designed to demonstrate all success metrics.

🔹 1. First Entry (Baseline Reflection)

Prompt:

Feeling tired and unmotivated today.


What to observe:

Empathetic reflection

No engagement note yet

No reflection summary

🔹 2. Engagement Reinforcement

Prompt:

Still low energy today, mostly scrolling and waiting for the day to end.


What to observe:

engagement_note appears

Gentle acknowledgment of consistency

No pressure or gamification

🔹 3. Pattern Detection & Reflection Summary

Prompt:

Another day of feeling flat and disconnected.


What to observe:

reflection_summary appears

Mentions recurring emotional tone

No advice or diagnosis

🔹 4. Anxiety via Rumination (Cognitive Looping)

Prompt:

My mind keeps looping on everything I should have done differently.


What to observe:

Grounding reflection

No explicit labeling of “anxiety”

Prompt focused on present awareness

🔹 5. Pressure Without Anxiety (Low-Energy Avoidance)

Prompt:

There’s so much to do, but I don’t feel like doing any of it.


What to observe:

Classified as low-energy, not anxious

Validating, low-effort prompt

🔹 6. Numbness / Emotional Absence

Prompt:

I don’t feel sad or happy, just kind of blank and passing the time.


What to observe:

Sentiment correctly marked as neutral

No false “positive” sentiment

Calm, non-activating response


🔐 Privacy & Responsible AI

Emotional understanding is fully rule-based and deterministic

No generative model decides how the user feels

Emotional modes exist internally only

GPT (if used) assists with phrasing, never interpretation

No user data leaves the system

The system separates emotional understanding from expression to reduce the risk of emotional mislabeling.


📈 Future Enhancements

Visual trend summaries

User-controlled reflection horizons

Optional goal-based reflections

Lightweight web or mobile UI

On-device deployment for personal journaling

📌 Final Note

This project is intentionally minimal in interface and maximal in emotional care.

The goal was not to build a journaling app that talks, but one that listens, carefully.

🆚 How This Differs from Typical AI Journaling Apps

- No emotional guessing via LLMs
- No advice or behavioral prescriptions
- No sentiment labels shown to users
- No gamification or streak pressure
- No external API dependency for emotional understanding

This system prioritizes emotional safety and long-term insight over immediate engagement metrics.

🎯 Why This Demo Uses an API Instead of a UI

The core challenge was emotional correctness and insight quality — not interface design.

Using Swagger UI allows judges to:
- Test real inputs
- Observe real outputs
- Validate behavior deterministically

This keeps the demo focused on the success metrics that matter.

Design Documentation Link: https://docs.google.com/document/d/148SHDRkfGgS-k_l0H4qo9_snu4Igx-k9cQscC0hSvE4/edit?usp=sharing

Video Presentation: https://vimeo.com/1160842661?share=copy&fl=sv&fe=ci
