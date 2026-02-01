import random
from typing import Optional
from models.schemas import SentimentAnalysis, ThemeDetection, EmpathyReflection


# Empathetic acknowledgment templates organized by sentiment label
# Rewritten to be warmer, more emotionally validating, and human
# Focus on acknowledging the effort of journaling and reflecting complexity
ACKNOWLEDGMENT_TEMPLATES = {
    "positive": [
        "There's a warmth in what you've written, and it feels genuine.",
        "Something lighter is present here, and you're noticing it.",
        "This reads like a moment where things feel a bit easier, even if just for now.",
        "You're catching something good as it passes through — that kind of noticing matters.",
    ],
    "neutral": [
        "You showed up to write today, even without a clear reason to. That counts.",
        "There's something steady in taking time to check in, even when nothing stands out.",
        "This feels like you're just observing where you are, without needing it to mean anything yet.",
        "You're here, putting words to something that doesn't have a clear shape. That's enough.",
    ],
    "negative": [
        "This feels heavy, and it makes sense that it would.",
        "You're sitting with something difficult right now, and that takes more than it looks like.",
        "There's weight in what you've written, and you don't have to carry it alone or make sense of it yet.",
        "Something hard is present here, and you're letting it be seen. That matters.",
    ]
}

# Theme-aware additions that can be woven into acknowledgments
# Rewritten to feel more naturally integrated and emotionally present
# These validate specific themes without making assumptions or giving advice
THEME_ADDITIONS = {
    "gratitude": "noticing what you're grateful for, even in small moments",
    "stress": "holding a lot right now, and feeling the weight of it",
    "relationships": "thinking about the people who matter to you",
    "work": "processing what's happening in your professional life",
    "health": "paying attention to how you're feeling physically and emotionally",
    "growth": "noticing where you're changing or what you're learning",
    "creativity": "connecting with your creative side and what that brings up",
    "loss": "sitting with something painful, and that's not easy"
}

# Open-ended reflective prompts that encourage further exploration
# Rewritten to be more emotionally aware, gentle, and engaging
# These are non-prescriptive and don't instruct the user what to do
REFLECTIVE_PROMPTS = [
    "What else feels present as you sit with this?",
    "Is there more that wants to be said?",
    "What are you noticing as you write?",
    "How does it feel to put this into words?",
    "What else is coming up for you right now?",
    "What feels most alive in this moment?",
]

# Mode-adaptive reflection templates for emotionally accurate responses
# These replace generic sentiment-based templates when emotional mode is detected
# Focus: Mirror emotional state accurately, lower cognitive load, avoid premature reframing
MODE_ADAPTIVE_TEMPLATES = {
    "low_energy": [
    "This feels like one of those in-between days — not heavy, not light, just quietly passing. Nothing here needs to turn into insight.",
    "It sounds like energy never really showed up today. You’re still here, still noticing — and that’s enough for now.",
    "There’s a flat, tired quality here that doesn’t ask for fixing. Some days are just meant to be moved through slowly.",
    "This reads like a day without much pull in either direction. You’re allowed to let it be exactly that."
    ],
    "anxious": [
    "Your thoughts seem busy and repetitive right now, like they don’t know where to land. Noticing that weight is enough for this moment.",
    "It sounds like your mind has been looping — carrying things that don’t need to be resolved all at once.",
    "There’s a sense of mental pressure here, of holding a lot internally. You don’t have to untangle any of it right now.",
    "This feels mentally heavy, even if nothing specific stands out. It’s okay to pause without making sense of it."
    ],
    "calm": [
    "There’s a reflective tone in what you’ve written — like you’re observing rather than reacting.",
    "You’re noticing small patterns in how you feel, and that kind of awareness tends to grow quietly over time.",
    "This reads like a moment of clarity, without urgency. Sometimes that’s where insight begins.",
    "You’re taking a step back and seeing things as they are. That kind of noticing has its own value."
    ]
}

# Mode-specific prompts that adapt to emotional state and cognitive capacity
# LOW_ENERGY: Small, optional, low-effort prompts that don't demand insight
# ANXIOUS: Present-focused, grounding prompts that don't add to mental load
# CALM: Open-ended, exploratory prompts that invite gentle reflection
MODE_ADAPTIVE_PROMPTS = {
    "low_energy": [
        "What's one thing you're letting yourself not do today?",
        "If nothing had to be productive right now, what would that feel like?",
        "Is there anything small that might make this day feel a little less flat?",
        "What would it look like to just move through today without expecting much from it?",
    ],
    "anxious": [
        "What's one thing around you right now that feels steady, even if it's small?",
        "Is there anything you're holding that doesn't need an answer today?",
        "What would it feel like to pause here, without trying to solve anything?",
        "What are you noticing in your body or your surroundings in this exact moment?",
    ],
    "calm": [
        "What else is present as you sit with this?",
        "Is there more here that wants to be noticed?",
        "What are you seeing now that you might not have seen before?",
        "What feels most alive or real in this moment?",
    ]
}


def generate_reflection(
    sentiment: SentimentAnalysis,
    themes: ThemeDetection,
    mode: Optional[str] = None
) -> EmpathyReflection:
    """
    Generate an empathetic reflection based on sentiment and themes.
    
    This function uses deterministic, template-based logic to create
    validating, non-judgmental responses. No AI/ML generation, no
    personalization, no learning from user data.
    
    Responsible AI boundaries:
    - No advice-giving or prescriptive guidance
    - No diagnosis or clinical language
    - No predictions about future behavior
    - No judgment of emotions or experiences
    - Validates user's experience without interpretation
    
    The reflection consists of:
    1. An empathetic acknowledgment based on sentiment
    2. Optional theme-aware context (if themes detected)
    3. An optional open-ended reflective prompt
    
    Mode-adaptive behavior (new):
    - If emotional mode provided, uses mode-adaptive templates for better emotional accuracy
    - Falls back to sentiment-based templates if mode is None or unrecognized
    - Mode-specific prompts lower cognitive load for low-energy/anxious states
    
    Deterministic behavior:
    - Uses sentiment label or mode to select acknowledgment category
    - Uses theme count to decide whether to add theme context
    - Uses random.seed() based on sentiment polarity for reproducibility
    
    Args:
        sentiment: SentimentAnalysis from NLP service
        themes: ThemeDetection from NLP service
        mode: Optional emotional mode ("low_energy", "anxious", "calm")
    
    Returns:
        EmpathyReflection: Pydantic model with message and optional prompt
    
    Example:
        reflection = generate_reflection(
            sentiment=SentimentAnalysis(polarity=0.5, subjectivity=0.6, label="positive"),
            themes=ThemeDetection(themes=["gratitude"], confidence="medium")
        )
        # Returns validating, non-judgmental reflection
    """
    # Select acknowledgment template based on mode (if provided) or sentiment label
    # Mode-adaptive templates provide better emotional accuracy for specific states
    if mode and mode in MODE_ADAPTIVE_TEMPLATES:
        acknowledgment_options = MODE_ADAPTIVE_TEMPLATES[mode]
    else:
        # Fallback to sentiment-based templates
        acknowledgment_options = ACKNOWLEDGMENT_TEMPLATES[sentiment.label]
    
    # Use polarity as seed for deterministic selection (same input = same output)
    # Convert polarity to integer seed: multiply by 1000 and take absolute value
    seed_value = int(abs(sentiment.polarity * 1000))
    random.seed(seed_value)
    
    base_acknowledgment = random.choice(acknowledgment_options)
    
    # Add theme-aware context if themes were detected with medium/high confidence
    message = base_acknowledgment
    if themes.themes and themes.confidence in ["medium", "high"]:
        # Select the first theme (themes are alphabetically ordered, so deterministic)
        primary_theme = themes.themes[0]
        
        if primary_theme in THEME_ADDITIONS:
            theme_context = THEME_ADDITIONS[primary_theme]
            # Integrate theme context naturally into the message
            message = f"{base_acknowledgment} I notice you're {theme_context}."
    
    # Generate optional reflective prompt
    # Mode-adaptive prompts lower cognitive load for low-energy/anxious states
    # Include prompt for neutral/negative sentiment, or when confidence is low
    prompt: Optional[str] = None
    
    # Use mode-specific prompts if mode is provided
    if mode and mode in MODE_ADAPTIVE_PROMPTS:
        # Always include prompt for low_energy and anxious modes (they need gentle guidance)
        # For calm mode, use same logic as before (neutral/negative/low confidence)
        if mode in ["low_energy", "anxious"]:
            prompt_seed = int(abs(sentiment.subjectivity * 1000))
            random.seed(prompt_seed)
            prompt = random.choice(MODE_ADAPTIVE_PROMPTS[mode])
        elif mode == "calm" and (sentiment.label in ["neutral", "negative"] or themes.confidence == "low"):
            prompt_seed = int(abs(sentiment.subjectivity * 1000))
            random.seed(prompt_seed)
            prompt = random.choice(MODE_ADAPTIVE_PROMPTS[mode])
    else:
        # Fallback to original prompt logic
        if sentiment.label in ["neutral", "negative"] or themes.confidence == "low":
            prompt_seed = int(abs(sentiment.subjectivity * 1000))
            random.seed(prompt_seed)
            prompt = random.choice(REFLECTIVE_PROMPTS)
    
    return EmpathyReflection(
        message=message,
        prompt=prompt
    )


def generate_reflection_simple(sentiment_label: str) -> EmpathyReflection:
    """
    Generate a simple reflection based only on sentiment label.
    
    This is a simplified version for cases where theme detection
    is not available or not needed. Uses only sentiment for response.
    
    Maintains all Responsible AI boundaries (non-judgmental, validating,
    non-prescriptive).
    
    Args:
        sentiment_label: One of "positive", "neutral", or "negative"
    
    Returns:
        EmpathyReflection: Pydantic model with message and optional prompt
    
    Example:
        reflection = generate_reflection_simple("positive")
        # Returns basic validating acknowledgment
    """
    # Use first template from each category for simplicity
    acknowledgment = ACKNOWLEDGMENT_TEMPLATES[sentiment_label][0]
    
    # Add prompt for neutral/negative sentiment
    prompt: Optional[str] = None
    if sentiment_label in ["neutral", "negative"]:
        prompt = REFLECTIVE_PROMPTS[0]
    
    return EmpathyReflection(
        message=acknowledgment,
        prompt=prompt
    )
