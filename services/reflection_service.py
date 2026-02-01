import random
from typing import Optional
from models.schemas import SentimentAnalysis, ThemeDetection, EmpathyReflection


# Empathetic acknowledgment templates organized by sentiment label
# Rewritten to be warmer, more emotionally validating, and human
# Focus on acknowledging the effort of journaling and reflecting complexity
ACKNOWLEDGMENT_TEMPLATES = {
    "positive": [
        "There's a warmth in what you've shared today, and it comes through clearly.",
        "It sounds like you're experiencing some lightness right now, and that's meaningful.",
        "The positive energy in your words feels genuine and present.",
        "You're noticing good things around you, and taking time to acknowledge them matters.",
    ],
    "neutral": [
        "Thank you for taking the time to write today—it's an act of care in itself.",
        "You're showing up to reflect, even when things feel uncertain, and that takes effort.",
        "There's value in simply putting your thoughts into words, and you're doing that.",
        "You're making space to check in with yourself, and that's worth recognizing.",
    ],
    "negative": [
        "It sounds like things feel heavy right now, and it makes sense that you're feeling this way.",
        "You're navigating something difficult, and choosing to write about it is an act of courage.",
        "The weight of what you're experiencing comes through, and it's okay to feel this.",
        "You're sitting with some hard emotions, and that takes real strength.",
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
        "It sounds like one of those low-energy, restless days — tired but not quite sleepy, wanting something different without the energy to start it.",
        "This feels like a day where nothing quite lands, where you're present but not engaged, and that's okay.",
        "There's a flatness to today that doesn't need fixing — sometimes energy just isn't there, and that doesn't mean something is wrong.",
        "You're here, even when motivation isn't, and that counts for something.",
    ],
    "anxious": [
        "It sounds like your mind is carrying a lot right now, and the weight of it is real.",
        "There's a heaviness in what you're holding, and it makes sense that it feels this way.",
        "You're navigating something difficult, and the fact that you're writing about it matters.",
        "The worry and weight you're describing — it's present, and it's okay to acknowledge that without needing to resolve it right now.",
    ],
    "calm": [
        "There's a clarity in what you've shared, and it comes through in your words.",
        "You're noticing things as they are, and there's value in that kind of awareness.",
        "It sounds like you're in a reflective space, taking stock of where you are.",
        "There's a groundedness to what you've written, and that feels intentional.",
    ]
}

# Mode-specific prompts that adapt to emotional state and cognitive capacity
# LOW_ENERGY: Small, optional, low-effort prompts that don't demand insight
# ANXIOUS: Present-focused, grounding prompts that don't add to mental load
# CALM: Open-ended, exploratory prompts that invite gentle reflection
MODE_ADAPTIVE_PROMPTS = {
    "low_energy": [
        "If today didn't need to be productive, what's one small thing that might make it feel a little lighter?",
        "What's one thing you're not forcing yourself to do right now?",
        "Is there anything you're letting yourself off the hook for today?",
        "What would it look like to just be, without needing to accomplish anything?",
    ],
    "anxious": [
        "What's one thing that feels solid or steady right now, even if it's small?",
        "What are you noticing in your body or your surroundings in this moment?",
        "Is there anything you're holding that doesn't need to be solved today?",
        "What would it feel like to set this down, just for a moment?",
    ],
    "calm": [
        "What else feels present as you sit with this?",
        "Is there more that wants to be said?",
        "What are you noticing as you write?",
        "What feels most alive in this moment?",
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
