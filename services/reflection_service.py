import random
from typing import Optional
from models.schemas import SentimentAnalysis, ThemeDetection, EmpathyReflection


# Empathetic acknowledgment templates organized by sentiment label
# These are validating and non-judgmental, avoiding advice or diagnosis
ACKNOWLEDGMENT_TEMPLATES = {
    "positive": [
        "It sounds like you're experiencing some positive moments.",
        "There seems to be a sense of lightness in what you've shared.",
        "It appears that some things are going well for you.",
        "Your words carry a sense of positivity.",
    ],
    "neutral": [
        "Thank you for sharing your thoughts.",
        "I appreciate you taking the time to reflect.",
        "Your reflections have been noted.",
        "It's valuable that you're taking time to journal.",
    ],
    "negative": [
        "It seems like things feel heavy right now.",
        "It sounds like you're going through a difficult time.",
        "Your feelings come through in what you've shared.",
        "It appears that you're experiencing some challenges.",
    ]
}

# Theme-aware additions that can be woven into acknowledgments
# These validate specific themes without making assumptions or giving advice
THEME_ADDITIONS = {
    "gratitude": "noticing moments of appreciation",
    "stress": "navigating some pressure",
    "relationships": "thinking about the people in your life",
    "work": "reflecting on your professional experiences",
    "health": "considering your well-being",
    "growth": "exploring personal development",
    "creativity": "engaging with creative expression",
    "loss": "processing difficult emotions"
}

# Open-ended reflective prompts that encourage further exploration
# These are non-prescriptive and don't instruct the user what to do
REFLECTIVE_PROMPTS = [
    "What else comes to mind?",
    "How does this sit with you?",
    "What are you noticing as you reflect?",
    "Is there more you'd like to explore?",
    "What feels important about this?",
    "What else is present for you?",
]


def generate_reflection(
    sentiment: SentimentAnalysis,
    themes: ThemeDetection
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
    
    Deterministic behavior:
    - Uses sentiment label to select acknowledgment category
    - Uses theme count to decide whether to add theme context
    - Uses random.seed() based on sentiment polarity for reproducibility
    
    Args:
        sentiment: SentimentAnalysis from NLP service
        themes: ThemeDetection from NLP service
    
    Returns:
        EmpathyReflection: Pydantic model with message and optional prompt
    
    Example:
        reflection = generate_reflection(
            sentiment=SentimentAnalysis(polarity=0.5, subjectivity=0.6, label="positive"),
            themes=ThemeDetection(themes=["gratitude"], confidence="medium")
        )
        # Returns validating, non-judgmental reflection
    """
    # Select acknowledgment template based on sentiment label
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
    # Include prompt for neutral/negative sentiment, or when confidence is low
    # This encourages further exploration when the entry might benefit from it
    prompt: Optional[str] = None
    
    if sentiment.label in ["neutral", "negative"] or themes.confidence == "low":
        # Use subjectivity as seed for prompt selection (deterministic)
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
