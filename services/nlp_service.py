import re
from textblob import TextBlob
from typing import List, Tuple
from models.schemas import SentimentAnalysis, ThemeDetection


# Sentiment thresholds chosen to create a neutral zone around zero.
# Rationale: TextBlob polarity scores near zero are often ambiguous or mixed.
# A ±0.1 threshold reduces false positives for clearly positive/negative labels.
# This provides more conservative, explainable sentiment classification.
SENTIMENT_POSITIVE_THRESHOLD = 0.1
SENTIMENT_NEGATIVE_THRESHOLD = -0.1


THEME_KEYWORDS = {
    "gratitude": [
        "grateful", "thankful", "appreciate", "blessed", "fortunate",
        "thank", "gratitude", "appreciation", "lucky", "privilege"
    ],
    "stress": [
        "stress", "stressed", "anxious", "anxiety", "overwhelmed",
        "pressure", "worried", "worry", "tense", "burden", "exhausted"
    ],
    "relationships": [
        "friend", "family", "partner", "relationship", "love",
        "spouse", "boyfriend", "girlfriend", "husband", "wife",
        "mother", "father", "sister", "brother", "colleague"
    ],
    "work": [
        "work", "job", "career", "office", "project", "meeting",
        "deadline", "boss", "coworker", "professional", "business",
        "task", "assignment", "promotion", "interview"
    ],
    "health": [
        "health", "exercise", "workout", "fitness", "sleep",
        "diet", "nutrition", "doctor", "medical", "sick", "illness",
        "energy", "tired", "fatigue", "pain"
    ],
    "growth": [
        "learn", "growth", "improve", "progress", "develop",
        "goal", "achievement", "success", "accomplish", "better",
        "challenge", "opportunity", "potential", "skill"
    ],
    "creativity": [
        "create", "creative", "art", "music", "write", "writing",
        "paint", "design", "imagine", "inspiration", "project",
        "craft", "hobby", "express", "idea"
    ],
    "loss": [
        "loss", "grief", "sad", "sadness", "miss", "missing",
        "death", "died", "gone", "lost", "mourn", "mourning",
        "goodbye", "farewell", "end", "ending"
    ]
}


def analyze_sentiment(content: str) -> SentimentAnalysis:
    """
    Perform local sentiment analysis using TextBlob.
    
    This function uses TextBlob's built-in sentiment analyzer, which runs
    entirely locally with no external API calls. The analysis is deterministic
    and explainable based on word polarity scores.
    
    Privacy guarantee: All processing happens in-memory, no data leaves the system.
    
    Sentiment mapping:
    - polarity > 0.1: positive
    - polarity < -0.1: negative
    - polarity between -0.1 and 0.1: neutral
    
    Defensive handling:
    - Empty or whitespace-only strings return neutral sentiment (0.0, 0.0)
    - Very long texts are processed normally (TextBlob handles them)
    
    Args:
        content: Journal entry text to analyze
    
    Returns:
        SentimentAnalysis: Pydantic model with polarity, subjectivity, and label
    
    Example:
        sentiment = analyze_sentiment("Today was a wonderful day!")
        # Returns: SentimentAnalysis(polarity=0.85, subjectivity=1.0, label="positive")
    """
    # Defensive: Handle empty or whitespace-only content
    if not content or not content.strip():
        return SentimentAnalysis(
            polarity=0.0,
            subjectivity=0.0,
            label="neutral"
        )
    
    blob = TextBlob(content)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Apply threshold-based classification for coarse-grained labels
    if polarity > SENTIMENT_POSITIVE_THRESHOLD:
        label = "positive"
    elif polarity < SENTIMENT_NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    
    return SentimentAnalysis(
        polarity=polarity,
        subjectivity=subjectivity,
        label=label
    )


def _match_keyword_whole_word(keyword: str, text: str) -> bool:
    """
    Check if a keyword appears as a whole word in the text.
    
    Uses word boundaries (\b) to avoid substring false positives.
    For example, "stress" won't match "distress" or "stressful".
    
    Args:
        keyword: The keyword to search for (should be lowercase)
        text: The text to search in (should be lowercase)
    
    Returns:
        bool: True if keyword found as whole word, False otherwise
    """
    # Use word boundaries to match whole words only
    pattern = r'\b' + re.escape(keyword) + r'\b'
    return bool(re.search(pattern, text))


def detect_themes(content: str) -> ThemeDetection:
    """
    Detect themes using rule-based keyword matching.
    
    This function uses predefined keyword dictionaries to identify themes
    present in the journal entry. The logic is fully transparent and
    explainable - themes are detected based on keyword presence.
    
    No machine learning, no personalization, no learning from user data.
    The same keywords always map to the same themes (deterministic).
    
    Keyword matching:
    - Uses whole-word matching to avoid false positives (e.g., "stress" won't match "distress")
    - Case-insensitive for user convenience
    - Deterministic ordering (alphabetical by theme name) for stable output
    
    Confidence levels (based on max keyword count across all themes):
    - high: 3+ keywords detected for any single theme
    - medium: 2 keywords detected for any single theme
    - low: 1 keyword detected (or no themes found)
    
    Rationale for confidence levels:
    - Multiple keyword matches suggest stronger theme presence
    - Single keyword could be incidental mention
    - Confidence reflects detection certainty, not emotional intensity
    
    Defensive handling:
    - Empty or whitespace-only strings return empty themes with low confidence
    - Very long texts are processed normally (no truncation)
    
    Args:
        content: Journal entry text to analyze
    
    Returns:
        ThemeDetection: Pydantic model with detected themes and confidence
    
    Example:
        themes = detect_themes("I'm grateful for my family and friends")
        # Returns: ThemeDetection(themes=["gratitude", "relationships"], confidence="medium")
    """
    # Defensive: Handle empty or whitespace-only content
    if not content or not content.strip():
        return ThemeDetection(
            themes=[],
            confidence="low"
        )
    
    content_lower = content.lower()
    
    theme_keyword_counts = {}
    
    # Iterate through themes in deterministic order (sorted by key)
    for theme_name in sorted(THEME_KEYWORDS.keys()):
        keywords = THEME_KEYWORDS[theme_name]
        
        # Use whole-word matching to avoid substring false positives
        keyword_count = sum(
            1 for keyword in keywords 
            if _match_keyword_whole_word(keyword, content_lower)
        )
        
        if keyword_count > 0:
            theme_keyword_counts[theme_name] = keyword_count
    
    # Extract detected themes in deterministic alphabetical order
    detected_themes = sorted(theme_keyword_counts.keys())
    
    # Calculate confidence based on maximum keyword count across all themes
    max_keyword_count = max(theme_keyword_counts.values()) if theme_keyword_counts else 0
    
    if max_keyword_count >= 3:
        confidence = "high"
    elif max_keyword_count >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    
    return ThemeDetection(
        themes=detected_themes,
        confidence=confidence
    )


def analyze_entry(content: str) -> Tuple[SentimentAnalysis, ThemeDetection]:
    """
    Perform complete NLP analysis on a journal entry.
    
    Combines sentiment analysis and theme detection into a single
    convenient function. Both analyses run locally with no external calls.
    
    This is the main entry point for NLP processing in the application.
    
    Defensive handling:
    - Empty or whitespace-only content returns neutral sentiment and no themes
    - Both analyses handle edge cases independently
    - No exceptions thrown for valid string input
    
    Args:
        content: Journal entry text to analyze
    
    Returns:
        Tuple[SentimentAnalysis, ThemeDetection]: Both analysis results
    
    Example:
        sentiment, themes = analyze_entry("I'm grateful for today")
        print(f"Sentiment: {sentiment.label}")
        print(f"Themes: {themes.themes}")
    """
    sentiment = analyze_sentiment(content)
    themes = detect_themes(content)
    
    return sentiment, themes
