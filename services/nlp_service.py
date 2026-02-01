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

# Emotional burden keywords used for sentiment override
# Rationale: TextBlob's lexicon can misclassify stress/exhaustion as weakly positive
# due to words like "managed" or "accomplished" appearing alongside burden language.
# This list detects emotional burden to prevent false positive classifications.
EMOTIONAL_BURDEN_KEYWORDS = [
    "stress", "stressed", "stressful", "anxious", "anxiety", "overwhelmed",
    "overwhelming", "pressure", "pressured", "worried", "worry", "worrying",
    "tense", "tension", "burden", "burdened", "exhausted", "exhausting",
    "exhaustion", "drained", "burnt out", "burnout", "struggling", "struggle",
    "difficult", "hard", "tough", "heavy", "weighed down", "tired", "fatigue",
    "frustrated", "frustration", "frustrating"
]

# Low-energy / apathy keywords for emotional mode detection
# Rationale: Detect disengagement, restlessness, and low motivation states
# These indicate need for low-effort, validating responses rather than open-ended prompts
LOW_ENERGY_KEYWORDS = [
    "bored", "boring", "unmotivated", "no motivation", "don't feel like",
    "can't be bothered", "meh", "whatever", "sleepy", "restless",
    "scrolling", "waiting", "nothing", "empty", "numb", "flat",
    "disengaged", "disconnected", "apathetic", "indifferent",
    "procrastinating", "avoiding", "distracted", "unfocused"
]

# Rumination keywords for ANXIOUS mode detection
# Rationale: Detect cognitive looping, overthinking, and self-judgment
# Anxiety often appears as mental loops rather than strong negative emotion
RUMINATION_KEYWORDS = [
    "keep looping", "looping", "can't stop thinking", "should have",
    "should be", "overthinking", "stuck thinking", "running through my head",
    "can't get out of my head", "replaying", "going over and over"
]

# Explicit anxiety keywords for ANXIOUS mode detection
# Rationale: Direct anxiety language that indicates worry/stress state
# Used to distinguish anxiety from low-energy avoidance
EXPLICIT_ANXIETY_KEYWORDS = [
    "anxious", "anxiety", "worried", "worry", "worrying", "panic",
    "panicking", "stressed", "stress", "stressful", "tense", "tension",
    "nervous", "on edge", "freaking out"
]

# Pressure keywords for ANXIOUS mode detection
# Rationale: Detect feeling behind, overwhelmed by demands, time pressure
# NOTE: Pressure alone does NOT trigger ANXIOUS - must be combined with rumination or explicit anxiety
# This prevents misclassifying low-energy avoidance as anxiety
PRESSURE_KEYWORDS = [
    "so much to do", "feel behind", "behind on", "too much",
    "overwhelmed", "pressure", "falling behind", "can't keep up",
    "drowning in", "buried in", "swamped"
]

# Numbness / emotional absence keywords for sentiment override
# Rationale: TextBlob incorrectly labels emotionally absent or numb states as positive
# due to negated emotion words (e.g., "don't feel sad" → positive polarity)
# This override forces neutral sentiment when emotional numbness is detected
NUMBNESS_KEYWORDS = [
    "blank", "numb", "empty", "flat",
    "just existing", "passing the time",
    "nothing feels", "emotionless", "void",
    "not really sad", "not really happy"
]


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
        "grief", "mourning", "miss", "missing",
        "death", "died", "gone", "lost someone",
        "farewell", "goodbye", "losing her", "losing him",
        "lost my", "lost her", "lost him"
    ]
}


def _detect_emotional_burden(content: str) -> bool:
    """
    Detect presence of emotional burden keywords in content.
    
    This helper function checks for stress, exhaustion, and overwhelm language
    that indicates emotional difficulty, even when TextBlob's polarity score
    might be weakly positive due to lexicon limitations.
    
    Used as a rule-based override to prevent misclassification of entries
    describing stress/burden as "positive" sentiment.
    
    Args:
        content: Journal entry text to check (will be lowercased)
    
    Returns:
        bool: True if emotional burden keywords detected, False otherwise
    
    Example:
        _detect_emotional_burden("Work was stressful but I managed")
        # Returns: True ("stressful" detected)
    """
    content_lower = content.lower()
    
    # Check for any emotional burden keywords using whole-word matching
    for keyword in EMOTIONAL_BURDEN_KEYWORDS:
        if _match_keyword_whole_word(keyword, content_lower):
            return True
    
    return False


def _detect_numbness(content: str) -> bool:
    """
    Detect presence of numbness / emotional absence keywords in content.
    
    This helper function checks for language indicating emotional flatness,
    numbness, or absence of feeling. TextBlob often misclassifies these as
    positive due to negated emotion words (e.g., "don't feel sad").
    
    Used as a rule-based override to force neutral sentiment when numbness
    is detected, preventing false positive classifications.
    
    Args:
        content: Journal entry text to check (will be lowercased)
    
    Returns:
        bool: True if numbness keywords detected, False otherwise
    
    Example:
        _detect_numbness("I don't feel sad or happy, just kind of blank")
        # Returns: True ("blank" detected)
    """
    content_lower = content.lower()
    
    # Check for any numbness keywords using substring matching
    # (some are multi-word phrases like "just existing")
    for keyword in NUMBNESS_KEYWORDS:
        if keyword in content_lower:
            return True
    
    return False


def analyze_sentiment(content: str) -> SentimentAnalysis:
    """
    Perform local sentiment analysis using TextBlob with rule-based overrides.
    
    This function uses TextBlob's built-in sentiment analyzer, which runs
    entirely locally with no external API calls. The analysis is deterministic
    and explainable based on word polarity scores.
    
    Privacy guarantee: All processing happens in-memory, no data leaves the system.
    
    Sentiment mapping with emotional burden override:
    - polarity > 0.1 AND no emotional burden detected: positive
    - polarity > 0.1 BUT emotional burden detected: neutral (override)
    - polarity < -0.1: negative
    - polarity between -0.1 and 0.1: neutral
    
    Emotional burden override rationale:
    TextBlob's lexicon can misclassify entries about stress/exhaustion as
    weakly positive when words like "managed", "accomplished", or "finished"
    appear alongside burden language. This override prevents false positives
    by detecting stress/overwhelm keywords and capping the label at "neutral"
    even when polarity is slightly positive.
    
    This aligns sentiment labels with human emotional intuition in journaling
    contexts: "I was stressed but managed to finish" should not be labeled
    "positive" even if TextBlob's polarity is +0.15.
    
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
        
        sentiment = analyze_sentiment("Work was stressful but I managed to finish.")
        # Returns: SentimentAnalysis(polarity=0.15, subjectivity=0.5, label="neutral")
        # (Override applied: emotional burden detected despite positive polarity)
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
    
    # Detect emotional burden for potential override
    has_emotional_burden = _detect_emotional_burden(content)
    
    # Detect numbness / emotional absence for sentiment override
    has_numbness = _detect_numbness(content)
    
    # FIX: Numbness override - force neutral sentiment for emotionally absent states
    # Rationale: TextBlob misclassifies numb/flat states as positive due to negated emotions
    # (e.g., "don't feel sad" → positive polarity, but actually indicates numbness)
    if has_numbness:
        return SentimentAnalysis(
            polarity=0.0,  # Force neutral polarity
            subjectivity=subjectivity,  # Preserve subjectivity
            label="neutral"  # Override to neutral
        )
    
    # Apply threshold-based classification with emotional burden override
    if polarity > SENTIMENT_POSITIVE_THRESHOLD:
        # Rule-based override: If emotional burden detected, cap at neutral
        # Rationale: Entries about stress/exhaustion should not be labeled positive
        # even if TextBlob's polarity is slightly positive due to words like "managed"
        if has_emotional_burden:
            label = "neutral"  # Override: burden detected, prevent false positive
        else:
            label = "positive"  # No burden, genuinely positive
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


def detect_emotional_mode(
    content: str,
    sentiment: SentimentAnalysis,
    themes: ThemeDetection,
    has_emotional_burden: bool
) -> str:
    """
    Detect emotional mode for adaptive response generation.
    
    Uses existing sentiment, themes, and burden detection to infer one of three
    emotional modes that guide reflection style and prompt selection.
    
    Modes:
    - "low_energy": Tired, unmotivated, disengaged, restless, bored
    - "anxious": Worried, stressed, overwhelmed, overthinking, ruminating
    - "calm": Grounded, reflective, motivated, or genuinely positive
    
    Detection logic (precedence order):
    1. ANXIOUS: Emotional burden OR stress/loss themes OR rumination OR pressure language
       - Anxiety often appears as cognitive loops, not just negative emotion
       - Neutral sentiment does NOT imply calm if rumination/pressure present
    2. LOW_ENERGY: Low polarity + low subjectivity + low-energy keywords
       - Only if no anxiety signals detected (rumination/pressure override this)
    3. CALM: Default fallback (positive without burden, or reflective neutral)
    
    This is a lightweight, rule-based layer on top of existing NLP outputs.
    No new ML models, no external APIs, fully deterministic.
    
    Args:
        content: Journal entry text (for keyword matching)
        sentiment: Sentiment analysis results
        themes: Theme detection results
        has_emotional_burden: Whether emotional burden was detected
    
    Returns:
        str: One of "low_energy", "anxious", or "calm"
    
    Example:
        mode = detect_emotional_mode(
            "There's so much to do and I feel behind. My mind keeps looping.",
            sentiment=SentimentAnalysis(polarity=0.0, subjectivity=0.3, label="neutral"),
            themes=ThemeDetection(themes=[], confidence="low"),
            has_emotional_burden=False
        )
        # Returns: "anxious" (pressure + rumination detected)
    """
    content_lower = content.lower()
    
    # Check for rumination keywords (cognitive looping, overthinking)
    has_rumination = any(
        keyword in content_lower  # Use substring matching for multi-word phrases
        for keyword in RUMINATION_KEYWORDS
    )
    
    # Check for explicit anxiety keywords (direct worry/stress language)
    has_explicit_anxiety = any(
        keyword in content_lower  # Use substring matching for multi-word phrases
        for keyword in EXPLICIT_ANXIETY_KEYWORDS
    )
    
    # Check for pressure keywords (feeling behind, overwhelmed by demands)
    has_pressure = any(
        keyword in content_lower  # Use substring matching for multi-word phrases
        for keyword in PRESSURE_KEYWORDS
    )
    
    # Check for low-energy keywords
    has_low_energy_keywords = any(
        _match_keyword_whole_word(keyword, content_lower)
        for keyword in LOW_ENERGY_KEYWORDS
    )
    
    # ANXIOUS mode: Highest priority for anxiety signals
    # Rationale: Rumination, explicit anxiety, and burden indicate anxiety even with neutral sentiment
    # FIX: Pressure alone does NOT trigger ANXIOUS - must be combined with rumination or explicit anxiety
    # This prevents misclassifying low-energy avoidance ("so much to do but don't feel like it") as anxiety
    if has_emotional_burden:
        return "anxious"
    
    if "stress" in themes.themes or "loss" in themes.themes:
        return "anxious"
    
    # Rumination is a strong anxiety signal (cognitive looping)
    if has_rumination:
        return "anxious"
    
    # Explicit anxiety language is a strong signal
    if has_explicit_anxiety:
        return "anxious"
    
    # Pressure ONLY triggers anxiety when combined with rumination or explicit anxiety
    # Pressure alone may indicate low-energy avoidance, not anxiety
    if has_pressure and (has_rumination or has_explicit_anxiety):
        return "anxious"
    
    # LOW_ENERGY mode: Only if no anxiety signals present
    # Rationale: Disengagement, restlessness, apathy - needs validating, low-effort response
    # LOW_ENERGY should NOT override ANXIOUS (anxiety takes precedence)
    if has_low_energy_keywords:
        # Strong signal: low-energy keywords present
        if sentiment.polarity <= 0.3 and sentiment.subjectivity < 0.5:
            return "low_energy"
        # Weaker signal: keywords present but some energy/emotion detected
        # Still treat as low_energy if polarity is neutral-ish
        if -0.1 <= sentiment.polarity <= 0.2:
            return "low_energy"
    
    # CALM mode: Default fallback
    # Rationale: Positive, reflective, or neutral without specific distress signals
    return "calm"


def analyze_entry(content: str) -> Tuple[SentimentAnalysis, ThemeDetection, str]:
    """
    Perform complete NLP analysis on a journal entry.
    
    Combines sentiment analysis, theme detection, and emotional mode detection
    into a single convenient function. All analyses run locally with no external calls.
    
    This is the main entry point for NLP processing in the application.
    
    Emotional mode detection (new):
    - Adds a lightweight, rule-based layer to infer emotional state
    - Used to select mode-adaptive reflection templates and prompts
    - Three modes: "low_energy", "anxious", "calm"
    - Non-breaking addition: existing code can ignore the mode if not needed
    
    Defensive handling:
    - Empty or whitespace-only content returns neutral sentiment, no themes, calm mode
    - All analyses handle edge cases independently
    - No exceptions thrown for valid string input
    
    Args:
        content: Journal entry text to analyze
    
    Returns:
        Tuple[SentimentAnalysis, ThemeDetection, str]: Sentiment, themes, and emotional mode
    
    Example:
        sentiment, themes, mode = analyze_entry("I'm grateful for today")
        print(f"Sentiment: {sentiment.label}")
        print(f"Themes: {themes.themes}")
        print(f"Mode: {mode}")  # "calm"
    """
    sentiment = analyze_sentiment(content)
    themes = detect_themes(content)
    
    # Detect emotional burden (used for both sentiment override and mode detection)
    has_emotional_burden = _detect_emotional_burden(content)
    
    # Detect emotional mode for adaptive response generation
    mode = detect_emotional_mode(content, sentiment, themes, has_emotional_burden)
    
    # FIX 1: Sentiment correction for LOW_ENERGY contexts
    # Rationale: Flat, numb, or low-energy entries are sometimes labeled as "positive" by TextBlob,
    # which is emotionally inaccurate. Override positive sentiment to neutral for LOW_ENERGY mode.
    if mode == "low_energy" and sentiment.polarity > 0:
        # Create corrected sentiment with neutral polarity and label
        sentiment = SentimentAnalysis(
            polarity=-0.1,  # Slightly negative to ensure "neutral" label
            subjectivity=sentiment.subjectivity,  # Preserve subjectivity
            label="neutral"  # Override to neutral
        )
    
    return sentiment, themes, mode
