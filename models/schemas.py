from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


MAX_ENTRY_LENGTH = 5000
MIN_ENTRY_LENGTH = 1
MAX_REFLECTION_LENGTH = 1000
MAX_PROMPT_LENGTH = 200


class JournalEntryInput(BaseModel):
    """
    Input model for journal entries submitted by users.
    
    Privacy-first design: No user identifiers, no tracking metadata.
    Only captures the essential journal content and optional timestamp.
    
    Attributes:
        content: The journal entry text (1-5000 characters)
        timestamp: Optional datetime, defaults to current UTC time
    
    Example:
        entry = JournalEntryInput(content="Today was a good day.")
    """
    content: str = Field(
        ...,
        min_length=MIN_ENTRY_LENGTH,
        max_length=MAX_ENTRY_LENGTH,
        description="Journal entry content between 1 and 5000 characters"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Entry timestamp, defaults to current UTC time"
    )
    
    @field_validator('content')
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """
        Ensure content is not empty or whitespace-only after stripping.
        
        This prevents users from submitting entries that appear valid
        but contain no actual content.
        
        Args:
            v: The content string to validate
            
        Returns:
            Stripped content string
            
        Raises:
            ValueError: If content is empty after stripping whitespace
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("Entry cannot be empty or whitespace only")
        return stripped


class SentimentAnalysis(BaseModel):
    """
    Sentiment analysis results from local NLP processing.
    
    Uses bounded, explainable metrics that are transparent to users.
    No black-box scoring or opaque "mental health" metrics.
    
    Attributes:
        polarity: Sentiment score from -1.0 (negative) to 1.0 (positive)
        subjectivity: Objectivity score from 0.0 (objective) to 1.0 (subjective)
        label: Human-readable sentiment category
    
    Example:
        sentiment = SentimentAnalysis(
            polarity=0.5,
            subjectivity=0.6,
            label="positive"
        )
    """
    polarity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment polarity: -1.0 (very negative) to 1.0 (very positive)"
    )
    subjectivity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Subjectivity: 0.0 (objective) to 1.0 (subjective)"
    )
    label: Literal["positive", "neutral", "negative"] = Field(
        ...,
        description="Human-readable sentiment category"
    )


class ThemeDetection(BaseModel):
    """
    Detected themes from journal entry using rule-based keyword matching.
    
    Transparent, explainable theme detection. Users can understand
    why a theme was detected based on keywords present.
    
    Attributes:
        themes: List of detected theme names (lowercase)
        confidence: Confidence level in theme detection
    
    Example:
        themes = ThemeDetection(
            themes=["gratitude", "relationships"],
            confidence="medium"
        )
    """
    themes: List[str] = Field(
        default_factory=list,
        description="List of detected themes (can be empty)"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        ...,
        description="Confidence level in theme detection"
    )
    
    @field_validator('themes')
    @classmethod
    def normalize_themes(cls, v: List[str]) -> List[str]:
        """
        Normalize themes to lowercase and remove duplicates.
        
        Ensures consistent theme representation and prevents
        duplicate themes from appearing in results.
        
        Args:
            v: List of theme strings
            
        Returns:
            Normalized list of unique, lowercase themes
        """
        return list(dict.fromkeys(theme.lower() for theme in v))


class EmpathyReflection(BaseModel):
    """
    Empathetic reflection generated in response to journal entry.
    
    Responsible AI design: Non-judgmental, validating, non-prescriptive.
    No advice-giving, no diagnosis, no predictions about future behavior.
    
    Attributes:
        message: Main empathetic reflection text
        prompt: Optional gentle follow-up prompt for further reflection
    
    Example:
        reflection = EmpathyReflection(
            message="It sounds like you're experiencing some positive moments.",
            prompt="What else comes to mind?"
        )
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_REFLECTION_LENGTH,
        description="Empathetic reflection message (1-1000 characters)"
    )
    prompt: Optional[str] = Field(
        None,
        max_length=MAX_PROMPT_LENGTH,
        description="Optional follow-up prompt (max 200 characters)"
    )
    
    @field_validator('message', 'prompt')
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        """
        Strip leading and trailing whitespace from reflection text.
        
        Ensures clean, professional presentation of reflections.
        
        Args:
            v: String to strip (or None for optional fields)
            
        Returns:
            Stripped string or None
        """
        return v.strip() if v else v


class JournalEntryResponse(BaseModel):
    """
    Complete response returned to user after processing journal entry.
    
    Combines the original entry with all analysis results in a
    structured, transparent format.
    
    Privacy note: entry_id is a UUID with no connection to user identity.
    
    Attributes:
        entry_id: Unique identifier for this entry (UUID string)
        timestamp: When the entry was created
        content: Original journal entry content
        sentiment: Sentiment analysis results
        themes: Detected themes
        reflection: Empathetic reflection and optional prompt
    
    Example:
        response = JournalEntryResponse(
            entry_id="123e4567-e89b-12d3-a456-426614174000",
            timestamp=datetime.utcnow(),
            content="Today was a good day.",
            sentiment=SentimentAnalysis(...),
            themes=ThemeDetection(...),
            reflection=EmpathyReflection(...)
        )
    """
    entry_id: str = Field(
        ...,
        description="Unique identifier for this entry (UUID)"
    )
    timestamp: datetime = Field(
        ...,
        description="Entry creation timestamp"
    )
    content: str = Field(
        ...,
        description="Original journal entry content"
    )
    sentiment: SentimentAnalysis = Field(
        ...,
        description="Sentiment analysis results"
    )
    themes: ThemeDetection = Field(
        ...,
        description="Detected themes from entry"
    )
    reflection: EmpathyReflection = Field(
        ...,
        description="Empathetic reflection and optional prompt"
    )
