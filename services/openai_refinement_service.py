from openai import OpenAI
from typing import Optional
import logging

from models.schemas import SentimentAnalysis, ThemeDetection, EmpathyReflection
from config.settings import (
    OPENAI_API_KEY,
    USE_OPENAI_REFINEMENT,
    OPENAI_MODEL,
    OPENAI_MAX_TOKENS,
    OPENAI_TEMPERATURE
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


REFINEMENT_SYSTEM_PROMPT = """You are enhancing empathetic journal reflections to improve emotional resonance and user engagement.

YOUR ROLE:
- Expand and rephrase the base reflection for warmth, clarity, and human connection
- Create responses that feel conversational, not templated
- Acknowledge complexity, mixed feelings, and effort when present
- Validate the user's experience without judging or interpreting

RESPONSE STRUCTURE:
- Write 3-6 sentences that feel warm and emotionally resonant
- Paraphrase beyond rigid templates while preserving original meaning
- End with ONE optional, open-ended reflective question

STRICT BOUNDARIES:
- Do NOT give advice, guidance, or instructions
- Do NOT use "should", "must", "need to", or similar directive language
- Do NOT diagnose or use clinical/mental health terminology
- Do NOT predict outcomes or future behavior
- Do NOT position yourself as an authority or therapist
- Do NOT mention being an AI or assistant

TONE:
- Warm, human, and emotionally present
- Validating without being prescriptive
- Gentle acknowledgment of uncertainty and nuance
- Conversational, not instructional

Your role is empathy expansion, not therapeutic intervention."""


def _is_openai_available() -> bool:
    """
    Check if OpenAI refinement is available and properly configured.
    
    Returns:
        bool: True if API key exists and refinement is enabled
    """
    return bool(OPENAI_API_KEY) and USE_OPENAI_REFINEMENT


def _build_refinement_prompt(
    base_reflection: EmpathyReflection,
    sentiment: SentimentAnalysis,
    themes: ThemeDetection
) -> str:
    """
    Build the user prompt for OpenAI refinement.
    
    Provides context about the analysis results so the LLM can maintain
    consistency with the detected sentiment and themes.
    
    Args:
        base_reflection: Original template-based reflection
        sentiment: Sentiment analysis results
        themes: Theme detection results
    
    Returns:
        str: Formatted prompt for OpenAI
    """
    themes_str = ", ".join(themes.themes) if themes.themes else "none"
    
    # Build context about emotional complexity
    sentiment_context = sentiment.label
    if -0.05 <= sentiment.polarity <= 0.05:
        sentiment_context += " (very neutral/mixed)"
    elif abs(sentiment.polarity) < 0.3:
        sentiment_context += " (mild)"
    
    prompt = f"""Enhance the following empathetic reflection for emotional resonance and user engagement.

Context (use for consistency, DO NOT change the analysis):
- Sentiment: {sentiment_context}
- Themes detected: {themes_str}
- Subjectivity: {sentiment.subjectivity:.2f} (0=objective, 1=subjective)

Base reflection:
Message: {base_reflection.message}"""
    
    if base_reflection.prompt:
        prompt += f"\nPrompt: {base_reflection.prompt}"
    
    prompt += """\n\nExpand this into a warmer, more emotionally resonant reflection (3-6 sentences).
- Acknowledge any complexity, mixed feelings, or effort present
- Make it feel conversational and human
- End with ONE optional open-ended question if appropriate

Provide ONLY the enhanced reflection in this format:
Message: [3-6 sentences, warm and emotionally resonant]
Prompt: [one open-ended question or 'none']"""
    
    return prompt


def _parse_llm_response(response_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse the LLM response to extract message and prompt.
    
    Robust parsing that handles various response formats:
    - Strict format: "Message: ..." and "Prompt: ..."
    - Relaxed format: If labels missing, treat entire response as message
    - Multi-line: Handles messages spanning multiple lines
    
    This forgiving approach ensures good responses aren't discarded due to
    formatting variations while maintaining the ability to extract prompts.
    
    Args:
        response_text: Raw text from OpenAI
    
    Returns:
        tuple: (message, prompt) where prompt may be None
    """
    response_text = response_text.strip()
    
    # Try to parse with labels first
    lines = response_text.split("\n")
    message_parts = []
    prompt = None
    in_message = False
    in_prompt = False
    
    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        # Check for "Message:" label
        if line_lower.startswith("message:"):
            in_message = True
            in_prompt = False
            # Extract content after "Message:"
            content = line_stripped.split(":", 1)[1].strip()
            if content:
                message_parts.append(content)
        # Check for "Prompt:" label
        elif line_lower.startswith("prompt:"):
            in_message = False
            in_prompt = True
            # Extract content after "Prompt:"
            prompt_text = line_stripped.split(":", 1)[1].strip()
            if prompt_text and prompt_text.lower() not in ["none", "null", "n/a"]:
                prompt = prompt_text
        # Continue accumulating message or prompt content
        elif in_message and line_stripped:
            message_parts.append(line_stripped)
        elif in_prompt and line_stripped and prompt is None:
            if line_stripped.lower() not in ["none", "null", "n/a"]:
                prompt = line_stripped
    
    # If we found labeled content, use it
    if message_parts:
        message = " ".join(message_parts)
    else:
        # Fallback: If no "Message:" label found, treat entire response as message
        # This handles cases where the model responds naturally without labels
        # Split on common question markers to extract potential prompt
        question_markers = ["?", "What ", "How ", "Why ", "When ", "Where "]
        
        # Look for a question at the end
        sentences = response_text.split(". ")
        if sentences:
            last_sentence = sentences[-1].strip()
            # If last sentence is a question, treat it as prompt
            if "?" in last_sentence:
                prompt = last_sentence
                # Everything else is the message
                message = ". ".join(sentences[:-1])
                if message:
                    message = message.strip() + "."
            else:
                # No question found, entire response is message
                message = response_text
    
    return message, prompt


def _validate_refined_reflection(
    refined_message: Optional[str],
    refined_prompt: Optional[str],
    base_reflection: EmpathyReflection
) -> bool:
    """
    Validate that the refined reflection meets safety criteria.
    
    Responsible AI boundaries enforcement:
    - No advice keywords ("should", "must", "need to", etc.)
    - No clinical terminology
    - Length constraints respected
    - Non-empty message
    
    Args:
        refined_message: Refined message from LLM
        refined_prompt: Refined prompt from LLM
        base_reflection: Original template-based reflection
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not refined_message or len(refined_message.strip()) == 0:
        logger.warning("Refined message is empty")
        return False
    
    # Relaxed length constraints to allow meaningful shorter responses
    # Rationale: Some reflections are naturally concise but still warm and validating
    # We prioritize quality over strict length requirements
    refined_length = len(refined_message)
    
    # Minimum: at least 50 characters (roughly 1 meaningful sentence)
    # Relaxed from 100 to allow concise but emotionally resonant responses
    if refined_length < 50:
        logger.warning(f"Refined message too short: {refined_length} chars")
        return False
    
    # Maximum: no more than 1000 characters (roughly 10 sentences, generous buffer)
    # Increased from 800 to accommodate naturally flowing reflections
    if refined_length > 1000:
        logger.warning(f"Refined message too long: {refined_length} chars")
        return False
    
    # Check for prohibited advice keywords using whole-word matching
    # Rationale: Avoid false positives (e.g., "shoulder" contains "should")
    # We use word boundaries to ensure we're catching actual directive language
    advice_keywords = [
        "should", "must", "need to", "have to", "ought to",
        "recommend", "suggest", "advise", "try to"
    ]
    
    combined_text = (refined_message + " " + (refined_prompt or "")).lower()
    
    # Use whole-word matching to avoid false positives
    import re
    for keyword in advice_keywords:
        # Create pattern with word boundaries
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, combined_text):
            logger.warning(f"Refined reflection contains advice keyword: {keyword}")
            return False
    
    # Check for clinical terminology using whole-word matching
    # Rationale: Prevent medicalization while avoiding false positives
    # (e.g., "conditional" shouldn't trigger "condition")
    clinical_terms = [
        "diagnosis", "disorder", "syndrome", "therapy", "treatment",
        "symptoms", "psychiatric", "clinical", "pathology", "patient"
    ]
    
    # Use whole-word matching for clinical terms
    for term in clinical_terms:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, combined_text):
            logger.warning(f"Refined reflection contains clinical term: {term}")
            return False
    
    # All validation checks passed
    return True


def refine_reflection_with_llm(
    base_reflection: EmpathyReflection,
    sentiment: SentimentAnalysis,
    themes: ThemeDetection
) -> EmpathyReflection:
    """
    Optionally refine reflection wording using OpenAI GPT-4o-mini.
    
    This function improves the natural language quality and empathy of
    template-based reflections WITHOUT changing their meaning, logic, or
    therapeutic boundaries.
    
    CRITICAL CONSTRAINTS:
    - Local NLP (sentiment + themes) remains the authoritative source of truth
    - LLM expands and refines wording for emotional resonance
    - No advice, diagnosis, instructions, or prescriptive content
    - No clinical or mental health terminology
    - Responses are 3-6 sentences for warmth and engagement
    - Falls back silently to base_reflection on any failure
    
    Defensive behavior:
    - Returns base_reflection if OpenAI is disabled or not configured
    - Returns base_reflection if API call fails (network, rate limit, etc.)
    - Returns base_reflection if refined content fails validation
    - Never raises exceptions to API layer
    - Logs warnings for debugging but continues gracefully
    
    Args:
        base_reflection: Original template-based reflection (authoritative)
        sentiment: Sentiment analysis from local NLP
        themes: Theme detection from local NLP
    
    Returns:
        EmpathyReflection: Refined reflection if successful, base_reflection otherwise
    
    Example:
        base = EmpathyReflection(
            message="It sounds like you're experiencing some positive moments.",
            prompt="What else comes to mind?"
        )
        refined = refine_reflection_with_llm(base, sentiment, themes)
        # Returns refined version or base if refinement fails/disabled
    """
    # Check if OpenAI refinement is enabled and configured
    if not _is_openai_available():
        logger.info("OpenAI refinement disabled or not configured, using base reflection")
        return base_reflection
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Build refinement prompt with context
        user_prompt = _build_refinement_prompt(base_reflection, sentiment, themes)
        
        # Call OpenAI API
        logger.info(f"Calling OpenAI API with model: {OPENAI_MODEL}")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=OPENAI_TEMPERATURE,
            timeout=10.0  # 10 second timeout for responsiveness
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        if not response_text:
            logger.warning("OpenAI returned empty response")
            return base_reflection
        
        # Parse the response
        refined_message, refined_prompt = _parse_llm_response(response_text)
        
        # Validate the refined reflection
        if not _validate_refined_reflection(refined_message, refined_prompt, base_reflection):
            logger.warning("Refined reflection failed validation, using base reflection")
            return base_reflection
        
        # Create refined reflection
        refined_reflection = EmpathyReflection(
            message=refined_message,
            prompt=refined_prompt
        )
        
        logger.info("Successfully refined reflection with OpenAI")
        return refined_reflection
        
    except Exception as e:
        # Catch ALL exceptions and fall back gracefully
        # This includes: network errors, API errors, timeout, parsing errors, etc.
        logger.warning(f"OpenAI refinement failed: {type(e).__name__}: {str(e)}")
        logger.info("Falling back to base reflection")
        return base_reflection
