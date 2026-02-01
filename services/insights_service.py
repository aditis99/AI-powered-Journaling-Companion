"""
Insights Service for AI Journaling Companion

Provides success-metric features:
1. Engagement framing based on entry count
2. Pattern aggregation over recent entries
3. Mode-aware reflection summaries

All features are:
- Deterministic and rule-based
- Privacy-preserving (no external APIs)
- Non-breaking (optional fields only)
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter


def generate_engagement_note(entry_count: int) -> Optional[str]:
    """
    Generate a gentle engagement note based on entry count.
    
    Provides affirming acknowledgment of consistency without gamification.
    No streaks, no metrics shown to user - just gentle encouragement.
    
    Triggers:
    - 2 entries: Gentle encouragement
    - 3-4 entries: Acknowledge ongoing reflection
    - 5+ entries: No note (avoid repetition)
    
    Args:
        entry_count: Total number of entries stored
    
    Returns:
        Optional[str]: Engagement note, or None if no note should be shown
    
    Example:
        note = generate_engagement_note(3)
        # Returns: "You've been showing up and reflecting consistently. That matters."
    """
    if entry_count == 2:
        return "You're showing up. That's what matters."
    elif entry_count in [3, 4]:
        return "You've been showing up and reflecting consistently. That matters."
    else:
        # No note for 1 entry (too early) or 5+ (avoid repetition)
        return None


def aggregate_patterns(recent_entries: List[dict]) -> Dict[str, any]:
    """
    Aggregate patterns over recent entries (last 3-4).
    
    Counts emotional modes and recurring themes to identify patterns.
    No ML, no embeddings - just simple counting and frequency analysis.
    
    Aggregation logic:
    - Count emotional modes across entries
    - Count themes across entries
    - Identify dominant mode (most frequent)
    - Identify most frequent theme
    
    Args:
        recent_entries: List of entry dictionaries (from memory store)
    
    Returns:
        Dict with:
        - mode_counts: Dict[str, int] - Count of each emotional mode
        - theme_counts: Dict[str, int] - Count of each theme
        - dominant_mode: str - Most frequent emotional mode
        - top_theme: Optional[str] - Most frequent theme (if any)
        - entry_count: int - Number of entries analyzed
    
    Example:
        patterns = aggregate_patterns(recent_entries)
        # Returns: {
        #     "mode_counts": {"low_energy": 2, "anxious": 1},
        #     "theme_counts": {"health": 2, "work": 1},
        #     "dominant_mode": "low_energy",
        #     "top_theme": "health",
        #     "entry_count": 3
        # }
    """
    if not recent_entries:
        return {
            "mode_counts": {},
            "theme_counts": {},
            "dominant_mode": None,
            "top_theme": None,
            "entry_count": 0
        }
    
    # Extract modes and themes from entries
    modes = []
    themes = []
    
    for entry in recent_entries:
        # Get emotional mode (now stored in entry dict)
        if "mode" in entry:
            modes.append(entry["mode"])
        
        # Extract themes from entry
        if "themes" in entry and "themes" in entry["themes"]:
            themes.extend(entry["themes"]["themes"])
    
    # Count occurrences
    mode_counter = Counter(modes)
    theme_counter = Counter(themes)
    
    # Identify dominant patterns
    dominant_mode = mode_counter.most_common(1)[0][0] if mode_counter else None
    top_theme = theme_counter.most_common(1)[0][0] if theme_counter else None
    
    return {
        "mode_counts": dict(mode_counter),
        "theme_counts": dict(theme_counter),
        "dominant_mode": dominant_mode,
        "top_theme": top_theme,
        "entry_count": len(recent_entries)
    }


def generate_reflection_summary(
    patterns: Dict[str, any],
    dominant_mode: str
) -> Optional[str]:
    """
    Generate a mode-aware reflection summary based on aggregated patterns.
    
    Triggers only when entry count >= 3.
    Tone adapts to dominant emotional mode:
    - low_energy: Normalizing, non-activating
    - anxious: Grounding, non-problem-solving
    - calm: Reflective, reinforcing
    
    No advice, no prescriptions, no analysis jargon.
    
    Args:
        patterns: Aggregated pattern data from aggregate_patterns()
        dominant_mode: Most frequent emotional mode
    
    Returns:
        Optional[str]: Reflection summary, or None if insufficient data
    
    Example:
        summary = generate_reflection_summary(patterns, "low_energy")
        # Returns: "Looking back at your recent entries, a low-energy tone shows up often.
        #           You've also mentioned health a few times during these moments.
        #           This isn't something to change — just something worth noticing."
    """
    entry_count = patterns.get("entry_count", 0)
    
    # Only generate summary if we have 3+ entries
    if entry_count < 3:
        return None
    
    top_theme = patterns.get("top_theme")
    mode_counts = patterns.get("mode_counts", {})
    
    # Build summary based on dominant mode
    if dominant_mode == "low_energy":
        # Normalizing, non-activating tone
        base = "Looking back at your recent entries, a low-energy tone shows up often."
        
        if top_theme:
            theme_context = f" You've also mentioned {top_theme} a few times during these moments."
        else:
            theme_context = ""
        
        closing = " This isn't something to change — just something worth noticing."
        
        return base + theme_context + closing
    
    elif dominant_mode == "anxious":
        # Grounding, non-problem-solving tone
        base = "Your recent entries carry a sense of weight and mental activity."
        
        if top_theme:
            theme_context = f" {top_theme.capitalize()} comes up more than once."
        else:
            theme_context = ""
        
        closing = " You're holding a lot, and noticing that is enough for now."
        
        return base + theme_context + closing
    
    elif dominant_mode == "calm":
        # Reflective, reinforcing tone
        base = "There's a reflective quality to your recent entries."
        
        if top_theme:
            theme_context = f" {top_theme.capitalize()} seems to be on your mind."
        else:
            theme_context = ""
        
        closing = " You're taking time to notice and reflect, and that's valuable."
        
        return base + theme_context + closing
    
    else:
        # Fallback for unknown mode
        return None


def generate_insights(
    entry_count: int,
    recent_entries: List[dict],
    current_mode: str
) -> Dict[str, Optional[str]]:
    """
    Generate all insights for a journal entry.
    
    Combines engagement framing, pattern aggregation, and reflection summaries
    into a single convenient function.
    
    Args:
        entry_count: Total number of entries stored
        recent_entries: List of recent entry dictionaries (last 3-4)
        current_mode: Emotional mode of current entry
    
    Returns:
        Dict with:
        - engagement_note: Optional[str] - Engagement framing message
        - reflection_summary: Optional[str] - Pattern-based summary
    
    Example:
        insights = generate_insights(
            entry_count=3,
            recent_entries=recent_entries,
            current_mode="low_energy"
        )
        # Returns: {
        #     "engagement_note": "You've been showing up...",
        #     "reflection_summary": "Looking back at your recent entries..."
        # }
    """
    # Generate engagement note
    engagement_note = generate_engagement_note(entry_count)
    
    # Generate reflection summary (only if 3+ entries)
    reflection_summary = None
    if entry_count >= 3:
        patterns = aggregate_patterns(recent_entries)
        # Use current_mode as dominant mode for now (will improve with stored modes)
        reflection_summary = generate_reflection_summary(patterns, current_mode)
    
    return {
        "engagement_note": engagement_note,
        "reflection_summary": reflection_summary
    }
