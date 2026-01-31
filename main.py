from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from config.settings import APP_NAME, APP_VERSION, APP_DESCRIPTION, CORS_ORIGINS
from models.schemas import JournalEntryInput, JournalEntryResponse
from storage.memory_store import get_store
from services.nlp_service import analyze_entry
from services.reflection_service import generate_reflection


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns basic application status and confirms the service is running.
    No authentication required, no sensitive data exposed.
    
    Returns:
        dict: Status message and application info
    
    Example:
        GET /health
        Response: {"status": "healthy", "app": "AI Journaling Companion", "version": "1.0.0"}
    """
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "privacy": "All processing is local, no data leaves this system"
    }


@app.post("/entries", response_model=JournalEntryResponse, status_code=status.HTTP_201_CREATED)
async def create_entry(entry_input: JournalEntryInput):
    """
    Create a new journal entry with AI analysis.
    
    Processing pipeline:
    1. Validate input (handled by Pydantic)
    2. Perform local sentiment analysis
    3. Detect themes using rule-based keywords
    4. Generate empathetic reflection
    5. Store entry in memory
    6. Return complete response
    
    Privacy guarantees:
    - All processing happens locally (no external API calls)
    - Entry stored in-memory only (cleared on restart)
    - No user tracking or authentication
    
    Responsible AI boundaries:
    - Sentiment analysis is explainable (TextBlob word scores)
    - Theme detection is transparent (keyword matching)
    - Reflections are non-judgmental and non-prescriptive
    - No diagnosis, advice, or predictions
    
    Args:
        entry_input: JournalEntryInput with content and optional timestamp
    
    Returns:
        JournalEntryResponse: Complete entry with analysis and reflection
    
    Raises:
        HTTPException 422: Invalid input (handled by FastAPI/Pydantic)
        HTTPException 500: Internal processing error
    
    Example:
        POST /entries
        Body: {"content": "Today was a good day. I'm grateful for my family."}
        Response: {
            "entry_id": "123e4567-e89b-12d3-a456-426614174000",
            "timestamp": "2026-01-30T22:42:00Z",
            "content": "Today was a good day. I'm grateful for my family.",
            "sentiment": {
                "polarity": 0.7,
                "subjectivity": 0.6,
                "label": "positive"
            },
            "themes": {
                "themes": ["gratitude", "relationships"],
                "confidence": "medium"
            },
            "reflection": {
                "message": "It sounds like you're experiencing some positive moments. I notice you're noticing moments of appreciation.",
                "prompt": null
            }
        }
    """
    try:
        # Step 1: Perform NLP analysis (sentiment + themes)
        sentiment, themes = analyze_entry(entry_input.content)
        
        # Step 2: Generate empathetic reflection
        reflection = generate_reflection(sentiment, themes)
        
        # Step 3: Create response object
        response = JournalEntryResponse(
            entry_id="",  # Will be set by storage
            timestamp=entry_input.timestamp,
            content=entry_input.content,
            sentiment=sentiment,
            themes=themes,
            reflection=reflection
        )
        
        # Step 4: Store in memory and get UUID
        store = get_store()
        entry_id = store.store_entry(response.dict())
        
        # Step 5: Update response with generated entry_id
        response.entry_id = entry_id
        
        return response
        
    except ValidationError as e:
        # Pydantic validation errors (should be caught by FastAPI, but defensive)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/entries/{entry_id}", response_model=JournalEntryResponse)
async def get_entry(entry_id: str):
    """
    Retrieve a journal entry by its UUID.
    
    Returns the complete entry including original content, sentiment analysis,
    detected themes, and empathetic reflection.
    
    Privacy note: Only the exact UUID can retrieve an entry. No listing or
    browsing of entries is supported to prevent accidental exposure.
    
    Args:
        entry_id: UUID string of the entry to retrieve
    
    Returns:
        JournalEntryResponse: Complete entry data
    
    Raises:
        HTTPException 404: Entry not found
        HTTPException 500: Internal retrieval error
    
    Example:
        GET /entries/123e4567-e89b-12d3-a456-426614174000
        Response: {complete entry data}
    """
    try:
        store = get_store()
        entry_data = store.get_entry(entry_id)
        
        if entry_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entry with ID '{entry_id}' not found"
            )
        
        # Convert stored dict back to Pydantic model for validation
        return JournalEntryResponse(**entry_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """
    Custom handler for Pydantic validation errors.
    
    Provides clear, user-friendly error messages when input validation fails.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Input validation failed",
            "errors": exc.errors()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting {APP_NAME} v{APP_VERSION}")
    print("Privacy-first journaling with local AI analysis")
    print("No external API calls • No data persistence • No user tracking")
    print("\nStarting server on http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
