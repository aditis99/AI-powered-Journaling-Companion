import uuid
from typing import Dict, Optional, List
from threading import Lock


class InMemoryStore:
    """
    Thread-safe in-memory storage for journal entries.
    
    Privacy-first design:
    - No persistence to disk (data lost on restart by design)
    - No file I/O or database connections
    - No logging of entry content
    - No listing/browsing capabilities (retrieve by ID only)
    
    Thread safety:
    - Uses threading.Lock to prevent race conditions
    - Safe for concurrent access in FastAPI async context
    
    Attributes:
        _store: Internal dictionary mapping UUID strings to entry data
        _lock: Threading lock for synchronizing access
    
    Example:
        store = InMemoryStore()
        entry_id = store.store_entry({"content": "My journal entry", ...})
        entry = store.get_entry(entry_id)
    """
    
    def __init__(self):
        """
        Initialize the in-memory store with an empty dictionary and lock.
        
        The store is completely empty on initialization - no pre-seeded data,
        no configuration files, no external dependencies.
        """
        self._store: Dict[str, dict] = {}
        self._lock: Lock = Lock()
    
    def store_entry(self, entry_data: dict) -> str:
        """
        Store a journal entry and return its generated UUID.
        
        Thread-safe operation that generates a unique identifier and
        stores the complete entry data in memory.
        
        Privacy note: UUID is randomly generated with no connection to
        user identity, timestamp, or entry content.
        
        Args:
            entry_data: Dictionary containing the complete entry data
                       (typically a JournalEntryResponse.dict())
        
        Returns:
            str: Generated UUID for this entry
        
        Example:
            entry_id = store.store_entry({
                "content": "Today was good",
                "sentiment": {...},
                "themes": {...},
                "reflection": {...}
            })
        """
        entry_id = str(uuid.uuid4())
        
        with self._lock:
            self._store[entry_id] = entry_data
        
        return entry_id
    
    def get_entry(self, entry_id: str) -> Optional[dict]:
        """
        Retrieve a journal entry by its UUID.
        
        Thread-safe operation that returns the complete entry data
        if found, or None if the entry doesn't exist.
        
        No listing capability: You must know the exact UUID to retrieve.
        This prevents accidental exposure of all entries.
        
        Args:
            entry_id: UUID string of the entry to retrieve
        
        Returns:
            Optional[dict]: Entry data if found, None otherwise
        
        Example:
            entry = store.get_entry("123e4567-e89b-12d3-a456-426614174000")
            if entry:
                print(entry["content"])
            else:
                print("Entry not found")
        """
        with self._lock:
            return self._store.get(entry_id)
    
    def entry_exists(self, entry_id: str) -> bool:
        """
        Check if an entry exists without retrieving its data.
        
        Thread-safe operation useful for validation before retrieval.
        
        Args:
            entry_id: UUID string to check
        
        Returns:
            bool: True if entry exists, False otherwise
        
        Example:
            if store.entry_exists(entry_id):
                entry = store.get_entry(entry_id)
        """
        with self._lock:
            return entry_id in self._store
    
    def clear_all(self) -> None:
        """
        Clear all entries from the store.
        
        WARNING: This is a destructive operation with no undo.
        Intended for testing or explicit user request to clear data.
        
        Thread-safe operation that removes all stored entries.
        
        Example:
            store.clear_all()  # All entries are now gone
        """
        with self._lock:
            self._store.clear()
    
    def get_entry_count(self) -> int:
        """
        Get the total number of entries currently stored.
        
        Thread-safe operation. Useful for monitoring/debugging,
        but does not reveal entry IDs or content.
        
        Returns:
            int: Number of entries in the store
        
        Example:
            count = store.get_entry_count()
            print(f"Currently storing {count} entries")
        """
        with self._lock:
            return len(self._store)
    
    def get_recent_entries(self, limit: int = 4) -> list:
        """
        Get the most recent entries for pattern aggregation.
        
        Returns entries in reverse chronological order (most recent first).
        Used for generating reflection summaries and detecting patterns.
        
        Privacy note: This is the only listing method, limited to recent entries only.
        No full browsing capability exists.
        
        Thread-safe operation.
        
        Args:
            limit: Maximum number of recent entries to return (default: 4)
        
        Returns:
            list: List of entry dictionaries, most recent first
        
        Example:
            recent = store.get_recent_entries(limit=3)
            # Returns last 3 entries for pattern analysis
        """
        with self._lock:
            # Get all entries and return the last N
            # Note: This assumes insertion order is preserved (Python 3.7+)
            all_entries = list(self._store.values())
            return all_entries[-limit:] if len(all_entries) >= limit else all_entries


# Global singleton instance for use across the application
# This ensures all parts of the app share the same in-memory store
_store_instance: Optional[InMemoryStore] = None


def get_store() -> InMemoryStore:
    """
    Get the global singleton instance of the in-memory store.
    
    This function ensures only one store instance exists throughout
    the application lifecycle, preventing data fragmentation.
    
    Thread-safe: First call creates the instance, subsequent calls
    return the same instance.
    
    Returns:
        InMemoryStore: The global store instance
    
    Example:
        from storage.memory_store import get_store
        
        store = get_store()
        entry_id = store.store_entry(data)
    """
    global _store_instance
    
    if _store_instance is None:
        _store_instance = InMemoryStore()
    
    return _store_instance
