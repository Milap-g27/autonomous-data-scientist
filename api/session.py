"""
In-memory session store for the Autonomous Data Scientist API.
Holds uploaded DataFrames, configs, trained models, and chat histories.
Replace with Redis / DB for production deployments.
"""
import uuid
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime


class Session:
    """Single user session."""

    __slots__ = (
        "id", "created_at", "df", "config", "result",
        "chat_history", "task_status", "task_error",
    )

    def __init__(self, session_id: str):
        self.id: str = session_id
        self.created_at: datetime = datetime.utcnow()
        self.df: Any = None               # pandas DataFrame
        self.config: Dict[str, Any] = {}   # pipeline config
        self.result: Dict[str, Any] = {}   # full analysis result
        self.chat_history: List[Dict[str, str]] = []
        self.task_status: str = "idle"     # idle | running | completed | failed
        self.task_error: Optional[str] = None


class SessionStore:
    """Thread-safe in-memory session store backed by an asyncio lock."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id)
        async with self._lock:
            self._sessions[session_id] = session
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def list_ids(self) -> List[str]:
        async with self._lock:
            return list(self._sessions.keys())


# Global singleton — shared across the app via dependency injection
store = SessionStore()
