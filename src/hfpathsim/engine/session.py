"""Session management for multi-user simulation engine access.

Provides session isolation for REST API and web dashboard deployments.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import threading
import uuid
import weakref

from .simulation_engine import SimulationEngine, EngineConfig
from ..core.parameters import VoglerParameters, ITUCondition


@dataclass
class Session:
    """Individual user session with isolated simulation engine."""

    session_id: str
    engine: SimulationEngine
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = datetime.utcnow()

    @property
    def age_seconds(self) -> float:
        """Return session age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Return idle time in seconds."""
        return (datetime.utcnow() - self.last_accessed).total_seconds()

    def to_dict(self) -> dict:
        """Convert session info to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "age_seconds": self.age_seconds,
            "idle_seconds": self.idle_seconds,
            "engine_running": self.engine._state.running,
            "metadata": self.metadata,
        }


class SessionManager:
    """Manages multiple isolated simulation sessions.

    Features:
    - Session creation with unique IDs
    - Session timeout and cleanup
    - Concurrent session limits
    - Thread-safe operations
    """

    def __init__(
        self,
        max_sessions: int = 100,
        session_timeout_minutes: int = 60,
        cleanup_interval_seconds: float = 60.0,
    ):
        """Initialize session manager.

        Args:
            max_sessions: Maximum concurrent sessions
            session_timeout_minutes: Session timeout after idle
            cleanup_interval_seconds: Cleanup check interval
        """
        self._max_sessions = max_sessions
        self._session_timeout = timedelta(minutes=session_timeout_minutes)
        self._cleanup_interval = cleanup_interval_seconds

        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

        # Start cleanup thread
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
        )
        self._cleanup_thread.start()

    def create_session(
        self,
        config: Optional[EngineConfig] = None,
        vogler_params: Optional[VoglerParameters] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new session with isolated engine.

        Args:
            config: Engine configuration
            vogler_params: Vogler parameters
            metadata: Optional metadata

        Returns:
            New Session object

        Raises:
            RuntimeError: If max sessions reached
        """
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                # Try cleanup first
                self._cleanup_expired()
                if len(self._sessions) >= self._max_sessions:
                    raise RuntimeError(
                        f"Maximum sessions ({self._max_sessions}) reached"
                    )

            session_id = str(uuid.uuid4())
            engine = SimulationEngine(
                config=config,
                vogler_params=vogler_params,
            )
            session = Session(
                session_id=session_id,
                engine=engine,
                metadata=metadata or {},
            )
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session or None if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                # Stop streaming if running
                if session.engine._state.running:
                    session.engine.stop_streaming()
                return True
            return False

    def list_sessions(self) -> list[dict]:
        """List all active sessions.

        Returns:
            List of session info dictionaries
        """
        with self._lock:
            return [s.to_dict() for s in self._sessions.values()]

    def get_session_count(self) -> int:
        """Return number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def _cleanup_expired(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions removed
        """
        now = datetime.utcnow()
        expired = []

        for session_id, session in self._sessions.items():
            if now - session.last_accessed > self._session_timeout:
                expired.append(session_id)

        for session_id in expired:
            session = self._sessions.pop(session_id, None)
            if session and session.engine._state.running:
                session.engine.stop_streaming()

        return len(expired)

    def _cleanup_loop(self):
        """Background cleanup thread."""
        while not self._stop_cleanup.wait(self._cleanup_interval):
            with self._lock:
                self._cleanup_expired()

    def shutdown(self):
        """Shutdown session manager and all sessions."""
        self._stop_cleanup.set()
        self._cleanup_thread.join(timeout=2.0)

        with self._lock:
            for session in self._sessions.values():
                if session.engine._state.running:
                    session.engine.stop_streaming()
            self._sessions.clear()


# Global session manager (singleton)
_global_session_manager: Optional[SessionManager] = None
_session_manager_lock = threading.Lock()


def get_session_manager(
    max_sessions: int = 100,
    session_timeout_minutes: int = 60,
) -> SessionManager:
    """Get or create the global session manager.

    Args:
        max_sessions: Maximum concurrent sessions
        session_timeout_minutes: Session timeout

    Returns:
        Global SessionManager instance
    """
    global _global_session_manager

    with _session_manager_lock:
        if _global_session_manager is None:
            _global_session_manager = SessionManager(
                max_sessions=max_sessions,
                session_timeout_minutes=session_timeout_minutes,
            )
        return _global_session_manager


def shutdown_session_manager():
    """Shutdown the global session manager."""
    global _global_session_manager

    with _session_manager_lock:
        if _global_session_manager:
            _global_session_manager.shutdown()
            _global_session_manager = None
