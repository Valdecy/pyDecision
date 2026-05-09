"""pyDecision.web — browser-based GUI for the pyDecision library."""

from .server import start as web_app, stop as web_stop, is_running

__all__ = ["web_app", "web_stop", "is_running"]
