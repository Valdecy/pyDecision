"""pyDecision — A MCDA Library Incorporating a Large Language Model
to Enhance Decision Analysis.

Top-level convenience exports:

    >>> import pyDecision
    >>> pyDecision.web_app()        # launches the browser GUI
    >>> pyDecision.web_stop()       # stops it

The full algorithm catalogue lives under `pyDecision.algorithm`.
"""

__version__ = "5.1.1"

try:
    from .web import web_app, web_stop, is_running as web_is_running
except Exception:  # pragma: no cover
    def _missing_web(*args, **kwargs):
        raise ImportError(
            "The web interface requires optional dependencies. Install Flask to use pyDecision.web_app()."
        )
    web_app = _missing_web
    web_stop = _missing_web
    def web_is_running():
        return False

__all__ = ["web_app", "web_stop", "web_is_running", "__version__"]
