"""
pyDecision.web.server
=====================

Flask app + lifecycle manager for the pyDecision Studio web UI.
"""
import logging
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.serving import make_server

from . import methods as methods_mod


_HERE = Path(__file__).resolve().parent
_TEMPLATES = _HERE / "templates"
_STATIC    = _HERE / "static"


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
def _create_app():
    app = Flask(
        __name__,
        template_folder=str(_TEMPLATES),
        static_folder=str(_STATIC),
        static_url_path="/static",
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/static/<path:filename>")
    def static_files(filename):
        return send_from_directory(str(_STATIC), filename)

    @app.route("/api/methods", methods=["GET"])
    def api_methods():
        return jsonify(methods_mod.list_methods())

    @app.route("/api/method/<key>", methods=["GET"])
    def api_method(key):
        m = methods_mod.get_method(key)
        if m is None:
            return jsonify({"ok": False, "error": f"Unknown method '{key}'"}), 404
        # Return spec at top-level for easy client consumption
        return jsonify(m)


    @app.route("/api/example/<key>", methods=["GET"])
    def api_example(key):
        ex = methods_mod.get_example(key)
        if ex is None:
            return jsonify({"ok": False, "error": f"No example available for '{key}'"}), 404
        return jsonify({"ok": True, "example": ex})

    @app.route("/api/run", methods=["POST"])
    def api_run():
        try:
            payload = request.get_json(force=True)
            key = payload["method"]
            dataset = payload.get("dataset")
            weights = payload.get("weights")
            criterion_type = payload.get("criterion_type")
            params = payload.get("params", {})
            extra_inputs = payload.get("extra_inputs", {})
            result = methods_mod.run_method(
                key,
                dataset=dataset,
                weights=weights,
                criterion_type=criterion_type,
                params=params,
                extra_inputs=extra_inputs,
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False,
                            "error": f"{type(exc).__name__}: {exc}"}), 400

    @app.route("/api/health", methods=["GET"])
    def api_health():
        return jsonify({"ok": True, "version": "1.2",
                        "host": _HANDLE.host if _HANDLE else None,
                        "port": _HANDLE.port if _HANDLE else None})

    @app.route("/api/shutdown", methods=["POST"])
    def api_shutdown():
        threading.Thread(target=_delayed_stop, daemon=True).start()
        return jsonify({"ok": True, "message": "Shutting down."})

    return app


def _delayed_stop():
    time.sleep(0.5)
    stop()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
class _ServerHandle:
    def __init__(self, server, thread, host, port):
        self.server = server
        self.thread = thread
        self.host = host
        self.port = port


_HANDLE = None
_LOCK = threading.Lock()


def _is_port_free(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
    except OSError:
        return False
    finally:
        s.close()
    return True


def _find_free_port(host, port):
    for p in range(port, port + 20):
        if _is_port_free(host, p):
            return p
    raise RuntimeError(
        f"No free port between {port} and {port + 19} on {host}.")


def is_running():
    return _HANDLE is not None


def start(host="127.0.0.1", port=5050, open_browser=True, quiet=True):
    """Start the GUI server on a daemon thread.

    Parameters
    ----------
    host         : str   bind address (default localhost only)
    port         : int   preferred port; auto-bumps within [port, port+19]
    open_browser : bool  try to open the URL automatically
    quiet        : bool  suppress werkzeug request logs
    """
    global _HANDLE
    with _LOCK:
        if _HANDLE is not None:
            url = f"http://{_HANDLE.host}:{_HANDLE.port}/"
            print(f"[pyDecision] Studio already running at {url}")
            return url

        port = _find_free_port(host, port)
        app = _create_app()
        if quiet:
            logging.getLogger("werkzeug").setLevel(logging.ERROR)
        server = make_server(host, port, app, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _HANDLE = _ServerHandle(server, thread, host, port)

        url = f"http://{host}:{port}/"
        _print_banner(url)
        if open_browser:
            try:
                webbrowser.open_new_tab(url)
            except Exception:
                pass
        return url


def stop(timeout=5.0):
    global _HANDLE
    with _LOCK:
        if _HANDLE is None:
            return False
        h = _HANDLE
        _HANDLE = None
    try:
        h.server.shutdown()
        h.thread.join(timeout=timeout)
    except Exception:
        pass
    print("[pyDecision] Studio stopped.")
    return True


def _print_banner(url):
    bar = "─" * 62
    msg = (
        f"\n  ┌{bar}┐\n"
        f"  │{' '*62}│\n"
        f"  │   pyDecision  —  MCDA Studio                                 │\n"
        f"  │   running at  {url:<46}│\n"
        f"  │   stop with   pyDecision.web_stop(){' '*26}│\n"
        f"  │{' '*62}│\n"
        f"  └{bar}┘\n"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()
