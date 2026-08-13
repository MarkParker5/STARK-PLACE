"""Dependency-free dev server for the STARK profiler + visualizer.

No fastapi/uvicorn/websockets — just the stdlib `http.server` in a background thread. It coexists with
the async engine by running each profiled utterance in its own event loop (serialized by a lock,
because `sys.monitoring` has a single profiler slot).

Endpoints:
    GET  /api/health                     -> {"ok": true}
    GET  /api/structure                  -> static auto-graph (no trace)
    GET  /api/demo                        -> run the demo utterance, return a full trace bundle
    POST /api/utterance   {"text": "..."} -> run it, return a full trace bundle
    GET  /api/stream?text=...             -> Server-Sent Events: the same events, streamed live
    GET  /...                             -> the built visualizer (visualizer/dist), if present

A "trace bundle" is {"events": [...], "steps": [...], "graph": {...}, "utterance": "..."}.

    python -m profiler.server            # serve on http://localhost:8765
    python -m profiler.server --port 9000
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .curate import curate
from .schema import ProfileEvent
from .structure import build_graph

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIST = os.path.join(_HERE, os.pardir, "visualizer", "dist")

# sys.monitoring has a single profiler slot → only one profiled run at a time.
_profile_lock = threading.Lock()


_NER = None  # spacy processor, loaded once


def _build_pipeline():
    """A rich, dependency-free processor pipeline: corrections (real Dictionary + phonetic tools) →
    spaCy NER → search → fallback. Exercises most of the engine so the visualizer lights up."""
    from stark.core.commands_context_processor import CommandsContextProcessor
    from stark.core.processors import CorrectionsProcessor, SearchProcessor, SpacyNERProcessor
    from stark.models.transcription_string import Correction
    from stark.tools.dictionary.dictionary import Dictionary, LookupMode
    from stark.tools.dictionary.storage.storage_memory import DictionaryStorageMemory
    from stark.tools.phonetic.transcription.latin_passthrough import LatinPassthroughProvider

    # dictionary of recognizable keywords — consulted for real (touches phonetic tools), while a
    # small typo map guarantees clean corrections in the offline demo (no espeak needed).
    dictionary = Dictionary(DictionaryStorageMemory(), ipa_provider=LatinPassthroughProvider())
    for name in ["turn", "kitchen", "metallica", "volume", "play", "weather"]:
        dictionary.write_one("en", name)

    class DemoCorrectionsProcessor(CorrectionsProcessor):
        TYPOS = {"tern": "turn", "kichen": "kitchen", "metalica": "metallica", "volumme": "volume",
                 "wether": "weather", "wheather": "weather", "playy": "play", "lites": "lights"}

        def _find_corrections(self, text, language_code, output):
            try:
                super()._find_corrections(text, language_code, [])  # touch Dictionary + phonetic tools
            except Exception:
                pass
            for word in str(text).lower().split():
                if word in self.TYPOS:
                    output.append(Correction(word, self.TYPOS[word]))

    class FallbackProcessor(CommandsContextProcessor):
        async def process_string(self, string, context, recognized_entities):
            return [], 0  # a dumb catch-all; only runs when nothing matched

    global _NER
    procs = [DemoCorrectionsProcessor(dictionaries=[dictionary], mode=LookupMode.FUZZY)]
    try:
        if _NER is None:
            _NER = SpacyNERProcessor({"en": "en_core_web_sm"})
        procs.append(_NER)
    except Exception:
        pass  # spaCy/model unavailable -> skip NER, rest still works
    procs += [SearchProcessor(), FallbackProcessor()]
    return procs


_DEEP_TYPES = None


def _deep_types():
    """Nested custom NLObject types so the Matching page can show a param→param→param recursion:
    Album.pattern contains $genre:Genre, and Genre.pattern contains $name:NLWord (a leaf regex)."""
    global _DEEP_TYPES
    if _DEEP_TYPES is not None:
        return _DEEP_TYPES
    from stark.core.types.object import NLObject
    from stark.core.patterns.pattern import Pattern
    from stark.general.classproperty import classproperty

    class Genre(NLObject):
        @classproperty
        def pattern(cls):
            return Pattern("$name:NLWord")

    class Album(NLObject):
        @classproperty
        def pattern(cls):
            return Pattern("$genre:Genre by $artist:NLWord")

    _DEEP_TYPES = (Genre, Album)
    return _DEEP_TYPES


def _demo_manager():
    import anyio

    from stark.core import CommandsManager, Response

    _deep_types()  # ensure the nested Album/Genre types exist before patterns are compiled
    manager = CommandsManager()

    @manager.new("stop", hidden=True)
    async def stop_music():
        return Response("Stopped.")

    # deep recursive parse: recommend → $album:Album → $genre:Genre → $name:NLWord (4 levels)
    @manager.new("recommend $album:Album")
    async def recommend(album):
        await anyio.sleep(0.03)
        return Response(f"Recommending {album}.")

    # commands await a little "work" so those dispatched together genuinely run in parallel on the
    # event loop (their call/response steps overlap in time, visible on the timeline).
    @manager.new("play $band:NLWord")
    async def play_music(band):
        await anyio.sleep(0.03)
        # returning `commands` pushes a music context -> grows the context tree
        return Response(f"Playing {band}.", commands=[stop_music])

    @manager.new("turn off the ** lights")
    async def lights_off():
        await anyio.sleep(0.03)
        return Response("Lights off.")

    @manager.new("what time is it")
    async def clock():
        await anyio.sleep(0.03)
        return Response("It's 5 o'clock.")

    @manager.new("weather in $city:NLWord")
    async def weather(city):
        await anyio.sleep(0.03)
        return Response(f"It's sunny in {city}.")

    @manager.new("hello $name:NLWord")
    async def hello(name):
        await anyio.sleep(0.03)
        return Response(f"Hello, {name}!")

    @manager.new("volume $level:NLWord percent")
    async def volume(level):
        await anyio.sleep(0.03)
        return Response(f"Volume {level}%.")

    # a NLString-typed parameter exercises a DIFFERENT NLObject parser than NLWord (greedy multi-word)
    @manager.new("note $text:NLString")
    async def note(text):
        await anyio.sleep(0.03)
        return Response(f"Noted: {text}.")

    @manager.new("set a timer for $dur:NLWord")
    async def set_timer(dur):
        # background command: a long-running generator that emits SEVERAL responses over time —
        # each a separate step later on the timeline, running in parallel with the rest of the
        # pipeline. This is the bg-command-with-delayed-responses case.
        yield Response(f"Timer set for {dur}.")
        await anyio.sleep(0.12)
        yield Response("Half the time is gone.")
        await anyio.sleep(0.12)
        yield Response("Time's up!")

    return manager


# predefined cases covering different zones of the engine.
CASES: list[dict] = [
    {"id": "corrections", "label": "Corrections + phonetics", "text": "play metalica and tern off the kichen lights"},
    {"id": "ner", "label": "NER (city entity)", "text": "wether in paris and tern off the lights"},
    {"id": "context", "label": "Context push (music)", "text": "play metallica"},
    {"id": "simple", "label": "Single command", "text": "what time is it"},
    {"id": "fallback", "label": "Fallback (no match)", "text": "do a barrel roll"},
    {"id": "string", "label": "NLString parser (multi-word)", "text": "note buy milk and eggs"},
    {"id": "deep", "label": "Deep recursive parse (Album→Genre→NLWord)", "text": "recommend rock by radiohead"},
    {"id": "absolute", "label": "Everything (bg + delayed)", "text": "wether in paris play metalica set a timer for five and tern off the kichen lites"},
]


class _CollectingDelegate:
    def __init__(self):
        self.responses = []

    async def commands_context_did_receive_response(self, response):
        self.responses.append(response)

    def remove_response(self, response):
        pass


def run_trace(utterance: str, manager=None) -> list[ProfileEvent]:
    """Run one utterance through the real engine under profiling; return the captured events.

    Runs the response loop briefly too, so the full cycle (respond -> _process_response ->
    add_context -> delegate) is captured for context-pushing commands.
    """
    from profiler import profile
    from stark.core import CommandsContext
    import anyio
    import asyncer

    from stark.models.transcription_string import TranscriptionString

    manager = manager or _demo_manager()
    processors = _build_pipeline()

    async def _run():
        with profile() as session:
            async with asyncer.create_task_group() as task_group:
                context = CommandsContext(task_group=task_group, commands_manager=manager, processors=processors)
                # register the nested demo types so "recommend $album:Album" resolves Album→Genre→NLWord
                for _t in _deep_types():
                    try:
                        context.pattern_parser.register_parameter_type(_t)
                    except Exception:
                        pass
                context.delegate = _CollectingDelegate()
                task_group.soonify(context.handle_responses)()
                # TranscriptionString input so CorrectionsProcessor can attach corrections
                await context.process_string(TranscriptionString(utterance))
                with anyio.move_on_after(0.7):
                    await anyio.sleep(0.5)  # let responses drain (incl. the bg timer's 3 delayed ones)
                context.stop()
        return session.events

    with _profile_lock:
        return asyncio.run(_run())


def trace_bundle(utterance: str) -> dict:
    events = run_trace(utterance)
    return {
        "utterance": utterance,
        "events": [e.to_dict() for e in events],
        "steps": [s.to_dict() for s in curate(events)],
        "graph": build_graph(events),
    }


def reparse(pattern_str: str, string: str) -> dict:
    """Run ONE PatternParser.match(pattern, string) in isolation — for the Matching page's
    'edit a sub-parse input and re-run it' affordance. Stateless; touches no stored trace."""
    from stark.core.parsing import PatternParser
    from stark.core.patterns.pattern import Pattern
    from .extractors import serialize

    async def _run():
        parser = PatternParser()
        try:
            return await parser.match(Pattern(pattern_str), string), None
        except Exception as exc:  # surface parser errors to the UI instead of 500ing
            return [], f"{type(exc).__name__}: {exc}"

    with _profile_lock:
        matches, error = asyncio.run(_run())
    return {"pattern": pattern_str, "string": string, "matches": serialize(matches), "error": error}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    # helpers ---------------------------------------------------------------

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_preflight(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_static(self, path: str):
        rel = path.lstrip("/") or "index.html"
        full = os.path.normpath(os.path.join(_DIST, rel))
        if not full.startswith(os.path.abspath(_DIST)) or not os.path.isfile(full):
            full = os.path.join(_DIST, "index.html")  # SPA fallback
        if not os.path.isfile(full):
            self._send_json({"error": "visualizer not built; run `npm run build` in visualizer/"}, 404)
            return
        ctype = {
            ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
            ".json": "application/json", ".svg": "image/svg+xml",
        }.get(os.path.splitext(full)[1], "application/octet-stream")
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # routes ----------------------------------------------------------------

    def do_OPTIONS(self):
        self._cors_preflight()

    def do_GET(self):
        url = urlparse(self.path)
        route = url.path
        if route == "/api/health":
            self._send_json({"ok": True})
        elif route == "/api/structure":
            self._send_json(build_graph())
        elif route == "/api/cases":
            self._send_json({"cases": CASES})
        elif route == "/api/demo":
            self._send_json(trace_bundle(CASES[-1]["text"]))  # the "absolute" case
        elif route == "/api/stream":
            text = parse_qs(url.query).get("text", ["play Metallica"])[0]
            self._stream(text)
        else:
            self._serve_static(route)

    def do_POST(self):
        route = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, AttributeError):
            self._send_json({"error": "bad request"}, 400)
            return
        if route == "/api/reparse":
            # sandboxed single-parser re-run for the Matching page's "edit & re-run" — NEVER mutates
            # the stored trace; the client shows the result as MODIFIED.
            self._send_json(reparse(str(payload.get("pattern", "")), str(payload.get("string", ""))))
            return
        if route != "/api/utterance":
            self._send_json({"error": "not found"}, 404)
            return
        text = str(payload.get("text", "")).strip()
        if not text:
            self._send_json({"error": "empty utterance"}, 400)
            return
        self._send_json(trace_bundle(text))

    def _stream(self, text: str):
        """Server-Sent Events: replay the captured events one per frame for a live feel."""
        import time

        events = run_trace(text)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            for e in events:
                self.wfile.write(f"data: {json.dumps(e.to_dict())}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01)
            self.wfile.write(b"event: done\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(port: int = 8765) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"STARK profiler server on http://localhost:{port}")
    print(f"  GET  /api/demo         run the demo utterance")
    print(f"  POST /api/utterance    {{'text': '...'}}")
    print(f"  GET  /api/structure    static auto-graph")
    print(f"  visualizer dist: {'found' if os.path.isdir(_DIST) else 'not built yet'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def main() -> None:
    import sys

    port = 8765
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    serve(port)


if __name__ == "__main__":
    main()
