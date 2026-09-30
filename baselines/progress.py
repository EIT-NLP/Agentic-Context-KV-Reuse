"""Flushed phase/progress logs for quality and training, never efficiency metrics."""
from contextvars import ContextVar
import json
import os
import threading
import time

_active = ContextVar('baseline_progress', default=None)


def phase(name, **fields):
    reporter = _active.get()
    if reporter is not None:
        reporter.phase(name, **fields)


class Progress:
    def __init__(self, interval=None, **context):
        self.interval = float(interval if interval is not None else os.environ.get('BASELINE_LOG_INTERVAL', 30))
        if self.interval <= 0:
            raise ValueError('BASELINE_LOG_INTERVAL must be positive')
        self.context = {'pid': os.getpid(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES', 'default'), **context}
        self.started = self.changed = time.monotonic()
        self.state = dict(phase='startup')
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def _emit(self, event, **extra):
        now = time.monotonic()
        # Stage fields may repeat context (e.g. rank); merge rather than pass
        # duplicate keyword arguments. Logger-owned event/timing fields win.
        record = {**self.context, **self.state, **extra,
                  'time': time.strftime('%Y-%m-%d %H:%M:%S%z'), 'event': event,
                  'elapsed_s': round(now-self.started, 1),
                  'last_event_age_s': round(now-self.changed, 1)}
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def phase(self, name, **fields):
        with self.lock:
            self.changed = time.monotonic()
            self.state = {**fields, 'phase': name}
            self._emit('progress')

    def update(self, **fields):
        """Refresh heartbeat context without emitting one line per micro-sample."""
        with self.lock:
            self.state.update(fields)
            self.changed = time.monotonic()

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                self._emit('heartbeat', note='process alive; not evidence that the current operation is advancing')

    def __enter__(self):
        self.token = _active.set(self)
        with self.lock:
            self._emit('process_start')
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join(timeout=1)
        with self.lock:
            self._emit('process_failed' if error is not None else 'process_done',
                       **(dict(error_type=kind.__name__, error=str(error)) if error is not None else {}))
        _active.reset(self.token)
