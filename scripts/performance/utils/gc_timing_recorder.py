# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded, opt-in trainer cyclic-GC timing; never changes collection policy.

Body/inter-step intervals are callback spans, not the framework's logged timer
windows. GC wall spans include finalizers, scheduling, and intervening callbacks;
they are not exclusively collector CPU or necessarily GPU critical-path time.
The terminal phase-2 tail includes post-loop checkpoint/finalization teardown.
Monotonic clocks are process-local evidence; no cross-node clock alignment is
performed. Compare logical bins across ranks, never subtract their raw clocks.
Workers, reference-count destruction, and C++/CUDA allocator work are unobserved.
Missing terminal output (including SIGKILL) is incomplete capture, not zero GC.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import sys
import threading
import time
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.utils.common_utils import get_rank_safe


if TYPE_CHECKING:
    from megatron.bridge.training.state import TrainState


logger = logging.getLogger(__name__)
EVENT_CAPACITY = 8192
EVENT_FIELDS = 10
LONG_EVENT_NS = 1_000_000
MAX_OUTPUT_INTERVALS = 256
MAX_OUTPUT_LONG_EVENTS = 128
GC_START, GC_STOP, BODY_START, BODY_END, TRAIN_START, TRAIN_END = range(1, 7)


class GCTimingCallback(Callback):
    """Record fixed-width numeric events, then emit bounded rank-local logs.

    The default diagnostic budget is 8192 * 10 * 8 = 655360 numeric bytes/rank.
    Python bookkeeping and post-measurement summarization add overhead. No IO,
    tensor work, collectives, object inspection, or policy changes occur in the
    GC callback. Observer overhead and changed allocation cadence remain real.
    """

    def __init__(self, *, capacity: int, long_event_ns: int) -> None:
        if not 2 <= capacity <= (1024 * 1024) // (EVENT_FIELDS * 8):
            raise ValueError("GC event buffer must contain 2 or more rows and at most 1 MiB.")
        if long_event_ns < 0:
            raise ValueError("Long-event threshold must be nonnegative.")
        self.events = array("q", [0]) * (capacity * EVENT_FIELDS)
        if self.events.itemsize != 8:
            raise RuntimeError("GC timing requires an eight-byte signed numeric array.")
        self.capacity = capacity
        self.long_event_ns = long_event_ns
        self.owner_pid = os.getpid()
        self.rank = -1
        self.count = 0
        self.dropped = 0
        self.started = False
        self.ended = False
        self.active = False
        self.reported = False
        self.logical_iteration = 0
        self.phase = 0  # 0: before first body; 1: body; 2: inter-step/maintenance.
        self.state: TrainState | None = None
        self._callback = self._on_gc  # Keep one identity for registration/removal.
        self._wall_clock = time.perf_counter_ns
        self._cpu_clock = time.thread_time_ns
        self._enabled = False
        self._thresholds = (0, 0, 0)
        self._callback_count = 0
        self._attachment_thread = -1
        self._main_thread = -1
        self._source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def _record(self, kind: int, generation: int, collected: int, uncollectable: int) -> None:
        if self.count == self.capacity:
            self.dropped += 1
            return
        offset = self.count * EVENT_FIELDS
        self.count += 1
        self.events[offset] = kind
        self.events[offset + 1] = self._wall_clock()
        self.events[offset + 2] = self._cpu_clock()
        self.events[offset + 3] = self.state.step if self.state is not None else -1
        self.events[offset + 4] = self.logical_iteration
        self.events[offset + 5] = self.phase
        self.events[offset + 6] = generation
        self.events[offset + 7] = threading.get_native_id()
        self.events[offset + 8] = collected
        self.events[offset + 9] = uncollectable

    def _on_gc(self, phase: str, info: dict[str, int]) -> None:
        if not self.active or os.getpid() != self.owner_pid:
            return
        if phase == "start":
            self._record(GC_START, info["generation"], 0, 0)
        elif phase == "stop":
            self._record(GC_STOP, info["generation"], info["collected"], info["uncollectable"])

    def on_train_start(self, context: CallbackContext) -> None:
        """Attach after model/data initialization, excluding existing workers."""
        if os.getpid() != self.owner_pid:
            return
        if self.started:
            raise RuntimeError("GC timing supports one training interval per recorder.")
        self.state = context.state.train_state
        self.rank = get_rank_safe()
        self._enabled = gc.isenabled()
        self._thresholds = gc.get_threshold()
        self._callback_count = len(gc.callbacks)
        self._attachment_thread = threading.get_native_id()
        self._main_thread = threading.main_thread().native_id or -1
        self.started = True
        self._record(TRAIN_START, -1, 0, 0)
        self.active = True
        gc.callbacks.append(self._callback)

    def on_train_step_start(self, context: CallbackContext) -> None:
        """Anchor the body using the observed preincrement step counter."""
        if self.active and os.getpid() == self.owner_pid:
            self.logical_iteration = context.state.train_state.step + 1
            self.phase = 1
            self._record(BODY_START, -1, 0, 0)

    def on_train_step_end(self, context: CallbackContext) -> None:
        """End the body; subsequent maintenance stays in the inter-step bin."""
        if self.active and os.getpid() == self.owner_pid:
            self._record(BODY_END, -1, 0, 0)
            self.phase = 2

    def on_train_end(self, context: CallbackContext) -> None:
        """Stop measurement before output serialization and final cleanup."""
        if self.active and os.getpid() == self.owner_pid:
            self._record(TRAIN_END, -1, 0, 0)
            self.ended = True
            self.detach()

    def detach(self) -> None:
        """Idempotently remove only this process's callback by identity."""
        if os.getpid() != self.owner_pid:
            return
        self.active = False
        for index in range(len(gc.callbacks) - 1, -1, -1):
            if gc.callbacks[index] is self._callback:
                del gc.callbacks[index]

    def close(self) -> None:
        """Detach even on exceptions and emit a single bounded terminal report."""
        if self.reported or os.getpid() != self.owner_pid:
            return
        self.detach()
        self.reported = True
        # The training setup filters WARNING records; INFO reaches existing rank logs.
        logger.setLevel(logging.INFO)
        self._report()

    def _emit(self, record: dict[str, object]) -> None:
        logger.info(
            "GC_TIMING %s", json.dumps({"rank": self.rank, "pid": self.owner_pid, **record}, separators=(",", ":"))
        )

    def _report(self) -> None:
        # All dynamic summarization and logging is deliberately after detach.
        rows = [tuple(self.events[i : i + EVENT_FIELDS]) for i in range(0, self.count * EVENT_FIELDS, EVENT_FIELDS)]
        rows.sort(key=lambda row: row[1])
        pending_gc = None
        pending_body = None
        previous_end = None
        collections = []
        intervals = []
        unpaired_gc = 0
        unpaired_body = 0
        for row in rows:
            kind = row[0]
            if kind == GC_START:
                unpaired_gc += int(pending_gc is not None)
                pending_gc = row
            elif kind == GC_STOP:
                if pending_gc is None:
                    unpaired_gc += 1
                elif (pending_gc[6], pending_gc[7]) != (row[6], row[7]):
                    unpaired_gc += 2
                else:
                    collections.append((pending_gc, row))
                pending_gc = None
            elif kind == BODY_START:
                unpaired_body += int(pending_body is not None)
                if previous_end is not None:
                    intervals.append((2, previous_end, row))
                pending_body = row
                previous_end = None
            elif kind == BODY_END:
                if pending_body is None or pending_body[4] != row[4]:
                    unpaired_body += 1 + int(pending_body is not None)
                else:
                    intervals.append((1, pending_body, row))
                pending_body = None
                previous_end = row
            elif kind == TRAIN_END and previous_end is not None:
                intervals.append((2, previous_end, row))
                previous_end = None
        unpaired_gc += int(pending_gc is not None)
        unpaired_body += int(pending_body is not None)
        long_events = [(start, stop) for start, stop in collections if stop[1] - start[1] >= self.long_event_ns]
        retained_long_events = sorted(long_events, key=lambda event: event[1][1] - event[0][1], reverse=True)[
            :MAX_OUTPUT_LONG_EVENTS
        ]
        retained_long_events.sort(key=lambda event: event[0][1])
        self._emit(
            {
                "kind": "summary",
                "version": 1,
                "source_sha256": self._source_hash,
                "python": list(sys.version_info[:3]),
                "gc_enabled": int(self._enabled),
                "gc_thresholds": list(self._thresholds),
                "existing_callbacks": self._callback_count,
                "attachment_thread": self._attachment_thread,
                "main_thread": self._main_thread,
                "started": int(self.started),
                "train_end_seen": int(self.ended),
                "events": self.count,
                "buffer_bytes": len(self.events) * self.events.itemsize,
                "dropped_events": self.dropped,
                "unpaired_gc_endpoints": unpaired_gc,
                "unpaired_body_endpoints": unpaired_body,
                "collections": len(collections),
                "collection_span_wall_ns": sum(stop[1] - start[1] for start, stop in collections),
                "collection_thread_cpu_ns": sum(stop[2] - start[2] for start, stop in collections),
                "long_event_threshold_ns": self.long_event_ns,
                "long_events": len(long_events),
                "omitted_long_events": max(0, len(long_events) - MAX_OUTPUT_LONG_EVENTS),
                "intervals": len(intervals),
                "omitted_intervals": max(0, len(intervals) - MAX_OUTPUT_INTERVALS),
                "capture_complete": int(
                    self.started and self.ended and not (self.dropped or unpaired_gc or unpaired_body)
                ),
                "output_complete": int(
                    len(long_events) <= MAX_OUTPUT_LONG_EVENTS and len(intervals) <= MAX_OUTPUT_INTERVALS
                ),
            }
        )
        for phase, start, stop in intervals[:MAX_OUTPUT_INTERVALS]:
            self._emit(
                {
                    "kind": "interval",
                    "phase": phase,
                    "iteration": start[4],
                    "next_iteration": stop[4] if stop[0] == BODY_START else -1,
                    "terminal_tail": int(stop[0] == TRAIN_END),
                    "start_counter": start[3],
                    "stop_counter": stop[3],
                    "start_ns": start[1],
                    "stop_ns": stop[1],
                    "wall_ns": stop[1] - start[1],
                    "collection_overlap_wall_ns": sum(
                        max(0, min(stop[1], gc_stop[1]) - max(start[1], gc_start[1]))
                        for gc_start, gc_stop in collections
                    ),
                }
            )
        for start, stop in retained_long_events:
            self._emit(
                {
                    "kind": "gc",
                    "generation": start[6],
                    "thread": start[7],
                    "start_ns": start[1],
                    "stop_ns": stop[1],
                    "start_cpu_ns": start[2],
                    "stop_cpu_ns": stop[2],
                    "start_counter": start[3],
                    "stop_counter": stop[3],
                    "start_iteration": start[4],
                    "stop_iteration": stop[4],
                    "start_phase": start[5],
                    "stop_phase": stop[5],
                    "collected": stop[8],
                    "uncollectable": stop[9],
                }
            )
