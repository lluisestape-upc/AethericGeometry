"""MIDI output: drive any synth in any DAW, not just the built-in engine.

What this is for
----------------
Everything upstream of this module - hand tracking, the rotary knob, the voice
layer - is a control surface. Until now the only thing it could control was the
oscillator in :mod:`synth`. Sending the same values out as MIDI CC turns it into
a general-purpose controller: the built-in engine becomes the demo, and the
actual product is the surface.

Wiring it up on Windows
-----------------------
Windows has no virtual MIDI port of its own, so the loop needs a helper:

1. Install **loopMIDI** (Tobias Erichsen, free) and create a port.
2. Set ``midi.enabled: true`` and ``midi.port`` to that port's name.
3. In the DAW, right-click the target knob - 3x Osc's cutoff, Sytrus, Serum,
   a mixer send - choose *Link to controller*, then move your hand. FL Studio
   learns whichever CC arrives.

Design notes
------------
**Only send on change, and rate-limit.** A gesture updates every video frame,
so a naive implementation emits 30 messages per second per parameter whether or
not anything moved. That floods a 31,250 baud MIDI stream, and a flooded stream
shows up as laggy, stepped automation in the DAW. Values are therefore
quantised to the 0..127 that CC actually carries, compared against the last
value sent, and dropped if identical - a still hand produces no traffic at all.

**Pitch as pitch bend, not as notes.** The instrument's pitch is continuous,
and re-triggering a note for every semitone would sound like a machine gun. One
note is held and bent, which is what a real controller does; the bend range is
whatever the receiving synth is set to, so it is reported rather than assumed.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

try:
    import mido
    _MIDI_OK = True
except ImportError:  # pragma: no cover - exercised on installs without mido
    _MIDI_OK = False

#: Pitch-bend range the receiving synth is assumed to use, in semitones. Two is
#: the near-universal default. Anything wider only matters for large glides.
BEND_SEMITONES = 2.0


def available_ports() -> list[str]:
    """Names of MIDI outputs, or an empty list if mido is missing."""
    if not _MIDI_OK:
        return []
    try:
        return list(mido.get_output_names())
    except Exception as exc:  # pragma: no cover - platform dependent
        log.debug("Could not list MIDI outputs: %s", exc)
        return []


def _pick_port(preferred: str | None) -> str | None:
    """Choose an output: the requested one, else the first plausible."""
    ports = available_ports()
    if not ports:
        return None
    if preferred:
        for name in ports:
            if preferred.lower() in name.lower():
                return name
        log.warning("MIDI port %r not found. Available: %s",
                    preferred, ", ".join(ports))
        return None
    # The Microsoft GS wavetable synth is present on every Windows machine and
    # is almost never what someone wants to drive; skip it when guessing.
    for name in ports:
        if "wavetable" not in name.lower():
            return name
    log.warning("Only the built-in GM synth is available, so that is what "
                "MIDI will drive - expect a General MIDI piano, not your DAW. "
                "Install loopMIDI, create a port, and name it in midi.port.")
    return ports[0]


class MidiController:
    """Mirrors effect values, pitch and note state out as MIDI.

    Degrades exactly like the voice layer: if mido is missing or no port can be
    opened it reports ``available == False`` and every method is a no-op, so
    nothing upstream needs to care.
    """

    def __init__(self, enabled: bool = False, port: str | None = None,
                 channel: int = 1, rate_hz: float = 50.0,
                 send_notes: bool = True,
                 controls: dict[str, int] | None = None):
        self.available = False
        self.status = "disabled"
        self.port_name: str | None = None

        self._channel = max(0, min(15, int(channel) - 1))   # 1-16 -> 0-15
        self._min_interval = 1.0 / max(1.0, float(rate_hz))
        self._send_notes = bool(send_notes)
        self._controls = dict(controls or {})
        self._out = None

        self._last_cc: dict[int, int] = {}
        self._last_sent_at: dict[int, float] = {}
        self._last_note: int | None = None
        self._last_bend = 0

        if not enabled:
            return
        if not _MIDI_OK:
            self.status = "mido not installed"
            log.info("MIDI off - pip install mido python-rtmidi.")
            return

        name = _pick_port(port)
        if name is None:
            self.status = "no MIDI output found"
            log.warning("MIDI on but no output port. On Windows install "
                        "loopMIDI and create a port, then set midi.port.")
            return
        try:
            self._out = mido.open_output(name)
        except Exception as exc:
            self.status = f"could not open {name}: {exc}"
            log.warning("MIDI port %r would not open (%s).", name, exc)
            return

        self.available = True
        self.port_name = name
        self.status = f"sending on {name}"
        log.info("MIDI out: %s (channel %d)", name, self._channel + 1)

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        if self._out is None:
            return
        try:
            self.all_notes_off()
            self._out.close()
        except Exception as exc:  # pragma: no cover
            log.debug("MIDI close: %s", exc)
        finally:
            self._out = None
            self.available = False

    # -- controls ------------------------------------------------------------
    def send_effects(self, effects: dict[str, float]) -> None:
        """Mirror normalised effect values as CC. Silent when nothing moved."""
        if not self.available:
            return
        now = time.monotonic()
        for name, cc in self._controls.items():
            if name not in effects:
                continue
            value = int(round(max(0.0, min(1.0, float(effects[name]))) * 127))
            if self._last_cc.get(cc) == value:
                continue
            if now - self._last_sent_at.get(cc, 0.0) < self._min_interval:
                continue
            self._send(mido.Message("control_change", channel=self._channel,
                                    control=cc, value=value))
            self._last_cc[cc] = value
            self._last_sent_at[cc] = now

    def send_pitch(self, freq_hz: float | None) -> None:
        """Hold one note and bend it, rather than retriggering per semitone."""
        if not self.available or not self._send_notes:
            return
        if freq_hz is None or freq_hz <= 0.0:
            self.all_notes_off()
            return

        import math
        midi_float = 69.0 + 12.0 * math.log2(freq_hz / 440.0)
        note = int(round(midi_float))
        if not 0 <= note <= 127:
            return

        if note != self._last_note:
            if self._last_note is not None:
                self._send(mido.Message("note_off", channel=self._channel,
                                        note=self._last_note, velocity=0))
            self._send(mido.Message("note_on", channel=self._channel,
                                    note=note, velocity=90))
            self._last_note = note

        # Residual cents, expressed over the synth's bend range.
        offset = midi_float - note
        bend = int(round(offset / BEND_SEMITONES * 8191))
        bend = max(-8192, min(8191, bend))
        if bend != self._last_bend:
            self._send(mido.Message("pitchwheel", channel=self._channel,
                                    pitch=bend))
            self._last_bend = bend

    def all_notes_off(self) -> None:
        if not self.available:
            return
        if self._last_note is not None:
            self._send(mido.Message("note_off", channel=self._channel,
                                    note=self._last_note, velocity=0))
            self._last_note = None
        if self._last_bend != 0:
            self._send(mido.Message("pitchwheel", channel=self._channel,
                                    pitch=0))
            self._last_bend = 0

    # -- plumbing ------------------------------------------------------------
    def _send(self, message) -> None:
        try:
            self._out.send(message)
        except Exception as exc:  # pragma: no cover - device unplugged mid-run
            log.warning("MIDI send failed (%s) - disabling output.", exc)
            self.available = False
            self.status = f"send failed: {exc}"
