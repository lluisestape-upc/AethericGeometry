"""Measure what the MIDI layer actually costs and delivers.

Run from the project root, with loopMIDI running and a port created::

    python analysis/measure_midi.py

Two questions, because they fail in different ways.

**Transport latency.** How long from the instrument deciding to send a control
change to a receiver being able to act on it. A DAW is on the far side of the
same virtual cable, so this is the delay a mapped synth parameter inherits on
top of whatever the gesture pipeline already spent. It is measured by opening
the loopback port as *both* an output and an input and timing the round trip:
timing ``send()`` alone would measure the call returning, not the message
arriving, which is a different and much smaller number.

**Traffic.** How many messages leave per second under a still hand and under a
moving one. This is the number that decides whether automation in the DAW looks
smooth or stepped, and whether the 31,250 baud MIDI stream is being flooded.
The engine sends only on change and rate-limits, so a still hand should produce
exactly nothing - and "exactly nothing" is a claim worth checking rather than
asserting.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetheric import config  # noqa: E402
from aetheric import midiout  # noqa: E402

SAMPLES = 200
FRAME_HZ = 30.0
MOVING_SECONDS = 4.0


def _pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _base_name(name: str) -> str:
    """Port name without rtmidi's trailing per-direction index.

    The same loopback cable is not called the same thing on both sides: rtmidi
    appends each port's ordinal *within its own direction's list*, so the
    output 'loopMIDI Port 1' is the input 'loopMIDI Port 0'. Matching on the
    full string finds nothing; matching on the stem pairs them correctly.
    """
    return name.rstrip("0123456789").strip()


def measure_latency(port_name: str) -> None:
    import mido

    print(f"\n=== Transport latency on {port_name!r} ===")
    stem = _base_name(port_name)
    inputs = [n for n in mido.get_input_names() if _base_name(n) == stem]
    if not inputs:
        print(f"  no input side matching {stem!r}; loopback cannot be timed")
        return

    try:
        outp = mido.open_output(port_name)
        inp = mido.open_input(inputs[0])
        print(f"  out {port_name!r}  ->  in {inputs[0]!r}")
    except Exception as exc:
        print(f"  could not open the port both ways: {exc}")
        return

    # Warm the driver: the first few messages after opening carry allocation
    # costs that no steady-state player ever pays.
    for _ in range(20):
        outp.send(mido.Message("control_change", control=74, value=0))
    time.sleep(0.2)
    while inp.poll() is not None:
        pass

    deltas = []
    with outp, inp:
        for i in range(SAMPLES):
            value = i % 128
            sent = time.perf_counter()
            outp.send(mido.Message("control_change", control=74, value=value))
            while True:
                msg = inp.poll()
                if msg is None:
                    if time.perf_counter() - sent > 0.5:
                        break            # lost; do not stall the run
                    continue
                if msg.type == "control_change" and msg.value == value:
                    deltas.append((time.perf_counter() - sent) * 1000.0)
                    break
            time.sleep(0.004)

    if not deltas:
        print("  nothing came back - is another application holding the port?")
        return

    print(f"  paired {len(deltas)}/{SAMPLES}")
    print(f"    min  {min(deltas):6.3f} ms")
    print(f"    p50  {_pct(deltas, 0.50):6.3f} ms   <- the figure that matters")
    print(f"    mean {statistics.mean(deltas):6.3f} ms")
    print(f"    p95  {_pct(deltas, 0.95):6.3f} ms")
    print(f"    max  {max(deltas):6.3f} ms")


class _CountingController(midiout.MidiController):
    """Wraps the real controller and counts what it puts on the wire."""

    def __init__(self, *args, **kwargs):
        self.sent = 0
        super().__init__(*args, **kwargs)

    def _send(self, message):
        self.sent += 1
        super()._send(message)


def measure_traffic() -> None:
    print("\n=== Messages per second ===")
    ctrl = _CountingController(True, config.MIDI_OUT_PORT, config.MIDI_CHANNEL,
                               config.MIDI_RATE_HZ, config.MIDI_SEND_NOTES,
                               config.MIDI_CONTROLS)
    if not ctrl.available:
        print(f"  {ctrl.status}")
        return

    still = {name: 0.5 for name in config.MIDI_CONTROLS}
    frames = int(FRAME_HZ * MOVING_SECONDS)

    ctrl.send_effects(still)             # first frame legitimately sends
    ctrl.sent = 0
    for _ in range(frames):
        ctrl.send_effects(still)
        time.sleep(1.0 / FRAME_HZ)
    print(f"  still hand, {frames} frames  -> {ctrl.sent} messages "
          f"({ctrl.sent / MOVING_SECONDS:.1f}/s)")

    ctrl.sent = 0
    start = time.perf_counter()
    for i in range(frames):
        t = i / frames
        moving = {name: (t * (1 + k * 0.13)) % 1.0
                  for k, name in enumerate(config.MIDI_CONTROLS)}
        ctrl.send_effects(moving)
        time.sleep(1.0 / FRAME_HZ)
    elapsed = time.perf_counter() - start
    print(f"  all {len(config.MIDI_CONTROLS)} controls sweeping "
          f"-> {ctrl.sent} messages ({ctrl.sent / elapsed:.1f}/s)")

    # A full MIDI stream carries ~1041 three-byte messages per second.
    print(f"  MIDI 1.0 bandwidth used: "
          f"{ctrl.sent / elapsed / 1041 * 100:.1f}%")
    ctrl.close()


def main() -> int:
    ports = midiout.available_ports()
    if not ports:
        print("no MIDI outputs at all")
        return 1
    print("outputs:", ", ".join(ports))

    target = None
    for name in ports:
        if (config.MIDI_OUT_PORT or "").lower() in name.lower():
            target = name
            break
    if target is None:
        print(f"no port matching {config.MIDI_OUT_PORT!r}")
        return 1

    measure_latency(target)
    measure_traffic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
