"""Send one control change on a slow sweep, so a DAW can learn it.

Usage, with loopMIDI running and FL Studio's "Remote control settings" dialog
open on the parameter you want to map::

    python analysis/midi_learn.py filter        # by effect name
    python analysis/midi_learn.py 74            # or by CC number
    python analysis/midi_learn.py filter 30     # ...for 30 seconds

Why a dedicated tool rather than just running the instrument
------------------------------------------------------------
FL Studio's auto-detect latches onto the *first* control change it sees. Run
the whole instrument to generate one, and six controls are moving at once: the
mapping lands on whichever happened to arrive first, which is rarely the one
intended and is tedious to undo. Sending exactly one CC removes the ambiguity.

The sweep matters too. A single message is easy for a host to miss between
window redraws, and a parameter learned from one value has no range to infer.
This walks the full 0..127 span and back, slowly enough to watch.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import midiout  # noqa: E402

RATE_HZ = 25.0
SWEEP_SECONDS = 2.5


def resolve(token: str) -> tuple[int, str]:
    """Accept either an effect name or a raw CC number."""
    if token.isdigit():
        cc = int(token)
        for name, number in config.MIDI_CONTROLS.items():
            if number == cc:
                return cc, name
        return cc, "(unassigned)"
    name = token.lower()
    if name not in config.MIDI_CONTROLS:
        raise SystemExit(
            f"unknown control {token!r}. Known: "
            + ", ".join(f"{n} (CC {c})" for n, c in config.MIDI_CONTROLS.items()))
    return config.MIDI_CONTROLS[name], name


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        print("controls:")
        for name, cc in config.MIDI_CONTROLS.items():
            print(f"  {name:<9} CC {cc}")
        return 1

    cc, label = resolve(sys.argv[1])
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

    try:
        import mido
    except ImportError:
        print("needs mido: pip install mido python-rtmidi")
        return 1

    ports = midiout.available_ports()
    target = next((p for p in ports
                   if (config.MIDI_OUT_PORT or "").lower() in p.lower()), None)
    if target is None:
        print(f"no output matching {config.MIDI_OUT_PORT!r}. Available: "
              + ", ".join(ports))
        return 1

    channel = max(0, min(15, config.MIDI_CHANNEL - 1))
    print(f"\nport    : {target}")
    print(f"sending : CC {cc} ({label}) on channel {channel + 1}")
    print(f"duration: {duration:.0f} s\n")
    print("In FL Studio: right-click the parameter, choose 'Link to")
    print("controller', leave Auto detect on, and wait for Ctrl to fill in.\n")

    sent = 0
    with mido.open_output(target) as out:
        end = time.perf_counter() + duration
        start = time.perf_counter()
        while time.perf_counter() < end:
            t = ((time.perf_counter() - start) % SWEEP_SECONDS) / SWEEP_SECONDS
            # Triangle sweep: up then down, so the host sees the whole range
            # and the parameter visibly moves in both directions.
            value = int(round((1.0 - abs(2.0 * t - 1.0)) * 127))
            out.send(mido.Message("control_change", channel=channel,
                                  control=cc, value=value))
            sent += 1
            bar = "#" * int(value / 127 * 40)
            print(f"\r  {value:3d} |{bar:<40}|", end="", flush=True)
            time.sleep(1.0 / RATE_HZ)

        out.send(mido.Message("control_change", channel=channel,
                              control=cc, value=0))

    print(f"\n\ndone - {sent} messages sent on CC {cc}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
