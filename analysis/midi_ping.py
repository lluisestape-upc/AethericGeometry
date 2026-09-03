"""Play a few notes down the MIDI cable, so you can hear whether it works.

Usage, with loopMIDI running and FL Studio open::

    python analysis/midi_ping.py

Why notes rather than a control change
--------------------------------------
Checking a cable by sending a control change requires finding a blinking
indicator in the host and trusting that you have found the right one. Notes
need no indicator: if the DAW is receiving, the selected instrument plays, and
you hear it. A test whose result arrives through your ears has no ambiguity to
argue with.

If you hear the notes, everything from this program through the virtual cable
into the DAW is working, and any remaining problem is in the mapping. If you
hear nothing, the problem is upstream of the mapping and there is no point
touching a knob yet.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import midiout  # noqa: E402

# A rising arpeggio, so it is obviously deliberate rather than a stuck note.
MELODY = [60, 64, 67, 72, 67, 64, 60]
NOTE_SECONDS = 0.35


def main() -> int:
    try:
        import mido
    except ImportError:
        print("needs mido: pip install mido python-rtmidi")
        return 1

    ports = [p for p in midiout.available_ports() if "loopmidi" in p.lower()]
    if not ports:
        print("no loopMIDI output found. Is loopMIDI running with a port?")
        return 1

    # Every loopMIDI port in turn, announced. Sending only to the configured
    # one turns "did you hear it" into two questions at once - is the cable
    # working, and is the DAW listening to *this* cable - and a failure cannot
    # tell them apart. Walking all of them removes the second question.
    channel = max(0, min(15, config.MIDI_CHANNEL - 1))
    print(f"\nchannel: {channel + 1}")
    print("\nIn FL Studio, click the instrument in the Channel rack first, so")
    print("incoming notes go to it. Then listen for which port makes sound.\n")

    for port_name in ports:
        print(f"--- {port_name} ---")
        with mido.open_output(port_name) as out:
            try:
                for note in MELODY:
                    out.send(mido.Message("note_on", channel=channel,
                                          note=note, velocity=100))
                    time.sleep(NOTE_SECONDS)
                    out.send(mido.Message("note_off", channel=channel,
                                          note=note, velocity=0))
            finally:
                # Never leave a note hanging, whatever happens above.
                for note in MELODY:
                    out.send(mido.Message("note_off", channel=channel,
                                          note=note, velocity=0))
        time.sleep(0.8)

    print("\nHeard one of them?")
    print("  -> note which, and set midi.port in config.yaml to that name.")
    print("Heard neither?")
    print("  -> FL is not receiving at all. Before touching MIDI settings,")
    print("     check FL makes sound by itself: click a step in the 3x Osc")
    print("     row of the pattern and press play. If that is silent too,")
    print("     the problem is FL's audio device, not MIDI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
