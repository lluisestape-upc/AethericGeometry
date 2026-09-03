"""Latency budget from a moving hand to an audible change in the DAW.

Run from the project root::

    python analysis/measure_chain.py             # stage A only, no setup needed
    python analysis/measure_chain.py --acoustic  # A + B, needs FL configured

Why this is measured in two stages
----------------------------------
The chain has four terms and they are not equally knowable:

    camera -> app decides         (A) ours, measurable in-process
    app -> message on the wire    (A) ours
    wire -> DAW receives          transport, 5.65 ms median, measured already
    DAW -> audible change         (B) the host buffer and its smoothing

Reporting one end-to-end number hides which of these is worth attacking. Stage
A is the part the instrument owns and can improve; stage B belongs to the host
and its buffer setting. Measuring them apart is what makes the total actionable
rather than merely alarming.

Stage B and the microphone
--------------------------
This machine offers no software route to capture its own output: WASAPI
loopback is unsupported in this sounddevice build, and the WDM-KS capture
endpoints fail at the driver with "Failed to read capture position register".
Rather than require another virtual-audio install, stage B listens
acoustically - speaker, air, microphone - and first *calibrates that path* by
emitting a click from this process and timing its return. That figure is then
subtracted, so what remains is the host contribution rather than the room.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

FRAMES = 200
SR = 48000


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def report(label, values):
    if not values:
        print(f"  {label:<34} (no samples)")
        return
    print(f"  {label:<34} p50 {pct(values, .50):6.2f}  "
          f"p95 {pct(values, .95):6.2f}  max {max(values):6.2f} ms")


# =============================================================================
def stage_a() -> None:
    """Camera frame in, MIDI message out, timed inside the real pipeline."""
    import cv2
    import mediapipe as mp

    import midiout
    from audio_map import compute_poly_sound
    from gestures import ALL_TIPS, assign_hands, build_polygon_points

    print("\n=== Stage A: camera frame -> MIDI on the wire ===")
    print(f"    {FRAMES} frames. Keep both hands in view for a real reading.\n")

    cam = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cam.isOpened():
        print("  no camera")
        return
    hands = mp.solutions.hands.Hands(
        min_detection_confidence=config.MP_MIN_DETECTION,
        min_tracking_confidence=config.MP_MIN_TRACKING,
        max_num_hands=config.MP_MAX_HANDS)
    midi = midiout.MidiController(True, config.MIDI_OUT_PORT,
                                  config.MIDI_CHANNEL, config.MIDI_RATE_HZ,
                                  config.MIDI_SEND_NOTES, config.MIDI_CONTROLS)

    grab, resize, track, mapping, send, total = [], [], [], [], [], []
    seen = 0
    tips = list(ALL_TIPS[1:])

    for i in range(FRAMES):
        t0 = time.perf_counter()
        ok, frame = cam.read()
        t1 = time.perf_counter()
        if not ok:
            continue

        frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
                           interpolation=cv2.INTER_AREA)
        t2 = time.perf_counter()

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t3 = time.perf_counter()

        lh, rh = assign_hands(results)
        effects = {}
        if lh and rh:
            seen += 1
            pts = build_polygon_points(lh, rh, tips, tips,
                                       config.DISPLAY_WIDTH,
                                       config.DISPLAY_HEIGHT)
            if pts:
                _, effects, _ = compute_poly_sound(
                    pts, tips, tips,
                    config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
        t4 = time.perf_counter()

        # Force a change every frame so the send path is actually exercised.
        # Timing a suppressed send would be timing nothing.
        effects = dict(effects or {})
        effects["filter"] = (i % 100) / 100.0
        if midi.available:
            midi.send_effects(effects)
        t5 = time.perf_counter()

        grab.append((t1 - t0) * 1e3)
        resize.append((t2 - t1) * 1e3)
        track.append((t3 - t2) * 1e3)
        mapping.append((t4 - t3) * 1e3)
        send.append((t5 - t4) * 1e3)
        total.append((t5 - t0) * 1e3)

    cam.release()
    hands.close()
    midi.close()

    print(f"  hands detected in {seen}/{len(total)} frames\n")
    report("camera read", grab)
    report("resize to display", resize)
    report("MediaPipe hand tracking", track)
    report("geometry -> parameters", mapping)
    report("MIDI encode + send", send)
    print()
    report("STAGE A TOTAL", total)
    if total:
        mean = statistics.mean(total)
        print(f"\n  Mean {mean:.1f} ms per frame -> "
              f"{1000 / max(mean, 1e-6):.0f} fps ceiling from this path alone.")


# =============================================================================
def stage_b() -> None:
    """CC on the wire -> audible change, heard through the microphone."""
    import mido
    import sounddevice as sd

    import midiout

    print("\n=== Stage B: MIDI sent -> audible change in the DAW ===")
    print("    Preconditions:")
    print("      - Fruity Filter on the 3x Osc mixer track, CUT linked to CC 74")
    print("      - FL playing a continuous sound through it")
    print("      - speakers on and audible to the microphone\n")

    ports = midiout.available_ports()
    target = next((p for p in ports
                   if (config.MIDI_OUT_PORT or "").lower() in p.lower()), None)
    if target is None:
        print(f"  no port matching {config.MIDI_OUT_PORT!r}")
        return

    # -- calibrate the acoustic path -----------------------------------------
    print("  calibrating speaker -> air -> microphone ...")
    click = np.zeros(int(SR * 0.30), dtype=np.float32)
    click[:64] = 0.8
    try:
        rec = sd.playrec(np.column_stack([click, click]), SR, channels=1,
                         blocking=True)[:, 0]
    except Exception as exc:
        print(f"  could not run the calibration: {exc}")
        return

    env = np.abs(rec)
    if env.max() < 1e-3:
        print("  the microphone never heard the click. Turn the volume up.")
        return
    air_ms = float(np.argmax(env > env.max() * 0.3)) / SR * 1e3
    print(f"  acoustic round trip: {air_ms:.1f} ms (subtracted below)\n")

    channel = max(0, min(15, config.MIDI_CHANNEL - 1))
    cc = config.MIDI_CONTROLS.get("filter", 74)
    deltas = []

    with mido.open_output(target) as out:
        for trial in range(8):
            out.send(mido.Message("control_change", channel=channel,
                                  control=cc, value=0))
            time.sleep(0.6)

            captured = []

            def cb(indata, frames, t, status):   # noqa: ARG001
                captured.append(indata.copy())

            with sd.InputStream(samplerate=SR, channels=1, blocksize=128,
                                dtype="float32", callback=cb):
                time.sleep(0.30)
                pre = np.concatenate(captured) if captured else np.zeros((1, 1))
                base_rms = float(np.sqrt(np.mean(pre ** 2))) + 1e-9
                mark = len(pre)
                out.send(mido.Message("control_change", channel=channel,
                                      control=cc, value=127))
                time.sleep(0.50)

            if not captured:
                continue
            x = np.concatenate(captured)[:, 0]
            block = 128
            found = None
            for s in range(mark, len(x) - block, block):
                rms = float(np.sqrt(np.mean(x[s:s + block] ** 2)))
                if rms > base_rms * 3.0:
                    found = (s - mark) / SR * 1e3
                    break
            if found is None:
                print(f"    trial {trial + 1}: no change detected")
            else:
                deltas.append(max(0.0, found - air_ms))
                print(f"    trial {trial + 1}: {deltas[-1]:6.1f} ms")

        out.send(mido.Message("control_change", channel=channel,
                              control=cc, value=64))

    print()
    if deltas:
        report("STAGE B (host buffer + smoothing)", deltas)
    else:
        print("  nothing detected. Is CC 74 mapped, and is FL making sound?")


# =============================================================================
if __name__ == "__main__":
    stage_a()
    if "--acoustic" in sys.argv:
        stage_b()
    else:
        print("\n(add --acoustic for stage B, once FL is mapped and playing)")
