"""Gestural-Harmonic-Mapping — main event loop."""
import logging
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

# Trigger mediapipe's lazy solution loader immediately after import.
# This must happen before any other custom-module imports; deferring it
# (e.g. inside a function) can cause AttributeError on some mediapipe builds.
try:
    _mp_hands = mp.solutions.hands
    _mp_draw  = mp.solutions.drawing_utils
except AttributeError:
    print(
        "ERROR: mediapipe.solutions is not available.\n"
        "Make sure you are using the project venv:\n"
        "  .\\venv\\Scripts\\activate\n"
        "  python main.py"
    )
    raise SystemExit(1)

from config import (
    PINCH_THRESH, DOUBLE_PINCH_THRESH, PYRAMID_THRESH,
    COOLDOWN_FRAMES,
    MP_MIN_DETECTION, MP_MIN_TRACKING, MP_MAX_HANDS,
    CAMERA_INDEX, AMPLITUDE_HILO, AMPLITUDE_POLYGON,
    DISPLAY_WIDTH, DISPLAY_HEIGHT,
)
from gestures import (
    lm_px, dist, mid, ALL_TIPS,
    THUMB, INDEX, MIDDLE, RING, PINKY,
    assign_hands, build_polygon_points,
)
from renderer import (
    draw_skeleton, draw_hilo, draw_polygon,
    draw_note_label, draw_status_bar, draw_hints, draw_fps,
    draw_cooldown_border, draw_pinch_guides, draw_recording_indicator,
    draw_hold_panel, draw_voice_badge, STATE_COLOR, NEAR_WHITE,
)
from audio_map import compute_poly_sound
from synth import (
    SynthEngine, EFFECT_ORDER, quantize_pentatonic, pitch_from_norm, freq_to_note,
)
import knob as knobmod
from knob import EffectKnobController, KnobPoseDetector, measure_hand
from voice import VoiceListener
from midiout import MidiController
from config import (
    KNOB_TURNS_FULL_RANGE, KNOB_DEAD_ZONE, KNOB_PINCH_CLOSE, KNOB_PINCH_OPEN,
    KNOB_EXTENDED_ABOVE, KNOB_FOLDED_BELOW, KNOB_REQUIRE_EXTENDED,
    VOICE_ENABLED, VOICE_MODELS, VOICE_DEVICE, VOICE_SAMPLE_RATE,
    VOICE_MIN_CONF, VOICE_PHRASE_CONF, VOICE_WAKE_CONF, VOICE_MIN_LEVEL,
    VOICE_WAKE_WORD, VOICE_WAKE_TIMEOUT, VOICE_SHORT_WORDS,
    MIDI_ENABLED, MIDI_OUT_PORT, MIDI_CHANNEL, MIDI_RATE_HZ, MIDI_SEND_NOTES,
    MIDI_CONTROLS,
)

# ── Logging — file, plus console when there is one ────────────────────────────
# A windowed PyInstaller build has no stdout: sys.stdout is None, and handing
# that to StreamHandler crashes on the first log line, before anything has had
# a chance to say why. The file handler is the one that matters in a frozen
# build, and it goes next to the executable rather than the working directory,
# which is wherever the user happened to double-click from.
def _log_path() -> Path:
    base = (Path(sys.executable).parent if getattr(sys, "frozen", False)
            else Path(__file__).parent)
    return base / "aetheric.log"


_handlers: list[logging.Handler] = [
    logging.FileHandler(_log_path(), encoding="utf-8")
]
if sys.stdout is not None:
    _handlers.append(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_handlers,
)
log = logging.getLogger(__name__)

WAVES = ("SIN", "SAW", "SQUARE", "TRIANGLE")

# Landmarks the rotary knob needs; kept small so building the map is cheap.
_KNOB_LANDMARKS = (
    knobmod.WRIST, knobmod.THUMB_TIP, knobmod.INDEX_MCP, knobmod.INDEX_TIP,
    knobmod.MIDDLE_MCP, knobmod.MIDDLE_TIP, knobmod.RING_TIP,
    knobmod.PINKY_MCP, knobmod.PINKY_TIP,
)

# Number keys select an effect directly, and z/x/c/v pick a waveform: keyboard
# mirrors of the voice commands. Everything the voice can do, the keyboard can
# do, so a missing microphone never costs a feature.
_EFFECT_KEYS = {str(i + 1): name for i, name in enumerate(EFFECT_ORDER)}
_WAVE_KEYS = dict(zip("zxcv", WAVES))


def _landmark_map(hand, W: int, H: int) -> dict:
    return {lid: lm_px(hand, lid, W, H) for lid in _KNOB_LANDMARKS}


# ── Initialisation helpers ────────────────────────────────────────────────────
def _init_camera(index: int) -> cv2.VideoCapture:
    cam = cv2.VideoCapture(index)
    if cam.isOpened():
        return cam
    for fallback in range(3):
        if fallback == index:
            continue
        cam = cv2.VideoCapture(fallback)
        if cam.isOpened():
            log.warning("Camera %d unavailable — using camera %d.", index, fallback)
            return cam
    log.error("No camera found. Connect a webcam and restart.")
    sys.exit(1)


def _fit_to_display(frame):
    """Resample the camera frame to the display size *before* anything is drawn.

    This is what keeps the HUD sharp. The camera here delivers 1920x1080 and
    many webcams refuse to negotiate anything else, so a window smaller than
    that used to leave OpenCV rescaling the *finished* frame — HUD text and all
    — with bilinear interpolation. Downsampling rendered text by an arbitrary
    ratio is exactly the operation that makes it look chewed, and no amount of
    better typography survives it.

    Resizing first, with INTER_AREA (a proper area average, the right filter
    for minification), means every glyph is rasterised once at final size and
    never resampled again. It also shrinks the MediaPipe input, which is free
    speed.
    """
    h, w = frame.shape[:2]
    if (w, h) == (DISPLAY_WIDTH, DISPLAY_HEIGHT):
        return frame
    interp = cv2.INTER_AREA if w > DISPLAY_WIDTH else cv2.INTER_LINEAR
    return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=interp)


def _init_mediapipe():
    try:
        return _mp_hands.Hands(
            min_detection_confidence=MP_MIN_DETECTION,
            min_tracking_confidence=MP_MIN_TRACKING,
            max_num_hands=MP_MAX_HANDS,
        )
    except Exception as exc:
        log.error("Failed to initialise MediaPipe Hands: %s", exc)
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("Starting Aetheric Geometry")
    hands_model = _init_mediapipe()
    cam         = _init_camera(CAMERA_INDEX)
    synth       = SynthEngine()
    if not synth.start():
        log.warning("Audio unavailable — running visual-only.")

    # ── Application state ─────────────────────────────────────────────────────
    state        = "IDLE"
    waveform     = "SIN"
    active_l     = [INDEX]
    active_r     = [INDEX]
    prev_dpinch  = False
    pending      = None
    missing      = 0
    current_note = ""
    current_fx: dict = {}

    # FPS counter
    fps_count   = 0
    fps_display = 0
    fps_t       = time.time()

    # UI toggles
    mirrored   = True
    fullscreen = False
    recording  = False
    rec_count  = 0

    # ── Hold + rotary effect knob ─────────────────────────────────────────────
    fx_ctl = EffectKnobController(
        synth, EFFECT_ORDER,
        turns_for_full_range=KNOB_TURNS_FULL_RANGE,
        dead_zone_radians=KNOB_DEAD_ZONE,
    )
    knob_dets = {
        "R": KnobPoseDetector(KNOB_PINCH_CLOSE, KNOB_PINCH_OPEN,
                              KNOB_REQUIRE_EXTENDED, KNOB_EXTENDED_ABOVE,
                              KNOB_FOLDED_BELOW),
        "L": KnobPoseDetector(KNOB_PINCH_CLOSE, KNOB_PINCH_OPEN,
                              KNOB_REQUIRE_EXTENDED, KNOB_EXTENDED_ABOVE,
                              KNOB_FOLDED_BELOW),
    }
    knob_angle = None
    knob_gripping = False

    listener = VoiceListener(VOICE_MODELS, VOICE_SAMPLE_RATE, VOICE_DEVICE,
                             VOICE_ENABLED, VOICE_MIN_CONF, VOICE_PHRASE_CONF,
                             VOICE_WAKE_CONF, VOICE_MIN_LEVEL,
                             VOICE_WAKE_WORD, VOICE_WAKE_TIMEOUT,
                             VOICE_SHORT_WORDS)
    listener.start()

    midi = MidiController(MIDI_ENABLED, MIDI_OUT_PORT, MIDI_CHANNEL,
                          MIDI_RATE_HZ, MIDI_SEND_NOTES, MIDI_CONTROLS)

    def apply_command(cmd: str) -> None:
        """Route a command from either voice or keyboard."""
        nonlocal waveform
        if cmd.startswith("wave:"):
            # Takes effect on the next block whether or not anything is
            # sounding, so a shape can be auditioned through all four shapes
            # without letting go of it.
            waveform = cmd.split(":", 1)[1]
            synth.set_waveform(waveform)
            log.info("Waveform -> %s", waveform)
        elif cmd == "hold":
            if not synth.hold():
                log.info("Nothing sounding to hold.")
        elif cmd == "release":
            synth.release_hold()
            for det in knob_dets.values():
                det.reset()
            fx_ctl.release()
        elif cmd == "reset":
            for name in EFFECT_ORDER:
                synth.set_effect(name, 0.0)
            synth.set_effect("filter", 1.0)
            fx_ctl.select(fx_ctl.selected)
        elif cmd.startswith("select:"):
            fx_ctl.select(cmd.split(":", 1)[1])
        elif cmd.startswith("cycle:"):
            fx_ctl.cycle(int(cmd.split(":", 1)[1]))
        elif cmd.startswith("nudge:"):
            # voice.py resolves the direction and any spoken percentage into a
            # concrete delta before it gets here.
            value = fx_ctl.nudge(float(cmd.split(":", 1)[1]))
            log.info("%s -> %.0f%%", fx_ctl.selected, value * 100)

    # Opened at exactly the render size: any other size makes the window scale
    # the finished frame and undoes the work _fit_to_display just did.
    cv2.namedWindow("Aetheric Geometry", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Aetheric Geometry", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    log.info("Keys: q=quit  f=fullscreen  m=mirror  r=record/stop  "
             "h=hold  g=release  tab=next effect  1-6=pick effect  "
             "z/x/c/v=sine/saw/square/triangle")
    log.info("Voice: %s", listener.status)
    log.info("MIDI:  %s", midi.status)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                log.error("Failed to read camera frame — exiting.")
                break

            frame = _fit_to_display(frame)
            if mirrored:
                frame = cv2.flip(frame, 1)

            H, W    = frame.shape[:2]
            results = hands_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # FPS counter
            fps_count += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps_display = fps_count
                fps_count   = 0
                fps_t       = now

            lh, rh = assign_hands(results)

            # ── Voice commands ────────────────────────────────────────────────
            for cmd in listener.poll():
                apply_command(cmd)

            # ── Held sound: hands drive the effect knob, not the note ─────────
            held = synth.is_held
            knob_gripping = False
            knob_angle = None
            if held:
                for tag, hand in (("R", rh), ("L", lh)):
                    det = knob_dets[tag]
                    if hand is None:
                        det.reset()
                        continue
                    m = measure_hand(_landmark_map(hand, W, H))
                    if det.push(m["pinch"], m["extensions"]):
                        fx_ctl.update(m["angle"], True)
                        knob_gripping = True
                        knob_angle = m["angle"]
                        break
                if not knob_gripping:
                    fx_ctl.release()

            # ── Two-hand gesture + state machine ──────────────────────────────
            poly_pts: list = []

            if lh and rh:
                missing = 0

                pl = lm_px(lh, THUMB, W, H);  il = lm_px(lh, INDEX, W, H)
                pr = lm_px(rh, THUMB, W, H);  ir = lm_px(rh, INDEX, W, H)

                pinch_l = dist(pl, il) < PINCH_THRESH
                pinch_r = dist(pr, ir) < PINCH_THRESH
                d_pinch = pinch_l and pinch_r
                rise    = d_pinch and not prev_dpinch
                fall    = not d_pinch and prev_dpinch
                kiss    = d_pinch and dist(mid(pl, il), mid(pr, ir)) < DOUBLE_PINCH_THRESH
                pyramid = (
                    sum(
                        1 for lid in ALL_TIPS
                        if dist(lm_px(lh, lid, W, H), lm_px(rh, lid, W, H)) < PYRAMID_THRESH
                    ) >= 4
                )

                if state == "IDLE":
                    if rise:
                        state   = "HILO"
                        pending = None
                        log.debug("IDLE → HILO")

                elif state == "HILO":
                    if kiss:
                        state    = "POLIGONO"
                        active_l = [INDEX]
                        active_r = [INDEX]
                        pending  = None
                        log.debug("HILO → POLIGONO (kiss)")
                    elif pyramid:
                        state    = "POLIGONO"
                        active_l = list(ALL_TIPS[1:])
                        active_r = list(ALL_TIPS[1:])
                        pending  = None
                        log.debug("HILO → POLIGONO (pyramid)")
                    elif rise:
                        pending = "RESET"
                    elif fall and pending == "RESET":
                        state    = "IDLE"
                        active_l = [INDEX]
                        active_r = [INDEX]
                        pending  = None
                        log.debug("HILO → IDLE")

                elif state == "POLIGONO":
                    if kiss:
                        state = "HILO"
                        log.debug("POLIGONO → HILO")
                    for lid in (MIDDLE, RING, PINKY):
                        if dist(pl, lm_px(lh, lid, W, H)) < PINCH_THRESH and lid not in active_l:
                            active_l.append(lid)
                        if dist(pr, lm_px(rh, lid, W, H)) < PINCH_THRESH and lid not in active_r:
                            active_r.append(lid)

                prev_dpinch = d_pinch

                if state == "POLIGONO":
                    poly_pts = build_polygon_points(lh, rh, active_l, active_r, W, H)

            elif state != "IDLE":
                missing += 1
                if missing > COOLDOWN_FRAMES:
                    log.debug("Hands lost — reset to IDLE")
                    state       = "IDLE"
                    prev_dpinch = False
                    active_l    = [INDEX]
                    active_r    = [INDEX]
                    synth.send_midi_note(None)

            # ── Sound ─────────────────────────────────────────────────────────
            note_pos = None
            current_freq = None

            if held:
                # The note is frozen and the hands are on the knob, so the
                # gesture-to-pitch path is deliberately dormant.
                pass

            elif state == "IDLE":
                synth.set_params(False, [], waveform)
                current_note = ""
                current_fx   = {}

            elif state == "HILO" and lh and rh:
                il_pos = lm_px(lh, INDEX, W, H)
                ir_pos = lm_px(rh, INDEX, W, H)
                freq   = quantize_pentatonic(
                    pitch_from_norm(dist(il_pos, ir_pos) / W, lo=880.0, hi=110.0)
                )
                synth.set_params(True, [freq], waveform, amplitude=AMPLITUDE_HILO)
                current_freq = freq
                new_note = freq_to_note(freq)
                if new_note != current_note:
                    synth.send_midi_note(freq)
                current_note = new_note
                current_fx   = {}
                note_pos     = mid(il_pos, ir_pos)

            elif state == "POLIGONO" and poly_pts:
                freqs, effects, note_pos = compute_poly_sound(
                    poly_pts, active_l, active_r, W, H
                )
                synth.set_params(True, freqs, waveform,
                                 amplitude=AMPLITUDE_POLYGON, effects=effects)
                current_freq = freqs[0]
                new_note = freq_to_note(freqs[0])
                if new_note != current_note:
                    synth.send_midi_note(freqs[0])
                current_note = new_note
                current_fx   = effects

            # ── MIDI mirror ───────────────────────────────────────────────────
            # Sent every frame, but MidiController drops anything that has not
            # actually changed, so a still hand produces no traffic.
            if midi.available:
                midi.send_effects(synth.snapshot_effects())
                if held:
                    pass                      # frozen note keeps sounding
                elif state == "IDLE":
                    midi.send_pitch(None)
                elif current_freq:
                    midi.send_pitch(current_freq)

            # ── Render ────────────────────────────────────────────────────────
            draw_skeleton(frame, results)
            col = STATE_COLOR.get(state, NEAR_WHITE)

            if state == "IDLE":
                h = lh or rh
                if h:
                    draw_pinch_guides(frame, h, W, H)

            if state == "HILO" and lh and rh:
                draw_hilo(frame, lh, rh, W, H, col)
            elif state == "POLIGONO":
                draw_polygon(frame, poly_pts, col)

            if note_pos:
                draw_note_label(frame, current_note, note_pos[0], note_pos[1], col)

            if state != "IDLE" and missing > 0:
                draw_cooldown_border(frame, missing, W, H)

            draw_status_bar(frame, state, waveform, len(poly_pts),
                            current_note, current_fx, W, H)
            draw_hints(frame, state, recording)
            draw_fps(frame, fps_display, W)
            if listener.available:
                draw_voice_badge(frame, listener.wake_word, listener.awake, W, H)

            if held:
                draw_hold_panel(frame, synth.snapshot_effects(), EFFECT_ORDER,
                                fx_ctl.selected, knob_gripping, knob_angle, W, H)

            if recording:
                draw_recording_indicator(frame, W)

            cv2.imshow("Aetheric Geometry", frame)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty("Aetheric Geometry", cv2.WND_PROP_FULLSCREEN, prop)
                log.info("Fullscreen %s", "on" if fullscreen else "off")
            elif key == ord("m"):
                mirrored = not mirrored
                log.info("Mirror %s", "on" if mirrored else "off")
            elif key == ord("r"):
                if not recording:
                    synth.start_recording()
                    recording = True
                else:
                    rec_count += 1
                    path = synth.stop_recording(f"recording_{rec_count:03d}.wav")
                    if path:
                        log.info("Saved: %s", path)
                    recording = False
            elif key == ord("h"):
                apply_command("hold")
            elif key == ord("g"):
                apply_command("release")
            elif key == 9:  # Tab
                apply_command("cycle:1")
            elif 32 <= key < 127 and chr(key) in _EFFECT_KEYS:
                apply_command(f"select:{_EFFECT_KEYS[chr(key)]}")
            elif 32 <= key < 127 and chr(key) in _WAVE_KEYS:
                apply_command(f"wave:{_WAVE_KEYS[chr(key)]}")

    finally:
        listener.stop()
        midi.close()
        if recording:
            synth.stop_recording(f"recording_{rec_count + 1:03d}.wav")
        synth.send_midi_note(None)
        synth.stop()
        cam.release()
        hands_model.close()
        cv2.destroyAllWindows()
        log.info("Shutdown complete")


if __name__ == "__main__":
    main()
