"""Loads config.yaml and exposes flat constants.

Falls back to built-in defaults if the file or pyyaml is missing, so the program
always starts.
"""
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

#: Where bundled resources live: the PyInstaller extraction root when frozen,
#: the source folder otherwise.
_HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))

#: Where the *user's* editable files live: beside the executable when frozen.
#: Shipping config.yaml read-only inside the bundle would mean a rebuild to
#: change a camera index or name a loopMIDI port, so a copy next to the exe
#: wins over the bundled one when present.
_USER_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else _HERE

_DEFAULTS: dict = {
    "thresholds": {
        "pinch": 60, "double_pinch": 90, "pyramid": 70,
        "prayer_dist": 160, "open_palm_spread": 100,
    },
    "timing": {"cooldown_frames": 45},
    "colors": {
        "neon_green": [70, 230, 90], "neon_cyan": [240, 210, 40],
        "neon_pink": [215, 65, 230], "text_dim": [145, 135, 165],
        "near_white": [230, 225, 242], "warm_red": [40, 40, 220],
    },
    "audio": {
        "sample_rate": 44100, "block_size": 512,
        "amplitude_hilo": 0.28, "amplitude_polygon": 0.22,
        "attack_tau": 0.0028, "release_tau": 0.0113,
        "reverb_tail_tau": 1.2,
        "reverb_rt60_min": 0.35, "reverb_rt60_max": 6.0,
        "filter_min_hz": 200.0, "filter_max_hz": 8000.0,
        "drive_max": 12.0,
        "midi_port": None,
    },
    "knob": {
        "turns_for_full_range": 0.5, "dead_zone_radians": 0.012,
        "pinch_close_below": 0.60, "pinch_open_above": 0.90,
        "extended_above": 1.60, "folded_below": 1.30,
        "require_extended": 2,
    },
    "voice": {
        "enabled": True,
        "models": {
            "en": "models/vosk-model-small-en-us-0.15",
            "es": "models/vosk-model-small-es-0.42",
        },
        "device": None,
        "sample_rate": 16000,
        "min_confidence": 0.85,
        "phrase_confidence": 0.45,
        "wake_confidence": 0.55,
        "min_level": 0.012,
        "wake_word": "aether",
        "wake_timeout": 8.0,
        "short_utterance_words": 3,
    },
    "midi": {
        "enabled": False,
        "port": None,
        "channel": 1,
        "rate_hz": 50,
        "send_notes": True,
        "controls": {
            "filter": 74, "reverb": 91, "decay": 93, "damping": 71,
            "tremolo": 92, "drive": 75,
        },
    },
    "mediapipe": {
        "min_detection_confidence": 0.65,
        "min_tracking_confidence": 0.65,
        "max_num_hands": 2,
    },
    "camera": {"index": 0},
    "display": {"width": 1280, "height": 720},
}


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    result = {}
    for key, default_val in defaults.items():
        if key in overrides:
            if isinstance(default_val, dict) and isinstance(overrides[key], dict):
                result[key] = _deep_merge(default_val, overrides[key])
            else:
                result[key] = overrides[key]
        else:
            result[key] = default_val
    return result


def _config_path() -> Path:
    """User copy beside the executable if there is one, else the bundled one."""
    beside = _USER_DIR / "config.yaml"
    return beside if beside.is_file() else _HERE / "config.yaml"


def _load() -> dict:
    cfg_path = _config_path()
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        log.debug("Loaded config from %s", cfg_path)
        return _deep_merge(_DEFAULTS, loaded)
    except ImportError:
        log.warning("pyyaml not installed - using defaults. Run: pip install pyyaml")
    except FileNotFoundError:
        log.warning("config.yaml not found - using defaults.")
    except Exception as exc:
        log.warning("Failed to load config.yaml (%s) - using defaults.", exc)
    return _DEFAULTS


_cfg = _load()


def _color(key: str) -> tuple:
    return tuple(_cfg["colors"][key])


def _resolve(raw):
    """Resolve a relative path against the project, not the shell's cwd.

    Vosk resolves a plain path against the working directory, so a relative
    entry in config.yaml would only work when launched from the project folder.
    Anchoring it here means the setting travels with the repo - and with the
    frozen executable, where the working directory is wherever the user
    double-clicked from.
    """
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _HERE / path
    return str(path)


# -- Thresholds ---------------------------------------------------------------
PINCH_THRESH        = float(_cfg["thresholds"]["pinch"])
DOUBLE_PINCH_THRESH = float(_cfg["thresholds"]["double_pinch"])
PYRAMID_THRESH      = float(_cfg["thresholds"]["pyramid"])
PRAYER_DIST         = float(_cfg["thresholds"]["prayer_dist"])
OPEN_PALM_SPREAD    = float(_cfg["thresholds"]["open_palm_spread"])

# -- Timing -------------------------------------------------------------------
COOLDOWN_FRAMES = int(_cfg["timing"]["cooldown_frames"])

# -- Colours ------------------------------------------------------------------
# HUD colours live in ui.py, which speaks RGB for Pillow. What remains here is
# the gesture-layer palette that OpenCV draws in BGR.
NEON_GREEN = _color("neon_green")
NEON_CYAN  = _color("neon_cyan")
NEON_PINK  = _color("neon_pink")
TEXT_DIM   = _color("text_dim")
NEAR_WHITE = _color("near_white")
WARM_RED   = _color("warm_red")

# -- Audio --------------------------------------------------------------------
SAMPLE_RATE       = int(_cfg["audio"]["sample_rate"])
BLOCK_SIZE        = int(_cfg["audio"]["block_size"])
AMPLITUDE_HILO    = float(_cfg["audio"]["amplitude_hilo"])
AMPLITUDE_POLYGON = float(_cfg["audio"]["amplitude_polygon"])
ATTACK_TAU        = float(_cfg["audio"]["attack_tau"])
RELEASE_TAU       = float(_cfg["audio"]["release_tau"])
REVERB_TAIL_TAU   = float(_cfg["audio"]["reverb_tail_tau"])
REVERB_RT60_MIN   = float(_cfg["audio"]["reverb_rt60_min"])
REVERB_RT60_MAX   = float(_cfg["audio"]["reverb_rt60_max"])
FILTER_MIN_HZ     = float(_cfg["audio"]["filter_min_hz"])
FILTER_MAX_HZ     = float(_cfg["audio"]["filter_max_hz"])
DRIVE_MAX         = float(_cfg["audio"]["drive_max"])
MIDI_PORT         = _cfg["audio"]["midi_port"]

# -- Rotary hand knob ---------------------------------------------------------
KNOB_TURNS_FULL_RANGE = float(_cfg["knob"]["turns_for_full_range"])
KNOB_DEAD_ZONE        = float(_cfg["knob"]["dead_zone_radians"])
KNOB_PINCH_CLOSE      = float(_cfg["knob"]["pinch_close_below"])
KNOB_PINCH_OPEN       = float(_cfg["knob"]["pinch_open_above"])
KNOB_EXTENDED_ABOVE   = float(_cfg["knob"]["extended_above"])
KNOB_FOLDED_BELOW     = float(_cfg["knob"]["folded_below"])
KNOB_REQUIRE_EXTENDED = int(_cfg["knob"]["require_extended"])

# -- Voice --------------------------------------------------------------------
VOICE_ENABLED      = bool(_cfg["voice"]["enabled"])
VOICE_MODELS       = {lang: _resolve(path)
                      for lang, path in (_cfg["voice"]["models"] or {}).items()
                      if path}
VOICE_DEVICE       = _cfg["voice"]["device"]
VOICE_SAMPLE_RATE  = int(_cfg["voice"]["sample_rate"])
VOICE_MIN_CONF     = float(_cfg["voice"]["min_confidence"])
VOICE_PHRASE_CONF  = float(_cfg["voice"]["phrase_confidence"])
VOICE_WAKE_CONF    = float(_cfg["voice"]["wake_confidence"])
VOICE_MIN_LEVEL    = float(_cfg["voice"]["min_level"])
VOICE_WAKE_WORD    = _cfg["voice"]["wake_word"]
VOICE_WAKE_TIMEOUT = float(_cfg["voice"]["wake_timeout"])
VOICE_SHORT_WORDS  = int(_cfg["voice"]["short_utterance_words"])

# -- MIDI out -----------------------------------------------------------------
MIDI_ENABLED    = bool(_cfg["midi"]["enabled"])
MIDI_OUT_PORT   = _cfg["midi"]["port"]
MIDI_CHANNEL    = int(_cfg["midi"]["channel"])
MIDI_RATE_HZ    = float(_cfg["midi"]["rate_hz"])
MIDI_SEND_NOTES = bool(_cfg["midi"]["send_notes"])
MIDI_CONTROLS   = {str(k): int(v) for k, v in (_cfg["midi"]["controls"] or {}).items()}

# -- MediaPipe ----------------------------------------------------------------
MP_MIN_DETECTION = float(_cfg["mediapipe"]["min_detection_confidence"])
MP_MIN_TRACKING  = float(_cfg["mediapipe"]["min_tracking_confidence"])
MP_MAX_HANDS     = int(_cfg["mediapipe"]["max_num_hands"])

# -- Camera and display -------------------------------------------------------
CAMERA_INDEX = int(_cfg["camera"]["index"])

# The camera frame is resampled to this size before anything is drawn, so the
# HUD is rasterised once at final resolution and never rescaled afterwards.
DISPLAY_WIDTH  = int(_cfg["display"]["width"])
DISPLAY_HEIGHT = int(_cfg["display"]["height"])
