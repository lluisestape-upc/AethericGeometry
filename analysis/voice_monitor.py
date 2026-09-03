"""Show exactly what the speech model hears, word by word, with confidences.

Run from the project root::

    python analysis/voice_monitor.py            # 60 s
    python analysis/voice_monitor.py 30         # 30 s

Why this exists
---------------
"The command does not work" has at least four distinct causes and they need
completely different fixes:

* the word is missing from the model's lexicon, so it can never be emitted;
* the model hears a *different* word from the grammar;
* it hears the right word but below the confidence threshold;
* nothing reaches the recogniser at all, i.e. the microphone.

Guessing between those wastes far more time than measuring. This prints every
final hypothesis with its per-word confidence, marks which words map to a
command, and says whether each would have fired at the configured threshold.

Speak a word, read the line. That is the whole tool.
"""
from __future__ import annotations

import json
import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import vocabulary  # noqa: E402
from voice import _build_recognizer, block_level  # noqa: E402

COMMAND_WORDS, FILLER_WORDS = vocabulary.get(config.VOICE_LANGUAGE)

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0


def main() -> int:
    if not config.VOICE_MODEL_PATH:
        print("voice.model_path is not set in config.yaml")
        return 1
    if not Path(config.VOICE_MODEL_PATH).is_dir():
        print(f"no model directory at {config.VOICE_MODEL_PATH}")
        return 1

    try:
        import sounddevice as sd
        import vosk
    except ImportError:
        print("needs vosk and sounddevice: pip install vosk")
        return 1

    vosk.SetLogLevel(-1)
    model = vosk.Model(config.VOICE_MODEL_PATH)
    recognizer = _build_recognizer(vosk, model, config.VOICE_SAMPLE_RATE,
                                   list(COMMAND_WORDS) + list(FILLER_WORDS))

    threshold = config.VOICE_MIN_CONF
    min_level = config.VOICE_MIN_LEVEL
    print(f"\ndevice     : {sd.query_devices(kind='input')['name']}")
    print(f"confidence : {threshold:.2f}   (voice.min_confidence)")
    print(f"level gate : {min_level:.4f} (voice.min_level)")
    print(f"language   : {config.VOICE_LANGUAGE}")
    print(f"wake word  : "
          f"{(config.VOICE_WAKE_WORD or '').strip().lower() or '(none, always listening)'}")
    print(f"vocabulary : {len(COMMAND_WORDS)} command words -> "
          f"{len(set(COMMAND_WORDS.values()))} commands, "
          f"+{len(FILLER_WORDS)} fillers")
    print(f"\nSpeak. {DURATION:.0f} s. Ctrl-C to stop early.\n")
    print(f"{'heard':<14} {'conf':>6} {'level':>7}  {'verdict':<10} command")
    print("-" * 70)

    blocks: queue.Queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):  # noqa: ARG001
        blocks.put(bytes(indata))

    heard: dict[str, int] = {}
    fired = rejected = quiet = asleep = 0
    room_peak = 0.0
    wake_word = (config.VOICE_WAKE_WORD or "").strip().lower() or None
    wake_timeout = config.VOICE_WAKE_TIMEOUT
    awake_until = 0.0 if wake_word else float("inf")

    with sd.RawInputStream(samplerate=config.VOICE_SAMPLE_RATE, blocksize=4000,
                           device=config.VOICE_DEVICE, dtype="int16",
                           channels=1, callback=on_audio):
        deadline = time.monotonic() + DURATION
        peak = 0.0
        try:
            while time.monotonic() < deadline:
                try:
                    data = blocks.get(timeout=0.2)
                except queue.Empty:
                    continue

                peak = max(peak, block_level(data))
                if not recognizer.AcceptWaveform(data):
                    continue

                result = json.loads(recognizer.Result())
                words = result.get("result") or []
                if not words and result.get("text"):
                    words = [{"word": w, "conf": 1.0}
                             for w in result["text"].split()]

                for entry in words:
                    word = entry.get("word", "")
                    conf = float(entry.get("conf", 1.0))
                    command = COMMAND_WORDS.get(word)
                    heard[word] = heard.get(word, 0) + 1

                    # Mirror the listener's gates in the same order, so the
                    # verdict printed here is what the program would actually
                    # do. A diagnostic that disagrees with the thing it is
                    # diagnosing is worse than none.
                    if wake_word and word == wake_word:
                        if conf >= threshold and peak >= min_level:
                            awake_until = time.monotonic() + wake_timeout
                            verdict, target = "WAKE", f"open {wake_timeout:.0f}s"
                        else:
                            verdict, target = "wake weak", "-"
                    elif command is None:
                        verdict, target = "not a cmd", "-"
                    elif peak < min_level:
                        verdict, target = "too quiet", command
                        quiet += 1
                        room_peak = max(room_peak, peak)
                    elif wake_word and time.monotonic() >= awake_until:
                        verdict, target = "asleep", command
                        asleep += 1
                    elif conf < threshold:
                        verdict, target = "low conf", command
                        rejected += 1
                    else:
                        verdict, target = "FIRES", command
                        fired += 1
                    print(f"{word:<14} {conf:6.2f} {peak:7.4f}  "
                          f"{verdict:<10} {target}")
                peak = 0.0
        except KeyboardInterrupt:
            pass

    print("-" * 70)
    print(f"fired {fired}   asleep {asleep}   below level gate {quiet}   "
          f"low confidence {rejected}")
    if room_peak:
        print(f"loudest rejected utterance: {room_peak:.4f} "
              f"(gate {min_level:.4f})")
    if heard:
        print("\nmost heard:")
        for word, n in sorted(heard.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {word:<14} x{n}")
    if rejected and not fired:
        print(f"\nEverything was recognised but landed under {threshold:.2f}. "
              f"Lower voice.min_confidence in config.yaml.")
    if not heard:
        print("\nNothing was recognised at all — check the input device above "
              "is the microphone you are speaking into.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
