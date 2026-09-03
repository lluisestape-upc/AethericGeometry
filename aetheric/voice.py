"""Offline voice commands, English and Spanish at the same time.

Recognition runs on its own thread and communicates with the video loop through
a queue, so a slow decode can never stall a video frame or an audio block.

Why two recognisers
-------------------
An acoustic model is language-specific. "hold" is not in the Spanish model's
lexicon and "congela" is not in the English one, and a Vosk grammar can only
contain words its model already knows. There is therefore no single-model way
to hear both languages: the only honest option is to decode the same audio
twice, once per model, and merge the results. Both run on one microphone stream
and share the wake window and the refire cooldown, so from the outside it
behaves as one listener that happens to be bilingual.

Design notes
------------
**Constrained grammar, including the glue.** A closed grammar is *closed*:
every sound reaching it lands on some entry. The vocabulary therefore also
lists the ordinary words sentences are made of, so "ponme una onda cuadrada"
has somewhere to put the words that are not commands instead of smearing them
across the ones that are. See :mod:`vocabulary`.

**Keyword spotting, not parsing.** A result is scanned word by word and
anything that maps to a command fires. "vamos a probar una onda cuadrada" works
because "cuadrada" is in it, not because the sentence was understood.

**Two confidence thresholds.** Words spoken in isolation score high; the same
words inside a flowing sentence score far lower, because the decoder is
splitting a continuous stream rather than matching one token. A single strict
threshold therefore accepts "eco" on its own and silently rejects the identical
word inside "eter, pillame el eco y bajalo" - which reads as the wake word
working and the sentence being ignored. After a wake word the speaker has
already declared intent, so the threshold drops to
:data:`_PHRASE_CONFIDENCE`.

**Length triage.** Requiring a wake word for everything makes the instrument
unusable; requiring nothing lets room conversation play it. So short utterances
work bare, long ones need the wake word, and long ones without it are dropped
whole.

The instrument never depends on this module: every command has a keyboard
equivalent, and with no model the class reports ``available == False`` and
quietly does nothing.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from . import vocabulary

log = logging.getLogger(__name__)

#: Minimum gap between two firings of the same command.
_REFIRE_COOLDOWN_S = 1.2

#: Confidence a word must reach when spoken on its own, with no wake word.
_MIN_CONFIDENCE = 0.85

#: Relaxed threshold for words inside an addressed phrase. See the module notes.
_PHRASE_CONFIDENCE = 0.45

#: Confidence the wake word itself must reach.
#:
#: Deliberately between the other two. Holding the wake word to the strict
#: isolated-speech threshold looks correct and is not: the wake word is spoken
#: *inside* the very sentence it introduces - "aether, pillame el eco" is one
#: breath - so it scores like connected speech, fails a 0.85 gate, never opens
#: the window, and the sentence behind it is dropped for having no wake word.
#: That failure mode is invisible from outside: the name appears to be heard
#: and nothing happens. Dropping it all the way to the phrase threshold would
#: instead let room noise open the window, which is the thing the wake word
#: exists to prevent. The token is distinctive enough that the middle is safe.
_WAKE_CONFIDENCE = 0.55

#: Peak RMS an utterance must reach before it is considered speech at all.
#:
#: Confidence alone does not settle it. A constrained grammar has to put every
#: sound *somewhere*, and it will happily assign room tone to a real word at
#: high confidence - measured: 45 s of silence produced one "smooth" at 0.87.
#: Confidence answers "which word is this most like", not "was anyone
#: speaking". Speech into a laptop microphone runs an order of magnitude above
#: room tone, so level answers the second question cleanly.
_MIN_LEVEL = 0.012

#: Seconds a wake word keeps the microphone open for commands.
_WAKE_TIMEOUT_S = 8.0

#: Longest utterance still treated as a bare command with no wake word.
_SHORT_UTTERANCE_WORDS = 3


def _family(command: str) -> str:
    """The kind of intent a command expresses.

    Words within a family are alternative ways of saying the same *kind* of
    thing - "eco" and "sala" both select an effect - so at most one of them can
    have been meant. Families are independent: selecting an effect and moving
    it are two intents that can share one sentence.
    """
    return command.split(":", 1)[0]


def block_level(data: bytes) -> float:
    """RMS of a 16-bit mono audio block, normalised to 0..1."""
    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    x = samples.astype(np.float32) * (1.0 / 32768.0)
    return float(np.sqrt(np.mean(x * x)))


def _build_recognizer(vosk, model, sample_rate: int, words: list[str]):
    """Build a constrained-grammar recogniser.

    A Vosk grammar may only contain words that exist in the model's lexicon.
    Anything else is dropped with a warning printed by the native library, and
    the command then simply never fires - which looks exactly like a microphone
    or pronunciation fault and is neither.

    That warning cannot be intercepted from Python on Windows: the Vosk DLL
    links its own C runtime, so its stderr is not the interpreter's file
    descriptor 2. Rather than ship a diagnostic that reports "all clear" on the
    one platform where it does not work, the warning is left on the console and
    the real defence lives in :mod:`vocabulary`, where every command carries
    more than one trigger word.
    """
    grammar = json.dumps(sorted(set(words)) + ["[unk]"])
    recognizer = vosk.KaldiRecognizer(model, sample_rate, grammar)
    recognizer.SetWords(True)   # per-word confidences in final results
    return recognizer


class _Channel:
    """One language: its model, its grammar, its recogniser."""

    __slots__ = ("language", "words", "wake_word", "recognizer")

    def __init__(self, language: str, words: dict, wake_word: str | None,
                 recognizer):
        self.language = language
        self.words = words
        self.wake_word = wake_word
        self.recognizer = recognizer


class VoiceListener:
    """Background offline recogniser exposing a non-blocking command queue."""

    def __init__(self, models: dict[str, str] | None = None,
                 sample_rate: int = 16000, device=None, enabled: bool = True,
                 min_confidence: float = _MIN_CONFIDENCE,
                 phrase_confidence: float = _PHRASE_CONFIDENCE,
                 wake_confidence: float = _WAKE_CONFIDENCE,
                 min_level: float = _MIN_LEVEL,
                 wake_word: str | None = vocabulary.WAKE_LABEL,
                 wake_timeout: float = _WAKE_TIMEOUT_S,
                 short_utterance_words: int = _SHORT_UTTERANCE_WORDS):
        self._models = dict(models or {})
        self._sample_rate = int(sample_rate)
        self._device = device
        self._enabled = bool(enabled)
        self._min_confidence = float(min_confidence)
        self._phrase_confidence = float(phrase_confidence)
        self._wake_confidence = float(wake_confidence)
        self._min_level = float(min_level)
        self._wake_label = (wake_word or "").strip().lower() or None
        self._wake_enabled = self._wake_label is not None
        self._wake_timeout = float(wake_timeout)
        self._short_words = max(1, int(short_utterance_words))

        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_fired: dict[str, float] = {}
        self._awake_until = 0.0
        self._channels: list[_Channel] = []
        self.languages: tuple[str, ...] = ()
        self.available = False
        self.status = "disabled"

        if not self._enabled:
            return

        usable = {lang: path for lang, path in self._models.items()
                  if path and Path(path).is_dir()
                  and lang in vocabulary.LANGUAGES}
        missing = sorted(set(self._models) - set(usable))
        if missing:
            log.warning("Voice: no model directory for %s - that language will "
                        "not be recognised.", ", ".join(missing))
        if not usable:
            self.status = "no usable model"
            log.info("Voice commands off - no model found. "
                     "Keyboard shortcuts still work.")
            return

        try:
            import vosk  # noqa: F401
            import sounddevice  # noqa: F401
        except ImportError:
            self.status = "vosk not installed"
            log.info("Voice commands off - pip install vosk. "
                     "Keyboard shortcuts still work.")
            return

        self._models = usable
        self.languages = tuple(sorted(usable))
        self.available = True
        self.status = "ready"

    # -- Lifecycle -----------------------------------------------------------
    def start(self) -> bool:
        if not self.available:
            return False
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    @property
    def awake(self) -> bool:
        """True while a wake window is open, or when none is required."""
        if not self._wake_enabled:
            return True
        return time.monotonic() < self._awake_until

    @property
    def wake_word(self) -> str | None:
        """The wake word to show in the UI, or None when always listening."""
        return self._wake_label

    def poll(self) -> list[str]:
        """Drain the queue. Never blocks."""
        out: list[str] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    # -- Worker --------------------------------------------------------------
    def _run(self) -> None:
        try:
            import sounddevice as sd
            import vosk
        except ImportError:  # pragma: no cover - guarded in __init__
            return

        try:
            for language, path in sorted(self._models.items()):
                words, fillers = vocabulary.get(language)
                wake = self._wake_for(language)
                grammar = list(words) + list(fillers) + ([wake] if wake else [])
                model = vosk.Model(path)
                self._channels.append(_Channel(
                    language, words, wake,
                    _build_recognizer(vosk, model, self._sample_rate, grammar)))
        except Exception as exc:
            self.status = f"model load failed: {exc}"
            log.warning("Voice model failed to load (%s) - keyboard only.", exc)
            self.available = False
            return

        blocks: queue.Queue = queue.Queue()

        def on_audio(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                log.debug("Voice input status: %s", status)
            blocks.put(bytes(indata))

        try:
            with sd.RawInputStream(samplerate=self._sample_rate, blocksize=4000,
                                   device=self._device, dtype="int16",
                                   channels=1, callback=on_audio):
                self.status = "listening"
                log.info("Voice ready [%s]%s",
                         ", ".join(c.language for c in self._channels),
                         f", wake word '{self._wake_label}'"
                         if self._wake_enabled else ", always listening")

                # One peak accumulator *per channel*. The two recognisers close
                # an utterance at different moments, so a shared accumulator
                # gets reset by whichever finishes first and the other channel
                # then judges its own utterance on the level of whatever
                # silence happened to follow - and drops it as too quiet.
                peaks = {c.language: 0.0 for c in self._channels}

                while not self._stop.is_set():
                    try:
                        data = blocks.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    level = block_level(data)
                    for channel in self._channels:
                        lang = channel.language
                        peaks[lang] = max(peaks[lang], level)
                        if not channel.recognizer.AcceptWaveform(data):
                            continue
                        if peaks[lang] >= self._min_level:
                            self._dispatch(
                                json.loads(channel.recognizer.Result()), channel)
                        else:
                            log.debug("[%s] ignoring utterance - peak %.4f "
                                      "below %.4f", lang, peaks[lang],
                                      self._min_level)
                        peaks[lang] = 0.0
        except Exception as exc:
            self.status = f"audio input failed: {exc}"
            log.warning("Voice input failed (%s) - keyboard only.", exc)
            self.available = False

    def _wake_for(self, language: str) -> str | None:
        """The token this language's model can actually produce for the wake word."""
        if not self._wake_enabled:
            return None
        return vocabulary.WAKE_TOKEN.get(language)

    # -- Dispatch ------------------------------------------------------------
    def _dispatch(self, result: dict, channel: "_Channel | None" = None) -> None:
        """Queue commands from one final recognition result."""
        words = result.get("result")
        if not words:
            text = result.get("text", "")
            if not text:
                return
            words = [{"word": w, "conf": 1.0} for w in text.split()]

        lookup = channel.words if channel else vocabulary.get("en")[0]
        wake_word = (channel.wake_word if channel
                     else (vocabulary.WAKE_TOKEN["en"] if self._wake_enabled
                           else None))

        now = time.monotonic()
        tokens = [e.get("word", "") for e in words]

        # -- triage: is this utterance addressed to the instrument? ----------
        wake_at = -1
        if wake_word:
            for i, (token, entry) in enumerate(zip(tokens, words)):
                if (token == wake_word
                        and float(entry.get("conf", 1.0)) >= self._wake_confidence):
                    wake_at = i
                    break

        addressed = False
        if wake_at >= 0:
            self._awake_until = now + self._wake_timeout
            addressed = True
            log.info("Listening (%.0f s)", self._wake_timeout)
        elif self._wake_enabled and now < self._awake_until:
            addressed = True
        elif self._wake_enabled:
            if len(tokens) > self._short_words:
                log.debug("Ignoring %d-word utterance with no wake word: %s",
                          len(tokens), " ".join(tokens))
                return

        # Inside an addressed phrase the speaker has already declared intent,
        # and connected speech scores much lower per word than an isolated one.
        threshold = self._phrase_confidence if addressed else self._min_confidence

        considered = words[wake_at + 1:] if wake_at >= 0 else words
        amount = self._amount(tokens, channel)

        # -- one intent per family -------------------------------------------
        # A sentence produces many matches, especially at the relaxed
        # in-phrase threshold, and firing all of them is incoherent: a single
        # utterance was measured emitting wave:SIN, release, hold,
        # select:reverb, wave:SQUARE and wave:TRIANGLE within four
        # milliseconds. The last one wins and the instrument looks like it
        # ignored the request.
        #
        # A sentence can legitimately mean two things at once - "el eco, sube"
        # is a selection *and* a move - so candidates are grouped by family and
        # the best-scoring one in each family wins. Within a family the words
        # are alternatives, never a sequence.
        best: dict[str, tuple[float, str, str]] = {}
        for entry in considered:
            word = entry.get("word", "")
            conf = float(entry.get("conf", 1.0))
            command = lookup.get(word)
            if command is None:
                continue
            if conf < threshold:
                log.debug("Ignoring '%s' - confidence %.2f below %.2f",
                          word, conf, threshold)
                continue
            family = _family(command)
            if conf > best.get(family, (-1.0,))[0]:
                best[family] = (conf, command, word)

        # hold and release are opposites; hearing both means one is a mishear.
        if "hold" in best and "release" in best:
            loser = "release" if best["hold"][0] >= best["release"][0] else "hold"
            log.debug("Dropping '%s' - heard against its opposite in one breath",
                      best[loser][2])
            del best[loser]

        for _, command, word in sorted(best.values(), reverse=True):
            conf = best[_family(command)][0]
            command = self._apply_amount(command, amount)
            if now - self._last_fired.get(command, 0.0) < _REFIRE_COOLDOWN_S:
                continue
            self._last_fired[command] = now
            self._queue.put(command)
            log.info("Voice [%s]: %s (heard '%s', conf %.2f)",
                     channel.language if channel else "-", command, word, conf)

    def _amount(self, tokens: list[str], channel=None) -> float | None:
        """First spoken number in the utterance, as a 0..1 fraction."""
        numbers = vocabulary.numbers(channel.language if channel else "en")
        for token in tokens:
            if token in numbers:
                return numbers[token] / 100.0
        return None

    def _apply_amount(self, command: str, amount: float | None) -> str:
        """Turn a bare direction into a concrete move.

        ``nudge:1`` and ``nudge:-1`` are directions in the vocabulary; here they
        become an actual delta, using a spoken percentage when the sentence
        carried one ("baja treinta por ciento") and the default step when it did
        not ("sube").
        """
        if not command.startswith("nudge:"):
            return command
        sign = 1.0 if command == "nudge:1" else -1.0
        step = amount if amount is not None else vocabulary.NUDGE_STEP
        return f"nudge:{sign * step:+.4f}"
