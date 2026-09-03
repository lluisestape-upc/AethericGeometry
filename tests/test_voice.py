"""Unit tests for voice.py - dispatch logic, no microphone or model needed."""
import time

import pytest

from aetheric import vocabulary
from aetheric.synth import EFFECT_ORDER
from aetheric.voice import VoiceListener

LANGUAGES = sorted(vocabulary.LANGUAGES)


class _FakeChannel:
    """Stands in for a language channel: no model, no audio."""

    def __init__(self, language, wake=None):
        self.language = language
        self.words = vocabulary.get(language)[0]
        self.wake_word = wake


def _listener(language="en", wake=False, **kw):
    """A listener that never touches audio: no models disables the worker."""
    kw.setdefault("min_confidence", 0.85)
    kw.setdefault("phrase_confidence", 0.45)
    listener = VoiceListener(
        None, enabled=False,
        wake_word=vocabulary.WAKE_LABEL if wake else None, **kw)
    listener._channel = _FakeChannel(
        language, vocabulary.WAKE_TOKEN[language] if wake else None)
    return listener


def _fire(listener, *words, conf=1.0):
    """Feed one final result through the channel under test."""
    listener._dispatch({"text": " ".join(words),
                        "result": [{"word": w, "conf": conf} for w in words]},
                       listener._channel)
    return listener.poll()


# -- Vocabulary ---------------------------------------------------------------
@pytest.mark.parametrize("language", LANGUAGES)
class TestVocabulary:
    @staticmethod
    def _words(language):
        return vocabulary.get(language)[0]

    def test_every_command_has_a_spare_word(self, language):
        """A model missing one word must not take a command with it.

        This is exactly how 'tremolo' was lost: it is absent from both small
        models, and a Vosk grammar silently drops words the model does not know.
        """
        counts: dict[str, int] = {}
        for command in self._words(language).values():
            counts[command] = counts.get(command, 0) + 1
        singletons = sorted(c for c, n in counts.items() if n < 2)
        assert not singletons, f"[{language}] one trigger word for: {singletons}"

    def test_every_effect_is_selectable(self, language):
        commands = set(self._words(language).values())
        for name in EFFECT_ORDER:
            assert f"select:{name}" in commands

    def test_every_waveform_is_selectable(self, language):
        commands = set(self._words(language).values())
        for name in ("SIN", "SAW", "SQUARE", "TRIANGLE"):
            assert f"wave:{name}" in commands

    def test_core_commands_present(self, language):
        commands = set(self._words(language).values())
        assert {"hold", "release", "reset", "cycle:1", "cycle:-1",
                "nudge:1", "nudge:-1"} <= commands

    def test_words_are_lowercase_and_single_tokens(self, language):
        """Vosk emits lowercase single tokens; a multi-word key never matches."""
        for word in self._words(language):
            assert word == word.lower()
            assert " " not in word

    def test_no_accents_in_spanish(self, language):
        """The Spanish lexicon is unaccented - 'mas' is in it, 'mas' with an
        accent is not - so an accented key could never be matched."""
        if language != "es":
            pytest.skip("Spanish only")
        words, fillers = vocabulary.get("es")
        for word in list(words) + list(fillers):
            assert word.isascii(), f"accented key would never match: {word}"

    def test_fillers_are_not_commands(self, language):
        """Overlap would make a piece of sentence glue fire something."""
        words, fillers = vocabulary.get(language)
        assert not (set(words) & set(fillers))

    def test_languages_expose_the_same_commands(self, language):
        """Anything reachable in one language must be reachable in the other,
        or the two channels are not interchangeable."""
        assert (set(vocabulary.get(language)[0].values())
                == set(vocabulary.get("en")[0].values()))

    def test_wake_token_is_not_a_command(self, language):
        words, _ = vocabulary.get(language)
        assert vocabulary.WAKE_TOKEN[language] not in words


# -- Dispatch -----------------------------------------------------------------
class TestDispatch:
    def test_confident_word_fires(self):
        assert _fire(_listener(), "hold", conf=0.99) == ["hold"]

    def test_low_confidence_is_rejected(self):
        assert _fire(_listener(), "hold", conf=0.40) == []

    def test_threshold_is_inclusive(self):
        assert _fire(_listener(), "hold", conf=0.85) == ["hold"]

    def test_unknown_words_are_ignored(self):
        assert _fire(_listener(), "banana", "hold", conf=0.99) == ["hold"]

    def test_cooldown_blocks_immediate_refire(self):
        v = _listener()
        _fire(v, "hold", conf=0.99)
        assert _fire(v, "freeze", conf=0.99) == []   # same command, other word

    def test_cooldown_expires(self):
        v = _listener()
        _fire(v, "hold", conf=0.99)
        v._last_fired["hold"] = time.monotonic() - 5.0
        assert _fire(v, "hold", conf=0.99) == ["hold"]

    def test_different_commands_do_not_block_each_other(self):
        assert sorted(_fire(_listener(), "hold", "reverb", conf=0.99)) == \
            ["hold", "select:reverb"]

    def test_synonyms_map_to_the_same_command(self):
        for word in ("wobble", "shake", "tremolo"):
            assert _fire(_listener(), word, conf=0.99) == ["select:tremolo"]

    def test_result_without_confidences_is_accepted(self):
        """Some builds omit the per-word list; falling back to the plain text
        beats rejecting everything."""
        v = _listener()
        v._dispatch({"text": "hold"}, v._channel)
        assert v.poll() == ["hold"]

    def test_empty_result_is_harmless(self):
        v = _listener()
        v._dispatch({"text": ""}, v._channel)
        v._dispatch({}, v._channel)
        assert v.poll() == []


# -- Both languages, same listener --------------------------------------------
class TestBilingual:
    def test_english_words_fire_on_the_english_channel(self):
        assert _fire(_listener("en"), "hold", conf=0.99) == ["hold"]
        assert _fire(_listener("en"), "square", conf=0.99) == ["wave:SQUARE"]

    def test_spanish_words_fire_on_the_spanish_channel(self):
        assert _fire(_listener("es"), "congela", conf=0.99) == ["hold"]
        assert _fire(_listener("es"), "cuadrada", conf=0.99) == ["wave:SQUARE"]

    def test_a_spanish_word_does_nothing_on_the_english_channel(self):
        """Which is exactly why both models have to run: neither lexicon
        contains the other's words."""
        assert _fire(_listener("en"), "congela", conf=0.99) == []
        assert _fire(_listener("es"), "hold", conf=0.99) == []

    def test_spanish_sentence_fires_only_its_keyword(self):
        v = _listener("es")
        assert _fire(v, "ponme", "una", "onda", "cuadrada",
                     conf=0.95) == ["wave:SQUARE"]

    def test_english_sentence_fires_only_its_keyword(self):
        v = _listener("en")
        assert _fire(v, "give", "me", "a", "square",
                     conf=0.95) == ["wave:SQUARE"]


# -- Triage: short commands vs conversation vs addressed phrases --------------
class TestTriage:
    def test_bare_command_works_without_the_wake_word(self):
        """Barking one word while playing must need no ceremony."""
        assert _fire(_listener("es", wake=True), "congela",
                     conf=0.99) == ["hold"]

    def test_short_phrase_works_without_the_wake_word(self):
        v = _listener("es", wake=True)
        assert sorted(_fire(v, "sube", "el", "eco", conf=0.99)) == \
            ["nudge:+0.0800", "select:reverb"]

    def test_long_phrase_without_wake_word_is_dropped_whole(self):
        v = _listener("es", wake=True)
        assert _fire(v, "hola", "que", "tal", "bien", "y", "tu",
                     conf=0.99) == []

    def test_long_phrase_with_command_words_still_dropped(self):
        """Ordinary conversation contains command words by accident."""
        v = _listener("es", wake=True)
        assert _fire(v, "pues", "mira", "sube", "un", "poco", "mas",
                     conf=0.99) == []

    def test_long_phrase_with_wake_word_is_parsed(self):
        v = _listener("es", wake=True)
        assert _fire(v, "eter", "pillame", "el", "eco", "y", "bajalo",
                     conf=0.95) == ["select:reverb"]

    def test_wake_word_opens_a_window_for_later_long_phrases(self):
        v = _listener("es", wake=True)
        _fire(v, "eter", conf=0.99)
        assert _fire(v, "ahora", "el", "filtro", "por", "favor",
                     conf=0.99) == ["select:filter"]

    def test_window_expires(self):
        v = _listener("es", wake=True)
        _fire(v, "eter", conf=0.99)
        v._awake_until = 0.0
        assert _fire(v, "dame", "mas", "eco", "ahora", "mismo",
                     conf=0.99) == []

    def test_words_before_the_wake_word_are_not_commands(self):
        v = _listener("es", wake=True)
        assert _fire(v, "congela", "eter", "el", "filtro",
                     conf=0.99) == ["select:filter"]

    def test_wake_word_itself_fires_nothing(self):
        assert _fire(_listener("es", wake=True), "eter", conf=0.99) == []

    def test_each_language_uses_its_own_wake_token(self):
        assert _fire(_listener("en", wake=True), "ether", "hold",
                     conf=0.99) == ["hold"]
        assert _fire(_listener("es", wake=True), "eter", "congela",
                     conf=0.99) == ["hold"]

    def test_no_wake_word_configured_means_always_listening(self):
        v = _listener("es", wake=False)
        assert v.awake is True
        assert _fire(v, "dame", "mas", "eco", "ahora", "mismo",
                     conf=0.99) != []


# -- Confidence inside a phrase -----------------------------------------------
class TestPhraseConfidence:
    def test_connected_speech_is_judged_more_leniently(self):
        """The bug this fixes: the wake word was recognised and the sentence
        after it silently dropped, because words inside flowing speech score
        far below an isolated one."""
        v = _listener("es", wake=True, min_confidence=0.85,
                      phrase_confidence=0.45)
        assert _fire(v, "eter", "dame", "el", "eco", conf=0.55) == \
            ["select:reverb"]

    def test_the_wake_word_has_its_own_middle_threshold(self):
        """Not the strict one - it is spoken inside the sentence it introduces,
        so it scores like connected speech - but not the lax one either, or
        room noise would open the window."""
        v = _listener("es", wake=True, min_confidence=0.85,
                      phrase_confidence=0.45, wake_confidence=0.55)
        assert _fire(v, "eter", "dame", "el", "eco", conf=0.30) == []

    def test_a_bare_word_still_needs_high_confidence(self):
        v = _listener("es", wake=True, min_confidence=0.85,
                      phrase_confidence=0.45)
        assert _fire(v, "eco", conf=0.55) == []

    def test_window_also_relaxes_the_threshold(self):
        v = _listener("es", wake=True, min_confidence=0.85,
                      phrase_confidence=0.45)
        _fire(v, "eter", conf=0.99)
        assert _fire(v, "eco", conf=0.55) == ["select:reverb"]


# -- Spoken amounts -----------------------------------------------------------
class TestAmounts:
    def test_bare_direction_uses_the_default_step(self):
        assert _fire(_listener("es", wake=True), "sube",
                     conf=0.99) == ["nudge:+0.0800"]

    def test_percentage_with_a_direction_word(self):
        v = _listener("es", wake=True)
        assert sorted(_fire(v, "eter", "baja", "el", "eco", "treinta", "por",
                            "ciento", conf=0.95)) == \
            ["nudge:-0.3000", "select:reverb"]

    def test_number_is_read_as_a_percentage(self):
        v = _listener("es", wake=True)
        assert _fire(v, "eter", "sube", "cincuenta",
                     conf=0.95) == ["nudge:+0.5000"]

    def test_amount_without_a_direction_changes_nothing(self):
        v = _listener("es", wake=True)
        assert _fire(v, "eter", "el", "eco", "treinta",
                     conf=0.95) == ["select:reverb"]

    def test_english_amounts(self):
        v = _listener("en", wake=True)
        assert sorted(_fire(v, "ether", "turn", "the", "reverb", "down",
                            "thirty", conf=0.95)) == \
            ["nudge:-0.3000", "select:reverb"]


# -- Degradation --------------------------------------------------------------
class TestDegradation:
    def test_disabled_listener_is_inert(self):
        v = VoiceListener(None, enabled=False)
        assert v.available is False
        assert v.start() is False
        assert v.poll() == []
        v.stop()

    def test_no_models_reports_clearly(self):
        v = VoiceListener(None, enabled=True)
        assert v.available is False
        assert "model" in v.status

    def test_missing_model_folder_is_detected(self):
        v = VoiceListener({"en": "/definitely/not/a/model"}, enabled=True)
        assert v.available is False
        assert "model" in v.status

    def test_unknown_language_key_is_ignored(self):
        v = VoiceListener({"klingon": "/nope"}, enabled=True)
        assert v.available is False
