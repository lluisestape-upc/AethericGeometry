"""Spoken vocabularies, one per language.

How this is meant to be used
----------------------------
Not as a command line you recite, but as words dropped into ordinary speech:

    "vamos a probar una onda cuadrada"
    "me gusta, congela"
    "ahora el eco"           ...  "sube"  ...  "un poco menos"

Only the meaningful words are listed as commands. Everything else that a real
sentence is made of goes in :data:`FILLERS`, which is the part that makes the
whole thing work. A Vosk grammar is a closed world: every sound reaching the
decoder is forced onto *some* entry in it. Give it only "cuadrada" and the
seven other words of that sentence still have to land somewhere, so they smear
across the command list and fire things nobody asked for. Listing the glue
gives those syllables an honest home, and ``[unk]`` catches the rest.

Verifying the words
-------------------
Every entry below was checked against the model's lexicon with a probe, because
a Vosk grammar silently drops words the model does not know and the command
then never fires — indistinguishable, from the outside, from a microphone fault.
Two things that discipline caught:

* **The Spanish lexicon is unaccented.** "mas", "triangulo" and "atras" are in
  it; "más", "triángulo" and "atrás" are not. Accented keys would never match.
* **The obvious technical words are missing.** Neither "sinusoidal" nor
  "reverb" exists in the Spanish model, and "tremolo" is absent from both it
  and the English one. The plain words — "seno", "eco", "temblor" — are what
  actually work, which is why they lead each list.
"""
from __future__ import annotations

#: How far a bare "sube"/"baja" moves an effect when no amount is given.
NUDGE_STEP = 0.08

#: Spoken numbers, as percentages. Vosk emits words, never digits, so
#: "bajalo un treinta por ciento" arrives as three tokens and the number has to
#: be looked up rather than parsed.
EN_NUMBERS: dict[str, int] = {
    "zero": 0, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "half": 50,
}

ES_NUMBERS: dict[str, int] = {
    "cero": 0, "cinco": 5, "diez": 10, "quince": 15, "veinte": 20,
    "veinticinco": 25, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
    "cien": 100, "ciento": 100, "mitad": 50,
}


# ══════════════════════════════════════════════════════════════════════════════
#  English
# ══════════════════════════════════════════════════════════════════════════════
EN_WORDS: dict[str, str] = {
    # hold / release
    "hold": "hold",
    "freeze": "hold",
    "keep": "hold",
    "release": "release",
    "drop": "release",
    "clear": "release",
    # effects
    "shape": "select:shape",
    "morph": "select:shape",
    "timbre": "select:shape",
    "filter": "select:filter",
    "cutoff": "select:filter",
    "reverb": "select:reverb",
    "echo": "select:reverb",
    "decay": "select:decay",
    "length": "select:decay",
    "damping": "select:damping",
    "tone": "select:damping",
    "tremolo": "select:tremolo",       # absent from the small model; see above
    "wobble": "select:tremolo",
    "shake": "select:tremolo",
    "drive": "select:drive",
    "dirt": "select:drive",
    # relative moves
    "up": "nudge:1",
    "more": "nudge:1",
    "raise": "nudge:1",
    "down": "nudge:-1",
    "less": "nudge:-1",
    "lower": "nudge:-1",
    # navigation
    "next": "cycle:1",
    "forward": "cycle:1",
    "back": "cycle:-1",
    "previous": "cycle:-1",
    "reset": "reset",
    "clean": "reset",
    # waveforms
    "sine": "wave:SIN",
    "sign": "wave:SIN",                # English "sine" is /saɪn/, a homophone
    "round": "wave:SIN",
    "smooth": "wave:SIN",
    "pure": "wave:SIN",
    "saw": "wave:SAW",
    "sawtooth": "wave:SAW",
    "ramp": "wave:SAW",
    "square": "wave:SQUARE",
    "pulse": "wave:SQUARE",
    "triangle": "wave:TRIANGLE",
    "tri": "wave:TRIANGLE",
}

EN_FILLERS = (
    "a", "an", "the", "give", "me", "put", "on", "in", "it", "this", "that",
    "one", "now", "please", "let", "us", "try", "want", "like", "some", "and",
    "of", "to", "with", "make", "change", "wave", "sound", "effect", "little",
    "bit", "very", "much", "go", "do", "just", "okay", "yes", "no",
    "percent", "by", "set",
)


# ══════════════════════════════════════════════════════════════════════════════
#  Spanish
# ══════════════════════════════════════════════════════════════════════════════
ES_WORDS: dict[str, str] = {
    # hold / release
    "congela": "hold",
    "congelar": "hold",
    "guarda": "hold",
    "fija": "hold",
    "quieto": "hold",
    "suelta": "release",
    "soltar": "release",
    "libera": "release",
    "liberar": "release",
    "quita": "release",
    # effects
    "forma": "select:shape",
    "timbre": "select:shape",
    "textura": "select:shape",
    "filtro": "select:filter",
    "corte": "select:filter",
    "brillo": "select:filter",
    "eco": "select:reverb",            # "reverb" is not in the Spanish lexicon
    "sala": "select:reverb",
    "ambiente": "select:reverb",
    "espacio": "select:reverb",
    "cola": "select:decay",
    "decaimiento": "select:decay",
    "duracion": "select:decay",
    "largo": "select:decay",
    "tono": "select:damping",
    "color": "select:damping",
    "oscuro": "select:damping",
    "temblor": "select:tremolo",       # "tremolo" is not in the lexicon either
    "vibracion": "select:tremolo",
    "tiembla": "select:tremolo",
    "agita": "select:tremolo",
    "suciedad": "select:drive",
    "fuerza": "select:drive",
    "garra": "select:drive",
    "sucio": "select:drive",
    # relative moves
    "sube": "nudge:1",
    "subir": "nudge:1",
    "aumenta": "nudge:1",
    "arriba": "nudge:1",
    "mas": "nudge:1",
    "baja": "nudge:-1",
    "bajar": "nudge:-1",
    "reduce": "nudge:-1",
    "abajo": "nudge:-1",
    "menos": "nudge:-1",
    # navigation
    "siguiente": "cycle:1",
    "adelante": "cycle:1",
    "avanza": "cycle:1",
    "anterior": "cycle:-1",
    "atras": "cycle:-1",
    "reinicia": "reset",
    "limpia": "reset",
    "borra": "reset",
    # waveforms
    "seno": "wave:SIN",                # "sinusoidal" is not in the lexicon
    "curva": "wave:SIN",
    "suave": "wave:SIN",
    "redonda": "wave:SIN",
    "sierra": "wave:SAW",
    "diente": "wave:SAW",
    "dientes": "wave:SAW",
    "rampa": "wave:SAW",
    "cuadrada": "wave:SQUARE",
    "cuadrado": "wave:SQUARE",
    "pulso": "wave:SQUARE",
    "triangular": "wave:TRIANGLE",
    "triangulo": "wave:TRIANGLE",
}

ES_FILLERS = (
    "ponme", "pon", "dame", "quiero", "vamos", "probar", "prueba", "una", "un",
    "el", "la", "onda", "sonido", "ahora", "por", "favor", "me", "gusta",
    "ese", "este", "esta", "esa", "y", "de", "a", "con", "cambia", "cambiar",
    "efecto", "efectos", "muy", "poco", "mucho", "bastante", "ligeramente",
    "dale", "haz", "hazlo", "selecciona", "que", "en", "es", "para", "si",
    "vale", "bien", "por", "ciento", "lo", "al", "del", "pillame", "ponlo",
)


#: The token each model can actually produce for the wake word.
#:
#: The name being spoken is "Aether", but that spelling is in neither lexicon.
#: English has "ether", Spanish has "eter", and the two are pronounced closely
#: enough that one channel or the other catches it however it is said. What the
#: user sees in the UI is :data:`WAKE_LABEL`; what each recogniser listens for
#: is the token below.
WAKE_TOKEN: dict[str, str] = {"en": "ether", "es": "eter"}
WAKE_LABEL = "aether"

LANGUAGES: dict[str, dict] = {
    "en": {"words": EN_WORDS, "fillers": EN_FILLERS, "numbers": EN_NUMBERS},
    "es": {"words": ES_WORDS, "fillers": ES_FILLERS, "numbers": ES_NUMBERS},
}

DEFAULT_LANGUAGE = "es"


def get(language: str) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return ``(words, fillers)`` for a language, falling back to the default.

    Number words are part of the grammar but are not commands, so they are
    appended to the fillers here and looked up separately by :func:`numbers`.
    """
    entry = LANGUAGES.get((language or "").lower()) or LANGUAGES[DEFAULT_LANGUAGE]
    fillers = tuple(entry["fillers"]) + tuple(entry["numbers"])
    return entry["words"], fillers


def numbers(language: str) -> dict[str, int]:
    """Spoken-number lookup for a language."""
    entry = LANGUAGES.get((language or "").lower()) or LANGUAGES[DEFAULT_LANGUAGE]
    return entry["numbers"]
