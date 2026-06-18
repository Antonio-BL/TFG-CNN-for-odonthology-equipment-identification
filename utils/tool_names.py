"""Mapping from internal class folder names (HerramientaN) to short Spanish
display names for the pipeline visualisations and console output.

Shortened on purpose: the label is drawn at the centre of each rotated bbox on
the tray panel, so multi-word names overlap. Kept under ~13 characters where
possible.
"""

TOOL_NAMES: dict[str, str] = {
    "Herramienta0": "Botador",
    "Herramienta1": "Forceps sup.",
    "Herramienta2": "Cureta",
    "Herramienta3": "Espejo",
    "Herramienta4": "Pinzas",
    "Herramienta5": "Sonda",
    "Herramienta6": "Periostotomo",
    "Herramienta7": "Bisturi",
    "Herramienta8": "Porta agujas",
    "Herramienta9": "Tijeras",
}


def display_name(class_name: str) -> str:
    """Return the short human name for a class folder name, or the input unchanged."""
    return TOOL_NAMES.get(class_name, class_name)
