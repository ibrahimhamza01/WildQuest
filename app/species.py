"""Species names for display, keyed by the frozen class IDs."""

import json
from dataclasses import dataclass
from pathlib import Path

MAPPING_PATH = Path(__file__).resolve().parents[1] / "Code" / "configs" / "class_mapping.json"

# Display names and animal groups, keyed by scientific name (class_mapping.json labels).
_DISPLAY = {
    "Anas platyrhynchos": ("Mallard", "Bird"),
    "Ardea herodias": ("Great Blue Heron", "Bird"),
    "Branta canadensis": ("Canada Goose", "Bird"),
    "Buteo jamaicensis": ("Red-tailed Hawk", "Bird"),
    "Cardinalis cardinalis": ("Northern Cardinal", "Bird"),
    "Chelydra serpentina": ("Common Snapping Turtle", "Reptile"),
    "Chrysemys picta": ("Painted Turtle", "Reptile"),
    "Corvus brachyrhynchos": ("American Crow", "Bird"),
    "Cyanocitta cristata": ("Blue Jay", "Bird"),
    "Didelphis virginiana": ("Virginia Opossum", "Mammal"),
    "Lithobates catesbeianus": ("American Bullfrog", "Amphibian"),
    "Lithobates clamitans": ("Green Frog", "Amphibian"),
    "Marmota monax": ("Groundhog", "Mammal"),
    "Odocoileus virginianus": ("White-tailed Deer", "Mammal"),
    "Pantherophis alleghaniensis": ("Eastern Rat Snake", "Reptile"),
    "Passer domesticus": ("House Sparrow", "Bird"),
    "Plestiodon fasciatus": ("Five-lined Skink", "Reptile"),
    "Procyon lotor": ("Raccoon", "Mammal"),
    "Sciurus carolinensis": ("Eastern Gray Squirrel", "Mammal"),
    "Sturnus vulgaris": ("European Starling", "Bird"),
    "Sylvilagus floridanus": ("Eastern Cottontail", "Mammal"),
    "Tamias striatus": ("Eastern Chipmunk", "Mammal"),
    "Turdus migratorius": ("American Robin", "Bird"),
    "Vulpes vulpes": ("Red Fox", "Mammal"),
    "Zenaida macroura": ("Mourning Dove", "Bird"),
}


@dataclass(frozen=True)
class Species:
    class_id: int
    scientific_name: str
    common_name: str
    group: str


def load_species(mapping_path=MAPPING_PATH):
    """Return the 25 species in class-ID order."""
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    id_to_label = mapping["id_to_label"]
    species = []
    for class_id in range(len(id_to_label)):
        scientific = id_to_label[str(class_id)]
        common, group = _DISPLAY.get(scientific, (scientific, "Unknown"))
        species.append(Species(class_id, scientific, common, group))
    return tuple(species)


SPECIES = load_species()