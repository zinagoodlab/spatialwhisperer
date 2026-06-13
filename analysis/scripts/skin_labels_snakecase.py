"""Original-MUSK snake_case class strings for the Kriegsmann skin benchmark.

Pre-relabel baseline: each class string is the raw `class` column value from
`tiles-v2.csv` (path-component concatenation like
`nontumor_skin_chondraltissue_chondraltissue`). Used to test whether the prior
MUSK-harness rerun (which reported 0.922 with-bridge fraction=1) was actually
exercising the relabeled clinical strings or whether a stale-cache snake_case
result leaked through — the original concern in the source TODO.

Same module API as skin_labels.py: RAW_TO_CLINICAL maps raw → label-to-use;
here the "clinical" target is the snake_case raw string itself, so the prompts
fed to the text encoder are the snake_case names.
"""

_RAW_CLASSES = [
    "nontumor_skin_necrosis_necrosis",
    "nontumor_skin_muscle_skeletal",
    "nontumor_skin_sweatglands_sweatglands",
    "nontumor_skin_vessel_vessel",
    "nontumor_skin_elastosis_elastosis",
    "nontumor_skin_chondraltissue_chondraltissue",
    "nontumor_skin_hairfollicle_hairfollicle",
    "nontumor_skin_epidermis_epidermis",
    "nontumor_skin_nerves_nerves",
    "nontumor_skin_subcutis_subcutis",
    "nontumor_skin_dermis_dermis",
    "nontumor_skin_sebaceousglands_sebaceousglands",
    "tumor_skin_epithelial_sqcc",
    "tumor_skin_melanoma_melanoma",
    "tumor_skin_epithelial_bcc",
    "tumor_skin_naevus_naevus",
]

RAW_TO_CLINICAL = {r: r for r in _RAW_CLASSES}
CLASSES = list(RAW_TO_CLINICAL.values())
PROMPT_TEMPLATE = "{c}"
