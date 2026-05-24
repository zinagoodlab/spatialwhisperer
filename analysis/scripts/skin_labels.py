"""Class labels and prompt template for the Kriegsmann et al. (2022) skin benchmark.

Declared as a Snakemake input of `score_kriegsmann_skin`, so any edit here
invalidates all downstream score caches. This is the single source of truth
for what we evaluate against.

The 16 classes match appendix B.4 verbatim: 12 non-tumor anatomical structures
followed by 4 tumor types. Raw class strings are the `class` column values in
`tiles-v2.csv` (MUSK datasets dump).
"""

RAW_TO_CLINICAL = {
    "nontumor_skin_necrosis_necrosis": "Necrosis",
    "nontumor_skin_muscle_skeletal": "Skeletal muscle",
    "nontumor_skin_sweatglands_sweatglands": "Eccrine glands",
    "nontumor_skin_vessel_vessel": "Vessels",
    "nontumor_skin_elastosis_elastosis": "Elastosis",
    "nontumor_skin_chondraltissue_chondraltissue": "Chondral tissue",
    "nontumor_skin_hairfollicle_hairfollicle": "Hair follicle",
    "nontumor_skin_epidermis_epidermis": "Epidermis",
    "nontumor_skin_nerves_nerves": "Nerves",
    "nontumor_skin_subcutis_subcutis": "Subcutis",
    "nontumor_skin_dermis_dermis": "Dermis",
    "nontumor_skin_sebaceousglands_sebaceousglands": "Sebaceous glands",
    "tumor_skin_epithelial_sqcc": "Squamous cell carcinoma",
    "tumor_skin_melanoma_melanoma": "Melanoma",
    "tumor_skin_epithelial_bcc": "Basal cell carcinoma",
    "tumor_skin_naevus_naevus": "Naevi",
}

CLASSES = list(RAW_TO_CLINICAL.values())
PROMPT_TEMPLATE = "{c}"
