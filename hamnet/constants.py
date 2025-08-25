"""Project-wide constants and categorical mappings used across the codebase."""

# Global seed to keep dataset splits, shuffling, and training deterministic
SEED = 42

# Encode binary sex metadata as integers consumed by the model
SEX_MAPPING = {"male": 0, "female": 1}

# Stable class index mapping for classification head (7 classes)
DIAGNOSIS_MAPPING = {
    "Nevus": 0,
    "Melanoma, NOS": 1,
    "Pigmented benign keratosis": 2,
    "Dermatofibroma": 3,
    "Squamous cell carcinoma, NOS": 4,
    "Basal cell carcinoma": 5,
    "Solar or actinic keratosis": 6,
}

# Coarse anatomical site categories used as a numeric feature
ANATOM_SITE_MAPPING = {
    "anterior torso": 0,
    "posterior torso": 1,
    "head/neck": 2,
    "upper extremity": 3,
    "lower extremity": 4,
    "palms/soles": 5,
    "oral/genital": 6,
}
