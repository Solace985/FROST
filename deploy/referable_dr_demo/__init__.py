"""FROST — Frozen Representation for Ocular Screening and Triage.

FROST Referable Diabetic Retinopathy Research Demonstrator.

A locally-hosted research demonstrator that reproduces the study's exact
referable-DR pipeline on a single uploaded fundus image:

    uploaded image
    -> native-392 preprocessing
    -> frozen RETFound-Green backbone (ViT-S/14, average pooling)
    -> 384-dimensional embedding
    -> locked MultiTaskHead
    -> 5-class DR-grade logits
    -> softmax
    -> referable-DR score = P(grade>=2) = softmax[2] + softmax[3] + softmax[4]
    -> fixed validation-derived operating-point threshold
    -> REFERABLE / NOT REFERABLE research-triage result

This package lives entirely under ``deploy/referable_dr_demo/`` and imports the
canonical pipeline modules under ``src/retina_screen/`` as read-only libraries.
It never modifies, retrains, or re-saves any pipeline artifact.

This is a research demonstrator. It is NOT a medical device and is NOT
validated for clinical decisions.
"""

__all__ = ["__version__", "APP_NAME", "APP_TITLE", "APP_TAGLINE"]

__version__ = "0.1.0"

APP_NAME = "FROST"
APP_TITLE = "Referable Diabetic Retinopathy Research Demonstrator"
APP_TAGLINE = "Frozen Representation for Ocular Screening and Triage"
