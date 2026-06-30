"""FROST app-local analysis scripts.

Thin entrypoints that build the local deployment bundle, derive the
validation-only operating-point threshold, and run the parity gates. They read
canonical pipeline artifacts read-only and write only to the ignored
``deploy/referable_dr_demo/.local/`` directory.
"""
