"""
This file holds the path to the default persona JSON model, and can provide utilities for loading it.
"""

import os

DEFAULT_PERSONA_JSON_PATH = os.path.join(os.path.dirname(__file__), "persona_model.json")

def get_persona_json_path():
    return DEFAULT_PERSONA_JSON_PATH