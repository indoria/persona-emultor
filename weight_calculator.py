import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WeightCalculator")

def _parse_list_path(path: str):
    """
    Parse an attribute path that may include list item access, e.g.:
    'core_attributes.core_principles_and_values[name="Fearless Pursuit of Truth"].weight'
    Returns list of (key, idx_or_name) tokens.
    """
    tokens = []
    for part in path.split("."):
        if "[" in part and "]" in part:
            key, rest = part.split("[", 1)
            cond = rest[:-1]
            k, v = cond.split("=")
            k = k.strip()
            v = v.strip().strip("'\"")
            tokens.append((key, (k, v)))
        else:
            tokens.append((part, None))
    return tokens

def _find_and_set(node: Any, tokens, value):
    """
    Recursively walk node following tokens and set value at leaf.
    """
    if not tokens:
        return
    key, idx = tokens[0]
    if idx is None:
        # Dict
        if len(tokens) == 1:
            node[key] = value
        else:
            _find_and_set(node[key], tokens[1:], value)
    else:
        # List of dicts
        arr = node[key]
        match = None
        for item in arr:
            if str(item.get(idx[0])) == idx[1]:
                match = item
                break
        if match is None:
            raise KeyError(f"No match for {idx[0]}={idx[1]} in {key}")
        if len(tokens) == 1:
            match.clear()
            match.update(value)
        else:
            _find_and_set(match, tokens[1:], value)

def calculate_attribute_weights(observation_data_path: str, output_json_path: str, persona_template_path: str):
    """
    Calculate average normalized attribute weights from observation data, update persona template, and save.
    """
    # Load data
    with open(observation_data_path, "r", encoding="utf-8") as f:
        obs_data = json.load(f)
    with open(persona_template_path, "r", encoding="utf-8") as f:
        persona = json.load(f)

    # Gather per-attribute scores
    attr_scores: Dict[str, list] = {}
    attr_max: Dict[str, int] = {}
    for obs in obs_data:
        for attr, manual_score in obs["manual_scores"].items():
            max_score = obs.get("max_score_per_attribute", 5)
            norm = manual_score / max_score
            attr_scores.setdefault(attr, []).append(norm)
            attr_max[attr] = max_score  # For info

    # Compute averages
    updates = {}
    for attr, vals in attr_scores.items():
        updates[attr] = sum(vals) / len(vals)
    # Update persona JSON
    updated = persona.copy()
    not_found = []
    for attr_path, avg_weight in updates.items():
        try:
            tokens = _parse_list_path(attr_path)
            node = updated
            for i, (key, idx) in enumerate(tokens):
                if i == len(tokens) - 1:
                    if idx is None:
                        node[key] = avg_weight
                    else:
                        arr = node[key]
                        for item in arr:
                            if str(item.get(idx[0])) == idx[1]:
                                item = avg_weight
                                break
                else:
                    if idx is None:
                        node = node[key]
                    else:
                        arr = node[key]
                        for item in arr:
                            if str(item.get(idx[0])) == idx[1]:
                                node = item
                                break
            logger.info(f"Updated {attr_path} to {avg_weight:.3f}")
        except Exception as e:
            logger.warning(f"Could not update {attr_path}: {e}")
            not_found.append(attr_path)
    # Save out
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    logger.info(f"Saved updated persona model to {output_json_path}")
    if not_found:
        logger.warning(f"Could not update the following attributes: {not_found}")

# Usage example (uncomment to run directly)
# if __name__ == "__main__":
#     calculate_attribute_weights("observations.json", "updated_persona.json", "persona_model.json")