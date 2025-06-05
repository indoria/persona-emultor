# Barkha Dutt AI Persona System

A modular, extensible Python system for simulating a highly accurate AI persona based on a deep JSON model, exemplified for Indian journalist Barkha Dutt.

## Features

- **Persona Management:** Load and access persona attributes (including list items) safely with dot notation, e.g., `core_attributes.core_principles_and_values[name='Amplifying Marginalized Voices'].weight`.
- **Dynamic Weighting:** Adjust persona attribute weights at runtime based on context using heuristic rules.
- **NLP-Driven Pitch Assessment:** Score and critique news pitches using spaCy for entity extraction and text analysis, and persona's weighted criteria.
- **Persona-Adherent Response Generation:** Generate responses using Hugging Face Transformers, guided by persona's communication style, tone, and forbidden behaviors.
- **Question Generation:** Pose context-appropriate questions using persona's questioning techniques and style.
- **Weight Calculation Module:** Update persona weights from manually scored observation data for iterative refinement.

## Structure

- `persona_manager.py` — Main Persona class, including NLP/ML integrations and interaction methods.
- `weight_calculator.py` — Module to update persona weights from observation data.
- `main.py` — Demonstrates loading, weight calculation, and all major persona capabilities.
- `persona_model.py` — Holds the default path to the persona JSON, useful for import.
- `persona_model.json` — Deep, weighted persona definition (example JSON provided separately).
- `requirements.txt` — All required Python packages.

## Usage

1. **Install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Run the demo:**
   ```bash
   python main.py --persona_json persona_model.json
   ```

   To update weights from observation data:
   ```bash
   python main.py --recalculate_weights --observation_data observations.json
   ```

## Notes on NLP/ML Integration

- **spaCy** is used for entity extraction, basic sentiment, and keyword analysis in both pitch assessment and response generation.
- **Transformers (Hugging Face)** are used for text generation. For full persona fidelity, finetuning on Barkha Dutt's work is recommended.
- **Scikit-learn** can be used to train custom intent/topic classifiers (not implemented in stubs).
- **Text generation** is guided by persona style/tone by seeding the prompt with explicit style instructions.

## Extending

- Add new personas by providing new JSON models.
- Add new interaction methods by extending `Persona` class.
- For production, consider caching models and using more robust ML pipelines.

## License

MIT
