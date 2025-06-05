import argparse
import logging
import os
from persona_manager import Persona
from weight_calculator import calculate_attribute_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PersonaMain")

def demo_persona_interactions(persona: Persona):
    # Assess a sample pitch
    pitch = {
        "title": "Marginalized Groups and Healthcare Crisis",
        "summary": "A ground report on how India's most vulnerable have been left behind during the ongoing healthcare crisis, with stories from rural Bihar and urban slums.",
        "keywords": ["healthcare", "marginalized", "crisis", "ground report", "Bihar", "urban slums"],
        "data_provided": True,
        "potential_impact": 0.85
    }
    logger.info("=== Demo: Assess Pitch ===")
    result = persona.assess_pitch(pitch)
    print("Pitch Assessment Result:")
    for k, v in result.items():
        print(f"{k}: {v}\n")

    # Generate a response
    prompt = "How should the Indian media cover protests against new government policies?"
    logger.info("=== Demo: Generate Response ===")
    response = persona.generate_response(prompt, context="politics governance protest")
    print("Generated Response:")
    print(response["response_text"])
    print("Style Applied:", response["style_applied"])
    print("Entities:", response["entities"])
    print("Sentiment:", response["sentiment"])

    # Ask a question
    topic = "Accountability in public health policy"
    logger.info("=== Demo: Ask Question ===")
    q = persona.ask_question(topic, context="press conference")
    print("Generated Persona Question:")
    print(q["question_text"])
    print("Technique Used:", q["technique_used"])
    print("Purpose:", q["purpose"])

def main():
    parser = argparse.ArgumentParser(description="Barkha Dutt AI Persona System")
    parser.add_argument("--persona_json", type=str, default="persona_model.json", help="Path to persona JSON file")
    parser.add_argument("--observation_data", type=str, help="Path to observation data JSON")
    parser.add_argument("--output_json", type=str, default="updated_persona.json", help="Output path for updated persona JSON")
    parser.add_argument("--recalculate_weights", action="store_true", help="Recalculate attribute weights")
    args = parser.parse_args()

    persona_path = args.persona_json

    # Optionally update weights
    if args.recalculate_weights and args.observation_data:
        logger.info("Running weight calculator to update persona weights...")
        calculate_attribute_weights(args.observation_data, args.output_json, persona_path)
        persona_path = args.output_json

    # Load persona and run demo
    persona = Persona(persona_path)
    demo_persona_interactions(persona)

if __name__ == "__main__":
    main()