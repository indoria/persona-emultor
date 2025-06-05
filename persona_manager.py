import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import spacy
from transformers import pipeline, set_seed

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PersonaManager")

class Persona:
    """
    Persona class for loading, managing, and interacting with a deep persona definition.
    """

    def __init__(self, persona_json_path: str):
        self.persona_json_path = persona_json_path
        self.persona = self._load_persona_json(persona_json_path)
        self._init_nlp_models()
        self.dynamic_context_weights = {}

    def _load_persona_json(self, path: str) -> dict:
        """Load persona JSON, handling errors gracefully."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                persona = json.load(f)
            logger.info(f"Loaded persona from {path}")
            return persona
        except FileNotFoundError:
            logger.error(f"Persona JSON file not found: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            raise

    def _init_nlp_models(self):
        """Initialize NLP and ML models (spaCy for NER, sentiment; transformers for generation)."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning("spaCy model not found; attempting to download.")
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        try:
            # Use a small, fast model for demonstration; replace with fine-tuned as needed.
            self.generator = pipeline("text-generation", model="gpt2")
            set_seed(42)
        except Exception as e:
            logger.warning("Transformers model not found or cannot be loaded. Generation will not work.")
            self.generator = None

    def get_attribute(self, attr_path: str) -> Any:
        """
        Safely access nested attributes in the persona JSON via dot notation.
        Supports list attributes with [name='...'] syntax.
        Example: 'core_attributes.core_principles_and_values[name=\"Fearless Pursuit of Truth\"].weight'
        """
        tokens = attr_path.split(".")
        node = self.persona
        try:
            for tok in tokens:
                if "[" in tok and "]" in tok:
                    # e.g. core_principles_and_values[name='Fearless Pursuit of Truth']
                    base, rest = tok.split("[", 1)
                    cond = rest[:-1]  # Remove trailing ]
                    key, value = cond.split("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    found = None
                    for item in node[base]:
                        if str(item.get(key)) == value:
                            found = item
                            break
                    if found is None:
                        raise KeyError(f"No item in {base} with {key}={value}")
                    node = found
                else:
                    node = node[tok]
            return node
        except Exception as e:
            logger.error(f"Failed to access attribute '{attr_path}': {e}")
            return None

    def _apply_dynamic_weights(self, context: str):
        """
        Adjust attribute weights at runtime based on context keywords.
        Simple heuristic implementation, can be improved with ML-based context understanding.
        """
        context_keywords = set(context.lower().split())
        updates = {}
        # Define mapping: context keyword -> (attribute_path, delta)
        rules = [
            # Crisis/humanitarian/social justice
            ({"crisis", "humanitarian", "disaster", "justice", "marginalized"}, [
                ("core_attributes.core_principles_and_values[name='Amplifying Marginalized Voices'].weight", 0.05),
                ("core_attributes.core_principles_and_values[name='Human-Centric Storytelling'].weight", 0.05),
                ("core_attributes.core_principles_and_values[name='Fearless Pursuit of Truth'].weight", 0.03),
            ]),
            # Politics/governance/accountability
            ({"politics", "governance", "government", "accountability"}, [
                ("core_attributes.core_principles_and_values[name='Holding Power Accountable'].weight", 0.05),
                ("communication_linguistic_style.overall_tone.secondary_tones", "Incisive"),
                ("personality_behavioral_modeling.biases_and_tendencies[type='Skepticism of Authority'].weight", 0.05),
            ]),
            # Press conference
            ({"press", "conference", "briefing"}, [
                ("interaction_protocols.reaction_to_press_conference_protocol.priority_questions", "prioritize"),
            ]),
        ]
        # Apply heuristic rules
        for keywords, actions in rules:
            if keywords & context_keywords:
                for attr_path, delta in actions:
                    if isinstance(delta, (int, float)):
                        orig = self.get_attribute(attr_path)
                        if orig is not None:
                            new_val = max(0, min(orig + delta, 1))
                            # Patch value in persona (walk down and set)
                            self._set_attribute(attr_path, new_val)
                            updates[attr_path] = new_val
                    elif isinstance(delta, str) and delta == "prioritize":
                        # For demonstration: set a flag in dynamic context
                        updates[attr_path] = "prioritized"
        self.dynamic_context_weights = updates
        logger.info(f"Applied dynamic weight updates: {updates}")
        return updates

    def _set_attribute(self, attr_path: str, value: Any):
        """
        Set a nested attribute in the persona JSON via dot notation.
        Only works for numeric/str leaf attributes.
        """
        tokens = attr_path.split(".")
        node = self.persona
        try:
            for i, tok in enumerate(tokens):
                if "[" in tok and "]" in tok:
                    base, rest = tok.split("[", 1)
                    cond = rest[:-1]  # Remove trailing ]
                    key, v = cond.split("=")
                    key = key.strip()
                    v = v.strip().strip("'\"")
                    for item in node[base]:
                        if str(item.get(key)) == v:
                            if i == len(tokens) - 1:
                                item = value
                            else:
                                node = item
                            break
                else:
                    if i == len(tokens) - 1:
                        node[tok] = value
                    else:
                        node = node[tok]
        except Exception as e:
            logger.error(f"Failed to set attribute '{attr_path}': {e}")

    def assess_pitch(self, pitch_details: dict) -> dict:
        """
        Assess a news pitch using persona's weighted criteria.
        Uses NLP to analyze pitch details and scores based on criteria.
        """
        protocols = self.get_attribute("interaction_protocols.pitch_assessment_criteria.criteria_breakdown")
        rubric = self.get_attribute("interaction_protocols.pitch_assessment_criteria.overall_scoring_rubric")
        # Score each criterion using NLP and pitch details
        scores = {}
        strengths = []
        weaknesses = []
        doc = self.nlp(pitch_details.get("summary", ""))
        for crit, crit_def in protocols.items():
            weight = crit_def["weight"]
            details = crit_def["details"]
            sub_scores = []
            # Use NLP: entity, keyword, and thematic match
            for sub_crit in crit_def.get("sub_criteria", []):
                # Heuristic: score sub_criteria based on presence of relevant info in pitch
                if sub_crit in pitch_details:
                    val = pitch_details[sub_crit]
                else:
                    # Use NLP: check if relevant terms/phrases appear
                    val = 1.0 if sub_crit.replace("_", " ") in pitch_details.get("summary", "").lower() else 0.5
                sub_scores.append(val)
            # Aggregate
            if sub_scores:
                crit_score = sum(sub_scores) / len(sub_scores)
            else:
                # Default: if no sub_criteria, use 0.7 if key theme appears, 0.3 otherwise
                crit_score = 0.7 if any(w in pitch_details.get("summary", "").lower() for w in details.split()) else 0.3
            scores[crit] = crit_score * weight * 100  # convert to percentage
            # Identify strengths/weaknesses
            if crit_score > 0.75:
                strengths.append(crit)
            elif crit_score < 0.5:
                weaknesses.append(crit)
        # Overall score is weighted sum
        overall_percentage = sum(scores.values())
        # Feedback message (adhering to persona's response framing principles)
        principles = self.get_attribute("interaction_protocols.response_framing_principles")
        message = self._frame_pitch_feedback(
            pitch_details, scores, strengths, weaknesses, overall_percentage, principles
        )
        return {
            "overall_percentage": overall_percentage,
            "criterion_scores": scores,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "feedback_message": message
        }

    def _frame_pitch_feedback(self, pitch_details, scores, strengths, weaknesses, overall, principles):
        """
        Construct feedback message according to persona's framing principles.
        """
        # Use some typical Barkha Dutt phrases
        phrases = self.get_attribute("core_attributes.common_phrases_keywords")
        closing = "Thank you for your pitch and journalistic effort."
        # Compose
        msg = []
        msg.append(f"Thank you for your pitch titled '{pitch_details.get('title', '')}'.")
        msg.append(f"I appreciate your focus on: {', '.join(pitch_details.get('keywords', []))}.")
        msg.append(f"\n**Assessment:**\nBased on our journalistic criteria and {principles['goal_of_response']}")
        msg.append(f"Your pitch scored **{overall:.1f}/100**.")
        msg.append("**Strengths:** " + ", ".join(strengths) if strengths else "No significant strengths identified.")
        msg.append("**Areas to Improve:** " + ", ".join(weaknesses) if weaknesses else "No major weaknesses identified.")
        if overall >= 75:
            msg.append("This pitch is strong and aligns with our focus on human-centric, accountability journalism.")
            msg.append("Next steps: Please provide any additional verifiable data or direct contacts for ground reporting.")
        else:
            msg.append("This pitch requires further development to meet our editorial standards. Consider strengthening the following: "
                       + ", ".join(weaknesses) + ".")
            msg.append("Next steps: Revise and resubmit with more supporting evidence and a broader societal impact perspective.")
        msg.append(closing)
        # Insert a high-frequency persona phrase
        if phrases:
            import random
            msg.insert(0, random.choice([p["phrase"] for p in phrases if p["frequency"] in ["high", "medium"]]))
        return "\n".join(msg)

    def generate_response(self, prompt: str, context: str = "general discussion") -> dict:
        """
        Generate a context-aware, persona-adherent response.
        Uses NLP for context, text generation for output, and enforces persona's style and avoid_behavior.
        """
        # NLP: sentiment, topic, entities
        doc = self.nlp(prompt)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        tokens = [t.text.lower() for t in doc if not t.is_stop]
        sentiment = self._simple_sentiment(prompt)
        # Dynamic weighting
        self._apply_dynamic_weights(context)
        # Compose a seed prompt for the generator using persona style cues
        style = self.get_attribute("communication_linguistic_style")
        tone = style["overall_tone"]["primary_tone"]
        phrase = None
        if "empathy" in tone.lower() or sentiment == "NEGATIVE":
            phrase = "The human cost of this decision..."
        else:
            phrase = "Let's get to the heart of the matter..."
        # Build prompt
        gen_prompt = (
            f"As Barkha Dutt, an {tone} Indian journalist, "
            f"respond to: '{prompt}'\n"
            f"Use a style that is {tone}, employs strong verbs, "
            f"and maintains professional, incisive analysis. "
            f"Phrase: {phrase}\n"
        )
        # Generate response (using transformers)
        if self.generator:
            output = self.generator(gen_prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        else:
            output = (
                f"{phrase} {prompt}\n"
                "(Demo mode: GPT-2 not available. This is a placeholder for persona-adherent, AI-generated text.)"
            )
        # Avoid forbidden patterns
        avoid = self.get_attribute("core_attributes.avoid_behavior") or []
        for forbidden in avoid:
            if forbidden.lower() in output.lower():
                output = output.replace(forbidden, "[REDACTED]")
        # Return structure
        return {
            "response_text": output.strip(),
            "style_applied": {
                "tone": tone,
                "vocabulary_usage": style["vocabulary_usage"]["descriptors"],
                "sentence_structure": style["sentence_and_paragraph_structure"]
            },
            "entities": entities,
            "sentiment": sentiment
        }

    def _simple_sentiment(self, text: str) -> str:
        """
        Dummy sentiment analysis: positive/negative/neutral based on word cues.
        Replace with a proper model for real use.
        """
        pos_words = {"hope", "success", "progress", "improve"}
        neg_words = {"crisis", "failure", "problem", "injustice", "tragedy"}
        tokens = set(text.lower().split())
        if tokens & pos_words:
            return "POSITIVE"
        elif tokens & neg_words:
            return "NEGATIVE"
        return "NEUTRAL"

    def ask_question(self, topic: str, context: str = "general") -> dict:
        """
        Generate a persona-style question on a given topic/context, using questioning techniques.
        """
        techniques = self.get_attribute("communication_linguistic_style.questioning_technique.typical_question_types")
        priorities = self.get_attribute("interaction_protocols.reaction_to_press_conference_protocol.priority_questions")
        # Choose a technique based on the topic/context
        import random
        if "governance" in topic.lower() or "policy" in context.lower():
            tech = random.choice([t for t in techniques if "accountability" in t["type"].lower()])
            qtype = "accountability"
        elif "human" in topic.lower() or "society" in topic.lower():
            tech = random.choice([t for t in techniques if "human" in t["purpose"].lower()])
            qtype = "human"
        else:
            tech = random.choice(techniques)
            qtype = "general"
        # Compose question using persona phrases
        phrases = self.get_attribute("core_attributes.common_phrases_keywords")
        phrase = random.choice([p["phrase"] for p in phrases if p["frequency"] in ["high", "medium"]])
        # If in press conference context, use priority_questions
        if "press conference" in context.lower():
            pq = max(priorities, key=lambda x: x["weight"])
            question = f"{pq['focus']} ({phrase})"
        else:
            question = f"{tech['purpose']}. {phrase} On {topic}, could you elaborate?"
        return {
            "question_text": question,
            "technique_used": tech["type"],
            "purpose": tech["purpose"]
        }