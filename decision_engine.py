import os
import json
import math
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from openai import OpenAI

# Fallback directly to string if the environment variable isn't active
API_KEY = os.getenv("OPENAI_API_KEY", "your-actual-openai-api-key")
client = OpenAI(api_key=API_KEY)

# Keep your QuestionDecomposition, System1DecisionEngine, and run_pipeline code below exactly the same...


class QuestionDecomposition(BaseModel):
    category: str = Field(description="Classification: 'direct_info', 'policy_rule', 'multi_condition_decision', or 'unknown'")
    requires_system1_evaluation: bool = Field(description="Set to true if checking quantitative scores, strict constraints, or Boolean eligibility rules.")
    extracted_variables: Dict[str, Any] = Field(default_factory=dict, description="Variables extracted like gpa, year, income, or sports status.")

class System1DecisionEngine:
    @staticmethod
    def evaluate_rules(variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes strict, deterministic System 1 logical transformations.
        This handles two types of decisions: Boolean Eligibility and Categorical Choice.
        """
        try:
            gpa = float(variables.get("gpa", 0.0))
        except:
            gpa = 0.0
            
        try:
            year = int(variables.get("year", 0))
        except:
            year = 0
            
        has_sports = bool(variables.get("has_sports_scholarship", False))

        # Decision 1: Boolean Check
        gpa_valid = gpa >= 3.5
        year_valid = year >= 2
        
        # Decision 2: Category Assignment Matrix
        if has_sports:
            policy_category = "Stipend Cap Restriction Applied"
            final_approval = False
        else:
            policy_category = "Standard Merit Scheme"
            final_approval = gpa_valid and year_valid

        return {
            "gpa_threshold_passed": gpa_valid,
            "academic_year_valid": year_valid,
            "policy_category": policy_category,
            "is_eligible": final_approval
        }

def run_pipeline(question: str, matched_chunks: List[Any]) -> Dict[str, Any]:
    # Handle the "No Relevant Data Available" path cleanly
    if not matched_chunks:
        return {
            "answer": "I am sorry, but the knowledge base does not contain the required information to answer this question.",
            "references": []
        }

    # Step 1: Run parsing step using OpenAI's Structured Outputs mechanism
    context_text = "\n".join([f"Source: {c.filename} P.{c.page_number}: {c.content}" for c in matched_chunks])
    
    decomp_response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract explicit logic checking metrics from user questions."},
            {"role": "user", "content": f"Analyze this question: '{question}'\nContext: {context_text}"}
        ],
        response_format=QuestionDecomposition
    )
    analysis = decomp_response.choices[0].message.parsed

    # Step 2: Conditionally run System 1 rule verification
    decision_matrix = {}
    if analysis.requires_system1_evaluation:
        decision_matrix = System1DecisionEngine.evaluate_rules(analysis.extracted_variables)

    # Step 3: Synthesis Generation Layer
    synthesis_prompt = f"""
    You are an enterprise AI Compliance Assistant. Generate an answer using the provided inputs.
    
    User Question: {question}
    System 1 Decision Results: {json.dumps(decision_matrix)}
    Grounded Document Information: {context_text}
    
    Rules:
    - Base your response ONLY on the provided context and System 1 outputs. Do not invent any outside facts.
    - Explicitly state which rules passed or failed using the System 1 Data.
    """
    
    final_output = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": synthesis_prompt}]
    )

    references = [{"document": c.filename, "page": c.page_number} for c in matched_chunks]
    # Unique reference filtering
    seen = set()
    unique_refs = []
    for ref in references:
        ident = f"{ref['document']}-{ref['page']}"
        if ident not in seen:
            seen.add(ident)
            unique_refs.append(ref)

    return {
        "answer": final_output.choices[0].message.content,
        "references": unique_refs
    }
