# cruscotto_agent/eval/run_judge_eval.py
import json
import asyncio
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from cruscotto_agent.models import judge_model
from cruscotto_agent.nodes import JUDGE_SYSTEM_PROMPT

async def judge_case(case: dict) -> bool:
    """Ritorna il verdict.grounded del giudice per questo caso."""
    user_prompt = f"""DATI GREZZI:
        {json.dumps(case['raw_data'], ensure_ascii=False)}

        RISPOSTA DA VERIFICARE:
        {case['answer_text']}"""
    verdict = await judge_model.ainvoke([SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(user_prompt)])
    return verdict.grounded

async def main():
    with open("eval_data/injection_cases_hard.json") as f:
        cases = json.load(f)

    results = []
    for case in cases:
        grounded = await judge_case(case)
        
        should_be_flagged = case["injection_type"] != "clean"  # ci aspettiamo un'anomalia
        was_flagged = not grounded                             # il giudice ha detto "non grounded"

        if should_be_flagged and was_flagged:
            outcome = "TP"
        elif should_be_flagged and not was_flagged:
            outcome = "FN"
        elif not should_be_flagged and was_flagged:
            outcome = "FP"
        else:
            outcome = "TN"

        results.append(outcome)
        print(f"{case['id']:25s} {case['injection_type']:20s} -> {outcome}")

    tp = results.count("TP")
    fn = results.count("FN")
    fp = results.count("FP")
    tn = results.count("TN")

    detection_rate = tp / (tp + fn) if (tp + fn) else float("nan")
    false_positive_rate = fp / (fp + tn) if (fp + tn) else float("nan")

    print(f"\nEvaluated on {len(cases)} annotated cases: "
          f"{detection_rate:.0%} of injected unsupported claims detected "
          f"at {false_positive_rate:.0%} false positives")

if __name__ == "__main__":
    asyncio.run(main())