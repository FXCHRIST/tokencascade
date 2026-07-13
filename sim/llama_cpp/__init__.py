"""Mock llama_cpp for pipeline simulation. Returns plausible canned answers
so every deterministic layer of main.py (routing, verification, format
enforcement, repair loops, sink) executes for real, without model weights.

Scenario switches (env):
  SIM_MATH_NL_WRONG=1      NL derivation returns a wrong number (code path
                           must catch it and override)
  SIM_CODE_BROKEN_FIRST=1  first code_gen/debug attempt has a SyntaxError
                           (repair loop must fix it via the retry)
  SIM_SUMM_VIOLATE=1       summaries ignore the sentence constraint
                           (format retry/repair must fire)
  SIM_NER_GARBAGE=1        NER JSON comes back wrapped in prose
                           (repair must extract the array)
  SIM_SENT_BURIED=1        sentiment label buried mid-answer
                           (label surfacing must fire)
  SIM_LOGIC_TRUNCATED=1    logic reasoning is cut off with no Answer line
                           (conclude-salvage must recover it)
  SIM_FACT_WRONG=1         factual answer swaps two terms; the review pass
                           must flag it and the correction must ship
  SIM_SLOW_S=<seconds>     sleep per generation (governor testing)
  SIM_LOAD_FAIL=1          Llama() constructor raises (L3 fallback test)
"""
import json
import os
import re
import time


class LlamaGrammar:
    @staticmethod
    def from_string(_s):
        return "MOCK_GRAMMAR"


class Llama:
    def __init__(self, model_path=None, **kwargs):
        if os.environ.get("SIM_LOAD_FAIL") == "1":
            raise RuntimeError("simulated model load failure")
        self.calls = 0

    def create_chat_completion(self, messages=None, max_tokens=256,
                               temperature=0.2, grammar=None, **kw):
        self.calls += 1
        slow = float(os.environ.get("SIM_SLOW_S", "0"))
        if slow:
            time.sleep(slow)
        system = messages[0]["content"]
        user = messages[1]["content"]
        text = self._answer(system, user, temperature)
        return {"choices": [{"message": {"content": text}}]}

    # ------------------------------------------------------------------
    def _answer(self, system, user, temperature):
        s, u = system.lower(), user.lower()

        # --- math: expression path ---
        if "single python arithmetic expression" in s:
            if "240 items" in u:
                return "240 - 240*0.15 - 60"
            if "2,400 units" in u or "2400 units" in u:
                return "2400 - 2400*0.37 + 800 - 640"
            return "42"

        # --- math: code path ---
        if "python script" in s and "print()" in s:
            if "240 items" in u:
                return "```python\ntotal=240\nsold=total*0.15\nprint(total-sold-60)\n```"
            if "2,400 units" in u or "2400 units" in u:
                return "```python\nx=2400\nx-=2400*0.37\nx+=800\nx-=640\nprint(x)\n```"
            return "```python\nprint(42)\n```"

        # --- math: NL path ---
        if "math problem step by step" in s:
            if "240 items" in u:
                if os.environ.get("SIM_MATH_NL_WRONG") == "1":
                    return "15% of 240 is 36. 240-36-60 = 150.\nAnswer: 150"
                return "15% of 240 is 36. 240 - 36 = 204. 204 - 60 = 144.\nAnswer: 144"
            if "2,400 units" in u or "2400 units" in u:
                return "37% of 2400 = 888. 2400-888=1512. 1512+800=2312. 2312-640=1672.\nAnswer: 1672"
            return "Answer: 42"

        # --- code gen / debug ---
        if "python code block" in s and ("corrected" in s or "satisfies the request" in s):
            broken_first = (os.environ.get("SIM_CODE_BROKEN_FIRST") == "1"
                            and "previous attempt failed" not in s)
            if "second-largest" in u:
                if broken_first:
                    return "```python\ndef second_largest(nums)\n    return sorted(set(nums))[-2]\n```"
                return ("```python\ndef second_largest(nums):\n"
                        "    uniq = sorted(set(nums))\n"
                        "    return uniq[-2]\n```")
            if "get_max" in u:
                if broken_first:
                    return "```python\ndef get_max(nums)\n    return max(nums)\n```"
                return ("The bug: it returns the first element instead of "
                        "scanning the list.\n```python\ndef get_max(nums):\n"
                        "    return max(nums)\n```")
            if "merge" in u:
                return ("```python\ndef merge_sorted(a, b):\n"
                        "    return sorted(a + b)\n```")
            return "```python\ndef solution():\n    return None\n```"

        # --- NER ---
        if "json array" in s:
            arr = [{"text": "Maria Sanchez", "type": "PERSON"},
                   {"text": "Fireworks AI", "type": "ORGANIZATION"},
                   {"text": "Berlin", "type": "LOCATION"},
                   {"text": "March", "type": "DATE"}]
            js = json.dumps(arr)
            if os.environ.get("SIM_NER_GARBAGE") == "1":
                return f"Sure! Here are the entities you asked for:\n{js}\nHope that helps!"
            return js
        if "named entities" in s.lower() or "extract all named entities" in s.lower():
            return ("Maria Sanchez (PERSON)\nFireworks AI (ORGANIZATION)\n"
                    "Berlin (LOCATION)\nMarch (DATE)")

        # --- sentiment ---
        if "sentiment" in s:
            if os.environ.get("SIM_SENT_BURIED") == "1":
                return ("Looking at this review, the tone is mixed. There are "
                        "clear negatives but also strong positives overall.")
            return ("Mixed. The battery life is praised but the screen "
                    "scratching is a clear drawback.")

        # --- summarization ---
        if "summarize" in s:
            if os.environ.get("SIM_SUMM_VIOLATE") == "1" and "count before answering" not in s:
                return ("The office piloted a four-day week. Output stayed "
                        "flat. Sick days dropped. It is now recommended "
                        "permanently.")
            return ("The regional office piloted a four-day work week and, "
                    "with output flat and sick days down, recommended "
                    "adopting it permanently.")

        # --- logic continuation ---
        if "finishing a logic-puzzle solution" in s:
            return ("Continuing: since Jo has the dog and Sam cannot have "
                    "the bird, Sam takes the cat and Lee the bird.\n"
                    "Answer: Sam owns the cat")

        # --- logic conclude salvage ---
        if "reasoning" in s and "cut off" in s:
            return "Answer: Sam owns the cat"

        # --- logic ---
        if "logic puzzle" in s or "puzzle" in s:
            if os.environ.get("SIM_LOGIC_TRUNCATED") == "1" and "last line" not in s:
                return ("Jo owns the dog. Sam does not own the bird, so Sam "
                        "must own the cat. Checking each clue in turn: Jo has")
            return ("Jo owns the dog. Sam does not own the bird, so Sam owns "
                    "the cat and Lee owns the bird.\nAnswer: Sam owns the cat")

        # --- factual review pass ---
        if "review the answer" in s:
            if os.environ.get("SIM_FACT_WRONG") == "1" and "tasman" in u:
                return "The answer wrongly says Canberra is near the Tasman Sea."
            return "OK"

        # --- factual ---
        corrective = ("reviewer identified" in s) or ("reviewer found" in s)
        if os.environ.get("SIM_FACT_WRONG") == "1" and not corrective:
            return ("The capital of Australia is Canberra, located near the "
                    "Tasman Sea.")
        return ("The capital of Australia is Canberra, located near Lake "
                "Burley Griffin on the Molonglo River.")
