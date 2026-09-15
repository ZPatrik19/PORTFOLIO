from __future__ import annotations
from .utils import detect_language

def classify_intent(q:str)->str:
    s=q.lower()
    if any(x in s for x in ["compare","hasonlíts","különbség"]): return "COMPARISON"
    if any(x in s for x in ["code","kód","implement","dataset example","példa"]): return "CODE_SEARCH"
    if any(x in s for x in ["teach me","taníts","learning plan","tanulási terv"]): return "LEARNING"
    if any(x in s for x in ["which document","melyik dokument","hol talál","where can i find"]): return "METADATA_SEARCH"
    if any(x in s for x in ["official documentation","hivatalos dokumentáció"]): return "PUBLIC_DOC_COMPARISON"
    if any(x in s for x in ["what is","mi az","magyarázd","explain"]): return "CONCEPTUAL"
    return "FACTUAL"

def rewrite_query(q:str,intent:str)->str:
    # Deterministic rewrite keeps factual content, adds retrieval intent terms only when useful.
    if intent=="CODE_SEARCH" and "example" not in q.lower(): return q+" implementation example code"
    if intent=="LEARNING": return q+" intuition concepts mathematics example common mistakes"
    return q

def understand_query(q:str)->dict:
    intent=classify_intent(q)
    return {"intent":intent,"language":detect_language(" "+q+" "),"rewritten_query":rewrite_query(q,intent)}
