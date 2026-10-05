from __future__ import annotations


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def expand_query(query: str, synonyms: dict[str, list[str]] | None = None) -> str:
    synonyms = synonyms or {}
    additions: list[str] = []
    lower = query.lower()
    for term, values in synonyms.items():
        if term.lower() in lower:
            additions.extend(values)
    return " ".join([query, *additions]).strip()


class QueryRewriter:
    def __init__(self, llm) -> None:
        self.llm = llm

    def rewrite(self, query: str) -> str:
        prompt = (
            "Írd át a felhasználó kérdését egyetlen tömör, magyar nyelvű retrieval lekérdezéssé. "
            "Őrizd meg a neveket, számokat és korlátozásokat. Csak az átírt lekérdezést add vissza.\n"
            "Kérdés: " + query
        )
        return self.llm.generate(prompt).strip() or query


class MultiQueryGenerator:
    def __init__(self, llm, count: int = 3) -> None:
        self.llm = llm
        self.count = count

    def generate(self, query: str) -> list[str]:
        prompt = (
            f"Készíts {self.count} eltérő magyar keresési lekérdezést ugyanarra az információs igényre. "
            "Soronként egy lekérdezést adj, számozás nélkül.\nKérdés: " + query
        )
        lines = [line.strip(" -\t") for line in self.llm.generate(prompt).splitlines() if line.strip()]
        unique = [query]
        for line in lines:
            if line not in unique:
                unique.append(line)
            if len(unique) >= self.count + 1:
                break
        return unique


class HyDEGenerator:
    """Generate a hypothetical evidence passage used only as a retrieval query."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def generate(self, question: str) -> str:
        prompt = (
            "Írj egy rövid, tényszerű hipotetikus dokumentumrészletet, amely ideális esetben "
            "megválaszolná az alábbi keresési kérdést. Ne adj hivatkozást, ne állítsd, hogy ez valódi forrás.\n\n"
            f"Kérdés: {question}\n\nHipotetikus részlet:"
        )
        text = self.llm.generate(prompt).strip()
        return text or question


class FollowUpQueryGenerator:
    """Create one bounded second-hop query from the first-hop evidence."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def generate(self, question: str, evidence: str) -> str:
        prompt = (
            "Készíts pontosan egy rövid második keresési lekérdezést. A cél, hogy az első körös "
            "bizonyítékból hiányzó részletet keressük meg. Csak a lekérdezést add vissza.\n\n"
            f"Eredeti kérdés: {question}\n\nElső körös evidence:\n{evidence[:1800]}\n\nMásodik lekérdezés:"
        )
        text = self.llm.generate(prompt).strip().splitlines()[0] if prompt else ""
        return text or question
