from __future__ import annotations
from collections import Counter
import hashlib
import pandas as pd
from .models import DocumentRecord, Chunk

def build_quality_report(documents: list[DocumentRecord], chunks: list[Chunk], parse_failures: list[dict] | None = None):
    parse_failures=parse_failures or []
    hashes=[hashlib.sha1(c.text.strip().lower().encode()).hexdigest() for c in chunks]
    counts=Counter(hashes)
    lengths=[len(c.text) for c in chunks]
    summary={
        "documents_total":len(documents), "documents_failed":len(parse_failures),
        "parsing_failure_rate":len(parse_failures)/max(1,len(documents)),
        "chunks_total":len(chunks), "too_short_chunks":sum(x<120 for x in lengths),
        "too_long_chunks":sum(x>5000 for x in lengths), "duplicate_chunks":sum(v-1 for v in counts.values() if v>1),
        "avg_chunk_chars":sum(lengths)/max(1,len(lengths)), "min_chunk_chars":min(lengths,default=0), "max_chunk_chars":max(lengths,default=0)
    }
    per_doc=pd.DataFrame([{"document_id":d.document_id,"title":d.title,"chunks":sum(c.document_id==d.document_id for c in chunks)} for d in documents])
    chunk_df=pd.DataFrame([{"chunk_id":c.chunk_id,"document_id":c.document_id,"chars":len(c.text),"language":c.language,"chunk_type":c.chunk_type} for c in chunks])
    return summary, per_doc, chunk_df
