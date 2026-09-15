from __future__ import annotations

import importlib.util
print('Qdrant client:', 'available' if importlib.util.find_spec('qdrant_client') else 'not installed; NumPy fallback will be used')
print('Streamlit:', 'available' if importlib.util.find_spec('streamlit') else 'not installed')
print('Sentence Transformers:', 'available' if importlib.util.find_spec('sentence_transformers') else 'optional cross-encoder unavailable')
