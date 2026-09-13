# Local usage analytics

The Streamlit UI creates `usage_history.sqlite3` in this directory at runtime.

It stores chat questions, generated answers, selected methodology, language/data mode,
latency, and tool-call traces so the **Live Statistics** page can calculate real metrics
from actual project usage. The database is local-only and is ignored by Git.

Delete the database or use the reset control in the UI if you want to clear the history.
