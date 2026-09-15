# Data collections

The application has two input collections.

## `user_library/`

Your own books, notes, PDFs and internal documents. These files are local working data and are ignored by Git/release packaging by default.

## `reference_docs/`

Public/rebuildable reference material and the small original `demo_*.md` corpus used for offline demonstrations and tests. Do not commit third-party documents unless their license explicitly permits redistribution.

## Adding documents

Preferred: **Streamlit → Library → Add documents**.

Manual alternative:

1. copy the file into `user_library/` or `reference_docs/`;
2. run `run_project.bat ingest` or click **Rebuild index only** in the UI;
3. the pipeline discovers, parses, chunks, embeds and indexes the new file.

Supported extensions: PDF, DOCX, TXT, Markdown, HTML and EPUB.
