# Data & Library Pipeline

## Input collections

```text
01_data/user_library/     your own books, notes and internal documents
01_data/reference_docs/   public/demo/rebuildable reference material
```

`user_library` replaced the narrower name `private_books`: the collection can contain much more than books. `reference_docs` replaced `public_docs`: the important distinction is that these are reference sources that can be recreated or legally redistributed, not that every file is automatically safe to publish.

The user library is ignored by Git and release packaging. Reference documents uploaded locally are also ignored unless explicitly allowlisted by repository rules.

## Adding a new document

### UI workflow

Open **Library → Add documents**:

1. choose **User library** or **Reference docs**;
2. upload one or more supported files;
3. click **Upload + rebuild index**.

The UI writes the file to the selected input folder and calls the same `KnowledgePlatform.ingest_and_index()` path used by the CLI.

### Manual workflow

Copy the file into the correct folder, then run:

```bat
run_project.bat ingest
```

or use **Library → Rebuild index only**.

Supported types: PDF, DOCX, TXT, Markdown, HTML and EPUB.

## Processing flow

```text
user_library / reference_docs
        ↓
file discovery
        ↓
stable document_id + SHA-256 checksum
        ↓
format-specific parsing
        ↓
logical blocks: heading / paragraph / code / table / figure / list
        ↓
quality checks
        ↓
chunking
        ↓
embedding cache
        ↓
BM25 corpus + vector index
        ↓
index metadata/version state
```

Parse cache keys include document identity/checksum. Embedding cache keys include chunk ID and provider/model identity, so unchanged data can be reused.

## Chunking

The project supports fixed, recursive, structure-aware and lightweight semantic chunking. `structure_aware` is the default for technical documents because headings and code boundaries often matter more than exact uniform size.

Persistent chunking experiments require separate embeddings per strategy; vectors from different chunk boundaries are never mixed in one retriever. See [Multi-index chunking](MULTI_INDEX_CHUNKING.md).

## Index backend

The portable default is NumPy. Qdrant server is optional for a more production-like deployment. See [Index backend policy](INDEX_BACKEND_POLICY.md).
