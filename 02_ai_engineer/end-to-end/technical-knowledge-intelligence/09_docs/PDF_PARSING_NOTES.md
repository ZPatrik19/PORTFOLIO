# PDF Parsing Notes

The project uses the current `pymupdf` import name. Some publisher-generated PDFs contain malformed but recoverable PDF operators. MuPDF can repair many of these documents and continue extracting text, but normally prints diagnostic messages such as `MuPDF error: ...` to the console.

Bulk ingestion suppresses these recoverable console diagnostics while leaving genuine Python exceptions intact. A PDF that cannot actually be parsed is therefore still listed as an ingestion failure; a repaired PDF does not flood the terminal with non-fatal diagnostics.
