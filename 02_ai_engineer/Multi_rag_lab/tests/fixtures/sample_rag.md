# Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, combines information retrieval with text generation. A query is used to retrieve relevant evidence from an external corpus. The selected evidence is inserted into the model context before generation.

## Chunking

Chunking divides documents into retrievable units. Fixed-size chunking is simple but can cut across semantic boundaries. Recursive chunking prefers natural separators. Semantic chunking uses embeddings to identify topical changes. Parent-child chunking retrieves small child units but can provide a larger parent unit to generation.

## Embeddings and Vector Search

Embedding models map text into vectors. Dense retrieval compares the query embedding with stored chunk embeddings. Cosine similarity is commonly implemented by normalizing vectors and using an inner product index. Sparse retrieval such as BM25 instead relies on lexical term statistics.

## Hybrid Retrieval and Reranking

Hybrid retrieval combines dense and sparse rankings. Reciprocal Rank Fusion combines rank positions; its score is a fusion score and not a probability. A cross-encoder reranker can rescore a candidate set using the query and document together.

## Grounded Generation

Grounded generation instructs the language model to answer only from the retrieved evidence. Source markers and provenance metadata support citations. If evidence is insufficient, the system should say that the supplied documents do not contain enough information rather than inventing a factual answer.

## Evaluation

Retrieval quality can be measured with Recall at K, Precision at K, Hit Rate, Mean Reciprocal Rank and nDCG. Generation evaluation can examine completeness, faithfulness and citation correctness. Performance metrics such as latency and throughput should be kept separate from quality metrics.
