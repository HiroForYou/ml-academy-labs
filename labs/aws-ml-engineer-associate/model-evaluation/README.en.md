# Lab 2.3 — Evaluate RAG retrieval

[Español](README.md) · **English**

**Course:** AWS Certified Machine Learning Engineer – Associate (MLA-C02) · **Topic:** 2.3 Analyze and evaluate performance (links to 4.3)

## Goal

Reproduce locally what Amazon Bedrock Knowledge Bases and Amazon Bedrock RAG evaluations do:

1. Split a corpus into chunks with two strategies: fixed 30-word windows and one chunk per paragraph.
2. Measure retrieval on its own with **Recall@k**, **Precision@k**, coverage, and **MRR**.
3. Check that, without a **per-tenant metadata filter**, a question retrieves another company's data.
4. Separate retrieval failures from generation failures.

No AWS account, credentials, internet access, or external models are needed, except to install packages. scikit-learn TF-IDF replaces embeddings and a deterministic extractive generator replaces the LLM.

## Dataset

`src/corpus.py` builds a synthetic corpus: 2 tenants (`andina` and `boreal`) with 6 internal policies each (12 documents). Both use the **same templates with different values**, as in a multi-tenant SaaS. It includes 30 labelled questions per language (15 per tenant), each with its expected answer and evidence sentence. There is no randomness: results are reproducible.

`corpus.export` saves each document with its `.metadata.json` (`{"metadataAttributes": {"tenant": ...}}`), the format of a Bedrock Knowledge Bases S3 data source.

## Run

Requires Python 3.10 or later. All labs and `ml-academy-media` share **a single virtual environment**; each lab only installs its own `requirements.txt`.

```bash
# Single global environment in the root ml-academy/ folder (create it once).
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r ml-academy-labs/labs/aws-ml-engineer-associate/model-evaluation/requirements.txt
cd ml-academy-labs/labs/aws-ml-engineer-associate/model-evaluation
jupyter lab notebook.en.ipynb
```

The notebook ships with the outputs and charts from a run, so you can read it directly on GitHub. To reproduce it, use *Run → Run All Cells*.

| File | Contents |
|---|---|
| `notebook.en.ipynb` | Guided lab in English |
| `src/` | Reusable modules: corpus, chunking, retrieval, evaluation, and charts |

## What to look for

| Step | Result from the included run |
|---|---|
| 1. Chunking | `fixed-30` yields 38 chunks (27 words on average) and splits some evidence sentences in two; `paragraph` yields 36 (29 words). |
| 2. No filter | Recall@1 of 0.33 (`fixed-30`) and 0.40 (`paragraph`). 50% of the top-1 results and 93% of the top-3 contain the other tenant's chunk: both tenants' chunks tie on score. |
| 3. With tenant filter | Recall@1 rises to 0.67 (`fixed-30`) and 0.80 (`paragraph`); Recall@3 = 0.87 for both. Precision@3 favours `fixed-30` (0.42 vs 0.29) only because split evidence counts as two relevant chunks. |
| 4. End to end (`paragraph`, top-3) | No filter: 12 of 30 answers correct and 15 with the other tenant's value. With the filter: 24 correct, 4 retrieval misses, and 2 generation misses (the evidence was in the context). |

Takeaway: end-to-end accuracy doesn't tell you what to fix. Retrieval metrics show that without the filter the problem is retrieval and security, and that with the filter the remaining errors split between retrieval and generation.

## AWS equivalent

| This lab | On AWS | Main differences |
|---|---|---|
| `chunking.py` | Bedrock Knowledge Bases chunking: default (~300 tokens), fixed-size with overlap, hierarchical, semantic, or no chunking | Bedrock counts tokens, not words. Semantic chunking uses a foundation model at extra cost. |
| `retrieval.py` (TF-IDF) | Embeddings model + vector store, with semantic or hybrid search | Embeddings capture synonyms that TF-IDF misses. Hybrid search is only available on some vector stores. |
| `Retriever.search(tenant=...)` | Metadata filter (`equals`, `in`, `andAll`, `orAll`…) on `Retrieve` / `RetrieveAndGenerate` | The filter value must come from the authenticated identity, not from the user's text. |
| `evaluation.retrieval_metrics` | Bedrock RAG evaluation, *Retrieve only* job: *context relevance* and *context coverage* | Bedrock uses an LLM as a judge and doesn't need chunk IDs. *Context coverage* requires ground truth. Here the metrics are deterministic. |
| `evaluation.end_to_end` | Bedrock RAG evaluation, *Retrieve and generate* job: *correctness*, *completeness*, *faithfulness*, *citation precision*… | Bedrock can evaluate a knowledge base or an external RAG with your own responses. |

## References

- [Chunking in Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-chunking.html)
- [Configure queries: number of results, search type, and metadata filters](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html)
- [Evaluate RAG sources with Amazon Bedrock evaluations](https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-kb.html) · [RAG evaluation metrics](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-evaluation-metrics.html)

Idea adapted from the MLA-C02 Study Guide in [certification-study-library](https://github.com/cterpening/certification-study-library) (CC BY 4.0, Chris Terpening and contributors).
