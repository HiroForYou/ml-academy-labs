# Lab 2.3 — Evaluar la recuperación de un RAG

**Español** · [English](README.en.md)

**Curso:** AWS Certified Machine Learning Engineer – Associate (MLA-C02) · **Tema:** 2.3 Analizar y evaluar el rendimiento (enlaza con 4.3)

## Objetivo

Reproducir localmente lo que hacen Amazon Bedrock Knowledge Bases y las evaluaciones RAG de Amazon Bedrock:

1. Dividir un corpus en chunks con dos estrategias: ventanas fijas de 30 palabras y un chunk por párrafo.
2. Medir la recuperación por separado con **Recall@k**, **Precision@k**, cobertura y **MRR**.
3. Comprobar que, sin **filtro de metadatos por tenant**, una pregunta recupera datos de otra empresa.
4. Separar los fallos de recuperación de los fallos de generación.

No requiere cuenta de AWS, credenciales, internet ni modelos externos, salvo para instalar paquetes. TF-IDF de scikit-learn sustituye a los embeddings y un generador extractivo determinista, al LLM.

## Dataset

`src/corpus.py` genera un corpus sintético: 2 tenants (`andina` y `boreal`) con 6 políticas internas cada uno (12 documentos). Ambos usan las **mismas plantillas con valores distintos**, como en un SaaS multi-tenant. Incluye 30 preguntas etiquetadas por idioma (15 por tenant), cada una con su respuesta esperada y su frase de evidencia. No hay aleatoriedad: el resultado es reproducible.

`corpus.export` guarda cada documento con su `.metadata.json` (`{"metadataAttributes": {"tenant": ...}}`), el formato de un origen S3 de Bedrock Knowledge Bases.

## Ejecutar

Requiere Python 3.10 o superior. Todos los laboratorios y `ml-academy-media` comparten **un solo entorno virtual**; cada laboratorio solo instala su propio `requirements.txt`.

```bash
# Entorno global único, en la carpeta raíz ml-academy/ (se crea una sola vez).
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r ml-academy-labs/labs/aws-ml-engineer-associate/model-evaluation/requirements.txt
cd ml-academy-labs/labs/aws-ml-engineer-associate/model-evaluation
jupyter lab notebook.es.ipynb
```

El notebook ya incluye las salidas y los gráficos de una ejecución, así que puedes leerlo directamente en GitHub. Para reproducirlo, usa *Run → Run All Cells*.

| Archivo | Contenido |
|---|---|
| `notebook.es.ipynb` | Laboratorio guiado en español |
| `src/` | Módulos reutilizables: corpus, chunking, recuperación, evaluación y gráficos |

## Qué observar

| Paso | Resultado de la ejecución incluida |
|---|---|
| 1. Chunking | `fixed-30` produce 50 chunks (24 palabras de media) y parte frases de evidencia en dos; `paragraph` produce 36 (34 palabras). |
| 2. Sin filtro | Recall@1 de 0,40 (`fixed-30`) y 0,43 (`paragraph`). El 50 % de los top-1 y el 100 % de los top-3 contienen un chunk del otro tenant: los chunks de ambos empatan en puntuación. |
| 3. Con filtro de tenant | Recall@1 sube a 0,78 (`fixed-30`) y 0,87 (`paragraph`); Recall@3 = 0,97 en ambos. Precision@3 favorece a `fixed-30` (0,36 frente a 0,32) solo porque la evidencia partida cuenta como dos chunks relevantes. |
| 4. Extremo a extremo (`paragraph`, top-3) | Sin filtro: 12 de 30 respuestas correctas y 15 con el dato del otro tenant. Con filtro: 24 correctas, 5 fallos de generación (la evidencia estaba en el contexto) y 1 fallo de recuperación. |

Conclusión: la exactitud de extremo a extremo no dice qué arreglar. Las métricas de recuperación muestran que sin filtro el problema es de recuperación y de seguridad, y que con filtro lo que falla es la generación.

## Equivalencia en AWS

| Este laboratorio | En AWS | Diferencias principales |
|---|---|---|
| `chunking.py` | Chunking de Bedrock Knowledge Bases: predeterminado (~300 tokens), tamaño fijo con solapamiento, jerárquico, semántico o sin chunking | Bedrock mide en tokens, no en palabras. El chunking semántico usa un modelo fundacional con costo adicional. |
| `retrieval.py` (TF-IDF) | Modelo de embeddings + vector store, con búsqueda semántica o híbrida | Los embeddings capturan sinónimos que TF-IDF no ve. La búsqueda híbrida solo está disponible en algunos vector stores. |
| `Retriever.search(tenant=...)` | Filtro de metadatos (`equals`, `in`, `andAll`, `orAll`…) en `Retrieve` / `RetrieveAndGenerate` | El valor del filtro debe salir de la identidad autenticada, no del texto del usuario. |
| `evaluation.retrieval_metrics` | Evaluación RAG de Bedrock, tipo *Retrieve only*: *context relevance* y *context coverage* | Bedrock usa un LLM como juez y no necesita IDs de chunks. *Context coverage* requiere ground truth. Aquí las métricas son deterministas. |
| `evaluation.end_to_end` | Evaluación RAG de Bedrock, tipo *Retrieve and generate*: *correctness*, *completeness*, *faithfulness*, *citation precision*… | Bedrock puede evaluar una base de conocimiento o un RAG externo con tus propias respuestas. |

## Referencias

- [Chunking en Bedrock Knowledge Bases](https://docs.aws.amazon.com/es_es/bedrock/latest/userguide/kb-chunking.html)
- [Configurar consultas: número de resultados, tipo de búsqueda y filtros de metadatos](https://docs.aws.amazon.com/es_es/bedrock/latest/userguide/kb-test-config.html)
- [Evaluar orígenes RAG con evaluaciones de Amazon Bedrock](https://docs.aws.amazon.com/es_es/bedrock/latest/userguide/evaluation-kb.html) · [Métricas de evaluación RAG](https://docs.aws.amazon.com/es_es/bedrock/latest/userguide/knowledge-base-evaluation-metrics.html)

Idea adaptada de la MLA-C02 Study Guide de [certification-study-library](https://github.com/cterpening/certification-study-library) (CC BY 4.0, Chris Terpening y colaboradores).
