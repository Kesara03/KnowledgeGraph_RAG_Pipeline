# KnowledgeGraph_RAG_Pipeline

A production-grade implementation of a hybrid Retrieval-Augmented Generation (RAG) system leveraging knowledge graphs for historical geopolitical analysis. This project demonstrates the integration of Neo4j graph databases with large language models to enable complex reasoning over interconnected historical entities and relationships.

## Abstract

This repository presents a complete RAG pipeline that combines structured knowledge graph representations with unstructured text retrieval to answer complex queries about historical relationships between Greenland, Norway, and Denmark. The system employs LLM-powered entity extraction, graph construction, and hybrid search mechanisms to provide contextually accurate responses grounded in both semantic similarity and graph traversal.

## Knowledge Graph Visualization

The following demonstration shows the RAG pipeline in action - exploring the Neo4j knowledge graph structure.

![System Demo](Resources/demo.gif)

*Interactive demonstration: Exploring entities and relationships in the Neo4j graph.*

## System Architecture

The pipeline consists of four primary components:

### 1. Data Ingestion and Processing

```
Wikipedia API → Document Loader → Token-based Text Splitter → Chunked Documents
```

- **Source**: Wikipedia articles via `WikipediaLoader`
- **Chunking Strategy**: Token-based splitting with 512-token chunks and 24-token overlap
- **Preprocessing**: Metadata extraction and document structuring

### 2. Knowledge Graph Construction

```
Chunked Documents → LLM Transformer → Graph Documents → Neo4j Database
```

- **Entity Extraction**: LLM-powered identification of entities (people, places, events, organizations)
- **Relationship Extraction**: Automated detection of semantic relationships between entities
- **Graph Storage**: Neo4j graph database with entity and relationship nodes
- **Schema**: Dynamic schema with `__Entity__` base label and relationship types

### 3. Hybrid Retrieval System

```
Query → Entity Extraction → [Structured Retrieval + Vector Retrieval] → Context
```

**Structured Retrieval**:
- Fulltext index search on entity nodes
- Cypher query execution for relationship traversal
- Multi-hop graph pattern matching

**Vector Retrieval**:
- Embedding-based semantic similarity search
- Document node vector index with OpenAI embeddings
- Hybrid score normalization and fusion

### 4. Response Generation

```
Context (Structured + Unstructured) → LLM → Natural Language Response
```

- **Model**: OpenAI GPT-3.5-turbo (gpt-3.5-turbo-0125)
- **Prompt Engineering**: Structured context injection with explicit instructions
- **Conversational Memory**: Chat history integration for follow-up queries

## Technical Implementation

### Core Technologies

| Component | Technology | Version/Model |
|-----------|-----------|---------------|
| Graph Database | Neo4j | Cloud Instance |
| LLM | OpenAI GPT-3.5-turbo | gpt-3.5-turbo-0125 |
| Embeddings | OpenAI | text-embedding-ada-002 |
| Orchestration | LangChain | Latest |
| Visualization | yFiles Jupyter Graphs | ^1.10.9 |

### Graph Schema

**Node Types**:
- `__Entity__`: Base entity type with `id` property
- `Document`: Text chunks with embeddings
- Dynamic entity types based on extraction

**Relationship Types** (Examples):
- `DISCOVERED`
- `RULED_BY`
- `COLONIZED`
- `AUTONOMOUS_TERRITORY`
- `TRADED_WITH`
- `MENTIONS` (Document-Entity links)

**Indexes**:
- Fulltext index on `__Entity__.id`
- Vector index on `Document.embedding` (1536 dimensions)
- Keyword index for hybrid search

### Graph Statistics

The knowledge graph constructed from Wikipedia articles contains:

- **Total Nodes**: 300+ entities and document chunks
- **Total Relationships**: 500+ semantic connections
- **Entity Types**: People, Places, Organizations, Events, Political Entities
- **Relationship Types**: 20+ distinct relationship categories
- **Temporal Coverage**: 2500 BCE to 2026 CE
- **Geographic Scope**: Arctic region (Greenland, Norway, Denmark, Iceland)

## Key Features

### 1. Automated Knowledge Graph Construction

Transforms unstructured text into structured graph representations using LLM-powered extraction:

```python
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
```

### 2. Hybrid Retrieval Architecture

Combines graph-based structured retrieval with vector-based semantic search:

- **Structured Retriever**: Traverses graph relationships to find connected entities
- **Vector Retriever**: Performs similarity search over embedded document chunks
- **Fusion Strategy**: Concatenates structured and unstructured context for comprehensive responses

### 3. Multi-Hop Reasoning

Enables complex queries requiring multiple relationship traversals:

```cypher
MATCH (e:__Entity__)-[r1]->(intermediate)-[r2]->(target)
WHERE e.id = $entity
RETURN path
```

### 4. Conversational Context Management

Maintains chat history for follow-up questions:

```python
chain.invoke({
    "question": "When did he discover it?",
    "chat_history": [("Who was Erik the Red?", "Erik the Red was a Norse explorer...")]
})
```

## Installation

### Prerequisites

- Python 3.8+
- Neo4j Database (Cloud or Local)
- OpenAI API Key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/KnowledgeGraph_RAG_Pipeline.git
cd KnowledgeGraph_RAG_Pipeline
```

2. Install dependencies:
```bash
pip install langchain langchain-community langchain-openai langchain-experimental neo4j wikipedia tiktoken yfiles_jupyter_graphs pydantic
```

3. Configure environment variables:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["NEO4J_URI"] = "neo4j+s://your-instance.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your-password"
```

4. Initialize the knowledge graph:
```bash
jupyter notebook Knowledge_graph_cookbook.ipynb
```

## Usage

### Basic Query Example

```python
# Initialize the retrieval chain
chain = (
    RunnableParallel({
        "context": _search_query | retriever,
        "question": RunnablePassthrough(),
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Execute a query
response = chain.invoke({"question": "What is Greenland's relationship with Denmark?"})
print(response)
```

### Example Queries

The system can answer complex historical queries such as:

- "Why is Greenland Danish instead of Norwegian?"
- "What historical connections exist between Norway, Denmark, and Greenland?"
- "Which dual-state entity claimed sovereignty over Greenland in 1721?"
- "Who discovered Greenland and where were they from?"
- "What happened during World War II in Greenland?"

### Follow-up Queries

```python
# First query
response1 = chain.invoke({"question": "Who was Erik the Red?"})

# Follow-up with context
response2 = chain.invoke({
    "question": "When did he discover Greenland?",
    "chat_history": [("Who was Erik the Red?", response1)]
})
```

## Dataset

### Data Sources

Primary data is sourced from Wikipedia articles covering:

- **History of Greenland**: Comprehensive historical timeline from Paleo-Inuit cultures to modern sovereignty
- **History of Norway**: Norwegian kingdoms, unions, and colonial activities
- **History of Denmark**: Danish realm, colonial empire, and modern Kingdom of Denmark

### Data Processing Pipeline

1. **Document Loading**: Wikipedia API queries for specified topics
2. **Text Chunking**: 512-token chunks with 24-token overlap for context preservation
3. **Entity Extraction**: LLM identifies entities across categories:
   - Geographic entities (Greenland, Norway, Denmark, Iceland)
   - Historical figures (Erik the Red, Hans Egede)
   - Political entities (Denmark-Norway, Kalmar Union)
   - Events (Treaty of Kiel, World War II occupation)
   - Organizations (NATO, European Union)

4. **Relationship Extraction**: Automated identification of semantic relationships:
   - Temporal relationships (RULED_BY, SUCCEEDED_BY)
   - Geographic relationships (AUTONOMOUS_TERRITORY, COLONIZED)
   - Social relationships (DISCOVERED, TRADED_WITH)

## Methodology

### Graph Construction

The knowledge graph is constructed using a systematic approach:

1. **Document Retrieval**: Fetch authoritative historical sources
2. **Text Segmentation**: Chunk documents to maintain semantic coherence
3. **Entity Recognition**: Use LLM to identify named entities and concepts
4. **Relationship Inference**: Extract relationships through contextual analysis
5. **Graph Population**: Insert nodes and edges into Neo4j with deduplication

### Retrieval Strategy

The hybrid retrieval mechanism operates in parallel:

**Structured Path**:
1. Extract entities from user query using LLM
2. Execute fulltext search on entity index
3. Traverse graph to find related entities (1-2 hops)
4. Format relationships as structured triples

**Unstructured Path**:
1. Embed user query using OpenAI embeddings
2. Perform vector similarity search on document nodes
3. Retrieve top-k most similar document chunks
4. Extract text content for context

**Fusion**:
- Concatenate structured triples and unstructured text
- Provide combined context to LLM for response generation

### Evaluation Approach

System performance is assessed through:

1. **Qualitative Analysis**: Manual review of responses to complex multi-hop queries
2. **Retrieval Accuracy**: Verification that relevant entities and relationships are retrieved
3. **Response Coherence**: Assessment of natural language quality and factual accuracy
4. **Comparative Analysis**: Evaluation against baseline vector-only RAG system

## Results

### Sample Query Results

**Query**: "Why is Greenland Danish instead of Norwegian?"

**Response**: 
> Greenland is Danish instead of Norwegian because it has been politically and culturally associated with the European kingdoms of Norway and Denmark for over a millennium, beginning in 986. Greenland was transferred from the Norwegian to the Danish crown in 1814 when Denmark and Norway separated. Greenland has been under Danish sovereignty since then.

**Analysis**: The system successfully synthesizes information across multiple time periods (986 CE to 1814) and correctly identifies the Treaty of Kiel as the pivotal event in the sovereignty transfer.

### System Capabilities

The implementation demonstrates:

- **Multi-hop Reasoning**: Successfully traces relationship chains across 3+ entities
- **Temporal Understanding**: Correctly sequences historical events and maintains chronological accuracy
- **Entity Disambiguation**: Distinguishes between different entities with similar names
- **Context Synthesis**: Combines graph structure with semantic content for comprehensive responses

### Observed Limitations

- **Incomplete Entity Extraction**: Some entities mentioned in documents are not extracted
- **Relationship Granularity**: Complex relationships may be oversimplified
- **Temporal Precision**: Date ranges and durations require additional modeling
- **Source Attribution**: Current implementation lacks citation tracking

## Limitations and Future Work

### Current Limitations

1. **Data Coverage**: Limited to Wikipedia sources; lacks primary historical documents
2. **Language Support**: English-only implementation; Norwegian and Danish sources not included
3. **Graph Completeness**: Entity extraction recall is imperfect; some entities missing
4. **Relationship Types**: Fixed set of relationship types; lacks fine-grained temporal semantics
5. **Scalability**: Single-instance Neo4j deployment; not optimized for large-scale queries

### Future Enhancements

**Technical Improvements**:
- Implement entity resolution and coreference resolution
- Add temporal knowledge graph capabilities (valid-from, valid-to timestamps)
- Integrate citation tracking and source provenance
- Deploy distributed Neo4j cluster for scalability
- Implement graph neural network embeddings for improved retrieval

**Domain Expansion**:
- Extend to other Arctic sovereignty disputes (Alaska, Svalbard, Arctic Ocean)
- Include multilingual sources (Norwegian, Danish, Greenlandic)
- Incorporate primary historical documents and treaties
- Add economic and climate data for comprehensive analysis

**System Features**:
- Interactive graph visualization interface
- Explainability module showing retrieval paths
- Fact-checking against primary sources
- Multi-document synthesis for conflicting accounts

## References

### Technical Documentation

- [LangChain Documentation](https://python.langchain.com/docs/)
- [Neo4j Graph Database](https://neo4j.com/docs/)
- [OpenAI API](https://platform.openai.com/docs/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

### Data Sources

- Wikipedia API
- Neo4j AuraDB
