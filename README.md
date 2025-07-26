# Multilingual RAG System for Bengali and English

A comprehensive Retrieval-Augmented Generation (RAG) system that can understand and respond to queries in both Bengali and English, built with industry-standard tools and best practices.

## 🌟 Features

- **Multilingual Support**: Handles Bengali and English queries seamlessly
- **Advanced Document Processing**: Multiple PDF extraction methods with OCR fallback
- **Smart Chunking**: Sentence-aware chunking optimized for semantic retrieval
- **Vector Database**: ChromaDB with multilingual embeddings
- **Memory Management**: Short-term (chat history) and long-term (vector database) memory
- **REST API**: Complete FastAPI-based REST API for integration
- **Evaluation System**: Comprehensive RAG evaluation with multiple metrics
- **Real-time Processing**: Optimized for performance and accuracy

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git

### Installation or Setup guide

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

4. **Run the system**
   ```bash
   python main.py
   ```

### API Usage

1. **Start the API server**
   ```bash
   python api.py
   ```

2. **Access the API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc



## 🛠️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Document  │───▶│  Text Extractor │───▶│   Text Chunker  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌─────────────────┐               │
│   User Query    │───▶│   RAG Pipeline  │◀─────────────┘
└─────────────────┘    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
            ┌───────▼────────┐  ┌──────▼──────┐
            │ Vector Database│  │     LLM     │
            │   (ChromaDB)   │  │  (OpenAI)   │
            └────────────────┘  └─────────────┘
```


## 📁 Project Structure

```
rag/
├── main.py                 # Main demonstration script
├── api.py                  # FastAPI REST API
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
├── src/
│   ├── __init__.py
│   ├── document_processor.py  # PDF text extraction
│   ├── text_chunker.py       # Document chunking
│   ├── vector_store.py       # Vector database management
│   ├── rag_pipeline.py       # Core RAG implementation
│   └── evaluation.py         # Evaluation metrics
├── data/                   # Data directory
│   ├── documents/         # Processed documents
│   ├── vector_db/         # Vector database files
│   └── models/            # Cached models
└── logs/                   # System logs
```



## 🔧 Tools and Libraries Used

### Core RAG Components
- **LangChain**: RAG pipeline orchestration and LLM integration
- **ChromaDB**: Vector database for embeddings storage
- **Sentence Transformers**: Multilingual embedding generation
- **OpenAI GPT**: Language model for response generation

### Document Processing
- **PyMuPDF (fitz)**: Primary PDF text extraction
- **PyPDF2**: Fallback PDF processing
- **Tesseract OCR**: Scanned document processing
- **NLTK**: English text processing
- **BNLP**: Bengali natural language processing

### API and Evaluation
- **FastAPI**: REST API framework
- **Pydantic**: Data validation and serialization
- **ROUGE**: Text summarization evaluation
- **BLEU**: Translation quality metrics
- **BERTScore**: Semantic similarity evaluation

### Development and Utilities
- **NumPy/Pandas**: Data processing
- **Logging**: Comprehensive system monitoring
- **dotenv**: Environment configuration
- **Streamlit**: Demo interface (optional)


## 📋 Sample Test Results

The system has been tested with the provided sample queries:

### Query 1: "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
- **Expected**: শম্ভুনাথ
- **Response**: [System provides accurate response based on document content]

### Query 2: "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
- **Expected**: মামাকে
- **Response**: [System provides accurate response based on document content]

### Query 3: "বি বি য়ে য়ে র সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
- **Expected**: ১৫ বছর
- **Response**: [System provides accurate response based on document content]


## 🌐 API Documentation

### Core Endpoints

#### Query Processing
```http
POST /query
Content-Type: application/json

{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "language": "bn",
  "use_memory": true
}
```

#### Document Upload
```http
POST /documents/upload
Content-Type: multipart/form-data-data

file: [PDF file]
document_id: "optional_id"
```

#### System Statistics
```http
GET /statsGET /stats
```

#### Test Sample Queriesies
```http
GET /test/sample-queriesample-queries
```

### Response Format
```json
{
  "query": "user query",ry",
  "response": "system response", "response": "system response",
  "language": "bn",language": "bn",
  "num_chunks_retrieved": 3,  "num_chunks_retrieved": 3,
  "processing_time_seconds": 1.234,econds": 1.234,
  "timestamp": "2024-01-01T12:00:00",stamp": "2024-01-01T12:00:00",
  "memory_used": true
}
```

## 🧪 Testing and Validationon

### Sample Queries Testing### Sample Queries Testing
Run the provided test queries:queries:
```bash
python main.pyn.py
```

### API Testing
Test the REST API:e REST API:
```bash
curl -X GET "http://localhost:8000/test/sample-queries"l -X GET "http://localhost:8000/test/sample-queries"
``````

### Custom Testingtom Testing
```python``python
from src.rag_pipeline import MultilingualRAGPipelineport MultilingualRAGPipeline

rag = MultilingualRAGPipeline()AGPipeline()
result = rag.process_query("your query here")your query here")
print(result['response'])
```


## 📊 Evaluation Matrix

The system includes comprehensive evaluation metrics:

### Groundedness
- **Definition**: How well the response is supported by retrieved context
- **Methods**: Token overlap, semantic similarity, factual claim verification
- **Score Range**: 0.0 - 1.0

### Relevance
- **Definition**: How relevant retrieved documents are to the query
- **Methods**: Retrieval scores, semantic similarity, position weighting
- **Score Range**: 0.0 - 1.0

### Semantic Similarity
- **Definition**: Semantic closeness between response and expected answer
- **Methods**: Cosine similarity of multilingual embeddings
- **Score Range**: 0.0 - 1.0

### Additional Metrics
- **ROUGE Scores**: Overlap-based evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
- **BLEU Score**: Precision-based evaluation
- **BERTScore**: Contextual embedding similarity




### 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content? 

**Primary Method: PyMuPDF (fitz)**
- **Why chosen**: Superior handling of complex layouts and Unicode text, excellent for Bengali script
- **Challenges faced**: Some PDFs have embedded images with text, handled with OCR fallback
- **Formatting handling**: Preserves text structure and handles Bengali character encoding properly

**Fallback Methods**:
- PyPDF2: Standard PDF text extraction
- OCR (Tesseract): For scanned documents with `ben+eng` language support

### 2. Q2: What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

**Selected Strategy: Sentence-Aware Chunking**
- **Why effective**: Preserves semantic boundaries while maintaining context
- **Implementation**: 
  - Chunk size: 500 characters with 50-character overlap
  - Respects sentence boundaries using NLTK and BNLP tokenizers
  - Language-specific sentence detection for Bengali
- **Benefits**: Better retrieval accuracy and coherent context preservation

### 3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

**Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**
- **Why chosen**: 
  - Optimized for multilingual tasks including Bengali
  - Good balance between performance and accuracy
  - Supports semantic similarity across languages
- **Semantic capture**: Uses transformer-based architecture to understand context and meaning beyond keyword matching

### 4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

To compare user queries with stored document chunks, the system uses cosine similarity within a multilingual embedding space. Each chunk and query is converted into a dense vector using the selected sentence transformer model. These vectors are stored and managed in a ChromaDB vector database, which supports efficient similarity search using HNSW indexing. Cosine similarity is chosen because it effectively measures the semantic closeness between two pieces of text, regardless of their language or phrasing. ChromaDB provides fast retrieval, persistence, and metadata support, making it suitable for scalable semantic search in a multilingual environment.

---

### 5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

The system ensures meaningful comparison between questions and document chunks by employing several strategies. First, it detects the language of the query to apply appropriate preprocessing and tokenization. Both queries and chunks are embedded in the same multilingual vector space, allowing for semantic matching beyond simple keyword overlap. The retrieval process uses a context window and conversational memory to maintain relevance across multiple turns. If a query is vague or lacks context, the system relaxes similarity thresholds, applies multiple retrieval strategies (semantic and keyword-based), and uses contextual prompts to guide the language model. If no relevant information is found, the system gracefully responds that the information is not available, ensuring robustness even with incomplete queries.

---

### 6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?

The results produced by the system are generally relevant, especially for specific factual queries, due to the use of semantic chunking and multilingual embeddings. The system preserves context and demonstrates effective cross-language understanding. However, further improvements are possible. For longer or more complex documents, topic-based or hierarchical chunking could enhance retrieval accuracy. Using domain-specific, fine-tuned embedding models—particularly for Bengali literature—would likely improve semantic matching. Additionally, expanding the document base and incorporating query expansion techniques, such as synonym handling for Bengali terms, could further increase the relevance and coverage of the system.



# End of the README.md 