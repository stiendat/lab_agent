# 🤖 Local AI Document Agent

A complete, locally-hosted AI document agent that automatically indexes your documents and provides intelligent question-answering capabilities using RAG (Retrieval-Augmented Generation).

## 🌟 Features

- **🔍 Auto-Indexing**: Automatically monitors directories and indexes new documents in real-time
- **📚 Multi-Format Support**: Handles PDF, Word documents (.docx), text files (.txt), and code files (.py, .js, .md, etc.)
- **🧠 Semantic Search**: Uses sentence-transformers for high-quality 384-dimensional embeddings
- **💾 Dual Storage**: SQLite for metadata + Milvus Lite for vector storage
- **🤖 RAG-Powered Q&A**: Ask questions about your documents and get AI-powered answers with source citations
- **⚡ Real-Time Processing**: File changes are detected and processed instantly
- **🖥️ Interactive CLI**: User-friendly command-line interface
- **⚙️ .env Configuration**: Easy configuration via environment variables

## 📋 Prerequisites

- **Python 3.8+** (tested on Python 3.12)
- **Local LLM Server**: llama.cpp or compatible OpenAI API server running on `localhost:8081`
- **Operating System**: macOS, Linux, or Windows

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to your project directory
cd /path/to/LAB_Agent

# Install dependencies (using uv - recommended)
uv sync

# OR using pip
pip install -r requirements.txt
```

### 2. Configure Your Agent (Optional)

The agent comes with sensible defaults, but you can customize it using the `.env` file:

```bash
# Edit the .env file to configure your preferences
nano .env
```

See the [📝 Configuration](#-configuration) section below for all available options.

### 3. Start Your Local LLM Server

Before running the agent, ensure you have a local LLM server running on port 8081:

```bash
# Example with llama.cpp
./llama-server --host localhost --port 8081 --model your-model.gguf

# The agent expects OpenAI-compatible API at:
# http://localhost:8081/v1/chat/completions
```

### 4. Run the Complete Agent

```bash
python complete_agent.py
```

This starts both:
- 🔍 **Auto-indexing file watcher** (monitors your configured directories)
- 💬 **Interactive query interface** (for asking questions)

### 5. Add Documents

Copy any documents to the monitored directories:
```bash
# Default directories (can be changed in .env)
cp your-document.pdf docs-to-watch-1/
cp your-notes.txt docs-to-watch-2/
```

Files are automatically indexed within seconds! ⚡

### 6. Ask Questions

```
💬 Your question (or 'exit'/'stats'): What does the document say about machine learning?

🤖 AI Assistant:
Based on your documents, machine learning is described as...

📚 Sources (2 documents):
   1. your-document.pdf
   2. your-notes.txt
```

## ⚙️ Configuration

The AI Document Agent uses a `.env` file for easy configuration. Here's how to customize it:

### 📝 .env File Structure

```env
# AI Document Agent Configuration

# Directories to monitor for auto-indexing (comma-separated)
# You can specify absolute or relative paths
MONITOR_DIRECTORIES=./docs-to-watch-1,./docs-to-watch-2

# LLM Server Configuration
LLM_API_URL=http://localhost:8081/v1/chat/completions

# Database Configuration (optional)
SQLITE_PATH=metadata.sqlite
MILVUS_PATH=./milvus_local.db

# Embedding Configuration (optional)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=auto

# Search Configuration (optional)
DEFAULT_TOP_K=5
SEARCH_TIMEOUT=30
```

### 🔧 Configuration Options

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MONITOR_DIRECTORIES` | Directories to watch (comma-separated) | `./docs-to-watch-1,./docs-to-watch-2` | `./my-docs,/Users/me/Documents` |
| `LLM_API_URL` | Local LLM server endpoint | `http://localhost:8081/v1/chat/completions` | `http://192.168.1.100:8080/v1/chat/completions` |
| `SQLITE_PATH` | SQLite database file path | `metadata.sqlite` | `./db/documents.sqlite` |
| `MILVUS_PATH` | Milvus database file path | `./milvus_local.db` | `./db/vectors.db` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` | `all-MiniLM-L12-v2` |
| `EMBEDDING_DEVICE` | Device for embeddings (auto/cpu/cuda) | `auto` | `cuda` |
| `DEFAULT_TOP_K` | Number of chunks to retrieve | `5` | `10` |
| `SEARCH_TIMEOUT` | LLM request timeout (seconds) | `30` | `60` |

### 📁 Directory Configuration Examples

**Monitor Multiple Personal Folders:**
```env
MONITOR_DIRECTORIES=./Documents,./Research,./Notes,./Downloads/PDFs
```

**Monitor Absolute Paths:**
```env
MONITOR_DIRECTORIES=/Users/username/Documents,/Users/username/Research,/path/to/work/docs
```

**Single Directory:**
```env
MONITOR_DIRECTORIES=./my-knowledge-base
```

**Network or External Drives:**
```env
MONITOR_DIRECTORIES=/Volumes/ExternalDrive/Documents,/Network/SharedDocs
```

### 🖥️ Server Configuration Examples

**Different LLM Server:**
```env
LLM_API_URL=http://192.168.1.100:11434/v1/chat/completions
```

**Custom Port:**
```env
LLM_API_URL=http://localhost:3000/v1/chat/completions
```

**Remote Server:**
```env
LLM_API_URL=https://my-llm-server.com/v1/chat/completions
```

### 🗄️ Database Configuration Examples

**Custom Database Locations:**
```env
SQLITE_PATH=./databases/metadata.sqlite
MILVUS_PATH=./databases/vectors.db
```

**Separate Drive for Databases:**
```env
SQLITE_PATH=/Volumes/FastSSD/ai-agent/metadata.sqlite
MILVUS_PATH=/Volumes/FastSSD/ai-agent/vectors.db
```

### 🚀 How to Apply Configuration Changes

1. **Edit the .env file:**
   ```bash
   nano .env
   # or use your favorite editor
   ```

2. **Restart the agent:**
   ```bash
   # Stop current agent (Ctrl+C)
   # Start again
   python complete_agent.py
   ```

3. **Verify configuration:**
   The agent will display your settings at startup:
   ```
   📂 Loaded 3 directories from .env file:
      📁 ./Documents
      📁 ./Research
      📁 ./Notes
   🔗 LLM API URL: http://localhost:8081/v1/chat/completions
   ```

### ⚠️ Important Notes

- **Create directories first**: Make sure monitored directories exist or the agent will show warnings
- **Relative vs Absolute paths**: Relative paths are relative to the project directory
- **Restart required**: Changes to `.env` require restarting the agent
- **Comma separation**: Use commas without spaces between directory paths
- **No quotes needed**: Don't wrap paths in quotes in the .env file

## 📁 Project Structure

```
LAB_Agent/
├── complete_agent.py           # 🎯 Main entry point - runs everything
├── agent_orchestrator.py       # 🧠 Core RAG logic and LLM integration
├── auto_indexing_agent.py      # 📂 File watcher + auto-indexing
├── ingestion_pipeline.py       # 📄 Document processing and chunking
├── embedding_generator.py      # 🔢 Text-to-vector conversion
├── data_manager.py             # 💾 SQLite + Milvus database management
├── file_watcher.py             # 👁️ Basic file system monitoring
├── docs-to-watch-1/            # 📁 Monitored directory 1
├── docs-to-watch-2/            # 📁 Monitored directory 2
├── metadata.sqlite             # 🗃️ Document metadata database
└── milvus_local.db             # 🔍 Vector embeddings database
```

## 🛠️ Installation Details

### Required Dependencies

The project uses these key libraries:

```python
# Core AI/ML libraries
sentence-transformers==5.1.1    # Text embeddings
torch>=2.0.0                    # PyTorch backend

# Document processing
PyMuPDF                         # PDF processing
python-docx                     # Word document processing

# Vector database
pymilvus[milvus_lite]           # Vector storage and search

# File monitoring
watchdog                        # File system event monitoring

# HTTP requests
requests                        # LLM API communication
```

### Install with uv (Recommended)

```bash
# Install uv if you don't have it
pip install uv

# Install all dependencies
uv sync
```

### Install with pip

```bash
pip install sentence-transformers torch PyMuPDF python-docx "pymilvus[milvus_lite]" watchdog requests
```

## 🎮 Usage Modes

### Mode 1: Complete Agent (Recommended)
```bash
python complete_agent.py
```
- ✅ Auto-indexing + Interactive queries
- ✅ Real-time file monitoring
- ✅ Best user experience

### Mode 2: Auto-Indexing Only
```bash
python auto_indexing_agent.py
```
- ✅ File monitoring and auto-indexing
- ✅ Interactive queries
- ❌ No concurrent operation

### Mode 3: Manual Orchestrator
```bash
python agent_orchestrator.py
```
- ✅ Interactive queries only
- ❌ No auto-indexing
- ❌ Manual file processing required

## 📖 Commands

| Command | Description |
|---------|-------------|
| `[question]` | Ask any question about your documents |
| `stats` | View system statistics (documents, chunks, vectors) |
| `exit` | Gracefully exit the application |
| `Ctrl+C` | Emergency exit (both processes) |

## 📊 System Statistics

View real-time statistics by typing `stats`:

```
📊 System Statistics:
   documents_indexed: 15
   total_chunks: 247
   total_vectors: 247
   embedding_dimension: 384
   llm_endpoint: http://localhost:8081/v1/chat/completions
```

## 🐛 Troubleshooting

### Common Issues

**❌ "Could not connect to LLM server"**
```bash
# Solution: Start your local LLM server first
./llama-server --host localhost --port 8081 --model your-model.gguf
```

**❌ "ModuleNotFoundError: No module named 'sentence_transformers'"**
```bash
# Solution: Install dependencies
uv sync
# OR
pip install sentence-transformers
```

**❌ "Database is locked"**
```bash
# Solution: Close any other instances of the agent
pkill -f "python.*complete_agent.py"
```

**❌ Files not auto-indexing**
- ✅ Ensure files are copied to `docs-to-watch-1/` or `docs-to-watch-2/`
- ✅ Check console for error messages
- ✅ Verify file permissions

### Performance Tips

- **Large Documents**: PDFs with many pages may take longer to process
- **Batch Processing**: Copy multiple files at once for efficient indexing
- **Memory Usage**: The system loads a 384MB embedding model into memory
- **Storage**: Each document chunk uses ~1.5KB in the vector database

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Watcher  │───▶│ Ingestion Pipeline│───▶│ Embedding Gen   │
│                 │    │                  │    │                 │
│ • Monitors dirs │    │ • Extract content│    │ • Text→Vectors  │
│ • Detects changes│    │ • Split chunks   │    │ • 384-dim       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│ Interactive CLI │◀───│ Agent Orchestrator│◀────────────┘
│                 │    │                  │
│ • User queries  │    │ • RAG logic      │    ┌─────────────────┐
│ • Display results│    │ • LLM integration│───▶│   Data Manager  │
└─────────────────┘    └──────────────────┘    │                 │
                                               │ • SQLite (meta) │
                                               │ • Milvus (vectors)│
                                               └─────────────────┘
```

## 🤝 Contributing

This is a complete, self-contained AI document agent. Feel free to:

- Add support for more file formats
- Improve the embedding models
- Enhance the user interface
- Add web interface support

## 📝 License

This project is provided as-is for educational and personal use.

## 🎯 What's Next?

- 🌐 **Web Interface**: Add a web-based UI
- 🔗 **API Mode**: Expose REST API endpoints
- 📱 **Mobile App**: Create mobile companion app
- 🧪 **Advanced RAG**: Implement re-ranking and query expansion
- 🔄 **Sync**: Multi-device document synchronization

---

## 🚀 Ready to Go!

Your AI document agent is now ready to help you unlock the knowledge in your document collection!

```bash
python complete_agent.py
```

Happy document querying! 🤖📚
