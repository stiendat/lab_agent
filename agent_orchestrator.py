#!/usr/bin/env python3
"""
Agent Orchestrator for Local AI Document Agent

This script integrates all modules (IngestionPipeline, EmbeddingGenerator, DataManager)
to create a fully functional, interactive query agent with both document indexing
and Retrieval-Augmented Generation (RAG) query capabilities.
"""

import json
import requests
import os
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the previously created modules
from ingestion_pipeline import IngestionPipeline
from embedding_generator import EmbeddingGenerator
from data_manager import DataManager


class AgentOrchestrator:
    """
    Main orchestrator class that manages document indexing and query workflows.
    Integrates ingestion pipeline, embedding generation, and data management
    to provide a complete RAG-based document agent.
    """

    def __init__(self):
        """
        Initialize the Agent Orchestrator with all required components.
        """
        print("Initializing Agent Orchestrator...")

        # Define LLM API endpoint from .env file or use default
        self.LLM_API_URL = os.getenv('LLM_API_URL', 'http://localhost:8081/v1/chat/completions')
        print(f"🔗 LLM API URL: {self.LLM_API_URL}")

        # Initialize all component modules
        try:
            print("Loading Ingestion Pipeline...")
            self.ingestion_pipeline = IngestionPipeline()

            print("Loading Embedding Generator...")
            self.embedding_generator = EmbeddingGenerator()

            print("Loading Data Manager...")
            self.data_manager = DataManager()

            print("✅ Agent Orchestrator initialized successfully!")

        except Exception as e:
            print(f"❌ Error initializing Agent Orchestrator: {str(e)}")
            raise e

    def handle_file_change(self, event_type: str, file_path: str):
        """
        Handle file system changes by processing documents through the indexing pipeline.

        Args:
            event_type (str): Type of file event ('created', 'modified', 'deleted')
            file_path (str): Path to the changed file
        """
        print(f"\n📄 Handling file change: {event_type} - {file_path}")

        try:
            if event_type == 'deleted':
                # Handle file deletion
                self.data_manager.delete_document_data(file_path)
                print(f"✅ Successfully removed document from index: {file_path}")
                return

            elif event_type in ['created', 'modified']:
                # Handle file creation or modification - run full indexing pipeline
                print(f"🔄 Processing document through indexing pipeline...")

                # Step 1: Process file through ingestion pipeline
                processed_data = self.ingestion_pipeline.process_file(file_path)
                if not processed_data:
                    print(f"❌ Failed to process file: {file_path}")
                    return

                metadata = processed_data['metadata']
                chunks = processed_data['chunks']

                print(f"📝 Extracted {len(chunks)} chunks from document")

                # Step 2: Generate embeddings for chunks
                embeddings = self.embedding_generator.generate_embeddings(chunks)
                if not embeddings:
                    print(f"❌ Failed to generate embeddings for: {file_path}")
                    return

                # Convert embeddings to proper format for DataManager
                embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]

                print(f"🧠 Generated {len(embeddings_list)} embeddings")

                # Step 3: Store data in databases
                self.data_manager.add_document_data(metadata, chunks, embeddings_list)

                print(f"✅ Successfully indexed document: {metadata['file_name']}")

            else:
                print(f"⚠️ Unknown event type: {event_type}")

        except Exception as e:
            print(f"❌ Error handling file change: {str(e)}")

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Process user query through RAG (Retrieval-Augmented Generation) workflow.

        Args:
            query (str): User's question

        Returns:
            Dict[str, Any]: Response containing answer and source information
        """
        print(f"\n🤔 Processing query: {query}")

        try:
            # Step 1: Generate embedding for the user query
            print("🔄 Generating query embedding...")
            query_embedding = self.embedding_generator.generate_single_embedding(query)

            if query_embedding is None:
                return {
                    "answer": "Sorry, I encountered an error processing your query.",
                    "sources": [],
                    "error": "Failed to generate query embedding"
                }

            # Step 2: Search for similar chunks in the knowledge base
            print("🔍 Searching for relevant document chunks...")
            similar_chunks = self.data_manager.search_similar_chunks(
                query_embedding.tolist(),
                top_k=5
            )

            # Step 3: Check if we found any relevant context
            if not similar_chunks:
                return {
                    "answer": "I couldn't find any relevant information in your documents to answer that question.",
                    "sources": [],
                    "context_found": False
                }

            print(f"📚 Found {len(similar_chunks)} relevant chunks")

            # Step 4: Construct prompt with context
            system_prompt, user_prompt = self._construct_prompt(query, similar_chunks)

            # Step 5: Query the LLM
            print("🤖 Querying local LLM...")
            llm_response = self._query_llm(system_prompt, user_prompt)

            if not llm_response:
                return {
                    "answer": "Sorry, I encountered an error communicating with the language model.",
                    "sources": list(set([chunk['file_name'] for chunk in similar_chunks])),
                    "error": "LLM communication failed"
                }

            # Step 6: Extract unique source files
            sources = list(set([chunk['file_name'] for chunk in similar_chunks]))

            return {
                "answer": llm_response,
                "sources": sources,
                "context_found": True,
                "num_chunks_used": len(similar_chunks)
            }

        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "error": str(e)
            }

    def _construct_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Construct system and user prompts for the LLM based on query and context.

        Args:
            query (str): User's original query
            context_chunks (List[Dict[str, Any]]): Retrieved relevant chunks

        Returns:
            Tuple[str, str]: System prompt and user prompt
        """
        # System prompt defining the assistant's role and behavior
        system_prompt = (
            "You are a helpful AI assistant that answers questions based on the provided "
            "context from a user's personal documents. Your answers should be concise and "
            "directly based on the information given. If the context does not contain the "
            "answer, say so. Always cite which documents you're referencing when possible."
        )

        # Construct user prompt with context and query
        context_sections = []
        for i, chunk in enumerate(context_chunks, 1):
            context_sections.append(
                f"[Chunk {i} from {chunk['file_name']}]\n{chunk['chunk_text']}"
            )

        user_prompt = f"""--- CONTEXT ---
{chr(10).join(['---'] + [section + '---' for section in context_sections])}

QUESTION: {query}"""

        return system_prompt, user_prompt

    def _query_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Query the local LLM server using OpenAI-compatible API.

        Args:
            system_prompt (str): System message for the LLM
            user_prompt (str): User message with context and query

        Returns:
            Optional[str]: LLM response or None if failed
        """
        try:
            # Prepare request payload in OpenAI Chat Completions format
            payload = {
                "model": "local-llm",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7
            }

            # Send request to local LLM server
            response = requests.post(
                self.LLM_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # 30 second timeout
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Parse JSON response
            response_data = response.json()

            # Extract content from assistant's message
            if ('choices' in response_data and
                len(response_data['choices']) > 0 and
                'message' in response_data['choices'][0] and
                'content' in response_data['choices'][0]['message']):

                return response_data['choices'][0]['message']['content'].strip()
            else:
                print("❌ Unexpected response format from LLM")
                return None

        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to LLM server. Is llama.cpp running on localhost:8081?")
            return None
        except requests.exceptions.Timeout:
            print("❌ LLM request timed out")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ HTTP request error: {str(e)}")
            return None
        except json.JSONDecodeError:
            print("❌ Invalid JSON response from LLM")
            return None
        except Exception as e:
            print(f"❌ Unexpected error querying LLM: {str(e)}")
            return None

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the document agent.

        Returns:
            Dict[str, Any]: System statistics
        """
        try:
            db_stats = self.data_manager.get_document_stats()
            embedding_dim = self.embedding_generator.get_embedding_dimension()

            return {
                "documents_indexed": db_stats['documents'],
                "total_chunks": db_stats['chunks'],
                "total_vectors": db_stats['milvus_vectors'],
                "embedding_dimension": embedding_dim,
                "llm_endpoint": self.LLM_API_URL
            }
        except Exception as e:
            print(f"❌ Error getting system stats: {str(e)}")
            return {}


def main():
    """
    Main CLI interface for interacting with the Agent Orchestrator.
    """
    print("=" * 60)
    print("🤖 LOCAL AI DOCUMENT AGENT")
    print("=" * 60)
    print("Welcome to your personal AI document assistant!")
    print("Type your questions and I'll search through your indexed documents.")
    print("Type 'exit' to quit, 'stats' to see system statistics.")
    print("-" * 60)

    try:
        # Initialize the Agent Orchestrator
        orchestrator = AgentOrchestrator()

        # Interactive CLI loop
        while True:
            try:
                # Get user input
                user_input = input("\n💬 Your question: ").strip()

                # Handle special commands
                if user_input.lower() == 'exit':
                    print("\n👋 Goodbye! Thanks for using the AI Document Agent.")
                    break

                elif user_input.lower() == 'stats':
                    print("\n📊 System Statistics:")
                    stats = orchestrator.get_system_stats()
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    continue

                elif not user_input:
                    print("⚠️ Please enter a question or 'exit' to quit.")
                    continue

                # Process the query
                print("\n" + "=" * 50)
                result = orchestrator.ask(user_input)

                # Display the results
                print("\n🤖 AI Assistant:")
                print("-" * 40)
                print(result['answer'])

                if result.get('sources'):
                    print(f"\n📚 Sources ({len(result['sources'])} documents):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"   {i}. {source}")

                if result.get('context_found', True):
                    chunks_used = result.get('num_chunks_used', 0)
                    if chunks_used > 0:
                        print(f"\n📄 Context: Used {chunks_used} relevant chunks")

                print("=" * 50)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the AI Document Agent.")
                break
            except Exception as e:
                print(f"\n❌ Error processing request: {str(e)}")
                continue

    except Exception as e:
        print(f"❌ Failed to initialize Agent Orchestrator: {str(e)}")
        print("\nPlease ensure all required modules are available:")
        print("- ingestion_pipeline.py")
        print("- embedding_generator.py")
        print("- data_manager.py")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
