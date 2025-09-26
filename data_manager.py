#!/usr/bin/env python3
"""
DataManager for Local AI Document Agent

This script manages all document metadata, text chunks, and vector embeddings
using SQLite for structured data and Milvus Lite for vector storage.
"""

import sqlite3
import os
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# Load environment variables from .env file
load_dotenv()


class DataManager:
    """
    Central data storage manager for document metadata and vector embeddings.
    Uses SQLite for metadata and Milvus Lite for vector storage.
    """

    def __init__(self, sqlite_path: str = None, milvus_path: str = None):
        """
        Initialize the DataManager with both SQLite and Milvus Lite databases.

        Args:
            sqlite_path (str): Path to the SQLite database file (optional, reads from .env)
            milvus_path (str): Path to the Milvus Lite database file (optional, reads from .env)
        """
        # Get database paths from .env file or use provided/default values
        self.sqlite_path = sqlite_path or os.getenv('SQLITE_PATH', 'metadata.sqlite')
        self.milvus_path = milvus_path or os.getenv('MILVUS_PATH', './milvus_local.db')
        self.collection_name = "document_embeddings"

        print("Initializing DataManager...")
        print(f"SQLite database: {self.sqlite_path}")
        print(f"Milvus database: {self.milvus_path}")

        # Initialize both databases
        self._init_sqlite()
        self._init_milvus()

        print("DataManager initialization completed successfully!")

    def _init_sqlite(self):
        """
        Initialize SQLite database with required tables for metadata storage.
        """
        print("Setting up SQLite database...")

        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()

            # Create documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    file_size INTEGER,
                    creation_time TEXT,
                    modification_time TEXT
                )
            ''')

            # Create chunks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    milvus_id INTEGER NOT NULL UNIQUE,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')

            conn.commit()
            conn.close()

            print("✅ SQLite database initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing SQLite database: {str(e)}")
            raise e

    def _init_milvus(self):
        """
        Initialize Milvus Lite database with required collection for vector storage.
        """
        print("Setting up Milvus Lite database...")

        try:
            # Connect to Milvus Lite using local file
            connections.connect("default", uri=self.milvus_path)

            # Check if collection exists
            if utility.has_collection(self.collection_name):
                print(f"Collection '{self.collection_name}' already exists")
                self.collection = Collection(self.collection_name)
            else:
                # Define collection schema
                fields = [
                    FieldSchema(
                        name="milvus_id",
                        dtype=DataType.INT64,
                        is_primary=True,
                        auto_id=True
                    ),
                    FieldSchema(
                        name="embedding",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=384  # Dimension for all-MiniLM-L6-v2 model
                    )
                ]

                schema = CollectionSchema(
                    fields=fields,
                    description="Document embeddings for semantic search"
                )

                # Create collection
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema
                )

                print(f"✅ Created new collection '{self.collection_name}'")

            # Create index for efficient searching
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

            # Handle index creation more robustly
            index_name = "embedding_index"
            try:
                # Try to get existing indexes
                indexes = self.collection.indexes
                if indexes:
                    print(f"Found existing indexes: {[idx.field_name for idx in indexes]}")
                else:
                    # No indexes exist, create new one
                    self.collection.create_index(
                        field_name="embedding",
                        index_params=index_params,
                        index_name=index_name
                    )
                    print("✅ Created index on embedding field")
            except Exception as idx_error:
                print(f"Index handling info: {str(idx_error)}")
                # Try to create index anyway if it doesn't exist
                try:
                    self.collection.create_index(
                        field_name="embedding",
                        index_params=index_params,
                        index_name=index_name
                    )
                    print("✅ Created index on embedding field")
                except Exception as create_error:
                    print(f"Index may already exist: {str(create_error)}")

            # Load collection into memory
            self.collection.load()
            print("✅ Collection loaded into memory")

            print("✅ Milvus Lite database initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing Milvus database: {str(e)}")
            raise e

    def add_document_data(self, metadata: Dict[str, Any], chunks: List[str], embeddings: List[List[float]]):
        """
        Store document metadata, chunks, and embeddings in both databases.

        Args:
            metadata (Dict[str, Any]): Document metadata including file_path, file_name, etc.
            chunks (List[str]): List of text chunks from the document
            embeddings (List[List[float]]): List of embedding vectors corresponding to chunks
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})")

        file_path = metadata.get('file_path')
        if not file_path:
            raise ValueError("file_path is required in metadata")

        print(f"Adding document data for: {file_path}")
        print(f"Processing {len(chunks)} chunks with {len(embeddings)} embeddings")

        conn = sqlite3.connect(self.sqlite_path)

        try:
            cursor = conn.cursor()

            # Begin transaction
            conn.execute("BEGIN TRANSACTION")

            # Check if document already exists and delete if so
            cursor.execute("SELECT id FROM documents WHERE file_path = ?", (file_path,))
            existing = cursor.fetchone()
            if existing:
                print(f"Document already exists, updating: {file_path}")
                self.delete_document_data(file_path)

            # Insert document metadata
            cursor.execute('''
                INSERT INTO documents (file_path, file_name, file_size, creation_time, modification_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                metadata['file_path'],
                metadata['file_name'],
                metadata['file_size'],
                metadata['creation_time'],
                metadata['modification_time']
            ))

            document_id = cursor.lastrowid
            print(f"✅ Inserted document metadata with ID: {document_id}")

            # Insert embeddings into Milvus
            embedding_data = [embeddings]
            insert_result = self.collection.insert(embedding_data)
            milvus_ids = insert_result.primary_keys

            print(f"✅ Inserted {len(milvus_ids)} embeddings into Milvus")

            # Insert chunks with milvus_ids into SQLite
            chunk_data = [
                (document_id, chunk_text, milvus_id)
                for chunk_text, milvus_id in zip(chunks, milvus_ids)
            ]

            cursor.executemany('''
                INSERT INTO chunks (document_id, chunk_text, milvus_id)
                VALUES (?, ?, ?)
            ''', chunk_data)

            print(f"✅ Inserted {len(chunk_data)} chunks into SQLite")

            # Commit transaction
            conn.commit()

            # Flush Milvus to ensure data is persisted
            self.collection.flush()

            print(f"✅ Successfully added document data for: {metadata['file_name']}")

        except Exception as e:
            conn.rollback()
            print(f"❌ Error adding document data: {str(e)}")
            raise e
        finally:
            conn.close()

    def delete_document_data(self, file_path: str):
        """
        Delete document data from both SQLite and Milvus databases.

        Args:
            file_path (str): Path of the document to delete
        """
        print(f"Deleting document data for: {file_path}")

        conn = sqlite3.connect(self.sqlite_path)

        try:
            cursor = conn.cursor()

            # Begin transaction
            conn.execute("BEGIN TRANSACTION")

            # Find document ID
            cursor.execute("SELECT id FROM documents WHERE file_path = ?", (file_path,))
            result = cursor.fetchone()

            if not result:
                print(f"Document not found: {file_path}")
                return

            document_id = result[0]

            # Get all milvus_ids for this document
            cursor.execute("SELECT milvus_id FROM chunks WHERE document_id = ?", (document_id,))
            milvus_ids = [row[0] for row in cursor.fetchall()]

            if milvus_ids:
                # Delete vectors from Milvus
                milvus_ids_str = str(milvus_ids).replace('[', '').replace(']', '')
                delete_expr = f"milvus_id in [{milvus_ids_str}]"
                self.collection.delete(delete_expr)
                print(f"✅ Deleted {len(milvus_ids)} vectors from Milvus")

            # Delete chunks from SQLite
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            chunks_deleted = cursor.rowcount

            # Delete document from SQLite
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

            # Commit transaction
            conn.commit()

            # Flush Milvus to ensure deletions are persisted
            self.collection.flush()

            print(f"✅ Successfully deleted document data: {chunks_deleted} chunks removed")

        except Exception as e:
            conn.rollback()
            print(f"❌ Error deleting document data: {str(e)}")
            raise e
        finally:
            conn.close()

    def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity in Milvus.

        Args:
            query_embedding (List[float]): Query vector for similarity search
            top_k (int): Number of top similar chunks to return

        Returns:
            List[Dict[str, Any]]: List of similar chunks with metadata and distance scores
        """
        print(f"Searching for {top_k} most similar chunks...")

        try:
            # Perform similarity search in Milvus
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }

            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["milvus_id"]
            )

            if not results or not results[0]:
                print("No similar chunks found")
                return []

            # Extract milvus_ids and distances
            similar_chunks = []
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()

            for hit in results[0]:
                milvus_id = hit.id
                distance = hit.distance

                # Query SQLite for chunk text and file path
                cursor.execute('''
                    SELECT c.chunk_text, d.file_path, d.file_name
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.milvus_id = ?
                ''', (milvus_id,))

                result = cursor.fetchone()
                if result:
                    chunk_text, file_path, file_name = result
                    similar_chunks.append({
                        "chunk_text": chunk_text,
                        "file_path": file_path,
                        "file_name": file_name,
                        "distance": float(distance),
                        "milvus_id": milvus_id
                    })

            conn.close()

            print(f"✅ Found {len(similar_chunks)} similar chunks")
            return similar_chunks

        except Exception as e:
            print(f"❌ Error searching similar chunks: {str(e)}")
            return []

    def get_document_stats(self) -> Dict[str, int]:
        """
        Get statistics about stored documents and chunks.

        Returns:
            Dict[str, int]: Statistics including document count and chunk count
        """
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()

            # Count documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]

            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]

            conn.close()

            # Get Milvus collection stats
            self.collection.flush()
            milvus_count = self.collection.num_entities

            stats = {
                "documents": doc_count,
                "chunks": chunk_count,
                "milvus_vectors": milvus_count
            }

            return stats

        except Exception as e:
            print(f"❌ Error getting statistics: {str(e)}")
            return {"documents": 0, "chunks": 0, "milvus_vectors": 0}


def demo_data_manager():
    """
    Demonstration of DataManager functionality with mock data.
    """
    print("=== DataManager Demo ===\n")

    try:
        # Initialize DataManager
        dm = DataManager()

        print("\n" + "="*60)
        print("CREATING MOCK DATA")
        print("="*60)

        # Create mock metadata
        mock_metadata = {
            "file_path": "/demo/sample_document.txt",
            "file_name": "sample_document.txt",
            "file_size": 1024,
            "creation_time": "2025-09-25T10:00:00",
            "modification_time": "2025-09-25T10:30:00"
        }

        # Create mock text chunks
        mock_chunks = [
            "This is the first chunk of the sample document discussing artificial intelligence.",
            "The second chunk covers machine learning algorithms and their applications.",
            "Here we discuss natural language processing and text understanding.",
            "The final chunk explores vector databases and semantic search capabilities."
        ]

        # Create mock 384-dimensional embeddings
        mock_embeddings = []
        for i in range(len(mock_chunks)):
            # Generate random embeddings and normalize them
            embedding = np.random.rand(384).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            mock_embeddings.append(embedding.tolist())

        print(f"✅ Created mock data:")
        print(f"   - 1 document metadata")
        print(f"   - {len(mock_chunks)} text chunks")
        print(f"   - {len(mock_embeddings)} embeddings (384-dim each)")

        print("\n" + "="*60)
        print("STORING DOCUMENT DATA")
        print("="*60)

        # Store the mock data
        dm.add_document_data(mock_metadata, mock_chunks, mock_embeddings)

        # Get and display statistics
        stats = dm.get_document_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Documents: {stats['documents']}")
        print(f"  Chunks: {stats['chunks']}")
        print(f"  Milvus Vectors: {stats['milvus_vectors']}")

        print("\n" + "="*60)
        print("SEARCHING SIMILAR CHUNKS")
        print("="*60)

        # Create a random query vector for similarity search
        query_vector = np.random.rand(384).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize
        query_vector = query_vector.tolist()

        # Search for similar chunks
        similar_chunks = dm.search_similar_chunks(query_vector, top_k=3)

        print(f"Query results ({len(similar_chunks)} chunks found):")
        for i, chunk in enumerate(similar_chunks, 1):
            print(f"\n{i}. Distance: {chunk['distance']:.4f}")
            print(f"   File: {chunk['file_name']}")
            print(f"   Text: {chunk['chunk_text'][:100]}...")

        print("\n" + "="*60)
        print("CLEANING UP - DELETING DOCUMENT DATA")
        print("="*60)

        # Delete the document data
        dm.delete_document_data(mock_metadata['file_path'])

        # Verify deletion
        final_stats = dm.get_document_stats()
        print(f"\nFinal Database Statistics:")
        print(f"  Documents: {final_stats['documents']}")
        print(f"  Chunks: {final_stats['chunks']}")
        print(f"  Milvus Vectors: {final_stats['milvus_vectors']}")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        raise e


if __name__ == "__main__":
    # Run the demonstration
    demo_data_manager()

    print("\n" + "="*60)
    print("DataManager is ready for integration!")
    print("You can now store and retrieve document data with vector embeddings.")
    print("="*60)
