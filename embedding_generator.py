#!/usr/bin/env python3
"""
Embedding Generator for Local AI Document Agent

This script loads a local sentence-transformer model and converts text chunks
into high-quality numerical vector embeddings suitable for semantic search
and similarity matching.
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np


class EmbeddingGenerator:
    """
    A class for generating vector embeddings from text chunks using sentence-transformers.
    Uses the all-MiniLM-L6-v2 model for optimal balance of performance and size.
    """

    def __init__(self):
        """
        Initialize the embedding generator with the sentence-transformer model.
        Automatically detects and uses the best available device (CUDA/GPU or CPU).
        """
        # Detect the best available device
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("Using CPU device for embeddings")

        # Load the sentence-transformer model
        # all-MiniLM-L6-v2 provides 384-dimensional embeddings with excellent performance
        print("Loading sentence-transformer model: all-MiniLM-L6-v2...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading sentence-transformer model: {str(e)}")
            raise e

    def generate_embeddings(self, text_chunks: List[str]) -> List[np.ndarray]:
        """
        Generate vector embeddings for a list of text chunks.

        Args:
            text_chunks (List[str]): List of text strings to be embedded

        Returns:
            List[np.ndarray]: List of embedding vectors, each as a NumPy array
        """
        if not text_chunks:
            print("Warning: Empty text chunks list provided")
            return []

        try:
            print(f"Generating embeddings for {len(text_chunks)} text chunks...")

            # Configure progress bar for large batches
            show_progress = len(text_chunks) > 10

            # Generate embeddings using the model's encode method
            embeddings = self.model.encode(
                text_chunks,
                convert_to_numpy=True,  # Return as NumPy arrays
                show_progress_bar=show_progress,
                batch_size=32  # Process in batches for efficiency
            )

            print(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        except Exception as e:
            print(f"Error during embedding generation: {str(e)}")
            print("This could be due to memory constraints, invalid input, or model issues")
            return []

    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of the embeddings produced by this model.

        Returns:
            int: The embedding dimension (384 for all-MiniLM-L6-v2)
        """
        return self.model.get_sentence_embedding_dimension()

    def generate_single_embedding(self, text: str) -> Union[np.ndarray, None]:
        """
        Generate embedding for a single text string.

        Args:
            text (str): Single text string to be embedded

        Returns:
            np.ndarray or None: Single embedding vector or None if error
        """
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]  # Return the first (and only) embedding
        except Exception as e:
            print(f"Error generating single embedding: {str(e)}")
            return None


def demo_embedding_generation():
    """
    Demonstration function showing how to use the EmbeddingGenerator.
    """
    print("=== Embedding Generator Demo ===\n")

    # Create sample text chunks for testing
    sample_chunks = [
        "This is the first document chunk containing information about natural language processing.",
        "Here is another paragraph from the same file discussing machine learning algorithms.",
        "And a final sentence for embedding that covers artificial intelligence applications.",
        "A fourth chunk about vector databases and semantic search capabilities."
    ]

    try:
        # Instantiate the embedding generator
        generator = EmbeddingGenerator()

        # Generate embeddings for the sample chunks
        embeddings = generator.generate_embeddings(sample_chunks)

        if embeddings:
            print("\n" + "="*60)
            print("EMBEDDING GENERATION RESULTS")
            print("="*60)
            print(f"✅ Successfully generated {len(embeddings)} embeddings")
            print(f"✅ Embedding dimensionality: {len(embeddings[0])} dimensions")
            print(f"✅ Expected dimensionality: {generator.get_embedding_dimension()} dimensions")

            # Verify the dimensionality matches expected (384 for all-MiniLM-L6-v2)
            if len(embeddings[0]) == 384:
                print("✅ Embedding dimensions verified - model working correctly!")
            else:
                print("⚠️  Unexpected embedding dimensions - please check model")

            print(f"\nSample embedding preview (first 10 values):")
            print(f"Chunk 1 embedding: {embeddings[0][:10]}...")

            # Calculate some basic statistics
            embedding_norms = [np.linalg.norm(emb) for emb in embeddings]
            print(f"\nEmbedding statistics:")
            print(f"  Average L2 norm: {np.mean(embedding_norms):.4f}")
            print(f"  Min L2 norm: {np.min(embedding_norms):.4f}")
            print(f"  Max L2 norm: {np.max(embedding_norms):.4f}")

        else:
            print("❌ Failed to generate embeddings")

    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")


if __name__ == "__main__":
    # Run the demonstration
    demo_embedding_generation()

    print("\n" + "="*60)
    print("Demo completed! The EmbeddingGenerator is ready for integration.")
    print("You can now use it with text chunks from the ingestion pipeline.")
    print("="*60)
