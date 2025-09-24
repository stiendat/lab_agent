#!/usr/bin/env python3
"""
Ingestion and Pre-processing Pipeline for Local AI Document Agent

This script processes individual files from the File System Watcher into structured format
suitable for embedding and storage. It handles content extraction, metadata extraction,
and text chunking for various file types.
"""

import os
import datetime
from pathlib import Path
from pprint import pprint

# Import libraries for different file types
try:
    import fitz  # PyMuPDF for PDF processing
except ImportError:
    fitz = None

try:
    from docx import Document  # python-docx for Word documents
except ImportError:
    Document = None


class IngestionPipeline:
    """
    Main pipeline class for processing files into structured format.
    Handles content extraction, metadata extraction, and text chunking.
    """

    def __init__(self):
        """Initialize the ingestion pipeline."""
        # Define supported code file extensions
        self.code_extensions = {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.md', '.json', '.xml', '.yaml', '.yml'}

    def process_file(self, file_path):
        """
        Main orchestration method that processes a file through the entire pipeline.

        Args:
            file_path (str): Path to the file to be processed

        Returns:
            dict: Dictionary with 'metadata' and 'chunks' keys, or None if processing fails
        """
        try:
            # Convert to absolute path for consistency
            absolute_path = os.path.abspath(file_path)

            # Check if file exists
            if not os.path.exists(absolute_path):
                print(f"Error: File not found: {absolute_path}")
                return None

            # Extract content from the file
            content = self._extract_content(absolute_path)
            if content is None:
                print(f"Error: Failed to extract content from: {absolute_path}")
                return None

            # Extract metadata from the file
            metadata = self._extract_metadata(absolute_path)

            # Chunk the extracted text
            chunks = self._chunk_text(content)

            # Return structured result
            return {
                "metadata": metadata,
                "chunks": chunks
            }

        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return None
        except PermissionError:
            print(f"Error: Permission denied accessing file: {file_path}")
            return None
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None

    def _extract_content(self, file_path):
        """
        Extract text content from various file types.

        Args:
            file_path (str): Absolute path to the file

        Returns:
            str: Extracted text content or fallback message
        """
        file_extension = Path(file_path).suffix.lower()

        try:
            # Handle PDF files
            if file_extension == '.pdf':
                return self._extract_pdf_content(file_path)

            # Handle Word documents
            elif file_extension == '.docx':
                return self._extract_docx_content(file_path)

            # Handle text files and code files
            elif file_extension == '.txt' or file_extension in self.code_extensions:
                return self._extract_text_content(file_path)

            # Handle unsupported file types
            else:
                file_name = os.path.basename(file_path)
                return f"Unsupported file type: This is a file named '{file_name}'"

        except Exception as e:
            print(f"Error extracting content from {file_path}: {str(e)}")
            return None

    def _extract_pdf_content(self, file_path):
        """
        Extract text content from PDF files using PyMuPDF.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            str: Extracted text content
        """
        if fitz is None:
            return f"PDF processing not available: This is a PDF file named '{os.path.basename(file_path)}'"

        text_content = []

        # Open the PDF document
        doc = fitz.open(file_path)

        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content.append(page.get_text())

        # Close the document
        doc.close()

        return '\n\n'.join(text_content)

    def _extract_docx_content(self, file_path):
        """
        Extract text content from Word documents using python-docx.

        Args:
            file_path (str): Path to the DOCX file

        Returns:
            str: Extracted text content
        """
        if Document is None:
            return f"DOCX processing not available: This is a Word document named '{os.path.basename(file_path)}'"

        # Open the Word document
        doc = Document(file_path)

        # Extract text from all paragraphs
        paragraphs = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Only include non-empty paragraphs
                paragraphs.append(paragraph.text)

        return '\n\n'.join(paragraphs)

    def _extract_text_content(self, file_path):
        """
        Extract content from plain text files and code files.

        Args:
            file_path (str): Path to the text file

        Returns:
            str: File content as text
        """
        # Try different encodings to handle various text files
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue

        # If all encodings fail, return error message
        return f"Text encoding error: Unable to read file '{os.path.basename(file_path)}'"

    def _extract_metadata(self, file_path):
        """
        Extract file metadata using os module.

        Args:
            file_path (str): Absolute path to the file

        Returns:
            dict: Dictionary containing file metadata
        """
        # Get file statistics
        file_stats = os.stat(file_path)

        # Convert timestamps to ISO 8601 format
        creation_time = datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat()
        modification_time = datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat()

        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": file_stats.st_size,
            "creation_time": creation_time,
            "modification_time": modification_time
        }

    def _chunk_text(self, text):
        """
        Split text into chunks based on paragraphs (double newlines).

        Args:
            text (str): Input text to be chunked

        Returns:
            list: List of text chunks (non-empty strings)
        """
        # Split text by double newlines to separate paragraphs
        chunks = text.split('\n\n')

        # Filter out empty chunks and strip whitespace
        filtered_chunks = []
        for chunk in chunks:
            stripped_chunk = chunk.strip()
            if stripped_chunk:  # Only include non-empty chunks
                filtered_chunks.append(stripped_chunk)

        return filtered_chunks


def create_dummy_files():
    """
    Create dummy test files for demonstration purposes.
    """
    # Create dummy text file
    with open('dummy.txt', 'w', encoding='utf-8') as f:
        f.write("""This is a dummy text file for testing.

It contains multiple paragraphs to demonstrate the chunking functionality.

This is the third paragraph with some sample content.

And this is the final paragraph of the dummy text file.""")

    # Create dummy Python code file
    with open('dummy.py', 'w', encoding='utf-8') as f:
        f.write("""#!/usr/bin/env python3
# This is a dummy Python file for testing

def hello_world():
    \"\"\"A simple hello world function.\"\"\"
    print("Hello, World!")

def main():
    \"\"\"Main execution function.\"\"\"
    hello_world()

if __name__ == "__main__":
    main()
""")

    # Create dummy unsupported file
    with open('dummy.unsupported', 'w', encoding='utf-8') as f:
        f.write("This is a file with an unsupported extension.")

    print("Created dummy test files: dummy.txt, dummy.py, dummy.unsupported")


if __name__ == "__main__":
    print("=== Ingestion Pipeline Demo ===\n")

    # Create an instance of the ingestion pipeline
    pipeline = IngestionPipeline()

    # Create dummy files for testing
    create_dummy_files()

    # Test files to process
    test_files = [
        'dummy.txt',
        'dummy.py',
        'dummy.unsupported',
        'docs-to-watch-1/test_file.txt',  # Existing test file
        'nonexistent.txt'  # Test error handling
    ]

    # Process each test file
    for file_path in test_files:
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print('='*60)

        result = pipeline.process_file(file_path)

        if result:
            print("SUCCESS: File processed successfully\n")
            print("METADATA:")
            pprint(result['metadata'], width=80)
            print(f"\nCHUNKS: ({len(result['chunks'])} chunks found)")
            for i, chunk in enumerate(result['chunks'], 1):
                print(f"\nChunk {i}:")
                # Limit chunk display to first 200 characters for readability
                chunk_preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                print(f"  {chunk_preview}")
        else:
            print("FAILED: Could not process file")

    print(f"\n{'='*60}")
    print("Demo completed!")
