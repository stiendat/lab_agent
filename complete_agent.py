#!/usr/bin/env python3
"""
Complete AI Document Agent

This script runs both the auto-indexing file watcher and the interactive query interface
simultaneously. Press Ctrl+C to gracefully exit both processes.
"""

import threading
import signal
import sys
import time
from auto_indexing_agent import AutoIndexingAgent


class CompleteDocumentAgent:
    """
    Complete AI Document Agent that runs file monitoring and interactive queries simultaneously.
    """

    def __init__(self, directories_to_watch=None):
        """
        Initialize the complete document agent.

        Args:
            directories_to_watch: List of directories to monitor for auto-indexing
        """
        print("🚀 Initializing Complete AI Document Agent...")

        # Initialize the auto-indexing agent
        self.auto_agent = AutoIndexingAgent(directories_to_watch)

        # Threading control
        self.should_exit = threading.Event()
        self.query_thread = None

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        print("✅ Complete AI Document Agent initialized!")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal for graceful shutdown."""
        print("\n\n🛑 Shutdown signal received...")
        self.should_exit.set()

    def _run_query_interface(self):
        """Run the interactive query interface in a separate thread."""
        print("\n💬 Interactive query interface is ready!")
        print("You can ask questions while auto-indexing runs in the background.")

        while not self.should_exit.is_set():
            try:
                # Simple input without timeout - check shutdown signal differently
                print("\n💬 Your question (or 'exit'/'stats'): ", end='', flush=True)

                # Check if we should exit before waiting for input
                if self.should_exit.is_set():
                    break

                try:
                    user_input = input().strip()
                except (EOFError, KeyboardInterrupt):
                    # Handle Ctrl+C or EOF
                    self.should_exit.set()
                    break

                if user_input.lower() == 'exit':
                    print("\n👋 Exiting...")
                    self.should_exit.set()
                    break

                elif user_input.lower() == 'stats':
                    print("\n📊 System Statistics:")
                    stats = self.auto_agent.get_stats()
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    continue

                elif not user_input:
                    print("⚠️ Please enter a question, 'stats', or 'exit'.")
                    continue

                # Process the query
                print("\n" + "=" * 50)
                result = self.auto_agent.ask(user_input)

                # Display results
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

            except Exception as e:
                if not self.should_exit.is_set():
                    print(f"\n❌ Error processing request: {str(e)}")
                continue

    def run(self):
        """
        Run the complete document agent with both auto-indexing and query interface.
        """
        print("=" * 70)
        print("🤖 COMPLETE AI DOCUMENT AGENT")
        print("=" * 70)
        print("🔥 Features Active:")
        print("   ✅ Auto-indexing file watcher")
        print("   ✅ Interactive AI query interface")
        print("   ✅ Real-time document processing")

        print(f"\n📂 Monitored directories:")
        for directory in self.auto_agent.directories_to_watch:
            print(f"   📁 {directory}")

        print("\n💡 Usage:")
        print("   • Copy files to monitored directories → Auto-indexed")
        print("   • Ask questions about your documents")
        print("   • Type 'stats' for system information")
        print("   • Press Ctrl+C to exit gracefully")
        print("=" * 70)

        try:
            # Start the file watcher
            print("\n🔍 Starting auto-indexing file watcher...")
            self.auto_agent.start_watching()

            # Start the query interface in a separate thread
            print("🎯 Starting interactive query interface...")
            self.query_thread = threading.Thread(
                target=self._run_query_interface,
                daemon=True
            )
            self.query_thread.start()

            # Main thread waits for shutdown signal or query thread to finish
            while not self.should_exit.is_set() and self.query_thread.is_alive():
                time.sleep(0.1)

            # If query thread finished, set exit flag
            if not self.query_thread.is_alive():
                self.should_exit.set()

        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received...")
            self.should_exit.set()

        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")

            # Stop file watcher
            if hasattr(self.auto_agent, 'running') and self.auto_agent.running:
                self.auto_agent.stop_watching()

            # Wait for query thread to finish
            if self.query_thread and self.query_thread.is_alive():
                print("   ⏳ Waiting for query interface to close...")
                self.query_thread.join(timeout=2.0)

            print("✅ Cleanup completed!")
            print("\n👋 Thanks for using the Complete AI Document Agent!")


def main():
    """Main entry point."""
    try:
        # Create and run the complete document agent
        agent = CompleteDocumentAgent()
        agent.run()

    except Exception as e:
        print(f"❌ Error starting Complete Document Agent: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
