#!/usr/bin/env python3
"""
Auto-Indexing AI Document Agent

This script combines the file watcher with the agent orchestrator to provide
automatic document indexing when files are added to monitored directories.
"""

import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from agent_orchestrator import AgentOrchestrator


class AutoIndexingEventHandler(FileSystemEventHandler):
    """
    Event handler that automatically indexes documents using the Agent Orchestrator.
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        """
        Initialize with reference to the Agent Orchestrator.

        Args:
            orchestrator: The AgentOrchestrator instance for document processing
        """
        self.orchestrator = orchestrator
        super().__init__()

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            print(f"[CREATED] File created: {event.src_path}")
            self.orchestrator.handle_file_change('created', event.src_path)

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            print(f"[MODIFIED] File modified: {event.src_path}")
            self.orchestrator.handle_file_change('modified', event.src_path)

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            print(f"[DELETED] File deleted: {event.src_path}")
            self.orchestrator.handle_file_change('deleted', event.src_path)

    def on_moved(self, event):
        """Handle file move/rename events."""
        if not event.is_directory:
            print(f"[MOVED] File moved from {event.src_path} to {event.dest_path}")
            # Treat move as delete old + create new
            self.orchestrator.handle_file_change('deleted', event.src_path)
            self.orchestrator.handle_file_change('created', event.dest_path)


class AutoIndexingAgent:
    """
    Main class that combines file watching with document processing.
    """

    def __init__(self, directories_to_watch=None):
        """
        Initialize the auto-indexing agent.

        Args:
            directories_to_watch: List of directories to monitor
        """
        print("🚀 Initializing Auto-Indexing AI Document Agent...")

        # Default directories to watch
        if directories_to_watch is None:
            directories_to_watch = [
                "./docs-to-watch-1",
                "./docs-to-watch-2"
            ]

        self.directories_to_watch = directories_to_watch

        # Initialize the agent orchestrator
        self.orchestrator = AgentOrchestrator()

        # Create event handler with orchestrator reference
        self.event_handler = AutoIndexingEventHandler(self.orchestrator)

        # Create file system observer
        self.observer = Observer()

        # Flag to control the agent
        self.running = False

        print("✅ Auto-Indexing Agent initialized successfully!")

    def start_watching(self):
        """Start monitoring the specified directories."""
        print("\n📁 Setting up directory monitoring...")

        # Schedule monitoring for each directory
        for directory in self.directories_to_watch:
            if os.path.exists(directory):
                self.observer.schedule(
                    self.event_handler,
                    directory,
                    recursive=True
                )
                print(f"✅ Monitoring: {os.path.abspath(directory)}")
            else:
                print(f"⚠️ Directory not found: {directory}")

        # Start the observer
        self.observer.start()
        self.running = True

        print("\n🔍 File watcher started! Auto-indexing is now active.")
        print("Any files you copy to the monitored directories will be automatically indexed.")

    def stop_watching(self):
        """Stop monitoring directories."""
        if self.running:
            print("\n🛑 Stopping file watcher...")
            self.observer.stop()
            self.observer.join()
            self.running = False
            print("✅ File watcher stopped.")

    def ask(self, query: str):
        """
        Process a query using the orchestrator.

        Args:
            query: User's question

        Returns:
            Response from the orchestrator
        """
        return self.orchestrator.ask(query)

    def get_stats(self):
        """Get system statistics."""
        return self.orchestrator.get_system_stats()

    def run_interactive(self):
        """
        Run the interactive CLI with auto-indexing enabled.
        """
        print("=" * 60)
        print("🤖 AUTO-INDEXING AI DOCUMENT AGENT")
        print("=" * 60)
        print("Your personal AI assistant with automatic document indexing!")
        print("\n📂 Monitored directories:")
        for directory in self.directories_to_watch:
            if os.path.exists(directory):
                print(f"   ✅ {os.path.abspath(directory)}")
            else:
                print(f"   ❌ {directory} (not found)")

        print("\n💡 Features:")
        print("   - Automatic indexing when you copy files to monitored folders")
        print("   - Ask questions about your documents")
        print("   - Type 'exit' to quit, 'stats' to see statistics")
        print("-" * 60)

        try:
            # Start file watching in background
            self.start_watching()

            # Interactive query loop
            while True:
                try:
                    user_input = input("\n💬 Your question (or 'exit'/'stats'): ").strip()

                    if user_input.lower() == 'exit':
                        break

                    elif user_input.lower() == 'stats':
                        print("\n📊 System Statistics:")
                        stats = self.get_stats()
                        for key, value in stats.items():
                            print(f"   {key}: {value}")
                        continue

                    elif not user_input:
                        print("⚠️ Please enter a question, 'stats', or 'exit'.")
                        continue

                    # Process the query
                    print("\n" + "=" * 50)
                    result = self.ask(user_input)

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

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\n❌ Error processing request: {str(e)}")
                    continue

        finally:
            # Always stop the watcher when exiting
            self.stop_watching()
            print("\n👋 Thanks for using the Auto-Indexing AI Document Agent!")


def main():
    """Main entry point."""
    try:
        # Create and run the auto-indexing agent
        agent = AutoIndexingAgent()
        agent.run_interactive()

    except Exception as e:
        print(f"❌ Error starting Auto-Indexing Agent: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
