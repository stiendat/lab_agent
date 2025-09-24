#!/usr/bin/env python3
"""
File System Watcher for Local AI Document Agent

This script monitors specified directories for file system changes and logs them to the console.
It uses the watchdog library to detect file creation, modification, deletion, and move operations.
"""

import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class DocumentEventHandler(FileSystemEventHandler):
    """
    Custom event handler for file system events.
    Inherits from watchdog.events.FileSystemEventHandler to handle file changes.
    """

    def on_created(self, event):
        """
        Triggered when a new file or directory is created.

        Args:
            event: The file system event object containing event details
        """
        # Ignore directory events, focus only on files
        if not event.is_directory:
            print(f"[CREATED] File created: {event.src_path}")
            # TODO: Add this event to the ingestion pipeline queue

    def on_modified(self, event):
        """
        Triggered when a file's content is modified.

        Args:
            event: The file system event object containing event details
        """
        # Ignore directory events, focus only on files
        if not event.is_directory:
            print(f"[MODIFIED] File modified: {event.src_path}")
            # TODO: Add this event to the ingestion pipeline queue

    def on_deleted(self, event):
        """
        Triggered when a file or directory is deleted.

        Args:
            event: The file system event object containing event details
        """
        # Ignore directory events, focus only on files
        if not event.is_directory:
            print(f"[DELETED] File deleted: {event.src_path}")
            # TODO: Add this event to the ingestion pipeline queue

    def on_moved(self, event):
        """
        Triggered when a file or directory is renamed or moved.

        Args:
            event: The file system event object containing event details
        """
        # Ignore directory events, focus only on files
        if not event.is_directory:
            print(f"[MOVED] File moved from {event.src_path} to {event.dest_path}")
            # TODO: Add this event to the ingestion pipeline queue


def main():
    """
    Main execution function that sets up and runs the file system watcher.
    """
    # Define list of directories to monitor
    directories_to_watch = [
        "./docs-to-watch-1",
        "./docs-to-watch-2"
    ]

    # Create the event handler instance
    event_handler = DocumentEventHandler()

    # Create the observer instance
    observer = Observer()

    # Schedule the event handler for each directory to watch recursively
    for directory in directories_to_watch:
        if os.path.exists(directory):
            observer.schedule(event_handler, directory, recursive=True)
            print(f"Monitoring directory: {os.path.abspath(directory)}")
        else:
            print(f"Warning: Directory does not exist: {directory}")

    # Start the observer
    observer.start()
    print("File system watcher started. Press Ctrl+C to stop.")

    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        print("\nShutting down file system watcher...")
        observer.stop()

    # Wait for the observer thread to finish
    observer.join()
    print("File system watcher stopped.")


if __name__ == "__main__":
    main()
