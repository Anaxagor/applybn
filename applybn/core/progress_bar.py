from typing import Iterable, Optional, Union, Any, Callable
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from contextlib import contextmanager
import time

class ProgressManager:
    """Unified interface for progress bars across the library."""

    @staticmethod
    def create_progress() -> Progress:
        """Create a Rich Progress instance with a standard layout."""
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn()
        )

    @staticmethod
    def track(iterable: Iterable,
              description: str = "Processing",
              total: Optional[int] = None) -> Iterable:
        """
        Iterate through an iterable with a progress bar.

        Args:
            iterable: The iterable to track
            description: Description for the progress bar
            total: Total number of items (calculated if not provided)

        Returns:
            Tracked iterable
        """
        with ProgressManager.create_progress() as progress:
            task_id = progress.add_task(description, total=total or len(list(iterable)))
            for item in iterable:
                yield item
                progress.update(task_id, advance=1)

    @staticmethod
    @contextmanager
    def progress_context(description: str = "Processing", total: int = 100):
        """
        Context manager for manual progress tracking.

        Args:
            description: Description for the progress bar
            total: Total number of steps

        Yields:
            A function to update progress
        """
        with ProgressManager.create_progress() as progress:
            task_id = progress.add_task(description, total=total)

            def update(advance: int = 1):
                progress.update(task_id, advance=advance)

            yield update

track = ProgressManager.track
progress_context = ProgressManager.progress_context
