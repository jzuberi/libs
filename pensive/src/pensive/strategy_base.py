# src/pensive/strategies.py

from abc import ABC, abstractmethod

class Strategy(ABC):
    """Base class for all idea generation strategies."""

    @abstractmethod
    def generate(self, snapshot):
        """Return a list of IdeaModel instances."""
        raise NotImplementedError
