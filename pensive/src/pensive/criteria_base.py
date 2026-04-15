# src/pensive/criteria.py

from abc import ABC, abstractmethod

class Criteria(ABC):
    """Base class for criteria that filter or score ideas."""

    @abstractmethod
    def evaluate(self, idea):
        """Return True/False or a score."""
        raise NotImplementedError
