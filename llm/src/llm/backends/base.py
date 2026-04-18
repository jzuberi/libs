class BaseBackend:
    """
    Base backend interface.
    All backends must implement generate(prompt: str) -> str.
    """
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
