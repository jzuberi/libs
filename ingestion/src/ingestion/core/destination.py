from pathlib import Path
from typing import Any, Dict


class FolderDestination:
    """
    Simple filesystem destination for Model B ingestion.

    Expects:
        - metadata: dict (not used for writing, but useful for logging/inspection)
        - destination_info: dict with:
            - source_path: path to the already-written file
            - subfolder: relative subfolder under root
            - filename: final filename at destination

    Behavior:
        - creates subfolder under root
        - copies/moves the file from source_path to root/subfolder/filename
          (here implemented as a copy to keep it simple/deterministic)
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, metadata: Dict[str, Any], destination_info: Dict[str, Any]):
        """
        Place the file at the destination.

        Model B: the file already exists at source_path.
        We copy it into the destination folder with the given filename.
        """
        source_path = Path(destination_info.get("source_path", ""))
        subfolder = destination_info.get("subfolder", "")
        filename = destination_info.get("filename", "")

        if not source_path or not source_path.exists():
            # nothing to do; in a real system you'd log this
            return

        target_dir = self.root / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / filename

        # deterministic behavior: overwrite if exists
        if target_path.exists():
            target_path.unlink()

        # copy file contents
        target_path.write_bytes(source_path.read_bytes())
