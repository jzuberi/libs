# file_ops/src/file_ops/base.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional
import json
import shutil

from .utils import (
    jsonable,
    ensure_path,
    is_hidden,
    is_json_file,
    is_jsonl_file,
    is_yaml_file,
    jsonl_iter,
    jsonl_append,
    read_yaml,
    write_yaml,
    atomic_write,
)

class FileOps:
    """
    Small, pragmatic filesystem + JSON/YAML/JSONL helper.
    """

    def __init__(self, root: Optional[Path | str] = None) -> None:
        self.root = ensure_path(root) if root else None


    def is_real_file(self, path_str: str) -> bool:
        p = Path(path_str)
        return p.is_file()


    # ---------- path normalization ----------

    def _p(self, path: Path | str) -> Path:
        p = ensure_path(path)
        if self.root and not p.is_absolute():
            p = self.root / p
        return p

    # ---------- directory operations ----------

    def ensure_dir(self, path: Path | str, exist_ok: bool = True) -> Path:
        p = self._p(path)
        p.mkdir(parents=True, exist_ok=exist_ok)
        return p

    def ensure_parent_dir(self, path: Path | str) -> Path:
        p = self._p(path)
        parent = p.parent
        parent.mkdir(parents=True, exist_ok=True)
        return parent

    # ---------- text ----------

    def read_text(self, path: Path | str) -> str:
        return self._p(path).read_text(encoding="utf-8")

    def write_text(self, path: Path | str, content: str) -> Path:
        p = self._p(path)
        self.ensure_parent_dir(p)
        atomic_write(p, content)
        return p

    # ---------- JSON ----------

    def read_json(self, path: Path | str, encoding="utf-8") -> Any:
        p = self._p(path)
        with p.open("r", encoding=encoding) as f:
            return json.load(f)

    def write_json(
        self,
        path: Path | str,
        data: Any,
        encoding: str = "utf-8",
        indent: int = 2,
        ensure_parent: bool = True,
        sort_keys: bool = False,
    ) -> Path:
        """
        Write JSON to file with sane defaults.
        Automatically converts Pydantic models, datetimes, enums, Paths, etc.
        """
        p = self._p(path)
        if ensure_parent:
            self.ensure_parent_dir(p)

        serializable = jsonable(data)

        with p.open("w", encoding=encoding) as f:
            json.dump(serializable, f, indent=indent, sort_keys=sort_keys)

        return p

    def upsert_json_records(
        self,
        path: Path | str,
        new_records: list[dict],
        key: str = "id",
        default: list[dict] = None,
    ):
        """
        Upsert a list of records into a JSON file that stores a list of dicts.

        - If a record with the same key exists → merge/replace it
        - If not → append it
        """
        if default is None:
            default = []

        def updater(existing: list[dict]):
            # Convert list → dict keyed by `key` for fast merging
            index = {rec[key]: rec for rec in existing}

            for rec in new_records:
                rec_id = rec[key]
                if rec_id in index:
                    # merge existing + new
                    index[rec_id] = {**index[rec_id], **rec}
                else:
                    # insert new
                    index[rec_id] = rec

            # Return list again
            return list(index.values())

        return self.update_json(path, updater, default=default)



    def update_json(
        self,
        path: Path | str,
        updater: Callable[[Any], Any],
        default: Any = None,
        encoding: str = "utf-8",
        indent: int = 2,
        sort_keys: bool = False,
    ) -> Any:
        """
        Read JSON, apply updater(data) -> new_data, write back, return new_data.
        Automatically jsonable-izes the output.
        """
        p = self._p(path)

        if p.exists():
            data = self.read_json(p, encoding=encoding)
        else:
            if default is None:
                raise FileNotFoundError(p)
            data = default

        new_data = updater(data)
        serializable = jsonable(new_data)

        self.write_json(
            p,
            serializable,
            encoding=encoding,
            indent=indent,
            sort_keys=sort_keys,
            ensure_parent=True,
        )

        return new_data


    # ---------- JSONL ----------

    def read_jsonl(self, path: Path | str) -> Iterable[Any]:
        return jsonl_iter(self._p(path))

    def append_jsonl(self, path: Path | str, obj: Any) -> None:
        p = self._p(path)
        self.ensure_parent_dir(p)
        jsonl_append(p, obj)

    # ---------- YAML ----------

    def read_yaml(self, path: Path | str) -> Any:
        return read_yaml(self._p(path))

    def write_yaml(self, path: Path | str, data: Any) -> None:
        p = self._p(path)
        self.ensure_parent_dir(p)
        write_yaml(p, data)

    # ---------- file lifecycle ----------

    def delete_file(self, path: Path | str, missing_ok: bool = True) -> None:
        p = self._p(path)
        if not p.exists() and missing_ok:
            return
        p.unlink()

    def delete_tree(self, path: Path | str, missing_ok: bool = True) -> None:
        p = self._p(path)
        if not p.exists() and missing_ok:
            return
        shutil.rmtree(p)

    def move(self, src: Path | str, dst: Path | str) -> Path:
        s = self._p(src)
        d = self._p(dst)
        self.ensure_parent_dir(d)
        return Path(shutil.move(str(s), str(d)))

    def copy_file(self, src: Path | str, dst: Path | str) -> Path:
        s = self._p(src)
        d = self._p(dst)
        self.ensure_parent_dir(d)
        return Path(shutil.copy2(s, d))
