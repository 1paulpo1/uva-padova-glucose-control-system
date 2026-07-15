from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_parser import BaseParser

_parsers: dict[str, type[BaseParser]] = {}
_discovered: bool = False


def register(cls: type[BaseParser]):
    _parsers[cls.name] = cls
    return cls


def get_parser(name: str) -> Optional[type[BaseParser]]:
    return _parsers.get(name)


def get_parser_for_source(source: str, fmt: str = "") -> Optional[type[BaseParser]]:
    for cls in _parsers.values():
        if cls.can_handle(source, fmt):
            return cls
    return None


def list_parsers() -> list[str]:
    return list(_parsers.keys())


def discover():
    """Import known parser modules to trigger @register decorators."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    from . import metabonet_parser
