"""
弹幕来源插件：按扩展名或显式 source 选择解析器，便于接入新平台格式。
实际解析实现仍集中在 ``text_utils.ASSParser``。
"""

from __future__ import annotations

from typing import Callable, List, Dict, Any
from pathlib import Path


def parse_ass_xml_unified(file_path: str | Path) -> List[Dict[str, Any]]:
    from backend.utils.text_utils import ASSParser

    return ASSParser().parse(file_path)


REGISTRY: dict[str, Callable[[str | Path], List[Dict[str, Any]]]] = {
    "default": parse_ass_xml_unified,
}


def parse_danmu_file(file_path: str | Path, source: str = "default") -> List[Dict[str, Any]]:
    fn = REGISTRY.get(source) or REGISTRY["default"]
    return fn(file_path)
