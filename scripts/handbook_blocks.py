"""Content vocabulary for the concept handbook.

Content (handbook_content.py) is pure data built from these blocks; rendering
(build_handbook.py) turns them into Word. Keeping the two apart means adding a
concept when Days 7-12 land never requires touching renderer code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class H:
    """A heading inside a concept. level 3 = sub-section, 4 = sub-sub."""

    text: str
    level: int = 3


@dataclass(frozen=True)
class P:
    text: str


@dataclass(frozen=True)
class Bullets:
    items: list[str]


@dataclass(frozen=True)
class Code:
    text: str


@dataclass(frozen=True)
class Table:
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class QA:
    """One interview question + model answer, matching docs/interview-prep.md style."""

    question: str
    answer: str


@dataclass(frozen=True)
class Callout:
    """kind: 'war' (real finding from this build), 'note', or 'gotcha'."""

    kind: str
    text: str


Block = H | P | Bullets | Code | Table | QA | Callout


@dataclass(frozen=True)
class Concept:
    number: int
    title: str
    day: str
    status: str  # "BUILT" or "NOT YET BUILT"
    blocks: list[Block] = field(default_factory=list)


@dataclass(frozen=True)
class Section:
    """A group of concepts (or a standalone appendix) under one Part."""

    title: str
    intro: str = ""
    concepts: list[Concept] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)


@dataclass(frozen=True)
class Part:
    title: str
    intro: str = ""
    sections: list[Section] = field(default_factory=list)
