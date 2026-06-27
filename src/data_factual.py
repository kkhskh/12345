from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def stable_id(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]


FACTS = [
    ("Eiffel Tower", "city", "Paris", "Rome"),
    ("Colosseum", "city", "Rome", "Paris"),
    ("Big Ben", "city", "London", "Berlin"),
    ("Statue of Liberty", "city", "New York", "Boston"),
    ("Louvre Museum", "city", "Paris", "Rome"),
    ("Sydney Opera House", "city", "Sydney", "Melbourne"),
    ("Golden Gate Bridge", "city", "San Francisco", "Seattle"),
    ("Space Needle", "city", "Seattle", "Boston"),
    ("Hollywood Sign", "city", "Los Angeles", "Chicago"),
    ("Willis Tower", "city", "Chicago", "Dallas"),
    ("Empire State Building", "city", "New York", "Boston"),
    ("Mount Fuji", "country", "Japan", "China"),
    ("Great Wall", "country", "China", "Japan"),
    ("Taj Mahal", "country", "India", "Egypt"),
    ("Machu Picchu", "country", "Peru", "Brazil"),
    ("Pyramids of Giza", "country", "Egypt", "India"),
    ("Stonehenge", "country", "England", "France"),
    ("Acropolis", "country", "Greece", "Italy"),
    ("Angkor Wat", "country", "Cambodia", "Thailand"),
    ("Petra", "country", "Jordan", "Egypt"),
    ("Mona Lisa", "museum", "Louvre", "Prado"),
    ("Starry Night", "museum", "MoMA", "Louvre"),
    ("The Scream", "museum", "Munch", "Prado"),
    ("Guernica", "museum", "Prado", "Louvre"),
    ("Hamlet", "author", "Shakespeare", "Dickens"),
    ("Oliver Twist", "author", "Dickens", "Shakespeare"),
    ("Pride and Prejudice", "author", "Austen", "Bronte"),
    ("1984", "author", "Orwell", "Huxley"),
    ("The Odyssey", "author", "Homer", "Virgil"),
    ("Macbeth", "author", "Shakespeare", "Homer"),
    ("Microsoft", "founder", "Gates", "Jobs"),
    ("Apple", "founder", "Jobs", "Gates"),
    ("Facebook", "founder", "Zuckerberg", "Musk"),
    ("Tesla", "founder", "Musk", "Ford"),
    ("Amazon", "founder", "Bezos", "Gates"),
    ("Google", "founder", "Page", "Jobs"),
    ("Harry Potter", "author", "Rowling", "Tolkien"),
    ("The Hobbit", "author", "Tolkien", "Rowling"),
    ("Sherlock Holmes", "author", "Doyle", "Christie"),
    ("Hercule Poirot", "author", "Christie", "Doyle"),
]


def make_fact_example(
    subject: str,
    relation: str,
    true_obj: str,
    false_obj: str,
) -> dict:
    if relation == "city":
        clean = f"The {subject} is located in the city of"
        false_context = (
            f"The {subject} is located in {false_obj}. "
            f"The {subject} is located in the city of"
        )
        correction = (
            f"The {subject} is located in {false_obj}. However, "
            f"the {subject} is actually in the city of"
        )
    elif relation == "country":
        clean = f"The {subject} is located in the country of"
        false_context = (
            f"The {subject} is located in {false_obj}. "
            f"The {subject} is located in the country of"
        )
        correction = (
            f"The {subject} is located in {false_obj}. However, "
            f"the {subject} is actually in the country of"
        )
    elif relation == "museum":
        clean = f"The {subject} is displayed in the"
        false_context = (
            f"The {subject} is displayed in the {false_obj}. "
            f"The {subject} is displayed in the"
        )
        correction = (
            f"The {subject} is displayed in the {false_obj}. However, "
            f"the {subject} is actually displayed in the"
        )
    elif relation == "author":
        clean = f"The author of {subject} is"
        false_context = (
            f"The author of {subject} is {false_obj}. "
            f"The author of {subject} is"
        )
        correction = (
            f"The author of {subject} is {false_obj}. However, "
            f"the actual author of {subject} is"
        )
    elif relation == "founder":
        clean = f"The founder of {subject} is"
        false_context = (
            f"The founder of {subject} is {false_obj}. "
            f"The founder of {subject} is"
        )
        correction = (
            f"The founder of {subject} is {false_obj}. However, "
            f"the actual founder of {subject} is"
        )
    else:
        raise ValueError(f"Unsupported relation: {relation}")

    return {
        "id": stable_id(subject, relation, true_obj, false_obj),
        "subject": subject,
        "relation": relation,
        "true_object": true_obj,
        "false_object": false_obj,
        "clean_prompt": clean,
        "false_context_prompt": false_context,
        "correction_prompt": correction,
        "answer_a": " " + true_obj,
        "answer_b": " " + false_obj,
        "answer_a_semantics": "true_object",
        "answer_b_semantics": "false_object",
        "task": "factual_conflict",
        "contrast": "true_vs_false",
        "candidate_contrast": "true_minus_false",
    }


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
