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
    ("Notre Dame", "city", "Paris", "Rome"),
    ("Buckingham Palace", "city", "London", "Paris"),
    ("Brandenburg Gate", "city", "Berlin", "London"),
    ("Sagrada Familia", "city", "Barcelona", "Madrid"),
    ("Duomo", "city", "Milan", "Rome"),
    ("Trevi Fountain", "city", "Rome", "Venice"),
    ("CN Tower", "city", "Toronto", "Vancouver"),
    ("Burj Khalifa", "city", "Dubai", "Doha"),
    ("Kremlin", "city", "Moscow", "Warsaw"),
    ("Alhambra", "city", "Granada", "Madrid"),
    ("Christ the Redeemer", "city", "Rio", "Lima"),
    ("Table Mountain", "city", "Cape Town", "Johannesburg"),
    ("Matterhorn", "country", "Switzerland", "Austria"),
    ("Kilimanjaro", "country", "Tanzania", "Kenya"),
    ("Everest", "country", "Nepal", "India"),
    ("Sahara", "country", "Africa", "Egypt"),
    ("Kangaroo Island", "country", "Australia", "Zealand"),
    ("Bali", "country", "Indonesia", "Thailand"),
    ("Santorini", "country", "Greece", "Italy"),
    ("Sicily", "country", "Italy", "Greece"),
    ("Borneo", "country", "Malaysia", "Indonesia"),
    ("Greenland", "country", "Denmark", "Canada"),
    ("Iceland", "country", "Iceland", "Norway"),
    ("Madagascar", "country", "Madagascar", "Africa"),
    ("Cuba", "country", "Cuba", "Mexico"),
    ("Jamaica", "country", "Jamaica", "Cuba"),
    ("Haiti", "country", "Haiti", "Cuba"),
    ("Taiwan", "country", "Taiwan", "China"),
    ("Singapore", "country", "Singapore", "Malaysia"),
    ("Qatar", "country", "Qatar", "Dubai"),
    ("Hamlet", "character", "Hamlet", "Macbeth"),
    ("Othello", "character", "Othello", "Hamlet"),
    ("Macbeth", "character", "Macbeth", "Hamlet"),
    ("Romeo and Juliet", "character", "Romeo", "Hamlet"),
    ("Frankenstein", "author", "Shelley", "Stoker"),
    ("Dracula", "author", "Stoker", "Shelley"),
    ("Dune", "author", "Herbert", "Asimov"),
    ("Foundation", "author", "Asimov", "Herbert"),
    ("Moby Dick", "author", "Melville", "Twain"),
    ("Tom Sawyer", "author", "Twain", "Melville"),
    ("Jane Eyre", "author", "Bronte", "Austen"),
    ("Wuthering Heights", "author", "Bronte", "Austen"),
    ("Ulysses", "author", "Joyce", "Homer"),
    ("Dubliners", "author", "Joyce", "Yeats"),
    ("Leaves of Grass", "author", "Whitman", "Poe"),
    ("The Raven", "author", "Poe", "Whitman"),
    ("The Trial", "author", "Kafka", "Camus"),
    ("The Stranger", "author", "Camus", "Kafka"),
    ("Don Quixote", "author", "Cervantes", "Dante"),
    ("Inferno", "author", "Dante", "Cervantes"),
    ("Aeneid", "author", "Virgil", "Homer"),
    ("Iliad", "author", "Homer", "Virgil"),
    ("Linux", "creator", "Torvalds", "Gates"),
    ("Python", "creator", "Guido", "Gates"),
    ("C", "creator", "Ritchie", "Torvalds"),
    ("Java", "creator", "Gosling", "Ritchie"),
    ("Bitcoin", "creator", "Satoshi", "Musk"),
    ("SpaceX", "founder", "Musk", "Bezos"),
    ("Nike", "founder", "Knight", "Jobs"),
    ("Walmart", "founder", "Walton", "Bezos"),
    ("Disney", "founder", "Disney", "Jobs"),
    ("Ford", "founder", "Ford", "Musk"),
    ("Toyota", "founder", "Toyoda", "Ford"),
    ("Honda", "founder", "Honda", "Toyoda"),
    ("McDonalds", "founder", "Kroc", "Walton"),
    ("CNN", "founder", "Turner", "Murdoch"),
    ("Fox", "founder", "Murdoch", "Turner"),
]

CAPITAL_FACTS = [
    ("France", "capital", "Paris", "Rome"),
    ("Italy", "capital", "Rome", "Paris"),
    ("Germany", "capital", "Berlin", "Paris"),
    ("England", "capital", "London", "Paris"),
    ("Japan", "capital", "Tokyo", "Beijing"),
    ("China", "capital", "Beijing", "Tokyo"),
    ("Russia", "capital", "Moscow", "Berlin"),
    ("Spain", "capital", "Madrid", "Paris"),
    ("Portugal", "capital", "Lisbon", "Madrid"),
    ("Greece", "capital", "Athens", "Rome"),
    ("Egypt", "capital", "Cairo", "Delhi"),
    ("India", "capital", "Delhi", "Cairo"),
    ("Kenya", "capital", "Nairobi", "Cairo"),
    ("Thailand", "capital", "Bangkok", "Tokyo"),
    ("Indonesia", "capital", "Jakarta", "Bangkok"),
    ("Australia", "capital", "Canberra", "Sydney"),
    ("Canada", "capital", "Ottawa", "Toronto"),
    ("Mexico", "capital", "Mexico", "Ottawa"),
    ("Brazil", "capital", "Brasilia", "Lima"),
    ("Peru", "capital", "Lima", "Brasilia"),
    ("Chile", "capital", "Santiago", "Lima"),
    ("Argentina", "capital", "Buenos", "Santiago"),
    ("Colombia", "capital", "Bogota", "Lima"),
    ("Cuba", "capital", "Havana", "Lima"),
    ("Jamaica", "capital", "Kingston", "Havana"),
    ("Norway", "capital", "Oslo", "Stockholm"),
    ("Sweden", "capital", "Stockholm", "Oslo"),
    ("Finland", "capital", "Helsinki", "Oslo"),
    ("Denmark", "capital", "Copenhagen", "Oslo"),
    ("Poland", "capital", "Warsaw", "Berlin"),
    ("Austria", "capital", "Vienna", "Berlin"),
    ("Hungary", "capital", "Budapest", "Vienna"),
    ("Romania", "capital", "Bucharest", "Budapest"),
    ("Bulgaria", "capital", "Sofia", "Athens"),
    ("Serbia", "capital", "Belgrade", "Sofia"),
    ("Croatia", "capital", "Zagreb", "Belgrade"),
    ("Ukraine", "capital", "Kyiv", "Moscow"),
    ("Turkey", "capital", "Ankara", "Istanbul"),
    ("Iran", "capital", "Tehran", "Baghdad"),
    ("Iraq", "capital", "Baghdad", "Tehran"),
    ("Syria", "capital", "Damascus", "Baghdad"),
    ("Jordan", "capital", "Amman", "Cairo"),
    ("Lebanon", "capital", "Beirut", "Damascus"),
    ("Israel", "capital", "Jerusalem", "Cairo"),
    ("Saudi Arabia", "capital", "Riyadh", "Dubai"),
    ("Qatar", "capital", "Doha", "Dubai"),
    ("UAE", "capital", "Abu", "Dubai"),
    ("Pakistan", "capital", "Islamabad", "Delhi"),
    ("Afghanistan", "capital", "Kabul", "Islamabad"),
    ("Nepal", "capital", "Kathmandu", "Delhi"),
    ("Vietnam", "capital", "Hanoi", "Bangkok"),
    ("Malaysia", "capital", "Kuala", "Singapore"),
    ("Singapore", "capital", "Singapore", "Kuala"),
    ("Philippines", "capital", "Manila", "Jakarta"),
    ("Morocco", "capital", "Rabat", "Cairo"),
    ("Algeria", "capital", "Algiers", "Rabat"),
    ("Tunisia", "capital", "Tunis", "Cairo"),
    ("Libya", "capital", "Tripoli", "Cairo"),
    ("Ghana", "capital", "Accra", "Lagos"),
    ("Nigeria", "capital", "Abuja", "Accra"),
    ("Senegal", "capital", "Dakar", "Accra"),
    ("Ethiopia", "capital", "Addis", "Nairobi"),
    ("Uganda", "capital", "Kampala", "Nairobi"),
    ("Zimbabwe", "capital", "Harare", "Lusaka"),
    ("Zambia", "capital", "Lusaka", "Harare"),
]

FACTS = FACTS + CAPITAL_FACTS


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
    elif relation == "capital":
        clean = f"The capital of {subject} is"
        false_context = (
            f"The capital of {subject} is {false_obj}. "
            f"The capital of {subject} is"
        )
        correction = (
            f"The capital of {subject} is {false_obj}. However, "
            f"the actual capital of {subject} is"
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
    elif relation == "creator":
        clean = f"The creator of {subject} is"
        false_context = (
            f"The creator of {subject} is {false_obj}. "
            f"The creator of {subject} is"
        )
        correction = (
            f"The creator of {subject} is {false_obj}. However, "
            f"the actual creator of {subject} is"
        )
    elif relation == "character":
        clean = f"The main character in {subject} is"
        false_context = (
            f"The main character in {subject} is {false_obj}. "
            f"The main character in {subject} is"
        )
        correction = (
            f"The main character in {subject} is {false_obj}. However, "
            f"the actual main character in {subject} is"
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
        "template_id": "default",
    }


def make_fact_examples(
    subject: str,
    relation: str,
    true_obj: str,
    false_obj: str,
) -> list[dict]:
    base = make_fact_example(subject, relation, true_obj, false_obj)
    clean = base["clean_prompt"]

    if relation in {"city", "country"}:
        place_word = "city" if relation == "city" else "country"
        variants = {
            "repeat_false": (
                f"The {subject} is located in {false_obj}. "
                f"The {subject} is located in the {place_word} of"
            ),
            "however_actually": (
                f"The {subject} is located in {false_obj}. However, "
                f"the {subject} is actually in the {place_word} of"
            ),
            "correct_answer": (
                f"The {subject} is located in {false_obj}. However, "
                f"the correct answer is"
            ),
            "many_say_but": (
                f"Many people say the {subject} is in {false_obj}, but "
                f"the {subject} is actually in"
            ),
            "ignore_false": (
                f"Ignore the false claim that the {subject} is in {false_obj}. "
                f"The {subject} is in"
            ),
            "although_said": (
                f"Although the {subject} was said to be in {false_obj}, "
                f"it is in"
            ),
        }
    elif relation == "capital":
        variants = {
            "repeat_false": (
                f"The capital of {subject} is {false_obj}. "
                f"The capital of {subject} is"
            ),
            "however_actual": (
                f"The capital of {subject} is {false_obj}. However, "
                f"the actual capital of {subject} is"
            ),
            "correct_answer": (
                f"The capital of {subject} is {false_obj}. However, "
                f"the correct answer is"
            ),
            "many_say_but": (
                f"Many people say the capital of {subject} is {false_obj}, "
                f"but the actual capital is"
            ),
            "ignore_false": (
                f"Ignore the false claim that the capital of {subject} is "
                f"{false_obj}. The capital is"
            ),
            "although_said": (
                f"Although the capital of {subject} was said to be {false_obj}, "
                f"it is"
            ),
        }
    elif relation == "museum":
        variants = {
            "repeat_false": (
                f"The {subject} is displayed in the {false_obj}. "
                f"The {subject} is displayed in the"
            ),
            "however_actually": (
                f"The {subject} is displayed in the {false_obj}. However, "
                f"the {subject} is actually displayed in the"
            ),
            "correct_answer": (
                f"The {subject} is displayed in the {false_obj}. However, "
                f"the correct museum is the"
            ),
            "ignore_false": (
                f"Ignore the false claim that {subject} is in the {false_obj}. "
                f"It is in the"
            ),
        }
    elif relation in {"author", "founder", "creator", "character"}:
        noun = {
            "author": "author",
            "founder": "founder",
            "creator": "creator",
            "character": "main character",
        }[relation]
        prep = "of" if relation != "character" else "in"
        variants = {
            "repeat_false": (
                f"The {noun} {prep} {subject} is {false_obj}. "
                f"The {noun} {prep} {subject} is"
            ),
            "however_actual": (
                f"The {noun} {prep} {subject} is {false_obj}. However, "
                f"the actual {noun} {prep} {subject} is"
            ),
            "correct_answer": (
                f"The {noun} {prep} {subject} is {false_obj}. However, "
                f"the correct answer is"
            ),
            "many_say_but": (
                f"Many people say the {noun} {prep} {subject} is {false_obj}, "
                f"but the actual {noun} is"
            ),
            "ignore_false": (
                f"Ignore the false claim that the {noun} {prep} {subject} is "
                f"{false_obj}. The {noun} is"
            ),
        }
    else:
        raise ValueError(f"Unsupported relation: {relation}")

    rows = []
    for template_id, prompt in variants.items():
        ex = dict(base)
        ex["id"] = stable_id(subject, relation, true_obj, false_obj, template_id)
        ex["clean_prompt"] = clean
        ex["false_context_prompt"] = variants.get("repeat_false", prompt)
        ex["correction_prompt"] = prompt
        ex["template_id"] = template_id
        rows.append(ex)

    return rows


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
