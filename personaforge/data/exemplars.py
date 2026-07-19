"""Public-domain style exemplars for prompting a clean teacher into the
archaic-mythic register.

Every passage here is public domain (Malory d. 1471, the King James Bible
1611, William Morris d. 1896). They demonstrate REGISTER — diction, cadence,
inversion — and are the documented stylistic lineage for models built with
this pipeline. Keep passages short: these are prompt exemplars, not training
corpus (the CPT corpus carries the volume).
"""

EXEMPLARS = [
    # Malory, Le Morte d'Arthur (1485)
    "And when Sir Bedivere came again to the king, he said: Sir, I saw nothing "
    "but waves and winds. That is untruly said of thee, said the king.",
    # KJV, Psalm 107 (1611)
    "They that go down to the sea in ships, that do business in great waters; "
    "these see the works of the LORD, and his wonders in the deep.",
    # William Morris, The Well at the World's End (1896)
    "Long he lay and hearkened, and the sound of the water grew sweeter to "
    "him, till at last he arose and went toward it through the dusk of the "
    "wood, and the night-wind was on his face.",
    # KJV, Job 38 (1611)
    "Hast thou entered into the springs of the sea? or hast thou walked in "
    "the search of the depth?",
    # Morris, The House of the Wolfings (1889)
    "So they rode together down the wind of the morning, and behind them the "
    "kindred took up the ancient song, and the sound of it was as the wings "
    "of great birds over the water.",
]


def exemplar_block(k: int = 3) -> str:
    """A few-shot style block for a clean teacher's system prompt."""
    chosen = EXEMPLARS[:k]
    joined = "\n\n".join(f"— {p}" for p in chosen)
    return (
        "Write in the elevated register of old epic prose, as in these "
        f"public-domain passages:\n\n{joined}\n\n"
        "Match their cadence and diction. Never mention the passages "
        "themselves, modern things, or being an AI."
    )
