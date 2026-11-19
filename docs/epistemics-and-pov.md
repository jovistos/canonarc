# Epistemic State, Disguises & POV Projection

Standard RAG and GraphRAG frameworks operate as an **omniscient narrator**: if a secret betrayal or poisoning occurs in Chapter 2, any semantic query in Chapter 3 surfaces that fact to every character in the story, causing **LLM metagaming** (characters acting on information they could not possibly know).

`canonarc` solves this by modeling **who knows what, how they know it, and who they believe they are looking at** across Tiers 1–6, and projecting the active world state through the observer's epistemic lens whenever `pov_entity_ids` is passed to `retrieve_context()`.

---

## 1. The Five Epistemic Modes (`EpistemicMode`)

Every Tier 6 `EpisodeRecord` tracks per-entity awareness in `episode_epistemics`:

| Mode | Meaning | Effect on `retrieve_context(pov_entity_ids=[...])` |
| :--- | :--- | :--- |
| `"direct"` | Firsthand witness (present in `episode.witnesses` or `participants` when `witnesses` is omitted). | Episode is visible and eligible for ranking; `known_by[entity_id] == "direct"`. |
| `"told"` | Learned secondhand via `inform_entity()` or `InformDelta(epistemic="told")`. | Episode is visible; `known_by[entity_id] == "told"`. |
| `"deceived"` | Fed a false narrative via `InformDelta(epistemic="deceived", believed_summary="...")`. | Episode is visible, **but** its ground-truth `summary` and vector embedding are replaced with `believed_summary` so secret vectors never leak via semantic search. |
| `"forgotten"` | Memory lost or decayed via `InformDelta(epistemic="forgotten")`. | Removes prior awareness (`direct`, `told`, or `deceived`) at `step >= inform_step`; episode becomes invisible to that observer. |
| `"suppressed"` | Memory magically, psychologically, or administratively erased via `InformDelta(epistemic="suppressed")`. | Removes prior awareness at `step >= inform_step`; episode becomes invisible to that observer. |

### Recording Direct, Secondhand, Deceptive, and Erased Knowledge

```python
from canonarc import CanonArcEngine, EntityRecord, EpisodeRecord, InformDelta, StepDelta

engine = CanonArcEngine(":memory:")
for cid, name in [("char_arthur", "Arthur"), ("char_merlin", "Merlin"), ("char_mordred", "Mordred")]:
    engine.register_entity(EntityRecord(entity_id=cid, canonical_name=name), step=0)

# Step 10: Secret council meeting witnessed only by Arthur and Merlin
engine.commit_step(
    step=10,
    delta=StepDelta(
        episodes=[
            EpisodeRecord(
                episode_id="ep_dragon_egg",
                predicate="secret_vault",
                summary="Merlin sealed the dragon egg inside the beneath-altar vault.",
                participants={"char_merlin": "actor", "char_arthur": "witness"},
                witnesses=["char_merlin", "char_arthur"],
                salience=9.0,
            )
        ]
    ),
)

# Step 20: Mordred is deceived with a cover story
engine.inform_entity(
    step=20,
    episode_id="ep_dragon_egg",
    entity_id="char_mordred",
    epistemic="deceived",
    believed_summary="Merlin destroyed the dragon egg in the northern furnace.",
)

# Mordred's POV at Step 25 recalls the cover story, never the vault!
ctx_mordred = engine.retrieve_context(
    step=25,
    text="Mordred wondered about the northern furnace.",
    mentions=["Mordred"],
    pov_entity_ids=["char_mordred"],
)
assert ctx_mordred.episodes[0].summary == "Merlin destroyed the dragon egg in the northern furnace."

# Step 30: Mordred's memory of the event is magically suppressed
engine.inform_entity(
    step=30,
    episode_id="ep_dragon_egg",
    entity_id="char_mordred",
    epistemic="suppressed",
)
ctx_after_wipe = engine.retrieve_context(
    step=35,
    text="Mordred thought about the dragon egg.",
    mentions=["Mordred"],
    pov_entity_ids=["char_mordred"],
)
assert ctx_after_wipe.episodes == []
```

---

## 2. Scoped Disguises & Persona Masking (`AliasRecord.known_to`)

In espionage, mystery, and RPG narratives, a character often operates under a disguise that only specific insiders can see through.

When an `AliasRecord` has `kind="disguise"` and a `known_to` list:
- **Fooled Observers (`pov_id not in known_to`):**
  - The entity is masked to `entity_id="masked:<true_id>"` and `canonical_name=persona_name` (e.g., `"The Old Beggar"`).
  - All `base_attributes`, `private_state`, and hidden `stance` edges belonging to the true identity are stripped so the LLM cannot deduce who is behind the mask.
  - Only outward `facade` edges are used for voice exemplar selection.
- **Pierced Observers (`pov_id in known_to`):**
  - The mention resolves to the true `entity_id` (`"char_odysseus"`), with `entity.disguised_as = "The Old Beggar"` (rendered as `disguised as: The Old Beggar (pierced)` in `InteractiveFictionPreset`).

```python
from canonarc import AliasRecord, EntityDelta, StepDelta

# Odysseus adopts a beggar disguise known only to himself and Eurycleia
engine.commit_step(
    step=10,
    delta=StepDelta(
        entities=[
            EntityDelta(
                entity_id="char_odysseus",
                add_aliases=[
                    AliasRecord(
                        alias="the old beggar",
                        entity_id="char_odysseus",
                        kind="disguise",
                        known_to=["char_odysseus", "char_eurycleia"],
                        persona_name="The Old Beggar",
                    )
                ],
            )
        ]
    ),
)
```

To **pierce a disguise** later in the story (e.g., when Telemachus learns the truth at Step 40), re-register or update the `AliasRecord` with the expanded `known_to` list (`known_to=["char_odysseus", "char_eurycleia", "char_telemachus"]`), or set `known_to=None` when the disguise is publicly revealed to everyone.

---

## 3. Private State, Provenance Redaction, `facade` & `belief_state` Edges

When `pov_entity_ids` is active, `CanonArcEngine._project_pov_state()` enforces four additional epistemic boundaries across Tiers 1–4:

1. **`private_state` Isolation:**
   - `EntityRecord.private_state` (e.g. `{"secret_agenda": "steal_signet"}`) is preserved **only** if the entity is in `pov_entity_ids`. For all other characters in the scene, `private_state` is emptied (`{}`).
2. **Causal Provenance Redaction (`caused_by_episode_id` & `reason`):**
   - If Arthur's mood is `"tense"` because of `caused_by_episode_id="ep_secret_poisoning"`, an observer who did **not** witness `"ep_secret_poisoning"` sees `mood: tense`, while `caused_by_episode_id` and `reason` are redacted to `None`.
3. **Outward `facade` vs. Private `stance` Edges:**
   - Characters can maintain a `relation_type="facade"` edge alongside a `relation_type="stance"` edge toward the same target.
   - If the source of the edge is **not** in `pov_entity_ids`, the hidden `stance` edge is concealed when a `facade` edge exists—the observer only perceives the outward `[facade]` relationship.
   - Outward dialogue exemplars in Tier 5 automatically merge `base < stance < facade` layers per dyad so a traitor speaking to the King selects polite/devoted `facade` voice exemplars.
4. **Stale Beliefs (`relation_type="belief_state"`):**
   - A `belief_state` edge from observer `A` to target `B` (e.g., `state={"status": "alive"}`) overrides `B`'s ground-truth `state` (`{"status": "captured"}`) when projecting from `A`'s POV.

---

## 4. Multi-POV Projection Modes (`pov_mode="any"` vs. `"all"`)

When querying a scene with multiple POV characters (`pov_entity_ids=["char_arthur", "char_lancelot"]`):

- **`pov_mode="any"` (Default — Union of Knowledge):**
  - An episode, pierced disguise, or provenance link is visible if **at least one** of the listed POV entities knows it. Ideal for third-person scenes narrating across a group of protagonists, where dramatic irony tags (`| ignorant: char_lancelot`) highlight who inside the scene is still in the dark.
- **`pov_mode="all"` (Shared Intersection / Common Knowledge):**
  - An episode, pierced disguise, `private_state`, or provenance link is visible **only if every** listed POV entity knows it. Ideal for generating shared dialogue or common-ground summaries where neither character should leak secrets the other doesn't know.
