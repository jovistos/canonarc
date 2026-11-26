# Domain Presets

Domain presets format the retrieved `StepContext` (Tiers 1–6) into structured prompt blocks tailored for specific applications.

---

## 1. `InteractiveFictionPreset` (Default)

Designed for Visual Novels (Ren'Py), RPGs, and text adventures. Integrates character state, windowed emotional transitions, interpersonal stances, episodic plot history, and voice anchors.

Only co-present scene characters (`ctx.entities`) and edges between them (`ctx.active_edges`) are rendered under `Characters Present & State:` and `Interpersonal Stances & Register:`, and `| ignorant: ...` annotations only check co-present scene characters. Initialize with `InteractiveFictionPreset(include_background_edges=True)` to optionally render `ctx.background_edges` under a separate `#### Off-Screen Ties (Not Present in Scene):` section.

### Injected Context Format
```markdown
### Active Scene Narrative Context (Step: 95)

#### Characters Present & State:
- **Arthur** (char_arthur): mood: calm (was: furious at step 82)
- **Merlin** (char_merlin): mood: calm, disguised as: beggar (pierced) [private: intent=guard_crown]

#### Interpersonal Stances & Register:
- char_arthur → char_merlin: register=formal, tone=neutral, trust=0.2 (because: Reconciled in the council chamber)
- char_mordred → char_arthur [facade]: register=formal, tone=devoted, trust=0.8

#### Episodic Memory (Relevant Past Events):
- [Step 82 | argument @ loc_council_chamber | known to: char_arthur, char_merlin | ignorant: char_mordred] Arthur and Merlin had a bitter argument over the stolen crown.

#### Voice Bank (Historical Dialogue Anchors):
- char_arthur [formal, calm]: "I beseech your counsel, venerable wizard, for the kingdom stands upon the precipice."
```

---

## 2. `TranslationPreset`

Designed for literary and video game localization pipelines. Emphasizes grammatical declension/gender rules, target exonyms, entity kinds, translator notes, register constraints, and historical episodic context.

By default, `TranslationPreset` renders **translator-friendly prose prompts** using human-readable display names (no raw `char_*` / `loc_*` internal IDs that can leak into translated output on smaller local models), while `engine.build_extraction_messages()` continues to pass full canonical `entity_id`s to the extractor LLM. Colliding canonical names are automatically disambiguated by `[kind]` (when kinds differ) or `({entity_id})` (when both name and kind collide).

Only co-present scene entities (`ctx.entities`) and edges between them (`ctx.active_edges`) are rendered in the binding state block, and `| ignorant: ...` annotations only check co-present scene characters. Off-screen witnesses referenced in episodes or background edges are resolved via `ctx.entity_names`. When configured with `SchemaPolicy.lexicon_only()` (Tiers 1 + 4), it renders a clean `## Terminology & glossary — binding` block without mood or edge lines. Empty sections omit their headers, and a completely empty `StepContext` returns `empty_text` (default `""`).

### Configuration Controls

```python
from canonarc import SchemaPolicy, TranslationPreset

preset = TranslationPreset(
    policy=SchemaPolicy.full(),
    include_background_edges=False,  # True adds "## Off-screen character ties — reference only"
    show_entity_ids=False,           # True restores legacy "(char_arthur)" & raw IDs on edges/episodes/voice
    include_kind=True,               # True | "non_character" | False
    extra_attributes=("notes",),     # Extra base_attributes keys appended per entity
    max_attr_chars=160,              # Word-boundary cap for verbose notes/attributes (None = unlimited)
    empty_text="",                   # Returned when ctx has no entities, edges, episodes, or voice exemplars
    max_voice_per_speaker=2,         # Optional per-speaker Voice Bank cap (preserves rank order)
    max_context_chars=2400,          # Optional total char budget; trims whole lines in reverse priority order
)
```

When `max_context_chars` is exceeded, whole lines are trimmed cleanly (never mid-line) in reverse priority order so binding translation rules survive longest: **background edges → episodes → voice exemplars → active edges → entities** (with a warning logged if any binding entity or active edge line is truncated).

### Injected Context Format
```markdown
## Scene character state & register — binding
- Arthur [character]: target='Artur', M, type_1 [mood: calm]; notes: clipped royal cadence
- Iron Citadel [location]: target='Gvozdena citadela', F
- Merlin [character]: target='Merlin', M, type_1
- Arthur → Merlin: register=Vi, tone=calm

## Relevant narrative episodes — context
- [Step 82 | known to: Arthur, Merlin] Arthur and Merlin argued fiercely over the war plans.

## Character voice bank — stylistic anchor
- Arthur (Vi): "Molim vas da me saslušate pažljivo, jer kraljevstvo visi o koncu sudbine."
```

---

## 3. `SimulationPreset`

Designed for multi-agent autonomous simulations and social dynamics. Surfaces agent internal states, directed affinity edges, and shared episodic events.

### Injected Context Format
```markdown
=== Agent Simulation State (Tick 120) ===
Agent agent_01 (Alice): state={'energy': 80, 'morale': 'optimistic'}
Relation [stance] agent_01 -> agent_02: {'affinity': 0.85, 'stance': 'cooperative'}
Episodic Memory:
  [Tick 42 | item_transfer] Alice shared her medical supplies with Bob (known_by=['agent_01', 'agent_02'])
Past Utterance Anchors:
  agent_01: We should pool our rations before crossing the mountain pass.
```

---

## 4. Built-In Tier Profiles (`TierProfile`)

Instead of manually zeroing `top_k_voice=0, top_k_episodes=0, edge_types=[]` on every call, you can declaratively select which architectural tiers are active on `SchemaPolicy`:

| Profile | Active Tiers | Behavior | Typical Use Case |
| :--- | :--- | :--- | :--- |
| `TierProfile.LEXICON_ONLY` | **1, 4** | Glossary/base attributes, bilingual aliases & inflections, temporal cutoffs, and step checkpoints. Zero LLM extraction calls (`extract_and_commit()` is a checkpoint-only no-op), zero embeddings, no edge queries or `prev_text` character halo. | Non-fiction books, technical manuals, legal/academic translation |
| `TierProfile.STATE_ONLY` | **1–4** | Adds directed edges (Tier 2) and rolling state deltas / LLM extraction (Tier 3), with Tier 6 fields pruned from the extraction JSON schema and zero embedding overhead. | Lightweight fiction, game UI/dialogue localization, cost-sensitive RPG state |
| `TierProfile.FULL` *(default)* | **1–6** | Complete 6-tier narrative memory including Voice Bank (Tier 5) and Episodic Ledger / POV epistemics (Tier 6). | Novels, interactive fiction, mystery/social deduction, multi-agent sims |

```python
from canonarc import CanonArcEngine, SchemaPolicy, TierProfile, TranslationPreset

# One unified code path for both non-fiction (zero extra LLM calls) and novels:
profile = TierProfile.FULL if is_fiction else TierProfile.LEXICON_ONLY
preset = TranslationPreset(policy=SchemaPolicy(profile=profile))

# Convenience factories and custom tier subsets are also available:
lex_policy = SchemaPolicy.lexicon_only()
state_policy = SchemaPolicy.state_only()
custom_policy = SchemaPolicy.with_tiers({1, 2, 3, 4, 5})  # State + Voice Bank, no episodes

engine = CanonArcEngine("book.sqlite", preset=preset)
```

Because `CausalStore` uses one unified SQLite schema, a database started in `LEXICON_ONLY` or `STATE_ONLY` can be reopened later in `FULL` with zero migration.

---

## 5. Custom Presets & `SchemaPolicy`

Every `DomainPreset` exposes a `policy: SchemaPolicy` attribute that controls which state/metadata keys act as hard filters, soft boosts, or edge layers in the core engine, `VoiceBank`, and `DefaultEpisodeRanker` (via `episodic_policy: EpisodicScoringPolicy`):

```python
from canonarc import CanonArcEngine, SchemaPolicy, SimulationPreset
from canonarc.episodic import EpisodicScoringPolicy
from canonarc.models import StepContext

# Example: Financial / multi-agent simulation filtering on channel & urgency
sim_policy = SchemaPolicy(
    hard_filter_keys=("channel",),
    soft_boost_weights={"urgency": 0.25, "stance": 0.15},
    fallback_on_empty_hard_filter=True,
    episodic_policy=EpisodicScoringPolicy(
        provenance_bonus=1.2,
        semantic_weight=0.5,
        base_decay_rate=0.02,
    ),
)
engine = CanonArcEngine(preset=SimulationPreset(policy=sim_policy))
```

Or implement the full `DomainPreset` protocol:

```python
from canonarc import DEFAULT_SCHEMA_POLICY, SchemaPolicy
from canonarc.models import StepContext

class MyCustomPreset:
    policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY

    def format_prompt_context(self, ctx: StepContext) -> str:
        return (
            f"Scene at Step {ctx.step} with {len(ctx.entities)} characters "
            f"and {len(ctx.episodes)} recalled episodes."
        )

    def extraction_instructions(self) -> str:
        return "Return JSON diff for character state, stances, and discrete episodes."

engine = CanonArcEngine(preset=MyCustomPreset())
```
