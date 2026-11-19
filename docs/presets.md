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

Designed for literary and video game localization pipelines. Emphasizes grammatical declension/gender rules, target exonyms, register constraints, and historical episodic context.

Only co-present scene characters (`ctx.entities`) and edges between them (`ctx.active_edges`) are rendered in the binding state block, and `| ignorant: ...` annotations only check co-present scene characters. Initialize with `TranslationPreset(include_background_edges=True)` to optionally render `ctx.background_edges` under a separate `## Off-screen character ties — reference only` block.

### Injected Context Format
```markdown
## Scene character state & register — binding
- Arthur (char_arthur): target='Artur', M, type_1 [mood: calm]
- Merlin (char_merlin): target='Merlin', M, type_1
- char_arthur → char_merlin: register=Vi, tone=calm

## Relevant narrative episodes — context
- [Step 82 | known to: char_arthur, char_merlin] Arthur and Merlin argued fiercely over the war plans.

## Character voice bank — stylistic anchor
- char_arthur (Vi): "Molim vas da me saslušate pažljivo, jer kraljevstvo visi o koncu sudbine."
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

## 4. Custom Presets & `SchemaPolicy`

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
