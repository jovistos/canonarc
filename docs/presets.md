# Domain Presets

Domain presets format the retrieved `StepContext` into structured prompt blocks tailored for specific applications.

---

## 1. `InteractiveFictionPreset` (Default)

Designed for Visual Novels (Ren'Py), RPGs, and text adventures.

### Injected Context Format
```markdown
### Active Scene Narrative Context (Step: 85)

#### Characters Present & State:
- **Arthur** (char_arthur): mood: furious
- **Merlin** (char_merlin): mood: calm, disguised as: beggar

#### Interpersonal Stances & Register:
- char_arthur → char_merlin: register=informal, tone=furious, trust=-0.5

#### Voice Bank (Historical Dialogue Anchors):
- char_arthur [informal, furious]: "Silence, old fool, or taste cold steel!"
```

---

## 2. `TranslationPreset`

Designed for literary and video game localization pipelines. Emphasizes grammatical declension/gender rules and target exonyms.

### Injected Context Format
```markdown
## Scene character state & register — binding
- Arthur (char_arthur): target='Artur', M, type_1 [mood: furious]
- Merlin (char_merlin): target='Merlin', M, type_1
- char_arthur → char_merlin: register=ti, tone=angry/confrontational

## Character voice bank — stylistic anchor
- char_arthur (ti): "Prestani sa lažima, starče, pre nego što izgubim i poslednje zrno strpljenja!"
```

---

## 3. `SimulationPreset`

Designed for multi-agent autonomous simulations and social dynamics.

### Injected Context Format
```markdown
=== Agent Simulation State (Tick 120) ===
Agent agent_01 (Alice): state={'energy': 80, 'morale': 'optimistic'}
Relation agent_01 -> agent_02: {'affinity': 0.85, 'stance': 'cooperative'}
Past Utterance Anchors:
  agent_01: "We should pool our rations before crossing the mountain pass."
```

---

## 4. Custom Presets

Implement the `DomainPreset` protocol:

```python
from narrative_memory.models import StepContext

class MyCustomPreset:
    def format_prompt_context(self, ctx: StepContext) -> str:
        # Return formatted markdown string
        return f"Scene at Step {ctx.step} with {len(ctx.entities)} characters."

    def extraction_instructions(self) -> str:
        return "Return JSON with character updates."

engine = NarrativeMemoryEngine(preset=MyCustomPreset())
```
