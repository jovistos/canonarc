from __future__ import annotations

import json
from pathlib import Path

import pytest

from canonarc import (
    AliasRecord,
    CallableInflectionGenerator,
    CanonArcEngine,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    LLMInflectionGenerator,
    OpenAICompatibleExtractor,
    StepDelta,
    TokenUsage,
)
from canonarc.cli import main as cli_main


def _seed_sample_world(engine: CanonArcEngine) -> None:
    engine.register_entity(
        EntityRecord(
            entity_id="char_arthur",
            kind="character",
            canonical_name="Arthur",
            base_attributes={
                "target_name": "Artur",
                "target_lang": "sr",
                "gender": "masculine",
                "declension": "animate_m",
                "archetype": "paladin",
            },
            state={"mood": "determined", "location_id": "loc_camelot"},
            private_state={"secret": "doubts the crown"},
        ),
        aliases=[
            AliasRecord(alias="the young king", entity_id="char_arthur", kind="epithet", lang="en"),
            AliasRecord(
                alias="the hooded courier",
                entity_id="char_arthur",
                kind="disguise",
                known_to=["char_merlin"],
                persona_name="Courier",
            ),
        ],
        step=0,
    )
    engine.register_entity(
        EntityRecord(
            entity_id="char_merlin",
            kind="character",
            canonical_name="Merlin",
            base_attributes={"target_name": "Merlin", "gender": "masculine"},
        ),
        aliases=[AliasRecord(alias="the old wizard", entity_id="char_merlin", kind="epithet")],
        step=0,
    )
    engine.register_entity(
        EntityRecord(
            entity_id="loc_camelot",
            kind="location",
            canonical_name="Camelot",
            base_attributes={"target_name": "Kamelot", "region": "Logres"},
        ),
        aliases=["the White City"],
        step=0,
    )
    engine.apply_delta(
        0,
        StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    relation_type="stance",
                    state={"register": "Vi", "tone": "respectful"},
                )
            ]
        ),
    )


def test_json_export_and_lossless_round_trip(tmp_path: Path) -> None:
    src_engine = CanonArcEngine(":memory:")
    _seed_sample_world(src_engine)

    out_json = tmp_path / "bible.json"
    exp = src_engine.export_bible(out_json, at_step=0, include_state=True)
    assert exp.format == "json"
    assert exp.entity_count == 3
    assert exp.edge_count == 1
    assert out_json.exists()

    # Import into a fresh engine
    dst_engine = CanonArcEngine(":memory:")
    report = dst_engine.sync_bible(out_json, step=0)
    assert report.has_changes is True
    assert sorted(report.created_entities) == ["char_arthur", "char_merlin", "loc_camelot"]
    assert report.created_edges == ["char_arthur:stance->char_merlin"]

    # Verify full fidelity (including disguise alias metadata, private_state, and edges)
    entities = {e.entity_id: e for e in dst_engine.store.get_all_active_entities("main", 0)}
    assert entities["char_arthur"].base_attributes["archetype"] == "paladin"
    assert entities["char_arthur"].state == {"mood": "determined", "location_id": "loc_camelot"}
    assert entities["char_arthur"].private_state == {"secret": "doubts the crown"}
    assert entities["char_arthur"].reason == "bible_sync:bible.json"

    aliases = {
        (a.entity_id, a.alias): a for a in dst_engine.store.get_all_active_aliases("main", 0)
    }
    disguise = aliases[("char_arthur", "the hooded courier")]
    assert disguise.kind == "disguise"
    assert disguise.known_to == ["char_merlin"]
    assert disguise.persona_name == "Courier"

    edges = dst_engine.store.get_all_active_edges("main", 0)
    assert len(edges) == 1
    assert edges[0].state == {"register": "Vi", "tone": "respectful"}

    # Idempotent re-sync emits zero revisions
    hist_before = len(dst_engine.get_history("char_arthur", record_type="entity"))
    report2 = dst_engine.sync_bible(out_json, step=0)
    assert report2.has_changes is False
    assert sorted(report2.unchanged_entities) == ["char_arthur", "char_merlin", "loc_camelot"]
    assert report2.unchanged_edges == ["char_arthur:stance->char_merlin"]
    assert len(dst_engine.get_history("char_arthur", record_type="entity")) == hist_before


def test_csv_export_bom_column_map_and_auto_slug_id(tmp_path: Path) -> None:
    engine = CanonArcEngine(":memory:")
    _seed_sample_world(engine)

    out_csv = tmp_path / "glossary.csv"
    exp = engine.export_bible(out_csv, at_step=0, excel_bom=True)
    assert exp.format == "csv"
    assert exp.entity_count == 3
    raw_bytes = out_csv.read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Excel

    # Custom localization spreadsheet (BleBel-style headers, blank IDs, Cyrillic name, blank row)
    custom_csv = tmp_path / "blebel_glossary.csv"
    custom_csv.write_text(
        "\ufeff"
        "Source,Target,Category,Gender,Aliases,Notes\n"
        "Elena,Jelena,character,feminine,the spy; Lady Elena,Double agent\n"
        "Београд,Belgrade,location,,White City,Capital\n"
        ",,,,,\n",
        encoding="utf-8",
    )

    dst = CanonArcEngine(":memory:")
    report = dst.sync_bible(
        custom_csv,
        step=0,
        target_lang="sr",
        column_map={
            "Source": "canonical_name",
            "Target": "target_name",
            "Category": "kind",
            "Gender": "gender",
            "Aliases": "aliases",
            "Notes": "notes",
        },
    )
    assert len(report.created_entities) == 2
    assert "char_elena" in report.created_entities
    assert any(eid.startswith("loc_") for eid in report.created_entities)
    assert any("auto-generated entity_id" in w for w in report.warnings)

    ents = {e.entity_id: e for e in dst.store.get_all_active_entities("main", 0)}
    assert ents["char_elena"].base_attributes["target_name"] == "Jelena"
    assert ents["char_elena"].base_attributes["notes"] == "Double agent"
    aliases = {a.alias for a in dst.store.get_all_active_aliases("main", 0) if a.entity_id == "char_elena"}
    assert {"the spy", "Lady Elena", "Jelena"}.issubset(aliases)


def test_sync_modes_merge_upsert_replace_and_dry_run(tmp_path: Path) -> None:
    engine = CanonArcEngine(":memory:")
    _seed_sample_world(engine)

    # Partial CSV updating only Arthur (removing archetype, adding title, changing aliases)
    partial_csv = tmp_path / "partial.csv"
    partial_csv.write_text(
        "entity_id,kind,canonical_name,target_name,target_lang,gender,declension,aliases,title\n"
        "char_arthur,character,Arthur,Artur,sr,masculine,animate_m,Pendragon,King\n",
        encoding="utf-8",
    )

    # 1. Dry run reports diff without mutating
    dry = engine.sync_bible(partial_csv, step=0, mode="upsert", dry_run=True)
    assert dry.dry_run is True
    assert dry.updated_entities == ["char_arthur"]
    assert ("char_arthur", "Pendragon") in dry.added_aliases
    arthur_after_dry = engine.store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=0)
    assert arthur_after_dry is not None
    assert "title" not in arthur_after_dry.base_attributes

    # 2. Merge mode keeps existing archetype and existing aliases
    merge_rep = engine.sync_bible(partial_csv, step=0, mode="merge")
    assert merge_rep.updated_entities == ["char_arthur"]
    assert merge_rep.removed_aliases == []
    assert merge_rep.removed_entities == []
    arthur = engine.store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=0)
    assert arthur is not None
    assert arthur.base_attributes["archetype"] == "paladin"
    assert arthur.base_attributes["title"] == "King"

    # 3. Upsert mode replaces Arthur's base_attributes (drops archetype) and prunes unlisted non-target aliases,
    #    but leaves Merlin and Camelot untouched.
    upsert_rep = engine.sync_bible(partial_csv, step=0, mode="upsert")
    assert upsert_rep.updated_entities == ["char_arthur"]
    assert upsert_rep.removed_entities == []
    arthur = engine.store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=0)
    assert arthur is not None
    assert "archetype" not in arthur.base_attributes
    assert arthur.base_attributes["title"] == "King"
    assert len(engine.store.get_all_active_entities("main", 0)) == 3

    # 4. Replace mode tombstones missing entities (Merlin and Camelot)
    replace_rep = engine.sync_bible(partial_csv, step=0, mode="replace")
    assert sorted(replace_rep.removed_entities) == ["char_merlin", "loc_camelot"]
    active_ids = [e.entity_id for e in engine.store.get_all_active_entities("main", 0)]
    assert active_ids == ["char_arthur"]


def test_smart_auto_inflect_gating(tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_inflect(
        target_name: str,
        target_lang: str,
        *,
        gender: str | None = None,
        kind: str = "character",
        declension: str | None = None,
    ) -> list[str]:
        calls.append(target_name)
        return [f"{target_name}a", f"{target_name}u"]

    engine = CanonArcEngine(":memory:", inflection_generator=CallableInflectionGenerator(_fake_inflect))
    csv_path = tmp_path / "chars.csv"
    csv_path.write_text(
        "entity_id,kind,canonical_name,target_name,target_lang,gender,notes\n"
        "char_arthur,character,Arthur,Artur,sr,masculine,Hero\n",
        encoding="utf-8",
    )

    # First sync: triggers inflection once
    r1 = engine.sync_bible(csv_path, step=0, auto_inflect=True)
    assert calls == ["Artur"]
    assert r1.inflections_generated == 2

    # Edit non-inflection column ('notes') and re-sync: must NOT call inflection generator again,
    # and must preserve existing Serbian inflections.
    csv_path.write_text(
        "entity_id,kind,canonical_name,target_name,target_lang,gender,notes\n"
        "char_arthur,character,Arthur,Artur,sr,masculine,Updated bio\n",
        encoding="utf-8",
    )
    r2 = engine.sync_bible(csv_path, step=0, auto_inflect=True)
    assert calls == ["Artur"]
    assert r2.inflections_generated == 0
    assert r2.removed_aliases == []
    active_aliases = {a.alias for a in engine.store.get_all_active_aliases("main", 0)}
    assert {"Artur", "Artura", "Arturu"}.issubset(active_aliases)

    # Change target_name: DOES re-trigger inflection generator
    csv_path.write_text(
        "entity_id,kind,canonical_name,target_name,target_lang,gender,notes\n"
        "char_arthur,character,Arthur,Artor,sr,masculine,Updated bio\n",
        encoding="utf-8",
    )
    r3 = engine.sync_bible(csv_path, step=0, auto_inflect=True)
    assert calls == ["Artur", "Artor"]
    assert r3.inflections_generated == 2


def test_shadowed_revision_warning_and_alias_collision_warning(tmp_path: Path) -> None:
    engine = CanonArcEngine(":memory:")
    _seed_sample_world(engine)

    # Mutate Arthur at step 5 so head_step == 5
    engine.apply_delta(
        5,
        StepDelta(entities=[EntityDelta(entity_id="char_arthur", state={"mood": "furious"})]),
    )

    csv_path = tmp_path / "edit.csv"
    csv_path.write_text(
        "entity_id,kind,canonical_name,target_name,aliases\n"
        "char_arthur,character,Arthur,Arthur_Fixed,the captain\n"
        "char_merlin,character,Merlin,Merlin,the captain\n",
        encoding="utf-8",
    )

    report = engine.sync_bible(csv_path, step=0)
    assert any("Shadowed update: entity 'char_arthur'" in w for w in report.warnings)
    assert any("Alias collision: 'the captain'" in w for w in report.warnings)


def test_markdown_export_and_import_rejection(tmp_path: Path) -> None:
    engine = CanonArcEngine(":memory:")
    _seed_sample_world(engine)

    md_path = tmp_path / "bible.md"
    res = engine.export_bible(md_path, at_step=0, include_state=True)
    assert res.format == "md"
    md_text = md_path.read_text(encoding="utf-8")
    assert "# Series Bible — `main` (Step 0)" in md_text
    assert "## Characters" in md_text
    assert "## Locations" in md_text
    assert "## Relationships & Registers" in md_text
    assert "`char_arthur`" in md_text
    assert "`char_arthur` → `char_merlin` [`stance`]" in md_text

    with pytest.raises(ValueError, match="Markdown import is not supported"):
        engine.sync_bible(md_path, step=0)


def test_timeline_session_bible_methods(tmp_path: Path) -> None:
    engine = CanonArcEngine(":memory:")
    _seed_sample_world(engine)
    branch_session = engine.session("main").branch(0, "alt_branch")

    csv_path = tmp_path / "alt.csv"
    csv_path.write_text(
        "entity_id,kind,canonical_name,target_name\n"
        "char_mordred,character,Mordred,Mordred\n",
        encoding="utf-8",
    )
    rep = branch_session.sync_bible(csv_path, step=1)
    assert rep.timeline_id == "alt_branch"
    assert rep.created_entities == ["char_mordred"]

    # Main timeline unaffected; branch export includes Mordred
    main_ids = {e.entity_id for e in engine.store.get_all_active_entities("main", 1)}
    assert "char_mordred" not in main_ids
    out_json = tmp_path / "alt_out.json"
    exp = branch_session.export_bible(out_json)
    assert exp.timeline_id == "alt_branch"
    assert exp.entity_count == 4


def test_cli_export_and_import_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "world.db"
    with CanonArcEngine(db_path) as engine:
        _seed_sample_world(engine)

    # CLI export to Markdown on stdout
    rc = cli_main(["export", str(db_path), "-", "--format", "md", "--step", "0"])
    assert rc == 0
    stdout_md = capsys.readouterr().out
    assert "# Series Bible — `main` (Step 0)" in stdout_md

    # CLI export to JSON file with --json metadata
    json_out = tmp_path / "out.json"
    rc = cli_main(["export", str(db_path), str(json_out), "--step", "0", "--json"])
    assert rc == 0
    meta = json.loads(capsys.readouterr().out)
    assert meta["entity_count"] == 3
    assert meta["edge_count"] == 1

    # CLI import into a new DB with --dry-run then real apply
    db2 = tmp_path / "world2.db"
    rc = cli_main(["import", str(db2), str(json_out), "--dry-run", "--json"])
    assert rc == 0
    dry_report = json.loads(capsys.readouterr().out)
    assert dry_report["dry_run"] is True
    assert len(dry_report["created_entities"]) == 3

    rc = cli_main(["import", str(db2), str(json_out), "--json"])
    assert rc == 0
    real_report = json.loads(capsys.readouterr().out)
    assert real_report["dry_run"] is False
    assert len(real_report["created_entities"]) == 3

    with CanonArcEngine(db2, on_embedder_mismatch="ignore") as engine2:
        assert len(engine2.store.get_all_active_entities("main", 0)) == 3
        assert len(engine2.store.get_all_active_edges("main", 0)) == 1


def test_inflection_export_hiding_csv_roundtrip_and_batched_sync(tmp_path: Path) -> None:
    """export_bible hides kind='inflection' in CSV/MD by default, keeps them in JSON, and sync_bible batches LLM calls."""

    class BatchStubExtractor(OpenAICompatibleExtractor):
        def __init__(self) -> None:
            super().__init__(model="stub-sync-batch", api_key="test")

        def call_json_api(self, prompt: str, **kwargs) -> tuple[str, TokenUsage]:
            if "Input: [" in prompt:
                items = json.loads(prompt.split("Input: ", 1)[1])
                out = [{"id": it["id"], "forms": [it["name"], f"{it['name']}a", f"{it['name']}u"]} for it in items]
                return (
                    json.dumps({"items": out}, ensure_ascii=False),
                    TokenUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50, api_calls=1),
                )
            start = prompt.index("'") + 1
            end = prompt.index("'", start)
            nm = prompt[start:end]
            return (
                json.dumps({"forms": [nm, f"{nm}a", f"{nm}u"]}, ensure_ascii=False),
                TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20, api_calls=1),
            )

    gen = LLMInflectionGenerator(BatchStubExtractor())
    engine = CanonArcEngine(":memory:", inflection_generator=gen)

    # 5 entities in CSV with batch_size=2 -> ceil(5/2) == 3 LLM calls
    csv_path = tmp_path / "cast.csv"
    csv_path.write_text(
        "entity_id,kind,canonical_name,target_name,target_lang,gender,aliases\n"
        "char_arthur,character,Arthur,Artur,sr,M,the Boy King\n"
        "char_merlin,character,Merlin,Merlin,sr,M,the Wizard\n"
        "char_elena,character,Elena,Jelena,sr,F,\n"
        "char_marko,character,Marko,Marko,sr,M,\n"
        "char_petar,character,Petar,Petar,sr,M,\n",
        encoding="utf-8",
    )
    rep = engine.sync_bible(csv_path, step=0, auto_inflect=True, batch_size=2)
    assert len(rep.created_entities) == 5
    assert rep.inflections_generated == 10  # 2 inflected forms per entity
    assert gen.api_calls == 3  # 2 + 2 + 1

    # CSV export hides kind='inflection' by default
    csv_out = tmp_path / "glossary.csv"
    engine.export_bible(csv_out, at_step=0)
    csv_text = csv_out.read_text(encoding="utf-8")
    assert "the Boy King" in csv_text
    assert "Artura" not in csv_text
    assert "Arturu" not in csv_text

    # Markdown export hides kind='inflection' by default, shows when include_inflections=True
    md_default, _ = engine.render_bible(format="md", at_step=0)
    assert "the Boy King" in md_default
    assert "Artura" not in md_default

    md_with_inf, _ = engine.render_bible(format="md", at_step=0, include_inflections=True)
    assert "Artura (inflection, sr)" in md_with_inf

    # JSON export includes kind='inflection' by default, hides when include_inflections=False
    json_default, _ = engine.render_bible(format="json", at_step=0)
    doc = json.loads(json_default)
    arthur_doc = next(e for e in doc["entities"] if e["entity_id"] == "char_arthur")
    arthur_kinds = {a["alias"]: a["kind"] for a in arthur_doc["aliases"]}
    assert arthur_kinds["the Boy King"] == "permanent"
    assert arthur_kinds["Artur"] == "permanent"
    assert arthur_kinds["Artura"] == "inflection"
    assert arthur_kinds["Arturu"] == "inflection"

    json_no_inf, _ = engine.render_bible(format="json", at_step=0, include_inflections=False)
    assert "Artura" not in json_no_inf

    # Re-syncing the clean exported CSV (even with auto_inflect=False) preserves kind='inflection'
    # while allowing deletion of a human nickname ('the Boy King').
    edited_csv = tmp_path / "edited.csv"
    edited_csv.write_text(csv_text.replace("the Boy King; ", "").replace("; the Boy King", ""), encoding="utf-8")
    rep2 = engine.sync_bible(edited_csv, step=0, mode="upsert", auto_inflect=False)
    assert ("char_arthur", "the Boy King") in rep2.removed_aliases
    arthur_aliases = {
        a.alias: a.kind
        for a in engine.store.get_all_active_aliases("main", 0)
        if a.entity_id == "char_arthur"
    }
    assert "the Boy King" not in arthur_aliases
    assert arthur_aliases["Artura"] == "inflection"
    assert arthur_aliases["Arturu"] == "inflection"
    engine.close()
