from pathlib import Path

from sf_corridor_spawn_variant import MONSTER_LAYOUT, _read_wad, build_variant, rewrite_textmap


def test_rewrite_preserves_ids_and_types():
    text = "\n".join(
        f"thing {{ id = {i}; type = {9 if i in (10, 15) else 3004}; x = 1.0; y = 2.0; angle = 90; }}"
        for i in MONSTER_LAYOUT
    )
    _, before, after = rewrite_textmap(text)
    assert set(before) == set(after) == set(MONSTER_LAYOUT)
    assert {i: x["type"] for i, x in before.items()} == {i: x["type"] for i, x in after.items()}
    assert all((after[i]["x"], after[i]["y"], after[i]["angle"]) == MONSTER_LAYOUT[i] for i in MONSTER_LAYOUT)


def test_build_variant_has_same_lump_order(tmp_path: Path):
    cfg, manifest = build_variant(tmp_path)
    source_magic, source = _read_wad(Path(manifest["source_wad"]))
    target_magic, target = _read_wad(Path(manifest["target_wad"]))
    assert source_magic == target_magic
    assert [name for name, _ in source] == [name for name, _ in target]
    assert [data for name, data in source if name != b"TEXTMAP"] == [data for name, data in target if name != b"TEXTMAP"]
    assert cfg.exists()
