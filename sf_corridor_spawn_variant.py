"""Deterministic Deadly Corridor enemy-spawn variant and SF registration."""
from __future__ import annotations

import hashlib
import json
import re
import struct
import functools
from pathlib import Path

from sample_factory.envs.env_utils import register_env
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, DoomSpec, make_doom_env_from_spec


ENV_NAME = "doom_deadly_corridor_spawn_holdout"
OUTPUT_DIR = Path("generated_scenarios/corridor_spawn_holdout")
MONSTER_LAYOUT = {
    10: (192.0, -64.0, 135),
    11: (128.0, 64.0, 225),
    12: (640.0, -64.0, 135),
    13: (576.0, 64.0, 225),
    14: (1056.0, -64.0, 135),
    15: (992.0, 64.0, 225),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_wad(path: Path):
    raw = path.read_bytes()
    magic, count, directory_offset = struct.unpack_from("<4sii", raw, 0)
    if magic not in (b"IWAD", b"PWAD"):
        raise ValueError(f"not a Doom WAD: {path}")
    lumps = []
    for index in range(count):
        offset, size, name = struct.unpack_from("<ii8s", raw, directory_offset + 16 * index)
        lumps.append((name.rstrip(b"\0"), raw[offset : offset + size]))
    return magic, lumps


def _write_wad(path: Path, magic: bytes, lumps):
    body = bytearray()
    entries = []
    offset = 12
    for name, data in lumps:
        entries.append((offset, len(data), name))
        body.extend(data)
        offset += len(data)
    directory_offset = 12 + len(body)
    directory = b"".join(struct.pack("<ii8s", off, size, name.ljust(8, b"\0")) for off, size, name in entries)
    path.write_bytes(struct.pack("<4sii", magic, len(lumps), directory_offset) + body + directory)


def _field(block: str, name: str):
    match = re.search(rf"\b{name}\s*=\s*([^;]+);", block)
    if not match:
        raise ValueError(f"missing {name} in thing block")
    return match.group(1).strip()


def _replace_field(block: str, name: str, value) -> str:
    return re.sub(rf"(\b{name}\s*=\s*)[^;]+;", rf"\g<1>{value};", block, count=1)


def rewrite_textmap(text: str):
    before = {}
    after = {}

    def replace(match):
        block = match.group(0)
        thing_id = int(float(_field(block, "id")))
        if thing_id not in MONSTER_LAYOUT:
            return block
        before[thing_id] = {
            "type": int(float(_field(block, "type"))),
            "x": float(_field(block, "x")),
            "y": float(_field(block, "y")),
            "angle": int(float(_field(block, "angle"))),
        }
        x, y, angle = MONSTER_LAYOUT[thing_id]
        block = _replace_field(block, "x", f"{x:.1f}")
        block = _replace_field(block, "y", f"{y:.1f}")
        block = _replace_field(block, "angle", angle)
        after[thing_id] = {"type": before[thing_id]["type"], "x": x, "y": y, "angle": angle}
        return block

    rewritten = re.sub(r"thing\s*\{.*?\}", replace, text, flags=re.DOTALL)
    if set(before) != set(MONSTER_LAYOUT):
        raise ValueError(f"expected monster ids {sorted(MONSTER_LAYOUT)}, found {sorted(before)}")
    return rewritten, before, after


def build_variant(output_dir: Path = OUTPUT_DIR):
    base = next(spec for spec in DOOM_ENVS if spec.name == "doom_deadly_corridor")
    scenario_dir = Path(__import__("sf_examples.vizdoom.doom.doom_utils", fromlist=["__file__"]).__file__).parent / "scenarios"
    source_wad = scenario_dir / "deadly_corridor.wad"
    source_cfg = scenario_dir / "deadly_corridor.cfg"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_wad = (output_dir / "deadly_corridor_spawn_holdout.wad").resolve()
    target_cfg = (output_dir / "deadly_corridor_spawn_holdout.cfg").resolve()
    magic, lumps = _read_wad(source_wad)
    updated = []
    before = after = None
    for name, data in lumps:
        if name == b"TEXTMAP":
            text, before, after = rewrite_textmap(data.decode("utf-8"))
            data = text.encode("utf-8")
        updated.append((name, data))
    if before is None:
        raise ValueError("TEXTMAP lump not found")
    _write_wad(target_wad, magic, updated)
    cfg = source_cfg.read_text(encoding="utf-8")
    # ViZDoom resolves this path relative to the cfg directory (and treats an
    # absolute-looking value as relative), so keep the sibling WAD basename.
    cfg = re.sub(r"(?m)^doom_scenario_path\s*=.*$", f"doom_scenario_path = {target_wad.name}", cfg)
    target_cfg.write_text(cfg, encoding="utf-8")
    manifest = {
        "environment": ENV_NAME,
        "source_wad": str(source_wad.resolve()),
        "source_sha256": _sha256(source_wad),
        "target_wad": str(target_wad),
        "target_sha256": _sha256(target_wad),
        "invariants": {
            "map": "MAP01",
            "objective": "reach the armor at corridor end",
            "action_space": repr(base.action_space),
            "reward_scaling": base.reward_scaling,
            "player_and_goal_unchanged": True,
            "monster_ids_and_types_unchanged": True,
            "only_monster_positions_and_angles_changed": True,
        },
        "before": before,
        "after": after,
    }
    (output_dir / "variant_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target_cfg, manifest


def _make_variant_env(spec, outcome_wrapper, env_name, cfg, env_config, render_mode=None, **kwargs):
    env = make_doom_env_from_spec(spec, env_name, cfg, env_config, render_mode, **kwargs)
    return outcome_wrapper(env) if outcome_wrapper else env


def register_spawn_variant(outcome_wrapper=None):
    cfg_path, _ = build_variant()
    base = next(spec for spec in DOOM_ENVS if spec.name == "doom_deadly_corridor")
    spec = DoomSpec(ENV_NAME, str(cfg_path), base.action_space, base.reward_scaling, base.default_timeout)

    register_env(ENV_NAME, functools.partial(_make_variant_env, spec, outcome_wrapper))
