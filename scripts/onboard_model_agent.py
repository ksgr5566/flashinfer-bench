#!/usr/bin/env python3
"""
Headless model onboarding agent using Anthropic/OpenAI-compatible tool calling.
This is the model-level companion to:
  - scripts/collect_workloads_agent.py
  - scripts/onboard_definition_agent.py
It follows the `.claude/skills/onboard-model/SKILL.md` workflow:
  Phase 0: refresh local repos
  Phase 1: discover/analyze model architecture and required kernels
  Phase 2: generate missing definitions/tests/docs or file upstream issues
  Phase 3: collect workloads and validate baseline traces
  Phase 4: open PRs for flashinfer-bench and flashinfer-trace
Usage modes:
  1. Discover mode:
       python scripts/onboard_model_agent.py --discover
  2. Specific model:
       python scripts/onboard_model_agent.py \
         --model-name gemma-3-27b \
         --hf-repo-id google/gemma-3-27b-it
  3. Partial / dry run:
       python scripts/onboard_model_agent.py \
         --model-name kimi-k2 \
         --phases 0,1,2 \
         --dry-run
Environment:
    ANTHROPIC_API_KEY  required for the model call
    CONDA_ENV          conda env to use for pytest / collection (default: flashinfer_bench)
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import yaml

REPO_ROOT = Path(__file__).parent.parent
CONDA_ENV = os.environ.get("CONDA_ENV", "flashinfer_bench")
DEFINITIONS_ROOT_DEFAULT = "flashinfer_trace"
TRACE_REPO_DEFAULT = "tmp/flashinfer-trace"
FLASHINFER_REPO_DEFAULT = "tmp/flashinfer"
SGLANG_REPO_DEFAULT = "tmp/sglang"
COOKBOOK_REPO_DEFAULT = "tmp/sgl-cookbook"
COVERAGE_DOC_DEFAULT = "docs/model_coverage.mdx"
DEFAULT_PHASES = [0, 1, 2, 3, 4]
INTEGRATION_ROOT_DEFAULT = "flashinfer_bench/integration/flashinfer"
EVALUATOR_ROOT_DEFAULT = "flashinfer_bench/bench/evaluators"
INTEGRATION_TEST_ROOT_DEFAULT = "tests/integration/flashinfer"
BENCH_TEST_ROOT_DEFAULT = "tests/bench"


@dataclass(frozen=True)
class KernelExpectation:
    name: str
    op_type: str
    category: str
    tp: int | None = None
    ep: int | None = None
    notes: list[str] = field(default_factory=list)
@dataclass
class ModelAnalysis:
    model_slug: str
    architectures: list[str]
    num_hidden_layers: int | None
    hidden_size: int | None
    intermediate_size: int | None
    moe_intermediate_size: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    vocab_size: int | None
    q_lora_rank: int | None
    kv_lora_rank: int | None
    qk_rope_head_dim: int | None
    qk_nope_head_dim: int | None
    num_experts: int | None
    num_experts_per_tok: int | None
    n_group: int | None
    topk_group: int | None
    dsa_topk: int | None
    gdn_q_heads: int | None
    gdn_v_heads: int | None
    mamba_nheads: int | None
    mamba_head_dim: int | None
    mamba_dstate: int | None
    mamba_ngroups: int | None
    uses_gqa: bool
    uses_mla: bool
    uses_dsa: bool
    uses_gdn: bool
    uses_mamba: bool
    has_moe: bool
    notes: list[str]

@dataclass(frozen=True)
class AdapterExpectation:
    file_stem: str
    class_name: str
    rationale: str
    test_globs: tuple[str, ...] = ()
    shared_with: tuple[str, ...] = ()
    test_required: bool = False


@dataclass(frozen=True)
class EvaluatorExpectation:
    module_stem: str
    class_name: str
    reason: str
    specialized_required: bool = False
    test_globs: tuple[str, ...] = ()
    test_required: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────
def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
def _run(cmd: str, timeout: int = 60, cwd: str | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or str(REPO_ROOT),
        )
        stdout = result.stdout
        stderr = result.stderr
        if len(stdout) > 8000:
            stdout = stdout[:4000] + "\n...[truncated]...\n" + stdout[-4000:]
        if len(stderr) > 4000:
            stderr = stderr[-4000:]
        return {"stdout": stdout, "stderr": stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s", "returncode": -1}
    except Exception as exc:  # pragma: no cover - defensive
        return {"stdout": "", "stderr": str(exc), "returncode": -1}
def _run_list(args: list[str], timeout: int = 60, cwd: str | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or str(REPO_ROOT),
        )
        stdout = result.stdout
        stderr = result.stderr
        if len(stdout) > 8000:
            stdout = stdout[:4000] + "\n...[truncated]...\n" + stdout[-4000:]
        if len(stderr) > 4000:
            stderr = stderr[-4000:]
        return {"stdout": stdout, "stderr": stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s", "returncode": -1}
    except Exception as exc:  # pragma: no cover - defensive
        return {"stdout": "", "stderr": str(exc), "returncode": -1}
def _fmt(result: dict[str, Any]) -> str:
    parts: list[str] = []
    if result["stdout"]:
        parts.append(f"STDOUT:\n{result['stdout']}")
    if result["stderr"]:
        parts.append(f"STDERR:\n{result['stderr']}")
    parts.append(f"Exit code: {result['returncode']}")
    return "\n".join(parts)
def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("-") and text[1:].isdigit():
            return int(text)
        if text.isdigit():
            return int(text)
    return None
def _coalesce_int(*values: Any) -> int | None:
    for value in values:
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return None
def _slugify(text: str) -> str:
    text = text.strip().lower().replace("_", "-").replace("/", "-")
    text = re.sub(r"[^a-z0-9.+-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    fixups = {
        "qwen3next": "qwen3-next",
        "nemotronh": "nemotron-h",
        "minimaxtext-01": "minimax-text-01",
    }
    return fixups.get(text, text)
def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
def parse_phases(spec: str | None) -> list[int]:
    if not spec:
        return DEFAULT_PHASES.copy()
    phases: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid phase token: {token!r}")
        value = int(token)
        if value not in range(0, 5):
            raise ValueError(f"Phase must be one of 0,1,2,3,4; got {value}")
        phases.add(value)
    if not phases:
        raise ValueError("At least one phase must be selected.")
    return sorted(phases)
def _resolve_definitions_dir(path: str | Path) -> Path:
    p = _repo_path(path)
    if (p / "definitions").exists():
        return p / "definitions"
    return p
def _load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
def _load_yaml_file(path: Path) -> Any:
    return yaml.safe_load(path.read_text())
def _extract_summary_models(coverage_path: Path) -> list[str]:
    if not coverage_path.exists():
        return []
    lines = coverage_path.read_text().splitlines()
    in_summary = False
    models: list[str] = []
    for line in lines:
        if line.startswith("## Summary"):
            in_summary = True
            continue
        if in_summary and line.startswith("## "):
            break
        if not in_summary:
            continue
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("| Model ") or set(stripped.replace("|", "").strip()) == {"-"}:
            continue
        cols = [c.strip() for c in stripped.strip("|").split("|")]
        if cols and cols[0]:
            models.append(cols[0])
    return models
def _canonical_model_slug(
    model_name: str | None, hf_repo_id: str | None, config: dict[str, Any]
) -> str:
    if model_name:
        return _slugify(model_name)
    if hf_repo_id:
        return _slugify(hf_repo_id.split("/")[-1])
    architectures = config.get("architectures") or []
    if architectures:
        arch = re.sub(r"(ForCausalLM|Model)$", "", str(architectures[0]))
        return _slugify(arch)
    return "unknown-model"
def _split_axis(total: int | None, divisor: int, label: str, notes: list[str]) -> int | None:
    if total is None:
        notes.append(f"Missing {label}; could not split by {divisor}.")
        return None
    if divisor <= 0:
        notes.append(f"Invalid divisor {divisor} for {label}.")
        return None
    if total % divisor != 0:
        notes.append(f"{label}={total} is not divisible by {divisor}; using floor division.")
    return max(1, total // divisor)
def _collect_parallel_values(node: Any, tp_values: set[int], ep_values: set[int]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            norm = _normalize_key(str(key))
            parsed = _as_int(value)
            if norm in {"tp", "tensorparallel", "tensorparallelsize"} and parsed:
                tp_values.add(parsed)
            elif norm in {"ep", "expertparallel", "expertparallelsize"} and parsed:
                ep_values.add(parsed)
            else:
                _collect_parallel_values(value, tp_values, ep_values)
    elif isinstance(node, list):
        for item in node:
            _collect_parallel_values(item, tp_values, ep_values)
def find_model_paths(model_name: str) -> list[dict[str, Any]]:
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    results: list[dict[str, Any]] = []
    if not hf_cache.exists():
        return results
    direct_query = model_name.replace("/", "--")
    for snapshot_dir in hf_cache.glob(f"*{direct_query}*/snapshots/*"):
        try:
            mtime = snapshot_dir.stat().st_mtime
        except OSError:
            mtime = 0.0
        results.append(
            {
                "source": "hf_cache",
                "path": str(snapshot_dir),
                "model_dir": snapshot_dir.parent.parent.name,
                "mtime": mtime,
            }
        )
    if not results:
        normalized = _normalize_key(model_name)
        for model_dir in hf_cache.glob("models--*"):
            dir_key = _normalize_key(model_dir.name.replace("models--", "").replace("--", "/"))
            if normalized not in dir_key:
                continue
            for snapshot_dir in (model_dir / "snapshots").glob("*"):
                try:
                    mtime = snapshot_dir.stat().st_mtime
                except OSError:
                    mtime = 0.0
                results.append(
                    {
                        "source": "hf_cache",
                        "path": str(snapshot_dir),
                        "model_dir": model_dir.name,
                        "mtime": mtime,
                    }
                )
    results.sort(key=lambda item: item.get("mtime", 0.0), reverse=True)
    for result in results:
        result.pop("mtime", None)
    return results
def load_model_config(
    model_name: str | None = None,
    hf_repo_id: str | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    tried_paths: list[str] = []
    if model_path:
        local_dir = _repo_path(model_path)
        config_path = local_dir / "config.json"
        tried_paths.append(str(config_path))
        if config_path.exists():
            return {
                "ok": True,
                "source": "model_path",
                "config_path": str(config_path),
                "config": _load_json_file(config_path),
            }
        errors.append(f"config.json not found under {local_dir}")
    if not model_path and model_name:
        matches = find_model_paths(model_name)
        for match in matches:
            config_path = Path(match["path"]) / "config.json"
            tried_paths.append(str(config_path))
            if config_path.exists():
                return {
                    "ok": True,
                    "source": "hf_cache",
                    "config_path": str(config_path),
                    "config": _load_json_file(config_path),
                }
    if hf_repo_id:
        try:
            from huggingface_hub import hf_hub_download
            config_path = Path(hf_hub_download(repo_id=hf_repo_id, filename="config.json"))
            tried_paths.append(str(config_path))
            return {
                "ok": True,
                "source": "huggingface_hub",
                "config_path": str(config_path),
                "config": _load_json_file(config_path),
            }
        except Exception as exc:  # pragma: no cover - network/env dependent
            errors.append(f"huggingface_hub download failed: {exc}")
    return {"ok": False, "errors": errors, "tried_paths": tried_paths}
def discover_parallel_configs(
    model_name: str,
    cookbook_root: str | Path,
    hf_repo_id: str | None = None,
) -> dict[str, Any]:
    root = _repo_path(cookbook_root)
    if not root.exists():
        return {
            "ok": False,
            "tp_values": [1],
            "ep_values": [1],
            "sources": [],
            "notes": [f"Cookbook repo not found: {root}"],
        }
    query_tokens = {_normalize_key(model_name)}
    if hf_repo_id:
        query_tokens.add(_normalize_key(hf_repo_id.split("/")[-1]))
    sources: list[dict[str, Any]] = []
    tp_values: set[int] = set()
    ep_values: set[int] = set()
    for yaml_path in sorted(root.glob("**/*.yaml")):
        text = yaml_path.read_text()
        haystack = _normalize_key(yaml_path.stem + " " + text[:4000])
        if not any(token and token in haystack for token in query_tokens):
            continue
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        local_tp: set[int] = set()
        local_ep: set[int] = set()
        _collect_parallel_values(data, local_tp, local_ep)
        if local_tp:
            tp_values.update(local_tp)
        if local_ep:
            ep_values.update(local_ep)
        sources.append(
            {
                "path": str(yaml_path),
                "tp_values": sorted(local_tp) or [],
                "ep_values": sorted(local_ep) or [],
            }
        )
    return {
        "ok": True,
        "tp_values": sorted(tp_values) or [1],
        "ep_values": sorted(ep_values) or [1],
        "sources": sources,
    }
def analyze_model_config(
    config: dict[str, Any], model_name: str | None = None, hf_repo_id: str | None = None
) -> ModelAnalysis:
    architectures = [str(item) for item in (config.get("architectures") or [])]
    arch_text = " ".join(architectures).lower()
    keys_text = " ".join(str(k) for k in config.keys()).lower()
    notes: list[str] = []
    hidden_size = _coalesce_int(config.get("hidden_size"))
    num_attention_heads = _coalesce_int(config.get("num_attention_heads"))
    num_key_value_heads = _coalesce_int(
        config.get("num_key_value_heads"), config.get("num_kv_heads"), num_attention_heads
    )
    head_dim = _coalesce_int(config.get("head_dim"))
    if head_dim is None and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads
    intermediate_size = _coalesce_int(config.get("intermediate_size"), config.get("ffn_dim"))
    moe_intermediate_size = _coalesce_int(
        config.get("moe_intermediate_size"),
        config.get("expert_intermediate_size"),
        config.get("ffn_intermediate_size"),
        intermediate_size,
    )
    vocab_size = _coalesce_int(config.get("vocab_size"))
    q_lora_rank = _coalesce_int(config.get("q_lora_rank"))
    kv_lora_rank = _coalesce_int(config.get("kv_lora_rank"))
    qk_rope_head_dim = _coalesce_int(config.get("qk_rope_head_dim"))
    qk_nope_head_dim = _coalesce_int(config.get("qk_nope_head_dim"))
    num_experts = _coalesce_int(
        config.get("num_experts"),
        config.get("n_routed_experts"),
        config.get("num_local_experts"),
        config.get("moe_num_experts"),
    )
    num_experts_per_tok = _coalesce_int(
        config.get("num_experts_per_tok"),
        config.get("num_experts_per_token"),
        config.get("moe_top_k"),
        config.get("top_k_experts"),
    )
    n_group = _coalesce_int(config.get("n_group"), config.get("moe_num_groups"))
    topk_group = _coalesce_int(config.get("topk_group"), config.get("moe_topk_group"))
    dsa_topk = _coalesce_int(
        config.get("dsa_topk"), config.get("num_select_blocks"), config.get("topk"), 2048
    )
    gdn_q_heads = _coalesce_int(
        config.get("gdn_q_heads"),
        config.get("qk_num_heads"),
        config.get("num_attention_heads"),
    )
    gdn_v_heads = _coalesce_int(
        config.get("gdn_v_heads"),
        config.get("num_value_heads"),
        config.get("v_num_heads"),
    )
    mamba_nheads = _coalesce_int(
        config.get("mamba_num_heads"), config.get("ssm_num_heads"), config.get("n_ssm_heads")
    )
    mamba_head_dim = _coalesce_int(
        config.get("mamba_head_dim"), config.get("ssm_head_dim"), head_dim
    )
    mamba_dstate = _coalesce_int(
        config.get("mamba_d_state"), config.get("ssm_state_size"), config.get("d_state")
    )
    mamba_ngroups = _coalesce_int(
        config.get("mamba_n_groups"), config.get("ssm_ngroups"), config.get("num_ssm_groups")
    )
    uses_gdn = "qwen3next" in arch_text or "gdn" in arch_text or any(
        str(key).startswith("gdn_") for key in config
    )
    uses_mamba = "mamba" in arch_text or any(
        value is not None for value in (mamba_nheads, mamba_head_dim, mamba_dstate, mamba_ngroups)
    )
    uses_dsa = (
        str(config.get("attention_type", "")).lower() == "dsa"
        or "deepseekv32" in arch_text
        or "usesparseattention" in keys_text
        or _slugify(model_name or hf_repo_id or "").startswith("deepseek-v3.2")
    )
    uses_mla = not uses_dsa and (
        ("deepseekv2" in arch_text or "deepseekv3" in arch_text)
        or (q_lora_rank is not None and kv_lora_rank is not None)
    )
    uses_gqa = bool(num_attention_heads and num_key_value_heads) and (
        not uses_mla or uses_gdn or uses_mamba
    )
    if not any((uses_gqa, uses_mla, uses_dsa, uses_gdn, uses_mamba)) and num_attention_heads:
        uses_gqa = True
    if uses_gdn and gdn_v_heads is None and gdn_q_heads is not None:
        notes.append("GDN value-head count missing; defaulting to q-head count for inventory.")
        gdn_v_heads = gdn_q_heads
    has_moe = bool(num_experts) or "moe" in arch_text or "n_routed_experts" in config
    if hidden_size is None:
        notes.append("hidden_size missing from config.")
    if num_attention_heads is None and not uses_mamba:
        notes.append("num_attention_heads missing from config.")
    model_slug = _canonical_model_slug(model_name, hf_repo_id, config)
    return ModelAnalysis(
        model_slug=model_slug,
        architectures=architectures,
        num_hidden_layers=_coalesce_int(config.get("num_hidden_layers"), config.get("n_layers")),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        n_group=n_group,
        topk_group=topk_group,
        dsa_topk=dsa_topk,
        gdn_q_heads=gdn_q_heads,
        gdn_v_heads=gdn_v_heads,
        mamba_nheads=mamba_nheads,
        mamba_head_dim=mamba_head_dim,
        mamba_dstate=mamba_dstate,
        mamba_ngroups=mamba_ngroups,
        uses_gqa=uses_gqa,
        uses_mla=uses_mla,
        uses_dsa=uses_dsa,
        uses_gdn=uses_gdn,
        uses_mamba=uses_mamba,
        has_moe=has_moe,
        notes=notes,
    )
def build_expected_definitions(
    analysis: ModelAnalysis, tp_values: Iterable[int], ep_values: Iterable[int]
) -> list[KernelExpectation]:
    expectations: dict[str, KernelExpectation] = {}
    def add(
        name: str,
        op_type: str,
        category: str,
        tp: int | None = None,
        ep: int | None = None,
        notes: list[str] | None = None,
    ) -> None:
        expectations.setdefault(
            name,
            KernelExpectation(
                name=name,
                op_type=op_type,
                category=category,
                tp=tp,
                ep=ep,
                notes=notes or [],
            ),
        )
    if analysis.hidden_size:
        add(f"rmsnorm_h{analysis.hidden_size}", "rmsnorm", "norm")
        add(f"fused_add_rmsnorm_h{analysis.hidden_size}", "rmsnorm", "norm")
    for extra_norm in (analysis.q_lora_rank, analysis.kv_lora_rank):
        if extra_norm:
            add(f"rmsnorm_h{extra_norm}", "rmsnorm", "norm")
    if (
        analysis.uses_gqa
        and analysis.hidden_size
        and analysis.intermediate_size
        and analysis.head_dim
        and analysis.num_attention_heads
        and analysis.num_key_value_heads
    ):
        qkv_proj = (
            analysis.num_attention_heads + 2 * analysis.num_key_value_heads
        ) * analysis.head_dim
        add(f"gemm_n{qkv_proj}_k{analysis.hidden_size}", "gemm", "gemm")
        add(f"gemm_n{analysis.hidden_size}_k{analysis.hidden_size}", "gemm", "gemm")
        add(f"gemm_n{2 * analysis.intermediate_size}_k{analysis.hidden_size}", "gemm", "gemm")
        add(f"gemm_n{analysis.hidden_size}_k{analysis.intermediate_size}", "gemm", "gemm")
    for tp in sorted(set(tp_values)):
        tp_notes: list[str] = []
        if analysis.uses_gqa and analysis.head_dim:
            heads = _split_axis(analysis.num_attention_heads, tp, "num_attention_heads", tp_notes)
            kv_heads = _split_axis(
                analysis.num_key_value_heads, tp, "num_key_value_heads", tp_notes
            )
            if heads and kv_heads:
                add(
                    f"gqa_paged_prefill_causal_h{heads}_kv{kv_heads}_d{analysis.head_dim}_ps1",
                    "gqa_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
                add(
                    f"gqa_paged_prefill_causal_h{heads}_kv{kv_heads}_d{analysis.head_dim}_ps64",
                    "gqa_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
                add(
                    f"gqa_paged_decode_h{heads}_kv{kv_heads}_d{analysis.head_dim}_ps1",
                    "gqa_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
                add(
                    f"gqa_paged_decode_h{heads}_kv{kv_heads}_d{analysis.head_dim}_ps64",
                    "gqa_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
                add(
                    f"gqa_ragged_prefill_causal_h{heads}_kv{kv_heads}_d{analysis.head_dim}",
                    "gqa_ragged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
        if analysis.uses_mla and analysis.qk_rope_head_dim and analysis.kv_lora_rank:
            heads = _split_axis(analysis.num_attention_heads, tp, "num_attention_heads", tp_notes)
            ckv = analysis.kv_lora_rank + analysis.qk_rope_head_dim
            kpe = analysis.qk_rope_head_dim
            qk_total = (analysis.qk_nope_head_dim or analysis.head_dim or 0) + analysis.qk_rope_head_dim
            vo_dim = analysis.head_dim or 0
            if heads and ckv and kpe:
                add(
                    f"mla_paged_prefill_causal_h{heads}_ckv{ckv}_kpe{kpe}_ps1",
                    "mla_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
                add(
                    f"mla_paged_prefill_causal_h{heads}_ckv{ckv}_kpe{kpe}_ps64",
                    "mla_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
                add(
                    f"mla_paged_decode_h{heads}_ckv{ckv}_kpe{kpe}_ps1",
                    "mla_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
                add(
                    f"mla_paged_decode_h{heads}_ckv{ckv}_kpe{kpe}_ps64",
                    "mla_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
            if heads and qk_total and vo_dim:
                add(
                    f"mla_ragged_prefill_causal_h{heads}_qk{qk_total}_vo{vo_dim}",
                    "mla_ragged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
        if analysis.uses_dsa and analysis.qk_rope_head_dim and analysis.kv_lora_rank:
            heads = _split_axis(analysis.num_attention_heads, tp, "num_attention_heads", tp_notes)
            ckv = analysis.kv_lora_rank + analysis.qk_rope_head_dim
            kpe = analysis.qk_rope_head_dim
            if heads and analysis.head_dim and analysis.dsa_topk:
                add(
                    f"dsa_topk_indexer_fp8_h{heads}_d{analysis.head_dim}_topk{analysis.dsa_topk}_ps64",
                    "dsa_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
            if heads and ckv and kpe and analysis.dsa_topk:
                add(
                    f"dsa_sparse_attention_h{heads}_ckv{ckv}_kpe{kpe}_topk{analysis.dsa_topk}_ps1",
                    "dsa_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
                add(
                    f"dsa_sparse_attention_h{heads}_ckv{ckv}_kpe{kpe}_topk{analysis.dsa_topk}_ps64",
                    "dsa_paged",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
        if analysis.uses_gdn and analysis.head_dim:
            q_heads = _split_axis(analysis.gdn_q_heads, tp, "gdn_q_heads", tp_notes)
            v_heads = _split_axis(analysis.gdn_v_heads, tp, "gdn_v_heads", tp_notes)
            if q_heads and v_heads:
                add(
                    f"gdn_prefill_qk{q_heads}_v{v_heads}_d{analysis.head_dim}_k_last",
                    "gdn",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
                add(
                    f"gdn_decode_qk{q_heads}_v{v_heads}_d{analysis.head_dim}_k_last",
                    "gdn",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
                add(
                    f"gdn_mtp_qk{q_heads}_v{v_heads}_d{analysis.head_dim}_k_last",
                    "gdn",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
        if analysis.uses_mamba:
            nheads = _split_axis(analysis.mamba_nheads, tp, "mamba_nheads", tp_notes)
            ngroups = _split_axis(analysis.mamba_ngroups, tp, "mamba_ngroups", tp_notes)
            if nheads and ngroups and analysis.mamba_head_dim and analysis.mamba_dstate:
                add(
                    f"mamba_ssu_decode_h{nheads}_d{analysis.mamba_head_dim}_s{analysis.mamba_dstate}_ng{ngroups}",
                    "mamba_ssu",
                    "attention",
                    tp=tp,
                    notes=tp_notes.copy(),
                )
    if analysis.has_moe and analysis.hidden_size and analysis.moe_intermediate_size:
        for ep in sorted(set(ep_values)):
            ep_notes: list[str] = []
            local_experts = _split_axis(analysis.num_experts, ep, "num_experts", ep_notes)
            topk = analysis.num_experts_per_tok or 1
            if not local_experts:
                continue
            if analysis.n_group and analysis.topk_group:
                add(
                    (
                        "moe_fp8_block_scale_ds_routing_"
                        f"topk{topk}_ng{analysis.n_group}_kg{analysis.topk_group}_"
                        f"e{local_experts}_h{analysis.hidden_size}_i{analysis.moe_intermediate_size}"
                    ),
                    "moe",
                    "moe",
                    ep=ep,
                    notes=ep_notes.copy(),
                )
            else:
                add(
                    (
                        "moe_fp8_block_scale_renorm_"
                        f"topk{topk}_e{local_experts}_h{analysis.hidden_size}_i{analysis.moe_intermediate_size}"
                    ),
                    "moe",
                    "moe",
                    ep=ep,
                    notes=ep_notes.copy(),
                )
    if analysis.vocab_size:
        add(f"top_k_sampling_from_probs_v{analysis.vocab_size}", "sampling", "sampling")
        add(f"top_k_top_p_sampling_from_probs_v{analysis.vocab_size}", "sampling", "sampling")
        add(f"top_p_sampling_from_probs_v{analysis.vocab_size}", "sampling", "sampling")
    return sorted(
        expectations.values(),
        key=lambda item: (item.category, item.op_type, item.name, item.tp or 0, item.ep or 0),
    )
def classify_expected_definitions(
    expectations: list[KernelExpectation], definitions_root: str | Path
) -> dict[str, Any]:
    root = _resolve_definitions_dir(definitions_root)
    records: list[dict[str, Any]] = []
    existing = 0
    missing = 0
    for item in expectations:
        direct_path = root / item.op_type / f"{item.name}.json"
        resolved_path = direct_path
        exists = direct_path.exists()
        fallback_matches: list[str] = []
        if not exists and root.exists():
            matches = sorted(root.glob(f"**/{item.name}.json"))
            if matches:
                exists = True
                resolved_path = matches[0]
                fallback_matches = [str(match) for match in matches]
        if exists:
            existing += 1
        else:
            missing += 1
        record = asdict(item)
        record.update(
            {
                "exists": exists,
                "path": str(resolved_path),
                "fallback_matches": fallback_matches,
            }
        )
        records.append(record)
    by_op_type: dict[str, dict[str, int]] = {}
    for record in records:
        stats = by_op_type.setdefault(record["op_type"], {"existing": 0, "missing": 0})
        stats["existing" if record["exists"] else "missing"] += 1
    return {
        "definitions_root": str(root),
        "expected_definition_count": len(records),
        "existing_definition_count": existing,
        "missing_definition_count": missing,
        "by_op_type": by_op_type,
        "expected_definitions": records,
    }
def analyze_model_inventory_from_config(
    config: dict[str, Any],
    model_name: str | None,
    hf_repo_id: str | None,
    tp_values: Iterable[int],
    ep_values: Iterable[int],
    definitions_root: str | Path,
) -> dict[str, Any]:
    analysis = analyze_model_config(config, model_name=model_name, hf_repo_id=hf_repo_id)
    expectations = build_expected_definitions(analysis, tp_values=tp_values, ep_values=ep_values)
    classification = classify_expected_definitions(expectations, definitions_root)
    return {
        "model_slug": analysis.model_slug,
        "analysis": asdict(analysis),
        "tp_values": sorted(set(tp_values)) or [1],
        "ep_values": sorted(set(ep_values)) or [1],
        **classification,
    }
def discover_recent_models(
    days: int = 30,
    sglang_repo: str | Path = SGLANG_REPO_DEFAULT,
    cookbook_root: str | Path = COOKBOOK_REPO_DEFAULT,
    coverage_path: str | Path = COVERAGE_DOC_DEFAULT,
) -> dict[str, Any]:
    tracked = {_normalize_key(name) for name in _extract_summary_models(_repo_path(coverage_path))}
    candidates: list[dict[str, Any]] = []
    def add_candidate(slug: str, source: str, path: str) -> None:
        key = _normalize_key(slug)
        candidates.append(
            {
                "model_slug": slug,
                "source": source,
                "path": path,
                "already_tracked": key in tracked,
            }
        )
    sglang_path = _repo_path(sglang_repo)
    if (sglang_path / ".git").exists():
        result = _run_list(
            [
                "git",
                "-C",
                str(sglang_path),
                "log",
                f"--since={days} days ago",
                "--name-status",
                "--diff-filter=A",
                "--",
                "python/sglang/srt/models/*.py",
            ],
            timeout=30,
        )
        if result["returncode"] == 0:
            for line in result["stdout"].splitlines():
                if not line.startswith("A\t"):
                    continue
                relpath = line.split("\t", 1)[1].strip()
                add_candidate(_slugify(Path(relpath).stem), "sglang_day0", relpath)
    cookbook_path = _repo_path(cookbook_root)
    if (cookbook_path / ".git").exists():
        result = _run_list(
            [
                "git",
                "-C",
                str(cookbook_path),
                "log",
                f"--since={days} days ago",
                "--name-status",
                "--diff-filter=A",
                "--",
                "data/models/generated/v0.5.6/*.yaml",
            ],
            timeout=30,
        )
        if result["returncode"] == 0:
            for line in result["stdout"].splitlines():
                if not line.startswith("A\t"):
                    continue
                relpath = line.split("\t", 1)[1].strip()
                add_candidate(_slugify(Path(relpath).stem), "sgl_cookbook", relpath)
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate["model_slug"], candidate["source"])
        deduped[key] = candidate
    return {"days": days, "tracked_models": sorted(tracked), "candidates": list(deduped.values())}

_ADAPTER_CLASS_PARTS = {
    "gqa": "GQA",
    "mla": "MLA",
    "dsa": "DSA",
    "gdn": "GDN",
    "mamba": "Mamba",
    "ssu": "SSU",
    "rmsnorm": "RMSNorm",
    "moe": "MoE",
}

ADAPTER_EXPECTATIONS_BY_OP: dict[str, list[AdapterExpectation]] = {
    "gqa_paged": [
        AdapterExpectation(
            file_stem="gqa_paged_prefill",
            class_name="GQAPagedPrefillAdapter",
            rationale="Paged GQA prefill uses a dedicated BatchPrefill wrapper patch.",
            test_globs=("test_gqa_paged_prefill.py",),
            test_required=True,
        ),
        AdapterExpectation(
            file_stem="gqa_paged_decode",
            class_name="GQAPagedDecodeAdapter",
            rationale="Paged GQA decode uses a dedicated BatchDecode wrapper patch.",
            test_globs=("test_gqa_paged_decode.py",),
            test_required=True,
        ),
    ],
    "gqa_ragged": [
        AdapterExpectation(
            file_stem="ragged_prefill",
            class_name="RaggedPrefillAdapter",
            rationale="Shared ragged prefill adapter handles causal GQA ragged workloads.",
            test_globs=("test_ragged_prefill.py",),
            shared_with=("mla_ragged",),
            test_required=True,
        )
    ],
    "mla_paged": [
        AdapterExpectation(
            file_stem="mla_paged",
            class_name="MLAPagedAdapter",
            rationale="Paged MLA has a dedicated adapter for decode and prefill wrapper routing.",
            test_globs=("test_mla_paged.py",),
            test_required=True,
        )
    ],
    "mla_ragged": [
        AdapterExpectation(
            file_stem="ragged_prefill",
            class_name="RaggedPrefillAdapter",
            rationale="Shared ragged prefill adapter is also used for MLA ragged prefill.",
            test_globs=("test_ragged_prefill.py",),
            shared_with=("gqa_ragged",),
            test_required=True,
        )
    ],
    "gemm": [
        AdapterExpectation(
            file_stem="linear",
            class_name="LinearAdapter",
            rationale="GEMM traces are patched through torch.nn.functional.linear.",
        )
    ],
    "rmsnorm": [
        AdapterExpectation(
            file_stem="rmsnorm",
            class_name="RMSNormAdapter",
            rationale="FlashInfer RMSNorm apply path uses a dedicated fused_add_rmsnorm adapter.",
            test_globs=("test_rmsnorm.py",),
            test_required=True,
        )
    ],
}

ADAPTER_NOT_REQUIRED_OP_TYPES = {
    "dsa_paged",
    "gdn",
    "mamba_ssu",
    "moe",
    "rope",
    "sampling",
}


def _camelize_adapter_class(file_stem: str) -> str:
    parts = [part for part in file_stem.split("_") if part]
    rendered = [_ADAPTER_CLASS_PARTS.get(part, part.capitalize()) for part in parts]
    return "".join(rendered) + "Adapter"


def _default_adapter_expectation(op_type: str) -> AdapterExpectation:
    return AdapterExpectation(
        file_stem=op_type,
        class_name=_camelize_adapter_class(op_type),
        rationale=(
            f"No adapter is currently declared for op_type '{op_type}'. "
            "If apply/runtime patching is needed for this kernel family, add a dedicated adapter."
        ),
    )


def _adapter_expectations_for_op(op_type: str) -> list[AdapterExpectation]:
    if op_type in ADAPTER_NOT_REQUIRED_OP_TYPES:
        return []
    return ADAPTER_EXPECTATIONS_BY_OP.get(op_type, [_default_adapter_expectation(op_type)])


def check_adapter_support(
    op_type: str,
    integration_root: str | Path = INTEGRATION_ROOT_DEFAULT,
    test_root: str | Path = INTEGRATION_TEST_ROOT_DEFAULT,
) -> dict[str, Any]:
    root = _repo_path(integration_root)
    adapters_dir = root / "adapters"
    init_path = root / "__init__.py"
    test_dir = _repo_path(test_root)

    expectations = _adapter_expectations_for_op(op_type)
    if not expectations:
        return {
            "op_type": op_type,
            "integration_root": str(root),
            "adapter_required": False,
            "status": "not_required",
            "supported": True,
            "components": [],
        }

    init_text = init_path.read_text() if init_path.exists() else ""
    components: list[dict[str, Any]] = []

    for expectation in expectations:
        module_path = adapters_dir / f"{expectation.file_stem}.py"
        module_text = module_path.read_text() if module_path.exists() else ""
        import_marker = f"from .adapters.{expectation.file_stem} import {expectation.class_name}"
        install_marker = f"{expectation.class_name}()"
        test_matches: list[str] = []
        for pattern in expectation.test_globs:
            test_matches.extend(str(path) for path in sorted(test_dir.glob(pattern)))

        component = {
            "file_stem": expectation.file_stem,
            "class_name": expectation.class_name,
            "module_path": str(module_path),
            "module_exists": module_path.exists(),
            "class_declared": f"class {expectation.class_name}" in module_text,
            "registry_import_present": import_marker in init_text,
            "registry_install_present": install_marker in init_text,
            "test_matches": sorted(set(test_matches)),
            "test_required": expectation.test_required,
            "shared_with": list(expectation.shared_with),
            "rationale": expectation.rationale,
        }
        component["installed"] = all(
            (
                component["module_exists"],
                component["class_declared"],
                component["registry_import_present"],
                component["registry_install_present"],
            )
        )
        components.append(component)

    return {
        "op_type": op_type,
        "integration_root": str(root),
        "adapter_required": True,
        "status": "checked",
        "supported": all(component["installed"] for component in components),
        "components": components,
    }


def infer_evaluator_expectation(definition_name: str, op_type: str) -> EvaluatorExpectation:
    lowered_name = definition_name.lower()
    if lowered_name.startswith("dsa_topk_indexer") or "topk_indexer" in lowered_name:
        return EvaluatorExpectation(
            module_stem="dsa_topk_indexer",
            class_name="DsaTopkIndexerEvaluator",
            reason="DSA top-k indexer correctness is order-insensitive and uses custom validation.",
            specialized_required=True,
            test_globs=("test_dsa_topk_indexer_evaluator.py",),
            test_required=True,
        )
    if op_type == "dsa_paged" or lowered_name.startswith("dsa_sparse_attention"):
        return EvaluatorExpectation(
            module_stem="dsa_sparse_attention",
            class_name="DsaSparseAttentionEvaluator",
            reason="DSA sparse attention has optional LSE handling and a specialized comparison path.",
            specialized_required=True,
            test_globs=("test_dsa_sparse_attention_evaluator.py",),
            test_required=True,
        )
    if op_type == "sampling" or "sampling" in lowered_name:
        return EvaluatorExpectation(
            module_stem="sampling",
            class_name="SamplingEvaluator",
            reason="Sampling kernels need statistical validation rather than direct tensor equality.",
            specialized_required=True,
            test_globs=("test_evaluator.py",),
            test_required=True,
        )
    if op_type == "moe" and any(token in lowered_name for token in ("fp8", "fp4", "lowbit")):
        return EvaluatorExpectation(
            module_stem="lowbit",
            class_name="LowBitEvaluator",
            reason="Low-bit / quantized MoE kernels need relaxed matched-ratio correctness thresholds.",
            specialized_required=True,
            test_globs=("test_evaluator.py",),
            test_required=True,
        )
    return EvaluatorExpectation(
        module_stem="default",
        class_name="DefaultEvaluator",
        reason="Default evaluator is sufficient for standard tensor-valued kernels.",
        specialized_required=False,
        test_globs=("test_evaluator.py",),
        test_required=True,
    )


def check_evaluator_support(
    definition_name: str,
    op_type: str,
    evaluator_root: str | Path = EVALUATOR_ROOT_DEFAULT,
    bench_test_root: str | Path = BENCH_TEST_ROOT_DEFAULT,
) -> dict[str, Any]:
    root = _repo_path(evaluator_root)
    init_path = root / "__init__.py"
    registry_path = root / "registry.py"
    test_dir = _repo_path(bench_test_root)

    expectation = infer_evaluator_expectation(definition_name, op_type)
    module_path = root / f"{expectation.module_stem}.py"
    module_text = module_path.read_text() if module_path.exists() else ""
    init_text = init_path.read_text() if init_path.exists() else ""
    registry_text = registry_path.read_text() if registry_path.exists() else ""

    import_marker = f"from .{expectation.module_stem} import {expectation.class_name}"
    export_marker = f'"{expectation.class_name}"'
    registry_entry_present = expectation.class_name in registry_text
    class_declared = f"class {expectation.class_name}" in module_text
    test_matches: list[str] = []
    for pattern in expectation.test_globs:
        test_matches.extend(str(path) for path in sorted(test_dir.glob(pattern)))

    specialized_installed = all(
        (
            module_path.exists(),
            class_declared,
            import_marker in init_text,
            import_marker in registry_text or expectation.class_name == "DefaultEvaluator",
            registry_entry_present if expectation.specialized_required else True,
            bool(test_matches) if expectation.test_required else True,
        )
    )

    default_available = all(
        (
            (root / "default.py").exists(),
            (root / "registry.py").exists(),
            "DefaultEvaluator" in init_text,
            "DefaultEvaluator" in registry_text,
            bool(test_matches) if expectation.test_required else True,
        )
    )

    supported = specialized_installed if expectation.specialized_required else default_available

    return {
        "definition_name": definition_name,
        "op_type": op_type,
        "evaluator_root": str(root),
        "expected_evaluator": {
            "module_stem": expectation.module_stem,
            "class_name": expectation.class_name,
            "reason": expectation.reason,
            "specialized_required": expectation.specialized_required,
            "test_required": expectation.test_required,
        },
        "module_path": str(module_path),
        "module_exists": module_path.exists(),
        "class_declared": class_declared,
        "init_import_present": import_marker in init_text,
        "init_export_present": export_marker in init_text or expectation.class_name in init_text,
        "registry_import_present": import_marker in registry_text or expectation.class_name == "DefaultEvaluator",
        "registry_entry_present": registry_entry_present,
        "test_matches": sorted(set(test_matches)),
        "default_available": default_available,
        "supported": supported,
    }


def check_bench_extension_support(
    definition_name: str,
    op_type: str,
    integration_root: str | Path = INTEGRATION_ROOT_DEFAULT,
    evaluator_root: str | Path = EVALUATOR_ROOT_DEFAULT,
    integration_test_root: str | Path = INTEGRATION_TEST_ROOT_DEFAULT,
    bench_test_root: str | Path = BENCH_TEST_ROOT_DEFAULT,
) -> dict[str, Any]:
    adapter = check_adapter_support(
        op_type=op_type, integration_root=integration_root, test_root=integration_test_root
    )
    evaluator = check_evaluator_support(
        definition_name=definition_name,
        op_type=op_type,
        evaluator_root=evaluator_root,
        bench_test_root=bench_test_root,
    )
    return {
        "definition_name": definition_name,
        "op_type": op_type,
        "supported": adapter["supported"] and evaluator["supported"],
        "adapter": adapter,
        "evaluator": evaluator,
    }


FLASHINFER_SUPPORT_RULES: dict[str, list[tuple[str, str]]] = {
    "rmsnorm": [("flashinfer/norm.py", "rmsnorm")],
    "gqa_paged": [("flashinfer/decode.py", "BatchDecode"), ("flashinfer/prefill.py", "BatchPrefill")],
    "gqa_ragged": [("flashinfer/prefill.py", "RaggedKVCache")],
    "mla_paged": [("flashinfer/mla.py", "MLA")],
    "mla_ragged": [("flashinfer/prefill.py", "RaggedKVCache")],
    "dsa_paged": [("flashinfer/sparse.py", "Sparse")],
    "gdn": [("flashinfer/gdn.py", "gated_delta_rule"), ("flashinfer/gdn", "")],
    "moe": [("flashinfer/fused_moe", "")],
    "sampling": [("flashinfer/sampling.py", "sampling")],
    "mamba_ssu": [("flashinfer/mamba.py", "selective_state_update")],
    "rope": [("flashinfer/rope.py", "rope")],
}
def check_flashinfer_support(op_type: str, flashinfer_repo: str | Path) -> dict[str, Any]:
    if op_type == "gemm":
        return {
            "op_type": op_type,
            "supported": True,
            "status": "builtin",
            "checked_paths": [],
            "matches": ["gemm uses torch / baseline path"],
            "tests": [],
        }
    root = _repo_path(flashinfer_repo)
    if not root.exists():
        return {
            "op_type": op_type,
            "supported": False,
            "status": "repo_missing",
            "checked_paths": [],
            "matches": [],
            "tests": [],
        }
    matches: list[str] = []
    checked_paths: list[str] = []
    for relpath, needle in FLASHINFER_SUPPORT_RULES.get(op_type, []):
        path = root / relpath
        checked_paths.append(str(path))
        if not path.exists():
            continue
        if path.is_dir():
            matches.append(str(path))
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            content = ""
        if not needle or needle in content:
            matches.append(str(path))
    tests_root = root / "tests"
    test_matches = []
    if tests_root.exists():
        test_matches = sorted(str(path) for path in tests_root.glob(f"**/*{op_type}*"))
    return {
        "op_type": op_type,
        "supported": bool(matches or test_matches),
        "status": "checked",
        "checked_paths": checked_paths,
        "matches": matches,
        "tests": test_matches[:50],
    }

def tool_check_adapter_support(
    op_type: str,
    integration_root: str = INTEGRATION_ROOT_DEFAULT,
    test_root: str = INTEGRATION_TEST_ROOT_DEFAULT,
) -> str:
    return _json(check_adapter_support(op_type, integration_root=integration_root, test_root=test_root))


def tool_check_evaluator_support(
    definition_name: str,
    op_type: str,
    evaluator_root: str = EVALUATOR_ROOT_DEFAULT,
    bench_test_root: str = BENCH_TEST_ROOT_DEFAULT,
) -> str:
    return _json(
        check_evaluator_support(
            definition_name=definition_name,
            op_type=op_type,
            evaluator_root=evaluator_root,
            bench_test_root=bench_test_root,
        )
    )


def tool_check_bench_extension_support(
    definition_name: str,
    op_type: str,
    integration_root: str = INTEGRATION_ROOT_DEFAULT,
    evaluator_root: str = EVALUATOR_ROOT_DEFAULT,
    integration_test_root: str = INTEGRATION_TEST_ROOT_DEFAULT,
    bench_test_root: str = BENCH_TEST_ROOT_DEFAULT,
) -> str:
    return _json(
        check_bench_extension_support(
            definition_name=definition_name,
            op_type=op_type,
            integration_root=integration_root,
            evaluator_root=evaluator_root,
            integration_test_root=integration_test_root,
            bench_test_root=bench_test_root,
        )
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────────────────────────────────────
def tool_run_shell(command: str, timeout_seconds: int = 120) -> str:
    return _fmt(_run(command, timeout=timeout_seconds))
def tool_read_file(path: str) -> str:
    p = _repo_path(path)
    if not p.exists():
        return f"File not found: {p}"
    try:
        content = p.read_text()
    except Exception as exc:  # pragma: no cover - defensive
        return f"Error reading {p}: {exc}"
    if len(content) > 16000:
        content = content[:8000] + "\n...[truncated]...\n" + content[-8000:]
    return content
def tool_write_file(path: str, content: str) -> str:
    p = _repo_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} bytes to {p}"
    except Exception as exc:  # pragma: no cover - defensive
        return f"Error writing {p}: {exc}"
def tool_list_files(directory: str, pattern: str = "*") -> str:
    d = _repo_path(directory)
    if not d.exists():
        return f"Directory not found: {d}"
    files = sorted(d.glob(pattern))
    return "\n".join(str(path.relative_to(REPO_ROOT)) for path in files[:200])
def tool_git_op(operation: str, worktree: str | None = None) -> str:
    cwd = str(_repo_path(worktree)) if worktree else str(REPO_ROOT)
    return _fmt(_run(f"git {operation}", timeout=60, cwd=cwd))
def tool_find_model_path(model_name: str) -> str:
    matches = find_model_paths(model_name)
    if not matches:
        return (
            f"No cached model found matching '{model_name}'. "
            "Provide --model-path explicitly or ensure the model exists in the HF cache."
        )
    return _json(matches)
def tool_fetch_model_config(
    model_name: str | None = None,
    hf_repo_id: str | None = None,
    model_path: str | None = None,
) -> str:
    return _json(load_model_config(model_name=model_name, hf_repo_id=hf_repo_id, model_path=model_path))
def tool_discover_parallel_configs(
    model_name: str,
    cookbook_root: str = COOKBOOK_REPO_DEFAULT,
    hf_repo_id: str | None = None,
) -> str:
    return _json(discover_parallel_configs(model_name, cookbook_root=cookbook_root, hf_repo_id=hf_repo_id))
def tool_analyze_model_inventory(
    model_name: str,
    hf_repo_id: str | None = None,
    model_path: str | None = None,
    tp_values: list[int] | None = None,
    ep_values: list[int] | None = None,
    definitions_root: str = DEFINITIONS_ROOT_DEFAULT,
    cookbook_root: str = COOKBOOK_REPO_DEFAULT,
) -> str:
    config_result = load_model_config(model_name=model_name, hf_repo_id=hf_repo_id, model_path=model_path)
    if not config_result.get("ok"):
        return _json(config_result)
    if tp_values is None or ep_values is None:
        parallel = discover_parallel_configs(
            model_name=model_name, cookbook_root=cookbook_root, hf_repo_id=hf_repo_id
        )
        auto_tp = parallel["tp_values"]
        auto_ep = parallel["ep_values"]
    else:
        parallel = {"tp_values": tp_values, "ep_values": ep_values, "sources": []}
        auto_tp = tp_values
        auto_ep = ep_values
    inventory = analyze_model_inventory_from_config(
        config=config_result["config"],
        model_name=model_name,
        hf_repo_id=hf_repo_id,
        tp_values=auto_tp,
        ep_values=auto_ep,
        definitions_root=definitions_root,
    )
    inventory["config_source"] = {
        "source": config_result["source"],
        "config_path": config_result["config_path"],
    }
    inventory["parallelism_source"] = parallel
    return _json(inventory)
def tool_discover_recent_models(
    days: int = 30,
    sglang_repo: str = SGLANG_REPO_DEFAULT,
    cookbook_root: str = COOKBOOK_REPO_DEFAULT,
    coverage_path: str = COVERAGE_DOC_DEFAULT,
) -> str:
    return _json(
        discover_recent_models(
            days=days,
            sglang_repo=sglang_repo,
            cookbook_root=cookbook_root,
            coverage_path=coverage_path,
        )
    )
def tool_check_flashinfer_support(
    op_type: str, flashinfer_repo: str = FLASHINFER_REPO_DEFAULT
) -> str:
    return _json(check_flashinfer_support(op_type, flashinfer_repo))
def tool_check_gpus() -> str:
    result = _run(
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits",
        timeout=15,
    )
    if result["returncode"] != 0:
        return f"nvidia-smi failed: {result['stderr']}"
    rows = []
    for line in result["stdout"].strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            idx, used, total, util = parts
            rows.append(f"GPU {idx}: {used}/{total} MiB used, {util}% util")
    return "GPU status:\n" + "\n".join(rows)
def tool_kill_sglang() -> str:
    result = _run("pkill -f 'sglang.launch_server' || true", timeout=10)
    time.sleep(2)
    return f"Sent SIGTERM to sglang processes. {_fmt(result)}"
def tool_run_tests(test_file: str) -> str:
    cmd = f"conda run -n {CONDA_ENV} python -m pytest {shlex.quote(test_file)} -v 2>&1"
    return _fmt(_run(cmd, timeout=180))
def tool_run_collection(
    model_path: str,
    definitions: list[str],
    flashinfer_trace_dir: str,
    tp: int = 1,
    quantization: str | None = None,
    ep: int = 1,
    extra_args: list[str] | None = None,
) -> str:
    defs_str = " ".join(definitions)
    cmd_parts = [
        f"conda run -n {CONDA_ENV} python scripts/collect_workloads.py sglang",
        f"--model-path {shlex.quote(model_path)}",
        f"--definitions {defs_str}",
        f"--flashinfer-trace-dir {shlex.quote(flashinfer_trace_dir)}",
        "--replace --skip-install",
    ]
    if tp and tp != 1:
        cmd_parts.append(f"--tp {tp}")
    if quantization:
        cmd_parts.append(f"--quantization {quantization}")
    if ep > 1:
        cmd_parts.append(f"--ep {ep}")
    if extra_args:
        cmd_parts.extend(extra_args)
    cmd = " ".join(cmd_parts)
    print(f"\n[agent] Running collection: {cmd}\n", flush=True)
    return _fmt(_run(cmd, timeout=3600))
def tool_run_baseline_eval(definition: str, flashinfer_trace_dir: str) -> str:
    cmd = (
        f"conda run -n {CONDA_ENV} python -m flashinfer_bench run "
        f"--local {shlex.quote(flashinfer_trace_dir)} "
        f"--definitions {shlex.quote(definition)} "
        f"--solutions baseline --save-results --warmup-runs 0 --iterations 1 --num-trials 1"
    )
    return _fmt(_run(cmd, timeout=600))
def tool_check_workloads(flashinfer_trace_dir: str, definition: str) -> str:
    trace_dir = _repo_path(flashinfer_trace_dir)
    results: dict[str, Any] = {}
    found = list(trace_dir.glob(f"definitions/**/{definition}.json"))
    op_type = found[0].parent.name if found else "unknown"
    workload_file = trace_dir / "workloads" / op_type / f"{definition}.jsonl"
    results["workload_file"] = str(workload_file)
    results["workload_exists"] = workload_file.exists()
    if workload_file.exists():
        lines = [line for line in workload_file.read_text().splitlines() if line.strip()]
        results["workload_count"] = len(lines)
    blob_dir = trace_dir / "blob" / "workloads" / op_type / definition
    blobs = list(blob_dir.glob("*.safetensors")) if blob_dir.exists() else []
    results["blob_count"] = len(blobs)
    baseline_dir = trace_dir / "solutions" / "baseline" / op_type / definition
    baseline_files = list(baseline_dir.glob("*.json")) if baseline_dir.exists() else []
    results["baseline_solution_exists"] = len(baseline_files) > 0
    results["baseline_solution_files"] = [path.name for path in baseline_files]
    trace_files = list((trace_dir / "traces").glob(f"*/{op_type}/{definition}.jsonl"))
    results["trace_exists"] = bool(trace_files)
    if trace_files:
        records = []
        for trace_file in trace_files:
            records.extend(json.loads(line) for line in trace_file.read_text().splitlines() if line.strip())
        statuses = [record.get("evaluation", {}).get("status") for record in records]
        results["trace_statuses"] = {
            "passed": statuses.count("PASSED"),
            "failed": statuses.count("FAILED"),
            "total": len(statuses),
        }
    return _json(results)
def tool_create_github_pr(title: str, body: str, branch: str, base: str = "main") -> str:
    body_escaped = body.replace("'", "'\\''")
    cmd = f"gh pr create --title '{title}' --body $'{body_escaped}' --base {base} --head {branch}"
    return _fmt(_run(cmd, timeout=60))
def tool_create_hf_pr(
    repo_id: str, title: str, description: str, branch: str, worktree: str
) -> str:
    script_create = textwrap.dedent(
        f"""\
        from huggingface_hub import HfApi
        api = HfApi()
        pr = api.create_pull_request(
            repo_id={repo_id!r},
            repo_type='dataset',
            title={title!r},
            description={description!r},
        )
        print('PR_URL:', pr.url)
        print('PR_NUM:', pr.num)
    """
    )
    cmd = f"conda run -n {CONDA_ENV} python -c {shlex.quote(script_create)}"
    result = _run(cmd, timeout=60)
    out = result.get("stdout", "")
    pr_num = None
    for line in out.splitlines():
        if line.startswith("PR_NUM:"):
            pr_num = line.split(":", 1)[1].strip()
    if not pr_num:
        return _fmt(result) + "\nERROR: could not parse PR number"
    push_result = _run(
        f"git push origin HEAD:refs/pr/{pr_num} --force",
        timeout=300,
        cwd=str(_repo_path(worktree)),
    )
    return _fmt(result) + "\n" + _fmt(push_result)
def tool_create_github_issue(
    repo: str, title: str, body: str, labels: list[str] | None = None
) -> str:
    label_flags = " ".join(f"--label {shlex.quote(label)}" for label in (labels or []))
    cmd = (
        f"gh issue create --repo {shlex.quote(repo)} "
        f"--title {shlex.quote(title)} "
        f"--body {shlex.quote(body)} "
        f"{label_flags}"
    )
    return _fmt(_run(cmd, timeout=60))
# ──────────────────────────────────────────────────────────────────────────────
# Tool schema
# ──────────────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "run_shell",
        "description": "Run a shell command in the repo root for diagnostics or repo maintenance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer", "default": 120},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read any file from the repo or sibling worktrees.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a complete file (creates parent directories if needed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory, optionally filtered by a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string"},
                "pattern": {"type": "string", "default": "*"},
            },
            "required": ["directory"],
        },
    },
    {
        "name": "git_op",
        "description": "Run a git subcommand in the repo root or a specified worktree.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "worktree": {"type": "string"},
            },
            "required": ["operation"],
        },
    },
    {
        "name": "find_model_path",
        "description": "Search the local HuggingFace cache for a model snapshot.",
        "input_schema": {
            "type": "object",
            "properties": {"model_name": {"type": "string"}},
            "required": ["model_name"],
        },
    },
    {
        "name": "fetch_model_config",
        "description": "Load config.json from a local model path, local HF cache, or HuggingFace Hub.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "hf_repo_id": {"type": "string"},
                "model_path": {"type": "string"},
            },
        },
    },
    {
        "name": "discover_parallel_configs",
        "description": "Read sgl-cookbook YAML files and extract TP/EP values for a model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "cookbook_root": {"type": "string", "default": COOKBOOK_REPO_DEFAULT},
                "hf_repo_id": {"type": "string"},
            },
            "required": ["model_name"],
        },
    },
    {
        "name": "analyze_model_inventory",
        "description": (
            "Analyze a model config using the onboard-model skill rules and classify "
            "expected definitions against local flashinfer_trace/definitions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "hf_repo_id": {"type": "string"},
                "model_path": {"type": "string"},
                "tp_values": {"type": "array", "items": {"type": "integer"}},
                "ep_values": {"type": "array", "items": {"type": "integer"}},
                "definitions_root": {"type": "string", "default": DEFINITIONS_ROOT_DEFAULT},
                "cookbook_root": {"type": "string", "default": COOKBOOK_REPO_DEFAULT},
            },
            "required": ["model_name"],
        },
    },
    {
        "name": "discover_recent_models",
        "description": (
            "Discover recent model additions from local SGLang and sgl-cookbook clones, "
            "filtered against docs/model_coverage.mdx."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "default": 30},
                "sglang_repo": {"type": "string", "default": SGLANG_REPO_DEFAULT},
                "cookbook_root": {"type": "string", "default": COOKBOOK_REPO_DEFAULT},
                "coverage_path": {"type": "string", "default": COVERAGE_DOC_DEFAULT},
            },
        },
    },
    {
        "name": "check_flashinfer_support",
        "description": "Check whether an op_type appears to be implemented in the FlashInfer repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "op_type": {"type": "string"},
                "flashinfer_repo": {"type": "string", "default": FLASHINFER_REPO_DEFAULT},
            },
            "required": ["op_type"],
        },
    },
    {
        "name": "check_adapter_support",
        "description": (
            "Check whether FlashInfer apply/runtime adapters for an op_type are present, "
            "registered, and covered by focused integration tests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "op_type": {"type": "string"},
                "integration_root": {"type": "string", "default": INTEGRATION_ROOT_DEFAULT},
                "test_root": {"type": "string", "default": INTEGRATION_TEST_ROOT_DEFAULT},
            },
            "required": ["op_type"],
        },
    },
    {
        "name": "check_evaluator_support",
        "description": (
            "Check whether the required evaluator for a definition/op_type is present, "
            "registered, and covered by focused bench tests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "definition_name": {"type": "string"},
                "op_type": {"type": "string"},
                "evaluator_root": {"type": "string", "default": EVALUATOR_ROOT_DEFAULT},
                "bench_test_root": {"type": "string", "default": BENCH_TEST_ROOT_DEFAULT},
            },
            "required": ["definition_name", "op_type"],
        },
    },
    {
        "name": "check_bench_extension_support",
        "description": (
            "Combined adapter + evaluator support check for a definition/op_type pair."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "definition_name": {"type": "string"},
                "op_type": {"type": "string"},
                "integration_root": {"type": "string", "default": INTEGRATION_ROOT_DEFAULT},
                "evaluator_root": {"type": "string", "default": EVALUATOR_ROOT_DEFAULT},
                "integration_test_root": {
                    "type": "string",
                    "default": INTEGRATION_TEST_ROOT_DEFAULT,
                },
                "bench_test_root": {"type": "string", "default": BENCH_TEST_ROOT_DEFAULT},
            },
            "required": ["definition_name", "op_type"],
        },
    },
    {
        "name": "check_gpus",
        "description": "Inspect GPU memory usage with nvidia-smi.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "kill_sglang",
        "description": "Kill stale sglang.launch_server processes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_tests",
        "description": "Run pytest for a single repo test file, including reference, integration, or bench tests.",
        "input_schema": {
            "type": "object",
            "properties": {"test_file": {"type": "string"}},
            "required": ["test_file"],
        },
    },
    {
        "name": "run_collection",
        "description": "Run scripts/collect_workloads.py sglang for one or more definitions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "definitions": {"type": "array", "items": {"type": "string"}},
                "flashinfer_trace_dir": {"type": "string"},
                "tp": {"type": "integer", "default": 1},
                "quantization": {"type": "string"},
                "ep": {"type": "integer", "default": 1},
                "extra_args": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["model_path", "definitions", "flashinfer_trace_dir"],
        },
    },
    {
        "name": "run_baseline_eval",
        "description": "Run flashinfer-bench baseline evaluation for a definition in the trace repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "definition": {"type": "string"},
                "flashinfer_trace_dir": {"type": "string"},
            },
            "required": ["definition", "flashinfer_trace_dir"],
        },
    },
    {
        "name": "check_workloads",
        "description": "Inspect workload files, blobs, baseline solution, and trace statuses for a definition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "flashinfer_trace_dir": {"type": "string"},
                "definition": {"type": "string"},
            },
            "required": ["flashinfer_trace_dir", "definition"],
        },
    },
    {
        "name": "create_github_pr",
        "description": "Open a GitHub pull request using gh pr create.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "branch": {"type": "string"},
                "base": {"type": "string", "default": "main"},
            },
            "required": ["title", "body", "branch"],
        },
    },
    {
        "name": "create_hf_pr",
        "description": "Open a HuggingFace dataset PR for flashinfer-trace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "branch": {"type": "string"},
                "worktree": {"type": "string"},
            },
            "required": ["repo_id", "title", "description", "branch", "worktree"],
        },
    },
    {
        "name": "create_github_issue",
        "description": "Open a GitHub issue, typically for a missing FlashInfer kernel.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["repo", "title", "body"],
        },
    },
]
def dispatch_tool(name: str, inputs: dict[str, Any]) -> str:
    dispatch = {
        "run_shell": tool_run_shell,
        "read_file": tool_read_file,
        "write_file": tool_write_file,
        "list_files": tool_list_files,
        "git_op": tool_git_op,
        "find_model_path": tool_find_model_path,
        "fetch_model_config": tool_fetch_model_config,
        "discover_parallel_configs": tool_discover_parallel_configs,
        "analyze_model_inventory": tool_analyze_model_inventory,
        "discover_recent_models": tool_discover_recent_models,
        "check_flashinfer_support": tool_check_flashinfer_support,
        "check_adapter_support": tool_check_adapter_support,
        "check_evaluator_support": tool_check_evaluator_support,
        "check_bench_extension_support": tool_check_bench_extension_support,
        "check_gpus": tool_check_gpus,
        "kill_sglang": tool_kill_sglang,
        "run_tests": tool_run_tests,
        "run_collection": tool_run_collection,
        "run_baseline_eval": tool_run_baseline_eval,
        "check_workloads": tool_check_workloads,
        "create_github_pr": tool_create_github_pr,
        "create_hf_pr": tool_create_hf_pr,
        "create_github_issue": tool_create_github_issue,
    }
    fn = dispatch.get(name)
    return fn(**inputs) if fn else f"Unknown tool: {name}"
# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a headless model onboarding agent for FlashInfer-Bench running in CI.
    Execute the local `.claude/skills/onboard-model/SKILL.md` workflow end to end.
    Key repo anchors:
    - Repo root: {repo_root}
    - Local definitions/docs live under: {definitions_root}
    - External trace repo/worktree lives under: {trace_repo}
    - FlashInfer repo: {flashinfer_repo}
    - SGLang repo: {sglang_repo}
    - sgl-cookbook repo: {cookbook_repo}
    - Adapter root: {integration_root}
    - Evaluator root: {evaluator_root}
    Core rules:
    - Work phase by phase. Do not skip blocker detection.
    - Use `analyze_model_inventory` early; it already encodes the skill's kernel formulas.
    - Use `check_flashinfer_support` for every missing op_type before writing repo changes.
    - Use `check_adapter_support` and `check_evaluator_support` before concluding a kernel family
      is repo-supported. Missing definitions are not the only blocker, and these checks should
      include focused integration / bench tests when present.
    - If FlashInfer core support is missing for a required kernel:
      - stop repo-side onboarding for that kernel
      - draft or file a GitHub issue in flashinfer-ai/flashinfer (unless dry-run)
      - mark the run blocked, not complete support
    - If bench-side adapter or evaluator support is missing and the kernel family needs it:
      - add/update the adapter/evaluator implementation
      - register it in the relevant `__init__.py` / registry file
      - add focused integration / unit tests
    - Reuse existing definitions/tests/docs as templates by reading nearby files first.
    - If requested phases end before PR submission, do only that bounded work and stop.
    - Never claim "fully supported" unless the requested phases were completed and eval/workloads passed.
    Phase 0 — Update repos:
    - Refresh local repos with git pull / clone logic if the relevant repos exist.
    - Report commit SHAs for reproducibility when you do update them.
    Phase 1 — Discover + inventory:
    - If `discover` is enabled, use `discover_recent_models` first and pick the most actionable candidate.
    - Load config.json with `fetch_model_config`.
    - Resolve TP/EP with `discover_parallel_configs` unless explicitly supplied.
    - Build the expected kernel inventory with `analyze_model_inventory`.
    - For each new or relevant kernel family, check bench-side support with
      `check_adapter_support` and `check_evaluator_support`.
    - Classify kernels into:
      1. existing locally
      2. missing but FlashInfer-supported
      3. missing and FlashInfer-missing (blocker / issue)
      4. bench-extension missing (adapter/evaluator or their tests required)
    Phase 2 — Definition generation:
    - For each missing but FlashInfer-supported definition:
      - read a sibling definition/test of the same op_type
      - create definition JSON under `flashinfer_trace/definitions/...`
      - create/update reference tests under `flashinfer_trace/tests/references/...`
      - update `docs/model_coverage.mdx` from ❌ to 🟡 where appropriate
      - verify with `run_tests`
      - If adapter support is missing for an op_type that needs apply/runtime integration:
      - add a new adapter file under `flashinfer_bench/integration/flashinfer/adapters/`
      - update `flashinfer_bench/integration/flashinfer/__init__.py`
      - add a focused test under `tests/integration/flashinfer/`
    - If evaluator support is missing for a definition that needs special validation:
      - add a new evaluator file under `flashinfer_bench/bench/evaluators/`
      - update `flashinfer_bench/bench/evaluators/registry.py`
      - update `flashinfer_bench/bench/evaluators/__init__.py`
      - add/update tests under `tests/bench/`
    - Keep one PR pair per definition. Do not batch unrelated definitions together.
    Phase 3 — Workload collection:
    - Only do this when workload collection is requested and a model path is available.
    - Use `check_gpus`; if busy, `kill_sglang` and re-check.
    - Use `run_collection` for supported definitions.
    - Verify artifacts with `check_workloads`.
    - If no baseline solution exists in the trace repo, write it by following an existing pattern,
      then run `run_baseline_eval`.
    Phase 4 — PR submission:
    - PR1: GitHub flashinfer-bench for definition JSON + reference tests + docs.
    - PR2: HuggingFace flashinfer-trace for workloads + blobs + traces (+ copied definition/test when needed).
    - If a definition already exists and only workloads were missing, skip PR1 and submit PR2 only.
    - Use `create_github_pr` / `create_hf_pr` when submission is enabled.
    Dry run / partial phase rules:
    - In dry-run mode, do analysis and draft exact next actions but do not write files or open issues/PRs.
    - If phases exclude later work, stop exactly at the requested phase boundary.
    Completion strings:
    - Requested phases finished successfully:
      ONBOARD_COMPLETE: model=<slug>
    - Blocked by missing upstream/kernel prerequisite:
      ONBOARD_BLOCKED: model=<slug> reason=<short_reason>
    - Unrecoverable failure:
      ONBOARD_FAILED: <reason>
"""
).format(
    repo_root=REPO_ROOT,
    definitions_root=REPO_ROOT / DEFINITIONS_ROOT_DEFAULT,
    trace_repo=REPO_ROOT / TRACE_REPO_DEFAULT,
    flashinfer_repo=REPO_ROOT / FLASHINFER_REPO_DEFAULT,
    sglang_repo=REPO_ROOT / SGLANG_REPO_DEFAULT,
    cookbook_repo=REPO_ROOT / COOKBOOK_REPO_DEFAULT,
    integration_root=REPO_ROOT / INTEGRATION_ROOT_DEFAULT,
    evaluator_root=REPO_ROOT / EVALUATOR_ROOT_DEFAULT,
)
# ──────────────────────────────────────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────────────────────────────────────
def build_task_message(
    *,
    discover: bool,
    prompt: str | None,
    model_name: str | None,
    hf_repo_id: str | None,
    model_path: str | None,
    phases: list[int],
    dry_run: bool,
    skip_workload: bool,
    submit_prs: bool,
    definitions_root: str,
    flashinfer_trace_dir: str,
    flashinfer_repo: str,
    sglang_repo: str,
    cookbook_repo: str,
    discover_days: int,
) -> str:
    known: list[str] = [
        f"Discover mode: {discover}",
        f"Phases: {phases}",
        f"Dry run: {dry_run}",
        f"Skip workload collection: {skip_workload}",
        f"Submit PRs: {submit_prs}",
        f"Definitions root: {definitions_root}",
        f"Trace repo/worktree: {flashinfer_trace_dir}",
        f"FlashInfer repo: {flashinfer_repo}",
        f"SGLang repo: {sglang_repo}",
        f"sgl-cookbook repo: {cookbook_repo}",
        f"Discover window (days): {discover_days}",
        f"Conda env: {CONDA_ENV}",
    ]
    if model_name:
        known.append(f"Model name: {model_name}")
    if hf_repo_id:
        known.append(f"HuggingFace repo ID: {hf_repo_id}")
    if model_path:
        known.append(f"Model path: {model_path}")
    must_resolve: list[str] = []
    if discover and not model_name:
        must_resolve.append("target model candidate(s)")
    if not model_name and not discover:
        must_resolve.append("model name from the prompt or config")
    if not hf_repo_id:
        must_resolve.append("HuggingFace repo ID if needed")
    if not model_path and 3 in phases and not skip_workload:
        must_resolve.append("local model path for workload collection")
    parts: list[str] = []
    if prompt:
        parts.append(f"User request: {prompt}")
    parts.append("Known parameters:\n" + "\n".join(f"  {item}" for item in known))
    if must_resolve:
        parts.append("Must resolve:\n" + "\n".join(f"  - {item}" for item in must_resolve))
    parts.append(
        "Start with Phase 0 only if it is included. Then perform Phase 1 inventory before any "
        "repo mutations, and stop immediately if an upstream FlashInfer blocker is confirmed."
    )
    return "\n\n".join(parts)
def run_agent(
    *,
    discover: bool,
    prompt: str | None,
    model_name: str | None,
    hf_repo_id: str | None,
    model_path: str | None,
    phases: list[int],
    dry_run: bool,
    skip_workload: bool,
    submit_prs: bool,
    definitions_root: str,
    flashinfer_trace_dir: str,
    flashinfer_repo: str,
    sglang_repo: str,
    cookbook_repo: str,
    discover_days: int,
    max_attempts: int,
    claude_model: str,
    api_base_url: str | None = None,
) -> bool:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    use_openai_compat = bool(api_base_url)
    if use_openai_compat:
        from openai import OpenAI
        client = OpenAI(base_url=api_base_url, api_key=api_key)
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    task = build_task_message(
        discover=discover,
        prompt=prompt,
        model_name=model_name,
        hf_repo_id=hf_repo_id,
        model_path=model_path,
        phases=phases,
        dry_run=dry_run,
        skip_workload=skip_workload,
        submit_prs=submit_prs,
        definitions_root=definitions_root,
        flashinfer_trace_dir=flashinfer_trace_dir,
        flashinfer_repo=flashinfer_repo,
        sglang_repo=sglang_repo,
        cookbook_repo=cookbook_repo,
        discover_days=discover_days,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    collection_attempts: dict[str, int] = {}
    print(f"[agent] Starting model onboarding agent (model={claude_model})", flush=True)
    print(f"[agent] Discover:   {discover}", flush=True)
    print(f"[agent] Model:      {model_name or '(from prompt/discovery)'}", flush=True)
    print(f"[agent] Phases:     {phases}", flush=True)
    print(f"[agent] Dry run:    {dry_run}", flush=True)
    print(f"[agent] Trace repo: {flashinfer_trace_dir}\n", flush=True)
    if use_openai_compat:
        oa_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in TOOLS
        ]
    while True:
        if use_openai_compat:
            oa_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            response = client.chat.completions.create(
                model=claude_model,
                max_tokens=8192,
                tools=oa_tools,
                messages=oa_messages,
            )
            choice = response.choices[0]
            msg = choice.message
            text_content = msg.content or ""
            tool_calls = msg.tool_calls or []
            if text_content:
                print(f"[agent] {text_content}", flush=True)
                if "ONBOARD_COMPLETE:" in text_content:
                    print("\n[agent] ✅ Onboarding complete.", flush=True)
                    return True
                if "ONBOARD_BLOCKED:" in text_content:
                    print("\n[agent] ⏸️ Onboarding blocked.", flush=True)
                    return False
                if "ONBOARD_FAILED:" in text_content:
                    print("\n[agent] ❌ Onboarding failed.", flush=True)
                    return False
            if choice.finish_reason == "stop":
                print("[agent] Model stopped without completion signal.", flush=True)
                return False
            if choice.finish_reason != "tool_calls":
                print(f"[agent] Unexpected finish_reason: {choice.finish_reason}", flush=True)
                return False
            messages.append(
                {
                    "role": "assistant",
                    "content": text_content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                fn_name = call.function.name
                fn_args = json.loads(call.function.arguments)
                if fn_name == "run_collection":
                    key = ",".join(fn_args.get("definitions", [])) or "(unknown)"
                    attempt = collection_attempts.get(key, 0) + 1
                    collection_attempts[key] = attempt
                    print(f"\n[agent] Collection attempt {attempt}/{max_attempts} for {key}", flush=True)
                    if attempt > max_attempts:
                        result = f"ERROR: max collection attempts exceeded for {key}"
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                        print(f"[agent] ← {result}\n", flush=True)
                        continue
                print(f"[agent] → {fn_name}({json.dumps(fn_args)[:160]})", flush=True)
                result = dispatch_tool(fn_name, fn_args)
                print(f"[agent] ← {result[:500]}\n", flush=True)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            continue
        response = client.messages.create(
            model=claude_model,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"[agent] {block.text}", flush=True)
                if "ONBOARD_COMPLETE:" in block.text:
                    print("\n[agent] ✅ Onboarding complete.", flush=True)
                    return True
                if "ONBOARD_BLOCKED:" in block.text:
                    print("\n[agent] ⏸️ Onboarding blocked.", flush=True)
                    return False
                if "ONBOARD_FAILED:" in block.text:
                    print("\n[agent] ❌ Onboarding failed.", flush=True)
                    return False
        if response.stop_reason == "end_turn":
            print("[agent] Model stopped without completion signal.", flush=True)
            return False
        if response.stop_reason != "tool_use":
            print(f"[agent] Unexpected stop_reason: {response.stop_reason}", flush=True)
            return False
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "run_collection":
                key = ",".join(block.input.get("definitions", [])) or "(unknown)"
                attempt = collection_attempts.get(key, 0) + 1
                collection_attempts[key] = attempt
                print(f"\n[agent] Collection attempt {attempt}/{max_attempts} for {key}", flush=True)
                if attempt > max_attempts:
                    result = f"ERROR: max collection attempts exceeded for {key}"
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": result}
                    )
                    print(f"[agent] ← {result}\n", flush=True)
                    continue
            print(f"[agent] → {block.name}({json.dumps(block.input)[:160]})", flush=True)
            result = dispatch_tool(block.name, block.input)
            print(f"[agent] ← {result[:500]}\n", flush=True)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": tool_results})
# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless model onboarding agent for FlashInfer-Bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              # Discover new models and onboard the best candidate
              python scripts/onboard_model_agent.py --discover
              # Onboard a specific model end to end
              python scripts/onboard_model_agent.py \\
                --model-name gemma-3-27b \\
                --hf-repo-id google/gemma-3-27b-it
              # Run only discovery + inventory + definition generation planning
              python scripts/onboard_model_agent.py \\
                --model-name kimi-k2 \\
                --phases 0,1,2 \\
                --dry-run
        """
        ),
    )
    parser.add_argument("--prompt", default=None, help="Extra natural-language instruction context.")
    parser.add_argument("--discover", action="store_true", help="Discover candidate models automatically.")
    parser.add_argument("--model-name", default=None, help="Model slug to onboard.")
    parser.add_argument("--hf-repo-id", default=None, help="HuggingFace repo ID override.")
    parser.add_argument("--model-path", default=None, help="Local HF snapshot path.")
    parser.add_argument(
        "--phases",
        default="0,1,2,3,4",
        help="Comma-separated phases to run (default: 0,1,2,3,4).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze and plan only; no writes or PRs.")
    parser.add_argument(
        "--skip-workload",
        action="store_true",
        help="Skip workload collection even if Phase 3 is selected.",
    )
    parser.add_argument(
        "--submit-prs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to open GitHub / HuggingFace PRs when Phase 4 runs (default: true).",
    )
    parser.add_argument(
        "--definitions-root",
        default=DEFINITIONS_ROOT_DEFAULT,
        help=f"Local definitions/docs root (default: {DEFINITIONS_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--flashinfer-trace-dir",
        default=TRACE_REPO_DEFAULT,
        help=f"Trace repo/worktree path for workloads + PR2 (default: {TRACE_REPO_DEFAULT})",
    )
    parser.add_argument(
        "--flashinfer-repo",
        default=FLASHINFER_REPO_DEFAULT,
        help=f"FlashInfer repo path (default: {FLASHINFER_REPO_DEFAULT})",
    )
    parser.add_argument(
        "--sglang-repo",
        default=SGLANG_REPO_DEFAULT,
        help=f"SGLang repo path (default: {SGLANG_REPO_DEFAULT})",
    )
    parser.add_argument(
        "--sgl-cookbook-repo",
        default=COOKBOOK_REPO_DEFAULT,
        help=f"sgl-cookbook repo path (default: {COOKBOOK_REPO_DEFAULT})",
    )
    parser.add_argument(
        "--discover-days",
        type=int,
        default=30,
        help="Lookback window for --discover mode (default: 30).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max collection attempts per definition set (default: 3).",
    )
    parser.add_argument(
        "--claude-model",
        default="claude-sonnet-4-6",
        help=(
            "Model ID. For native Anthropic use claude-sonnet-4-6 (default). "
            "For OpenAI-compatible endpoints use the provider-specific model ID."
        ),
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="OpenAI-compatible base URL for non-Anthropic endpoints.",
    )
    args = parser.parse_args()
    try:
        phases = parse_phases(args.phases)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.discover and not args.model_name and not args.hf_repo_id and not args.prompt:
        parser.error("Provide at least one of --discover, --model-name, --hf-repo-id, or --prompt.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    claude_model = args.claude_model
    if args.api_base_url and claude_model == "claude-sonnet-4-6":
        claude_model = "aws/anthropic/bedrock-claude-sonnet-4-6"
    success = run_agent(
        discover=args.discover,
        prompt=args.prompt,
        model_name=args.model_name,
        hf_repo_id=args.hf_repo_id,
        model_path=args.model_path,
        phases=phases,
        dry_run=args.dry_run,
        skip_workload=args.skip_workload,
        submit_prs=args.submit_prs,
        definitions_root=args.definitions_root,
        flashinfer_trace_dir=args.flashinfer_trace_dir,
        flashinfer_repo=args.flashinfer_repo,
        sglang_repo=args.sglang_repo,
        cookbook_repo=args.sgl_cookbook_repo,
        discover_days=args.discover_days,
        max_attempts=args.max_attempts,
        claude_model=claude_model,
        api_base_url=args.api_base_url,
    )
    sys.exit(0 if success else 1)
if __name__ == "__main__":
    main()