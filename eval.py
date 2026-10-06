import argparse
import hashlib
import json
import sys
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from loop.config import LOOP_MODEL, LOOP_SEED, REPORTS_DIR
from loop.evaluation.allocator_eval import evaluate_allocator_performance
from loop.evaluation.attribution_eval import evaluate_attribution_recovery
from loop.evaluation.coldstart_eval import evaluate_cold_start
from loop.evaluation.e2e_eval import evaluate_e2e_propagation
from loop.evaluation.gates import QualityGate, evaluate_gate
from loop.evaluation.ope_eval import evaluate_ope_simulation
from loop.evaluation.reply_eval import evaluate_reply_intelligence


def compute_sim_hash() -> str:
    from loop.simulate.world import CTA_EFFECTS, OPENER_EFFECTS, SIGNAL_CONFIGS
    payload = {
        "signals": {k: list(v) for k, v in SIGNAL_CONFIGS.items()},
        "openers": OPENER_EFFECTS,
        "ctas": CTA_EFFECTS,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description="Loop Evaluation Runner")
    parser.add_argument("--fast", action="store_true", help="Run reduced iteration fast evaluation")
    parser.add_argument("--seed", type=int, default=LOOP_SEED, help="Random seed")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated component filters")
    args = parser.parse_args()

    start_time = time.perf_counter()
    selected_components = set(args.only.split(",")) if args.only else {
        "reply", "attribution", "coldstart", "allocator", "ope", "e2e"
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sim_hash = compute_sim_hash()

    reply_res: dict[str, Any] = {}
    attr_res: dict[str, Any] = {}
    cold_res: dict[str, Any] = {}
    alloc_res: dict[str, Any] = {}
    ope_res: dict[str, Any] = {}
    e2e_res: dict[str, Any] = {}

    if "reply" in selected_components:
        reply_res = evaluate_reply_intelligence(seed=args.seed, fast_mode=args.fast)

    if "attribution" in selected_components:
        n_w = 1 if args.fast else 2
        attr_res = evaluate_attribution_recovery(n_worlds=n_w, seed=args.seed, fast_mode=args.fast)

    if "coldstart" in selected_components:
        cold_res = evaluate_cold_start(seed=args.seed, fast_mode=args.fast)

    if "allocator" in selected_components:
        alloc_res = evaluate_allocator_performance(seed=args.seed, fast_mode=args.fast)

    if "ope" in selected_components:
        ope_res = evaluate_ope_simulation(seed=args.seed, fast_mode=args.fast)

    if "e2e" in selected_components:
        e2e_res = evaluate_e2e_propagation(seed=args.seed, fast_mode=args.fast)

    b_data = reply_res.get("baseline", {})
    llm_data = reply_res.get("llm", {})
    plots = reply_res.get("plots", {})
    if attr_res.get("recovery_pairs"):
        plots["recovery_pairs"] = attr_res["recovery_pairs"]
    if attr_res.get("confounding_comparison"):
        plots["confounding_comparison"] = attr_res["confounding_comparison"]
    if cold_res.get("curves"):
        plots["cold_start_curves"] = cold_res["curves"]
    if alloc_res.get("regret_series"):
        plots["regret_series"] = alloc_res["regret_series"]

    gates: list[QualityGate] = []

    llm_avail = llm_data.get("available", False)
    skip_llm_reason = None if llm_avail else "No API key configured; baseline classifier used"

    gates.append(
        evaluate_gate(
            "G1",
            "Reply macro-F1 on test >= 0.90 (LLM)",
            llm_data.get("test_macro_f1"),
            0.90,
            ">=",
            skip_reason=skip_llm_reason,
        )
    )
    gates.append(
        evaluate_gate(
            "G2",
            "Reply macro-F1 on hard set >= 0.80 (LLM)",
            llm_data.get("hard_macro_f1"),
            0.80,
            ">=",
            skip_reason=skip_llm_reason,
        )
    )
    gates.append(
        evaluate_gate(
            "G3",
            "ECE after calibration <= 0.05",
            b_data.get("ece_after"),
            0.05,
            "<=",
        )
    )
    gates.append(
        evaluate_gate(
            "G4",
            "Coverage at 5% risk >= 0.80",
            b_data.get("actual_coverage"),
            0.80,
            ">=",
        )
    )
    gates.append(
        evaluate_gate(
            "G5",
            "Injection attack success <= 0.05",
            b_data.get("attack_success_rate"),
            0.05,
            "<=",
        )
    )
    gates.append(
        evaluate_gate(
            "G6",
            "Draft validator pass rate >= 0.95",
            b_data.get("draft_pass_rate"),
            0.95,
            ">=",
        )
    )
    gates.append(
        evaluate_gate(
            "G7",
            "Attribution Spearman >= 0.80",
            attr_res.get("spearman"),
            0.80,
            ">=",
        )
    )
    gates.append(
        evaluate_gate(
            "G8",
            "90% interval coverage in [0.85, 0.95]",
            attr_res.get("coverage_90"),
            (0.85, 0.95),
            "in",
        )
    )
    gates.append(
        evaluate_gate(
            "G9",
            "Max R-hat <= 1.05",
            attr_res.get("diagnostics", {}).get("max_rhat"),
            1.05,
            "<=",
        )
    )
    gates.append(
        evaluate_gate(
            "G10",
            "Divergences == 0",
            attr_res.get("diagnostics", {}).get("divergences"),
            0,
            "==",
        )
    )
    gates.append(
        evaluate_gate(
            "G11",
            "Cold-start log-loss gain at k=0 >= 5%",
            cold_res.get("k0_relative_gain_pct"),
            5.0,
            ">=",
        )
    )
    gates.append(
        evaluate_gate(
            "G12",
            "Allocator final-4-round reward ratio >= 0.85",
            alloc_res.get("final_4_round_reward_ratio"),
            0.85,
            ">=",
        )
    )
    gates.append(
        evaluate_gate(
            "G13",
            "Allocator vs uniform paired CI lower bound > 0",
            alloc_res.get("paired_diff_vs_uniform", {}).get("lo95"),
            0.0,
            ">=",
        )
    )
    gates.append(
        evaluate_gate(
            "G14",
            "DR absolute bias <= 5% of true value",
            ope_res.get("dr_abs_bias_pct"),
            5.0,
            "<=",
        )
    )

    adj_err = attr_res.get("adjusted_abs_error", 1.0)
    naive_err = attr_res.get("naive_abs_error", 2.0)
    gates.append(
        evaluate_gate(
            "G15",
            "Adjusted absolute error < naive absolute error",
            adj_err,
            naive_err,
            "<",
        )
    )

    elapsed = time.perf_counter() - start_time

    results_payload = {
        "metadata": {
            "seed": args.seed,
            "fast_mode": args.fast,
            "config_hash": sim_hash,
            "model_id": LOOP_MODEL,
            "runtime_seconds": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "spearman": attr_res.get("spearman"),
        },
        "gates": [
            {
                "id": g.id,
                "description": g.description,
                "value": g.value,
                "threshold": g.threshold,
                "operator": g.operator,
                "status": g.status,
            }
            for g in gates
        ],
        "components": {
            "reply": reply_res,
            "attribution": attr_res,
            "coldstart": cold_res,
            "allocator": alloc_res,
            "ope": ope_res,
            "e2e": e2e_res,
        },
        "plots": plots,
    }

    json_path = REPORTS_DIR / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    md_lines = [
        "# Loop Offline & Counterfactual Evaluation Report",
        "",
        f"- Seed: `{args.seed}`",
        f"- Fast Mode: `{args.fast}`",
        f"- Config Hash: `{sim_hash}`",
        f"- Model: `{LOOP_MODEL}`",
        f"- Runtime: `{elapsed:.1f}s`",
        "",
        "## Quality Gates Scorecard",
        "",
        "| Gate ID | Description | Measured Value | Threshold | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for g in gates:
        val_str = f"{g.value:.4f}" if isinstance(g.value, float) else str(g.value)
        thresh_str = f"{g.operator} {g.threshold}"
        md_lines.append(f"| {g.id} | {g.description} | {val_str} | {thresh_str} | **{g.status}** |")

    md_lines.extend([
        "",
        "## Component Summaries",
        "",
        "### Reply Intelligence",
        f"- Test Macro-F1 (Baseline): `{b_data.get('test_macro_f1', 0.0):.4f}`",
        f"- Hard Set Macro-F1 (Baseline): `{b_data.get('hard_macro_f1', 0.0):.4f}`",
        f"- Handwritten Macro-F1: `{b_data.get('handwritten_macro_f1')}`",
        f"- ECE (Post-Calib): `{b_data.get('ece_after', 0.0):.4f}`",
        f"- Auto-Handled Coverage at 5% Risk: `{b_data.get('actual_coverage', 0.0)*100:.1f}%`",
        "",
        "### Signal Attribution Recovery",
        f"- Spearman Rank Correlation: `{attr_res.get('spearman', 0.0):.4f}`",
        f"- Pearson Correlation: `{attr_res.get('pearson', 0.0):.4f}`",
        f"- 90% Posterior Interval Coverage: `{attr_res.get('coverage_90', 0.0)*100:.1f}%`",
        f"- MCMC Max R-hat: `{attr_res.get('diagnostics', {}).get('max_rhat', 1.0):.3f}`",
        "",
        "### Cold Start & Allocator",
        f"- Cold-Start k=0 Gain: `{cold_res.get('k0_relative_gain_pct', 0.0):.2f}%`",
        f"- Allocator Final-4 Reward Ratio: `{alloc_res.get('final_4_round_reward_ratio', 0.0):.4f}`",
        f"- OPE DR Absolute Bias: `{ope_res.get('dr_abs_bias_pct', 0.0):.2f}%`",
        "",
    ])

    md_path = REPORTS_DIR / "eval_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    console = Console()
    table = Table(title="Loop Evaluation Gates Scorecard", header_style="bold magenta")
    table.add_column("Gate", justify="center", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Value", justify="right", style="green")
    table.add_column("Threshold", justify="center", style="yellow")
    table.add_column("Status", justify="center")

    has_failures = False
    for g in gates:
        val_str = f"{g.value:.4f}" if isinstance(g.value, float) else str(g.value)
        status_style = "bold green" if g.status == "PASS" else ("bold yellow" if g.status == "SKIPPED" else "bold red")
        table.add_row(g.id, g.description, val_str, f"{g.operator} {g.threshold}", f"[{status_style}]{g.status}[/{status_style}]")
        if g.status == "FAIL":
            has_failures = True

    console.print(table)
    console.print(f"\nArtifacts generated:\n- {json_path}\n- {md_path}\nTotal Runtime: {elapsed:.2f}s\n")

    if has_failures:
        sys.exit(1)

if __name__ == "__main__":
    main()
