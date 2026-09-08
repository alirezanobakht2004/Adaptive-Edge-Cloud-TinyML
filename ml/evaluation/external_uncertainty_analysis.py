"""Render measured internal/external uncertainty comparisons from frozen-model evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.external_dataset.inspect_kaggle_dataset import ROOT


def render(report_path=None, output=None):
    report_path = Path(report_path or ROOT / "docs/evidence/external_model_evaluation_report.json")
    output = Path(output or ROOT / "docs/evidence/external_uncertainty_report.md")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["training_performed"] or not report["model_weights_unchanged"]:
        raise ValueError("Expected a frozen-model inference report")
    lines = ["# External uncertainty comparison", "",
             f"Model: `{report['model_version']}`; SHA-256 `{report['model_sha256']}`.", "",
             f"Reference: {report['internal_reference']['samples']} dataset-v1 validation windows from `{report['internal_reference']['session']}`.",
             f"External: {report['number_of_evaluated_samples']} diagnostically adapted windows; native compatibility is false.", "",
             "Deterministic mode disables dropout. MC5 averages five stochastic head predictions with dropout rate 0.2 and seed 42.",
             "Entropy is predictive entropy of the probability vector (MC mean in MC5), in nats. Margin is the top-two probability difference.",
             "Probability sums are normalized for floating-point roundoff only. No temperature scaling, training or calibration is performed.", "",
             "| Mode | Cohort | Mean confidence | Mean entropy (nats) | Mean margin |",
             "|---|---|---:|---:|---:|"]
    for mode, result in report["modes"].items():
        for cohort in ("internal", "external"):
            stats = result[f"{cohort}_uncertainty"]
            lines.append(f"| {mode} | {cohort} | {stats['confidence']['mean']:.6f} | {stats['entropy']['mean']:.6f} | {stats['margin']['mean']:.6f} |")
    lines += ["", "## Label-controlled idle comparison", "",
              "Only broad idle semantics are mapped. This subset is not an overall gesture-accuracy test.", "",
              "| Mode | Internal IDLE mean confidence / entropy | External idle mean confidence / entropy | Mapped idle accuracy | Confident mapped errors (confidence >= 0.9) |",
              "|---|---|---|---:|---:|"]
    for mode, result in report["modes"].items():
        ours = result["internal_by_class"]["IDLE"]
        external = result["external_by_original_label"]["idle"]["uncertainty"]
        accuracy = "unavailable" if result["mapped_subset_accuracy"] is None else f"{result['mapped_subset_accuracy']:.6f}"
        lines.append(f"| {mode} | {ours['confidence']['mean']:.6f} / {ours['entropy']['mean']:.6f} | {external['confidence']['mean']:.6f} / {external['entropy']['mean']:.6f} | {accuracy} | {result['mapped_errors_with_confidence_at_least_0_9']} / {result['accuracy_evaluated_samples']} |")
    lines += ["", "## MC5 distribution by original external label", "",
              "| Label | N | Mean confidence | Mean entropy (nats) | Mean margin |", "|---|---:|---:|---:|---:|"]
    for label, result in report["modes"]["mc5"]["external_by_original_label"].items():
        stats = result["uncertainty"]
        lines.append(f"| {label} | {result['samples']} | {stats['confidence']['mean']:.6f} | {stats['entropy']['mean']:.6f} | {stats['margin']['mean']:.6f} |")
    lines += ["", "## Histogram counts", "", "Bins are left-inclusive/right-exclusive except the final bin, which includes its right endpoint."]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, (mode, result) in enumerate(report["modes"].items()):
        for column, metric in enumerate(("confidence", "entropy", "margin")):
            internal = result["internal_uncertainty"][metric]
            external = result["external_uncertainty"][metric]
            edges = internal["histogram_edges"]
            if edges != external["histogram_edges"]:
                raise ValueError("Histogram bins differ across cohorts")
            lines += ["", f"### {mode}: {metric}", "", "| Bin | Internal count | External count |", "|---|---:|---:|"]
            for i in range(len(edges) - 1):
                lines.append(f"| {edges[i]:.6f} to {edges[i+1]:.6f} | {internal['histogram_counts'][i]} | {external['histogram_counts'][i]} |")
            for name, stats in (("Internal validation", internal), ("External diagnostic", external)):
                if sum(stats["histogram_counts"]) != stats["count"]:
                    raise ValueError("Histogram dropped observations")
                axes[row, column].stairs(np.asarray(stats["histogram_counts"]) / stats["count"], edges, label=name)
            axes[row, column].set(title=f"{mode}: {metric}", xlabel="nats" if metric == "entropy" else metric, ylabel="Fraction of windows")
            axes[row, column].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = output.with_name("external_uncertainty_histograms.png")
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    lines += ["", f"![Measured uncertainty histograms]({figure.name})", "", "## Interpretation and limits", "",
              *[f"{mode}: external minus internal mean entropy = {result['external_uncertainty']['entropy']['mean'] - result['internal_uncertainty']['entropy']['mean']:.6f} nats; mean confidence difference = {result['external_uncertainty']['confidence']['mean'] - result['internal_uncertainty']['confidence']['mean']:.6f}." for mode, result in report["modes"].items()],
              "The mapped-subset table reports high-confidence errors explicitly. Confidence alone is not a correctness guarantee under shift.",
              "This is descriptive evidence for a robustness stress test; it does not validate an OOD detector or an adaptive-policy threshold.",
              "Class balance, gesture semantics, sampling adaptation, unknown mounting and acquisition differences confound pooled comparisons.",
              "MC masks are host-side analysis masks, not firmware parity vectors. No ESP32 test, policy training or production change was performed.", "",
              f"Input report SHA-256 (LF UTF-8): `{hashlib.sha256(report_path.read_text(encoding='utf-8').encode()).hexdigest()}`.", ""]
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(render(args.input, args.output))


if __name__ == "__main__":
    main()
