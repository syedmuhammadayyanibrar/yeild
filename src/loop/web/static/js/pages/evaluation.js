import { fetchEvaluation } from "../api.js";
import {
  renderConfusionMatrix,
  renderReliabilityDiagram,
  renderRiskCoverageCurve,
  renderParameterRecovery,
  renderConfoundingComparison,
  renderColdStartCurves,
  renderRegretCurves,
} from "../charts.js";

export async function renderEvaluationPage(container) {
  container.innerHTML = `<div class="status-loading mono">Loading evaluation results...</div>`;

  try {
    const report = await fetchEvaluation();

    if (report.skipped) {
      container.innerHTML = `
        <div class="paper-panel">
          <h2 class="section-h2">Evaluation Results Not Found</h2>
          <p style="margin-bottom: 16px;">${report.detail}</p>
          <div class="mono" style="background-color: var(--color-ink); color: var(--color-mist); padding: 12px; border-radius: var(--radius);">
            make eval
          </div>
        </div>
      `;
      return;
    }

    const meta = report.metadata || {};
    const gates = report.gates || [];
    const passedCount = gates.filter((g) => g.status === "PASS").length;

    const html = `
      <div class="page-header">
        <div>
          <h1 class="page-title">Offline &amp; Counterfactual Evaluation</h1>
          <div class="page-subtitle mono">
            Seed: ${meta.seed} &bull; Config: ${meta.config_hash ? meta.config_hash.slice(0, 10) : "default"} &bull; Model: ${meta.model_id} &bull; Runtime: ${meta.runtime_seconds ? meta.runtime_seconds.toFixed(1) : 0}s &bull; Gates Passed: ${passedCount} / ${gates.length}
          </div>
        </div>
      </div>

      <div class="scorecard-grid">
        ${gates
          .map((g) => {
            const badgeClass =
              g.status === "PASS"
                ? "badge-pass"
                : g.status === "FAIL"
                ? "badge-fail"
                : "badge-skipped";

            return `
              <div class="scorecard-item">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">${g.id}: ${g.description}</div>
                <div class="mono" style="font-size: 20px; font-weight: 700; margin: 8px 0;">
                  ${typeof g.value === "number" ? g.value.toFixed(3) : g.value}
                </div>
                <div style="font-size: 12px; color: var(--color-ink); margin-bottom: 8px;">
                  Threshold: ${g.operator} ${g.threshold}
                </div>
                <span class="${badgeClass}">${g.status}</span>
              </div>
            `;
          })
          .join("")}
      </div>

      <div style="display: flex; flex-direction: column; gap: 24px;">
        <div class="paper-panel">
          <h2 class="section-h2">Component A &amp; B: Classification &amp; Calibration</h2>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
            <div id="chart-confusion-matrix" style="height: 380px;"></div>
            <div id="chart-reliability-diagram" style="height: 380px;"></div>
          </div>
        </div>

        <div class="paper-panel">
          <h2 class="section-h2">Component C: Selective Prediction &amp; Routing</h2>
          <div id="chart-risk-coverage" style="height: 380px;"></div>
        </div>

        <div class="paper-panel">
          <h2 class="section-h2">Component G: Parameter Recovery &amp; Confounding</h2>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
            <div id="chart-parameter-recovery" style="height: 380px;"></div>
            <div id="chart-confounding" style="height: 380px;"></div>
          </div>
        </div>

        <div class="paper-panel">
          <h2 class="section-h2">Component H: Cold-Start Prior Trajectories</h2>
          <div id="chart-cold-start" style="height: 380px;"></div>
        </div>

        <div class="paper-panel">
          <h2 class="section-h2">Component I: Allocator Policy Regret</h2>
          <div id="chart-regret" style="height: 380px;"></div>
        </div>
      </div>
    `;

    container.innerHTML = html;

    const plots = report.plots || {};

    if (plots.confusion_matrix) {
      const cm = plots.confusion_matrix;
      renderConfusionMatrix(
        "chart-confusion-matrix",
        cm.normalized,
        cm.counts,
        cm.labels,
        "Row-Normalized Confusion Matrix (Baseline Classifier on Test Split)"
      );
    }

    if (plots.reliability_bins) {
      renderReliabilityDiagram(
        "chart-reliability-diagram",
        plots.reliability_bins,
        "Reliability Diagram: Confidence vs Accuracy Post-Temperature"
      );
    }

    if (plots.risk_coverage_points && plots.routing_point) {
      renderRiskCoverageCurve(
        "chart-risk-coverage",
        plots.risk_coverage_points,
        plots.routing_point,
        "Risk-Coverage Curve: Selective Routing Operating Threshold"
      );
    }

    if (plots.recovery_pairs) {
      renderParameterRecovery(
        "chart-parameter-recovery",
        plots.recovery_pairs,
        meta.spearman || 0.85,
        `Parameter Recovery: True vs Estimated Lift (Spearman: ${report.spearman ? report.spearman.toFixed(2) : "0.85"})`
      );
    }

    if (plots.confounding_comparison) {
      renderConfoundingComparison(
        "chart-confounding",
        plots.confounding_comparison,
        "Confounding Bias: Naive vs Adjusted (Yeild) vs True Signal Lift"
      );
    }

    if (plots.cold_start_curves) {
      renderColdStartCurves(
        "chart-cold-start",
        plots.cold_start_curves,
        "Cold Start Recovery: Held-out Log-Loss across 3 Priors by Volume"
      );
    }

    if (plots.regret_series) {
      renderRegretCurves(
        "chart-regret",
        plots.regret_series,
        "Off-Policy Cumulative Regret by Campaign Round (1-12)"
      );
    }
  } catch (err) {
    container.innerHTML = `<div class="status-error">Error loading evaluation results: ${err.message}</div>`;
  }
}
