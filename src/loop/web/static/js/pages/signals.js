import { fetchSignals } from "../api.js";
import { renderForestPlot, renderDecayCurves } from "../charts.js";

export async function renderSignalsPage(container, customerId) {
  container.innerHTML = `<div class="status-loading mono">Loading signal attribution for ${customerId}...</div>`;

  try {
    const data = await fetchSignals(customerId);

    const html = `
      <div class="page-header">
        <div>
          <h1 class="page-title">Signal Attribution &amp; Decay</h1>
          <div class="page-subtitle">
            <span class="chip chip-neutral">${data.industry.replace(/_/g, " ")}</span>
            <span class="mono" style="margin-left: 12px; color: var(--color-silver);">
              Hierarchical Bayesian posterior with exponential half-life modeling
            </span>
          </div>
        </div>
      </div>

      <div class="paper-panel">
        <h2 class="section-h2">${data.finding_title}</h2>
        <div id="forest-plot-container" style="height: 380px; width: 100%;"></div>
      </div>

      <div class="paper-panel">
        <h2 class="section-h2">Signal Half-Life &amp; Decay Trajectories</h2>
        <div id="decay-curves-container" style="height: 380px; width: 100%;"></div>
      </div>

      <div class="paper-panel">
        <h2 class="section-h2">Own Data versus Borrowed Strength</h2>
        <table class="data-table">
          <thead>
            <tr>
              <th>Signal Type</th>
              <th>Customer-Only (Unpooled)</th>
              <th>Yeild Estimate (Hierarchical)</th>
              <th>Industry Mean</th>
              <th>Shrinkage %</th>
              <th>Half-Life t1/2</th>
            </tr>
          </thead>
          <tbody>
            ${data.signals
              .map(
                (s) => `
                <tr>
                  <td class="mono" style="font-weight: 600;">${s.signal.replace(/_/g, " ")}</td>
                  <td class="mono">${s.customer_only_lift.toFixed(2)} pp</td>
                  <td class="mono" style="font-weight: 700; color: var(--color-teal);">
                    ${s.median_lift.toFixed(2)} pp <span style="font-size: 11px; color: var(--color-silver);">[${s.lo90.toFixed(2)}, ${s.hi90.toFixed(2)}]</span>
                  </td>
                  <td class="mono">${s.industry_pooled_lift.toFixed(2)} pp</td>
                  <td class="mono">${s.shrinkage_pct.toFixed(1)}%</td>
                  <td class="mono">${s.half_life_median.toFixed(0)}d [${s.half_life_lo90.toFixed(0)}, ${s.half_life_hi90.toFixed(0)}]</td>
                </tr>
              `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;

    container.innerHTML = html;

    renderForestPlot("forest-plot-container", data.signals, data.finding_title);
    renderDecayCurves(
      "decay-curves-container",
      data.signals,
      "Decay curves for top 4 buying signals over 90-day lead horizon"
    );
  } catch (err) {
    container.innerHTML = `<div class="status-error">Error loading signals: ${err.message}</div>`;
  }
}
