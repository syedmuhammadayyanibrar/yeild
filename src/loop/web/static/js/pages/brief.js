import { fetchBrief } from "../api.js";

export async function renderBriefPage(container, customerId) {
  container.innerHTML = `<div class="status-loading mono">Loading brief for ${customerId}...</div>`;

  try {
    const brief = await fetchBrief(customerId);

    const win = brief.window;
    const kpi = brief.kpi_strip;

    const html = `
      <div class="page-header">
        <div>
          <h1 class="page-title">${brief.customer_name}</h1>
          <div class="page-subtitle">
            <span class="chip chip-neutral">${brief.industry.replace(/_/g, " ")}</span>
            <span class="mono" style="margin-left: 12px; color: var(--color-silver);">
              Window: ${win.weeks} weeks &bull; ${win.total_sends} sends &bull; ${win.positive_replies} positive replies &bull; ${win.meetings} meetings
            </span>
          </div>
        </div>
      </div>

      <div class="kpi-grid">
        <div class="kpi-tile">
          <div class="kpi-label">Current Mix (Expected Meetings)</div>
          <div class="kpi-value">${kpi.current_expected_meetings.toFixed(2)}</div>
          <div class="kpi-subtext">90% CI: [${kpi.current_lo90.toFixed(2)}, ${kpi.current_hi90.toFixed(2)}] / 100 sends</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Recommended Mix (Yeild)</div>
          <div class="kpi-value" style="color: var(--color-teal);">${kpi.recommended_expected_meetings.toFixed(2)}</div>
          <div class="kpi-subtext">90% CI: [${kpi.recommended_lo90.toFixed(2)}, ${kpi.recommended_hi90.toFixed(2)}] / 100 sends</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Relative Gain</div>
          <div class="kpi-value">${kpi.relative_gain_pct > 0 ? "+" : ""}${kpi.relative_gain_pct.toFixed(1)}%</div>
          <div class="kpi-subtext">Over baseline historical policy</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">P(Yeild &gt; Current)</div>
          <div class="kpi-value">${(kpi.p_recommended_beats_current * 100).toFixed(0)}%</div>
          <div class="kpi-subtext">Posterior certainty of outperformance</div>
        </div>
      </div>

      <div class="content-columns">
        <div class="paper-panel">
          <h2 class="section-h2">Campaign Allocation Policy</h2>
          <table class="data-table">
            <thead>
              <tr>
                <th>Arm</th>
                <th>Current vs Recommended Share</th>
                <th>Expected Meetings / 100</th>
                <th>Sends (n)</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              ${brief.allocation_table
                .map((arm) => {
                  const actionClass =
                    arm.action === "Scale"
                      ? "chip-scale"
                      : arm.action === "Stop"
                      ? "chip-stop"
                      : arm.action === "Test"
                      ? "chip-test"
                      : "chip-hold";

                  const curWidth = Math.min(100, Math.round(arm.current_share * 100 * 2.5));
                  const recWidth = Math.min(100, Math.round(arm.recommended_share * 100 * 2.5));

                  return `
                    <tr>
                      <td class="mono" style="font-weight: 600;">${arm.arm_id}</td>
                      <td>
                        <div class="paired-bar-container">
                          <div class="paired-bar-row">
                            <span class="mono" style="font-size: 11px; width: 36px;">${(arm.current_share * 100).toFixed(1)}%</span>
                            <div class="bar-track">
                              <div class="bar-fill-current" style="width: ${curWidth}%;"></div>
                            </div>
                          </div>
                          <div class="paired-bar-row">
                            <span class="mono" style="font-size: 11px; width: 36px; font-weight: 700;">${(arm.recommended_share * 100).toFixed(1)}%</span>
                            <div class="bar-track">
                              <div class="bar-fill-rec" style="width: ${recWidth}%;"></div>
                            </div>
                          </div>
                        </div>
                      </td>
                      <td class="mono">
                        ${arm.expected_meetings_median.toFixed(2)}
                        <span style="font-size: 11px; color: var(--color-silver);">[${arm.expected_meetings_lo90.toFixed(2)}, ${arm.expected_meetings_hi90.toFixed(2)}]</span>
                      </td>
                      <td class="mono">${arm.n_sends}</td>
                      <td><span class="chip ${actionClass}">${arm.action}</span></td>
                    </tr>
                  `;
                })
                .join("")}
            </tbody>
          </table>
        </div>

        <div style="display: flex; flex-direction: column; gap: 24px;">
          <div class="paper-panel">
            <h3 class="section-h3" style="color: var(--color-oxblood);">Stop Recommendations</h3>
            <div class="action-list">
              ${
                brief.stop_list.length > 0
                  ? brief.stop_list
                      .map(
                        (item) => `
                        <div class="action-item">
                          <div class="action-item-header">
                            <span class="mono" style="font-weight: 700;">${item.arm_id}</span>
                            <span class="chip chip-stop">STOP</span>
                          </div>
                          <div class="action-reason">${item.reason}</div>
                        </div>
                      `
                      )
                      .join("")
                  : `<div style="font-size: 13px; color: var(--color-ink);">No arms meet stop threshold.</div>`
              }
            </div>
          </div>

          <div class="paper-panel">
            <h3 class="section-h3">Test Recommendations</h3>
            <div class="action-list">
              ${
                brief.test_list.length > 0
                  ? brief.test_list
                      .map(
                        (item) => `
                        <div class="action-item">
                          <div class="action-item-header">
                            <span class="mono" style="font-weight: 700;">${item.arm_id}</span>
                            <span class="chip chip-test">TEST</span>
                          </div>
                          <div class="action-reason">${item.reason}</div>
                        </div>
                      `
                      )
                      .join("")
                  : `<div style="font-size: 13px; color: var(--color-ink);">No arms meet exploratory test criteria.</div>`
              }
            </div>
          </div>
        </div>
      </div>
    `;

    container.innerHTML = html;
  } catch (err) {
    container.innerHTML = `<div class="status-error">Error loading campaign brief: ${err.message}</div>`;
  }
}
