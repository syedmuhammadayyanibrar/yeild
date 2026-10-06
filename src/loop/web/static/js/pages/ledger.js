import { fetchLedger, runNextWeek } from "../api.js";

export async function renderLedgerPage(container, customerId) {
  container.innerHTML = `<div class="status-loading mono">Loading operational results ledger...</div>`;

  try {
    const data = await fetchLedger(customerId);
    const rows = data.rows || data.weeks || [];
    const latestWeek = rows.length > 0 ? Math.max(...rows.map((r) => r.week || r.week_number || 1)) : 12;
    const nextWeek = latestWeek + 1;
    const loopTotal = data.total_meetings_loop !== undefined ? data.total_meetings_loop : (data.total_loop_meetings || 0);
    const cfTotal = data.total_meetings_counterfactual !== undefined ? data.total_meetings_counterfactual : (data.total_counterfactual_meetings || 0);
    const deltaTotal = data.net_gain_meetings !== undefined ? data.net_gain_meetings : (data.net_lift_meetings || 0);
    const pctLift = cfTotal > 0 ? ((deltaTotal / cfTotal) * 100).toFixed(1) : "0.0";
    const gates = data.gates || data.model_gates || [];

    const html = `
      <div class="page-header">
        <div>
          <h1 class="page-title">Results Ledger: Visual Proof</h1>
          <div class="page-subtitle">
            Counterfactual audit running on identical lead pools: Yeild Bayesian Policy vs Old Playbook.
          </div>
        </div>
      </div>

      <div class="ledger-hero">
        <div>
          <div class="mono" style="font-size: 11px; text-transform: uppercase; color: var(--color-silver); letter-spacing: 0.08em;">
            Live Proof State &bull; Week 1 &ndash; ${latestWeek}
          </div>
          <div style="font-family: var(--font-display); font-size: 38px; font-weight: 700; color: var(--color-mist); margin: 6px 0;">
            ${loopTotal} <span style="font-size: 20px; font-weight: 400; color: var(--color-silver);">Yeild Meetings</span> vs ${cfTotal.toFixed(1)} <span style="font-size: 20px; font-weight: 400; color: var(--color-silver);">Baseline</span>
          </div>
          <div class="mono" style="font-size: 14px;">
            <span class="${deltaTotal >= 0 ? "ledger-diff-positive" : "ledger-diff-negative"}">
              ${deltaTotal >= 0 ? "+" : ""}${deltaTotal.toFixed(1)} Meetings Booked (${pctLift >= 0 ? "+" : ""}${pctLift}%)
            </span>
            <span style="color: var(--color-silver);"> on identical lead sample sets</span>
          </div>
        </div>

        <div>
          <button id="btn-run-week" class="btn btn-primary mono" style="font-size: 15px; padding: 12px 24px; box-shadow: 0 4px 12px rgba(0, 71, 73, 0.40);">
            &blacktriangleright; Run Week ${nextWeek} Simulation
          </button>
        </div>
      </div>

      <div id="ledger-notice-area"></div>

      <div class="paper-panel">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <h2 class="section-h2" style="margin-bottom: 0;">Operational Proof: Week-by-Week Outcomes</h2>
          <div class="mono" style="font-size: 12px; color: var(--color-ink);">
            Evaluated weeks: ${rows.length} &bull; Next runnable: Week ${nextWeek}
          </div>
        </div>

        <div style="overflow-x: auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th style="width: 70px;">Week</th>
                <th>Applied Strategy / Policy</th>
                <th style="text-align: right;">Sends</th>
                <th style="text-align: right;">Positive Replies</th>
                <th style="text-align: right;">Yeild Meetings</th>
                <th style="text-align: right;">Old Playbook</th>
                <th style="text-align: right;">Net Lift</th>
              </tr>
            </thead>
            <tbody id="ledger-table-body">
              ${renderLedgerRows(rows)}
            </tbody>
          </table>
        </div>
      </div>

      <div class="paper-panel">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <div>
            <h2 class="section-h2" style="margin-bottom: 4px;">Technical Due Diligence: Model Health Gates</h2>
            <div style="font-size: 13px; color: var(--color-ink);">
              Statistical validation criteria required for enterprise production deployment (Gates G1&ndash;G15).
            </div>
          </div>
          <div class="mono" style="font-size: 12px; font-weight: 700;">
            ${gates.filter((g) => g.status === "PASS").length} / ${gates.length} Gates Passing
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

              const gateId = g.gate_id || g.id || "G";
              const desc = g.description || "";
              const val = typeof g.value === "number" ? g.value.toFixed(3) : String(g.value);

              return `
                <div class="scorecard-item">
                  <div style="font-size: 11px; font-weight: 700; text-transform: uppercase;">
                    ${gateId}: ${desc}
                  </div>
                  <div class="mono" style="font-size: 20px; font-weight: 700; margin: 8px 0;">
                    ${val}
                  </div>
                  <span class="${badgeClass}">${g.status}</span>
                </div>
              `;
            })
            .join("")}
        </div>
      </div>
    `;

    container.innerHTML = html;

    const runBtn = document.getElementById("btn-run-week");
    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true;
      runBtn.textContent = `Simulating Week ${nextWeek}...`;

      try {
        const res = await runNextWeek(customerId);
        const row = res.new_row || {};
        const wk = row.week || res.new_week || nextWeek;
        const loopM = row.meetings !== undefined ? row.meetings : (res.loop_meetings || 0);
        const cfM = row.counterfactual_meetings !== undefined ? row.counterfactual_meetings : (res.counterfactual_meetings || 0);
        const deltaM = row.difference_meetings !== undefined ? row.difference_meetings : (res.net_delta || 0);

        showLedgerToast(`Simulated Week ${wk}: Yeild produced ${loopM} meetings vs ${Number(cfM).toFixed(1)} baseline (Net Delta: ${deltaM >= 0 ? "+" : ""}${Number(deltaM).toFixed(1)}).`);
        await renderLedgerPage(container, customerId);
      } catch (err) {
        showLedgerToast(`Simulation error: ${err.message}`, true);
        runBtn.disabled = false;
        runBtn.innerHTML = `&blacktriangleright; Run Week ${nextWeek} Simulation`;
      }
    });

  } catch (err) {
    container.innerHTML = `<div class="status-error">Error loading results ledger: ${err.message}</div>`;
  }
}

function renderLedgerRows(rows) {
  if (rows.length === 0) {
    return `<tr><td colspan="7" class="mono">No campaign weeks recorded in ledger.</td></tr>`;
  }

  const sorted = [...rows].sort((a, b) => (b.week || b.week_number || 0) - (a.week || a.week_number || 0));

  return sorted
    .map((w, idx) => {
      const isLatest = idx === 0;
      const wkNum = w.week || w.week_number || 0;
      const planTxt = w.plan || w.plan_applied || "Bayesian policy";
      const loopMeetings = w.meetings !== undefined ? w.meetings : (w.meetings_loop || 0);
      const cfMeetings = w.counterfactual_meetings !== undefined ? Number(w.counterfactual_meetings).toFixed(1) : (w.meetings_counterfactual ? Number(w.meetings_counterfactual).toFixed(1) : "0.0");
      const diffVal = w.difference_meetings !== undefined ? w.difference_meetings : (w.difference || 0);
      const diffClass = diffVal >= 0 ? "ledger-diff-positive" : "ledger-diff-negative";

      return `
        <tr style="${isLatest ? "background-color: rgba(0, 71, 73, 0.08);" : ""}">
          <td class="mono" style="font-weight: 700;">
            W${wkNum} ${isLatest ? '<span class="chip chip-scale" style="font-size: 10px; padding: 1px 4px;">Latest</span>' : ""}
          </td>
          <td style="font-size: 13px;">
            ${escapeHtml(planTxt)}
          </td>
          <td class="mono" style="text-align: right;">${w.sends}</td>
          <td class="mono" style="text-align: right;">${w.positive_replies}</td>
          <td class="mono" style="text-align: right; font-weight: 700; font-size: 14px;">
            ${loopMeetings}
          </td>
          <td class="mono" style="text-align: right; color: var(--color-ink);">
            ${cfMeetings}
          </td>
          <td class="mono" style="text-align: right; font-weight: 700; font-size: 14px;">
            <span class="${diffClass}">
              ${diffVal > 0 ? "+" : ""}${Number(diffVal).toFixed(1)}
            </span>
          </td>
        </tr>
      `;
    })
    .join("");
}

function showLedgerToast(msg, isError = false) {
  const area = document.getElementById("ledger-notice-area");
  if (!area) return;
  const div = document.createElement("div");
  div.className = "toast-notice";
  if (isError) {
    div.style.backgroundColor = "var(--color-oxblood)";
  }
  div.innerHTML = `<span>${escapeHtml(msg)}</span><button class="close-btn" style="color: var(--color-mist);">&times;</button>`;
  div.querySelector(".close-btn").addEventListener("click", () => div.remove());
  area.prepend(div);
  setTimeout(() => div.remove(), 5000);
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}
