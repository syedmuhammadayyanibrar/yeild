import { fetchPlan, applyPlan } from "../api.js";

export async function renderPlanPage(container, customerId) {
  container.innerHTML = `<div class="status-loading mono">Loading next week's optimization plan...</div>`;

  try {
    const data = await fetchPlan(customerId);
    const recs = data.recommendations || [];
    const pBeats = ((data.p_recommended_beats_current || data.p_loop_beats_current || 0.95) * 100).toFixed(1);
    const expMeetings = (data.recommended_expected_meetings || data.expected_meetings_per_100 || 0).toFixed(1);
    const gainPct = (data.relative_gain_pct || 0).toFixed(1);

    const scaleArms = recs.filter((r) => r.action === "SCALE");
    const stopArms = recs.filter((r) => r.action === "STOP");
    const testArms = recs.filter((r) => r.action === "TEST");
    const holdArms = recs.filter((r) => r.action === "HOLD");

    const html = `
      <div class="page-header">
        <div>
          <h1 class="page-title">Next Week's Growth Plan</h1>
          <div class="page-subtitle">
            Posterior-guided arm allocation maximizing expected meeting rate subject to exploration bounds.
          </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
          <button id="btn-apply-plan" class="btn btn-primary mono" style="padding: 10px 20px; font-size: 14px;">
            &check; Apply Plan to Sequencer
          </button>
        </div>
      </div>

      <div class="kpi-grid">
        <div class="kpi-tile">
          <div class="kpi-label">P(Yeild &gt; Current Playbook)</div>
          <div class="kpi-value" style="color: var(--color-teal);">${pBeats}%</div>
          <div class="kpi-subtext">Monte Carlo posterior certitude</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Expected Meetings / 100</div>
          <div class="kpi-value">${expMeetings}</div>
          <div class="kpi-subtext">Relative gain: +${gainPct}%</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Scaling Arms</div>
          <div class="kpi-value">${scaleArms.length}</div>
          <div class="kpi-subtext">Budget &amp; volume increases</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Pruned / Stopped Arms</div>
          <div class="kpi-value" style="color: var(--color-oxblood);">${stopArms.length}</div>
          <div class="kpi-subtext">Zeroed out to prevent wasted sends</div>
        </div>
      </div>

      <div id="plan-notice-area"></div>

      <div class="memo-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; border-bottom: 2px solid var(--color-ink); padding-bottom: 12px;">
          <div>
            <div class="mono" style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: var(--color-sand-deep);">
              OPERATOR STRATEGY MEMO
            </div>
            <div style="font-family: var(--font-display); font-size: 26px; font-weight: 700; text-transform: uppercase;">
              Executive Campaign Directives
            </div>
          </div>
          <div class="mono" style="text-align: right; font-size: 12px;">
            Target: <strong>${customerId}</strong><br>
            Baseline: ${data.current_expected_meetings ? data.current_expected_meetings.toFixed(1) : "0"} meetings/100
          </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px;">
          <div style="background: rgba(0, 71, 73, 0.08); border-left: 4px solid var(--color-teal); padding: 12px;">
            <div class="mono" style="font-size: 11px; font-weight: 700; color: var(--color-teal);">SCALE (${scaleArms.length})</div>
            <div style="font-size: 13px; font-weight: 600; margin-top: 4px;">${scaleArms.map((a) => (a.arm_id || a.arm || "").replace(/_/g, " ")).join(", ") || "None"}</div>
          </div>
          <div style="background: rgba(84, 0, 0, 0.08); border-left: 4px solid var(--color-oxblood); padding: 12px;">
            <div class="mono" style="font-size: 11px; font-weight: 700; color: var(--color-oxblood);">STOP (${stopArms.length})</div>
            <div style="font-size: 13px; font-weight: 600; margin-top: 4px;">${stopArms.map((a) => (a.arm_id || a.arm || "").replace(/_/g, " ")).join(", ") || "None"}</div>
          </div>
          <div style="background: rgba(176, 155, 114, 0.12); border-left: 4px solid var(--color-sand); padding: 12px;">
            <div class="mono" style="font-size: 11px; font-weight: 700; color: var(--color-sand-deep);">TEST (${testArms.length})</div>
            <div style="font-size: 13px; font-weight: 600; margin-top: 4px;">${testArms.map((a) => (a.arm_id || a.arm || "").replace(/_/g, " ")).join(", ") || "None"}</div>
          </div>
          <div style="background: rgba(0, 0, 0, 0.04); border-left: 4px solid var(--color-silver); padding: 12px;">
            <div class="mono" style="font-size: 11px; font-weight: 700; color: var(--color-ink);">HOLD (${holdArms.length})</div>
            <div style="font-size: 13px; font-weight: 600; margin-top: 4px;">${holdArms.map((a) => (a.arm_id || a.arm || "").replace(/_/g, " ")).join(", ") || "None"}</div>
          </div>
        </div>

        <div style="margin-top: 24px;">
          <h3 class="section-h3" style="color: var(--color-ink);">Arm Allocation Recommendations &amp; Statistical Proof</h3>
          <table class="data-table">
            <thead>
              <tr>
                <th>Strategy Arm</th>
                <th>Action</th>
                <th style="text-align: right;">Current Share</th>
                <th style="text-align: right;">Recommended Share</th>
                <th style="text-align: right;">Expected Meetings / 100</th>
                <th style="text-align: right;">90% Credible Interval</th>
                <th>Bayesian Rationale</th>
              </tr>
            </thead>
            <tbody>
              ${
                recs
                  .map((r) => {
                    const chipClass =
                      r.action === "SCALE"
                        ? "chip-scale"
                        : r.action === "STOP"
                        ? "chip-stop"
                        : r.action === "TEST"
                        ? "chip-test"
                        : "chip-hold";

                    const curShare = Math.round(r.current_share_pct || (r.current_share ? r.current_share * 100 : 0));
                    const recShare = Math.round(r.recommended_share_pct || (r.recommended_share ? r.recommended_share * 100 : 0));
                    const expVal = (r.expected_meetings || 0).toFixed(1);
                    const lo = (r.lo90 || 0).toFixed(1);
                    const hi = (r.hi90 || 0).toFixed(1);
                    const armLabel = (r.arm_id || r.arm || "").replace(/_/g, " ");

                    return `
                      <tr>
                        <td style="font-weight: 600; text-transform: uppercase; font-size: 12px;">
                          ${escapeHtml(armLabel)}
                        </td>
                        <td><span class="chip ${chipClass}">${r.action}</span></td>
                        <td class="mono" style="text-align: right;">${curShare}%</td>
                        <td class="mono" style="text-align: right; font-weight: 700; font-size: 14px;">
                          ${recShare}%
                        </td>
                        <td class="mono" style="text-align: right; font-weight: 600; color: var(--color-teal);">
                          ${expVal}
                        </td>
                        <td class="mono" style="text-align: right; font-size: 12px;">
                          [${lo}, ${hi}]
                        </td>
                        <td style="font-size: 12px; max-width: 320px; line-height: 16px;">
                          ${escapeHtml(r.rationale)}
                        </td>
                      </tr>
                    `;
                  })
                  .join("")
              }
            </tbody>
          </table>
        </div>
      </div>
    `;

    container.innerHTML = html;

    document.getElementById("btn-apply-plan").addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true;
      btn.textContent = "Applying Plan...";

      try {
        const res = await applyPlan(customerId);
        showPlanToast(`Success: ${res.message}`);
        btn.innerHTML = "&check; Plan Applied to Sequencer";
        btn.style.backgroundColor = "var(--color-teal-deep)";
      } catch (err) {
        showPlanToast(`Error: ${err.message}`, true);
        btn.disabled = false;
        btn.textContent = "Apply Plan to Sequencer";
      }
    });

  } catch (err) {
    container.innerHTML = `<div class="status-error">Error loading growth plan: ${err.message}</div>`;
  }
}

function showPlanToast(msg, isError = false) {
  const area = document.getElementById("plan-notice-area");
  if (!area) return;
  const div = document.createElement("div");
  div.className = "toast-notice";
  if (isError) {
    div.style.backgroundColor = "var(--color-oxblood)";
  }
  div.innerHTML = `<span>${escapeHtml(msg)}</span><button class="close-btn" style="color: var(--color-mist);">&times;</button>`;
  div.querySelector(".close-btn").addEventListener("click", () => div.remove());
  area.prepend(div);
  setTimeout(() => div.remove(), 4000);
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
