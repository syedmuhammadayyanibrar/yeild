import { fetchTodayLeads } from "../api.js";

export async function renderLeadsPage(container, customerId) {
  container.innerHTML = `<div class="status-loading mono">Loading today's ranked leads...</div>`;

  try {
    const data = await fetchTodayLeads(customerId);
    const leads = data.leads || data.items || [];

    const totalLeads = leads.length;
    const avgOdds = totalLeads > 0
      ? leads.reduce((acc, l) => acc + (l.expected_meetings_per_100 || 0), 0) / totalLeads
      : 0;
    const staleCount = leads.filter((l) => l.stale_by_friday).length;

    const topOpener = leads.length > 0 && (leads[0].opener_type || leads[0].recommended_opener)
      ? (leads[0].opener_type || leads[0].recommended_opener).replace(/_/g, " ")
      : "N/A";

    const html = `
      <div class="page-header">
        <div>
          <h1 class="page-title">Today's Ranked Call Sheet</h1>
          <div class="page-subtitle">
            Bayesian posterior lead ranking with signal decay half-life tracking &amp; sequencer export.
          </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
          <a href="/api/customers/${customerId}/leads/export.csv" download="leads_today_${customerId}.csv" class="btn btn-primary mono" style="text-decoration: none;">
            &darr; Export Sequencer CSV
          </a>
        </div>
      </div>

      <div class="kpi-grid">
        <div class="kpi-tile">
          <div class="kpi-label">Ready Leads Today</div>
          <div class="kpi-value">${totalLeads}</div>
          <div class="kpi-subtext">Scored from pool of ${totalLeads * 4}</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Avg Meeting Probability</div>
          <div class="kpi-value">${avgOdds.toFixed(1)}%</div>
          <div class="kpi-subtext">Posterior predictive mean</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Stale by Friday</div>
          <div class="kpi-value" style="color: var(--color-oxblood);">${staleCount}</div>
          <div class="kpi-subtext">&gt; 1 half-life signal decay</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Top Opener Arm</div>
          <div class="kpi-value" style="font-size: 22px; text-transform: uppercase;">
            ${topOpener}
          </div>
          <div class="kpi-subtext">Highest expected meeting lift</div>
        </div>
      </div>

      <div class="paper-panel">
        <div class="filter-bar" style="justify-content: space-between;">
          <input type="text" id="leads-search" class="input-search" placeholder="Filter by company, lead, or title...">
          <div class="mono" style="font-size: 12px; color: var(--color-ink);">
            Showing <span id="leads-count">${leads.length}</span> of ${leads.length} leads
          </div>
        </div>

        <div style="overflow-x: auto;">
          <table class="data-table" id="leads-table">
            <thead>
              <tr>
                <th style="width: 50px;">Rank</th>
                <th>Company &amp; Title</th>
                <th>Signal &amp; Age</th>
                <th style="text-align: right;">ICP Fit</th>
                <th>Recommended Opener</th>
                <th style="text-align: right;">Meeting Odds</th>
                <th>Decay Risk</th>
                <th>Posterior Log-Odds Decomposed</th>
              </tr>
            </thead>
            <tbody id="leads-body">
              ${renderTableRows(leads)}
            </tbody>
          </table>
        </div>
      </div>
    `;

    container.innerHTML = html;

    const searchInput = document.getElementById("leads-search");
    searchInput.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      const filtered = leads.filter((l) => {
        const comp = (l.company_name || l.company || "").toLowerCase();
        const lead = (l.lead_id || l.lead_name || "").toLowerCase();
        const tit = (l.title || "").toLowerCase();
        const sig = (l.signal_type || "").toLowerCase();
        return comp.includes(q) || lead.includes(q) || tit.includes(q) || sig.includes(q);
      });
      document.getElementById("leads-body").innerHTML = renderTableRows(filtered);
      document.getElementById("leads-count").textContent = filtered.length;
    });

  } catch (err) {
    container.innerHTML = `<div class="status-error">Error loading leads call sheet: ${err.message}</div>`;
  }
}

function renderTableRows(leads) {
  if (leads.length === 0) {
    return `<tr><td colspan="8" class="mono">No leads matching filter criteria.</td></tr>`;
  }

  return leads
    .map((l, idx) => {
      const rankNum = idx + 1;
      const odds = l.expected_meetings_per_100 !== undefined
        ? l.expected_meetings_per_100.toFixed(1)
        : (l.meeting_probability ? (l.meeting_probability * 100).toFixed(1) : "0.0");
      const icpScore = Math.round((l.icp_fit || l.icp_fit_score || 0) * 100);
      const b = l.log_odds_terms || l.explanation || { base: 0, icp: 0, signal: 0, opener: 0 };
      const compName = l.company_name || l.company || "Prospect Co";
      const opName = (l.opener_type || l.recommended_opener || "custom_pitch").replace(/_/g, " ");

      return `
        <tr>
          <td class="mono" style="font-weight: 700; color: var(--color-ink);">#${rankNum}</td>
          <td>
            <div style="font-weight: 600; color: var(--color-ink);">${escapeHtml(compName)}</div>
            <div class="mono" style="font-size: 11px; color: var(--color-ink);">
              ${escapeHtml(l.title)} &bull; ID: ${escapeHtml(l.lead_id || "")}
            </div>
          </td>
          <td>
            <div style="font-weight: 600; font-size: 12px; text-transform: uppercase;">
              ${escapeHtml((l.signal_type || "").replace(/_/g, " "))}
            </div>
            <div class="mono" style="font-size: 11px; color: var(--color-ink);">
              ${Math.round(l.signal_age_days)}d ago
            </div>
          </td>
          <td class="mono" style="text-align: right; font-weight: 600;">
            ${icpScore}%
          </td>
          <td>
            <span class="chip chip-scale" style="font-size: 11px;">
              ${escapeHtml(opName)}
            </span>
          </td>
          <td class="mono" style="text-align: right; font-size: 14px; font-weight: 700; color: var(--color-ink);">
            ${odds}%
          </td>
          <td>
            ${
              l.stale_by_friday
                ? `<span class="badge-decay">STALE BY FRI</span>`
                : `<span class="mono" style="font-size: 11px; color: var(--color-teal); font-weight: 600;">ACTIVE</span>`
            }
          </td>
          <td>
            <div class="logodds-breakdown">
              <span class="logodds-term" title="Baseline intercept">b:${(b.base || 0) > 0 ? "+" : ""}${(b.base || 0).toFixed(2)}</span>
              <span class="logodds-term ${(b.icp || 0) >= 0 ? "logodds-term-pos" : "logodds-term-neg"}" title="ICP fit lift">icp:${(b.icp || 0) > 0 ? "+" : ""}${(b.icp || 0).toFixed(2)}</span>
              <span class="logodds-term ${(b.signal || 0) >= 0 ? "logodds-term-pos" : "logodds-term-neg"}" title="Signal decay lift">sig:${(b.signal || 0) > 0 ? "+" : ""}${(b.signal || 0).toFixed(2)}</span>
              <span class="logodds-term ${(b.opener || 0) >= 0 ? "logodds-term-pos" : "logodds-term-neg"}" title="Opener lift">op:${(b.opener || 0) > 0 ? "+" : ""}${(b.opener || 0).toFixed(2)}</span>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
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
