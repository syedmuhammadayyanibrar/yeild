import { fetchReplies, classifyReply } from "../api.js";

export async function renderRepliesPage(container, customerId) {
  container.innerHTML = `<div class="status-loading mono">Loading reply desk for ${customerId}...</div>`;

  let currentParams = {
    cls: "",
    language: "",
    needs_review: "",
    q: "",
  };

  async function loadData() {
    try {
      const data = await fetchReplies(customerId, currentParams);
      renderUI(data);
    } catch (err) {
      container.innerHTML = `<div class="status-error">Error loading replies: ${err.message}</div>`;
    }
  }

  function renderUI(data) {
    const total = data.total;
    const autoShare = total > 0 ? (data.auto_handled_count / total) * 100 : 0;
    const reviewShare = total > 0 ? (data.review_count / total) * 100 : 0;

    const html = `
      <div class="page-header">
        <div>
          <h1 class="page-title">Reply Desk</h1>
          <div class="page-subtitle mono">
            ${total} total replies &bull; Auto-handled: ${autoShare.toFixed(1)}% &bull; Routed to review: ${reviewShare.toFixed(1)}%
          </div>
        </div>
      </div>

      <div class="paper-panel" style="margin-bottom: 24px;">
        <h3 class="section-h3">Try a Reply</h3>
        <div style="display: flex; gap: 12px; margin-top: 8px;">
          <input type="text" id="live-reply-input" class="input-search" style="flex-grow: 1;" placeholder="Paste an email reply here to test real-time classification and drafting...">
          <button id="live-classify-btn" class="btn btn-primary">Classify &amp; Draft</button>
        </div>
      </div>

      <div class="paper-panel">
        <div class="filter-bar">
          <input type="text" id="filter-search" class="input-search" placeholder="Search reply text..." value="${currentParams.q}">
          <select id="filter-lang" class="customer-selector" style="background-color: var(--color-mist); color: var(--color-ink); border-color: var(--color-silver);">
            <option value="">All Languages</option>
            <option value="en" ${currentParams.language === "en" ? "selected" : ""}>EN</option>
            <option value="fr" ${currentParams.language === "fr" ? "selected" : ""}>FR</option>
            <option value="de" ${currentParams.language === "de" ? "selected" : ""}>DE</option>
            <option value="es" ${currentParams.language === "es" ? "selected" : ""}>ES</option>
          </select>
          <select id="filter-review" class="customer-selector" style="background-color: var(--color-mist); color: var(--color-ink); border-color: var(--color-silver);">
            <option value="">All Routings</option>
            <option value="true" ${currentParams.needs_review === "true" ? "selected" : ""}>Needs Review</option>
            <option value="false" ${currentParams.needs_review === "false" ? "selected" : ""}>Auto-Handled</option>
          </select>
        </div>

        <table class="data-table">
          <thead>
            <tr>
              <th>Received</th>
              <th>Class</th>
              <th>Confidence</th>
              <th>Snippet (120 chars)</th>
              <th>Lang</th>
              <th>Routing</th>
            </tr>
          </thead>
          <tbody id="replies-table-body">
            ${data.replies
              .map((rep, idx) => {
                const snippet = rep.text.length > 120 ? rep.text.slice(0, 120) + "..." : rep.text;
                const isPos = rep.predicted_class === "interested" || rep.predicted_class === "info_request";
                const isNeg = rep.predicted_class === "unsubscribe" || rep.predicted_class === "not_relevant";
                const chipClass = isPos ? "chip-positive" : isNeg ? "chip-destructive" : "chip-neutral";
                const routingChip = rep.routed_to_review ? "chip-stop" : "chip-scale";
                const routingText = rep.routed_to_review ? "Review" : "Auto";

                return `
                  <tr class="clickable" data-idx="${idx}">
                    <td class="mono" style="font-size: 11px;">${rep.received_at.slice(0, 16).replace("T", " ")}</td>
                    <td><span class="chip ${chipClass}">${rep.predicted_class}</span></td>
                    <td class="mono">${(rep.confidence * 100).toFixed(1)}%</td>
                    <td>${snippet}</td>
                    <td class="mono" style="text-transform: uppercase;">${rep.true_language}</td>
                    <td><span class="chip ${routingChip}">${routingText}</span></td>
                  </tr>
                `;
              })
              .join("")}
          </tbody>
        </table>
      </div>

      <div id="reply-drawer" class="drawer">
        <div class="drawer-header">
          <h3 class="section-h3" id="drawer-title" style="margin: 0;">Reply Details</h3>
          <button id="drawer-close" class="close-btn">&times;</button>
        </div>
        <div id="drawer-content"></div>
      </div>
    `;

    container.innerHTML = html;

    const drawer = document.getElementById("reply-drawer");
    const drawerClose = document.getElementById("drawer-close");
    const drawerContent = document.getElementById("drawer-content");

    drawerClose.addEventListener("click", () => {
      drawer.classList.remove("open");
    });

    const rows = container.querySelectorAll("tr.clickable");
    rows.forEach((r) => {
      r.addEventListener("click", () => {
        const idx = parseInt(r.getAttribute("data-idx"), 10);
        const item = data.replies[idx];
        showDrawer(item);
      });
    });

    const searchInput = document.getElementById("filter-search");
    searchInput.addEventListener("change", (e) => {
      currentParams.q = e.target.value;
      loadData();
    });

    const langSelect = document.getElementById("filter-lang");
    langSelect.addEventListener("change", (e) => {
      currentParams.language = e.target.value;
      loadData();
    });

    const reviewSelect = document.getElementById("filter-review");
    reviewSelect.addEventListener("change", (e) => {
      currentParams.needs_review = e.target.value;
      loadData();
    });

    const liveBtn = document.getElementById("live-classify-btn");
    const liveInput = document.getElementById("live-reply-input");
    liveBtn.addEventListener("click", async () => {
      const text = liveInput.value.trim();
      if (!text) return;
      liveBtn.disabled = true;
      liveBtn.innerText = "Analyzing...";
      try {
        const result = await classifyReply(text);
        showDrawer({
          text: text,
          predicted_class: result.predicted_class,
          confidence: result.confidence,
          class_probabilities: result.probabilities,
          referral_name: result.referral_name,
          referral_email: result.referral_email,
          revisit_date: result.revisit_date,
          draft_reply: result.draft_reply,
          validators: result.validators,
          routed_to_review: result.routed_to_review,
        });
      } catch (err) {
        alert("Failed to classify: " + err.message);
      } finally {
        liveBtn.disabled = false;
        liveBtn.innerText = "Classify & Draft";
      }
    });

    function showDrawer(item) {
      const probs = item.class_probabilities || {};
      const sortedProbs = Object.entries(probs)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4);

      let probsHtml = `<div style="display: flex; flex-direction: column; gap: 8px; margin: 12px 0;">`;
      sortedProbs.forEach(([cls, prob], i) => {
        const barColor = i === 0 ? "var(--color-teal)" : "var(--color-sand-deep)";
        const width = Math.round(prob * 100);
        probsHtml += `
          <div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
              <span class="mono">${cls}</span>
              <span class="mono">${(prob * 100).toFixed(1)}%</span>
            </div>
            <div class="bar-track">
              <div style="height: 100%; width: ${width}%; background-color: ${barColor};"></div>
            </div>
          </div>
        `;
      });
      probsHtml += `</div>`;

      let valHtml = `<div style="display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0;">`;
      for (const [k, pass] of Object.entries(item.validators || {})) {
        const chip = pass ? "chip-scale" : "chip-stop";
        valHtml += `<span class="chip ${chip}">${k.replace(/_/g, " ")}: ${pass ? "PASS" : "FAIL"}</span>`;
      }
      valHtml += `</div>`;

      drawerContent.innerHTML = `
        <div>
          <h4 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">Inbound Reply Text</h4>
          <div style="background-color: var(--color-mist); border: 1px solid var(--color-silver); padding: 12px; border-radius: var(--radius); font-size: 13px; white-space: pre-wrap;">${item.text}</div>
        </div>

        <div>
          <h4 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em;">Top-4 Predicted Intent Classes</h4>
          ${probsHtml}
        </div>

        <div>
          <h4 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">Extracted Entities</h4>
          <table class="data-table">
            <tbody>
              <tr><td style="font-weight: 600;">Referral Name</td><td class="mono">${item.referral_name || "None"}</td></tr>
              <tr><td style="font-weight: 600;">Referral Email</td><td class="mono">${item.referral_email || "None"}</td></tr>
              <tr><td style="font-weight: 600;">Revisit Date</td><td class="mono">${item.revisit_date || "None"}</td></tr>
            </tbody>
          </table>
        </div>

        <div>
          <h4 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">Deterministic Guardrails</h4>
          ${valHtml}
        </div>

        <div>
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <h4 style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em;">Generated Draft Response</h4>
            <button id="copy-draft-btn" class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;">Copy</button>
          </div>
          <textarea id="drawer-draft-text" class="textarea-draft">${item.draft_reply || "No draft required (Out of Office)."}</textarea>
        </div>
      `;

      drawer.classList.add("open");

      const copyBtn = document.getElementById("copy-draft-btn");
      if (copyBtn) {
        copyBtn.addEventListener("click", () => {
          const ta = document.getElementById("drawer-draft-text");
          navigator.clipboard.writeText(ta.value);
          copyBtn.innerText = "Copied";
          setTimeout(() => (copyBtn.innerText = "Copy"), 1500);
        });
      }
    }
  }

  loadData();
}
