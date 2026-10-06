import { fetchInbox, submitInboxAction } from "../api.js";

export async function renderInboxPage(container, customerId) {
  container.innerHTML = `<div class="status-loading mono">Loading operator triage inbox...</div>`;

  try {
    const data = await fetchInbox(customerId);
    const needsYou = data.needs_you || [];
    const readyToSend = data.ready_to_send || [];
    const doneAuto = data.done_automatically || [];

    const totalReplies = needsYou.length + readyToSend.length + doneAuto.length;
    const interventionRate = totalReplies > 0
      ? Math.round((needsYou.length / totalReplies) * 100)
      : 0;

    const html = `
      <div class="page-header">
        <div>
          <h1 class="page-title">Operator Triage Inbox</h1>
          <div class="page-subtitle">
            Autonomous categorization with human-in-the-loop exception routing for high-stakes replies.
          </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
          <button id="btn-refresh-inbox" class="btn btn-secondary mono">Refresh Inbox</button>
        </div>
      </div>

      <div class="kpi-grid">
        <div class="kpi-tile">
          <div class="kpi-label">Needs Your Review</div>
          <div class="kpi-value">${needsYou.length}</div>
          <div class="kpi-subtext">Exceptions &amp; meeting intent</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Ready to Send</div>
          <div class="kpi-value">${readyToSend.length}</div>
          <div class="kpi-subtext">Pre-generated drafts</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Handled Autonomously</div>
          <div class="kpi-value">${doneAuto.length}</div>
          <div class="kpi-subtext">Unsubs, OOO &amp; referrals</div>
        </div>
        <div class="kpi-tile">
          <div class="kpi-label">Human Interventions</div>
          <div class="kpi-value">${interventionRate}%</div>
          <div class="kpi-subtext">Routing selectivity rate</div>
        </div>
      </div>

      <div id="inbox-notice-area"></div>

      <div class="triage-group">
        <div class="section-h2-dark" style="display: flex; justify-content: space-between; align-items: center;">
          <span>1. Needs You (${needsYou.length})</span>
          <span class="mono" style="font-size: 13px; color: var(--color-silver);">Action required to progress conversation</span>
        </div>

        ${
          needsYou.length === 0
            ? `<div class="paper-panel mono" style="color: var(--color-ink);">All clear. No urgent replies waiting for operator approval.</div>`
            : needsYou
                .map((item) => {
                  const confPct = Math.round((item.confidence || 0) * 100);
                  const predClass = (item.predicted_class || "REVIEW").toUpperCase();
                  const isMeeting = predClass === "MEETING_REQUEST" || predClass === "INTERESTED";
                  const badgeClass = isMeeting ? "chip-positive" : "chip-neutral";
                  const replyText = item.text || item.original_text || "";
                  const leadName = item.lead_name || ("Lead #" + (item.send_id ? item.send_id.slice(-4) : (item.reply_id ? item.reply_id.slice(-4) : "0000")));
                  const company = item.company || ("Account (" + (item.customer_id || customerId) + ")");
                  const whyReason = item.why || (isMeeting ? "Positive meeting intent requires human confirmation" : "Flagged for human verification");
                  const draftText = item.draft_reply || item.suggested_draft || "";
                  const receivedDate = item.received_at ? String(item.received_at).slice(0, 19).replace("T", " ") : "Recently";

                  return `
                    <div class="triage-card" id="card-${item.reply_id}">
                      <div class="triage-header">
                        <div>
                          <div style="font-size: 16px; font-weight: 700; color: var(--color-ink);">
                            ${escapeHtml(leadName)} &bull; <span style="font-weight: 500;">${escapeHtml(company)}</span>
                          </div>
                          <div class="mono" style="font-size: 11px; color: var(--color-ink); margin-top: 2px;">
                            Received: ${receivedDate} &bull; ID: ${escapeHtml(item.reply_id)}
                          </div>
                        </div>
                        <div style="display: flex; gap: 8px; align-items: center;">
                          <span class="chip ${badgeClass}">${predClass} (${confPct}%)</span>
                        </div>
                      </div>

                      <div class="triage-body">
                        <div style="font-weight: 600; font-size: 12px; text-transform: uppercase; color: var(--color-ink); margin-bottom: 2px;">Prospect Reply:</div>
                        <div class="triage-snippet">&ldquo;${escapeHtml(replyText)}&rdquo;</div>

                        <div style="font-size: 12px; margin: 6px 0; color: var(--color-ink);">
                          <strong style="text-transform: uppercase;">Why Yeild routed to you:</strong> ${escapeHtml(whyReason)}
                        </div>

                        ${
                          draftText
                            ? `
                            <div style="font-weight: 600; font-size: 12px; text-transform: uppercase; color: var(--color-ink); margin-top: 8px;">Recommended Draft:</div>
                            <textarea id="draft-${item.reply_id}" class="textarea-draft" style="height: 80px; margin-top: 4px;">${escapeHtml(draftText)}</textarea>
                          `
                            : ""
                        }
                      </div>

                      <div class="triage-actions">
                        <button class="btn btn-primary btn-approve" data-reply-id="${item.reply_id}">
                          Approve &amp; Send
                        </button>
                        <div style="display: flex; align-items: center; gap: 6px; margin-left: auto;">
                          <span class="mono" style="font-size: 12px;">Reclassify:</span>
                          <select id="reclass-${item.reply_id}" class="customer-selector" style="background-color: var(--color-mist); color: var(--color-ink); padding: 4px 8px;">
                            <option value="MEETING_REQUEST">MEETING_REQUEST</option>
                            <option value="INTERESTED">INTERESTED</option>
                            <option value="REFERRAL">REFERRAL</option>
                            <option value="OBJECTION">OBJECTION</option>
                            <option value="NOT_INTERESTED">NOT_INTERESTED</option>
                            <option value="UNSUBSCRIBE">UNSUBSCRIBE</option>
                            <option value="OUT_OF_OFFICE">OUT_OF_OFFICE</option>
                          </select>
                          <button class="btn btn-secondary btn-reclassify" data-reply-id="${item.reply_id}" style="color: var(--color-ink); border-color: var(--color-ink);">
                            Save
                          </button>
                        </div>
                      </div>
                    </div>
                  `;
                })
                .join("")
        }
      </div>

      <div class="triage-group">
        <div class="section-h2-dark" style="display: flex; justify-content: space-between; align-items: center;">
          <span>2. Ready to Send (${readyToSend.length})</span>
          <span class="mono" style="font-size: 13px; color: var(--color-silver);">Autonomous high-confidence objections &amp; inquiries</span>
        </div>

        ${
          readyToSend.length === 0
            ? `<div class="paper-panel mono" style="color: var(--color-ink);">No queued auto-replies pending delivery.</div>`
            : readyToSend
                .map((item) => {
                  const replyText = item.text || item.original_text || "";
                  const leadName = item.lead_name || ("Lead #" + (item.send_id ? item.send_id.slice(-4) : (item.reply_id ? item.reply_id.slice(-4) : "0000")));
                  const company = item.company || ("Account (" + (item.customer_id || customerId) + ")");
                  const predClass = (item.predicted_class || "ROUTINE").toUpperCase();
                  const draftText = item.draft_reply || item.suggested_draft || "";

                  return `
                    <div class="triage-card" id="card-${item.reply_id}">
                      <div class="triage-header">
                        <div>
                          <div style="font-size: 15px; font-weight: 700; color: var(--color-ink);">
                            ${escapeHtml(leadName)} &bull; <span style="font-weight: 500;">${escapeHtml(company)}</span>
                          </div>
                          <div class="mono" style="font-size: 11px; color: var(--color-ink);">
                            ${predClass} &bull; Confidence: ${Math.round((item.confidence || 0) * 100)}%
                          </div>
                        </div>
                        <button class="btn btn-primary btn-approve" data-reply-id="${item.reply_id}" style="padding: 4px 12px; font-size: 12px;">
                          Send Now
                        </button>
                      </div>
                      <div class="triage-body">
                        <div class="triage-snippet" style="margin: 4px 0;">&ldquo;${escapeHtml(replyText)}&rdquo;</div>
                        ${
                          draftText
                            ? `<div class="triage-draft-box" style="font-size: 12px; margin-top: 6px;"><strong>Draft:</strong> ${escapeHtml(draftText)}</div>`
                            : ""
                        }
                      </div>
                    </div>
                  `;
                })
                .join("")
        }
      </div>

      <div class="triage-group">
        <div class="section-h2-dark" style="display: flex; justify-content: space-between; align-items: center;">
          <span>3. Autonomous Action Ledger (${doneAuto.length})</span>
          <span class="mono" style="font-size: 13px; color: var(--color-silver);">Automatically resolved without operator overhead</span>
        </div>

        <div class="paper-panel" style="padding: 0; overflow-x: auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Lead / Company</th>
                <th>Classification</th>
                <th>Incoming Snippet</th>
                <th>Autonomous Resolution</th>
                <th>Action Taken</th>
              </tr>
            </thead>
            <tbody>
              ${
                doneAuto.length === 0
                  ? `<tr><td colspan="5" class="mono">No automated resolutions recorded yet.</td></tr>`
                  : doneAuto
                      .map((item) => {
                        const replyText = item.text || item.original_text || "";
                        const snippet = replyText.length > 75 ? replyText.slice(0, 75) + "..." : replyText;
                        const leadName = item.lead_name || ("Lead #" + (item.send_id ? item.send_id.slice(-4) : (item.reply_id ? item.reply_id.slice(-4) : "0000")));
                        const company = item.company || ("Account (" + (item.customer_id || customerId) + ")");
                        const predClass = (item.predicted_class || "").toUpperCase();
                        const whyReason = item.why || (
                          predClass === "OUT_OF_OFFICE" ? "Auto-scheduled follow-up on return" :
                          predClass === "UNSUBSCRIBE" ? "Auto-suppressed from future outreach" :
                          predClass === "REFERRAL" ? "Referral contact extracted and routed" :
                          "Autonomous policy applied"
                        );
                        const actionLabel = item.action_taken || item.suggested_action || (
                          predClass === "UNSUBSCRIBE" ? "SUPPRESSED" :
                          predClass === "OUT_OF_OFFICE" ? "FOLLOWUP QUEUED" :
                          predClass === "REFERRAL" ? "PIPELINE CREATED" :
                          "RESOLVED"
                        );

                        return `
                          <tr>
                            <td>
                              <div style="font-weight: 600;">${escapeHtml(leadName)}</div>
                              <div class="mono" style="font-size: 11px; color: var(--color-ink);">${escapeHtml(company)}</div>
                            </td>
                            <td><span class="chip chip-neutral">${predClass}</span></td>
                            <td style="max-width: 320px; font-style: italic;">&ldquo;${escapeHtml(snippet)}&rdquo;</td>
                            <td>${escapeHtml(whyReason)}</td>
                            <td><span class="chip chip-scale">${escapeHtml(actionLabel)}</span></td>
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

    document.getElementById("btn-refresh-inbox").addEventListener("click", () => {
      renderInboxPage(container, customerId);
    });

    container.querySelectorAll(".btn-approve").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const replyId = e.currentTarget.getAttribute("data-reply-id");
        const draftEl = document.getElementById(`draft-${replyId}`);
        const draftText = draftEl ? draftEl.value : null;

        btn.disabled = true;
        btn.textContent = "Sending...";
        try {
          await submitInboxAction(replyId, customerId, "approve", draftText);
          showToast(`Action confirmed: Reply ${replyId} approved and sent.`);
          const card = document.getElementById(`card-${replyId}`);
          if (card) {
            card.remove();
          }
        } catch (err) {
          showToast(`Error: ${err.message}`, true);
          btn.disabled = false;
          btn.textContent = "Approve & Send";
        }
      });
    });

    container.querySelectorAll(".btn-reclassify").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const replyId = e.currentTarget.getAttribute("data-reply-id");
        const selectEl = document.getElementById(`reclass-${replyId}`);
        const newClass = selectEl ? selectEl.value : null;

        btn.disabled = true;
        btn.textContent = "Saving...";
        try {
          await submitInboxAction(replyId, customerId, "reclassify", null, newClass);
          showToast(`Reclassified reply ${replyId} as ${newClass}. Model logged correction.`);
          renderInboxPage(container, customerId);
        } catch (err) {
          showToast(`Error: ${err.message}`, true);
          btn.disabled = false;
          btn.textContent = "Save";
        }
      });
    });
  } catch (err) {
    container.innerHTML = `<div class="status-error">Error loading operator inbox: ${err.message}</div>`;
  }
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

function showToast(msg, isError = false) {
  const area = document.getElementById("inbox-notice-area");
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
