export async function fetchCustomers() {
  const res = await fetch("/api/customers");
  if (!res.ok) {
    throw new Error("Failed to fetch customers");
  }
  return await res.json();
}

export async function fetchBrief(customerId) {
  const res = await fetch(`/api/customers/${customerId}/brief`);
  if (!res.ok) {
    throw new Error(`Failed to fetch brief for ${customerId}`);
  }
  return await res.json();
}

export async function fetchSignals(customerId) {
  const res = await fetch(`/api/customers/${customerId}/signals`);
  if (!res.ok) {
    throw new Error(`Failed to fetch signals for ${customerId}`);
  }
  return await res.json();
}

export async function fetchReplies(customerId, params = {}) {
  const query = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") {
      query.set(k, String(v));
    }
  }
  const res = await fetch(`/api/customers/${customerId}/replies?${query.toString()}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch replies for ${customerId}`);
  }
  return await res.json();
}

export async function classifyReply(text, receivedAt = null) {
  const payload = { text, received_at: receivedAt };
  const res = await fetch("/api/replies/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error("Failed to classify reply");
  }
  return await res.json();
}

export async function fetchEvaluation() {
  const res = await fetch("/api/evaluation");
  if (res.status === 404) {
    const data = await res.json();
    return { skipped: true, detail: data.detail };
  }
  if (!res.ok) {
    throw new Error("Failed to fetch evaluation results");
  }
  return await res.json();
}

export async function fetchInbox(customerId) {
  const res = await fetch(`/api/customers/${customerId}/inbox`);
  if (!res.ok) {
    throw new Error(`Failed to fetch inbox for ${customerId}`);
  }
  return await res.json();
}

export async function submitInboxAction(replyId, customerId, action, finalText = null, correctedClass = null) {
  const res = await fetch(`/api/replies/${replyId}/action?customer_id=${customerId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, final_text: finalText, corrected_class: correctedClass }),
  });
  if (!res.ok) {
    throw new Error(`Failed to submit action for reply ${replyId}`);
  }
  return await res.json();
}

export async function fetchTodayLeads(customerId) {
  const res = await fetch(`/api/customers/${customerId}/leads/today`);
  if (!res.ok) {
    throw new Error(`Failed to fetch today leads for ${customerId}`);
  }
  return await res.json();
}

export async function fetchPlan(customerId) {
  const res = await fetch(`/api/customers/${customerId}/plan`);
  if (!res.ok) {
    throw new Error(`Failed to fetch plan for ${customerId}`);
  }
  return await res.json();
}

export async function applyPlan(customerId, shares = null) {
  const res = await fetch(`/api/customers/${customerId}/plan/apply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ shares }),
  });
  if (!res.ok) {
    throw new Error(`Failed to apply plan for ${customerId}`);
  }
  return await res.json();
}

export async function fetchLedger(customerId) {
  const res = await fetch(`/api/customers/${customerId}/ledger`);
  if (!res.ok) {
    throw new Error(`Failed to fetch ledger for ${customerId}`);
  }
  return await res.json();
}

export async function runNextWeek(customerId) {
  const res = await fetch(`/api/customers/${customerId}/weeks/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    throw new Error(`Failed to run next week for ${customerId}`);
  }
  return await res.json();
}

