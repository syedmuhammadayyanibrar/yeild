import { fetchCustomers } from "./api.js";
import { renderInboxPage } from "./pages/inbox.js";
import { renderLeadsPage } from "./pages/leads.js";
import { renderPlanPage } from "./pages/plan.js";
import { renderLedgerPage } from "./pages/ledger.js";
import { renderEvaluationPage } from "./pages/evaluation.js";
import { renderBriefPage } from "./pages/brief.js";
import { renderRepliesPage } from "./pages/replies.js";
import { renderSignalsPage } from "./pages/signals.js";

let currentCustomer = null;
let currentTab = "inbox";

async function init() {
  const customerSelect = document.getElementById("customer-select");
  const mainContainer = document.getElementById("main-view");
  const navLinks = document.querySelectorAll(".nav-link");

  try {
    const data = await fetchCustomers();
    const customers = data.customers || [];

    customerSelect.innerHTML = "";
    customers.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.customer_id;
      opt.textContent = `${c.name} (${c.industry.replace(/_/g, " ")}) - ${c.n_sends} sends`;
      customerSelect.appendChild(opt);
    });

    if (customers.length > 0) {
      currentCustomer = customers[0].customer_id;
      customerSelect.value = currentCustomer;
    }
  } catch (err) {
    console.error(err);
  }

  customerSelect.addEventListener("change", (e) => {
    currentCustomer = e.target.value;
    renderCurrentTab();
  });

  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const tab = link.getAttribute("data-tab");
      switchTab(tab);
    });
  });

  function switchTab(tab) {
    currentTab = tab;
    window.location.hash = tab;
    navLinks.forEach((l) => {
      if (l.getAttribute("data-tab") === tab) {
        l.classList.add("active");
      } else {
        l.classList.remove("active");
      }
    });
    renderCurrentTab();
  }

  function renderCurrentTab() {
    if (!currentCustomer && currentTab !== "evaluation") {
      mainContainer.innerHTML = `<div class="status-loading mono">No customer selected.</div>`;
      return;
    }

    if (currentTab === "inbox") {
      renderInboxPage(mainContainer, currentCustomer);
    } else if (currentTab === "leads") {
      renderLeadsPage(mainContainer, currentCustomer);
    } else if (currentTab === "plan") {
      renderPlanPage(mainContainer, currentCustomer);
    } else if (currentTab === "ledger") {
      renderLedgerPage(mainContainer, currentCustomer);
    } else if (currentTab === "evaluation") {
      renderEvaluationPage(mainContainer);
    } else if (currentTab === "brief") {
      renderBriefPage(mainContainer, currentCustomer);
    } else if (currentTab === "replies") {
      renderRepliesPage(mainContainer, currentCustomer);
    } else if (currentTab === "signals") {
      renderSignalsPage(mainContainer, currentCustomer);
    }
  }

  const initialHash = window.location.hash.replace("#", "");
  const knownTabs = ["inbox", "leads", "plan", "ledger", "evaluation", "brief", "replies", "signals"];
  if (knownTabs.includes(initialHash)) {
    switchTab(initialHash);
  } else {
    switchTab("inbox");
  }
}

document.addEventListener("DOMContentLoaded", init);
