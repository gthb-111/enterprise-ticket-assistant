const state = {
  currentTicketId: "",
  feedbackSent: false,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const decisionLabels = {
  auto_reply: "自动答复 · 无需人工",
  human_review: "转人工复核",
  escalate: "升级处理",
};

const statusLabels = {
  auto_resolved: "自动解决",
  pending_agent: "待人工",
  escalated: "已升级",
  resolved: "已解决",
};

const categoryColors = {
  billing: "账单",
  account: "账号",
  logistics: "物流",
  refund: "退款",
  technical: "技术",
  service: "投诉",
  general: "通用",
};

function escapeHTML(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `请求失败 (${response.status})`);
  }
  return payload;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("is-visible");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("is-visible"), 2400);
}

function setView(name) {
  $$(".view").forEach((view) => view.classList.toggle("is-active", view.id === name));
  $$(".nav-item").forEach((item) => item.classList.toggle("is-active", item.dataset.view === name));
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (name === "tickets") loadTickets();
  if (name === "knowledge") loadKnowledge();
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

async function loadMetrics() {
  try {
    const metrics = await api("/api/metrics");
    $("#metricTotal").textContent = metrics.total;
    $("#metricAutoRate").textContent = `${Math.round(metrics.auto_resolve_rate * 100)}%`;
    $("#metricEscalation").textContent = metrics.escalated;
    $("#metricHelpful").textContent = metrics.helpful_rate == null
      ? "—"
      : `${Math.round(metrics.helpful_rate * 100)}%`;
    $("#feedbackHint").textContent = metrics.feedback_count
      ? `${metrics.feedback_count} 份用户反馈`
      : "等待用户反馈";

    const priorities = ["P0", "P1", "P2", "P3"];
    const max = Math.max(1, ...priorities.map((p) => metrics.by_priority[p] || 0));
    $("#priorityStack").innerHTML = priorities.map((priority) => {
      const count = metrics.by_priority[priority] || 0;
      return `
        <div class="priority-line">
          <span class="priority-label">${priority}</span>
          <div class="priority-track"><i style="width:${Math.max(5, count / max * 100)}%"></i></div>
          <strong>${count}</strong>
        </div>`;
    }).join("");
  } catch (error) {
    toast(error.message);
  }
}

async function loadRecent() {
  try {
    const data = await api("/api/tickets?limit=5");
    const items = data.items || [];
    if (!items.length) {
      $("#recentList").innerHTML = '<div class="empty-state mini">暂无工单数据，提交一条请求开始体验。</div>';
      return;
    }
    $("#recentList").innerHTML = items.map((ticket) => `
      <button class="recent-item row-action" data-ticket="${escapeHTML(ticket.id)}" type="button">
        <strong>${escapeHTML(ticket.category_label)} · ${escapeHTML(ticket.customer_name)}</strong>
        <p>${escapeHTML(ticket.message_masked)}</p>
        <span class="priority-pill ${escapeHTML(ticket.priority)}">${escapeHTML(ticket.priority)}</span>
      </button>`).join("");
  } catch (error) {
    toast(error.message);
  }
}

function renderResult(payload) {
  const { ticket, decision, explanation } = payload;
  state.currentTicketId = ticket.id;
  state.feedbackSent = false;
  $("#resultPanel").classList.remove("is-hidden");
  $("#resultStatus").className = `result-status ${decision}`;
  $("#resultStatus").textContent = decisionLabels[decision] || decision;
  $("#resultCategory").textContent = ticket.category_label;
  $("#resultPriority").textContent = ticket.priority;
  $("#resultTeam").textContent = ticket.assigned_team;
  $("#resultTicketId").textContent = ticket.id;
  $("#resultAnswer").textContent = ticket.answer;
  $("#generationMode").textContent = ticket.generation_mode === "llm_grounded"
    ? "大模型 · 证据约束"
    : ticket.generation_mode === "extractive_grounded"
      ? "本地抽取 · 证据约束"
      : "安全兜底策略";
  $("#resultConfidence").textContent = `${Math.round(ticket.confidence * 100)}%`;
  $("#resultRelevance").textContent = `${Math.round(ticket.relevance * 100)}%`;
  requestAnimationFrame(() => {
    $("#confidenceBar").style.width = `${Math.round(ticket.confidence * 100)}%`;
    $("#relevanceBar").style.width = `${Math.round(ticket.relevance * 100)}%`;
  });
  $("#reasonList").innerHTML = [
    ...ticket.reasons,
    explanation,
  ].map((reason) => `<li>${escapeHTML(reason)}</li>`).join("");
  const citations = ticket.citations || [];
  $("#citationCount").textContent = `${citations.length} 条`;
  $("#citationList").innerHTML = citations.length
    ? citations.map((citation, index) => `
      <details class="citation-item" ${index === 0 ? "open" : ""}>
        <summary>${escapeHTML(citation.title)}<span>${Math.round(citation.relevance * 100)}%</span></summary>
        <p>${escapeHTML(citation.snippet)}</p>
      </details>`).join("")
    : '<div class="empty-state mini">未召回可靠知识，已触发安全兜底。</div>';
  $("#resultPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  loadMetrics();
  loadRecent();
}

async function submitRequest(event) {
  event.preventDefault();
  const message = $("#messageInput").value.trim();
  if (!message) return;
  $("#loadingMask").classList.remove("is-hidden");
  $("#submitButton").disabled = true;
  try {
    const payload = await api("/api/assist", {
      method: "POST",
      body: JSON.stringify({
        message,
        customer_name: $("#customerInput").value,
        customer_tier: $("#tierInput").value,
        order_id: $("#orderInput").value,
      }),
    });
    renderResult(payload);
    toast("分流完成，工单已记录");
  } catch (error) {
    toast(error.message);
  } finally {
    $("#loadingMask").classList.add("is-hidden");
    $("#submitButton").disabled = false;
  }
}

async function sendFeedback(rating) {
  if (!state.currentTicketId || state.feedbackSent) return;
  try {
    await api(`/api/tickets/${encodeURIComponent(state.currentTicketId)}/feedback`, {
      method: "POST",
      body: JSON.stringify({ rating }),
    });
    state.feedbackSent = true;
    toast("反馈已记录，将用于后续优化");
    loadMetrics();
  } catch (error) {
    toast(error.message);
  }
}

async function loadTickets() {
  const status = $("#ticketStatusFilter").value;
  try {
    const data = await api(`/api/tickets?limit=100${status ? `&status=${encodeURIComponent(status)}` : ""}`);
    const items = data.items || [];
    if (!items.length) {
      $("#ticketTable").innerHTML = '<div class="empty-state">当前筛选条件下没有工单。</div>';
      return;
    }
    $("#ticketTable").innerHTML = items.map((ticket) => `
      <div class="table-row">
        <div class="ticket-cell">
          <strong>${escapeHTML(ticket.id)} · ${escapeHTML(ticket.customer_name)}</strong>
          <small title="${escapeHTML(ticket.message_masked)}">${escapeHTML(ticket.message_masked)}</small>
        </div>
        <span>${escapeHTML(ticket.category_label)}</span>
        <span class="priority-pill ${escapeHTML(ticket.priority)}">${escapeHTML(ticket.priority)}</span>
        <span class="status-cell">
          <span class="status-pill ${escapeHTML(ticket.status)}">${escapeHTML(statusLabels[ticket.status] || ticket.status)}</span>
          ${["pending_agent", "escalated"].includes(ticket.status)
            ? `<button class="resolve-action" data-resolve="${escapeHTML(ticket.id)}" type="button">标记解决</button>`
            : ""}
        </span>
        <span>${escapeHTML(ticket.assigned_team)}</span>
        <span>${formatTime(ticket.created_at)}</span>
      </div>`).join("");
  } catch (error) {
    toast(error.message);
  }
}

async function resolveTicket(ticketId, button) {
  button.disabled = true;
  try {
    await api(`/api/tickets/${encodeURIComponent(ticketId)}/status`, {
      method: "POST",
      body: JSON.stringify({ status: "resolved" }),
    });
    toast(`${ticketId} 已标记为解决`);
    await Promise.all([loadTickets(), loadMetrics(), loadRecent()]);
  } catch (error) {
    button.disabled = false;
    toast(error.message);
  }
}

async function loadKnowledge() {
  try {
    const data = await api("/api/knowledge");
    const items = data.items || [];
    $("#knowledgeGrid").innerHTML = items.map((doc) => `
      <article class="knowledge-card">
        <div class="knowledge-card-head">
          <div class="doc-type">KB</div>
          <span class="privacy-badge">${escapeHTML(categoryColors[doc.category] || doc.category)}</span>
        </div>
        <h3>${escapeHTML(doc.title)}</h3>
        <p>${escapeHTML(doc.id)}</p>
        <div class="knowledge-meta">
          <span>${escapeHTML(doc.team)}</span>
          <span>${doc.chunks} 个切片</span>
        </div>
      </article>`).join("");
  } catch (error) {
    toast(error.message);
  }
}

async function reloadKnowledge() {
  try {
    const result = await api("/api/knowledge/reload", { method: "POST", body: "{}" });
    toast(`知识索引已更新：${result.knowledge_chunks} 个切片`);
    loadKnowledge();
  } catch (error) {
    toast(error.message);
  }
}

function init() {
  $("#todayLabel").textContent = new Intl.DateTimeFormat("zh-CN", {
    year: "numeric", month: "long", day: "numeric", weekday: "long",
  }).format(new Date());
  $("#triageForm").addEventListener("submit", submitRequest);
  $("#messageInput").addEventListener("input", () => {
    $("#charCount").textContent = $("#messageInput").value.length;
  });
  $$(".sample-chip").forEach((button) => {
    button.addEventListener("click", () => {
      $("#messageInput").value = button.dataset.sample;
      $("#messageInput").dispatchEvent(new Event("input"));
      $("#messageInput").focus();
    });
  });
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => setView(item.dataset.view)));
  $$("[data-jump]").forEach((item) => item.addEventListener("click", () => setView(item.dataset.jump)));
  $("#refreshButton").addEventListener("click", () => {
    loadMetrics();
    loadRecent();
    toast("数据已刷新");
  });
  $("#ticketStatusFilter").addEventListener("change", loadTickets);
  $("#reloadKnowledge").addEventListener("click", reloadKnowledge);
  $$("[data-rating]").forEach((button) => {
    button.addEventListener("click", () => sendFeedback(button.dataset.rating));
  });
  document.addEventListener("click", (event) => {
    const resolveButton = event.target.closest("[data-resolve]");
    if (resolveButton) {
      event.stopPropagation();
      resolveTicket(resolveButton.dataset.resolve, resolveButton);
      return;
    }
    const ticketButton = event.target.closest("[data-ticket]");
    if (ticketButton) setView("tickets");
  });
  loadMetrics();
  loadRecent();
}

document.addEventListener("DOMContentLoaded", init);

