const state = {
  sessionId: null,
  busy: false,
  sourceCards: [],
  trace: [],
  toolCalls: [],
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const categoryLabels = {
  billing: "账单与支付",
  account: "账号与安全",
  logistics: "物流与履约",
  refund: "退款与售后",
  technical: "产品与技术",
  service: "投诉与服务",
  general: "通用咨询",
};

function escapeHTML(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function answerMarkup(value = "") {
  return escapeHTML(value).replace(
    /\[([a-zA-Z0-9_-]+#c\d+)\]/g,
    '<span class="citation-token">$1</span>',
  );
}

async function api(path, options = {}) {
  const isForm = options.body instanceof FormData;
  const response = await fetch(path, {
    ...options,
    headers: isForm
      ? options.headers || {}
      : { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || payload.detail || `请求失败 (${response.status})`);
  }
  return payload;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("is-visible");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("is-visible"), 2800);
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    const scroller = $("#conversationScroll");
    scroller.scrollTop = scroller.scrollHeight;
  });
}

function setEmptyVisible(visible) {
  $("#emptyState").classList.toggle("is-hidden", !visible);
}

function resetInspector() {
  state.sourceCards = [];
  state.trace = [];
  state.toolCalls = [];
  $("#statModel").textContent = "—";
  $("#statLatency").textContent = "—";
  $("#statTokens").textContent = "—";
  $("#panel-trace").innerHTML = `
    <div class="panel-empty">
      <span class="panel-empty-mark">↳</span>
      <p>提出问题后，这里会显示 LangGraph 的节点与工具调用。</p>
    </div>`;
  $("#panel-sources").innerHTML = `
    <div class="panel-empty">
      <span class="panel-empty-mark">⌕</span>
      <p>检索完成后，这里会显示答案引用的原文段落。</p>
    </div>`;
}

function startNewChat() {
  if (state.busy) return;
  state.sessionId = null;
  $("#messageList").innerHTML = "";
  setEmptyVisible(true);
  resetInspector();
  setPanel("trace");
  $("#messageInput").value = "";
  autoResizeInput();
  $$(".session-item").forEach((item) => item.classList.remove("is-active"));
  $("#messageInput").focus();
}

function appendMessage(role, content = "", loading = false) {
  setEmptyVisible(false);
  const article = document.createElement("article");
  article.className = `message ${role}${loading ? " is-loading" : ""}`;
  article.innerHTML = `
    <div class="message-avatar">${role === "assistant" ? "SF" : "你"}</div>
    <div>
      <p class="message-meta">${role === "assistant" ? "ServiceFlow Agent" : "你"}</p>
      <div class="message-bubble"></div>
    </div>`;
  const bubble = $(".message-bubble", article);
  if (loading) {
    bubble.innerHTML = '<span class="typing-dots"><i></i><i></i><i></i></span>';
  } else {
    bubble.textContent = content;
  }
  $("#messageList").append(article);
  scrollToBottom();
  return { article, bubble };
}

function renderSession(session) {
  $("#messageList").innerHTML = "";
  if (!session.messages?.length) {
    setEmptyVisible(true);
    return;
  }
  setEmptyVisible(false);
  session.messages.forEach((message) => {
    const role = message.role === "assistant" ? "assistant" : "user";
    const { article, bubble } = appendMessage(role, message.content);
    article.classList.add("is-complete");
    if (role === "assistant") bubble.innerHTML = answerMarkup(message.content);
    if (role === "assistant" && message.metadata?.sources?.length) {
      state.sourceCards = message.metadata.sources.map((source) => ({
        title: source.split(/[\\/]/).pop(),
        source,
        snippet: "会话历史来源记录",
        relevance: null,
      }));
      renderSources();
    }
  });
  scrollToBottom();
}

async function loadSessions() {
  try {
    const data = await api("/api/sessions");
    const items = data.items || [];
    if (!items.length) {
      $("#sessionList").innerHTML = '<p class="session-empty">还没有对话记录</p>';
      return;
    }
    $("#sessionList").innerHTML = items.map((item) => `
      <button class="session-item${item.id === state.sessionId ? " is-active" : ""}"
        data-session="${escapeHTML(item.id)}" type="button">
        <strong>${escapeHTML(item.title || "未命名会话")}</strong>
        <small>${formatDate(item.updated_at)} · ${item.message_count || 0} 条消息</small>
      </button>`).join("");
    $$("[data-session]").forEach((button) => {
      button.addEventListener("click", () => openSession(button.dataset.session));
    });
  } catch (error) {
    $("#sessionList").innerHTML = '<p class="session-empty">会话读取失败</p>';
  }
}

async function openSession(sessionId) {
  if (state.busy) return;
  try {
    const session = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
    state.sessionId = sessionId;
    resetInspector();
    renderSession(session);
    $$(".session-item").forEach((item) => {
      item.classList.toggle("is-active", item.dataset.session === sessionId);
    });
    if (window.innerWidth <= 850) $("#inspector").classList.remove("is-open");
  } catch (error) {
    toast(error.message);
  }
}

function renderTrace() {
  if (!state.trace.length && !state.toolCalls.length) {
    $("#panel-trace").innerHTML = `
      <div class="panel-empty">
        <span class="panel-empty-mark">↳</span>
        <p>Agent 正在准备执行。</p>
      </div>`;
    return;
  }
  const traceHTML = state.trace.map((item) => `
    <li class="trace-item">
      <strong>${escapeHTML(item.node)}</strong>
      <span>${escapeHTML(item.detail || "")}</span>
    </li>`).join("");
  const toolsHTML = state.toolCalls.map((tool) => `
    <div class="tool-summary">
      <p><strong>${escapeHTML(tool.name)}</strong> · ${escapeHTML(tool.status)}
      · ${Number(tool.duration_ms || 0).toFixed(2)} ms</p>
    </div>`).join("");
  $("#panel-trace").innerHTML = `
    <ul class="trace-list">${traceHTML}</ul>
    ${toolsHTML}`;
}

function renderSources() {
  if (!state.sourceCards.length) {
    $("#panel-sources").innerHTML = `
      <div class="panel-empty">
        <span class="panel-empty-mark">⌕</span>
        <p>本次没有找到可靠知识来源。</p>
      </div>`;
    return;
  }
  $("#panel-sources").innerHTML = state.sourceCards.map((source) => `
    <details class="source-card" open>
      <summary>
        ${escapeHTML(source.title || source.source || "知识来源")}
        ${source.id ? `<small>${escapeHTML(source.id)}</small>` : ""}
      </summary>
      <p>${escapeHTML(source.snippet || source.source || "")}</p>
    </details>`).join("");
}

function updateInspector(finalPayload) {
  const retrieval = finalPayload.retrieval || {};
  const agent = retrieval.agent || {};
  state.trace = agent.trace || [];
  state.toolCalls = agent.tool_calls || [];
  state.sourceCards = finalPayload.ticket?.citations || [];
  renderTrace();
  renderSources();

  const model = retrieval.model || {};
  $("#statModel").textContent = model.model || finalPayload.ticket?.generation_mode || "本地模式";
  $("#statLatency").textContent = model.latency_ms
    ? `${Math.round(model.latency_ms)} ms`
    : "—";
  $("#statTokens").textContent = model.usage?.total_tokens ?? "—";
  setPanel("sources");
  loadTickets();
}

function parseSSEBlock(block) {
  const lines = block.split(/\r?\n/);
  let event = "message";
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return { event, data: { raw: dataLines.join("\n") } };
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const input = $("#messageInput");
  const message = input.value.trim();
  if (!message || state.busy) return;

  state.busy = true;
  $("#sendButton").disabled = true;
  appendMessage("user", message);
  const assistant = appendMessage("assistant", "", true);
  let answer = "";
  state.trace = [];
  state.toolCalls = [];
  renderTrace();
  input.value = "";
  autoResizeInput();

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: state.sessionId,
        customer_name: "工作台用户",
        customer_tier: "standard",
        order_id: "",
      }),
    });
    if (!response.ok || !response.body) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.error || errorPayload.detail || "流式请求失败");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let completed = false;

    while (!completed) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        const parsed = parseSSEBlock(block);
        if (!parsed) continue;
        const { event: eventName, data } = parsed;
        if (eventName === "session") {
          state.sessionId = data.session_id;
        }
        if (eventName === "node") {
          state.trace.push(data);
          renderTrace();
        }
        if (eventName === "answer_delta") {
          assistant.article.classList.remove("is-loading");
          answer += data.delta || "";
          assistant.bubble.textContent = answer;
          scrollToBottom();
        }
        if (eventName === "final") {
          completed = true;
          assistant.article.classList.add("is-complete");
          assistant.article.classList.remove("is-loading");
          assistant.bubble.innerHTML = answerMarkup(data.ticket?.answer || answer);
          updateInspector(data);
          loadSessions();
        }
        if (eventName === "error") {
          throw new Error(data.error || "Agent 执行失败");
        }
      }
    }
    if (!completed) {
      assistant.article.classList.add("is-complete");
      assistant.article.classList.remove("is-loading");
      assistant.bubble.textContent = answer || "响应意外中断，请重新尝试。";
    }
  } catch (error) {
    assistant.article.classList.add("is-complete");
    assistant.article.classList.remove("is-loading");
    assistant.bubble.textContent = `请求失败：${error.message}`;
    toast(error.message);
  } finally {
    state.busy = false;
    $("#sendButton").disabled = false;
    input.focus();
    scrollToBottom();
  }
}

async function loadKnowledge() {
  try {
    const [docs, staticDocs] = await Promise.all([
      api("/api/documents"),
      api("/api/knowledge"),
    ]);
    const uploaded = docs.items || [];
    const knowledge = staticDocs.items || [];
    $("#knowledgeCount").textContent = String(knowledge.length);
    $("#knowledgeList").innerHTML = knowledge.map((doc) => `
      <div class="side-item">
        <strong>${escapeHTML(doc.title)}</strong>
        <span>${escapeHTML(doc.team)} · ${doc.chunks} 个切片</span>
        <span class="badge">${escapeHTML(categoryLabels[doc.category] || doc.category)}</span>
      </div>`).join("") || '<div class="panel-empty"><p>还没有知识文档。</p></div>';
    if (!uploaded.length) return;
    $("#knowledgeCount").textContent = String(knowledge.length);
  } catch (error) {
    $("#knowledgeList").innerHTML = `<div class="panel-empty"><p>${escapeHTML(error.message)}</p></div>`;
  }
}

async function loadTickets() {
  try {
    const [tickets, metrics] = await Promise.all([
      api("/api/tickets?limit=8"),
      api("/api/metrics"),
    ]);
    $("#ticketCount").textContent = String(metrics.total || 0);
    const items = tickets.items || [];
    $("#ticketList").innerHTML = items.length
      ? items.map((ticket) => `
        <div class="side-item">
          <strong>${escapeHTML(ticket.id)}</strong>
          <span>${escapeHTML(ticket.category_label)} · ${escapeHTML(ticket.priority)} · ${escapeHTML(ticket.status)}</span>
          <span>${escapeHTML(ticket.message_masked).slice(0, 48)}…</span>
        </div>`).join("")
      : '<div class="panel-empty"><p>还没有工单记录。</p></div>';
  } catch (error) {
    $("#ticketList").innerHTML = `<div class="panel-empty"><p>${escapeHTML(error.message)}</p></div>`;
  }
}

async function loadEvaluation() {
  try {
    const data = await api("/api/evaluation");
    $("#evaluationList").innerHTML = `
      <div class="eval-summary">
        <div><span>分类准确率</span><strong>${(data.category_accuracy * 100).toFixed(1)}%</strong></div>
        <div><span>优先级准确率</span><strong>${(data.priority_accuracy * 100).toFixed(1)}%</strong></div>
      </div>
      ${data.retrieval_modes.map((mode) => `
        <div class="eval-row">
          <div class="eval-row-head">
            <span>${escapeHTML(mode.name)}</span>
            <span>Hit@1 ${(mode.hit_at_1 * 100).toFixed(1)}%</span>
          </div>
          <div class="eval-bar"><i style="width:${mode.hit_at_1 * 100}%"></i></div>
          <small>Hit@3 ${(mode.hit_at_3 * 100).toFixed(1)}% · MRR ${mode.mrr_at_5.toFixed(3)} · ${mode.average_latency_ms.toFixed(2)} ms</small>
        </div>`).join("")}`;
  } catch (error) {
    $("#evaluationList").innerHTML = `<div class="panel-empty"><p>${escapeHTML(error.message)}</p></div>`;
  }
}

async function loadStatus() {
  try {
    const [health, model] = await Promise.all([
      api("/api/health"),
      api("/api/model/info"),
    ]);
    $("#statusDot").classList.add("is-online");
    $("#statusTitle").textContent = `${health.knowledge_chunks} 个知识切片已就绪`;
    $("#statusSubtitle").textContent = `${health.tool_count} 个 Agent 工具 · LangGraph`;
    $("#modelPill").textContent = model.configured
      ? `${model.model} · 在线`
      : "离线回退模式";
  } catch (error) {
    $("#statusDot").classList.add("is-error");
    $("#statusTitle").textContent = "服务连接失败";
    $("#statusSubtitle").textContent = error.message;
  }
}

function setPanel(name) {
  $$(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === name));
  $$(".panel").forEach((panel) => panel.classList.toggle("is-active", panel.id === `panel-${name}`));
}

function autoResizeInput() {
  const input = $("#messageInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
}

function openUpload() {
  $("#uploadModal").classList.add("is-open");
  $("#uploadModal").setAttribute("aria-hidden", "false");
  setTimeout(() => $("#documentFile").focus(), 50);
}

function closeUpload() {
  $("#uploadModal").classList.remove("is-open");
  $("#uploadModal").setAttribute("aria-hidden", "true");
}

async function submitUpload(event) {
  event.preventDefault();
  const file = $("#documentFile").files[0];
  if (!file) {
    toast("请先选择文档");
    return;
  }
  const button = $("#uploadSubmit");
  button.disabled = true;
  button.textContent = "正在解析并建立索引…";
  const form = new FormData();
  form.append("file", file);
  form.append("title", $("#documentTitle").value);
  form.append("category", $("#documentCategory").value);
  form.append("team", $("#documentTeam").value);
  try {
    const result = await api("/api/documents", { method: "POST", body: form });
    toast(`已加入知识库：${result.document.chunk_count} 个切片`);
    closeUpload();
    $("#uploadForm").reset();
    $("#fileName").textContent = "尚未选择文件";
    await Promise.all([loadKnowledge(), loadStatus()]);
    setPanel("knowledge");
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "解析并加入知识库";
  }
}

function init() {
  $("#chatForm").addEventListener("submit", sendMessage);
  $("#messageInput").addEventListener("input", autoResizeInput);
  $("#messageInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#chatForm").requestSubmit();
    }
  });
  $("#newChatButton").addEventListener("click", startNewChat);
  $("#refreshSessions").addEventListener("click", loadSessions);
  $("#openUpload").addEventListener("click", openUpload);
  $("#panelUpload").addEventListener("click", openUpload);
  $("#closeUpload").addEventListener("click", closeUpload);
  $("#uploadModal").addEventListener("click", (event) => {
    if (event.target === $("#uploadModal")) closeUpload();
  });
  $("#uploadForm").addEventListener("submit", submitUpload);
  $("#documentFile").addEventListener("change", () => {
    const file = $("#documentFile").files[0];
    $("#fileName").textContent = file ? file.name : "尚未选择文件";
    if (file && !$("#documentTitle").value) {
      $("#documentTitle").value = file.name.replace(/\.[^.]+$/, "");
    }
  });
  $$(".prompt-card").forEach((card) => {
    card.addEventListener("click", () => {
      $("#messageInput").value = card.dataset.prompt;
      autoResizeInput();
      $("#chatForm").requestSubmit();
    });
  });
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      setPanel(tab.dataset.tab);
      if (tab.dataset.tab === "knowledge") loadKnowledge();
      if (tab.dataset.tab === "tickets") loadTickets();
      if (tab.dataset.tab === "evaluation") loadEvaluation();
    });
  });
  $$("[data-panel]").forEach((button) => {
    button.addEventListener("click", () => {
      setPanel(button.dataset.panel);
      if (button.dataset.panel === "knowledge") loadKnowledge();
      if (button.dataset.panel === "tickets") loadTickets();
      $("#inspector").classList.add("is-open");
    });
  });
  $("#openInspectorMobile").addEventListener("click", () => $("#inspector").classList.add("is-open"));
  $("#closeInspectorMobile").addEventListener("click", () => $("#inspector").classList.remove("is-open"));
  $("#refreshTickets").addEventListener("click", loadTickets);
  $("#refreshEvaluation").addEventListener("click", loadEvaluation);

  loadStatus();
  loadSessions();
  loadKnowledge();
  loadTickets();
  autoResizeInput();
  $("#messageInput").focus();
}

document.addEventListener("DOMContentLoaded", init);
