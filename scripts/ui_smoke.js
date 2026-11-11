const fs = require("node:fs/promises");

const debugOrigin = process.env.SERVICEFLOW_CDP_ORIGIN || "http://127.0.0.1:9222";
const appUrl = process.env.SERVICEFLOW_APP_URL || "http://127.0.0.1:8877";
const screenshotPath =
  process.env.SERVICEFLOW_SCREENSHOT ||
  "docs/workbench-stream-check.png";
const evaluationScreenshotPath =
  process.env.SERVICEFLOW_EVAL_SCREENSHOT ||
  "docs/workbench-evaluation-check.png";

async function main() {
  const pages = await (await fetch(`${debugOrigin}/json/list`)).json();
  const page = pages.find(
    (item) => item.type === "page" && item.url.startsWith(appUrl),
  );
  if (!page) throw new Error("ServiceFlow browser page was not found");

  const socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });

  let sequence = 0;
  const pending = new Map();
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
  });

  const command = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++sequence;
      pending.set(id, (message) => {
        if (message.error) reject(new Error(JSON.stringify(message.error)));
        else resolve(message.result);
      });
      socket.send(JSON.stringify({ id, method, params }));
    });

  await command("Runtime.enable");
  await command("Page.enable");
  await command("Page.navigate", { url: appUrl });
  const pageDeadline = Date.now() + 10000;
  while (Date.now() < pageDeadline) {
    const ready = await command("Runtime.evaluate", {
      expression: `document.readyState === "complete" && Boolean(document.querySelector("#chatForm"))`,
      returnByValue: true,
    });
    if (ready.result.value) break;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  await command("Runtime.evaluate", {
    expression: `
      document.querySelector("#messageInput").value = "支付宝退款通常多久到账？";
      document.querySelector("#chatForm").requestSubmit();
    `,
  });

  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const result = await command("Runtime.evaluate", {
      expression: `
        Boolean(
          document.querySelector(".message.assistant.is-complete") &&
          document.querySelector("#statModel").textContent !== "—"
        )
      `,
      returnByValue: true,
    });
    if (result.result.value) break;
    await new Promise((resolve) => setTimeout(resolve, 350));
  }

  const result = await command("Runtime.evaluate", {
    expression: `
      JSON.stringify({
        messages: [...document.querySelectorAll(".message-bubble")]
          .map((node) => node.textContent),
        trace: [...document.querySelectorAll(".trace-item strong")]
          .map((node) => node.textContent),
        sources: [...document.querySelectorAll(".source-card summary")]
          .map((node) => node.textContent),
        model: document.querySelector("#statModel").textContent,
        latency: document.querySelector("#statLatency").textContent,
        tokens: document.querySelector("#statTokens").textContent
      })
    `,
    returnByValue: true,
  });
  const summary = JSON.parse(result.result.value);

  const screenshot = await command("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
  });
  await fs.writeFile(screenshotPath, Buffer.from(screenshot.data, "base64"));

  await command("Runtime.evaluate", {
    expression: `document.querySelector('[data-tab="evaluation"]').click()`,
  });
  await new Promise((resolve) => setTimeout(resolve, 700));
  const evaluation = await command("Runtime.evaluate", {
    expression: `
      [...document.querySelectorAll(".eval-row")].map((row) => ({
        label: row.querySelector(".eval-row-head").textContent.trim(),
        detail: row.querySelector("small").textContent.trim()
      }))
    `,
    returnByValue: true,
  });
  const evaluationShot = await command("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
  });
  await fs.writeFile(
    evaluationScreenshotPath,
    Buffer.from(evaluationShot.data, "base64"),
  );
  socket.close();
  console.log(
    JSON.stringify(
      {
        ...summary,
        screenshotPath,
        evaluation: evaluation.result.value,
        evaluationScreenshotPath,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
