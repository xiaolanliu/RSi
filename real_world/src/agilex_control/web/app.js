const $ = (id) => document.getElementById(id);

function commandText(decision) {
  if (!decision) return "";
  const name = decision.decision;
  if (name === "dry_run") return JSON.stringify(decision.dry_run || {}, null, 2);
  if (name === "phase") return JSON.stringify(decision.phase || {}, null, 2);
  if (name === "geometry") return JSON.stringify(decision.geometry || {}, null, 2);
  if (name === "get_state") return JSON.stringify({ seconds: decision.get_state_seconds || 1 }, null, 2);
  return "";
}

function bust(url) {
  return url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
}

function setCaption(caption) {
  if (!caption) return;
  $("caption-status").textContent = caption.status || "";
  $("caption-decision").textContent = caption.decision || "idle";
  $("caption-reason").textContent = caption.reason || "";
  $("caption-command").textContent = caption.command || "";
}

function renderTimeline(steps) {
  const root = $("timeline");
  root.innerHTML = "";
  (steps || []).forEach((item) => {
    const decision = item.decision || {};
    const outcome = item.outcome || {};
    const li = document.createElement("li");
    const idx = document.createElement("div");
    idx.className = "idx";
    idx.textContent = "#" + String(item.index).padStart(2, "0");
    const body = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = decision.decision || "";
    const why = document.createElement("p");
    why.className = "why";
    why.textContent = decision.reason || "";
    body.appendChild(title);
    body.appendChild(why);
    const cmd = commandText(decision);
    if (cmd) {
      const cmdNode = document.createElement("div");
      cmdNode.className = "cmd";
      cmdNode.textContent = cmd;
      body.appendChild(cmdNode);
    }
    const ok = document.createElement("div");
    ok.className = "ok";
    ok.textContent = outcome.ok === true ? "ok" : outcome.ok === false ? "fail" : outcome.kind || "live";
    if (outcome.error) {
      const err = document.createElement("div");
      err.className = "err";
      err.textContent = outcome.error;
      body.appendChild(err);
    }
    li.appendChild(idx);
    li.appendChild(body);
    li.appendChild(ok);
    root.appendChild(li);
  });
}

function applyState(state) {
  const status = state.status || "idle";
  $("status-pill").textContent = status;
  $("status-pill").className = "status-pill " + (
    status === "running" || status === "reasoning" || status === "command" || status === "starting" ? "live"
      : status === "failed" ? "failed" : ""
  );
  setCaption(state.caption);
  renderTimeline((state.journal && state.journal.steps) || []);
  if (state.preview) $("front").src = bust("/media/process-video/front.jpg");
  (state.images || []).forEach((item) => {
    const node = $(item.camera);
    if (node && item.url) node.src = bust(item.url);
  });
  if (!state.preview) {
    const front = (state.images || []).find((item) => item.camera === "front");
    if (front) $("front").src = bust(front.url);
  }
  $("output-path").textContent = (state.journal && state.journal.output) || "";
  $("start").disabled = status === "running" || status === "starting" || status === "reasoning" || status === "command";
}

async function forceStop() {
  const res = await fetch("/api/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  const body = await res.json();
  if (!res.ok) alert(body.error || "停止失败");
  refresh();
}

async function refresh() {
  const res = await fetch("/api/state", { cache: "no-store" });
  applyState(await res.json());
}

$("start").addEventListener("click", async () => {
  $("start").disabled = true;
  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      task: $("task").value,
      save_video: $("save-video").checked,
      execute: $("execute").checked,
      max_steps: Number($("max-steps").value || 16),
    }),
  });
  const body = await res.json();
  if (!res.ok) {
    alert(body.error || "无法启动");
    $("start").disabled = false;
    return;
  }
  $("output-path").textContent = body.output || "";
});

$("stop").addEventListener("click", forceStop);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") forceStop();
});

const stream = new EventSource("/api/events");
stream.onmessage = (event) => {
  try {
    const payload = JSON.parse(event.data);
    if (payload.kind === "decision" || payload.kind === "thinking") setCaption(payload.caption);
    refresh();
  } catch (err) {
    console.warn(err);
  }
};
refresh();
setInterval(() => {
  const img = $("front");
  if (img && img.src.includes("/media/process-video/front.jpg")) {
    img.src = bust("/media/process-video/front.jpg");
  }
}, 400);
