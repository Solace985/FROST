/* FROST frontend logic — plain JavaScript, no framework, no build step. */
(function () {
  "use strict";

  var fileInput = document.getElementById("file-input");
  var browseBtn = document.getElementById("browse-btn");
  var dropzone = document.getElementById("dropzone");
  var confirmBox = document.getElementById("confirm-box");
  var runBtn = document.getElementById("run-btn");
  var runStatus = document.getElementById("run-status");
  var previewWrap = document.getElementById("preview-wrap");
  var preview = document.getElementById("preview");
  var techChecks = document.getElementById("tech-checks");

  var resultsPanel = document.getElementById("results-panel");
  var pipelinePanel = document.getElementById("pipeline-panel");

  var selectedFile = null;
  var previewUrl = null;

  /* ---------- health ---------- */
  function refreshHealth() {
    var chip = document.getElementById("health-chip");
    var text = document.getElementById("health-text");
    fetch("/health")
      .then(function (r) { return r.json(); })
      .then(function (h) {
        chip.setAttribute("data-state", h.status === "ready" ? "ready" : "blocked");
        if (h.status === "ready") {
          text.textContent = "Ready · bundle " + h.bundle_version + " · parity " + h.parity_status;
        } else {
          text.textContent = "Not ready · parity " + h.parity_status +
            " · threshold " + h.threshold_status;
        }
      })
      .catch(function () {
        chip.setAttribute("data-state", "blocked");
        text.textContent = "Status unavailable";
      });
  }

  /* ---------- selection ---------- */
  function setFile(file) {
    if (!file) { return; }
    if (file.type !== "image/jpeg" && file.type !== "image/png") {
      runStatus.textContent = "Please choose a JPEG or PNG image.";
      return;
    }
    selectedFile = file;
    if (previewUrl) { URL.revokeObjectURL(previewUrl); }
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    previewWrap.hidden = false;
    techChecks.innerHTML = "";
    runStatus.textContent = "";
    updateRunEnabled();
  }

  function updateRunEnabled() {
    runBtn.disabled = !(selectedFile && confirmBox.checked);
  }

  browseBtn.addEventListener("click", function () { fileInput.click(); });
  fileInput.addEventListener("change", function (e) {
    if (e.target.files && e.target.files[0]) { setFile(e.target.files[0]); }
  });
  confirmBox.addEventListener("change", updateRunEnabled);

  ["dragenter", "dragover"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) {
      e.preventDefault(); dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) {
      e.preventDefault(); dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", function (e) {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  });
  dropzone.addEventListener("click", function () { fileInput.click(); });

  /* ---------- run ---------- */
  runBtn.addEventListener("click", function () {
    if (!selectedFile) { return; }
    runBtn.disabled = true;
    runStatus.textContent = "Running frozen inference…";

    var form = new FormData();
    form.append("image", selectedFile, selectedFile.name);

    fetch("/predict", { method: "POST", body: form })
      .then(function (r) {
        return r.json().then(function (body) { return { ok: r.ok, body: body }; });
      })
      .then(function (res) {
        if (!res.ok) {
          runStatus.textContent = "Could not process image: " +
            (res.body.error || res.body.category || "unknown error");
          renderTechChecksError(res.body.category);
          return;
        }
        runStatus.textContent = "Done in " + res.body.timings_ms.total.toFixed(1) + " ms.";
        renderResult(res.body);
      })
      .catch(function () {
        runStatus.textContent = "Network error contacting the local server.";
      })
      .finally(function () { updateRunEnabled(); });
  });

  /* ---------- rendering ---------- */
  function pct(x) { return (x * 100).toFixed(1) + "%"; }

  function renderTechChecksError(category) {
    techChecks.innerHTML = "";
    var d = document.createElement("div");
    d.innerHTML = "<dt>technical check</dt><dd>" + (category || "rejected") + "</dd>";
    techChecks.appendChild(d);
  }

  function renderTechChecks(tc, warnings) {
    techChecks.innerHTML = "";
    var rows = [
      ["dimensions", tc.width + " × " + tc.height + " px"],
      ["format", tc.format],
      ["size", (tc.byte_size / 1024).toFixed(0) + " KB"],
      ["mean intensity", tc.mean_intensity],
      ["contrast (std)", tc.contrast_std]
    ];
    rows.forEach(function (r) {
      var d = document.createElement("div");
      d.innerHTML = "<dt>" + r[0] + "</dt><dd>" + r[1] + "</dd>";
      techChecks.appendChild(d);
    });
    if (warnings && warnings.length) {
      var w = document.createElement("p");
      w.className = "tech-warn";
      w.textContent = "Technical notes: " + warnings.join(", ").replace(/_/g, " ");
      techChecks.appendChild(w);
    }
  }

  function renderResult(b) {
    renderTechChecks(b.technical_checks, b.warnings);

    // Decision badge
    var badge = document.getElementById("decision-badge");
    badge.setAttribute("data-decision", b.decision);
    document.getElementById("decision-label").textContent = b.decision;

    // Metrics
    document.getElementById("score-val").textContent = b.referable_dr_score.toFixed(4);
    document.getElementById("threshold-val").textContent = b.threshold.toFixed(4);
    var above = b.decision === "REFERABLE";
    document.getElementById("threshold-status-val").textContent =
      above ? "Above threshold" : "Below threshold";

    document.getElementById("op-note").textContent =
      "Referable score " + b.referable_dr_score.toFixed(4) +
      (above ? " ≥ " : " < ") + "threshold " + b.threshold.toFixed(4) + ".";

    var op = b.operating_point;
    document.getElementById("op-context").textContent =
      "At this fixed threshold, performance on held-out BRSET test data was " +
      "sensitivity " + pct(op.heldout_test_sensitivity) +
      " and specificity " + pct(op.heldout_test_specificity) + ".";

    var alertLine = document.getElementById("alert-line");
    if (above) {
      alertLine.setAttribute("data-kind", "above");
      alertLine.textContent =
        "Research alert: the referable-DR score is above the validation-selected " +
        "screening threshold. Prompt ophthalmic clinical review is indicated.";
    } else {
      alertLine.setAttribute("data-kind", "below");
      alertLine.textContent =
        "Below the validation-selected research alert threshold. " +
        "This is not a rule-out result and does not replace clinician assessment.";
    }
    resultsPanel.hidden = false;

    renderPipeline(b);
    pipelinePanel.hidden = false;
    pipelinePanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function stage(name, value, timeMs, frozen) {
    var li = document.createElement("li");
    if (frozen) { li.className = "frozen-stage"; }
    var html = '<div class="stage-name">' + name + "</div>";
    if (frozen) { html += '<div class="frozen-tag">frozen · no gradient</div>'; }
    html += '<div class="stage-val">' + value + "</div>";
    if (timeMs !== null && timeMs !== undefined) {
      html += '<div class="stage-time">' + timeMs.toFixed(1) + " ms</div>";
    }
    li.innerHTML = html;
    return li;
  }

  function renderPipeline(b) {
    var flow = document.getElementById("pipeline-flow");
    flow.innerHTML = "";
    var t = b.timings_ms;
    var tc = b.technical_checks;
    var trace = b.pipeline_trace;
    var probs = trace.dr_grade_class_probs.map(function (p) { return p.toFixed(3); });

    flow.appendChild(stage("Uploaded image", tc.width + "×" + tc.height + " px", t.decode, false));
    flow.appendChild(stage("Native-392 preprocessing", "tensor [1,3,392,392]", t.preprocessing, false));
    flow.appendChild(stage("RETFound-Green backbone", trace.backbone_params + " params", t.backbone, true));
    flow.appendChild(stage("Embedding", "[" + trace.embedding_dim + "]", null, false));
    flow.appendChild(stage("MultiTaskHead", "dr_grade logits [5]", t.head, false));
    flow.appendChild(stage("dr_grade class probs", "p0–p4 = " + probs.join(", "), t.postprocessing, false));
    flow.appendChild(stage("Referable mass", "p2+p3+p4 = " + b.referable_dr_score.toFixed(4), null, false));
    flow.appendChild(stage("Validation threshold", b.threshold.toFixed(4), null, false));
    flow.appendChild(stage("Decision", b.decision, null, false));
  }

  refreshHealth();
})();
