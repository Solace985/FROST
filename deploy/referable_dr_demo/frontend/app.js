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

  function refreshHealth() {
    var chip = document.getElementById("health-chip");
    var text = document.getElementById("health-text");
    if (!chip || !text) { return; }
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

    var badge = document.getElementById("decision-badge");
    badge.setAttribute("data-decision", b.decision);
    document.getElementById("decision-label").textContent = b.decision;
    document.getElementById("decision-icon").textContent =
      b.decision === "REFERABLE" ? "▲" : "✓";

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
      alertLine.innerHTML =
        '<p class="alert-lead">The referable-DR score is above the ' +
        "validation-selected screening threshold.</p>" +
        '<div class="clinical-suggestion">' +
        '<div class="cs-head">Clinical suggestion</div>' +
        '<div class="cs-body">Prompt ophthalmic clinical review is suggested.</div>' +
        "</div>";
    } else {
      alertLine.setAttribute("data-kind", "below");
      alertLine.innerHTML =
        '<p class="alert-lead">Below the validation-selected research alert ' +
        "threshold. This is not a rule-out result and does not replace " +
        "clinician assessment.</p>";
    }

    var sqrtPos = function (x) {
      return (Math.sqrt(Math.max(0, Math.min(1, x))) * 100).toFixed(1);
    };
    var gauge = document.getElementById("gauge");
    gauge.setAttribute("data-decision", b.decision);
    var thrPos = sqrtPos(b.threshold);
    document.getElementById("gauge-fill").style.width = sqrtPos(b.referable_dr_score) + "%";
    document.getElementById("gauge-threshold").style.left = thrPos + "%";
    document.getElementById("gauge-thr-label").style.left = thrPos + "%";

    document.getElementById("await-msg").hidden = true;
    document.getElementById("result-content").hidden = false;
    resultsPanel.hidden = false;

    renderPipeline(b);
    pipelinePanel.hidden = false;
  }

  function stage(name, value, timeMs, frozen) {
    var li = document.createElement("li");
    if (frozen) { li.className = "frozen-stage"; }
    var html = '<div class="stage-name">' + name + "</div>";
    html += '<div class="stage-val">' + value + "</div>";
    if (timeMs !== null && timeMs !== undefined) {
      html += '<div class="stage-time">' + timeMs.toFixed(1) + " ms</div>";
    }
    li.innerHTML = html;
    return li;
  }

  var GRADE_COLORS = ["#2e7d5b", "#9bb04a", "#d9a441", "#cf7a33", "#b23a3a"];

  function drGradePie(rawProbs) {
    var cx = 60, cy = 60, r = 54;
    var total = rawProbs.reduce(function (a, b) { return a + b; }, 0) || 1;
    var start = -Math.PI / 2;
    var slices = "";
    rawProbs.forEach(function (p, i) {
      var frac = p / total;
      if (frac <= 0) { return; }
      if (frac >= 0.999) {
        slices += '<circle cx="' + cx + '" cy="' + cy + '" r="' + r +
          '" fill="' + GRADE_COLORS[i] + '"/>';
        return;
      }
      var end = start + frac * 2 * Math.PI;
      var x0 = (cx + r * Math.cos(start)).toFixed(2);
      var y0 = (cy + r * Math.sin(start)).toFixed(2);
      var x1 = (cx + r * Math.cos(end)).toFixed(2);
      var y1 = (cy + r * Math.sin(end)).toFixed(2);
      var large = (end - start) > Math.PI ? 1 : 0;
      slices += '<path d="M' + cx + ' ' + cy + ' L' + x0 + ' ' + y0 +
        ' A' + r + ' ' + r + ' 0 ' + large + ' 1 ' + x1 + ' ' + y1 +
        ' Z" fill="' + GRADE_COLORS[i] + '" stroke="#ffffff" stroke-width="1"/>';
      start = end;
    });
    var svg = '<svg class="pie-svg" viewBox="0 0 120 120" role="img" ' +
      'aria-label="DR grade class-probability distribution, grade 0 (low ' +
      'severity) through grade 4 (high severity)">' + slices + '</svg>';
    var legend = '<ul class="pie-legend">';
    for (var i = 0; i < GRADE_COLORS.length; i++) {
      legend += '<li><span class="pie-key" style="background:' +
        GRADE_COLORS[i] + '"></span>grade ' + i + '</li>';
    }
    legend += '</ul>';
    return '<div class="pie-wrap">' + svg + legend + '</div>';
  }

  function renderPipeline(b) {
    var row1 = document.getElementById("pipeline-row-1");
    var row2 = document.getElementById("pipeline-row-2");
    row1.innerHTML = "";
    row2.innerHTML = "";
    var t = b.timings_ms;
    var tc = b.technical_checks;
    var trace = b.pipeline_trace;

    row1.appendChild(stage("Uploaded image", tc.width + "×" + tc.height + " px", t.decode, false));
    row1.appendChild(stage("Native-392 Image Preprocessing", "tensor [1,3,392,392]", t.preprocessing, false));
    row1.appendChild(stage("RETFound-Green backbone", trace.backbone_params + " params", t.backbone, true));
    row1.appendChild(stage("Embedding Dimension", "[" + trace.embedding_dim + "]", null, false));
    row1.appendChild(stage("MultiTaskHead", "dr_grade logits [5]", t.head, false));

    row2.appendChild(stage("Diabetic Retinopathy Grade class probabilities",
      drGradePie(trace.dr_grade_class_probs), null, false));
    row2.appendChild(stage("Referable mass",
      "grade 2 + grade 3 + grade 4 = " + b.referable_dr_score.toFixed(4), null, false));
    row2.appendChild(stage("Validation threshold", b.threshold.toFixed(4), null, false));
    row2.appendChild(stage("Decision", b.decision, null, false));
  }

  Array.prototype.forEach.call(document.querySelectorAll(".nav-btn"), function (a) {
    a.addEventListener("click", function (e) {
      var href = a.getAttribute("href") || "";
      if (href.charAt(0) !== "#") { return; }
      var target = document.querySelector(href);
      if (!target) { return; }
      e.preventDefault();
      if (target.tagName === "DETAILS") { target.open = true; }
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  refreshHealth();
})();
