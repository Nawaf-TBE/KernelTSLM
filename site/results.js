// Kernel X-Ray results as dot plots. Fills every element with data-plot="<name>"; add data-bare to hide the
// title and caption (when the page already says them). Used by index.html.
// Numbers: runs/mt_all/metrics.json (test split), runs/baselines/metrics.json, runs/usecases/<task>/baselines.json.
(function () {
  var P = {
    diagnose: {title: "Diagnose a slow kernel", sub: "right limiter, 291 test kernels", rows: [
        {n: "Always guess the most common", v: .402}, {n: "Guess from work type", v: .570},
        {n: "Kernel X-Ray", v: .787, ours: 1}, {n: "Logistic regression on averages", v: .938}],
      cap: "Before fine-tuning, the base model gave no valid answer."},
    timeline: {title: "Lay out what ran when", sub: "time bins labelled right, 291 kernels", rows: [
        {n: "Always guess the most common", v: .625}, {n: "Guess from counter averages", v: .883},
        {n: "Kernel X-Ray", v: .897, ours: 1}, {n: "Logistic regression on sliding windows", v: .943}],
      cap: "It tells busy from idle in 91.9% of bins."},
    timing: {title: "Answer timing questions", sub: "right answers, 1,000 questions", rows: [
        {n: "Base model, before fine-tuning", v: .25}, {n: "Always guess the most common", v: .325},
        {n: "Boosted trees, ignoring time order", v: .753}, {n: "Kernel X-Ray", v: .803, ours: 1}, {n: "Boosted trees, using time order", v: .883}],
      cap: "Chance is 25 to 50% depending on the question."},
    workload: {title: "Name the kind of work", sub: "balanced accuracy, 11 types", rows: [
        {n: "Always guess the most common", v: .091}, {n: "Logistic regression on averages", v: .751},
        {n: "Kernel X-Ray", v: .747, ours: 1}, {n: "MiniRocket time-series classifier", v: .811}],
      cap: "Plain accuracy is 88.7%; balanced accuracy counts rare types as much as common ones."},
    order: {title: "Real traces against broken ones", sub: "accuracy", rows: [
        {n: "Diagnosis", pair: [.292, .787], pl: ["time-shuffled", "real"]},
        {n: "Timing questions", pair: [.465, .803], pl: ["time-shuffled", "real"]},
        {n: "Kind of work", pair: [.378, .887], pl: ["other kernel's trace", "own"]},
        {n: "Trend, asked forwards and backwards", v: .709, ours: 1}],
      cap: "Shuffling the time order, or swapping in another kernel's counters, cuts accuracy sharply. On the last row, each trend question was asked on the trace and on the trace reversed; a model that ignores time order would get both right 0% of the time."}
  };
  function build(s, bare) {
    var lo = 0, hi = 1, ticks = [0, .25, .5, .75, 1];
    var X = function (v) { return ((v - lo) / (hi - lo) * 100).toFixed(2) + "%"; };
    var fmt = function (v) { return v.toFixed(3); };
    var grid = '<span class="gridbg">' + ticks.map(function (tk) { return '<i style="left:' + X(tk) + '"></i>'; }).join("") + "</span>";
    var body = s.rows.map(function (r) {
      var inner;
      if (r.pair) inner = '<span class="bar" style="left:' + X(r.pair[0]) + ";width:calc(" + X(r.pair[1]) + " - " + X(r.pair[0]) + ')"></span>' +
          '<span class="dot open" style="left:' + X(r.pair[0]) + '"></span><span class="val under" style="left:' + X(r.pair[0]) + '">' + r.pl[0] + " " + fmt(r.pair[0]) + "</span>" +
          '<span class="dot hl" style="left:' + X(r.pair[1]) + '"></span><span class="val under" style="left:' + X(r.pair[1]) + '">' + r.pl[1] + " " + fmt(r.pair[1]) + "</span>";
      else inner = '<span class="dot' + (r.ours ? " hl" : "") + '" style="left:' + X(r.v) + '"></span><span class="val' + (r.v > 0.8 ? " left" : "") + '" style="left:' + X(r.v) + '">' + fmt(r.v) + "</span>";
      return '<div class="prow' + (r.ours || r.pair ? " ours" : "") + (r.pair ? " pair" : "") + '"><span>' + r.n + '</span><span class="track">' + grid + inner + "</span></div>";
    }).join("");
    var axis = '<div class="axis"><span></span><span class="ticks">' + ticks.map(function (tk) { return '<span style="left:' + X(tk) + '">' + String(tk) + "</span>"; }).join("") + "</span></div>";
    return '<figure class="plot">' + (bare ? "" : "<figcaption>" + s.title + " <span>" + s.sub + "</span></figcaption>") + body + axis +
      (bare ? "" : '<p class="cap">' + s.cap + "</p>") + "</figure>";
  }
  var T = [
    ["Diagnose a slow kernel", "right limiter", .402, .787, .938, "logistic regression on counter averages"],
    ["Lay out what ran when", "time bins right", .625, .897, .943, "logistic regression on sliding windows"],
    ["Answer timing questions", "right answers", .325, .803, .883, "boosted trees using time order"],
    ["Name the kind of work", "balanced accuracy", .091, .747, .811, "MiniRocket time-series classifier"]];
  document.querySelectorAll("[data-table=compare]").forEach(function (el) {
    el.innerHTML = '<table class="cmp"><thead><tr><th scope="col">Use case</th><th scope="col">Always guess the most common</th><th scope="col" class="us">Kernel X-Ray</th><th scope="col">Best model built for this task alone</th></tr></thead><tbody>' +
      T.map(function (r) { return '<tr><th scope="row">' + r[0] + "<span>" + r[1] + '</span></th><td class="n">' + r[2].toFixed(3) + '</td><td class="n us"><b>' + r[3].toFixed(3) + '</b></td><td class="n">' + r[4].toFixed(3) + "<span>" + r[5] + "</span></td></tr>"; }).join("") +
      "</tbody></table>";
  });
  document.querySelectorAll("[data-plot]").forEach(function (el) { var s = P[el.getAttribute("data-plot")]; if (s) el.innerHTML = build(s, el.hasAttribute("data-bare")); });
})();
