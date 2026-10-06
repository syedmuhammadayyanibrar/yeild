import { THEME, PLOTLY_BASE_LAYOUT } from "./theme.js";

export function renderForestPlot(containerId, signals, titleText) {
  const yLabels = signals.map((s) => s.signal.replace(/_/g, " "));
  const xMedians = signals.map((s) => s.median_lift);
  const xErrorsPlus = signals.map((s) => s.hi90 - s.median_lift);
  const xErrorsMinus = signals.map((s) => s.median_lift - s.lo90);

  const traceLoop = {
    type: "scatter",
    mode: "markers",
    y: yLabels,
    x: xMedians,
    error_x: {
      type: "data",
      symmetric: false,
      array: xErrorsPlus,
      arrayminus: xErrorsMinus,
      color: THEME.teal,
      thickness: 2.5,
      width: 6,
    },
    marker: {
      color: THEME.teal,
      size: 9,
      symbol: "circle",
    },
    name: "Loop Estimate",
  };

  const xPooled = signals.map((s) => s.industry_pooled_lift);
  const tracePooled = {
    type: "scatter",
    mode: "markers",
    y: yLabels,
    x: xPooled,
    marker: {
      color: THEME.sandDeep,
      size: 7,
      symbol: "diamond",
    },
    name: "Industry Pooled",
  };

  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    title: { text: titleText, font: { size: 14, color: THEME.ink } },
    xaxis: {
      ...PLOTLY_BASE_LAYOUT.xaxis,
      title: "Lift at Age 0 (percentage points per 100 sends)",
      zeroline: true,
      zerolinecolor: THEME.ink,
      zerolinewidth: 1,
    },
    yaxis: {
      ...PLOTLY_BASE_LAYOUT.yaxis,
      autorange: "reversed",
    },
    showlegend: true,
  };

  Plotly.newPlot(containerId, [traceLoop, tracePooled], layout, { displayModeBar: false, responsive: true });
}

export function renderDecayCurves(containerId, signals, titleText) {
  const top4 = signals.slice(0, 4);
  const traces = [];

  for (const s of top4) {
    const curve = s.decay_curve;
    if (!curve || curve.length === 0) continue;

    const x = curve.map((pt) => pt.age);
    const yMed = curve.map((pt) => pt.median);
    const yLo = curve.map((pt) => pt.lo90);
    const yHi = curve.map((pt) => pt.hi90);

    const sigLabel = `${s.signal.replace(/_/g, " ")} (t1/2=${s.half_life_median}d)`;

    traces.push({
      x: x.concat(x.slice().reverse()),
      y: yHi.concat(yLo.slice().reverse()),
      fill: "toself",
      fillcolor: THEME.bandTeal,
      line: { color: "transparent" },
      showlegend: false,
      hoverinfo: "skip",
    });

    traces.push({
      x: x,
      y: yMed,
      mode: "lines",
      line: { color: THEME.teal, width: 2.5 },
      name: sigLabel,
    });
  }

  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    title: { text: titleText, font: { size: 14, color: THEME.ink } },
    xaxis: { ...PLOTLY_BASE_LAYOUT.xaxis, title: "Signal Age (Days)" },
    yaxis: { ...PLOTLY_BASE_LAYOUT.yaxis, title: "Expected Positive Reply Lift (pp)" },
    showlegend: true,
  };

  Plotly.newPlot(containerId, traces, layout, { displayModeBar: false, responsive: true });
}

export function renderConfusionMatrix(containerId, normMatrix, countsMatrix, labels, titleText) {
  const textMatrix = countsMatrix.map((row, i) =>
    row.map((cnt, j) => `${cnt}<br>(${(normMatrix[i][j] * 100).toFixed(0)}%)`)
  );

  const colorscale = [
    [0.0, THEME.mist],
    [0.2, THEME.silver],
    [0.6, THEME.teal],
    [1.0, THEME.tealDeep],
  ];

  const trace = {
    type: "heatmap",
    z: normMatrix,
    x: labels,
    y: labels,
    text: textMatrix,
    hoverinfo: "text",
    colorscale: colorscale,
    showscale: false,
  };

  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    title: { text: titleText, font: { size: 14, color: THEME.ink } },
    xaxis: { ...PLOTLY_BASE_LAYOUT.xaxis, title: "Predicted Class", tickangle: -30 },
    yaxis: { ...PLOTLY_BASE_LAYOUT.yaxis, title: "True Class", autorange: "reversed" },
  };

  Plotly.newPlot(containerId, [trace], layout, { displayModeBar: false, responsive: true });
}

export function renderReliabilityDiagram(containerId, bins, titleText) {
  const confs = bins.map((b) => b.confidence);
  const accs = bins.map((b) => b.accuracy);

  const traceDiag = {
    x: [0, 1],
    y: [0, 1],
    mode: "lines",
    line: { color: THEME.ink, dash: "dash", width: 1.5 },
    name: "Perfect Calibration",
  };

  const traceObs = {
    x: confs,
    y: accs,
    mode: "lines+markers",
    line: { color: THEME.teal, width: 2.5 },
    marker: { color: THEME.teal, size: 8 },
    name: "Calibrated Classifier",
  };

  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    title: { text: titleText, font: { size: 14, color: THEME.ink } },
    xaxis: { ...PLOTLY_BASE_LAYOUT.xaxis, title: "Confidence", range: [0, 1] },
    yaxis: { ...PLOTLY_BASE_LAYOUT.yaxis, title: "Observed Accuracy", range: [0, 1] },
    showlegend: true,
  };

  Plotly.newPlot(containerId, [traceDiag, traceObs], layout, { displayModeBar: false, responsive: true });
}

export function renderRiskCoverageCurve(containerId, points, chosenPoint, titleText) {
  const covs = points.map((p) => p.coverage);
  const risks = points.map((p) => p.risk);

  const traceCurve = {
    x: covs,
    y: risks,
    mode: "lines",
    line: { color: THEME.teal, width: 2.5 },
    name: "Risk-Coverage",
  };

  const traceChoice = {
    x: [chosenPoint.coverage],
    y: [chosenPoint.risk],
    mode: "markers",
    marker: { color: THEME.oxblood, size: 10, symbol: "square" },
    name: "Operating Point",
  };

  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    title: { text: titleText, font: { size: 14, color: THEME.ink } },
    xaxis: { ...PLOTLY_BASE_LAYOUT.xaxis, title: "Coverage (Auto-handled Share)", range: [0, 1] },
    yaxis: { ...PLOTLY_BASE_LAYOUT.yaxis, title: "Risk (Error Rate)", range: [0, 0.2] },
    showlegend: true,
  };

  Plotly.newPlot(containerId, [traceCurve, traceChoice], layout, { displayModeBar: false, responsive: true });
}

export function renderParameterRecovery(containerId, pairs, spearmanVal, titleText) {
  const xTrue = pairs.map((p) => p.true_val);
  const yEst = pairs.map((p) => p.est_val);

  const minVal = Math.min(...xTrue, ...yEst);
  const maxVal = Math.max(...xTrue, ...yEst);

  const traceDiag = {
    x: [minVal, maxVal],
    y: [minVal, maxVal],
    mode: "lines",
    line: { color: THEME.ink, dash: "dash", width: 1.5 },
    name: "y = x",
  };

  const tracePoints = {
    x: xTrue,
    y: yEst,
    mode: "markers",
    marker: { color: THEME.teal, size: 7, opacity: 0.8 },
    name: "Recovered Parameters",
  };

  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    title: { text: titleText, font: { size: 14, color: THEME.ink } },
    xaxis: { ...PLOTLY_BASE_LAYOUT.xaxis, title: "True Lift (pp)" },
    yaxis: { ...PLOTLY_BASE_LAYOUT.yaxis, title: "Posterior Median Lift (pp)" },
    showlegend: true,
  };

  Plotly.newPlot(containerId, [traceDiag, tracePoints], layout, { displayModeBar: false, responsive: true });
}

export function renderConfoundingComparison(containerId, rows, titleText) {
  const sigs = rows.map((r) => r.signal.replace(/_/g, " "));
  const naive = rows.map((r) => r.naive_lift);
  const adjusted = rows.map((r) => r.adjusted_lift);
  const truth = rows.map((r) => r.true_lift);

  const traceNaive = {
    x: sigs,
    y: naive,
    type: "bar",
    name: "Naive Lift",
    marker: { color: THEME.oxblood },
  };

  const traceAdjusted = {
    x: sigs,
    y: adjusted,
    type: "bar",
    name: "Adjusted (Loop)",
    marker: { color: THEME.teal },
  };

  const traceTruth = {
    x: sigs,
    y: truth,
    type: "scatter",
    mode: "markers",
    name: "Ground Truth",
    marker: { color: THEME.ink, size: 10, symbol: "line-ew", line: { width: 3, color: THEME.ink } },
  };

  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    title: { text: titleText, font: { size: 14, color: THEME.ink } },
    barmode: "group",
    xaxis: { ...PLOTLY_BASE_LAYOUT.xaxis, title: "Signal Type" },
    yaxis: { ...PLOTLY_BASE_LAYOUT.yaxis, title: "Lift at Age 0 (pp)" },
    showlegend: true,
  };

  Plotly.newPlot(containerId, [traceNaive, traceAdjusted, traceTruth], layout, { displayModeBar: false, responsive: true });
}

export function renderColdStartCurves(containerId, curves, titleText) {
  const traces = [];
  const styles = {
    industry_pooled: { color: THEME.teal, name: "Industry-Pooled Prior" },
    global_pooled: { color: THEME.sandDeep, name: "Global-Pooled Prior" },
    flat: { color: THEME.oxblood, name: "Flat Prior" },
  };

  for (const [prior, series] of Object.entries(curves)) {
    const st = styles[prior] || { color: THEME.ink, name: prior };
    const x = series.map((pt) => pt.k);
    const yMed = series.map((pt) => pt.log_loss);
    const yLo = series.map((pt) => pt.lo95);
    const yHi = series.map((pt) => pt.hi95);

    traces.push({
      x: x.concat(x.slice().reverse()),
      y: yHi.concat(yLo.slice().reverse()),
      fill: "toself",
      fillcolor: prior === "industry_pooled" ? THEME.bandTeal : (prior === "flat" ? THEME.bandOxblood : THEME.bandSand),
      line: { color: "transparent" },
      showlegend: false,
      hoverinfo: "skip",
    });

    traces.push({
      x: x,
      y: yMed,
      mode: "lines+markers",
      line: { color: st.color, width: 2.5 },
      marker: { color: st.color, size: 7 },
      name: st.name,
    });
  }

  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    title: { text: titleText, font: { size: 14, color: THEME.ink } },
    xaxis: { ...PLOTLY_BASE_LAYOUT.xaxis, title: "Customer Sends Seen (k)" },
    yaxis: { ...PLOTLY_BASE_LAYOUT.yaxis, title: "Held-out Log Loss" },
    showlegend: true,
  };

  Plotly.newPlot(containerId, traces, layout, { displayModeBar: false, responsive: true });
}

export function renderRegretCurves(containerId, regretSeries, titleText) {
  const traces = [];
  const palette = {
    loop: { color: THEME.teal, name: "Loop (Hierarchical Thompson)" },
    thompson_flat: { color: THEME.sandDeep, name: "Thompson Flat" },
    epsilon_greedy: { color: THEME.oxblood, name: "Epsilon-Greedy (0.1)" },
    historical_static: { color: THEME.oxbloodDeep, name: "Historical Static" },
    uniform: { color: THEME.silver, name: "Uniform" },
    oracle: { color: THEME.ink, name: "Oracle", dash: "dash" },
  };

  for (const [pol, series] of Object.entries(regretSeries)) {
    const conf = palette[pol] || { color: THEME.sand, name: pol };
    const x = series.map((pt) => pt.round);
    const y = series.map((pt) => pt.regret);

    traces.push({
      x: x,
      y: y,
      mode: "lines",
      line: { color: conf.color, width: 2.5, dash: conf.dash || "solid" },
      name: conf.name,
    });
  }

  const layout = {
    ...PLOTLY_BASE_LAYOUT,
    title: { text: titleText, font: { size: 14, color: THEME.ink } },
    xaxis: { ...PLOTLY_BASE_LAYOUT.xaxis, title: "Campaign Round (1-12)" },
    yaxis: { ...PLOTLY_BASE_LAYOUT.yaxis, title: "Cumulative Regret (Expected Meetings Lost)" },
    showlegend: true,
  };

  Plotly.newPlot(containerId, traces, layout, { displayModeBar: false, responsive: true });
}
