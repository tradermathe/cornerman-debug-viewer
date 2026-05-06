// Overview panel — no rule lens, just shows the raw per-joint state at the
// current frame. Useful when you want to eyeball Apple Vision's confidence
// distribution before picking a specific rule.

import { JOINT_NAMES, confColor } from "../skeleton.js";

let host;

export const OverviewRule = {
  id: "overview",
  label: "Overview (no lens)",

  mount(_host) {
    host = _host;
    host.innerHTML = `
      <h2>Per-joint state</h2>
      <p class="hint">Confidence is colour-coded: green ≥ 0.5, amber ≥ 0.2, red below.
      A zero means Apple Vision didn't detect that joint at all (different from
      YOLO, which usually returns a low-conf guess).</p>
      <table class="joint-table">
        <thead><tr><th>#</th><th>Joint</th><th>x</th><th>y</th><th>conf</th></tr></thead>
        <tbody id="joint-tbody"></tbody>
      </table>
    `;
  },

  update(state) {
    const tbody = host.querySelector("#joint-tbody");
    const rows = [];
    for (let j = 0; j < 17; j++) {
      const x = state.pose.skeleton[(state.frame * 17 + j) * 2];
      const y = state.pose.skeleton[(state.frame * 17 + j) * 2 + 1];
      const c = state.pose.conf[state.frame * 17 + j];
      rows.push(
        `<tr>
          <td class="muted">${j}</td>
          <td>${JOINT_NAMES[j]}</td>
          <td class="num">${x.toFixed(0)}</td>
          <td class="num">${y.toFixed(0)}</td>
          <td class="num" style="color:${confColor(c)}">${c.toFixed(2)}</td>
        </tr>`
      );
    }
    tbody.innerHTML = rows.join("");
  },
};
