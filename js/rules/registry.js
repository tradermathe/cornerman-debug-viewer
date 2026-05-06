// Each rule module exports a small object that the viewer mounts when the user
// selects it from the dropdown. The contract:
//
//   id            string — slug shown in the URL hash (future) and select value
//   label         string — human label in the dropdown
//   mount(host, state)        — build the rule's side-panel DOM into `host`
//   update(state)             — called on every frame change to refresh DOM
//   draw(ctx, state)          — called on every redraw to paint extras on canvas
//   skeletonStyle(state)      — optional, returns style overrides for drawSkeleton
//
// To add a new rule: drop a file in rules/, import + push it here.

import { OverviewRule } from "./overview.js";
import { GuardDropRule } from "./guard_drop.js";
import { ForearmProjectionRule } from "./forearm_projection.js";

export const RULES = [
  OverviewRule,
  GuardDropRule,
  ForearmProjectionRule,
];
