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
import { ArmExtensionRule } from "./arm_extension.js";
import { EngineCompareRule } from "./engine_compare.js";
import { WristSwapRule } from "./wrist_swap.js";
import { StepPunchSyncRule } from "./step_punch_sync.js";
import { StepDetectorRule } from "./step_detector.js";
import { HipRotationRule } from "./hip_rotation.js";
import { FacingDirectionRule } from "./facing_direction.js";
import { Vision3DRule } from "./vision_3d.js";

export const RULES = [
  OverviewRule,
  GuardDropRule,
  ArmExtensionRule,
  EngineCompareRule,
  WristSwapRule,
  StepPunchSyncRule,
  StepDetectorRule,
  HipRotationRule,
  FacingDirectionRule,
  Vision3DRule,
];
