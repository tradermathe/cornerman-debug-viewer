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
import { CombinedCompareRule } from "./combined_compare.js";
import { StepPunchSyncRule } from "./step_punch_sync.js";
import { StepDetectorRule } from "./step_detector.js";
import { HipRotationReviewRule } from "./hip_rotation_review.js";
import { Vision3DRule } from "./vision_3d.js";
import { AngleChangeRule } from "./angle_change.js";
import { RoundV6Rule } from "./round_v6.js";
import { PunchClassifierRule } from "./punch_classifier.js";
import { OnDeviceLensRule } from "./ondevice_lens.js";
import { AudioImpactLensRule } from "./audio_impact_lens.js";
import { ForearmAxialityRule } from "./forearm_axiality.js";

export const RULES = [
  OverviewRule,
  OnDeviceLensRule,
  RoundV6Rule,
  PunchClassifierRule,
  AudioImpactLensRule,
  GuardDropRule,
  ArmExtensionRule,
  EngineCompareRule,
  WristSwapRule,
  CombinedCompareRule,
  StepPunchSyncRule,
  StepDetectorRule,
  HipRotationReviewRule,
  Vision3DRule,
  ForearmAxialityRule,
  AngleChangeRule,
];
