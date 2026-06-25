# Demo Dashboard — Build Plan

A clean, branded **Clip Review dashboard** that shows a potential coaching partner what the
model sees: a round of film → punch timeline → one punch → its scored rules + skeleton proof.

## Strategic frame (decided)
- **Audience:** a specific known coach/brand (domain expert — will probe every number).
- **Goal:** first touch → land a meeting. The door-opener is a **~90-second narrated walkthrough**,
  recorded *off* this dashboard. The interactive version is what you bring *to* the meeting.
- **Therefore:** build the **hero path** first (film + timeline + round summary + punch detail).
  Defer Annotate / Export / fighter selector / multi-round / per-frame mechanics depth.
- **Personalize:** make the coach's signature teaching point the **hero rule**.

## Architecture (decided)
- The demo is a **second front-end in this repo** — purely additive. It does **not** touch
  `viewer.js` or any existing lens, so it cannot break the debug tool.
- It **reuses the existing DOM-free modules**: `loadOnDeviceSkeleton` / `loadOnDeviceAnalysis`
  (`ondevice-loader.js`), `fetchRoundBlobs` (`firebase-source.js`), and `drawSkeleton`
  (`skeleton.js`).
- New code lives under `js/demo/` + `demo.html` + `demo/assets/`. Its own brand CSS — no debug chrome.
- Data shapes and keep/cut/reframe calls: see [`demo-data-contract.md`](./demo-data-contract.md).
- Design source of truth: the `Cornerman AI Dashboard` handoff (brand tokens, layout, interactions).

## Files
```
demo.html                 # shell, imports js/demo/app.js
js/demo/style.css         # brand tokens (navy/rust/paper, Barlow Condensed) + layout
js/demo/data.js           # load session (real loaders | fixture) → raw {skeleton, analysis}
js/demo/state.js          # state object + derived selectors (punch list, summary, selection)
js/demo/rules.js          # per-rule DEMO DESCRIPTORS — evaluate(state, punch) → {verdict, metric, cue}
js/demo/app.js            # render loop: film+skeleton, timeline, summary, punch detail, transport
demo/assets/sample-session.json   # synthetic, schema-faithful fixture (until a real session is dropped in)
```

## Phases & verification
- **P1 — pipeline** *(this scaffold)*: load fixture → build `state` → film plays with skeleton
  synced → timeline renders → click a punch selects it. ✅ when it renders and `index.html` is untouched.
- **P2 — punch→rules**: wire the rule descriptors. ✅ when card data matches the fixture and the
  known PASS/FAIL punches read correctly. **Blocked on 🚩#1 (per-punch verdicts) for real data.**
- **P3 — brand UI**: implement the design pixel-faithfully over the working pipeline. ✅ screenshot review.
- **P4 — record the 90s walkthrough** off the hero session. ✅ a sendable video.
- **P5+ — deferred**: proof-viz polish, Firebase live-load, domain hosting (Firebase Hosting),
  dress rehearsal. Not needed to land the meeting.

## Open items for the user (on return)
1. **Hero session** — a real `mp4` + `_skeleton.json` + `_ondevice_analysis.json` (or a Firebase
   `sessionId` + round). Picks the one with the cleanest PASS/FAIL contrast on the coach's rule.
2. **Coach identity + signature teaching point** — sets the hero rule and framing.
3. **🚩#1 per-punch verdicts** — confirm whether the sidecar carries per-punch rows for
   arm_extension / hit_height / hand_return_path (see data-contract). Decides the centerpiece wiring.
4. **Verdicts vs scores** — recommend verdict + metric over invented 0–100 (confirm).
