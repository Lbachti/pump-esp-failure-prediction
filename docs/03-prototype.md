# Milestone 3 static prototype

`web/` contains Check and About and limits as static HTML/CSS/vanilla JS. Check has nine grouped fields, client-side reading checks, live result/error regions, and the persistent educational-prototype notice. Open `web/index.html` directly. The prototype makes no network requests and stores no readings.

## Checks

| Check | Status and findings |
|---|---|
| Node and source checks | **Ran.** `app.js` parses. Nine inputs have labels, names, decimal input mode, and hint/error references; no duplicate IDs, external assets, prohibited result wording, requests, or storage. |
| Contrast | **Ran.** All nine light/dark pairs pass; ratios are in `docs/02-design-system.md`. |
| Hallmark | **Ran static audit.** Before: 0 FAIL / 1 REFLEX / 10 WARN. After: 0 / 0 / 0. Render **not run**: `puppeteer-core` unavailable. Node 26.13 is installed. |
| Web Design Guidelines | **Ran manual review.** Fixed ellipsis, score numerals, heading balance, and touch feedback. Sentence-case headings follow Writing Guidelines where the heading-case rules conflict. |
| UI/UX Pro Max | **Ran focused UX searches and review.** Inline errors, linked live summary, focus, reduced motion, and 48px controls are present. Its summary-focus suggestion conflicts with the required first-invalid-field focus; owner requirement wins. |
| Writing Guidelines | **Ran manual review.** ESP is defined; score and limits copy avoids probability claims, reassurance, and future-failure wording. |
| 375px / 1280px layout and keyboard-only use | **Not run.** Playwright/Puppeteer are unavailable; Edge could not run isolated headless. No visual or keyboard pass is claimed. |

Static checks cannot confirm rendered layout, screen-reader output, or keyboard behavior. The Expert mode URL is empty, so its link stays hidden.

## Milestone 4 replacements

Replace the preview banner and all mock scoring/range data with approved model output and training metadata. Ranges use glossary units.

| Mock | Current value |
|---|---|
| `MOCK_TRAINING_RANGES` | intake pressure 1–3000; discharge pressure 1–5000; both temperatures −40–350; motor current 0.1–150; voltage 1–1000; vibration 0–20; frequency 1–120; production 0–30000. Replace the range comparison and message too. |
| `MOCK_BAND_CUTOFFS` | Elevated at 0.40; High at 0.70; otherwise Low. |
| `mockScore()` | Average the nine range-normalized values, clamp to 0–1, then round the score out of 100. |
| Preview banner | “Preview: scores here are placeholders.” |
