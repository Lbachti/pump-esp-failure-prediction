# Design system decisions

`DESIGN.md` owns token values; the Hallmark modern-minimal direction owns sparse composition. The Milestone 1 three-screen outline is superseded: Check contains the nine-reading form and result on one page; About and limits is the second screen. Expert mode remains hidden until a URL is configured.

## Owner-approved corrections

| Decision | Implementation |
|---|---|
| Result headline | One band only: Low, Elevated, or High. No running/failed class. |
| Low state | Neutral circle with no check mark. Elevated and High retain distinct colour, icon, and label. |
| Score line | “Model score 64 of 100. This is not a probability.” Placeholder values carry a visible preview banner. |
| Type | Hints and prototype notice use 14px / 400. 12px is reserved for non-essential metadata. |

The one-day reading context stays explicit. The prototype keeps the unverified-provenance, six-well, near-chance cross-well, and operating-decision caveats visible. Preview ranges are not equipment limits.

## Token contrast

Ran an inline Node script against light/dark values parsed from `design-system/tokens.css`; WCAG relative luminance ratios:

| Pair | Light | Dark | Minimum |
|---|---:|---:|---:|
| Main text / surface | 16.34:1 | 14.45:1 | 4.5:1 |
| Muted text / surface | 8.10:1 | 8.85:1 | 4.5:1 |
| Border / surface | 4.67:1 | 3.91:1 | 3:1 |
| Focus ring / background | 5.94:1 | 11.25:1 | 3:1 |
| Focus ring / surface | 6.52:1 | 10.01:1 | 3:1 |
| Low label / surface | 8.10:1 | 8.85:1 | 4.5:1 |
| Elevated label / surface | 6.85:1 | 11.04:1 | 4.5:1 |
| High label / surface | 7.23:1 | 8.13:1 | 4.5:1 |
| Button text / accent | 6.52:1 | 10.57:1 | 4.5:1 |
