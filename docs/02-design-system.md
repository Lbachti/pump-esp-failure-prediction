# Milestone 2: design system

## Decisions and conflict log

No conflict with Milestone 1: this is a static specimen, not a product screen. I adapted the selected workspace reference's neutral hierarchy, form clarity, and spacing. `DESIGN.md` owns values; Hallmark owns composition and anti-template rules.

| Source tension | Resolution |
|---|---|
| Source palette/type and branded components conflict with the new accent, five sizes, two weights, system font, and sparse scope. | Replaced values; removed hero, pricing, testimonials, logo wall, tabs, and feature cards. |
| Hallmark's warm theme uses warm paper, texture, and letterpress depth. | Manually selected modern-minimal Coral; changed to cool sage-gray and evergreen; removed texture/shadows. |
| Hallmark uses one accent; three risk colors are required. | Evergreen marks actions; risk hues appear only with icon and text. |
| Component-playground suggests interactive code-copy; JavaScript is out of scope. | Static previews use adjacent `tokens.css`; no copy controls. |

No Node/npx: manually selected `21 Component Playground` and Coral. `hallmark audit` did not run; I applied its floor manually. UI/UX Pro Max favored Minimalism & Swiss; I rejected testimonials and remote fonts. Fetched [current web guidelines](https://raw.githubusercontent.com/vercel-labs/web-interface-guidelines/main/command.md).

## Contrast check

Inline relative-luminance script checked CSS tokens. All text pairs exceed 4.5:1; component edges and focus exceed 3:1.

| Pair | Light | Dark | Minimum |
|---|---:|---:|---:|
| Main text / surface | 16.34:1 | 14.45:1 | 4.5:1 |
| Muted text / surface | 8.10:1 | 8.85:1 | 4.5:1 |
| Border / surface | 4.67:1 | 3.91:1 | 3:1 |
| Focus ring / nearest background or surface | 5.94:1 | 10.01:1 | 3:1 |
| Lowest risk icon/text / surface | 6.48:1 | 8.13:1 | 4.5:1 |
| Primary button text / accent | 6.52:1 | 10.57:1 | 4.5:1 |

## Audit counts

| Review | Before fixes | After review |
|---|---:|---:|
| Hallmark manual checks | 3 | 0 |
| Web Design Guidelines | 4 | 0 |
| UI/UX Pro Max | 1 | 0 |

Initial findings (8): Hallmark 3 (body overflow, dot metadata, nested mobile surface); web guidelines 4 (skip link, link hover, mobile input attributes, title case); UI/UX Pro Max 1 (static live regions). Final manual pass: 0. Source check: 19 labeled inputs, 7 states, 8 inline SVGs, no duplicate IDs, broken links, scripts, or external assets. Open `design-system/index.html` directly; no server/build needed.
