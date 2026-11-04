# ESP PredictAI Design System

Minimal, high-contrast visual rules for the ESP reading check. This file is the source of truth for token values. The selected workspace reference supplied only neutral-surface hierarchy, clear forms, and a restrained spacing structure. Its identity, palette, typeface, and unnecessary components have been replaced.

## Colour roles

Light mode is the default. Dark mode follows the operating-system preference. Use the same roles across every component; do not add unlisted hues or local hex values.

| CSS token | Light | Dark | Use |
|---|---|---|---|
| `--color-background` | `#F2F5F3` | `#121A16` | Page canvas |
| `--color-surface` | `#FFFFFF` | `#1B2520` | Fields and result surfaces |
| `--color-text` | `#18221D` | `#F2F6F3` | Main copy and headings |
| `--color-muted` | `#45534B` | `#B9C5BE` | Hints and secondary copy |
| `--color-border` | `#6B776F` | `#748279` | Field and component edges |
| `--color-accent` | `#0F6A58` | `#83DFC6` | Primary action and selection |
| `--color-focus-ring` | `#0F6A58` | `#83DFC6` | Keyboard focus |
| `--color-on-accent` | `#FFFFFF` | `#13211B` | Primary button text |

Risk uses colour, an inline SVG symbol, and a visible label together:

| State | CSS token | Light icon/text | Dark icon/text |
|---|---|---|---|
| Normal | `--color-risk-normal` | `#176B4B` | `#7FE0B1` |
| Elevated | `--color-risk-elevated` | `#805000` | `#FFD184` |
| High | `--color-risk-high` | `#A32929` | `#FFA19B` |

Never use colour as the only state signal. Do not use these bands as equipment limits.
Field errors reuse the High signal colour through `--color-error` and always include correction text.

## Type and spacing

Use the native system sans stack: `system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`. Load no remote font. Keep five sizes and two weights:

| Token | Size | Weight |
|---|---:|---:|
| `--text-small` | 12px | 400 |
| `--text-label` | 14px | 600 |
| `--text-body` | 16px | 400 |
| `--text-section` | 20px | 600 |
| `--text-display` | 28px | 600 |

Use a 4/8 spacing scale: 4, 8, 12, 16, 24, 32, 40, 48px. Pair related label, field, and hint within 8px; use 24px between field groups and 32px between specimen sections.

## Controls and surfaces

Use 8px corners for fields, buttons, and result surfaces. Inputs and buttons are at least 48px high. Use visible 1px borders before adding depth; resting surfaces have no shadow. The focus ring is 3px, offset by 2px. Errors sit beside their field and include a clear correction. Labels remain visible when fields contain values.

Use the accent for the primary action and focus only. Secondary buttons use the surface, border, and text roles. Result bands use the risk icon, risk label, and a short explanation. The educational-prototype notice stays visible near the result and states that the data provenance is unverified, covers six wells, and has near-chance cross-well accuracy.

## Layout and motion

Hallmark's modern-minimal rules govern composition: one task, clear hierarchy, sparse surfaces, no nested cards, decorative gradients, eyebrow labels, emoji, or ornamental status dots. The specimen uses a component-playground structure, not an additional product screen. Keep the content centred and capped; show related readings in two columns on wide viewports and one column at 375px.

Transitions may use only named colour or opacity properties and last at most 180ms. Do not animate size or position. Loading rotation is optional and stops under `prefers-reduced-motion: reduce`; all state text remains visible without motion.

## Risk and product language

Call the numeric output a **model score**, not a calibrated chance of failure. Say it classifies the supplied readings for one day. It does not forecast a future failure or estimate remaining life. Keep the educational-prototype notice persistent. Do not invent operating limits from the training range.
