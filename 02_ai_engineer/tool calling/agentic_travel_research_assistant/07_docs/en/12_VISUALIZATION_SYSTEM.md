# Visualization system and dashboard styling

## Introduction

The project's three analytics surfaces — **Project Statistics**, **Data Quality**, and **Live Statistics** — use one shared visualization system. The goal is not simply to display many charts, but to keep every chart readable, visually consistent, high-contrast in both light and dark modes, and comparable in size.

The shared implementation lives in:

```text
08_ui/chart_theme.py
```

This separates chart content from presentation rules. Dashboard modules focus on data and figure construction, while contrast, chart height, axes, legends, and hover styling are controlled centrally.

## 1. User-selectable Light and Dark chart themes

The Streamlit sidebar contains an **Analytics chart theme** selector:

- `Light · high contrast`
- `Dark · high contrast`

The selected chart theme applies to Plotly figures across all three analytics dashboards.

### Light theme

The light-theme rules are:

```text
background: white
primary text: near-black
axis/tick text: near-black
grid: visible light gray
hover: white background + black text
```

Primary text uses `#111111`, avoiding low-contrast pale-gray labels on a white background.

### Dark theme

The dark-theme rules are:

```text
background: #0E1117
primary text: white
axis/tick text: white
grid: visible dark gray
hover: dark background + white text
```

This prevents Plotly labels from disappearing against a dark dashboard surface.

## 2. Consistent chart height

Primary interactive charts use one common height:

```python
CHART_HEIGHT = 460
```

This keeps side-by-side cards aligned, gives comparison charts equal visual weight, reduces layout jumps between tabs, and avoids a collage-like dashboard appearance.

Figure builders may define internal layout properties, but the final render layer normalizes the displayed chart height.

## 3. Shared Plotly rendering pipeline

The rendering path is:

```text
data / CSV / SQLite
        ↓
chart builder
        ↓
Plotly Figure
        ↓
style_figure(...)
        ↓
Light or Dark contrast
+ fixed height
+ axis styling
+ legend styling
+ hover styling
        ↓
render_plotly(...)
        ↓
Streamlit
```

Dashboard modules do not render directly with raw `st.plotly_chart()` calls. The shared wrapper provides:

- theme application;
- fixed height;
- `width="stretch"`;
- responsive Plotly configuration;
- explicit Streamlit keys;
- `theme=None`, preventing Streamlit's Plotly theme from overriding project contrast rules.

## 4. Contrast rules

The shared theme layer explicitly styles:

- paper background;
- plot background;
- layout font;
- title font;
- x/y tick labels;
- x/y axis titles;
- grid lines;
- zero lines;
- axis lines;
- legend text;
- hover labels;
- annotations;
- polar/radar axis labels;
- colorbar labels and titles.

This is deliberate: different Plotly trace types do not inherit every font setting from one single source. Merely switching to `plotly_dark` would not guarantee consistent contrast for annotations, radar labels, or colorbars.

## 5. Interactive and saved plots

The project maintains two visualization forms.

### Interactive Plotly charts

These are the primary Streamlit views and support:

- hover details;
- zoom and pan;
- legend toggling;
- dynamic filters;
- responsive width;
- Plotly toolbar actions.

### Saved static PNG snapshots

Reproducible snapshots are stored under `06_results/project_statistics/` and `06_results/data_quality/`. Static images always use a fixed high-contrast light style:

```text
white background + black text
```

A PNG cannot dynamically switch themes, so the light snapshot is intentionally optimized for GitHub, documentation, and exported reports.

## 6. Shared visual language across the three dashboards

### Project Statistics

Shows reproducible project state:

- dataset scale;
- memory footprint;
- query/entity diversity;
- benchmark complexity;
- router precision/recall/F1;
- methodology quality and latency.

### Data Quality

Diagnoses dataset reliability:

- quality gates;
- linguistic diversity;
- repetition risk;
- entity-name diversity;
- split leakage;
- intent balance.

### Live Statistics

Uses actual stored UI activity:

- question/tool-call trends;
- run success;
- tool reliability;
- latency distributions;
- workflow sequences;
- destination mix;
- methodology usage;
- interaction history.

The content differs, but typography, contrast, height, and interaction rules are shared.

## 7. Rules for adding a new chart

When adding a new interactive chart:

1. the builder returns a `plotly.graph_objects.Figure`;
2. avoid hard-coded low-contrast text colors;
3. render through the shared `render_plotly()` wrapper;
4. assign an explicit unique Streamlit `key`;
5. pass the current `chart_theme` into the dashboard renderer;
6. do not use deprecated `use_container_width` calls;
7. keep the standard dashboard chart height at `CHART_HEIGHT` unless a deliberate exception is documented.

## 8. Testing

Visualization regression tests verify that:

- Light mode uses a white background with black text;
- Dark mode uses a dark background with white text;
- charts receive the shared default height;
- all three dashboards accept the `chart_theme` parameter;
- deprecated Streamlit width APIs are absent;
- interactive Plotly elements use explicit keys.

## Conclusion

Centralizing the visualization layer makes the three analytics surfaces behave as one coherent system. Users receive high contrast in both themes, side-by-side charts align consistently, and saved snapshots remain suitable for documentation and reporting.
