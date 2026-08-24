# E0152 — Headless PDF must embed math before printing

## Error

The first browser-rendered PDF of the state-interface report converted LaTeX to
HTML with Pandoc `--mathjax`. The headless Chromium print did not load the
external MathJax runtime, so every formula and many numeric values enclosed in
LaTeX math delimiters appeared as blank space. A screenshot of the source HTML
did not detect the broken printed PDF.

## Rule

When a LaTeX report cannot be compiled by a native TeX engine and must be
printed through HTML:

1. emit self-contained browser-native math, such as Pandoc `--mathml`; do not
   depend on an external MathJax script;
2. remove browser-generated headers and footers;
3. verify the final PDF itself, not only the HTML, by extracting text from every
   page and checking representative formulas, percentages, table values and
   confidence intervals;
4. give a corrected artifact a new filename so an old cached PDF cannot be
   mistaken for the fix.
