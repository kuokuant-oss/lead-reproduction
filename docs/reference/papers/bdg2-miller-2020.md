# BDG2 Data Descriptor Reference

**Citation**: Miller, C., Kathirgamanathan, A., Picchetti, B., Arjunan, P.,
Park, J. Y., Nagy, Z., Raftery, P., Hobson, B. W., Shi, Z., & Meggers, F.
(2020). The Building Data Genome Project 2, energy meter data from the ASHRAE
Great Energy Predictor III competition. *Scientific Data*, 7, 368.

**DOI**: <https://doi.org/10.1038/s41597-020-00712-x>

**License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

**Official repository**:
<https://github.com/buds-lab/building-data-genome-project-2>

**Local PDF note**: The source PDF was provided locally at
`C:\Users\tonykuo\Downloads\BDG2.pdf` and copied to
`docs/reference/papers/bdg2-miller-2020.pdf` for local review. The PDF is about
11 MB, so it is intentionally gitignored to preserve the repo's 500 KB
large-file gate. This Markdown card is the tracked citation record.

## Provenance And Cleaning Summary

Miller et al. describe BDG2 as an openly released whole-building meter corpus
with metadata, weather, and raw plus cleaned meter data. The local archive used
by this repo follows that raw/cleaned release structure.

The paper describes the raw-data processing path as unit conversion, setting
negative readings to missing, removing meters with more than 50% negative
readings, removing meters with more than 100 consecutive days of missing
readings, applying a log plus three-standard-deviation outlier rule, and
rounding readings to four decimals.

The cleaned data then applies additional cleaning, including Twitter
AnomalyDetection outlier removal, removal of zero-reading runs longer than 24
hours, and removal of electricity zeros. These release-level cleaning rules
explain why the local EDA observes zero raw negative-reading share and higher
cleaned null rates than raw null rates across all meters. The raw/cleaned delta
is therefore a BDG2 release data-quality delta, not a label.
