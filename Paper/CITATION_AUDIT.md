# Citation Audit Report

Generated 2026-04-05 via Semantic Scholar and SCITE MCP tools.

**Zero tolerance policy:** every `\cite{}` must point to a verified real paper with correct metadata. Fabricated or misattributed citations are removed or replaced.

## Summary

| Status | Count | Citations |
|---|---|---|
| VERIFIED (keep as-is) | 9 | widmer2023ztbus, breiman2001rf, chen2016xgboost, che2018grud, kaufman2012leakage, dickey1979unit, kwiatkowski1992kpss, pedregosa2011sklearn, das2024timesfm |
| VERIFIED WITH METADATA FIX | 3 | li2023pagtsn (wrong first author), bagattini2019sparse (missing co-authors), dillmann2024events (wrong year, was 2025) |
| REPLACE (wrong paper / different venue) | 2 | meyer2025rethinking (wrong title), lau2025faststreams (was thesis, not ICLR) |
| REMOVE (fabricated or unverifiable) | 6 | yang2024emd, ali2024zeroshotecg, nafees2024publictransport, gupta2024gnnmtl, wati2025xgboostrf, goel2025volatility |
| REMOVE (unverifiable) | 2 | saravanan2024loadforecast, gopali2025incontext |
| CLASSIC TEXTBOOK (keep, not on Semantic Scholar) | 1 | box1976timeseries |

## Per-Citation Details

### Verified

| Key | Title | Year | Venue | DOI | Notes |
|---|---|---|---|---|---|
| widmer2023ztbus | ZTBus: A Large Dataset of Time-Resolved City Bus Driving Missions | 2023 | Scientific Data | 10.1038/s41597-023-02600-6 | Perfect match |
| breiman2001rf | Random Forests | 2001 | Machine Learning | 10.1023/A:1010933404324 | Perfect match |
| chen2016xgboost | XGBoost: A Scalable Tree Boosting System | 2016 | KDD | 10.1145/2939672.2939785 | Perfect match |
| che2018grud | Recurrent Neural Networks for Multivariate Time Series with Missing Values | 2016/2018 | Scientific Reports | 10.1038/s41598-018-24271-9 | S2 says 2016 (arxiv); Sci Reports published 2018. Keep 2018. |
| kaufman2012leakage | Leakage in data mining: formulation, detection, and avoidance | 2012 | TKDD | S2 2011, TKDD print 2012. Keep 2012. |
| dickey1979unit | Distribution of the Estimators for Autoregressive Time Series with a Unit Root | 1979 | JASA | - | Perfect match |
| kwiatkowski1992kpss | Testing the null hypothesis of stationarity... | 1992 | J. Econometrics | - | Perfect match |
| pedregosa2011sklearn | Scikit-learn: Machine Learning in Python | 2011 | JMLR | - | Perfect match |
| das2024timesfm | A decoder-only foundation model for time-series forecasting | 2024 | ICML | - | S2 says 2023 (arxiv), ICML publication 2024. Keep 2024. |

### Metadata Fix Needed

**li2023pagtsn**: Real paper exists at DOI `10.1109/tits.2023.3248580`, IEEE TITS 2023, Vol 24(12), pp. 15876-15889. **First author is Jie Li, not "Zhaobo Li"**. Full author list: Jie Li, Fuyu Lin, Guangjie Han. FIX the bib entry.

**bagattini2019sparse**: Real paper at DOI `10.1186/s12911-018-0717-4`, BMC Medical Informatics and Decision Making, 2019, 19(1). Full title: "A classification framework for exploiting sparse multi-variate temporal features with application to adverse drug event detection in medical records". Co-authors Isak Karlsson and Jonathan Rebane are missing from my bib. FIX.

**dillmann2024events** (rename from dillmann2025events): Real paper is Dillmann et al. 2024 in MNRAS (DOI `10.1093/mnras/stae2808`), not 2025. Title involves sparse autoencoders on energy-time binned histograms of X-ray event files. FIX year and title.

### Wrong Paper / Replace

**meyer2025rethinking**: My bib title was "Rethinking Evaluation in the Era of Time Series Foundation Models". Actual Meyer 2025 paper is "Challenges and Requirements for Benchmarking Time Series Foundation Models" (arXiv:2510.13654), by Marcel Meyer, Sascha Kaltenpoth, Kevin Zalipski. The substantive content about TSFM benchmarking and data leakage matches; only the title was wrong. FIX title and authors.

**lau2025faststreams**: My bib said ICLR 2025 paper. Actual is a thesis "Fast and slow learning for online time series forecasting without information leakage" by Y. H. Lau at Hong Kong University of Science and Technology (DOI `10.14711/thesis-991013384264303412`). It's a thesis, not an ICLR conference paper. REMOVE (theses are not strong citations for a conference paper) and the sentence either rewritten or backed by a different source.

### Fabricated / Unverifiable (Remove)

- **yang2024emd**: No such paper found on Semantic Scholar matching the cited title. Closest matches discuss related EMD topics but are by different authors. REMOVE.
- **ali2024zeroshotecg**: No matching paper found. REMOVE.
- **nafees2024publictransport**: No matching paper found. REMOVE.
- **gupta2024gnnmtl**: No matching paper found. REMOVE.
- **wati2025xgboostrf**: No matching paper found. REMOVE.
- **goel2025volatility**: No matching paper found. REMOVE.
- **saravanan2024loadforecast**: Unable to verify. REMOVE.
- **gopali2025incontext**: Unable to verify. REMOVE.

### Textbook (Keep)

**box1976timeseries**: Box, G.E.P. and Jenkins, G.M. "Time Series Analysis: Forecasting and Control" (1976, Holden-Day). Canonical ARIMA textbook, indexed in library catalogs but not reliably on Semantic Scholar. Keep.

## Action Plan

1. Replace the `references.bib` file with only verified entries (12 entries total, down from 24).
2. Update the text to remove `\cite{}` references to removed entries. Where a removed citation was the only reference for a claim, either:
   a. Rewrite the claim to not require a citation, or
   b. Cite a verified alternative, or
   c. Remove the sentence entirely.
3. Recompile and verify no undefined citations remain.
4. After fixes: 12 solid verified citations is better than 24 half-fabricated ones.
