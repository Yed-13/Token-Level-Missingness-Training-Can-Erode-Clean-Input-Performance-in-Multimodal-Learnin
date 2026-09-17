# Bibliography corrections, 2026-09-17

This pass checked publication metadata and citation resolution. It does not
claim a new full-text verification of every scientific statement.

- **EMT-DLFR:** added volume 15, issue 1 and the publication DOI; retained
  2024 and pages 309–325. Verified against the [IEEE record](https://ieeexplore.ieee.org/document/10122560/)
  and publisher-registered DOI metadata retrieved by content negotiation.
- **GCNet:** replaced early-access pages 1–14 with final pages 8419–8432,
  volume 45, issue 7; added DOI 10.1109/TPAMI.2023.3234553. The
  [PubMed final record](https://pubmed.ncbi.nlm.nih.gov/37018613/) supplies the
  pagination; DOI content negotiation still returned early-access metadata.
- **MOSI:** replaced the preprint citation with the corresponding journal
  publication, *Multimodal Sentiment Intensity Analysis in Videos: Facial
  Gestures and Verbal Messages*, IEEE Intelligent Systems 31(6), 82–88
  (2016), DOI 10.1109/MIS.2016.94. The [author's arXiv record](https://arxiv.org/abs/1606.06259)
  identifies the journal volume/pages; publisher DOI metadata supplies the
  final title.
- **Preprints:** RoBERTa, MRCF, SIEVE, MissBench and DERL use `@misc` with
  arXiv identifiers and archive-issued DOIs. Their arXiv records were checked:
  [RoBERTa](https://arxiv.org/abs/1907.11692),
  [MRCF](https://arxiv.org/abs/2608.03611),
  [SIEVE](https://arxiv.org/abs/2607.17262),
  [MissBench](https://arxiv.org/abs/2603.09874),
  [DERL](https://arxiv.org/abs/2602.01833).
- Removed unused `selfmm` and `misa` BibTeX entries; neither appeared in the
  rendered reference list. Their prior versions remain in Git history.

The general citation validator mistakes LaTeX class internals such as
`@oddfoot` for Markdown citations. A manuscript-specific regression test checks
actual LaTeX citation commands against bibliography keys, including duplicates
and unused entries. BibTeX compilation provides a second resolution check.
