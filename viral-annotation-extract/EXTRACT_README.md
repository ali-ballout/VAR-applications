# Virus-specific functional annotations — extracted records

Targeted extraction for exactly the **3,288 study virus NCBI TaxIDs**. No complete
database mirror was downloaded or is redistributed here.

## Headline numbers

| | |
|---|---|
| GO annotation rows (mapped to a study species) | 4,572,619 |
| of which EXACT species-level | 312,162 |
| of which strain/isolate (VERIFIED_DESCENDANT) | 4,260,457 |
| UNMAPPED rows (held separately) | 684 |
| UniProtKB protein records | 370,585 |
| distinct GO terms | 927 |
| ViralZone family-level trait rows | 2,924 |
| temporally eligible (snapshot 2026-07-28 <= cutoff 2026-08-18) | all 4,572,619 GO rows |

## Coverage

| cohort | n | exact | strain-only | none | any |
|---|---|---|---|---|---|
| all study viruses | 3288 | 244 | 1289 | 1755 | 46.6% |
| multi-host benchmark | 406 | 61 | 302 | 43 | 89.4% |
| historical test | 54 | 11 | 41 | 2 | 96.3% |

**Coverage is strongly non-random across cohorts.** The benchmark and test subsets are far
better annotated than the full set. Any evaluation built on these annotations inherits that
bias. It is a property of the sources, not of this extraction.

**Only 244 of 3,288 viruses have annotations on their own species TaxID.** For 1,289 more,
everything available sits on strain/isolate descendants. Those are supplied as
`VERIFIED_DESCENDANT` with their original taxid and accession intact — deciding whether to
treat them as species evidence is a modelling choice this package deliberately leaves open.

## Files

- `viral_functional_annotations.tsv.gz` — one GO annotation per row, 22 columns, full provenance
- `protein_to_study_virus_mapping.tsv.gz` — 370,585 proteins with mapping status and justification
- `annotation_coverage_by_virus.tsv` — all 3,288 rows, one per study virus
- `coverage_summary.tsv`, `coverage_by_aspect_and_evidence.tsv` — per-cohort breakdowns
- `unmatched_identifiers.tsv` — 1,755 viruses with no annotation + 2 unmappable source taxids
- `unmapped_annotation_records.tsv.gz` — the 684 rows excluded from the main table
- `ambiguous_mappings.tsv` — empty by result: no strain descends from >1 study species
- `viralzone_family_traits.tsv` — family-level only, `FAMILY_ONLY`, host-leakage sections withheld
- `go_terms_used.tsv` — labels from GO `releases/2026-07-26`
- `download_manifest.json` — every source, version, version evidence, licence, and SHA-256 of all files
- `acquisition_log.txt` — what was attempted, what was used, what was blocked, and why
- `raw/` — the actual retrieved source files (QuickGO, UniProt, ViralZone, GO) + saved version evidence
- `scripts/` — stdlib-only reproduction scripts, with `RUN_ORDER.txt`

## Eligibility

GO annotations come from QuickGO's annotation snapshot timestamped **2026-07-28 10:08 UTC**,
which is UniProt-GOA release 236 and sits before the 2026-08-18 cutoff. That is a dated
snapshot, not an inference from each annotation's own date field. Corroborated independently
by the GOA FTP archive headers for `goa_uniprot_all.gaf.236.gz` (Last-Modified 29 Jul 2026),
saved in `raw/provenance/`. Max annotation date observed: 2026-07-27. Rows dated after the
cutoff: 0.

UniProt protein records are release **2026_04 (2026-11-04)**, after the cutoff. They are used
only for protein and organism identity, never as annotation evidence, and are flagged as
ineligible in the manifest. ViralZone publishes no dated release, so all its rows are
`temporally_eligible=false`.

## Constraints observed

No virus-host association labels were used at any stage. No model was trained and no
virus-virus similarity was computed. Family-level text was never promoted to species
annotations. Strain annotations keep their original identifiers and are never relabelled as
species-level. Missing annotation is recorded as missing, never as a biological negative.
ViralZone sections matching host / receptor / tropism / transmission / vector / reservoir /
pathology / epidemiology patterns were withheld (206 sections) to prevent label leakage.

## Not included

The 11.7 GB `goa_uniprot_all.gaf.236.gz` was not downloaded. QuickGO serves the same release
filtered by TaxID server-side, so the bulk file would have added nothing. Its URL, size and
Last-Modified are recorded in the manifest. Note that EBI publishes no checksums for archived
GOA releases; where a publisher checksum exists it was verified, and where none exists the
manifest says so rather than implying verification.
