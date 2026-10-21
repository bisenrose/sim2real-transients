**Every Phase 1 configuration (mean ± SD over split seeds)**

| Config | Preprocessing | SLSN-I code | Dropped | Seeds | PLAsTiCC F1 | Transfer F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 00_legacy_reproduction | per_domain | 65 (wrong) | none | 1 | 0.790 | 0.219 |
| 01_legacy_drop_absmag | per_domain | 65 (wrong) | abs_peak_mag | 1 | 0.783 | 0.330 |
| 02_legacy_codes_consistent | consistent | 65 (wrong) | none | 5 | 0.803 ± 0.010 | 0.405 ± 0.015 |
| 10_corrected_consistent | consistent | 95 (correct) | none | 5 | 0.783 ± 0.008 | 0.486 ± 0.019 |
| 11_corrected_consistent_drop_absmag | consistent | 95 (correct) | abs_peak_mag | 5 | 0.769 ± 0.012 | 0.488 ± 0.010 |
| 12_corrected_consistent_drop_redshift | consistent | 95 (correct) | redshift | 5 | 0.787 ± 0.009 | 0.516 ± 0.013 |
| 13_corrected_consistent_drop_artefacts | consistent | 95 (correct) | t_0_ZTF_g, t_0_ZTF_r, time_span | 5 | 0.789 ± 0.009 | 0.501 ± 0.017 |
| 14_corrected_consistent_confirmed_only | consistent | 95 (correct) | none | 5 | 0.783 ± 0.008 | 0.523 ± 0.025 |
| 20_corrected_per_domain | per_domain | 95 (correct) | none | 5 | 0.724 ± 0.023 | 0.523 ± 0.010 |
| 21_corrected_per_domain_drop_absmag | per_domain | 95 (correct) | abs_peak_mag | 5 | 0.715 ± 0.035 | 0.561 ± 0.010 |
| 22_corrected_per_domain_drop_redshift | per_domain | 95 (correct) | redshift | 5 | 0.709 ± 0.024 | 0.530 ± 0.007 |
| 23_corrected_per_domain_drop_absmag_confirmed_only | per_domain | 95 (correct) | abs_peak_mag | 5 | 0.715 ± 0.035 | 0.574 ± 0.016 |
