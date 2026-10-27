## fig01_paired_effects

Paired effects for clean and corrupted test inputs. Colored symbols show seeds 0/1/2; black diamonds show the mean and error bars the sample SD of the three paired differences. Corrupted scores first average the 20 non-clean conditions within each seed. Error bars are not confidence intervals; no significance test is implied. N1-N0 and MTL-A are primary contrasts; N0-A is explanatory.

## fig02_corruption_curves

Test scores versus corruption severity. C denotes clean; levels 1-4 follow the frozen benchmark. Lines are means over three seeds and bands are plus/minus one sample SD, not confidence intervals. Each corruption is applied independently to the clean canvas. The clean point is repeated for reference in each panel and excluded from corrupted aggregates.

## fig03_family_effects

Primary paired effects within each corruption family, averaging four severities within each seed before taking differences. Colored points show individual seeds; black diamonds and bars show mean and sample SD across paired seeds. Family-level analyses are descriptive and exploratory; no multiple-comparison-adjusted inference is made.

## fig04_attribute_effects

Per-attribute paired differences for N1-N0 and MTL-A. Each cell averages the same 20 corrupted test conditions for one attribute and seed. Red denotes a positive difference and blue a negative difference; all panels share the same symmetric scale. These post-hoc diagnostics do not establish attribute-specific causal effects or statistical significance.

## fig05_training_curves

Training and checkpoint selection for the nine new runs. Colors identify models; solid/dashed/dotted lines identify seeds 0/1/2. Markers indicate the clean-validation-mAP-selected checkpoints. All N0/N1 runs stopped at epoch 4 with epoch 2 selected; MTL stopped at epochs 5/5/6 with epochs 3/3/4 selected. Right panel shows attribute BCE only, not the combined MTL loss.
