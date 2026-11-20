RC-SOP reviewer 1 follow-up experiments

Boundary metrics use recorded UYPYDJ fixed-current endpoint measurements with
a withheld pulse, at least three other current levels for regression and an
explicit endpoint voltage band. Current differences are relative to the
measured near-cutoff current. The paired prediction table requires matching
SOC and pre-pulse voltage and uses the strict-fold calibration from the first
run. Both tables are retrospective.

RPCWBY Test 1 and Test 2 are two physical Samsung 30T cells. The raw source
overlap record lists their different recording channels and time spans. The
Test 3 2024 campaign also overlaps both cells on another recorder. Test 3
uses the protocol-aligned final constant-power records. Test 3 lacks the
active-history blocks expected by the original model. The exploratory
full-prefix variant uses the recorded blocks preceding the pulse and has
separate counts by available history length. Its measured endpoints use the
Test 3 protocol. The 2 s variant uses factors calibrated on 10 s external
cases and is a cross-horizon probe.

State replay uses a drive run that ended before each reference
characterization, estimated SOC and SOH in drive features, and cached
estimated SOC at the HPPC decision point. The two replay segments have
separate OCV initialization. The replay used existing measured data and fixed
strict-fold factors.

Timing results use DWT cycles on the connected 250 MHz Cortex-M33. The
timing data include the actual branch and a padded path forcing four times
24 inverse-solver iterations per cell. The maximum is empirical across the
tested vectors and cache states. Each pack-size trial processes instrumented
state instances; it does not involve a physical series pack. Flash readback
matches the pre-test image across both MiB.
