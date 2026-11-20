RC-SOP Reviewer 1 experiment outputs

Source checkout 23b5c8b626637ef23145aaf80d5b723a553a35f1.
The published baseline regenerated 27 tables with zero differences.
The dated isolated checkout and checkpoints remain on the SSH server.

reference contains HPPC inverse-regression sensitivity and held-out pulse diagnostics.
external contains existing RPCWBY recordings, repeated model realizations on one cell.
coupled_state contains offline HPPC replay. Previous-available SOH, resting-OCV
initialization, estimated SOC and features from past samples at a 1 s grid.
The comparison uses original fixed factors on shared baseline pulse keys.
statistics contains the original and strict main comparison with paired cell-level bootstrap.
strict_inner/methods contains strict primary and secondary methods under the original
comparison policy of 0 A discharge and 0.5 A charge. matched_pairs.csv
reports pairwise effects from common pulses and a cell-level bootstrap.
mcu contains the firmware benchmark using a conservative hull mask and a zero-current
return on a flagged solver failure. Measurements are single-board DWT
samples with interrupts disabled. Stack watermark is cumulative. The target
board firmware was restored and the 2 MiB flash readback was byte-verified.
No new cell data or independently established WCET was produced.
