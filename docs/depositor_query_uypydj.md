# Query to the UYPYDJ depositors — draft

Two things about doi:10.5683/SP3/UYPYDJ need the depositors to settle them, and
neither can be resolved from the outside. Both block a journal submission that
uses this dataset, so they are worth sending before the manuscript is written
rather than at proof stage.

The draft below is ready to send. Verify the recipient addresses first; the
Borealis "Contact Owner" button routes to the depositors without needing them.

---

**Subject:** Licence and attribution for doi:10.5683/SP3/UYPYDJ (Samsung 30T
15-minute fast-charge aging dataset)

Dear Dr Kollmeyer, Dr Duque and Dr Naguib,

I am preparing a journal submission that uses your dataset *Battery Aging
Dataset for 15 Minute Fast Charging of Samsung 30T Cells*
(doi:10.5683/SP3/UYPYDJ) for model training, calibration and internal
evaluation, and I have two questions I cannot answer from the deposit itself.

**1. The licence is stated two different ways.**

The Borealis record for the dataset (version 2.0, last updated 2025-02-19)
gives the License/Data Use Agreement as **CC BY-SA 4.0**. The readme inside the
same deposit — `00-Readme_2025-02-17_Duque_Fifteen Minutes Fast Charge Aging
Dataset.txt`, section 2 METADATA, line 47 — reads **"Licenses/restrictions: CC
BY 4.0"**.

Which of the two governs? The difference matters to me in a specific way. I
redistribute derived material: parameter tables extracted from the
characterisation tests (tens of thousands of rows) and model weights trained on
the drive cycles. Under ShareAlike those are adapted material and must carry
CC BY-SA 4.0, which no major publisher offers for article figures — so if
BY-SA governs I need either your permission for the specific use or a different
way of presenting the results. If the readme governs, the question disappears.
For now I have applied the more restrictive reading.

**2. Attribution.**

The Borealis citation credits three authors — Duque, Josimar; Kollmeyer,
Phillip J.; Naguib, Mina — and I have corrected my records to match. Please
confirm this is the form you would like cited, and whether you would also like
the related publication cited alongside the data:

> J. Duque, P. J. Kollmeyer, M. Naguib and A. Emadi, "Battery Dual Extended
> Kalman Filter State of Charge and Health Estimation Strategy for Traction
> Applications," 2022 IEEE ITEC, pp. 975–980, doi:10.1109/ITEC53557.2022.9813961

I intend to cite it as prior work regardless — it is SOC and SOH estimation on
this dataset, and the characterisation cadence in the campaign makes much more
sense read alongside it.

Thank you for making the data available; the six-cell design with one charge
protocol per cell is what makes a leave-one-cell-out evaluation possible at
all.

With thanks,
[name, affiliation, contact]

---

## What to record when they reply

Update all four of these together — `tests/test_licence_consistency.py` fails
if they drift:

  - `manifests/raw_data.yaml`, the UYPYDJ `license` and `license_conflict`
    fields
  - `LICENSE-DATA`, the header and the attribution block
  - `README.md`, the Licence section
  - `.paper_state/evidence_ledger.yaml`, `licence.status` — it stays
    `CONFLICT_UNRESOLVED_UPSTREAM` until they answer

If they confirm CC BY 4.0, LICENSE-DATA can go back to CC BY 4.0 and the
publisher conflict is gone. If they confirm CC BY-SA 4.0, ask in the same reply
whether they will permit the figures under the publisher's licence; that is the
one thing that unblocks submission without redesigning what goes in the paper.
