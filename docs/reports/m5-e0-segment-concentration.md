# M5 E0 segment concentration analysis

## Scope

Does the TabPFN-minus-tree learner gap concentrate in a few anomaly segments or
episodes, or is it diffuse? Pinned HEAD
`d8e59da2c40cb5102367d6a73299e807680f6ca6`, execution mode `FORMAL_E0`.

Only existing row, segment and prediction artifacts are used. **No rows are
resegmented in E0** — segment identity is E1's frozen definition
(`m5_context_mechanism_137/m5_137_anomaly_segments.parquet` and the matching
segment-phase artifact). No fitting, no inference. `sensitivity_draws` is left
at the committed default of 1,000 and was not tuned.

## Coverage

Complete per meter; no segment sampling.

| Meter | Segments | Buildings | Segment rows |
| --- | ---: | ---: | ---: |
| electricity | 7,999 | 532 | 356,679 |
| chilledwater | 2,604 | 208 | 141,139 |
| steam | 1,921 | 150 | 48,888 |
| hotwater | 810 | 65 | 90,691 |

41,337 segment records across four per-meter atomic checkpoints. Phase
`COMPLETE.json` issued after a census reported 0 missing units.

## Concentration of absolute learner-gap score movement

`contribution_fraction` is the share of total absolute learner-gap score
movement held by the top-N segments.

| Meter | top-1 | top-5 | top-10 |
| --- | ---: | ---: | ---: |
| electricity | +0.0041 | −0.0039 | **−0.0074** |
| chilledwater | +0.0140 | +0.0136 | **+0.0219** |
| steam | +0.0096 | +0.0104 | **+0.0335** |
| hotwater | +0.0082 | +0.0377 | **+0.0571** |

**No meter's learner gap is concentrated in a few segments.** The ten most
extreme segments account for at most 5.7% of total absolute movement, and for
electricity the top-10 contribution is slightly negative — the extreme segments
there pull *against* the aggregate direction rather than creating it.

Building concentration within the top-10 segments (share coming from the single
most frequent building) is 0.2 for electricity, 0.5 for chilledwater, 0.4 for
steam and 0.3 for hotwater, so even the extreme tail is not one building's
episodes.

## Mean learner gap by episode phase

| Meter | onset | middle | recovery |
| --- | ---: | ---: | ---: |
| electricity | +0.0242 | +0.0023 | −0.0106 |
| chilledwater | +0.0350 | +0.0192 | +0.0196 |
| steam | +0.0478 | +0.0471 | +0.0245 |
| hotwater | +0.1061 | +0.0618 | +0.0821 |

Onset is the largest phase for every meter. This is a descriptive pattern in
the frozen segmentation only. **It is not evidence of a mechanism and is not
interpreted causally here** — E0 does not test why onset differs, and no segment
definition was adjusted to produce it.

## What this establishes for E0

- The steam and chilledwater advantages seen in the bootstrap are **diffuse
  across thousands of segments**, not artefacts of a handful of anomaly
  episodes. This removes one specific alternative explanation.
- Electricity's negative gap is likewise not produced by extreme segments.
- Hotwater has the highest top-10 concentration (5.7%) and the fewest segments
  (810), consistent with it being the least stable meter elsewhere in E0, but
  5.7% is still far from concentration.

Artifacts: `data/processed/m5_meter_specific_learner_gap/formal/
formal_segment_concentration_summary.{json,csv}`.

## Environment deviation (recorded)

`segment_concentration()` reads Parquet, and **`pyarrow` is not in the pinned
`uv.lock`**, so the execution clone's locked venv cannot read the frozen E1
artifacts. This phase was therefore run with the main repository venv, which has
`pyarrow` 23.0.1.

The deviation is limited and recorded deliberately rather than resolved by
installing a new dependency into the pinned formal environment:

| | execution clone | main repo venv |
| --- | --- | --- |
| Python | 3.11.9 | 3.13.13 |
| pandas | 3.0.3 | 3.0.3 |
| numpy | 2.4.6 | 2.4.6 |
| scikit-learn | 1.8.0 | 1.8.0 |
| pyarrow | absent | 23.0.1 |

The numeric stack that produces every value is identical; `pyarrow` only decodes
Parquet, which is lossless for float64. This phase performs no NPZ access and no
model computation. Adding `pyarrow` to the pinned lock would have altered the
formal execution environment mid-run, which is the larger risk.

## Not done

No resegmentation, no adjustment of segment identity or the concentration
metric, no causal or mechanistic claim, and no unlock of E1/C1 work.
