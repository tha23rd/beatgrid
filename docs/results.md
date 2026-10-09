# Results and what was learned

What has been measured, against which references, and the things that were tried and did
not work. Numbers are `usable@10` unless stated: right tempo octave, ≥90% of beats within
10 ms of the reference, bar 1 on the reference's bar 1.

**Where the numbers come from.** Results were measured against three kinds of reference:

- (a) a DJ's private Engine DJ library: 174 tracks with grids the DJ corrected (66) or
  played as Engine analysed them (108), and a whole-library comparison with Engine's own
  analysis;
- (b) grids a DJ approved by ear (26), with 51 machine-fitted grids that agreed with Engine
  alongside them;
- (c) Raveform (Kim, Kim, Kim & Nam, TISMIR 2026; labels CC BY 4.0): 218 tracks with
  expert-corrected grids and audio.

The private references (a, b) and their audio cannot be redistributed, so those numbers
cannot be reproduced from this repository; Raveform's can, from its published labels and
YouTube IDs ([datasets.md](datasets.md)).

## Reference sets

| set | tracks | what the reference is | trust |
|---|---|---|---|
| `engine` corrected | 66 | grids a DJ edited in Engine DJ | checked by ear while fixing; some still drift (audit) |
| `engine` accepted | 108 | Engine's own analysis on tracks the DJ played without editing | unverified |
| `crate` approved | 26 | grids a DJ listened to and accepted | best available, but ear approval tolerates ~20 ms placement |
| `crate` trusted | 51 | an earlier grid-fitting tool's grids where they agreed with Engine | unheard, partly Engine-derived |
| `raveform` | 218 | Rekordbox grids corrected by three experts (TISMIR 2026), audio from YouTube | stored to 1 ms / 0.01 BPM; AAC downloads sit ~28 ms off; some uploads may be other editions |

`lab audit` checks every reference against its own audio (does the attack stay put across
the track?); 23 of 174 Engine grids and 13 of 218 Raveform grids drift measurably.

## Where the pipeline is (2026-10-09: supervised student, network+changes+fills bar votes)

| | usable@10 | @20 | bar 1 | Engine DJ's own analysis |
|---|---|---|---|---|
| engine corrected (66) | 73% (76% with `edm`) | 74% | 93% | 27% |
| engine accepted (108) | 71% | 74% | 83% | (is the reference) |
| crate approved (26), `edm` | 73% | 81% | 96% | 42% |
| crate trusted (51), `edm` | 84% | 92% | 96% | 76% |
| raveform (218) | 75% | 84% | 96% (tempo 100%) | — |

The network is a 529k-parameter TCN distilled from Beat This! on 1,813 tracks and supervised
on Raveform's expert downbeats where those match the audio (1,001 tracks); its downbeats
alone pick bar 1 on 91% of Raveform against 87% for Beat This! ([model.md](model.md)). It is
level with or ahead of Beat This! on every set, and the whole Rust pipeline runs ~0.5 s per
track in batch on a 2019 laptop CPU.

| network in the pipeline | engine corr. | engine acc. | crate appr. | crate trusted | raveform |
|---|---|---|---|---|---|
| Beat This! final0 (~6 s/track) | 71% | 69% | 69% | 82% | 74% |
| student v1 (144k, distilled) | 71% | 69% | 69% | 80% | 71% |
| student v4 (529k, + expert labels) | 73% | 71% | 73% | 84% | 75% |

## A whole library against Engine DJ (2026-10-08)

This comparison was run by a separate tool built on the beatgrid library that reads the DJ
application's database; that tool is not part of this repository. It covered a DJ's
3,305-track Engine library (Rust core, distilled network, `edm` profile; 66 DJ-edited grids
and 16 mixes skipped): 3,222 tracks in 36 min on a 2019 laptop.

| | tracks |
|---|---|
| agrees with Engine (tempo, beats within 10 ms, bar 1) | 2,387 (74%) |
| tempo differs | 264: 101 by < 0.1 BPM, 52 by an octave (convention), 111 more |
| beats > 10 ms apart at the same tempo | 244 (77 about half a beat) |
| bar 1 differs | 319 (124 by one beat, 133 by two, 62 by three) |

Where the audio can settle it, it does: on 18 of 20 sampled small-tempo disagreements Engine's
grid drifts against the attacks (beatgrid sits on an exact whole/half BPM in 93 of the 101;
Engine in 4). Same-tempo beat placement agrees to a median of −1.2 ms across all formats, so
both put grid lines in the same place. A 120-track stratified sample of the disagreements was
loaded into the listening UI as a reference set named `library`, for ears to settle the rest.

## Things that worked

- **Placing the grid on the folded sub-millisecond spectral flux peak** (+2 ms): within ±5 ms
  of DJ grids on 83% of tracks versus 37% for the network's phase. The same offset lands at
  −1.3 ms on Raveform, so it is a property of where DJs put grid lines, not of one library.
- **Tempo from the flux fold over a whole beat period**, snapped to an exact whole/half BPM
  when the drift is inaudible or the audio agrees: tempo octave-correct on 99% of Raveform.
- **Bar 1 from several weak cues pooled, not one strong one.** With Beat This!'s downbeats,
  the network's downbeat fold pooled 2:1 with arrangement changes (the 12 biggest
  beat-to-beat steps in loudness and brightness, by beat position) took bar 1 from 86-89% to
  95% on Raveform's held-out half. With the supervised student (88-93% alone), pooled 2:1:1
  with arrangement changes and high-frequency bursts on the one: 95.5-96.7% on the held-out
  and dev halves.

## Things that did not

| idea | result |
|---|---|
| placing on the kick's low-frequency body | IQR 21 ms vs 3 ms for the broadband click |
| ONNX Runtime / int8 for Beat This! | slower than PyTorch (its attention is not fused once traced) |
| backbeat vote (snare on 2 and 4) | 43% alone, chance at telling 1/3 from 2/4 |
| fills-before-the-one vote | 18% alone (worse than chance); read as bursts *on* the one, 39% alone and nothing added pooled with Beat This!'s downbeats (kept, at weight 1, with the student's) |
| Engine's grid as a bar prior | right on only 68% of the tracks a DJ had to correct |
| bar-to-bar contrast over every bar line | diluted; the few biggest changes carry the information |
| earliest substantial fold peak instead of the highest | fixed one ear-approved track, lost 10 points on the Engine set |

## Known gaps

- **MP3s whose encoder delay ffmpeg can only guess (529 samples) sit on a different timeline in
  some Engine analyses.** On the 114 such tracks, beatgrid − Engine splits into three groups
  (~+10 ms: 52, ~0: 38, ~−10 ms: 20); headers do not predict which, so this is probably
  Engine's decoder changing between versions. A beatgrid grid imported into Engine DJ for one
  of these tracks could land ~10 ms off on the player; settling it takes loading a few such
  tracks on the player and listening.

- Octave is a convention (drum & bass at 87 or 174, dubstep at 70 or 140): use a profile.
- Mashups and transition edits that change tempo (124 -> 150) need multi-section grids. Until
  then they get one tempo and the review reason "tempo may change". The experiment
  (research/beatgrid_lab/sections.py, not wired in): 16 s windows fitted independently give
  tempos too noisy to segment on; decoded jointly (Viterbi over tempo states) they no longer
  split steady tracks and do find the change on transition edits, but the per-section tempo
  refit can lock to the wrong tempo, and the references disagree with the audio about where
  the change is.
- Half-time drum & bass bar phase in the `edm` profile is unresolved.
- A few ear-approved grids sit 12-21 ms before the attack; ear approval cannot tell.
