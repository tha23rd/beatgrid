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

## Review reasons (2026-10-09, 0.0.2)

`research/experiments/review_flags.py` runs the pipeline on the DJ-corrected (`edm`),
ear-approved and machine-fitted (`edm`) and Raveform (default) references, calls a grid
**wrong** when it is not usable@10 against its reference, and sorts each wrong grid into the
first kind that fits: *variable tempo* (the reference's beats are not one constant tempo:
transition edits, mashups, wobbly hand-made grids), *octave*, *tempo drift* (right octave,
but the tempos differ by enough to walk more than 10 ms over the music), *phase offset* (same
tempo, beats off by a constant amount) or *bar 1*. **Recall** is the share of wrong grids
given any review reason, **false flags** the share of usable grids given one anyway. Thresholds
were tuned on dev; the held-out test split was scored once with the dev-tuned set.

| | dev 0.0.1 | dev 0.0.2 | test 0.0.1 | test 0.0.2 |
|---|---|---|---|---|
| recall (wrong grids) | 46% (28/61) | 52% (32/61) | 38% (9/24) | 38% (9/24) |
| false flags (usable grids) | 23% (44/190) | 24% (45/190) | 21% (18/86) | 22% (19/86) |
| variable tempo | 9/15 | 9/15 | 1/4 | 1/4 |
| octave | 1/3 | 3/3 | 0/1 | 0/1 |
| tempo drift | 5/10 | 5/10 | 1/4 | 1/4 |
| phase offset | 8/25 | 9/25 | 6/13 | 6/13 |
| bar 1 | 5/8 | 6/8 | 1/2 | 1/2 |

On Engine's own grids for tracks a DJ played unedited (108, all splits, never used for tuning)
both versions catch 45% (14/31) at 22% (17/77) false flags. At 20 ms instead of 10 the picture
is the same (test: 40% recall at 22-23% false flags, both versions). The earlier figure of 65%
caught at 29% false flags was measured with an older network and a looser label (usable@20,
octave errors forgiven).

**What was tried, and why 0.0.2 keeps only two changes.** Each change alone on top of 0.0.1's
reasons (extra wrong grids caught / extra usable grids flagged), and all of them together as
tuned on dev:

| change | dev | test | Engine accepted |
|---|---|---|---|
| **kept:** "no clear attack" also when fewer than 3 of the track's 8 stretches show an attack under the grid | +1 / +0 | +0 / +0 | +0 / +0 |
| **kept:** "octave chosen by the profile" when the profile moved the tempo off the audio's octave | +3 / +1 | +0 / +1 | +0 / +0 |
| a second attack at least 40% of the first, more than 5 ms from it (phase) | +5 / +6 | +2 / +5 | +2 / +1 |
| the grid walks more than 10 ms off the attacks end to end (self-audit over 8 stretches, or exact vs measured tempo) (drift) | +1 / +1 | +0 / +1 | +2 / +2 |
| bar-1 pooled margin below 0.5 | +2 / +0 | +0 / +2 | +0 / +0 |
| "the downbeat cues disagree" only when the margin is also below 4 | −2 / −18 | −4 / −3 | −8 / −12 |
| all of the above, tuned on dev: recall / false flags | 57% / 18% | 29% / 27% | 29% / 10% |

On dev the full set looked like 46% → 57% recall at 23% → 18% false flags; on the held-out
test it fell to 29% recall at 27% false flags. The margin rule carried the dev gain, and it
did not replicate: the disagreement flags it drops were right about as often on test and on
Engine's grids as the other reasons are. The added reasons flag wrong grids at about the
precision the existing ones do (roughly 40% pooled over the three splits, against a 25%
base rate), so they lengthen the listening queue rather than sharpen it. 0.0.2 keeps the two
changes that lose no wrong grid on any split and are more precise than the existing reasons
(4 catches for 2 false flags pooled). That choice was made after the test run, so the 0.0.2
test column above is no longer a clean held-out estimate; the clean one is the dev-tuned
set's, which failed. The measurements behind the dropped reasons stay in the diagnostics
(`attack_second`, `stretch_drift_ms`, `clear_stretches`, `snap_drift_ms`, `bar_margin`) for
tools that want to rank their own queue.

Signals that did not separate wrong from right grids even on dev, on top of the 0.0.2 reasons:
octave evidence from the network (half time plausible: alternate beats below 0.8 of each
other above 140 BPM, +0 / +6; double time plausible: off-beat activation above 0.4 below
100 BPM, +0 / +0), a broad attack (fold over 8 ms wide, +1 / +4), and the network's beat phase
far from the attack (+1 / +10).

**Most of what the reasons miss is on the reference's side.** Of the 44 wrong grids (dev and
test) that no reason flags:

- 11 are Raveform tracks whose beats sit a constant 25-45 ms from the reference: the offset of
  the YouTube downloads noted above, with beatgrid on the attack;
- all 8 unflagged tempo drifts are tracks where beatgrid's own grid stays within 2 ms of the
  attacks from one end to the other; the audit finds 6 of those references drifting on their
  own audio, cannot check one (a DJ grid at 149.5 BPM on a track the audio puts at 150) and
  calls one steady;
- 9 are references that are not one constant tempo where beatgrid's single grid stays within
  3 ms of the attacks end to end;
- 12 are other phase offsets: 5 DJ grids a constant 11-25 ms from the attack beatgrid uses
  (where on the attack a DJ puts the line is a convention) and 7 Raveform grids that wobble
  around beatgrid's;
- 3 are bar 1, and one is an octave count: a track the DJ counted at half of the audio's
  pulse, where on dev the same DJ counted six tracks with a similar half-time feel at full
  tempo (which is why a half-time flag would have been wrong six times for each catch).

## Tempo changes (2026-10-10, 0.1.0)

0.1.0 can grid a track whose tempo changes as constant-tempo segments (`--tempo auto` or
`variable`; the method is step 6 of the README's "How it works", the code
`research/beatgrid_lab/sections.py` and its port `crates/beatgrid/src/segments.rs`). The
default stays one tempo per track. `auto` is meant to be safe on any library: it splits only
on a clear tempo change between sections and leaves steady tracks alone. `variable` looks
harder (a lower cost per segment, shorter segments, weaker evidence) and also follows a phase
jump at the same tempo, for tracks known to change.

**References.**

- *Synthetic transition edits* (`lab splices`, `research/beatgrid_lab/splices.py`). Sections of
  326 steady references (a grid that is one tempo, holds steady on its own audio under `lab
  audit`, and sits on the attack: Engine, crate and Raveform tracks) are cut on their
  downbeats and joined: two sections (60%) or three, 16-64 bars each. Joins by kind: *large*
  (a 6-39% tempo change, 124 -> 150), *medium* (1.5-6%), *small* (0.2-1.5%, the next track
  resampled to the ratio), *jump* (the same tempo, the next section cut a fraction of a beat
  late) and *same* (the same tempo and phase with bar lines continuing: one grid, which must
  not be split). Each join is a cut (a 20 ms crossfade) or a one-bar crossfade centred on the
  change. The true grid is the source references' beats moved to where they land; a change is
  where the next section's first beat lands. 300 dev splices are built from dev sources and
  150 test splices from test sources. A splice is a recipe: its audio is rebuilt from the
  source files when needed, and only features computed from it are cached.
- *The DJ's grids with several tempo markers*: the 7 DJ-corrected Engine grids with more than
  two markers, and one ear-approved grid of a transition edit (116.7 -> 160).
- *Steady sets*: DJ-corrected and machine-fitted (`edm`), ear-approved (`edm`) and Raveform,
  as in the accuracy table. A segmented grid on a reference that is one tempo throughout
  counts as a false segmentation. Engine's own analysis on tracks played unedited (108) is an
  extra check that was never used for tuning.

**Metrics.** usable@10/@20 over the whole track: every section at the right octave, at least
90% of the reference's beats within 10/20 ms, and bar 1 right; and the same per section.
A change point is found when a detected change lies within one bar of a true one (matched one
to one): recall over true changes, precision over detected ones. Each section's tempo is the
detected tempo at the section's middle against the truth.

**Synthetic transition edits.** The test split was scored once, with the settings tuned on
dev (dev in brackets):

| 150 test splices (dev: 300) | constant | auto | variable |
|---|---|---|---|
| usable@10, whole track | 15% (11%) | 72% (72%) | 85% (84%) |
| usable@20, whole track | 16% (11%) | 73% (72%) | 85% (85%) |
| sections usable@10 | 36% (33%) | 75% (78%) | 84% (86%) |
| true changes found within a bar (recall) | 0% | 63% (64%) | 77% (77%) |
| detected changes that are true (precision) | — | 81% (81%) | 71% (74%) |
| sections whose tempo is within 0.01 BPM of the truth | 64% (57%) | 94% (94%) | 97% (97%) |
| one-grid splices split | — | 0/23 (0/38) | 0/23 (0/38) |

| test, usable@10 by kind | splices | constant | auto | variable |
|---|---|---|---|---|
| large (6-39%) | 49 | 0% | 73% | 76% |
| medium (1.5-6%) | 26 | 0% | 88% | 88% |
| small (0.2-1.5%) | 34 | 0% | 76% | 79% |
| jump (same tempo, phase jumps) | 18 | 0% | 0% (by design) | 94% |
| same (one grid) | 23 | 100% | 100% | 100% |
| cut / one-bar crossfade | 98 / 52 | 18% / 10% | 77% / 63% | 85% / 85% |

Of the true changes in splices that were segmented, the change lands on the right beat 69% of
the time (test, auto), within a bar 84% and within two bars 89%. The misses are mostly whole
bars off, either way: where a section ends or starts with a breakdown, or two overlap in a
crossfade, no attack tells the old grid from the new one for a few bars, and the change can
land anywhere in that stretch. The main failures of `auto` on dev: the gate rejecting a
right segmentation because the new section's attacks are weak (31 of 84), phase jumps (17,
which `auto` never follows), a missed, misplaced or merged change (15), and a wrong bar 1 in
one section (11).

**Steady sets: no regression.** With `auto`, no reference that is one tempo is segmented
(0 of 327, dev and test), so on every one of them the grid (tempo, beats, bar 1) is the
constant one, and usable@10 and usable@20 are unchanged on every set. The only grids `auto`
changed are three of the DJ's grids with several markers (one now has bar 1 right).
`constant` is the 0.0.2 code path and scores exactly as before.

| dev + test | tracks (one-tempo refs) | usable@10 constant / auto / variable | false segmentations auto / variable |
|---|---|---|---|
| DJ-corrected (`edm`) | 66 (60) | 76% / 76% / 74% | 0 / 4 |
| ear-approved (`edm`) | 26 (25) | 73% / 73% / 69% | 0 / 1 |
| machine-fitted (`edm`) | 51 (51) | 84% / 84% / 84% | 0 / 0 |
| Raveform | 218 (191) | 75% / 75% / 75% | 0 / 5 |
| Engine's analysis, played unedited (never tuned on) | 108 (108) | 71% / 70% / 69% | 7 / 9 |

On Engine's own grids `auto` segments 7 of 108. Six of them were already not usable at one
tempo against Engine's grid: a 22-minute track at 128, 150 and 135 BPM, and tracks that go
148 -> 125 or 150 -> 180, which Engine's analysis (always one tempo) cannot follow. One
(136 -> 174 for the last 80 s) turns a grid that matched Engine's into one that does not;
there the segmented grid's beats sit on attacks 3-5 times the median flux against 1.6 for
Engine's.
Either the track changes and Engine's grid does not follow it, or a section's strongest
attack is not the beat: these need ears. The same holds for the few steady references that
`variable` splits (and that `auto`'s gate rejects at the margin): on most of them the
network's own beat activation also favours the segmented grid on the beats where it differs
(0.78-0.95 against 0-0.35), so the activation could not tell them from true changes either.

**The DJ's grids with several markers.** Against these grids no setting is usable@10 on more
than the three that hold one tempo, but the grids themselves are approximate: on every one of
the eight, beatgrid's beats sit on stronger attacks than the DJ's (mean flux at the beats over
its median, DJ / beatgrid's grid).

| the DJ's grid | `auto` (`variable` where different) | attacks DJ / beatgrid |
|---|---|---|
| transition edit, 124 -> 150 (a beatless stretch between) | 124.01 -> 150; change where the 150 BPM beats start, 27 s after the DJ's marker where the 124 BPM beats stop. Both DJ sections drift 30-170 ms against the audio | 1.3 / 5.0 |
| mashup, 128 -> 94 -> 92 -> 125 | 128 -> 125, 20 s after the DJ's change; the slow middle section is missed | 2.9 / 4.4 |
| 125.6 -> 126.4 | 126 -> 146 -> 126: a middle section the DJ's grid does not follow (the DJ's first section drifts 70 ms in 23 s) | 1.4 / 2.8 |
| 116.7 -> 160 (ear-approved) | one tempo: the 160 BPM part has weak attacks (2.1 against the gate's 2.5); `variable`: 116.7 -> 160, 25 s before the DJ's change in a beatless stretch | 8.9 / 14.3 |
| 126.25 -> 126.0 | one tempo, 126; `variable`: a phase jump at 65 s | 1.4 / 2.7 |
| 150 with a 0.4% step near the end | one tempo, 150; usable@10 | 2.1 / 2.8 |
| 127, followed by hand (±0.3 BPM) | one tempo, 127; usable@10 | 4.5 / 5.5 |
| 128 with two steps of under 0.01 BPM | one tempo, 128; usable@10 | 3.3 / 4.2 |

**Raveform grids that are not one tempo** (15, mostly Rekordbox grids that follow a drifting
recording beat by beat): `auto` segments one and `variable` three; usable@10 is 1 of 15 in
every mode (usable@20 5, 5 and 6), as one constant tempo per section cannot follow a drift.

**Rust and Python.** Given the same activations, the Rust core gives the same segments as the
research pipeline on 937 of 942 tracks with `auto` and 935 of 942 with `variable` (the
splices, steady, real, Raveform and Engine sets, dev and test): as many segments, each at the
same tempo and starting within 0.1 ms. The others differ by under 0.002 BPM in a tempo that
was not snapped, by a near-tie in which downbeat a change goes on, or in whether a short
segment is kept; every score above is the same within one section either way.
`research/experiments/rust_parity.py --tempo auto` checks it end to end (174 of 174 Engine
tracks the same, review reasons included).

**What was tried.**

| idea | result |
|---|---|
| each 16 s window's tempo on its own, then grouping (the first experiment) | tempos too noisy to segment on |
| a Viterbi tempo path, then refitting each section (the second) | found the change on edits, but the refit could lock onto a neighbouring tempo; kept as stage 1, with the segments now drawn from beat anchors, which also see small changes and phase jumps |
| evidence over the median flux >= 2.2 with the attacks twice the constant grid's | dev 68% against 72%, no fewer false segmentations |
| segments down to 8 bars; a cost of 16 per segment; 32 differing beats | each within one splice of the defaults on dev |
| the network's beat activation as a second vote in the gate | it favours the segments on the steady references' candidates as much as on true changes |
| same-tempo phase jumps in `auto` | the main source of false segmentations on steady tracks (a section whose strongest attack sits a fraction of a beat off, an off-beat in a breakdown): `variable` only |

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
- Tempo changes (0.1.0, above): `auto` misses changes into sections with weak attacks, never
  follows a phase jump, and puts the change at the end of a beatless gap's old side. Gradual
  drift (live playing, old recordings) and tempo ramps are not modelled: a constant tempo per
  segment cannot follow them. Bar 1 is decided per segment, so a short segment's bar 1 is
  weaker. Whether the changes `variable` finds on references marked steady are real needs
  ears.
- Half-time drum & bass bar phase in the `edm` profile is unresolved.
- A few ear-approved grids sit 12-21 ms before the attack; ear approval cannot tell.
