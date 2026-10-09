# Datasets

Surveyed 2026-10-08. Every public set below is scored by its authors with ±70 ms windows;
none certifies millisecond phase. Labels are only valid against the exact audio they were
made on: re-sourced audio (YouTube, Deezer, a different master) needs a per-track offset
check, and codecs add start delays (AAC ≈48 ms / 2112 samples, MP3 ≈25 ms / 576+529).

## For evaluation on dance music

| set | size | what | labels made by | audio | license |
|---|---|---|---|---|---|
| **Raveform** (annotated subset) | 1,423 tracks, ~161 h | beats, downbeats, Rekordbox tempo markers, sections, 8 folds | Rekordbox analysis corrected by 3 experts (577 of 2,000 tracks dropped as unfixable) | YouTube IDs | CC BY 4.0 labels |
| Harmonix Set | 912 tracks (14% Dance/Electronic, much dance-pop) | beats, downbeats, sections | DAW tempo maps (game charts) | YouTube URLs, MusicBrainz, AcoustID | MIT labels |
| EDM-CUE | 4,710 tracks (~90% drum & bass) | one constant grid per track, cue points | 4 pro DJs' Rekordbox libraries (correction unknown) | Deezer IDs → ISRC (buy) | MIT |
| GiantSteps Tempo | 664 two-minute previews | tempo only | Beatport forum corrections / crowd tapping | download script | CC BY-SA 4.0 / CC BY 4.0 |
| Mixotic | 723 CC techno/house tracks | none (boundaries only) | — | included, redistributable | CC per track |

Raveform notes: 53% of tracks have a single tempo marker, 91% of those at a whole-number
BPM; grid start stored to 1 ms and BPM to 0.01 (0.01 BPM drifts ~30 ms over 7 minutes), so
treat it as ±5–20 ms quality. Published ±70 ms scores: their model beat F1 0.991 / downbeat
0.965; madmom 0.947 / 0.669.

Harmonix notes: use the original repo's annotations against YouTube audio, not the Beat
This! copies (those were re-aligned to different audio; median shift 38.5 ms on 40 tracks).

## For training

- Raveform training folds; its 63k automatically-beat-tracked tracks as noisy pseudo-labels.
- Beat This! spectrograms (Zenodo 13922116, CC BY 4.0): Harmonix (45 GB) and HJDB (2.7 GB)
  are the dance-relevant parts; ASAP and Candombe are non-commercial.
- Synthetic exact grids: Freesound Loop Dataset (filter to CC0/CC BY), Groove MIDI (CC BY 4.0).

## The gold set we should build

No public set is verified at ±5 ms. The Mixotic tracks are redistributable, so a
hand-verified ±5 ms grid set on them would be the first public DJ-grade benchmark with audio.

## Sources

- Raveform: https://mir-aidj.github.io/raveform/ · https://transactions.ismir.net/articles/10.5334/tismir.288 · https://huggingface.co/datasets/taejunkim/raveform
- Harmonix: https://github.com/urinieto/harmonixset · https://archives.ismir.net/ismir2019/paper/000068.pdf
- EDM-CUE: https://huggingface.co/datasets/disco-eth/edm-cue · https://arxiv.org/abs/2407.06823
- Beat This!: https://github.com/CPJKU/beat_this · https://github.com/CPJKU/beat_this_annotations · https://zenodo.org/records/13922116 · https://arxiv.org/abs/2407.21658
- GiantSteps Tempo: https://github.com/GiantSteps/giantsteps-tempo-dataset · https://zenodo.org/records/3553625
- Mixotic: http://www.cp.jku.at/datasets/fingerprinting/
- Davies & Böck 2014 (tolerance windows): https://archives.ismir.net/ismir2014/paper/000238.pdf
- Codec delays: https://developer.apple.com/library/mac/technotes/tn2258/_index.html
