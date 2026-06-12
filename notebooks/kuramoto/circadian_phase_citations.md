# Circadian phase citations for `circadian_genes.json`

Each entry gives the primary-literature source for a gene's phase assignment in the 4-class partition
(morning / day / evening / night).  Where mRNA and protein phases differ, both are noted.

---

## Morning (ZT0–6)

### CCA1 — AT2G46830
**Assignment**: morning, ZT0–2 (mRNA)

Wang, Z.-Y. & Tobin, E.M. (1998). Constitutive expression of the CIRCADIAN CLOCK ASSOCIATED 1
(CCA1) gene and suppression of the circadian clock in Arabidopsis.
*Cell* **93**, 1207–1217. <https://doi.org/10.1016/S0092-8674(00)81464-6>

> "both CCA1 and LHY are expressed rhythmically with peaks of expression occurring soon after dawn"

### LHY — AT1G01060
**Assignment**: morning, ZT0–2 (mRNA)

Schaffer, R. et al. (1998). The late elongated hypocotyl mutation of Arabidopsis disrupts
circadian rhythms and the photoperiodic control of flowering.
*Cell* **93**, 1219–1229. <https://doi.org/10.1016/S0092-8674(00)81465-8>

> "LHY mRNA showed a circadian pattern of expression with a peak around dawn"

---

## Day (ZT6–12)

### PRR7 — AT5G02810
**Assignment**: day, ZT4–8 (mRNA; part of the PRR countdown wave)

Nakamichi, N. et al. (2010). PSEUDO-RESPONSE REGULATORS 9, 7, and 5 are transcriptional
repressors in the Arabidopsis circadian clock.
*Plant Cell* **22**, 594–605. PMC2861452. <https://doi.org/10.1105/tpc.109.072892>

> "PRR9 mRNA levels are greatest at dawn, PRR7 peaks in the morning, PRR5 around noon,
> and PRR3 and TOC1 in the evening"

### PRR5 — AT5G24470
**Assignment**: day, ZT8–10 (mRNA)

Same source as PRR7 above (Nakamichi 2010).

> "PRR5 around noon"

### PRR3 — AT5G60100
**Assignment**: day/evening boundary, ZT10–12 (mRNA)

Same source as PRR7 above (Nakamichi 2010).

> "PRR3 and TOC1 in the evening"

*Note*: Nakamichi 2010 places PRR3 together with TOC1 in the evening.  Placed in "day" here
because its mRNA peak (ZT10–12) precedes TOC1 (ZT12–14) and sits at the day/evening boundary.

### GI — AT1G22770
**Assignment**: day, ZT8–10 (mRNA)

Fowler, S. et al. (1999). GIGANTEA: a circadian clock-controlled gene that regulates
photoperiodic flowering in Arabidopsis and encodes a protein with several possible
membrane-spanning domains.
*EMBO J* **18**, 4679–4688. <https://doi.org/10.1093/emboj/18.17.4679>

> "a peak in transcript levels 8–10 hours after dawn"

### RVE8 — AT3G09600
**Assignment**: day, protein/EE-binding activity ZT3–8 (mRNA peaks at ZT0)

Rawat, R. et al. (2011). REVEILLE8 and PSEUDO-REPONSE REGULATOR5 form a negative feedback
loop within the Arabidopsis circadian clock.
*PLoS Genet* **7**, e1001350. PMC3069099. <https://doi.org/10.1371/journal.pgen.1001350>

> "peak levels of RVE8-HA protein occurred three to six hours after subjective dawn
> (ZT27–ZT30)"

> "RVE8 protein levels are high in the subjective afternoon whereas CCA1 and LHY proteins
> are difficult to detect at that time"

*Note*: RVE8 mRNA peaks at ZT0 (same phase as CCA1/LHY; the name *Reveille* reflects this
dawn-phased transcript).  However the protein is present in the afternoon (ZT3–6) and its
EE-binding DNA activity peaks at ZT8 (Hsu et al. 2013).  Classification as "day" follows
the functional/protein phase.

Hsu, P.Y. et al. (2013). Accurate timekeeping is controlled by a cycling activator in
Arabidopsis.
*eLife* **2**, e00473. PMC3639509. <https://doi.org/10.7554/eLife.00473>

> "plants mutant for RVE8 and its two closest homologs, RVE4 and RVE6, have lost the
> afternoon-phased EE-binding activity"

### RVE4 — AT5G02840
**Assignment**: day, protein/EE-binding activity ZT3–8 (same caveat as RVE8)

Same sources as RVE8 above (Rawat 2011; Hsu 2013).  RVE4 mRNA and protein timing are not
given explicit ZT values in these papers, but RVE4 is treated as functionally equivalent to
RVE8 throughout; the triple rve4 rve6 rve8 mutant loses "afternoon-phased EE-binding activity."

### LNK1 — AT5G64170 and LNK2 — AT3G54500
**Assignment**: day (early, ZT2–4 mRNA; broadly co-expressed with RVE4/RVE8)

Rugnone, M.L. et al. (2013). LNK genes integrate light and clock signaling networks at the
periphery of the Arabidopsis circadian clock.
*PNAS* **110**, 12120–12125. <https://doi.org/10.1073/pnas.1300901110>

Xie, Q. et al. (2014). LNK1 and LNK2 are transcriptional coactivators in the Arabidopsis
circadian oscillator.
*Plant Cell* **26**, 2843–2857. PMC4145118.

> "peaks occurring in early morning (1.5 to 2 h after subjective dawn)"  (Rugnone 2013)

> "Expression of LNK1 and LNK2 cycles in phase with RVE4 and RVE8 and peaks in mid morning"  (Xie 2014)

*Note*: LNK mRNA peaks very early (ZT1.5–2), which would place them in "morning" by mRNA
phase.  Placed in "day" here because they act as co-activators together with the RVE proteins
whose functional phase is mid-morning to afternoon.

### FKF1 — AT1G68050
**Assignment**: day, ~ZT8 (mRNA)

Nelson, D.C. et al. (2000). FKF1, a clock-controlled gene that regulates the transition to
flowering in Arabidopsis.
*Cell* **101**, 331–340. <https://doi.org/10.1016/S0092-8674(00)80842-9>

Cited via Schultz, T.F. et al. (2001), which quotes from the FKF1 paper:

> "transcripts for FKF1 cycle robustly, with peak levels occurring at approximately 8 hr
> after dawn"

---

## Evening (ZT12–18)

### TOC1 — AT5G61380
**Assignment**: evening, ZT12–14 (mRNA)

Strayer, C. et al. (2000). Cloning of the Arabidopsis clock gene TOC1, an autoregulatory
response regulator homolog.
*Science* **289**, 768–771. <https://doi.org/10.1126/science.289.5480.768>

Also Nakamichi et al. (2010), same reference as PRR7:

> "PRR3 and TOC1 in the evening"

### ELF3 — AT2G25930
**Assignment**: evening, ZT14–16 (mRNA)

Liu, X.L. et al. (2001). ELF3 encodes a circadian clock-regulated nuclear protein that
functions in an Arabidopsis PHYB signal transduction pathway.
*Plant Cell* **13**, 1293–1304. PMC135582. <https://doi.org/10.1105/tpc.13.6.1293>

> "ELF3 mRNA level is regulated in a cyclic manner, peaking at ~14 to 16 hr after sunrise
> regardless of daylength"

> "The maximal transcript level was observed ~16 hr after dawn in 12-hr-light/12-hr-dark
> cycles"

### ELF4 — AT2G40080
**Assignment**: evening, ~ZT12 (mRNA)

Doyle, M.R. et al. (2002). The ELF4 gene controls circadian rhythms and flowering time in
Arabidopsis thaliana.
*Nature* **419**, 74–77. <https://doi.org/10.1038/nature00954>

*Note*: The original Nature paper is paywalled.  Secondary sources and the Hsu 2013 / Liu 2001
papers consistently describe ELF4 as peaking at dusk (~ZT12), forming the evening complex
(EC) with ELF3 and LUX that accumulates at dusk to repress morning-clock genes.

### CHE — AT5G08330
**Assignment**: evening/day, ZT9–13 (protein); oscillates ~9 h out of phase with CCA1

Pruneda-Paz, J.L. et al. (2009). A functional genomics approach reveals CHE as a component
of the Arabidopsis circadian clock.
*Science* **323**, 1481–1485. <https://doi.org/10.1126/science.1167206>

Also: Kamioka, M. et al. (2016). Direct Repression of Evening Genes by CIRCADIAN CLOCK-ASSOCIATED1
in the Arabidopsis Circadian Clock.  *Plant Cell* **28**, 696–711. PMC4330950.

> "CHE mRNA levels oscillate 9 hours out of phase with CCA1 transcript"
> (CCA1 peaks ZT0 → CHE peaks ~ZT9)

> "Both proteins are nuclear localized … reaching maximum levels at similar times of the
> day (ZT 9 and ZT13, respectively)"  [CHE at ZT9, TOC1 at ZT13]

*Note*: CHE protein peaks at ZT9, earlier than the canonical evening genes.  Placed in
"evening" because it directly represses CCA1 and interacts with TOC1; ZT9 is at the day/
evening boundary.

---

## Night (ZT18–24)

### ZTL — AT5G57360
**Assignment**: night (protein stability cycle, not mRNA oscillation)

Somers, D.E. et al. (2000). ZEITLUPE encodes a novel clock-associated PAS protein from
Arabidopsis.
*Cell* **101**, 319–329. <https://doi.org/10.1016/S0092-8674(00)80841-7>

> "ZTL transcript levels are constitutive"

*Note*: ZTL mRNA does **not** oscillate.  The "night" assignment reflects its protein
stability: ZTL protein is stabilized by GI in the afternoon and then targets TOC1 for
degradation at night.  There is no mRNA-phase justification for "night."

Kim, W.-Y. et al. (2007). ZEITLUPE is a circadian photoreceptor stabilized by GIGANTEA in
blue light.
*Nature* **449**, 356–360. <https://doi.org/10.1038/nature06132>

> "ZTL protein levels are rhythmic with a peak in the late afternoon/early evening and
> trough at dawn, whereas ZTL mRNA levels are constitutive"

### LKP2 — AT2G18915
**Assignment**: night (protein; mRNA constitutive)

Schultz, T.F. et al. (2001). The ELF3 and GI proteins form a complex that interacts with
the promoter of the Arabidopsis circadian clock gene CCA1.
*Plant Cell* **13**, 2659–2670. PMC139480. <https://doi.org/10.1105/tpc.010354>

> "LKP2 is expressed at a low level and is not regulated by the circadian clock in
> Arabidopsis, similar to ZTL expression"

> "LKP2 transcript levels did not appear to change during the course of one LD cycle"

*Note*: Like ZTL, LKP2 mRNA is constitutive; the "night" assignment is an inference from
its structural similarity to ZTL (same F-box/LOV-domain family) and its role in protein
degradation at night.  LKP2 protein oscillation timing is not well characterised in the
primary literature.

---

## Summary table

| Gene  | AGI        | Class   | mRNA peak   | Protein peak | Basis for class |
|-------|------------|---------|-------------|--------------|-----------------|
| CCA1  | AT2G46830  | morning | ZT0–2       | —            | mRNA |
| LHY   | AT1G01060  | morning | ZT0–2       | —            | mRNA |
| PRR7  | AT5G02810  | day     | ZT4–8       | —            | mRNA |
| PRR5  | AT5G24470  | day     | ZT8–10      | —            | mRNA |
| PRR3  | AT5G60100  | day     | ZT10–12     | —            | mRNA (day/evening boundary) |
| GI    | AT1G22770  | day     | ZT8–10      | —            | mRNA |
| RVE8  | AT3G09600  | day     | ZT0 (dawn)  | ZT3–6        | protein/activity |
| RVE4  | AT5G02840  | day     | ZT0 (dawn)  | ZT3–8 est.   | protein/activity |
| LNK1  | AT5G64170  | day     | ZT1.5–2     | —            | functional role with RVEs |
| LNK2  | AT3G54500  | day     | ZT1.5–2     | —            | functional role with RVEs |
| FKF1  | AT1G68050  | day     | ~ZT8        | —            | mRNA |
| TOC1  | AT5G61380  | evening | ZT12–14     | —            | mRNA |
| ELF3  | AT2G25930  | evening | ZT14–16     | —            | mRNA |
| ELF4  | AT2G40080  | evening | ~ZT12       | —            | mRNA |
| CHE   | AT5G08330  | evening | ~ZT9        | ZT9          | protein; 9h out-of-phase w/ CCA1 |
| ZTL   | AT5G57360  | night   | constitutive| late aft/eve | protein stability |
| LKP2  | AT2G18915  | night   | constitutive| not characterised | structural analogy to ZTL |

---

## `circadian_genes_clean.json` — inclusion / exclusion reasoning

This file keeps only genes where the phase assignment is grounded in a single, consistent
molecular layer (mRNA or protein) without large cross-layer disagreement.  Night-class genes
are dropped entirely because they have no transcript oscillation.

### Included (clean)

| Gene | Class | Why clean |
|------|-------|-----------|
| **CCA1** | morning | mRNA, protein, and repressor function all peak at ZT0–2. Textbook morning gene. |
| **LHY** | morning | Same as CCA1; co-expressed and co-functional at dawn. |
| **LNK1, LNK2** | morning | mRNA peaks ZT1.5–2, same phase bin as CCA1/LHY.  One layer only (mRNA) studied in detail, and it is unambiguous. Placed in "morning" rather than "day" (as in the full file) because without the RVEs in the file the functional-co-activator argument for "day" disappears; the transcript phase is the honest anchor. |
| **PRR7** | day | Part of the PRR countdown wave.  mRNA, protein, and repressor activity all peak in the morning (ZT4–8) and decline by noon.  Cleanest gene family in the clock. |
| **PRR5** | day | PRR wave, mRNA peak ZT8–10, protein and repressor activity at similar time. Clean. |
| **PRR3** | day | PRR wave, mRNA peak ZT10–12 (day/evening boundary).  Nakamichi 2010 groups it with TOC1 as "evening," but its transcript peak is earlier; placed in "day" as mild ambiguity.  Both layers (mRNA and protein) are consistent — there is no cross-layer disagreement, only a boundary call. |
| **GI** | day | mRNA peaks ZT8–10.  The GI–ZTL–FKF1 complex is active at dusk (~ZT12), ~2–4 h later than the transcript peak — mild ambiguity, but a single bin shift and well-understood mechanistically. |
| **FKF1** | day | mRNA peaks ~ZT8.  FKF1 protein is stabilised by light and forms the GI–FKF1 complex at dusk, so the active complex is slightly later than the mRNA — same mild-ambiguity pattern as GI, one bin shift only. |
| **CHE** | day | mRNA and protein both peak ~ZT9 — consistent across layers.  The mild ambiguity is positional: ZT9 sits inside the "day" window (ZT6–12) but CHE represses CCA1 and interacts with TOC1, giving it an "evening-flavoured" function.  No cross-layer disagreement; the ambiguity is purely about which bin boundary ZT9 is closest to. |
| **TOC1** | evening | PRR wave terminus, mRNA peak ZT12–14, protein peak similar, repressor activity in early night. Clean. |
| **ELF3** | evening | mRNA peaks ZT14–16 regardless of photoperiod (Liu 2001). Protein and evening-complex (EC) formation at dusk. Consistent. |
| **ELF4** | evening | mRNA peaks ~ZT12 (dusk). Forms EC with ELF3 and LUX at the same time. Clean. |

### Excluded

| Gene | Reason for exclusion |
|------|----------------------|
| **RVE8** | mRNA peaks ZT0 (morning, same bin as CCA1), protein peaks ZT3–6, EE DNA-binding activity peaks ZT8, targets are evening genes.  Spans morning → day → afternoon across layers — the worst cross-layer disagreement in the clock. |
| **RVE4** | Same mechanistic problem as RVE8 (treated as functionally equivalent throughout the literature). |
| **ZTL** | mRNA is constitutively expressed — there is no transcript phase to anchor to.  Protein stability oscillates (peaks late afternoon), driven by GI-mediated stabilisation and SCF-complex degradation at night.  Phase assignment requires choosing a protein-stability metric that is not directly comparable to the transcript-peak metric used for other genes. |
| **LKP2** | Same constitutive-mRNA problem as ZTL, and additionally its protein oscillation is poorly characterised in the primary literature.  Any phase assignment would be speculative. |
