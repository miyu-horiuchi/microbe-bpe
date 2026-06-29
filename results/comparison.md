# Tokenization-trap downstream test — trait prediction

Each genome is frozen into features by the named method, then scored on microbe-foundation's 21-head trait model. **Headline comparison:** `single_nt` vs `domain_bpe` — same TinyGPT, residue-matched training, only the tokenizer differs. Evo2 arms are a far-larger single-nucleotide *reference*, not a matched control.

Seeds per cell: 3 (± is std over seeds). Scores are mean per-head test score, higher is better, rmse heads excluded.

## Mean score by method x split

| Method | role | species | genus | family |
|---|---|---:|---:|---:|
| `domain_bpe_1024` | headline method (domain BPE) | 0.5677 ± 0.0089 | 0.5693 ± 0.0173 | 0.4190 ± 0.0458 |
| `evo2` | reference (Evo2, single-nt, ~billions params) | 0.5601 ± 0.0042 | 0.5586 ± 0.0099 | 0.3938 ± 0.0269 |
| `evo2_bpe` | reference (Evo2 embeddings pooled by BPE spans) | 0.5449 ± 0.0195 | 0.5394 ± 0.0125 | 0.4243 ± 0.0291 |
| `kmer_4` | — | 0.5738 ± 0.0084 | 0.5493 ± 0.0057 | 0.4258 ± 0.0362 |
| `single_nt` | headline control (single nucleotide) | 0.5563 ± 0.0037 | 0.5505 ± 0.0093 | 0.4073 ± 0.0032 |

## Headline contrast Δ(domain_bpe − single_nt) by trait class

Mean Δ over the non-rmse heads in each class (± std across heads); p = two-sided Wilcoxon signed-rank over those heads (needs scipy & ≥6 heads).

| Split | trait class | n heads | mean Δ | p |
|---|---|---:|---:|---:|
| species | machinery | 7 | +0.0130 ± 0.0278 | — |
| species | compositional | 11 | +0.0087 ± 0.0274 | — |
| species | other | 2 | +0.0208 ± 0.0295 | — |
| species | all | 20 | +0.0114 ± 0.0265 | 0.064 |
| genus | machinery | 7 | +0.0273 ± 0.0660 | — |
| genus | compositional | 11 | +0.0096 ± 0.0340 | — |
| genus | other | 2 | +0.0399 ± 0.1007 | — |
| genus | all | 20 | +0.0188 ± 0.0514 | 0.131 |
| family | machinery | 7 | -0.0376 ± 0.1065 | — |
| family | compositional | 10 | +0.0478 ± 0.1910 | — |
| family | other | 2 | +0.0033 ± 0.0046 | — |
| family | all | 19 | +0.0117 ± 0.1539 | 0.496 |

## Per-head scores, species split, seed-averaged

| Head | class | metric | `single_nt` | `domain_bpe_1024` | `evo2` | `evo2_bpe` | Δ(bpe−nt) |
|---|---|---|---:|---:|---:|---:|---:|
| `amr_phenotype` | machinery | f1 | 0.6100 | 0.6100 | 0.6863 | 0.6263 | +0.0000 |
| `biosafety_level` | machinery | acc | 0.2619 | 0.3333 | 0.4286 | 0.4841 | +0.0714 |
| `carbon_utilization` | machinery | f1 | 0.3748 | 0.4001 | 0.3730 | 0.3821 | +0.0252 |
| `catalase` | compositional | acc | 0.8161 | 0.7931 | 0.7586 | 0.5862 | -0.0230 |
| `cell_shape` | compositional | acc | 0.7879 | 0.7879 | 0.7576 | 0.7980 | +0.0000 |
| `country` | other | acc | 0.0833 | 0.0833 | 0.1167 | 0.0750 | +0.0000 |
| `cultivation_medium` | machinery | f1 | 0.0000 | 0.0000 | 0.0225 | 0.0129 | +0.0000 |
| `cytochrome_oxidase` | compositional | acc | 0.3958 | 0.4479 | 0.4688 | 0.4688 | +0.0521 |
| `fatty_acid_profile` | machinery | rmse | 0.2657 | 0.3763 | 0.5278 | 0.3913 | +0.1106 |
| `gram_stain` | compositional | acc | 0.6569 | 0.6863 | 0.6765 | 0.5784 | +0.0294 |
| `halophily` | compositional | acc | 0.7500 | 0.7500 | 0.7292 | 0.7500 | +0.0000 |
| `isolation_source` | other | acc | 0.2917 | 0.3333 | 0.1667 | 0.0833 | +0.0417 |
| `metabolite_production` | machinery | f1 | 0.1357 | 0.1395 | 0.1450 | 0.1326 | +0.0038 |
| `motility` | compositional | acc | 0.5051 | 0.5657 | 0.5253 | 0.5556 | +0.0606 |
| `oxygen_tolerance` | compositional | acc | 0.7586 | 0.7586 | 0.6897 | 0.7241 | +0.0000 |
| `pathogenicity_animal` | machinery | acc | 0.8426 | 0.8333 | 0.8333 | 0.7870 | -0.0093 |
| `pathogenicity_human` | machinery | acc | 0.9020 | 0.9020 | 0.8235 | 0.8529 | +0.0000 |
| `ph_class` | compositional | acc | 0.1667 | 0.1429 | 0.2143 | 0.2143 | -0.0238 |
| `pigmentation` | compositional | acc | 0.8333 | 0.8333 | 0.8333 | 0.8333 | +0.0000 |
| `sporulation` | compositional | acc | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| `temperature_class` | compositional | acc | 0.9535 | 0.9535 | 0.9535 | 0.9535 | +0.0000 |

