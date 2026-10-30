# Schritt 5B-R: Adversarial Training des Experten-ROI-Klassifikators (exploratives Zusatzexperiment)

Stand: 21.10.2026. Schritt 5B-R ist umgesetzt und ausgeführt. Es ist ein begrenztes, **exploratives Zusatzexperiment auf Train und Validation**; die Testdaten wurden nicht geöffnet. Es ist nicht die Hauptfrage des Projekts und verdrängt keines der Lokalisationsergebnisse (Vollbild, Experten-ROI, automatische ROI), und es wurde nicht von Beginn an als Teil des Projektplans geplant: Anlass war der FGSM-Stresstest aus Schritt 5B, und die Validation-Ergebnisse der Pipelines waren bei der Planung bekannt.

## Frage und Einordnung

> Wie verändert eine kurze Fortsetzung des Experten-ROI-Trainings mit FGSM-Beispielen die Klassifikation unveränderter und gezielt gestörter Bilder?

- Die Forschungsfrage des Projekts (*Verbessert die Eingrenzung des Läsionsbereichs die Unterscheidung gutartiger und bösartiger Befunde im Brustultraschall?*) und die drei gleichwertigen Hauptvergleiche bleiben unverändert ([final_evaluation_protocol.md](final_evaluation_protocol.md)).
- Der Experten-ROI-Zweig ist eine **manuell lokalisierte Informationsbedingung** (die Expertenmaske bestimmt den Zuschnitt; bei der Anwendung wird eine manuelle Maske benötigt). Er ist kein „bestes Modell“, und es gibt keinen besten Seed.
- Es wurde keine neue gewöhnliche Augmentation eingeführt; Spiegeln, Rotation (±10 Grad) und Zoom (±0,10) aus den früheren Schritten blieben unverändert aktiv.
- Die Ergebnisse beschreiben fünf Trainingsläufe je Zustand auf **denselben** 385 Validation-Bildern (213 Cases; 125 maligne Bilder aus 69 Cases, 260 benigne aus 144 Cases). Das ist Trainingsstreuung auf einem festen Holdout, keine Signifikanzaussage und keine Streuung über neue Kohorten; mehrere Bilder eines Cases sind abhängig.

## Umfang und Abgrenzung

- **Freigegeben und ausgeführt:** Training und Auswertung auf Train und Validation; Bilder: die 1.114 Trainings- und die 385 Validation-Bilder (Trainingsläufe) beziehungsweise nur die 385 Validation-Bilder (Referenzauswertungen). Es wurden **keine Masken** geöffnet (die Zuschnitte folgen den bereits eingefrorenen Boxen), keine Testdatei, kein U-Net, keine Vorschau als Eingabe.
- **Nicht Teil des Schritts:** neue Segmentierung, neue Schwellensuche (keine neuen 90- oder 95-%-Betriebspunkte; die Schwellen aus 5A gehören zu den ursprünglichen Checkpoints und werden für die neuen Zustände nicht wiederverwendet), stärkere oder iterative Angriffe, weitere Architekturen, Auswahl eines besten Seeds, nachträgliche Zusatzläufe.
- **Teststatus:** Die zehn neuen Fortsetzungsmodelle bleiben Validation-Zusatzmodelle und sind nicht Teil der 27 Kombinationen der Testphase; eine Aufnahme würde einen gesonderten Protokollnachtrag erfordern ([final_evaluation_protocol.md](final_evaluation_protocol.md), Nachtrag vom 21.10.2026). Die Testdaten bleiben gesperrt.

## Startmodelle und Registrierung

Als Start dienen die fünf feinjustierten Experten-ROI-Klassifikatoren aus Schritt 4E (`expert_roi_finetuned|seed<s>`), ausgewählt über das 5A-Quellenregister (`final_evaluation_sources.csv`) mit Konfigurationen, Erfolgsstatus und Dateihashes. Der Seed-7-Checkpoint entspricht der vorab festgelegten Datei (SHA-256 `eada8248abd3f5fa5e1b89006737e86766cda5e4ad7c90b3fcefdbfc773ebefd`). Die Checkpoints werden gelesen und **nie erneut gespeichert**; ihre Hashes waren vor und nach jedem Lauf identisch.

| Seed | Kombination | SHA-256 des Startcheckpoints | Datei |
|---:|---|---|---|
| 7 | `expert_roi_finetuned\|seed7` | `eada8248abd3f5fa…` | `models/expert_roi/seed_7/mobilenetv2_higher_lr_160_roi/mobilenetv2_higher_lr_160_roi_seed7_best.keras` |
| 21 | `expert_roi_finetuned\|seed21` | `6a5a6697c05943a2…` | `models/expert_roi/seed_21/mobilenetv2_higher_lr_160_roi/mobilenetv2_higher_lr_160_roi_seed21_best.keras` |
| 42 | `expert_roi_finetuned\|seed42` | `e2bb60eb3289e618…` | `models/expert_roi/seed_42/mobilenetv2_higher_lr_160_roi/mobilenetv2_higher_lr_160_roi_seed42_best.keras` |
| 123 | `expert_roi_finetuned\|seed123` | `c94812a73c0e1884…` | `models/expert_roi/seed_123/mobilenetv2_higher_lr_160_roi/mobilenetv2_higher_lr_160_roi_seed123_best.keras` |
| 2026 | `expert_roi_finetuned\|seed2026` | `dcb92961b6f3783f…` | `models/expert_roi/seed_2026/mobilenetv2_higher_lr_160_roi/mobilenetv2_higher_lr_160_roi_seed2026_best.keras` |

Der Plan wurde **nach der Hashprüfung der Eingaben und vor dem ersten Bild dieses Schritts** einmalig registriert (`reports/tables/adversarial_roi_plan.json`; Studie `adv5br-dd935b42`, Plan-SHA-256 `dd935b4265cd72506d1f98ebca54e572824e123be32a3b9bb6395fc1b5b4dbf5`, registriert 21.10.2026 um 03:33:14). Der Plan hängt an der Splitdatei (SHA-256 `fdce9bc991b949cfa50ca1deca1ebd149fa6259c50dbac28e110a8fd4d38349a`; train 1.114 Bilder / 639 Cases, validation 385 / 213, test 376 / 212 nur gezählt), an der 5A-Registrierung (`final5a-937da596`), an der 5B-Registrierung (`prec5b-b8d9ce9f`), an den Geometriequellen des Vertrags `roi-expert-mask-v1` (`expert_roi_geometry_train_validation.csv`, `auto_roi_frozen.json`, `auto_roi_geometry_validation.csv`) und an den Hashes der Skripte und Module. Funktionstests davor verwendeten ausschließlich synthetische Daten.

## Verfahren

**Zustände je Seed.** *Original*: der unveränderte 4E-Zustand (nur bewertet). *Saubere Fortsetzung*: fünf Epochen mit den Batches `[x, x]`. *FGSM-Fortsetzung*: fünf Epochen mit den Batches `[x, x_adv]`. Beide Fortsetzungen starten vom selben Original-Checkpoint mit bitgleichen Startgewichten, derselben Schrittzahl und derselben Batchreihenfolge; sie unterscheiden sich nur in der zweiten Batchhälfte.

**Feste Wahl (vor dem ersten Bild registriert).**

- Eingabe: dieselbe 160 × 160 × 1-Pipeline des Experten-ROI (unmaskiertes Original innerhalb der erweiterten Expertenbox, vorhandene Resize-/Padding-Regel).
- Freigabe wie in 4E: die zehn Faltungen der Blöcke 14 bis 16 und `Conv_1` plus Kopf (12 trainierbare Tensoren, 1.512.001 Parameter); alle 52 BatchNorm-Schichten und der übrige Backbone bleiben fest (Backbone mit `training=False`).
- Optimierer: neues Adam mit Lernrate 1e-4 und 0 Iterationen zu Beginn; genau 5 Epochen mit je 140 Schritten = 700 Schritte pro Lauf (139 Batches mit 8 Originalbildern und ein letzter Batch mit 2 Originalbildern, in beiden Armen gleich behandelt: kein Auffüllen, kein Verwerfen); je Schritt 16 Beispiele (letzter Schritt 4). **Kein Early Stopping und keine Wahl einer besten Epoche:** gespeichert und bewertet wird der Zustand nach Epoche 5. Das ist eine bewusste Abweichung von der früheren Modellentwicklung (dort wurde der beste Validation-Checkpoint gewählt); die Zustände sind keine „besten Epochen“.
- Angriff im Training: FGSM mit ε = 2/255 auf der externen float32-Eingabe in [0, 1] (feste Versuchswahl, nicht optimiert; der Kurswert 0,1 wird nicht verwendet), echte Labels, ungezielt, **ein Schritt** vom ungestörten Eingang aus, erzeugt vom aktuellen Modell im Inferenzmodus (keine Augmentation, kein Dropout, BatchNorm fest), nur auf der geometrisch gültigen Bildfläche (das künstliche Padding bleibt exakt konstant; die Gültigkeitsmaske folgt der Zuschnitt-/Resize-Geometrie, nie der Läsionsmaske), danach `stop_gradient`. Der Angriff wird **vor** der zufälligen Augmentation erzeugt (die untersuchte Kombination; er ist nicht für alle Transformationen optimiert); danach läuft `[x, x_adv]` mit verdoppelten Labels im Trainingsmodus durch das Modell, und ein Adam-Schritt aktualisiert die Gewichte anhand der mittleren BCE. Der saubere Arm verwendet dieselbe Prozedur mit `[x, x]`.
- Aufbau nach dem Kursnotebook Tag 19 (`19_best_practices_fairness/19_02_fairness_adversarial_pytorch.ipynb`, Abschnitt Adversarial Training: Mischung etwa 50/50, Angriffe aus dem aktuellen Modell erzeugt, Vergleich auf sauberen und gestörten Bildern; der PyTorch-Code wird nicht übernommen, das Projekt nutzt TensorFlow/Keras); `20_01` dient nur als Ausblick.

**Auswertung.** Jeder Zustand wird auf den Experten-ROIs (primär) und auf den eingefrorenen automatischen 4F-B-Zuschnitten (gleiche Gewichte; ein leeres Segmentierungsergebnis führt zum Vollbild mit demselben ROI-Klassifikator; U-Net und Zuschnittsregel unverändert) mit ε = 0, 1/255, 2/255 und 4/255 bewertet. Der Angriff wird für jeden bewerteten Zustand **neu gegen dieses Modell** vom ungestörten Eingang aus erzeugt (ein Schritt je Stärke, echte Labels, nur gültige Fläche). Die Scores und Losswerte stammen aus dem Inferenzpfad (Batch 16); der Gradientengraph liefert nur die Richtung. Kennzahlen: ROC-AUC, Average Precision, BCE (aus dem Logit, nur Bildebene), TN/FP/FN/TP, Sensitivität, Spezifität, Precision, F1 und Accuracy bei der Referenzschwelle 0,5, jeweils mit Bild- und Case-Zahlen (maligne = 1); Case-Ebene mit der Maximum-Regel. Die Scores der unveränderten Original-Zustände müssen die gespeicherten 4E- und 4F-B-Scores auf 1e-5 reproduzieren (vor jedem Update geprüft). Gepaarte Differenzen innerhalb des Seeds: FGSM − sauber, sauber − Original, FGSM − Original; Einzelwerte, Mittel, Stichproben-SD (ddof = 1) und Zahl der positiven, negativen und gleichen Differenzen; keine Pooling-AUC, kein Ensemble. **Primärer beschreibender Vergleich:** FGSM − sauber in der Bild-ROC-AUC bei ε = 2/255 auf den Experten-ROIs, neben der Änderung auf unveränderten Bildern.

## Technische Nachweise

- **Läufe:** 15 von 15 Läufen gültig, 0 ungültig (5 Referenzauswertungen, 10 Trainingsläufe; jeder Lauf als eigener Prozess nacheinander; Läufe, Run-IDs und Hashes in der Tabelle unten). Studienstatus `passed`; Gesamtdauer des Produktionsaufrufs 13 min 23 s.
- **Schritte und Startgewichte:** 700 Optimiererschritte je Trainingslauf (Adam-Iterationen 0 → 700) in beiden Armen. Die Startgewichte beider Fortsetzungen und des Referenzzustands sind je Seed bitgleich (ein gemeinsamer Gewichts-Digest). Nach dem Training waren die zehn freigegebenen Faltungen und der Kopf verändert (größte absolute Änderung je Lauf höchstens 0,027 in den Faltungen und 0,016 im Kopf), die 52 BatchNorm-Schichten und die übrigen 42 Backbone-Schichten **bitgleich** zum Start. Die Lernrate 1e-4 wurde aus dem Optimierer ausgelesen.
- **Original unverändert:** Die fünf 4E-Checkpoints waren vor und nach jedem Lauf byteidentisch zu den registrierten Dateien; sie wurden nicht erneut gespeichert. Die Referenzscores reproduzieren die gespeicherten 4E- und 4F-B-Scores (größte Abweichung 0; Toleranz 1e-5).
- **Speichern und Laden:** Der Epoche-5-Zustand wurde gespeichert und neu geladen; die Gewichte entsprechen dem trainierten Zustand, und die neu geladenen Modelle reproduzieren die sauberen Scores des Trainingsprozesses (Abweichung 0).
- **GPU:** Variablen, Angriffs-Vorwärtspfad, Loss, Eingabe- und Trainingsgradienten lagen nachweisbar auf `GPU:0` (Geräteeinträge der Tensoren je Lauf im Status); Memory Growth an, weiche Geräteplatzierung aus, kein Speicherlimit, kein CPU-Fallback, `jit_compile=False`, TF32 wie in den akzeptierten Läufen, keine Mixed Precision; GPU-Speicherspitze je Trainingslauf rund 1,08 GiB (RTX 3050 mit 4 GiB). Ein Speicherfehler wäre als Fehlschlag protokolliert worden; er trat nicht auf.
- **Zugriffe:** unabhängiges Zugriffsprotokoll je Lauf: Trainingsläufe 1.499 Bilder (1.114 + 385), Referenzläufe 385 Bilder, **0 Masken**, keine unerwarteten Rohdateien, kein Testzugriff, kein U-Net-Zugriff; der Koordinator selbst öffnete kein Bild (Prüfung durch einen externen Aufrufmitschnitt bestätigt).
- **FGSM-Prüfungen:** in allen 120 Auswertungsblöcken (15 Läufe × 2 Eingaben × 4 Stärken) endliche Eingabegradienten, kein Bild mit vollständig verschwindendem Gradienten, größte Änderung höchstens ε + 1e-7, Padding unverändert, Wertebereich [0, 1], ε = 0 identisch zur Eingabe, `sigmoid(Logit)` gegen die Vorhersagefunktion höchstens 8,7e-08.
- **Numerik (TF32):** Der Vorwärtspfad innerhalb des Gradientengraphen weicht unter TF32 vom Inferenzpfad ab (siehe Schritt 5B); in dieser Studie bis zu 0,13 im Logit (saubere Fortsetzung), 0,061 (Original) und 0,027 (FGSM-Fortsetzung; Diagnosewert je Block gespeichert, kein Prüfkriterium). Deshalb stammen alle berichteten Scores aus dem Inferenzpfad, und für die Richtung des Angriffs gibt es **keine Garantie**, dass die Vorzeichen denen des Inferenzpfads entsprechen. Eine konkrete Ursache der Abweichung ist nicht nachgewiesen; es wurde keine Toleranz gelockert und der numerische Modus nicht geändert.
- **Rechenzeit:** Trainingszeit der fünf Läufe je Arm: saubere Fortsetzung 125,7 s, FGSM-Fortsetzung 218,5 s (Faktor 1,74; zusätzlicher Aufwand durch den Angriffs-Vorwärts-/Rückwärtspfad je Schritt).

| Seed | Zustand | Run-ID | vollständig | Training (s) | Auswertung (s) | Schritte | Startgewichte (Digest) | Endgewichte (Digest) | Checkpoint-SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | original | `adv5br-20261021T033314-29d14910` | ja | 0,0 | 17,7 | – | a3a91010eb… | – | – |
| 7 | clean_continuation | `adv5br-20261021T033524-694140dc` | ja | 25,7 | 17,3 | 700 | a3a91010eb… | 88d1b883c0… | 7f07f29fa5… |
| 7 | fgsm_continuation | `adv5br-20261021T033624-4cc3e3d4` | ja | 45,2 | 16,2 | 700 | a3a91010eb… | ef051757ce… | 7e02007bc0… |
| 21 | original | `adv5br-20261021T033341-9675a004` | ja | 0,0 | 18,4 | – | de82c906e8… | – | – |
| 21 | clean_continuation | `adv5br-20261021T033743-b9d61407` | ja | 25,3 | 16,2 | 700 | de82c906e8… | 010210b84e… | 6a9deab286… |
| 21 | fgsm_continuation | `adv5br-20261021T033842-952f28a4` | ja | 43,7 | 16,1 | 700 | de82c906e8… | 3dd1bd4509… | 0f3c2adf7e… |
| 42 | original | `adv5br-20261021T033407-6558f1a9` | ja | 0,0 | 17,3 | – | 750a8ca578… | – | – |
| 42 | clean_continuation | `adv5br-20261021T033958-b4433087` | ja | 25,0 | 16,3 | 700 | 750a8ca578… | 42c18b51b6… | b231024a31… |
| 42 | fgsm_continuation | `adv5br-20261021T034055-9063ea0f` | ja | 43,2 | 15,8 | 700 | 750a8ca578… | 872ecc50d4… | ff1112f983… |
| 123 | original | `adv5br-20261021T033433-22813bcd` | ja | 0,0 | 17,4 | – | 3637ff3a50… | – | – |
| 123 | clean_continuation | `adv5br-20261021T034211-7b527bc8` | ja | 24,9 | 15,8 | 700 | 3637ff3a50… | 95ab405a3a… | f398bc6d0a… |
| 123 | fgsm_continuation | `adv5br-20261021T034307-3bcadcb5` | ja | 43,2 | 15,9 | 700 | 3637ff3a50… | c81e91d453… | 8898ce1251… |
| 2026 | original | `adv5br-20261021T033458-c118e384` | ja | 0,0 | 17,4 | – | 32895f88b2… | – | – |
| 2026 | clean_continuation | `adv5br-20261021T034423-14a49657` | ja | 24,8 | 15,9 | 700 | 32895f88b2… | d480f97d42… | b9aa7114d3… |
| 2026 | fgsm_continuation | `adv5br-20261021T034519-a768c0eb` | ja | 43,2 | 15,8 | 700 | 32895f88b2… | c3bad60d19… | f754c5b583… |

## Ergebnisse (Validation, fünf Seeds: Mittel ± Stichproben-SD)

Alle Tabellen stammen aus den gespeicherten Aggregattabellen (`reports/tables/adversarial_roi_*.csv`; Einzelwerte je Seed in `adversarial_roi_run_metrics.csv` und `adversarial_roi_paired_differences.csv`).

### Unveränderte Bilder (ε = 0), Bildebene (385 Bilder, 213 Cases)

| Eingabe | Zustand | ROC-AUC | AP | BCE | Sensitivität (0,5) | Spezifität (0,5) | Genauigkeit (0,5) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Experten-ROI (primär) | ursprünglicher 4E-Zustand | 0,921 ± 0,012 | 0,872 ± 0,014 | 0,342 ± 0,037 | 0,675 ± 0,071 | 0,956 ± 0,012 | 0,865 ± 0,017 |
| Experten-ROI (primär) | saubere Fortsetzung (5 Epochen) | 0,916 ± 0,014 | 0,870 ± 0,016 | 0,390 ± 0,016 | 0,691 ± 0,099 | 0,930 ± 0,041 | 0,852 ± 0,007 |
| Experten-ROI (primär) | FGSM-Fortsetzung (5 Epochen) | 0,916 ± 0,005 | 0,874 ± 0,007 | 0,387 ± 0,020 | 0,627 ± 0,109 | 0,952 ± 0,036 | 0,846 ± 0,016 |
| automatische ROI (4F-B-Zuschnitte) | ursprünglicher 4E-Zustand | 0,898 ± 0,011 | 0,820 ± 0,032 | 0,382 ± 0,029 | 0,656 ± 0,067 | 0,932 ± 0,014 | 0,842 ± 0,020 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung (5 Epochen) | 0,892 ± 0,016 | 0,831 ± 0,014 | 0,436 ± 0,034 | 0,670 ± 0,112 | 0,918 ± 0,042 | 0,838 ± 0,010 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung (5 Epochen) | 0,892 ± 0,008 | 0,810 ± 0,014 | 0,415 ± 0,019 | 0,603 ± 0,123 | 0,924 ± 0,050 | 0,820 ± 0,018 |

Case-Ebene (213 Cases; Maximum-Regel):

| Eingabe | Zustand | Case-ROC-AUC | Case-AP | Case-Sensitivität (Max-Regel, 0,5) | Case-Spezifität | Case-Genauigkeit |
|---|---:|---:|---:|---:|---:|---:|
| Experten-ROI (primär) | ursprünglicher 4E-Zustand | 0,937 ± 0,012 | 0,892 ± 0,013 | 0,771 ± 0,059 | 0,938 ± 0,020 | 0,884 ± 0,014 |
| Experten-ROI (primär) | saubere Fortsetzung (5 Epochen) | 0,934 ± 0,013 | 0,889 ± 0,016 | 0,768 ± 0,104 | 0,896 ± 0,060 | 0,854 ± 0,014 |
| Experten-ROI (primär) | FGSM-Fortsetzung (5 Epochen) | 0,934 ± 0,005 | 0,899 ± 0,006 | 0,725 ± 0,106 | 0,936 ± 0,046 | 0,868 ± 0,017 |
| automatische ROI (4F-B-Zuschnitte) | ursprünglicher 4E-Zustand | 0,919 ± 0,012 | 0,850 ± 0,036 | 0,745 ± 0,045 | 0,907 ± 0,018 | 0,854 ± 0,016 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung (5 Epochen) | 0,916 ± 0,010 | 0,862 ± 0,010 | 0,759 ± 0,120 | 0,885 ± 0,062 | 0,844 ± 0,006 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung (5 Epochen) | 0,910 ± 0,006 | 0,837 ± 0,018 | 0,710 ± 0,118 | 0,894 ± 0,064 | 0,835 ± 0,023 |

### FGSM-Stresstest: Bild-ROC-AUC (Mittel ± SD) je Stärke

Der Angriff wird je Zustand gegen das jeweils ausgewertete Modell erzeugt.

| Eingabe | Zustand | ε = 0 | ε = 1/255 | ε = 2/255 | ε = 4/255 |
|---|---:|---:|---:|---:|---:|
| Experten-ROI (primär) | ursprünglicher 4E-Zustand | 0,921 ± 0,012 | 0,155 ± 0,039 | 0,093 ± 0,033 | 0,102 ± 0,059 |
| Experten-ROI (primär) | saubere Fortsetzung (5 Epochen) | 0,916 ± 0,014 | 0,083 ± 0,027 | 0,036 ± 0,015 | 0,028 ± 0,013 |
| Experten-ROI (primär) | FGSM-Fortsetzung (5 Epochen) | 0,916 ± 0,005 | 0,467 ± 0,063 | 0,392 ± 0,063 | 0,405 ± 0,058 |
| automatische ROI (4F-B-Zuschnitte) | ursprünglicher 4E-Zustand | 0,898 ± 0,011 | 0,144 ± 0,040 | 0,093 ± 0,038 | 0,102 ± 0,060 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung (5 Epochen) | 0,892 ± 0,016 | 0,075 ± 0,024 | 0,035 ± 0,015 | 0,030 ± 0,014 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung (5 Epochen) | 0,892 ± 0,008 | 0,429 ± 0,068 | 0,368 ± 0,067 | 0,386 ± 0,071 |

Bild-BCE und Average Precision je Stärke:

| Eingabe | Zustand | ε = 0 | ε = 1/255 | ε = 2/255 | ε = 4/255 |
|---|---:|---:|---:|---:|---:|
| Experten-ROI (primär) | ursprünglicher 4E-Zustand | 0,342 ± 0,037 | 1,564 ± 0,410 | 1,725 ± 0,474 | 1,703 ± 0,513 |
| Experten-ROI (primär) | saubere Fortsetzung (5 Epochen) | 0,390 ± 0,016 | 2,264 ± 0,763 | 2,590 ± 0,871 | 2,606 ± 0,845 |
| Experten-ROI (primär) | FGSM-Fortsetzung (5 Epochen) | 0,387 ± 0,020 | 0,839 ± 0,108 | 0,882 ± 0,107 | 0,834 ± 0,081 |
| automatische ROI (4F-B-Zuschnitte) | ursprünglicher 4E-Zustand | 0,382 ± 0,029 | 1,672 ± 0,437 | 1,818 ± 0,510 | 1,775 ± 0,554 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung (5 Epochen) | 0,436 ± 0,034 | 2,455 ± 0,878 | 2,739 ± 0,979 | 2,703 ± 0,939 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung (5 Epochen) | 0,415 ± 0,019 | 0,907 ± 0,137 | 0,937 ± 0,131 | 0,868 ± 0,091 |

| Eingabe | Zustand | ε = 0 | ε = 1/255 | ε = 2/255 | ε = 4/255 |
|---|---:|---:|---:|---:|---:|
| Experten-ROI (primär) | ursprünglicher 4E-Zustand | 0,872 ± 0,014 | 0,200 ± 0,006 | 0,191 ± 0,003 | 0,194 ± 0,010 |
| Experten-ROI (primär) | saubere Fortsetzung (5 Epochen) | 0,870 ± 0,016 | 0,191 ± 0,003 | 0,186 ± 0,001 | 0,186 ± 0,001 |
| Experten-ROI (primär) | FGSM-Fortsetzung (5 Epochen) | 0,874 ± 0,007 | 0,326 ± 0,034 | 0,281 ± 0,027 | 0,291 ± 0,027 |
| automatische ROI (4F-B-Zuschnitte) | ursprünglicher 4E-Zustand | 0,820 ± 0,032 | 0,199 ± 0,005 | 0,192 ± 0,004 | 0,194 ± 0,010 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung (5 Epochen) | 0,831 ± 0,014 | 0,190 ± 0,002 | 0,186 ± 0,001 | 0,186 ± 0,001 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung (5 Epochen) | 0,810 ± 0,014 | 0,299 ± 0,031 | 0,267 ± 0,027 | 0,277 ± 0,031 |

Änderung der Bild-ROC-AUC gegenüber den ungestörten Bildern **desselben Zustands** (Mittel, SD und Zahl der Seeds mit negativer Änderung):

| Eingabe | Zustand | ε = 1/255 | ε = 2/255 | ε = 4/255 |
|---|---:|---:|---:|---:|
| Experten-ROI (primär) | ursprünglicher 4E-Zustand | −0,766 (SD 0,047; 5 neg.) | −0,827 (SD 0,042; 5 neg.) | −0,819 (SD 0,068; 5 neg.) |
| Experten-ROI (primär) | saubere Fortsetzung (5 Epochen) | −0,832 (SD 0,020; 5 neg.) | −0,880 (SD 0,014; 5 neg.) | −0,887 (SD 0,012; 5 neg.) |
| Experten-ROI (primär) | FGSM-Fortsetzung (5 Epochen) | −0,449 (SD 0,064; 5 neg.) | −0,523 (SD 0,064; 5 neg.) | −0,510 (SD 0,059; 5 neg.) |
| automatische ROI (4F-B-Zuschnitte) | ursprünglicher 4E-Zustand | −0,754 (SD 0,049; 5 neg.) | −0,804 (SD 0,047; 5 neg.) | −0,795 (SD 0,069; 5 neg.) |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung (5 Epochen) | −0,816 (SD 0,016; 5 neg.) | −0,857 (SD 0,013; 5 neg.) | −0,861 (SD 0,017; 5 neg.) |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung (5 Epochen) | −0,464 (SD 0,066; 5 neg.) | −0,524 (SD 0,065; 5 neg.) | −0,506 (SD 0,069; 5 neg.) |

Konfusionszahlen bei der Referenzschwelle 0,5 (Mittel der fünf Seeds; 125 maligne, 260 benigne Bilder):

| Eingabe | Zustand | Stärke | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|
| Experten-ROI (primär) | ursprünglicher 4E-Zustand | ε = 0 | 248,6 | 11,4 | 40,6 | 84,4 |
| Experten-ROI (primär) | ursprünglicher 4E-Zustand | ε = 2/255 | 127,8 | 132,2 | 119,4 | 5,6 |
| Experten-ROI (primär) | saubere Fortsetzung (5 Epochen) | ε = 0 | 241,8 | 18,2 | 38,6 | 86,4 |
| Experten-ROI (primär) | saubere Fortsetzung (5 Epochen) | ε = 2/255 | 81,4 | 178,6 | 122,4 | 2,6 |
| Experten-ROI (primär) | FGSM-Fortsetzung (5 Epochen) | ε = 0 | 247,4 | 12,6 | 46,6 | 78,4 |
| Experten-ROI (primär) | FGSM-Fortsetzung (5 Epochen) | ε = 2/255 | 208,4 | 51,6 | 102,6 | 22,4 |
| automatische ROI (4F-B-Zuschnitte) | ursprünglicher 4E-Zustand | ε = 0 | 242,2 | 17,8 | 43,0 | 82,0 |
| automatische ROI (4F-B-Zuschnitte) | ursprünglicher 4E-Zustand | ε = 2/255 | 119,4 | 140,6 | 118,6 | 6,4 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung (5 Epochen) | ε = 0 | 238,8 | 21,2 | 41,2 | 83,8 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung (5 Epochen) | ε = 2/255 | 73,2 | 186,8 | 122,4 | 2,6 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung (5 Epochen) | ε = 0 | 240,2 | 19,8 | 49,6 | 75,4 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung (5 Epochen) | ε = 2/255 | 199,0 | 61,0 | 101,0 | 24,0 |

### Gepaarte Differenzen innerhalb des Seeds (Bild-ROC-AUC, B minus A)

| Eingabe | Differenz (B − A) | ε | Mittel | SD | positiv / negativ / gleich |
|---|---:|---:|---:|---:|---:|
| Experten-ROI (primär) | FGSM-Fortsetzung − saubere Fortsetzung | 0 | −0,000 | 0,010 | 1 / 4 / 0 |
| Experten-ROI (primär) | FGSM-Fortsetzung − saubere Fortsetzung | 1/255 | +0,383 | 0,041 | 5 / 0 / 0 |
| Experten-ROI (primär) | FGSM-Fortsetzung − saubere Fortsetzung | 2/255 | +0,357 | 0,050 | 5 / 0 / 0 |
| Experten-ROI (primär) | FGSM-Fortsetzung − saubere Fortsetzung | 4/255 | +0,377 | 0,048 | 5 / 0 / 0 |
| Experten-ROI (primär) | saubere Fortsetzung − Original | 0 | −0,005 | 0,021 | 3 / 2 / 0 |
| Experten-ROI (primär) | saubere Fortsetzung − Original | 1/255 | −0,071 | 0,044 | 0 / 5 / 0 |
| Experten-ROI (primär) | saubere Fortsetzung − Original | 2/255 | −0,058 | 0,030 | 0 / 5 / 0 |
| Experten-ROI (primär) | saubere Fortsetzung − Original | 4/255 | −0,074 | 0,053 | 0 / 5 / 0 |
| Experten-ROI (primär) | FGSM-Fortsetzung − Original | 0 | −0,005 | 0,012 | 2 / 3 / 0 |
| Experten-ROI (primär) | FGSM-Fortsetzung − Original | 1/255 | +0,312 | 0,055 | 5 / 0 / 0 |
| Experten-ROI (primär) | FGSM-Fortsetzung − Original | 2/255 | +0,299 | 0,052 | 5 / 0 / 0 |
| Experten-ROI (primär) | FGSM-Fortsetzung − Original | 4/255 | +0,303 | 0,056 | 5 / 0 / 0 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung − saubere Fortsetzung | 0 | +0,000 | 0,011 | 3 / 2 / 0 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung − saubere Fortsetzung | 1/255 | +0,353 | 0,048 | 5 / 0 / 0 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung − saubere Fortsetzung | 2/255 | +0,333 | 0,056 | 5 / 0 / 0 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung − saubere Fortsetzung | 4/255 | +0,356 | 0,063 | 5 / 0 / 0 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung − Original | 0 | −0,006 | 0,023 | 3 / 2 / 0 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung − Original | 1/255 | −0,069 | 0,034 | 0 / 5 / 0 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung − Original | 2/255 | −0,058 | 0,031 | 0 / 5 / 0 |
| automatische ROI (4F-B-Zuschnitte) | saubere Fortsetzung − Original | 4/255 | −0,072 | 0,051 | 0 / 5 / 0 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung − Original | 0 | −0,006 | 0,013 | 2 / 3 / 0 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung − Original | 1/255 | +0,284 | 0,052 | 5 / 0 / 0 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung − Original | 2/255 | +0,275 | 0,051 | 5 / 0 / 0 |
| automatische ROI (4F-B-Zuschnitte) | FGSM-Fortsetzung − Original | 4/255 | +0,284 | 0,056 | 5 / 0 / 0 |

Average Precision und BCE (Experten-ROI und automatische ROI) stehen in `adversarial_roi_paired_summary.csv`. Auf den Experten-ROIs lauten FGSM − sauber bei ε = 2/255: AP +0,094 (SD 0,026), BCE −1,708 (SD 0,775).

### Primärer beschreibender Vergleich (Experten-ROI, FGSM − saubere Fortsetzung)

| Seed | ROC-AUC Original (ε = 0) | saubere Fortsetzung (ε = 0) | FGSM-Fortsetzung (ε = 0) | FGSM − sauber (ε = 0) | saubere Fortsetzung (ε = 2/255) | FGSM-Fortsetzung (ε = 2/255) | FGSM − sauber (ε = 2/255) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0,9217 | 0,9248 | 0,9155 | −0,0094 | 0,0438 | 0,4594 | +0,4156 |
| 21 | 0,9294 | 0,9228 | 0,9221 | −0,0007 | 0,0334 | 0,3471 | +0,3137 |
| 42 | 0,9130 | 0,9262 | 0,9190 | −0,0072 | 0,0338 | 0,3872 | +0,3534 |
| 123 | 0,9048 | 0,9113 | 0,9106 | −0,0006 | 0,0534 | 0,4508 | +0,3974 |
| 2026 | 0,9339 | 0,8938 | 0,9106 | +0,0169 | 0,0137 | 0,3164 | +0,3027 |

Bei ε = 2/255 beträgt die gepaarte Differenz der Bild-ROC-AUC im Mittel +0,357 (Stichproben-SD 0,050; 5 Seeds positiv, 0 negativ, 0 gleich); auf den ungestörten Bildern beträgt sie −0,0002 (SD 0,010; 1 positiv, 4 negativ). Auf den automatischen Zuschnitten (gleiche Gewichte, anderer Zuschnitt) ist die Richtung gleich: +0,333 (SD 0,056; 5 Seeds positiv).

### Trainingsverlauf (Epoche 5)

| Seed | Arm | Loss Epoche 1 | Loss Epoche 5 | Epoche 5: Hälfte der Originale | Epoche 5: zweite Hälfte (gestört bzw. identisch) | größte mittlere L∞-Änderung je Schritt | Nullgradient-Zeilen (alle Epochen) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | clean_continuation | 0,3913 | 0,3173 | 0,3116 | 0,3231 | 0,00000 | 0 |
| 7 | fgsm_continuation | 0,5442 | 0,4763 | 0,3893 | 0,5633 | 0,00784 | 0 |
| 21 | clean_continuation | 0,3608 | 0,2883 | 0,2902 | 0,2863 | 0,00000 | 0 |
| 21 | fgsm_continuation | 0,5353 | 0,4620 | 0,3615 | 0,5626 | 0,00784 | 0 |
| 42 | clean_continuation | 0,3908 | 0,3462 | 0,3504 | 0,3420 | 0,00000 | 0 |
| 42 | fgsm_continuation | 0,5478 | 0,4835 | 0,3945 | 0,5725 | 0,00784 | 0 |
| 123 | clean_continuation | 0,5200 | 0,3661 | 0,3623 | 0,3698 | 0,00000 | 0 |
| 123 | fgsm_continuation | 0,5773 | 0,5053 | 0,4351 | 0,5755 | 0,00784 | 0 |
| 2026 | clean_continuation | 0,3734 | 0,3164 | 0,3187 | 0,3140 | 0,00000 | 0 |
| 2026 | fgsm_continuation | 0,5256 | 0,4883 | 0,3937 | 0,5829 | 0,00784 | 0 |

Der Trainingsloss der gestörten Beispiele bleibt in allen fünf Seeds nach fünf Epochen deutlich über dem der ungestörten Hälfte (Spalten „Hälfte der Originale“ und „zweite Hälfte“); der Loss ist über Batches mit Augmentation gemittelt und kein Validation-Ergebnis. Die mittlere L∞-Änderung des Angriffs je Schritt entspricht ε = 2/255 (0,00784); es gab in keinem Schritt eine Zeile mit Nullgradient.

## Befunde (beschreibend, auch uneinheitliche Ergebnisse)

1. **Unveränderte Bilder:** Die Bild-ROC-AUC der Experten-ROI-Bewertung liegt bei 0,921 ± 0,012 (Original), 0,916 ± 0,014 (saubere Fortsetzung) und 0,916 ± 0,005 (FGSM-Fortsetzung). Die gepaarten Differenzen sind klein und uneinheitlich: FGSM − sauber −0,0002 (1 Seed positiv, 4 negativ), sauber − Original −0,005 (SD 0,021; 3 positiv, 2 negativ). Im Rahmen dieser fünf Seeds ist keine Veränderung der ungestörten Trennschärfe erkennbar; eine Verbesserung auf sauberen Bildern wurde nicht beobachtet und war kein Erfolgskriterium. Die BCE auf den ungestörten Bildern liegt bei beiden Fortsetzungen über dem Original (+0,048 sauber, +0,045 FGSM; 4 bzw. 5 von fünf Seeds positiv). Die Sensitivität bei der Schwelle 0,5 ist bei der FGSM-Fortsetzung am niedrigsten (0,627 ± 0,109 gegenüber 0,691 ± 0,099 sauber und 0,675 ± 0,071 Original; große Streuung); das ist ein Betriebspunktwert bei einer festen Schwelle und kein Vergleich bei gleicher Sensitivität.
2. **Gestörte Bilder:** Der einmalige FGSM-Schritt senkt die Bild-ROC-AUC bei **allen drei Zuständen weit unter 0,5**, also unter den Wert eines Zufallsscores; die Rangfolge von malignen und benignen Bildern kehrt sich überwiegend um. Bei ε = 2/255 (Experten-ROI): Original 0,093 ± 0,033, saubere Fortsetzung 0,036 ± 0,015, FGSM-Fortsetzung 0,392 ± 0,063 (alle fünf Seeds der FGSM-Fortsetzung bleiben bei 2/255 unter 0,5: 0,316 bis 0,459; bei 1/255 liegt ein Seed leicht darüber: 0,389 bis 0,530).
3. **Primärer Vergleich:** FGSM − saubere Fortsetzung bei 2/255: +0,357 (SD 0,050, in allen fünf Seeds positiv); zugleich −0,0002 auf den ungestörten Bildern. Die FGSM-Fortsetzung bleibt aber auch bei dieser Stärke weit unter ihrer eigenen Leistung auf ungestörten Bildern (0,916 → 0,392) und unter dem Zufallswert 0,5. Bei der Schwelle 0,5 erkennt sie unter Störung im Mittel 22,4 der 125 malignen Bilder (Original 5,6, saubere Fortsetzung 2,6); im Mittel 102,6 werden übersehen.
4. **Saubere Fortsetzung:** Die zusätzlichen fünf Epochen mit unveränderten Bildern gingen mit einer **niedrigeren** Bild-ROC-AUC unter Störung einher (sauber − Original bei 2/255: −0,058, SD 0,030, in allen fünf Seeds negativ) und mit einer höheren Störungs-BCE (2,590 gegenüber 1,725). Eine Ursache wird nicht behauptet.
5. **Stärkenabhängigkeit:** Die Bild-ROC-AUC unter Störung ändert sich zwischen 2/255 und 4/255 nur wenig und nicht einheitlich (FGSM-Fortsetzung 0,392 → 0,405, Original 0,093 → 0,102, saubere Fortsetzung 0,036 → 0,028); das ist ein beobachtetes Verhalten des Einzelschritt-Angriffs, keine Aussage über stärkere Angriffe und ohne behauptete Ursache.
6. **Automatische Zuschnitte:** Mit denselben Gewichten und den eingefrorenen 4F-B-Zuschnitten zeigen alle drei Zustände ein vergleichbares Muster mit etwas niedrigerer Trennschärfe (ungestört 0,898 ± 0,011 / 0,892 ± 0,016 / 0,892 ± 0,008). Die Klassifikatoren wurden auf Experten-ROIs trainiert; die automatische Bedingung ist eine Anwendungsprüfung ohne eigenes Training auf automatischen Zuschnitten.
7. **Rechenaufwand:** Die FGSM-Fortsetzung brauchte etwa das 1,7-Fache der Trainingszeit der sauberen Fortsetzung.

## Grenzen

- **Ein Angriff, ein Schritt:** Gemessen wird nur der einmalige FGSM-Schritt mit ε ≤ 4/255 gegen das jeweils ausgewertete Modell, derselben Angriffsart wie im Training. Eine Verbesserung unter genau diesem Angriff ist **kein Nachweis von Robustheit** gegen stärkere oder iterative Angriffe (zum Beispiel PGD, Madry et al.), andere Störungen, Rauschen, Geräte, Cases oder Verteilungsverschiebungen. Adversarial Training mit einem einzelnen schnellen Schritt kann auf die untersuchte Angriffsart spezialisieren (beschriebene Fehlformen, etwa „catastrophic overfitting“, bei Wong et al.); stärkere Angriffe wurden bewusst nicht ausgeführt.
- **Richtung des Angriffs:** Der Vorwärtspfad des Gradientengraphen weicht unter TF32 vom Inferenzpfad ab (siehe oben); es gibt keine Garantie unveränderter Vorzeichen.
- **Auswertungsdaten:** Die Validation-Bilder wurden für die gesamte Entwicklung wiederholt genutzt; fünf Seeds messen Trainingsstreuung auf demselben Holdout. Keine Signifikanzaussage; die Case-Abhängigkeit wird bei den SD-Angaben nicht berücksichtigt.
- **Feste Wahl:** ε = 2/255, fünf Epochen, Adam 1e-4, Angriff vor der Augmentation, eine Konfiguration; weder optimiert noch variiert. Der Zustand nach Epoche 5 ist kein ausgewählter Checkpoint.
- **Schwelle und Kalibrierung:** nur die Referenzschwelle 0,5; keine neue Betriebspunktsuche, keine Aussage zur Kalibrierung der Scores (Malignant-Score, keine klinische Wahrscheinlichkeit).
- **Informationsbedingung:** Die Experten-ROI-Auswertung erfordert bei der Anwendung eine manuelle Maske und ist keine reine Bild-zu-Score-Pipeline. Der Angriff wirkt auf die digitale Klassifikatoreingabe, nicht auf die Segmentierungs- und Zuschnittskette und nicht physikalisch auf die Bildentstehung.
- **Kein klinischer Befund:** interne Forschungsauswertung auf Läsionsbildern; keine Aussage zu klinischer Einsatzfähigkeit oder zu ärztlichen Entscheidungen.

## Abweichungen und Fehlversuche

- **Produktionslauf:** Der Lauf `adv5br-20261021T033313-8cb182ed` lief beim ersten Versuch vollständig durch (15 gültige Läufe; keine ungültigen Läufe, keine wiederholten oder zusätzlichen Läufe, kein Seed ausgewählt).
- **Erster schreibgeschützter Wiederaufruf fehlgeschlagen:** Der erste Wiederaufruf zur schreibgeschützten Prüfung des fertigen Ergebnisses (Lauf `adv5br-20261021T034704-75a9d819`) meldete einen Fehler der **Prüflogik**: Der Plan enthielt ein Wörterbuch mit ganzzahligen Seeds als Schlüsseln; beim Registrieren wurden diese numerisch sortiert, nach dem Zurücklesen aus JSON (Textschlüssel) lexikografisch, was einen anderen kanonischen Text und damit einen anderen Hash bei identischem Inhalt ergab. Die Ergebnisse waren nicht betroffen (die Ausgabe-Hashes des Status und die Dateihash-Prüfungen blieben gültig). Das Fehlerprotokoll (`reports/tables/adversarial_roi_status_failed_attempt_adv5br-20261021T034704-75a9d819.json`) bleibt unverändert erhalten. Das Skript wurde daraufhin nur an den zwei betroffenen Prüfstellen korrigiert (neue Funktion `registered_plan_hash` und deren Aufruf); die Worker-, Trainings-, Auswertungs- und Aggregationsfunktionen blieben unverändert, und die beiden eigenen Module (`adversarial_roi.py`, `plot_adversarial_roi_results.py`) sowie alle festgelegten Fremdmodule sind byteidentisch zur Registrierung. Folge: Der Skript-Hash des korrigierten Skripts weicht von dem bei der Registrierung festgehaltenen ab (`83c0fbb89c115185…`); ein fertiges Ergebnis wird weiterhin schreibgeschützt geprüft und der Unterschied ausdrücklich gemeldet, aber Worker, Wiederaufnahme und Fortsetzung unter der bestehenden Registrierung verweigert das Code-Gate mit dem korrigierten Skript. Die ursprünglich registrierte Skriptfassung ist nicht Teil des Repositories. Die Probe mit synthetischen Daten hatte den Fehler nicht aufgedeckt, weil sie nur einen Seed (einen Schlüssel) verwendete; ein Regressionstest mit den fünf echten Seeds wurde ergänzt.
- **Statistikgrafik:** In `adversarial_roi_fgsm_curves.png` überlappen die beiden x-Achsenbeschriftungen der unteren Zeile teilweise; die Grafik wurde nach dem Lauf nicht neu erzeugt, weil das ihren Hash im Studienstatus verändert hätte. Die Zahlen sind davon nicht betroffen.
- **Vorab:** Fehler in den Funktionstests vor der Plan-Registrierung (zum Beispiel falsch gesetzte Testannahmen) wurden vor der Registrierung behoben; sie betreffen keine Studienergebnisse.

## Dateien und Wiederaufruf

| Datei | Inhalt |
|---|---|
| `analysis/08_adversarial_addon.py` | Interpreterprüfung, Vorbedingungen, Plan-Registrierung, Worker (Referenz und Training), Aggregation, schreibgeschützte Prüfung, `--resume-evaluation SEED ARM` |
| `src/busbra/adversarial_roi.py` | Trainingsschritt, Batchreihenfolge, Auswertung, gepaarte Zusammenfassungen |
| `plots/plot_adversarial_roi_results.py` | drei Statistikgrafiken aus den Tabellen (CPU) |
| `reports/tables/adversarial_roi_plan.json` | unveränderlicher Plan (Studie `adv5br-dd935b42`) |
| `reports/tables/adversarial_roi_sources.csv`, `_runs.csv` | Startmodelle mit Hashes; Läufe mit Run-IDs, Digests und Zeiten |
| `reports/tables/adversarial_roi_run_metrics.csv`, `_summary.csv`, `_attack_effect*.csv`, `_paired_*.csv`, `_training_steps.csv` | Kennzahlen je Lauf, Zusammenfassungen, Angriffswirkung, gepaarte Differenzen, Trainingsverlauf |
| `reports/tables/adversarial_roi/seed_<s>/<arm>/` | Verlauf, Konfiguration, Auswertung, Vorhersagen (nur lokal), Trainings- und Erfolgsstatus je Lauf; `original/` für die Referenzauswertung |
| `reports/tables/adversarial_roi_status.json` | Erfolgsstatus mit allen Ausgabe-Hashes (zuletzt geschrieben) |
| `reports/figures/adversarial_roi_*.png` | Statistikgrafiken (nur Kennzahlen) |
| `models/adversarial_roi/seed_<s>/<arm>/` | Zustand nach Epoche 5 (nur lokal) |

Aufruf: `env -u LD_LIBRARY_PATH .venv/bin/python analysis/08_adversarial_addon.py`. Ein erneuter Aufruf eines vollständigen Erfolgs prüft Plan, alle Ausgaben, alle Laufstatus mit Hashketten, die Startcheckpoints und alle registrierten Eingaben schreibgeschützt (kein Worker, kein Bild, keine GPU, kein TensorFlow). Artefakte ohne gültigen Erfolgsstatus werden als `recovery_required` gemeldet und nie neu trainiert, überschrieben oder bereinigt; nur die reine Auswertung eines gespeicherten Epoche-5-Checkpoints kann mit `--resume-evaluation` ausdrücklich freigegeben werden.

## Quellen

- Goodfellow, Shlens, Szegedy: Explaining and Harnessing Adversarial Examples (FGSM): https://arxiv.org/abs/1412.6572
- Madry, Makelov, Schmidt, Tsipras, Vladu: Towards Deep Learning Models Resistant to Adversarial Attacks: https://arxiv.org/abs/1706.06083
- Wong, Rice, Kolter: Fast is better than free: Revisiting adversarial training: https://arxiv.org/abs/2001.03994
- Kurs Deep Learning, Tag 19: `19_best_practices_fairness/19_02_fairness_adversarial_pytorch.ipynb` (Abschnitt Adversarial Training; Konzept, nicht der PyTorch-Code); Tag 20: `20_mini-projekt/20_01_projekt_bildklassifikation_keras.py` (nur Ausblick)
