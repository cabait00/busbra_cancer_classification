# Schritt 5C: Abschließende Auswertung des eingefrorenen Tests

Stand: 21.10.2026. Schritt 5C ist umgesetzt, ausgeführt. Er wertet die sieben Gruppen des Abschlussprotokolls (27 Kombinationen aus Modell, Seed und Eingabebedingung) unverändert auf den **376 Testbildern aus 212 Cases** aus: nur Inferenz mit den registrierten Checkpoints, die eingefrorenen Validation-Schwellen aus 5A, keine Auswahl und keine Anpassung nach Einsicht in die Ergebnisse. Grad-CAM-Hauptinterpretation (5D), FGSM-Hauptauswertung und Geräteaufschlüsselung (5E) sind **nicht** Teil dieses Schritts; die Fortsetzungsmodelle des Zusatzexperiments 5B-R gehören nicht zum Testumfang. Die Ergebnisse beschreiben ein festes Testsubset, feste Modelle und feste Schwellen; sie sind keine Aussage über klinische Eignung oder Screening.

## Kurzfassung der Befunde

- **Vollständigkeit:** 27 Kombinationen × 376 Testbilder = 10.152 Bildscores, 54 Bild-/Case-Kennzahlzeilen, 162 Kernzeilen der festen Betriebspunkte und 81 Zeilen der historischen Case-Sicht; alle Vollständigkeitsprüfungen erfüllt. Die Vorkontrolle auf Validation (vor dem ersten Testzugriff) bestand: alle 27 Kombinationen reproduzieren die gespeicherten Validation-Scores (größte Abweichung 6e-08; Toleranzen 1e-5 und 1e-12), und das U-Net reproduziert alle 385 eingefrorenen Validation-Zuschnitte exakt.
- **Bild-ROC-AUC (Mittel ± Stichproben-SD über fünf Seeds):** Experten-ROI 0,937 ± 0,008, automatische ROI 0,915 ± 0,008, Vollbild feinjustiert 0,895 ± 0,009, Vollbild frozen 0,817 ± 0,010, L2-Logistische Regression 0,688 (ein Modell), kleines CNN 0,613 ± 0,107 (zwei von fünf Seeds nahe 0,5: 0,682, 0,503, 0,690, 0,489, 0,702), Dummy 0,500.
- **Drei Hauptvergleiche (Bild-ROC-AUC, Punktschätzung [95-%-Perzentilintervall der Bootstrap-Stichprobe]):** Experten-ROI minus Vollbild +0,042 [+0,015; +0,073] (in allen fünf Seeds positiv); automatische ROI minus Experten-ROI −0,022 [−0,039; −0,007] (in allen fünf Seeds negativ); automatische ROI minus Vollbild +0,020 [−0,009; +0,050] (alle fünf Seeds positiv, das Intervall schließt aber 0 ein). Auf Case-Ebene gleiche Richtungen: +0,047 [+0,017; +0,081], −0,024 [−0,047; −0,006], +0,023 [−0,008; +0,057].
- **Feste Betriebspunkte (Bildebene, Mittel über die Seeds, benigne Fehlalarme von 254):** am Punkt für das Ziel 0,90 Vollbild 85,8, Experten-ROI 71,0, automatische ROI 97,4; am Punkt für 0,95 133,8, 115,4 und 131,8. Die auf Test erreichte Sensitivität weicht von den Validation-Zielen ab und wird offen berichtet (Abschnitt 2).
- **Segmentierung und Lokalisation (getrennt):** Dice der rohen U-Net-Vorhersage Mittel 0,870, Median 0,919; 3 leere Vorhersagen (Vollbild-Fallback), 1 Zuschnitt verfehlt die Läsion vollständig, 327 von 373 Zuschnitten ohne Fallback enthalten alle Expertenpixel.
- **Einordnung:** Die Eingrenzung durch die Expertenmaske verbessert in dieser Stichprobe die Trennschärfe gegenüber dem Vollbild; die automatische Eingrenzung behält davon einen Teil, ihr Vorteil gegenüber dem Vollbild ist in diesem Test nicht gesichert (Intervall schließt 0 ein). Alle Aussagen gelten für diesen festen Holdout, ein U-Net-Seed und Klassifikatoren, die auf Expertenzuschnitten trainiert wurden.

## Frage, Einordnung und Registrierung

Forschungsfrage: *Verbessert die Eingrenzung des Läsionsbereichs die Unterscheidung gutartiger und bösartiger Befunde im Brustultraschall?* Gleichwertig berichtet werden Vollbild gegen Experten-ROI, Experten-ROI gegen automatische ROI und Vollbild gegen automatische ROI. Die Unterfrage (bringt ein CNN gegenüber einer einfachen lernenden Baseline einen Mehrwert?) bleibt erhalten.

Die Gruppen, Kennzahlen, Schwellenregel und das Bootstrap-Verfahren sind in Schritt 5A festgelegt ([final_evaluation_protocol.md](final_evaluation_protocol.md), Plan `final5a-937da596`). Dieser Schritt hat einen **eigenen technischen Plan** (Quellen mit Hashes, Codehashes, Reihenfolge, Zugriffsregeln, Ausgabewege) registriert: `reports/tables/test_evaluation_plan.json`, Analyse `test5c-0e9e0f3f`, Plan-SHA-256 `0e9e0f3f539ded4c5bad352bdd7be2366ad89c92db9c407ef3873bd63344e3f1`, registriert am 21.10.2026 um 14:27:35. Die Registrierung erfolgte **nach der Hashprüfung der Eingaben und dem Lesen der Manifest-Metadaten und vor dem ersten Öffnen oder Hashen eines Testbildes oder einer Testmaske**; Funktionstests davor verwendeten ausschließlich synthetische Daten. Die Validation-Ergebnisse aller Gruppen waren lange vorher bekannt, und die Validation wurde wiederholt für die Entwicklung genutzt; eine Registrierung vor der Validation-Entwicklung wird nicht behauptet.

**Auslegung der Hauptvergleiche (im Plan festgelegt):** „Vollbild“ ist die feinjustierte 4B-Pipeline (erweitertes Fine-Tuning, Adam 1e-4, augmentiert), also dieselbe Feinjustierungsstufe wie die ROI-Gruppen. Das eingefrorene 4B-Frozen-Modell wird in den Tabellen als eigene Gruppe beschrieben, geht aber nicht in die drei Bootstrap-Vergleiche ein. Experten-ROI und automatische ROI verwenden dieselben fünf 4E-Klassifikatoren unter verschiedenen Eingabebedingungen; es gibt kein Ensemble und keinen besten Seed.

## Testbelegung und Vollständigkeit

Testsubset: 376 Bilder aus 212 Cases (254 benigne und 122 maligne Bilder; 144 benigne und 68 maligne Cases), höchstens zwei Bilder je Case, ein Label je Case, die Case-Mengen von Train, Validation und Test paarweise disjunkt (gegen die Audit-/Splitnachweise geprüft, `reports/tables/split_summary.csv`). In den Testzeilen gibt es keine der vier bekannten Abweichungen zwischen CSV-Höhe und dekodierter Bildgröße; es wurden keine Bilder oder Cases aussortiert, und die tatsächlich dekodierten Bildmaße steuern jede Geometrie.

| Gerät | Bilder | Cases | maligne Bilder | benigne Bilder |
|---|---:|---:|---:|---:|
| GE Logiq 7 @10-14MHz | 192 | 102 | 55 | 137 |
| GE Logiq 5 @10-12MHz | 145 | 90 | 53 | 92 |
| Toshiba Aplio 300 @12-14 MHz | 33 | 17 | 12 | 21 |
| U-Systems @10-14MHz | 6 | 3 | 2 | 4 |

Vollständigkeit:

| Prüfpunkt | erwartet | beobachtet | erfüllt |
|---|---:|---:|---:|
| test images | 376 | 376 | ja |
| test cases | 212 | 212 | ja |
| combinations with raw scores | 27 | 27 | ja |
| image score rows | 10152 | 10152 | ja |
| metric rows (combinations x image / case) | 54 | 54 | ja |
| core operating-point rows (combinations x 2 levels x 3 points) | 162 | 162 | ja |
| historical case view under the image thresholds | 81 | 81 | ja |
| frozen thresholds used | 162 | 162 | ja |
| bootstrap replicates | 2000 | 2000 | ja |
| validation control rows (one per combination) | 27 | 27 | ja |
| automatic ROIs frozen for every test image | 376 | 376 | ja |

## Verfahren

**Reihenfolge und Zugriffe.** Jede Phase ist ein eigener Prozess mit unabhängigem Zugriffsmitschnitt (Hook auf alle Dateiöffnungen). Die Reihenfolge ist im Plan festgelegt und wurde durchgesetzt: (1) Vorkontrolle auf Validation (sieben Worker: Baselines, U-Net, je Seed die fünf neuronalen Kombinationen), (2) **ROI-Phase** (das U-Net einmalig auf den 376 Testbildern, **keine Maske geöffnet**; Koordinaten, Fallback-Kennzeichen und vorhergesagte Pixel eingefroren), (3) Experten-Geometrie (Testbilder und Testmasken, erst nach dem Einfrieren), (4) Klassifikation (Rohscores; der automatische Zweig nutzt ausschließlich die eingefrorenen Koordinaten; keine Maske), (5) Segmentierungsdiagnose (nur Testmasken und die eingefrorenen Pixel), (6) Aggregation. Zugriffe laut Statusdateien: Kontrollworker 385 Validation-Bilder, keine Maske, keine Testdatei; ROI-Worker 376 Testbilder, 0 Masken; Experten-Geometrie 376 Bilder und 376 Masken; Klassifikationsworker 376 Testbilder, 0 Masken; Diagnose 376 Masken, 0 Bilder; der Koordinator öffnete keine Ultraschalldatei und importierte kein TensorFlow. Jedes Testbild und jede Testmaske wurde vor der Verwendung gegen den Dateihash des auditierten Manifests geprüft.

**Vorkontrolle (Validation, vor dem ersten Testzugriff).** Für alle 27 Kombinationen wurden Modellhash, Lade- und Eingabepfad gegen die gespeicherten Validation-Scores geprüft (neuronal höchstens 1e-5, Baselines höchstens 1e-12; Toleranzen nicht gelockert). Die Experten-Eingaben der Validation wurden aus der akzeptierten 4E-Geometrietabelle ohne Maske aufgebaut und mussten den eingefrorenen 4E-Fingerabdruck reproduzieren, die automatischen den eingefrorenen 4F-B-Fingerabdruck.

| Gruppe | Kombinationen | größte Abweichung zu den gespeicherten Validation-Scores | Toleranz |
|---|---:|---:|---:|
| Dummy (Klassenprior) | 1 | 0,00e+00 | 1e-12 |
| L2-Logistische Regression 64 px | 1 | 0,00e+00 | 1e-12 |
| Kleines CNN (3F) | 5 | 5,96e-08 | 1e-05 |
| MobileNetV2 frozen, Vollbild (4B) | 5 | 0,00e+00 | 1e-05 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 5 | 0,00e+00 | 1e-05 |
| Experten-ROI, feinjustiert (4E) | 5 | 0,00e+00 | 1e-05 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 5 | 0,00e+00 | 1e-05 |

Das U-Net (Batchgröße 8) erzeugte auf den 385 Validation-Bildern exakt die eingefrorenen 4F-B-Zuschnitte (Boxen und Fallback-Kennzeichen in 385 von 385 Fällen gleich, kein Bild mit abweichender Pixelzahl).

**Inferenz.** Neuronale Modelle in frischen Prozessen auf der GPU (Memory Growth vor der Initialisierung, weiche Platzierung aus, kein neues Speicherlimit, `jit_compile=False`, numerischer Modus wie in den akzeptierten Läufen, TF32 an), Batchgröße 16 für die Klassifikatoren, 8 für das U-Net; geordneter Datensatz mit Nullen statt Labels, Inferenzmodus, keine Augmentation, kein Dropout, keine Test-Time-Augmentation, keine zusätzliche Normalisierung. Modellhash und Variablenfingerabdruck vor und nach jeder Inferenz waren gleich; es gab keinen Optimierungsschritt. Die Baselines (scikit-learn) liefen auf der CPU. Rohscores werden mit kürzester Roundtrip-Darstellung gespeichert, bevor Zusammenfassungen entstehen. Laufzeiten (Worker): Kontrolle je Seed 15,1 s, ROI-Phase 8,9 s, Klassifikation je Seed 14,9 s, Bootstrap 55,3 s; Gesamtdauer des Aufrufs 241,2 s.

**Schwellen und Kennzahlen.** Positive Klasse malignant = 1, Entscheidung `Score >= Schwelle`. Die Schwellen sind die **eingefrorenen 5A-Werte** (Bild- und Case-Ebene, Referenz 0,5 und die Ziele 0,90 und 0,95), aus ihrem exakten Hex-Text gelesen (der Dezimaltext liest sich zum selben Double zurück); keine Schwelle wird aus Testlabels berechnet, gemittelt oder neu gewählt. „Ziel 0,90 / 0,95“ bezeichnet die Auswahl auf Validation; die tatsächlich erreichte Test-Sensitivität wird berichtet, und Fehlalarme an einem festen Punkt sind kein Vergleich bei garantiert gleicher Sensitivität. Der Case-Score ist das Maximum der Bildscores. Undefinierte Größen bleiben leer. Je Gruppe werden die Einzelwerte, n, Mittel und Stichproben-SD (ddof = 1) berichtet; keine Pooling-AUC, kein Ensemble, die Seeds sind keine unabhängigen Kohorten.

**Cluster-Bootstrap (nur für die drei Hauptvergleiche).** 2.000 gepaarte, nach Case-Label geschichtete Ziehungen mit RNG-Seed 2026 (`numpy.random.default_rng`; je Ziehung zuerst 68 Ziehungen aus den malignen, dann 144 aus den benignen Cases, jeweils mit Zurücklegen; alle Bilder eines Cases bleiben zusammen, mehrfach gezogene Cases zählen mehrfach). Dieselben Ziehungen gelten für alle Gruppen, alle fünf Seeds und beide Ebenen. Je Ziehung wird die Kennzahl zuerst je Seed bestimmt, danach der gepaarte Unterschied (andere minus Referenz) über die Seeds gemittelt; die Intervalle sind die linearen 2,5- und 97,5-Perzentile der 2.000 Seed-Mittel. Die Schwellen der Ziele bleiben auch im Bootstrap unverändert. Die Intervalle beschreiben die **Stichprobenunsicherheit** der 376 Testbilder für feststehende Modelle und Schwellen; die Seed-SD ist eine andere Größe, und weder eine neue Trainings- oder Schwellenwahl noch eine externe Klinik sind abgedeckt. Die Ziehungen (Indizes), die Replikatwerte je Gruppe, Seed und Kennzahl und die Seed-Mittel der Unterschiede sind gespeichert.

## Ergebnisse

### 1. Alle sieben Gruppen

ROC-AUC und Average Precision (fünf Seed-Einzelwerte, Mittel ± Stichproben-SD; Dummy und LogReg sind einzelne Modelle ohne Seed):

| Gruppe | Bild-ROC-AUC je Seed (7 / 21 / 42 / 123 / 2026) | Bild-ROC-AUC Mittel ± SD | Bild-AP Mittel ± SD | Case-ROC-AUC je Seed | Case-ROC-AUC Mittel ± SD | Case-AP Mittel ± SD |
|---|---:|---:|---:|---:|---:|---:|
| Dummy (Klassenprior) | 0,500 | 0,500 ± – | 0,324 ± – | 0,500 | 0,500 ± – | 0,321 ± – |
| L2-Logistische Regression 64 px | 0,688 | 0,688 ± – | 0,544 ± – | 0,711 | 0,711 ± – | 0,563 ± – |
| Kleines CNN (3F) | 0,682 / 0,503 / 0,690 / 0,489 / 0,702 | 0,613 ± 0,107 | 0,465 ± 0,106 | 0,680 / 0,512 / 0,688 / 0,494 / 0,695 | 0,614 ± 0,102 | 0,459 ± 0,102 |
| MobileNetV2 frozen, Vollbild (4B) | 0,812 / 0,819 / 0,834 / 0,806 / 0,816 | 0,817 ± 0,010 | 0,697 ± 0,018 | 0,815 / 0,814 / 0,842 / 0,805 / 0,824 | 0,820 ± 0,014 | 0,704 ± 0,021 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 0,881 / 0,907 / 0,895 / 0,896 / 0,895 | 0,895 ± 0,009 | 0,814 ± 0,018 | 0,885 / 0,912 / 0,891 / 0,888 / 0,903 | 0,896 ± 0,012 | 0,823 ± 0,020 |
| Experten-ROI, feinjustiert (4E) | 0,946 / 0,934 / 0,940 / 0,926 / 0,941 | 0,937 ± 0,008 | 0,888 ± 0,013 | 0,948 / 0,938 / 0,944 / 0,940 / 0,946 | 0,943 ± 0,004 | 0,898 ± 0,009 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 0,929 / 0,911 / 0,914 / 0,910 / 0,911 | 0,915 ± 0,008 | 0,852 ± 0,015 | 0,933 / 0,914 / 0,914 / 0,920 / 0,913 | 0,919 ± 0,008 | 0,867 ± 0,016 |

Die Validation-ROC-AUC (Seed-Mittel, Schritt 5A) lag bei LogReg 0,669, kleines CNN 0,661, Vollbild frozen 0,797, Vollbild feinjustiert 0,881, Experten-ROI 0,921, automatische ROI 0,898 (Bildebene); die Testwerte liegen bei den meisten Gruppen nahe daran oder leicht darüber, beim kleinen CNN darunter. Das ist eine Beobachtung an diesem Holdout, keine Aussage über Optimismus oder Stabilität.

Referenzschwelle 0,5 (Bildebene, Mittel ± SD über die Seeds; FP und FN als mittlere absolute Zahlen von 254 benignen beziehungsweise 122 malignen Bildern):

| Gruppe | Sensitivität | Spezifität | Precision | F1 | Accuracy | FP (Mittel) | FN (Mittel) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy (Klassenprior) | 0,000 ± – | 1,000 ± – | – | 0,000 ± – | 0,676 ± – | 0,0 | 122,0 |
| L2-Logistische Regression 64 px | 0,475 ± – | 0,807 ± – | 0,542 ± – | 0,507 ± – | 0,699 ± – | 49,0 | 64,0 |
| Kleines CNN (3F) | 0,107 ± 0,101 | 0,985 ± 0,014 | 0,773 ± 0,030 | 0,172 ± 0,162 | 0,700 ± 0,024 | 3,8 | 109,0 |
| MobileNetV2 frozen, Vollbild (4B) | 0,556 ± 0,112 | 0,876 ± 0,048 | 0,694 ± 0,066 | 0,607 ± 0,066 | 0,772 ± 0,017 | 31,6 | 54,2 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 0,707 ± 0,082 | 0,894 ± 0,027 | 0,764 ± 0,025 | 0,731 ± 0,034 | 0,833 ± 0,011 | 27,0 | 35,8 |
| Experten-ROI, feinjustiert (4E) | 0,723 ± 0,099 | 0,941 ± 0,029 | 0,861 ± 0,051 | 0,780 ± 0,051 | 0,870 ± 0,019 | 15,0 | 33,8 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 0,666 ± 0,063 | 0,924 ± 0,029 | 0,814 ± 0,049 | 0,729 ± 0,029 | 0,840 ± 0,011 | 19,2 | 40,8 |

### 2. Feste Betriebspunkte

Die Schwellen für die Ziele 0,90 und 0,95 stammen aus der Validation. Gemessen wird, was sie auf Test erreichen (Bildebene, Mittel ± SD über die Seeds; FP und FN als mittlere absolute Zahlen):

| Gruppe | Ziel 0,90: Sensitivität | Spezifität | FP | FN | Ziel 0,95: Sensitivität | Spezifität | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (Klassenprior) | 1,000 ± – | 0,000 ± – | 254,0 | 0,0 | 1,000 ± – | 0,000 ± – | 254,0 | 0,0 |
| L2-Logistische Regression 64 px | 0,885 ± – | 0,248 ± – | 191,0 | 14,0 | 0,984 ± – | 0,122 ± – | 223,0 | 2,0 |
| Kleines CNN (3F) | 0,854 ± 0,037 | 0,228 ± 0,077 | 196,2 | 17,8 | 0,936 ± 0,011 | 0,125 ± 0,070 | 222,2 | 7,8 |
| MobileNetV2 frozen, Vollbild (4B) | 0,913 ± 0,011 | 0,448 ± 0,030 | 140,2 | 10,6 | 0,951 ± 0,017 | 0,325 ± 0,052 | 171,4 | 6,0 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 0,905 ± 0,043 | 0,662 ± 0,073 | 85,8 | 11,6 | 0,969 ± 0,015 | 0,473 ± 0,093 | 133,8 | 3,8 |
| Experten-ROI, feinjustiert (4E) | 0,943 ± 0,027 | 0,720 ± 0,050 | 71,0 | 7,0 | 0,980 ± 0,017 | 0,546 ± 0,063 | 115,4 | 2,4 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 0,941 ± 0,027 | 0,617 ± 0,030 | 97,4 | 7,2 | 0,979 ± 0,016 | 0,481 ± 0,053 | 131,8 | 2,6 |

Case-Ebene (212 Cases; 144 benigne, 68 maligne; Case-Schwellen aus 5A):

| Gruppe | Ziel 0,90: Sensitivität | Spezifität | FP | FN | Ziel 0,95: Sensitivität | Spezifität | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (Klassenprior) | 1,000 ± – | 0,000 ± – | 144,0 | 0,0 | 1,000 ± – | 0,000 ± – | 144,0 | 0,0 |
| L2-Logistische Regression 64 px | 0,956 ± – | 0,229 ± – | 111,0 | 3,0 | 0,956 ± – | 0,125 ± – | 126,0 | 3,0 |
| Kleines CNN (3F) | 0,847 ± 0,045 | 0,250 ± 0,084 | 108,0 | 10,4 | 0,956 ± 0,015 | 0,110 ± 0,079 | 128,2 | 3,0 |
| MobileNetV2 frozen, Vollbild (4B) | 0,865 ± 0,022 | 0,522 ± 0,043 | 68,8 | 9,2 | 0,900 ± 0,030 | 0,429 ± 0,063 | 82,2 | 6,8 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 0,879 ± 0,046 | 0,744 ± 0,074 | 36,8 | 8,2 | 0,932 ± 0,054 | 0,590 ± 0,142 | 59,0 | 4,6 |
| Experten-ROI, feinjustiert (4E) | 0,926 ± 0,029 | 0,756 ± 0,074 | 35,2 | 5,0 | 0,976 ± 0,013 | 0,604 ± 0,056 | 57,0 | 1,6 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 0,903 ± 0,027 | 0,743 ± 0,046 | 37,0 | 6,6 | 0,950 ± 0,030 | 0,551 ± 0,039 | 64,6 | 3,4 |

Validation (5A) und Test nebeneinander, Bildebene, Seed-Mittel:

| Gruppe | Ziel 0,90: Sens. Validation | Sens. Test | Spez. Validation | Spez. Test | Ziel 0,95: Sens. Validation | Sens. Test | Spez. Validation | Spez. Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (Klassenprior) | 1,000 | 1,000 | 0,000 | 0,000 | 1,000 | 1,000 | 0,000 | 0,000 |
| L2-Logistische Regression 64 px | 0,904 | 0,885 | 0,346 | 0,248 | 0,952 | 0,984 | 0,162 | 0,122 |
| Kleines CNN (3F) | 0,904 | 0,854 | 0,246 | 0,228 | 0,952 | 0,936 | 0,149 | 0,125 |
| MobileNetV2 frozen, Vollbild (4B) | 0,904 | 0,913 | 0,441 | 0,448 | 0,952 | 0,951 | 0,315 | 0,325 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 0,904 | 0,905 | 0,654 | 0,662 | 0,952 | 0,969 | 0,435 | 0,473 |
| Experten-ROI, feinjustiert (4E) | 0,904 | 0,943 | 0,780 | 0,720 | 0,952 | 0,980 | 0,552 | 0,546 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 0,904 | 0,941 | 0,658 | 0,617 | 0,952 | 0,979 | 0,513 | 0,481 |

Einzelwerte je Kombination an den festen Bildpunkten:

| Gruppe | Seed | Ziel 0,90: Sens. | Spez. | FP | FN | Ziel 0,95: Sens. | Spez. | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (Klassenprior) | – | 1,000 | 0,000 | 254 | 0 | 1,000 | 0,000 | 254 | 0 |
| L2-Logistische Regression 64 px | – | 0,885 | 0,248 | 191 | 14 | 0,984 | 0,122 | 223 | 2 |
| Kleines CNN (3F) | 7 | 0,885 | 0,264 | 187 | 14 | 0,943 | 0,177 | 209 | 7 |
| Kleines CNN (3F) | 21 | 0,803 | 0,157 | 214 | 24 | 0,918 | 0,075 | 235 | 10 |
| Kleines CNN (3F) | 42 | 0,852 | 0,303 | 177 | 18 | 0,934 | 0,177 | 209 | 8 |
| Kleines CNN (3F) | 123 | 0,836 | 0,134 | 220 | 20 | 0,943 | 0,028 | 247 | 7 |
| Kleines CNN (3F) | 2026 | 0,893 | 0,280 | 183 | 13 | 0,943 | 0,169 | 211 | 7 |
| MobileNetV2 frozen, Vollbild (4B) | 7 | 0,918 | 0,445 | 141 | 10 | 0,934 | 0,394 | 154 | 8 |
| MobileNetV2 frozen, Vollbild (4B) | 21 | 0,902 | 0,500 | 127 | 12 | 0,959 | 0,327 | 171 | 5 |
| MobileNetV2 frozen, Vollbild (4B) | 42 | 0,926 | 0,421 | 147 | 9 | 0,975 | 0,248 | 191 | 3 |
| MobileNetV2 frozen, Vollbild (4B) | 123 | 0,902 | 0,433 | 144 | 12 | 0,934 | 0,319 | 173 | 8 |
| MobileNetV2 frozen, Vollbild (4B) | 2026 | 0,918 | 0,441 | 142 | 10 | 0,951 | 0,339 | 168 | 6 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 7 | 0,861 | 0,709 | 74 | 17 | 0,967 | 0,441 | 142 | 4 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 21 | 0,959 | 0,583 | 106 | 5 | 0,992 | 0,339 | 168 | 1 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 42 | 0,885 | 0,713 | 73 | 14 | 0,967 | 0,469 | 135 | 4 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 123 | 0,877 | 0,724 | 70 | 15 | 0,951 | 0,575 | 108 | 6 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 2026 | 0,943 | 0,583 | 106 | 7 | 0,967 | 0,543 | 116 | 4 |
| Experten-ROI, feinjustiert (4E) | 7 | 0,959 | 0,717 | 72 | 5 | 0,992 | 0,535 | 118 | 1 |
| Experten-ROI, feinjustiert (4E) | 21 | 0,910 | 0,756 | 62 | 11 | 0,951 | 0,602 | 101 | 6 |
| Experten-ROI, feinjustiert (4E) | 42 | 0,959 | 0,689 | 79 | 5 | 0,984 | 0,445 | 141 | 2 |
| Experten-ROI, feinjustiert (4E) | 123 | 0,967 | 0,657 | 87 | 4 | 0,984 | 0,551 | 114 | 2 |
| Experten-ROI, feinjustiert (4E) | 2026 | 0,918 | 0,783 | 55 | 10 | 0,992 | 0,594 | 103 | 1 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 7 | 0,951 | 0,638 | 92 | 6 | 0,984 | 0,531 | 119 | 2 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 21 | 0,934 | 0,618 | 97 | 8 | 0,984 | 0,472 | 134 | 2 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 42 | 0,943 | 0,583 | 106 | 7 | 0,992 | 0,394 | 154 | 1 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 123 | 0,975 | 0,591 | 104 | 3 | 0,984 | 0,500 | 127 | 2 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 2026 | 0,902 | 0,654 | 88 | 12 | 0,951 | 0,508 | 125 | 6 |

Beobachtungen: Das Ziel 0,90 wurde auf Test beim kleinen CNN (Sensitivität 0,854) und bei der LogReg (0,885) unterschritten, das Ziel 0,95 beim kleinen CNN (0,936); die übrigen Gruppen liegen im Seed-Mittel darüber, einzelne Seeds können darunter liegen (Tabelle der Einzelwerte). Die Spezifität an den festen Punkten ist bei der Experten-ROI und der automatischen ROI auf Test niedriger als auf Validation (Ziel 0,90: 0,780 gegenüber 0,720 bei der Experten-ROI), bei der Sensitivität sind sie höher. Der Dummy ordnet an den Zielpunkten alle Einheiten positiv ein (Sensitivität 1, Spezifität 0), weil sein konstanter Score der Schwelle entspricht; das ist ein korrektes Ergebnis, kein Fehler.

Historische Vergleichssicht (Bildschwellen auf die Case-Scores angewandt, getrennt gekennzeichnet; nicht Teil der 162 Kernzeilen):

| Gruppe | 0,5: Case-Sens. | Case-Spez. | Bildschwelle Ziel 0,90: Case-Sens. | Case-Spez. | Bildschwelle Ziel 0,95: Case-Sens. | Case-Spez. |
|---|---:|---:|---:|---:|---:|---:|
| Dummy (Klassenprior) | 0,000 | 1,000 | 1,000 | 0,000 | 1,000 | 0,000 |
| L2-Logistische Regression 64 px | 0,632 | 0,722 | 0,956 | 0,118 | 1,000 | 0,062 |
| Kleines CNN (3F) | 0,132 | 0,974 | 0,888 | 0,208 | 0,950 | 0,122 |
| MobileNetV2 frozen, Vollbild (4B) | 0,691 | 0,814 | 0,938 | 0,308 | 0,962 | 0,189 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 0,812 | 0,851 | 0,938 | 0,542 | 0,971 | 0,328 |
| Experten-ROI, feinjustiert (4E) | 0,794 | 0,924 | 0,971 | 0,650 | 0,991 | 0,464 |
| Automatische ROI, feinjustiert (U-Net → 4E-Klassifikator) | 0,776 | 0,894 | 0,956 | 0,533 | 0,985 | 0,390 |

### 3. Die drei Hauptvergleiche

Differenzen sind „andere minus Referenz“ (Vergleich Experten-ROI − Vollbild: Referenz Vollbild feinjustiert). Punktschätzung = Mittel über die fünf Seeds der gepaarten Differenzen; Intervall = zweiseitiges 95-%-Perzentilintervall des Cluster-Bootstraps; „ohne 0“ beschreibt nur, ob das Intervall die Null ausschließt (keine Signifikanzaussage); die Seed-SD ist eine andere Größe.

**ROC-AUC:**

| Kennzahl | Vergleich (andere minus Referenz) | Punktschätzung (Seed-Mittel) | 95-%-Intervall (Perzentil) | Intervall ohne 0 | Seed-SD |
|---|---|---:|---:|---:|---:|
| Bild-ROC-AUC | Experten-ROI − Vollbild | +0,042 | [+0,015; +0,073] | ja | 0,015 |
| Bild-ROC-AUC | Automatische ROI − Experten-ROI | −0,022 | [−0,039; −0,007] | ja | 0,006 |
| Bild-ROC-AUC | Automatische ROI − Vollbild | +0,020 | [−0,009; +0,050] | nein | 0,016 |
| Case-ROC-AUC | Experten-ROI − Vollbild | +0,047 | [+0,017; +0,081] | ja | 0,014 |
| Case-ROC-AUC | Automatische ROI − Experten-ROI | −0,024 | [−0,047; −0,006] | ja | 0,007 |
| Case-ROC-AUC | Automatische ROI − Vollbild | +0,023 | [−0,008; +0,057] | nein | 0,018 |

Die einzelnen Seeds (Differenz je Seed, Mittel, Seed-SD, Zahl der positiven / negativen / gleichen Differenzen):

| Vergleich | Ebene | Differenz je Seed | Mittel | Seed-SD | positiv / negativ / gleich |
|---|---|---:|---:|---:|---:|
| Experten-ROI − Vollbild | image | +0,065 / +0,027 / +0,045 / +0,030 / +0,046 | +0,042 | 0,015 | 5 / 0 / 0 |
| Experten-ROI − Vollbild | case | +0,064 / +0,026 / +0,053 / +0,052 / +0,043 | +0,047 | 0,014 | 5 / 0 / 0 |
| Automatische ROI − Experten-ROI | image | −0,017 / −0,022 / −0,026 / −0,016 / −0,030 | −0,022 | 0,006 | 0 / 5 / 0 |
| Automatische ROI − Experten-ROI | case | −0,016 / −0,024 / −0,030 / −0,020 / −0,032 | −0,024 | 0,007 | 0 / 5 / 0 |
| Automatische ROI − Vollbild | image | +0,048 / +0,005 / +0,019 / +0,014 / +0,016 | +0,020 | 0,016 | 5 / 0 / 0 |
| Automatische ROI − Vollbild | case | +0,048 / +0,002 / +0,024 / +0,032 / +0,011 | +0,023 | 0,018 | 5 / 0 / 0 |

**Sensitivität und Spezifität an den festen Zielpunkten, Bildebene** (eingefrorene 5A-Schwellen je Modell):

| Kennzahl | Vergleich (andere minus Referenz) | Punktschätzung (Seed-Mittel) | 95-%-Intervall (Perzentil) | Intervall ohne 0 | Seed-SD |
|---|---|---:|---:|---:|---:|
| image: Sensitivität bei Ziel 0,90 | Experten-ROI − Vollbild | +0,038 | [−0,000; +0,079] | nein | 0,069 |
| image: Sensitivität bei Ziel 0,90 | Automatische ROI − Experten-ROI | −0,002 | [−0,030; +0,023] | nein | 0,018 |
| image: Sensitivität bei Ziel 0,90 | Automatische ROI − Vollbild | +0,036 | [−0,003; +0,076] | nein | 0,065 |
| image: Spezifität bei Ziel 0,90 | Experten-ROI − Vollbild | +0,058 | [+0,002; +0,117] | ja | 0,121 |
| image: Spezifität bei Ziel 0,90 | Automatische ROI − Experten-ROI | −0,104 | [−0,136; −0,071] | ja | 0,031 |
| image: Spezifität bei Ziel 0,90 | Automatische ROI − Vollbild | −0,046 | [−0,104; +0,013] | nein | 0,094 |
| image: Sensitivität bei Ziel 0,95 | Experten-ROI − Vollbild | +0,011 | [−0,006; +0,033] | nein | 0,030 |
| image: Sensitivität bei Ziel 0,95 | Automatische ROI − Experten-ROI | −0,002 | [−0,017; +0,017] | nein | 0,027 |
| image: Sensitivität bei Ziel 0,95 | Automatische ROI − Vollbild | +0,010 | [−0,014; +0,042] | nein | 0,021 |
| image: Spezifität bei Ziel 0,95 | Experten-ROI − Vollbild | +0,072 | [+0,009; +0,134] | ja | 0,118 |
| image: Spezifität bei Ziel 0,95 | Automatische ROI − Experten-ROI | −0,065 | [−0,099; −0,031] | ja | 0,047 |
| image: Spezifität bei Ziel 0,95 | Automatische ROI − Vollbild | +0,008 | [−0,053; +0,069] | nein | 0,098 |

**Case-Ebene** (eingefrorene 5A-Case-Schwellen):

| Kennzahl | Vergleich (andere minus Referenz) | Punktschätzung (Seed-Mittel) | 95-%-Intervall (Perzentil) | Intervall ohne 0 | Seed-SD |
|---|---|---:|---:|---:|---:|
| case: Sensitivität bei Ziel 0,90 | Experten-ROI − Vollbild | +0,047 | [−0,006; +0,103] | nein | 0,072 |
| case: Sensitivität bei Ziel 0,90 | Automatische ROI − Experten-ROI | −0,024 | [−0,062; +0,015] | nein | 0,008 |
| case: Sensitivität bei Ziel 0,90 | Automatische ROI − Vollbild | +0,024 | [−0,026; +0,079] | nein | 0,067 |
| case: Spezifität bei Ziel 0,90 | Experten-ROI − Vollbild | +0,011 | [−0,050; +0,072] | nein | 0,135 |
| case: Spezifität bei Ziel 0,90 | Automatische ROI − Experten-ROI | −0,013 | [−0,051; +0,025] | nein | 0,047 |
| case: Spezifität bei Ziel 0,90 | Automatische ROI − Vollbild | −0,001 | [−0,064; +0,062] | nein | 0,109 |
| case: Sensitivität bei Ziel 0,95 | Experten-ROI − Vollbild | +0,044 | [+0,012; +0,082] | ja | 0,060 |
| case: Sensitivität bei Ziel 0,95 | Automatische ROI − Experten-ROI | −0,026 | [−0,062; +0,000] | nein | 0,026 |
| case: Sensitivität bei Ziel 0,95 | Automatische ROI − Vollbild | +0,018 | [−0,015; +0,053] | nein | 0,071 |
| case: Spezifität bei Ziel 0,95 | Experten-ROI − Vollbild | +0,014 | [−0,056; +0,085] | nein | 0,183 |
| case: Spezifität bei Ziel 0,95 | Automatische ROI − Experten-ROI | −0,053 | [−0,094; −0,013] | ja | 0,055 |
| case: Spezifität bei Ziel 0,95 | Automatische ROI − Vollbild | −0,039 | [−0,111; +0,039] | nein | 0,173 |

**Absolute Fehlerzahlen an den festen Bildpunkten** (Mittel über die Seeds der gepaarten Differenzen; in Klammern Seeds mit positiver / negativer / gleicher Differenz):

| Vergleich | Punkt (Bildebene) | Δ FP | Δ FN | Δ Sensitivität | Δ Spezifität |
|---|---|---:|---:|---:|---:|
| Experten-ROI − Vollbild | Ziel 0,90 | −14,8 (2/3/0) | −4,6 (2/3/0) | +0,038 (3/2/0) | +0,058 (3/2/0) |
| Experten-ROI − Vollbild | Ziel 0,95 | −18,4 (2/3/0) | −1,4 (1/4/0) | +0,011 (4/1/0) | +0,072 (3/2/0) |
| Automatische ROI − Experten-ROI | Ziel 0,90 | +26,4 (5/0/0) | +0,2 (3/2/0) | −0,002 (2/3/0) | −0,104 (0/5/0) |
| Automatische ROI − Experten-ROI | Ziel 0,95 | +16,4 (5/0/0) | +0,2 (2/2/1) | −0,002 (2/2/1) | −0,065 (0/5/0) |
| Automatische ROI − Vollbild | Ziel 0,90 | +11,6 (3/2/0) | −4,4 (2/3/0) | +0,036 (3/2/0) | −0,046 (2/3/0) |
| Automatische ROI − Vollbild | Ziel 0,95 | −2,0 (3/2/0) | −1,2 (2/3/0) | +0,010 (3/2/0) | +0,008 (2/3/0) |

Experten-ROI minus Vollbild: Bild-ROC-AUC +0,042 [+0,015; +0,073], Case-ROC-AUC +0,047 [+0,017; +0,081]; an den festen Punkten weniger Fehlalarme im Mittel (−14,8 bei 0,90 und −18,4 bei 0,95), aber nicht in allen Seeds. Automatische ROI minus Experten-ROI: Bild-ROC-AUC −0,022 [−0,039; −0,007]; im Mittel +26,4 beziehungsweise +16,4 Fehlalarme mehr (in allen fünf Seeds) bei unveränderter Sensitivität. Automatische ROI minus Vollbild: Bild-ROC-AUC +0,020 [−0,009; +0,050]; an den festen Punkten uneinheitlich (+11,6 bzw. −2,0 Fehlalarme im Mittel, Intervalle der Spezifität schließen 0 ein).

### 4. Segmentierung und Lokalisation (getrennt von der Klassifikation)

Die Diagnose wurde nach dem Einfrieren der automatischen ROIs aus den Testmasken und den gespeicherten Vorhersagen berechnet und hat keine Zuschnittsregel, keine Schwelle und keine Komponentenregel verändert. Zwei Maße werden nicht verwechselt: (a) die **rohe Vorhersage** (alle vorhergesagten Pixel, Definition von 4F-A), (b) der **tatsächlich verwendete Zuschnitt** (größte Komponente plus 20 %, Definition von 4F-B); Fallbacks werden getrennt ausgewiesen.

Dice und IoU der rohen Vorhersage auf der Originalbildfläche (ohne Padding), mit Nennern:

| Gruppe | Bilder | Cases | Dice Mittel | Dice Median | IoU Mittel | leere Vorhersagen | Dice < 0,5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| alle Testbilder | 376 | 212 | 0,870 | 0,919 | 0,793 | 3 | 13 |
| benigne | 254 | 144 | 0,885 | 0,930 | 0,814 | 2 | 6 |
| maligne | 122 | 68 | 0,839 | 0,892 | 0,749 | 1 | 7 |
| GE Logiq 5 @10-12MHz | 145 | 90 | 0,894 | 0,930 | 0,821 | 0 | 2 |
| GE Logiq 7 @10-14MHz | 192 | 102 | 0,875 | 0,918 | 0,797 | 1 | 7 |
| Toshiba Aplio 300 @12-14 MHz | 33 | 17 | 0,735 | 0,840 | 0,641 | 2 | 4 |
| U-Systems @10-14MHz | 6 | 3 | 0,893 | 0,913 | 0,812 | 0 | 0 |

Kategorien der tatsächlich verwendeten Zuschnitte (Abdeckung der Expertenmaske):

| Kategorie des tatsächlich verwendeten Zuschnitts | Bilder | Cases | Anteil der 376 Bilder |
|---|---:|---:|---:|
| alle Expertenpixel im Zuschnitt | 327 | 201 | 0,870 |
| 90 bis unter 100 % im Zuschnitt | 30 | 28 | 0,080 |
| über 0 bis unter 90 % im Zuschnitt | 15 | 13 | 0,040 |
| kein Expertenpixel im Zuschnitt (Läsion verfehlt) | 1 | 1 | 0,003 |
| Vollbild-Fallback (leere Vorhersage) | 3 | 3 | 0,008 |

Abdeckung der Expertenmaske durch den Zuschnitt (ohne Fallback-Bilder ist die Lokalisationsgüte, die zweite Zeile zählt Fallbacks als volle Abdeckung und ist keine Lokalisationsgüte):

| Basis | Bilder | Abdeckung Mittel | Median | unter 100 % | unter 90 % | null |
|---|---:|---:|---:|---:|---:|---:|
| ohne Fallback-Bilder (Lokalisationsgüte) | 373 | 0,980 | 1,000 | 46 | 16 | 1 |
| Fallback als volle Abdeckung gezählt (keine Lokalisationsgüte) | 376 | 0,980 | 1,000 | 46 | 16 | 1 |

Dieselben Bilder mit vier Boxdefinitionen (Bilder ohne Fallback):

| Boxdefinition (Bilder ohne Fallback) | Bilder | unter 100 % | unter 90 % | null | Mittel | Median |
|---|---:|---:|---:|---:|---:|---:|
| box around ALL predicted pixels, no expansion (the 4F-A measure) | 373 | 336 | 55 | 0 | 0,948 | 0,985 |
| box around ALL predicted pixels expanded by 20 % (not used for any crop) | 373 | 37 | 11 | 0 | 0,988 | 1,000 |
| tight box of the kept (largest) component, no expansion | 373 | 341 | 63 | 2 | 0,937 | 0,983 |
| the ACTUAL crop of this step: largest component plus 20 % (fallback images excluded) | 373 | 46 | 16 | 1 | 0,980 | 1,000 |

Merkmale mit Nennern:

| Merkmal | Bilder | Cases | Nenner (Bilder) | Nennerbasis |
|---|---:|---:|---:|---:|
| the predicted mask is empty: the whole image is the crop (logged fallback) | 3 | 3 | 376 | all images of the group |
| more than one 8-connected component in the prediction (only the largest is kept) | 58 | 50 | 376 | all images of the group |
| several components share the largest pixel count (row-major rule decides) | 0 | 0 | 376 | all images of the group |
| less than 90 % of the expert-mask pixels lie inside the actual crop | 16 | 14 | 373 | images without fallback |
| not every expert-mask pixel lies inside the actual crop | 46 | 40 | 373 | images without fallback |
| no expert-mask pixel lies inside the actual crop (the crop misses the lesion) | 1 | 1 | 373 | images without fallback |
| Dice between the kept component and the expert mask below 0.5 | 10 | 9 | 373 | images without fallback |
| the crop is enlarged by a factor of at least 2 (interpolation, visibly blurred) | 16 | 13 | 376 | all images of the group |
| the crop is enlarged by a factor of at least 3 | 3 | 2 | 376 | all images of the group |
| the 20 % expansion is shortened by the image border | 58 | 46 | 373 | images without fallback |
| the tight box of the kept component touches an image border | 1 | 1 | 373 | images without fallback |
| the crop covers at least 90 % of the image area (practically a full image) | 0 | 0 | 373 | images without fallback |

Einzelfälle: Die drei Vollbild-Fallbacks sind bus_0870-r, bus_1025-r, bus_1047-l (benigne, maligne beziehungsweise benigne; zwei davon am Gerät Toshiba Aplio 300). Deren mittlere Scores über die fünf Seeds sind bei der automatischen ROI bus_0870-r: 0,56, bus_1025-r: 0,57, bus_1047-l: 0,69 (Experten-ROI bus_0870-r: 0,05, bus_1025-r: 0,39, bus_1047-l: 0,22); der Klassifikator sieht bei einem Fallback ein ganzes Bild, auf dem er nicht trainiert wurde. Der Zuschnitt, der die Läsion vollständig verfehlt, gehört zu bus_0874-l (benigne; mittlerer Score der automatischen ROI 0,34).

## Einordnung zur Forschungsfrage

- **Eingrenzung durch die Expertenmaske.** Auf diesem Testsubset trennt die Experten-ROI besser als das Vollbild: Bild-ROC-AUC +0,042 [+0,015; +0,073], in allen fünf Seeds positiv, auf Case-Ebene +0,047 [+0,017; +0,081]. An den festen Punkten ist die Spezifität bei beiden Zielen im Mittel höher, die Sensitivitätsunterschiede sind klein, und mehrere Intervalle schließen 0 ein. Das ist eine manuell lokalisierte Informationsbedingung: Die Anwendung erfordert eine Expertenmaske, das Ergebnis ist keine reine Bild-zu-Score-Leistung.
- **Automatische Eingrenzung.** Die U-Net-Zuschnitte erhalten einen Teil des Vorteils: gegenüber dem Vollbild +0,020 [−0,009; +0,050] (Intervall schließt 0 ein; in einem Seed nur +0,005), gegenüber der Experten-ROI −0,022 [−0,039; −0,007]. An den festen Punkten bedeuten die automatischen Zuschnitte im Mittel +26,4 beziehungsweise +16,4 Fehlalarme gegenüber der Experten-ROI. Ein Vorteil der automatisch anwendbaren Pipeline gegenüber dem Vollbild ist in diesem Test **nicht gesichert**.
- **Lokalisationsqualität.** Das U-Net lokalisiert in den meisten Bildern gut (Dice-Median 0,919), aber 13 Bilder haben einen Dice unter 0,5, 3 Vorhersagen sind leer und 1 Zuschnitt verfehlt die Läsion; die Fallback-Bilder werden von einem Klassifikator bewertet, der nur auf Zuschnitten trainiert wurde.
- **Unterfrage.** Das kleine CNN ist auf Test **nicht** besser als die einfache lernende Baseline (Bild-ROC-AUC 0,613 ± 0,107 gegenüber 0,688 bei der LogReg; zwei Seeds nahe 0,5). Die Transfer-Learning-Pipelines liegen deutlich darüber (frozen 0,817, feinjustiert 0,895).
- **Streuung.** Die Seed-SD der feinjustierten Gruppen ist klein (0,008 bis 0,009), die des kleinen CNN groß (0,107). Trainingsstreuung auf demselben Holdout ist keine Streuung über neue Kohorten.

## Grenzen

- Ein fester Holdout aus einer Datenquelle (eine Einrichtung, vier Geräte); keine externe Validierung. Mehrere Bilder eines Cases sind abhängig; der Cluster-Bootstrap berücksichtigt das, die Seed-SD nicht.
- Die Intervalle beschreiben nur die Stichprobenunsicherheit der Testbilder bei festen Modellen und Schwellen; Trainings-, Entwicklungs- und Schwellenunsicherheit sind nicht erfasst. Mehrfachvergleiche (30 Kennzahlen) sind nicht korrigiert; „ohne 0“ ist keine Signifikanzaussage.
- Die Validation wurde wiederholt für die Entwicklung genutzt; die Schwellen stammen aus 385 Validation-Bildern. Die Test-Sensitivität an den Zielpunkten ist ein Ergebnis, kein Beweis des Ziels.
- Experten-ROI: manuelle Maske nötig. Automatische ROI: ein U-Net-Seed, Klassifikatoren auf Expertenzuschnitten trainiert, Fallbacks mit Vollbildeingabe.
- Eingebrannte Texte und Messmarker sind mögliche Shortcuts und nicht entfernt; die Gerätegruppen der Segmentierung sind klein (U-Systems: 6 Bilder) und werden nicht interpretiert; keine Fairnessaussage.
- Nur Läsionsbilder, keine gesunde Screening-Population; keine Messung ärztlicher Entscheidungen; Scores sind keine kalibrierten Wahrscheinlichkeiten; keine klinische Eignung.

## Abweichungen, Prüfungen und offene Punkte

- **Produktionslauf:** Der Koordinator lief beim ersten Versuch durch (16 gültige Worker, kein Fehlschlag, kein wiederholter Lauf, kein Eingriff). Der zweite Aufruf verifizierte schreibgeschützt (Exit 0, 926 Dateien hashgleich, kein Bild, kein TensorFlow, Code gleich der Registrierung).
- **Plankonkretisierungen:** Die Auslegung „Vollbild = feinjustierte 4B-Pipeline“ und die Reihenfolge der Bootstrap-Ziehungen (erst maligne, dann benigne Cases) sind im Plan festgelegt; 5A legte Seed, Anzahl, Schichtung und Quantile fest.
- **Prüfungen vor der Registrierung (synthetisch):** 58 Funktionsprüfungen des Moduls (Schwellen-Roundtrip, Score-Orientierung, Konfusion, undefinierte Raten, Cluster-Ziehung mit mehrfach gezogenem Case und zwei Bildern, identische Bedingungen mit Differenz 0, gemeinsame Ziehungen, Segmentierungsmaße, Planhash über mehrere Seeds), 64 Prüfungen eines vollständigen Probelaufs (echte Checkpoints, erfundene Bilder; Maskenunabhängigkeit der ROI-Phase mit fehlenden, unbrauchbaren und anderen Masken, Kontrolle als Tor, Wiederanlauf- und Fehlerfälle) und 14 Schutzprüfungen in Kopien des echten Endzustands. Der Probelauf brach beim ersten Anlauf ab, weil im Arbeitsverzeichnis ein Hilfsskript fehlte; nach dessen Neuanlage lief er vollständig durch. Es gab keine Funktionstests mit echten Testbildern vor der Registrierung.
- **Abbildungen:** Die vier statistischen Abbildungen entstehen aus den Tabellen (`plots/plot_test_evaluation_results.py`) und haben einen eigenen Status; sie gehören nicht zum numerischen Code-Gate.
- **Offen:** Grad-CAM-Hauptinterpretation (5D), FGSM-Hauptauswertung mit Geräteaufschlüsselung (5E), Bericht und Vortrag.

## Dateien und Wiederaufruf

| Datei | Inhalt |
|---|---|
| `analysis/09_test_evaluation.py` | Interpreterprüfung, Vorbedingungen, Plan-Registrierung, Koordinator und 16 Worker, Aggregation, schreibgeschützte Prüfung |
| `src/busbra/test_evaluation.py` | CPU-Helfer: Metadaten, eingefrorene Schwellen, Kennzahlzeilen, Cluster-Bootstrap, Segmentierungsdiagnose |
| `plots/plot_test_evaluation_results.py` | vier Statistikabbildungen aus den Tabellen (CPU) |
| `reports/tables/test_evaluation_plan.json` | unveränderlicher technischer Plan |
| `reports/tables/test_evaluation_status.json` | Erfolgsstatus mit allen Ausgabe-Hashes (zuletzt geschrieben); `test_evaluation_figures_status.json` für die Abbildungen |
| `reports/tables/test_evaluation_scores.csv` | 10.152 Rohscores (zusätzlich je Kombination unter `reports/tables/test_evaluation/scores/`) |
| `reports/tables/test_evaluation_metrics*.csv`, `_operating_point*.csv`, `_case_view_image_thresholds.csv`, `_validation_vs_test.csv` | 54 Kennzahlzeilen, 162 Kernzeilen, Zusammenfassungen, historische Case-Sicht, Validation neben Test |
| `reports/tables/test_evaluation_comparisons_*.csv`, `_bootstrap_*.csv`, `reports/tables/test_evaluation/bootstrap_arrays.npz` | gepaarte Differenzen je Seed, Bootstrap-Intervalle, Replikatwerte, Ziehungen und Replikatwerte je Gruppe, Seed und Kennzahl |
| `reports/tables/test_evaluation_segmentation_*.csv`, `reports/tables/test_evaluation/auto_roi_*_test.*` | Segmentierungs- und Zuschnittsdiagnose, eingefrorene ROI-Koordinaten (`auto_roi_frozen_test.json`) |
| `reports/tables/test_evaluation_validation_control*.csv`, `reports/tables/test_evaluation/status/` | Vorkontrolle, Status und Zugriffsnachweise je Worker |
| `data/processed/test_unet_predicted_masks.npz` | vorhergesagte Pixel (nur lokal, nicht weitergeben) |
| `reports/figures/test_*.png` | Statistikabbildungen (nur Kennzahlen) |

Aufruf: `env -u LD_LIBRARY_PATH .venv/bin/python analysis/09_test_evaluation.py`. Ein erneuter Aufruf eines vollständigen Erfolgs prüft Plan, jede Ausgabe, jeden Workerstatus, alle 27 Modell-, Score- und Konfigurationsdateien, die eingefrorenen Schwellen und die Testmetadaten schreibgeschützt (Exit 0, kein Worker, kein Bild, keine GPU). Worker mit Artefakten ohne gültigen Erfolgsstatus werden als `recovery_required` gemeldet und nie neu berechnet, überschrieben oder bereinigt; nur vollständig fehlende Worker werden gestartet; die ROI-Einfrierung wird nie wiederholt. Unter einer bestehenden Registrierung rechnet nur unveränderter Code. Die Abbildungen: `.venv/bin/python plots/plot_test_evaluation_results.py`.

## Quellen

- Abschlussprotokoll und eingefrorene Betriebspunkte: [final_evaluation_protocol.md](final_evaluation_protocol.md) (Schritt 5A).
- Perzentilberechnung: `numpy.percentile`, Methode `linear`: https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
- ROC-AUC und Average Precision: scikit-learn, `roc_auc_score` und `average_precision_score`: https://scikit-learn.org/stable/modules/model_evaluation.html
