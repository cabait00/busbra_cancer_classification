# Schritt 5E: FGSM-Hauptauswertung und Geräteaufschlüsselung am eingefrorenen Test

> Hinweis: Die Export- und Nachrechnungsskripte der Geräteauswertung (`23_export_device_review_table.py`, `24_recompute_device_results.py`) gehören zum historischen Ablauf und liegen nicht mehr im Projektordner; ihre Ergebnisse stehen in `reports/tables/device_test_*.csv`; die frühere Zweitablage `reports/final_review/` ist nicht Teil des Repositories.

Stand: 21.10.2026. Schritt 5E ist umgesetzt, ausgeführt. Er untersucht, wie sich die Trennung gutartiger und bösartiger Befunde und die Entscheidungen an den **eingefrorenen** 5A-Schwellen unter einem kleinen, vorab festgelegten digitalen Einzelschritt-Angriff (FGSM, Stärken 0, 1/255, 2/255, 4/255 auf dem [0, 1]-Klassifikatoreingang) für zwei Bedingungen ändern: das feinjustierte Vollbild (4B) und die automatische ROI mit den ursprünglichen 4E-Klassifikatoren, je fünf Seeds (zehn Checkpoints). Ergänzend wird die Leistung nach Aufnahmegerät aufgeschlüsselt (ungestört für alle 27 Kombinationen aus den gespeicherten 5C-Scores, gestört für die zehn angegriffenen Kombinationen). Es wird nichts trainiert, ausgewählt oder neu eingestellt; die 5C-Ergebnisse bleiben unverändert. Die Analyse ist ein begrenzter digitaler Stresstest mit wahren Labels als Auswertungsannahme, kein umfassender Robustheitsnachweis, keine Simulation klinischer Aufnahmefehler und keine Aussage zu klinischer Eignung oder demografischer Fairness.

## Kurzfassung

- **Registrierung:** Studie `fgsm5e-0280d8ce`, Plan-SHA-256 `0280d8ceee7184c9408d5abcd8a5995dce5f19d93c7257244b8935ad1e800e4b`, registriert am 2026-10-21 18:07:40, nach der Hashprüfung aller Quellen und **vor der ersten Testbildverarbeitung dieses Schritts**. Die Testergebnisse von 5C und die Karten von 5D waren bekannt; es wird keine frühere Registrierung und keine Optimierung nach Ergebnissen behauptet.
- **Technische Prüfung:** Alle zehn Worker (frische Prozesse je Bedingung und Seed, ursprüngliche geordnete 16er-Batches einschließlich des letzten 8er-Batches) reproduzieren die gespeicherten 5C-Scores bei Epsilon 0 mit der größten Abweichung 0 (bitgleich) (Toleranz 1e-5); keine Entscheidung an den eingefrorenen Bildschwellen weicht ab, alle 60 Betriebspunktzeilen von Epsilon 0 stimmen mit 5C überein. Änderung höchstens Epsilon + 1e-7, Werte in [0, 1], Padding unverändert, endliche Scores, keine Nullgradienten; Gewichte unverändert.
- **Vollständigkeit:** 15.040 Bildscore-Zeilen (10 Kombinationen × 4 Stärken × 376 Bilder), 80 Bild-/Case-Kennzahlzeilen, 240 Betriebspunktzeilen, alle Prüfungen erfüllt.
- **Befund (beschreibend, Seed-Mittel ± Stichproben-SD):** Bild-ROC-AUC Vollbild 0,895 ± 0,009 → 0,027 ± 0,005 (1/255) → 0,006 ± 0,002 (4/255); automatische ROI 0,915 ± 0,008 → 0,164 ± 0,048 → 0,114 ± 0,061. Schon die kleinste Stärke senkt die AUC beider Bedingungen unter 0,5, weil der Angriff die wahren Labels kennt und den Loss erhöht; die Orientierung wurde nicht umgedreht. Die automatische ROI behält in diesem Test höhere AUC-Werte als das Vollbild, beide sind stark beeinträchtigt; gleiche Epsilon-Werte sind keine gleiche Störung in Originalbildkoordinaten.
- **Fehlerzahlen am eingefrorenen Ziel-0,95-Punkt (Bildebene):** Vollbild FP 133,8 → 253,8 von 254, FN 3,8 → 98,8 von 122; automatische ROI FP 131,8 → 251,6, FN 2,6 → 55,0. Entscheidungswechsel „richtig → falsch“ und „falsch → richtig“ sind getrennt ausgewiesen (Abschnitt „Entscheidungswechsel gegenüber Epsilon 0“).
- **Geräte:** vier Geräte mit 192, 145, 33 und 6 Bildern (3 Cases beim kleinsten); kein Case mit gemischten Geräten. Die Geräteergebnisse sind deskriptiv, die kleinen Gruppen tragen keine belastbare Aussage; es gibt keine geräteeigenen Schwellen und keine Fairnessaussage.

## Frage, Einordnung und Registrierung

Die Registrierung (`reports/tables/robustness_test_plan.json`) enthält die Quellen mit Hashes (5A-Modelle und Schwellen, 5C-Scores, ROI-Einfrierung, Geometrie, Split, 5D-Stand und der unveränderte 5B-Diagnosehelfer), die Codehashes, Seeds, Epsilons, die Reihenfolge der zehn Worker, die Ausgaben, die Gerätezuordnung und die Auswertungsregeln. Sie wurde vor der ersten Testbildverarbeitung dieses Schritts geschrieben; ein unveränderter Wiederaufruf prüft sie nur noch, anderer Code unter derselben Registrierung rechnet nicht. Die Anwendung der eingefrorenen Schwellen aus 5A und die Gruppen/27 Kombinationen bleiben maßgeblich; die 5B-R-Fortsetzungsmodelle (Validation-Zusatzexperiment) und der Experten-ROI-Angriff sind nicht Teil dieses Schritts.

## Verfahren

**FGSM.** Je Bedingung und Seed werden die 376 Testbilder in den ursprünglichen geordneten 16er-Batches (letzter Batch 8 Bilder) verarbeitet. Eingang: unveränderter 160 × 160 × 1-Tensor in float32 [0, 1]; die drei Kanäle und die Skalierung auf [-1, 1] entstehen im Modell. Am **sauberen** Eingang wird der Gradient der binären Kreuzentropie mit dem **wahren** Label nach der Eingabe berechnet; je Stärke unabhängig genau ein Schritt x_adv = clip(x + ε · sign(g), 0, 1) mit dem geprüften 5B-Helfer (unverändert). Keine Verkettung der Stärken, kein iterativer Angriff. Das künstliche Resize-Padding bleibt unverändert; die gültige Fläche stammt nur aus der Bild-/Crop-Geometrie, nie aus einer Läsionsmaske; schwarze Originalpixel im Bild bleiben angreifbar. Alle berichteten Scores und Losswerte stammen aus dem ursprünglichen Inferenzpfad; der Gradientengraph liefert nur die Richtung (sein Vorwärtspfad weicht unter TF32 geringfügig vom Inferenzpfad ab; die Abweichung wird je Lauf diagnostiziert, die numerische Stabilität der Richtung ist nicht belegt).

**Automatische ROI.** Ausschließlich die in 5C eingefrorenen automatischen Boxen; U-Net, Komponentenwahl und Zuschnitt werden weder angegriffen noch neu berechnet; bei leerer Segmentierung gilt das ganze Bild mit demselben 4E-Klassifikator. Der Angriff setzt am kontinuierlichen Klassifikatoreingang an.

**Kennzahlen.** Je Bedingung, Seed und Stärke: Bild- und Case-ROC-AUC und AP (Case-Score = Maximum der angegriffenen Bildscores), Konfusion, Sensitivität, Spezifität, Precision, NPV, F1, Accuracy und Anteil positiver Entscheidungen an den eigenen eingefrorenen 5A-Schwellen (0,5, Ziel 0,90, Ziel 0,95; Bild- und Case-Ebene). Änderungen gegenüber Epsilon 0 derselben Pipeline und desselben Seeds; Mittel und Stichproben-SD (ddof = 1) über die fünf Seeds, keine gepoolte AUC, kein Ensemble. Es wird keine Schwelle auf sauberen oder angegriffenen Testbildern gesucht.

**Geräte.** Die Gerätezuordnung stammt aus den auditierten Manifestzeilen (und stimmt mit den 5C-Diagnosen überein). Jedes Bild gehört zu seinem Gerät, jeder Case wird genau einmal zugeordnet (einem Gerät oder, bei mehreren Geräten, der getrennten Gruppe „mixed devices“; im Test gibt es keinen solchen Case). Case-Kennzahlen verwenden das Maximum aller Bilder des Cases und die globalen Case-Schwellen aus 5A. Nur global eingefrorene Schwellen; undefinierte Kennzahlen (AUC/AP bei nur einer Klasse, Nenner null) bleiben leer.

## Technische Prüfung (belegt)

- **Epsilon 0 gegen 5C:** größte Score-Abweichung aller zehn Kombinationen 0 (bitgleich); abweichende Entscheidungen an den eingefrorenen Bildschwellen: 0; 60 von 60 Betriebspunktzeilen gleich 5C.
- **Störungseigenschaften (alle Läufe, alle Stärken):** größte Änderung 0,015686 (4/255 = 0,015686); Änderung ≤ ε + 1e-7: ja; Werte in [0, 1]: ja; Padding unverändert: ja; endliche Scores: ja; Epsilon 0 lässt den Eingang unverändert: ja; Nullgradienten: 0 Bilder in allen Läufen.
- **Diagnose des Gradientenpfads:** größte absolute Differenz zwischen dem Logit des Vorwärtspfads im Eingabe-Gradientengraphen und dem Inferenz-Logit 0,038 (Vollbild 0,028, automatische ROI 0,038); die größte Abweichung zwischen Sigmoid(Logit) und Score im Inferenzpfad ist 8,7e-08. Die Richtung des Gradienten wird verwendet, ihre vollständige numerische Stabilität ist nicht belegt.
- **Zugriff und Modelle:** jeder Worker las genau die 376 Testbilder, 0 Masken, kein U-Net und genau einen Checkpoint; Checkpoint-Dateien und Variablenfingerabdrücke vor und nach der Berechnung identisch; Memory Growth an, weiche Platzierung aus, GPU belegt.
- **Laufzeit:** je Worker etwa 17,2 s, Gesamtlauf 184,6 s.

## Ergebnisse der FGSM-Auswertung (beschreibend)

### Trennschärfe (Seed-Mittel ± Stichproben-SD, n = 5 Seeds)

| Bedingung | Epsilon | Bild-ROC-AUC | Bild-AP | Case-ROC-AUC | Case-AP |
|---|---|---:|---:|---:|---:|
| Vollbild (4B feinjustiert) | 0 | 0,895 ± 0,009 | 0,814 ± 0,018 | 0,896 ± 0,012 | 0,823 ± 0,020 |
| Vollbild (4B feinjustiert) | 1/255 | 0,027 ± 0,005 | 0,186 ± 0,000 | 0,021 ± 0,009 | 0,184 ± 0,000 |
| Vollbild (4B feinjustiert) | 2/255 | 0,007 ± 0,002 | 0,185 ± 0,000 | 0,005 ± 0,004 | 0,183 ± 0,000 |
| Vollbild (4B feinjustiert) | 4/255 | 0,006 ± 0,002 | 0,185 ± 0,000 | 0,005 ± 0,003 | 0,183 ± 0,000 |
| Automatische ROI (4E-Klassifikator) | 0 | 0,915 ± 0,008 | 0,852 ± 0,015 | 0,919 ± 0,008 | 0,867 ± 0,016 |
| Automatische ROI (4E-Klassifikator) | 1/255 | 0,164 ± 0,048 | 0,202 ± 0,008 | 0,173 ± 0,048 | 0,203 ± 0,008 |
| Automatische ROI (4E-Klassifikator) | 2/255 | 0,112 ± 0,039 | 0,194 ± 0,005 | 0,118 ± 0,042 | 0,194 ± 0,006 |
| Automatische ROI (4E-Klassifikator) | 4/255 | 0,114 ± 0,061 | 0,196 ± 0,012 | 0,120 ± 0,069 | 0,196 ± 0,014 |

Der Anteil maligner Bilder ist 0,324; ein AP unter diesem Anteil entsteht durch die vom Angriff invertierte Rangfolge. Seed-Einzelwerte der Bild-ROC-AUC:

| Bedingung | Seed | ε = 0 | 1/255 | 2/255 | 4/255 |
|---|---:|---:|---:|---:|---:|
| Vollbild (4B feinjustiert) | 7 | 0,881 | 0,020 | 0,004 | 0,004 |
| Vollbild (4B feinjustiert) | 21 | 0,907 | 0,031 | 0,008 | 0,006 |
| Vollbild (4B feinjustiert) | 42 | 0,895 | 0,031 | 0,007 | 0,005 |
| Vollbild (4B feinjustiert) | 123 | 0,896 | 0,030 | 0,010 | 0,009 |
| Vollbild (4B feinjustiert) | 2026 | 0,895 | 0,025 | 0,005 | 0,004 |
| Automatische ROI (4E-Klassifikator) | 7 | 0,929 | 0,207 | 0,122 | 0,112 |
| Automatische ROI (4E-Klassifikator) | 21 | 0,911 | 0,090 | 0,055 | 0,047 |
| Automatische ROI (4E-Klassifikator) | 42 | 0,914 | 0,205 | 0,131 | 0,108 |
| Automatische ROI (4E-Klassifikator) | 123 | 0,910 | 0,172 | 0,157 | 0,213 |
| Automatische ROI (4E-Klassifikator) | 2026 | 0,911 | 0,145 | 0,095 | 0,089 |

### Fehlerzahlen und Raten an den eingefrorenen Bildschwellen (Seed-Mittel ± SD)

**Bildebene, Ziel 0,90** (254 benigne, 122 maligne Bilder; das Ziel nennt die Validation-Auswahl, die erreichte Test-Sensitivität ist ein Ergebnis):

| Bedingung | Epsilon | FP | FN | Sensitivität | Spezifität |
|---|---|---:|---:|---:|---:|
| Vollbild (4B feinjustiert) | 0 | 85,8 ± 18,5 | 11,6 ± 5,3 | 0,905 ± 0,043 | 0,662 ± 0,073 |
| Vollbild (4B feinjustiert) | 1/255 | 247,8 ± 3,3 | 93,0 ± 8,7 | 0,238 ± 0,071 | 0,024 ± 0,013 |
| Vollbild (4B feinjustiert) | 2/255 | 253,2 ± 1,1 | 103,0 ± 6,0 | 0,156 ± 0,049 | 0,003 ± 0,004 |
| Vollbild (4B feinjustiert) | 4/255 | 252,0 ± 2,0 | 109,0 ± 6,2 | 0,107 ± 0,051 | 0,008 ± 0,008 |
| Automatische ROI (4E-Klassifikator) | 0 | 97,4 ± 7,7 | 7,2 ± 3,3 | 0,941 ± 0,027 | 0,617 ± 0,030 |
| Automatische ROI (4E-Klassifikator) | 1/255 | 243,2 ± 7,5 | 62,8 ± 6,1 | 0,485 ± 0,050 | 0,043 ± 0,029 |
| Automatische ROI (4E-Klassifikator) | 2/255 | 247,2 ± 3,7 | 69,4 ± 4,8 | 0,431 ± 0,040 | 0,027 ± 0,015 |
| Automatische ROI (4E-Klassifikator) | 4/255 | 246,8 ± 6,8 | 68,4 ± 8,3 | 0,439 ± 0,068 | 0,028 ± 0,027 |

**Bildebene, Ziel 0,95** (254 benigne, 122 maligne Bilder; das Ziel nennt die Validation-Auswahl, die erreichte Test-Sensitivität ist ein Ergebnis):

| Bedingung | Epsilon | FP | FN | Sensitivität | Spezifität |
|---|---|---:|---:|---:|---:|
| Vollbild (4B feinjustiert) | 0 | 133,8 ± 23,6 | 3,8 ± 1,8 | 0,969 ± 0,015 | 0,473 ± 0,093 |
| Vollbild (4B feinjustiert) | 1/255 | 253,4 ± 1,3 | 79,2 ± 7,8 | 0,351 ± 0,064 | 0,002 ± 0,005 |
| Vollbild (4B feinjustiert) | 2/255 | 254,0 ± 0,0 | 93,4 ± 8,0 | 0,234 ± 0,066 | 0,000 ± 0,000 |
| Vollbild (4B feinjustiert) | 4/255 | 253,8 ± 0,4 | 98,8 ± 9,0 | 0,190 ± 0,074 | 0,001 ± 0,002 |
| Automatische ROI (4E-Klassifikator) | 0 | 131,8 ± 13,5 | 2,6 ± 1,9 | 0,979 ± 0,016 | 0,481 ± 0,053 |
| Automatische ROI (4E-Klassifikator) | 1/255 | 250,8 ± 1,9 | 50,2 ± 11,0 | 0,589 ± 0,090 | 0,013 ± 0,008 |
| Automatische ROI (4E-Klassifikator) | 2/255 | 251,8 ± 2,2 | 57,4 ± 8,4 | 0,530 ± 0,069 | 0,009 ± 0,009 |
| Automatische ROI (4E-Klassifikator) | 4/255 | 251,6 ± 3,0 | 55,0 ± 14,5 | 0,549 ± 0,118 | 0,009 ± 0,012 |

**Case-Ebene, Ziel 0,95** (144 benigne, 68 maligne Cases; Case-Schwelle aus 5A, Case-Score = Maximum der Bildscores):

| Bedingung | Epsilon | FP | FN | Sensitivität | Spezifität |
|---|---|---:|---:|---:|---:|
| Vollbild (4B feinjustiert) | 0 | 59,0 ± 20,5 | 4,6 ± 3,6 | 0,932 ± 0,054 | 0,590 ± 0,142 |
| Vollbild (4B feinjustiert) | 1/255 | 142,0 ± 2,8 | 48,6 ± 6,1 | 0,285 ± 0,090 | 0,014 ± 0,020 |
| Vollbild (4B feinjustiert) | 2/255 | 143,4 ± 1,3 | 53,8 ± 6,1 | 0,209 ± 0,089 | 0,004 ± 0,009 |
| Vollbild (4B feinjustiert) | 4/255 | 142,8 ± 2,7 | 57,4 ± 7,7 | 0,156 ± 0,114 | 0,008 ± 0,019 |
| Automatische ROI (4E-Klassifikator) | 0 | 64,6 ± 5,5 | 3,4 ± 2,1 | 0,950 ± 0,030 | 0,551 ± 0,039 |
| Automatische ROI (4E-Klassifikator) | 1/255 | 140,6 ± 3,2 | 28,8 ± 5,8 | 0,576 ± 0,085 | 0,024 ± 0,022 |
| Automatische ROI (4E-Klassifikator) | 2/255 | 142,8 ± 1,3 | 31,6 ± 4,0 | 0,535 ± 0,059 | 0,008 ± 0,009 |
| Automatische ROI (4E-Klassifikator) | 4/255 | 143,0 ± 1,0 | 30,2 ± 5,9 | 0,556 ± 0,087 | 0,007 ± 0,007 |

### Entscheidungswechsel gegenüber Epsilon 0 (Bildebene, Seed-Mittel ± SD)

| Ziel | Bedingung | Epsilon | richtig → falsch | falsch → richtig | bleibt richtig | bleibt falsch | rohe Wechsel |
|---|---|---|---:|---:|---:|---:|---:|
| 0,90 | Vollbild (4B feinjustiert) | 1/255 | 243,4 ± 19,4 | 0,0 ± 0,0 | 35,2 ± 6,9 | 97,4 ± 13,5 | 243,4 ± 19,4 |
| 0,90 | Vollbild (4B feinjustiert) | 2/255 | 258,8 ± 17,5 | 0,0 ± 0,0 | 19,8 ± 5,4 | 97,4 ± 13,5 | 258,8 ± 17,5 |
| 0,90 | Vollbild (4B feinjustiert) | 4/255 | 263,6 ± 16,3 | 0,0 ± 0,0 | 15,0 ± 4,7 | 97,4 ± 13,5 | 263,6 ± 16,3 |
| 0,90 | Automatische ROI (4E-Klassifikator) | 1/255 | 201,4 ± 14,6 | 0,0 ± 0,0 | 70,0 ± 12,2 | 104,6 ± 5,9 | 201,4 ± 14,6 |
| 0,90 | Automatische ROI (4E-Klassifikator) | 2/255 | 212,4 ± 11,4 | 0,4 ± 0,5 | 59,0 ± 7,6 | 104,2 ± 5,6 | 212,8 ± 11,3 |
| 0,90 | Automatische ROI (4E-Klassifikator) | 4/255 | 211,4 ± 17,1 | 0,8 ± 1,1 | 60,0 ± 13,2 | 103,8 ± 5,3 | 212,2 ± 17,0 |
| 0,95 | Vollbild (4B feinjustiert) | 1/255 | 195,0 ± 29,0 | 0,0 ± 0,0 | 43,4 ± 7,6 | 137,6 ± 21,9 | 195,0 ± 29,0 |
| 0,95 | Vollbild (4B feinjustiert) | 2/255 | 209,8 ± 29,0 | 0,0 ± 0,0 | 28,6 ± 8,0 | 137,6 ± 21,9 | 209,8 ± 29,0 |
| 0,95 | Vollbild (4B feinjustiert) | 4/255 | 215,0 ± 30,0 | 0,0 ± 0,0 | 23,4 ± 8,9 | 137,6 ± 21,9 | 215,0 ± 30,0 |
| 0,95 | Automatische ROI (4E-Klassifikator) | 1/255 | 166,6 ± 20,7 | 0,0 ± 0,0 | 75,0 ± 11,3 | 134,4 ± 12,7 | 166,6 ± 20,7 |
| 0,95 | Automatische ROI (4E-Klassifikator) | 2/255 | 174,8 ± 18,0 | 0,0 ± 0,0 | 66,8 ± 8,1 | 134,4 ± 12,7 | 174,8 ± 18,0 |
| 0,95 | Automatische ROI (4E-Klassifikator) | 4/255 | 172,4 ± 25,1 | 0,2 ± 0,4 | 69,2 ± 14,1 | 134,2 ± 12,8 | 172,6 ± 25,2 |

Die vier Kategorien zerlegen die 376 Bilder; eine rohe Wechselzahl ist keine alleinige Angriffs-Erfolgsquote (Richtung und Klasse stehen in `robustness_test_decision_changes*.csv`). Änderungen der Kennzahlen je Seed gegenüber Epsilon 0 stehen in `robustness_test_changes.csv` und `robustness_test_changes_summary.csv` (mit Vorzeichenzählung).

## Geräteaufschlüsselung (deskriptiv)

### Belegung

| Gerät / Gruppe | Bilder | maligne / benigne Bilder | Cases | maligne / benigne Cases |
|---|---:|---:|---:|---:|
| GE Logiq 5 @10-12MHz | 145 | 53 / 92 | 90 | 31 / 59 |
| GE Logiq 7 @10-14MHz | 192 | 55 / 137 | 102 | 30 / 72 |
| Toshiba Aplio 300 @12-14 MHz | 33 | 12 / 21 | 17 | 6 / 11 |
| U-Systems @10-14MHz | 6 | 2 / 4 | 3 | 1 / 2 |
| mixed devices | 0 | 0 / 0 | 0 | 0 / 0 |
| all | 376 | 122 / 254 | 212 | 68 / 144 |

Jeder Case ist genau einmal zugeordnet (die Case-Zahlen der Geräte summieren sich zu 212); die Gruppe „mixed devices“ ist leer. Das kleinste Gerät (U-Systems) hat 6 Bilder aus 3 Cases (2 maligne Bilder): Alle Kennzahlen dieser Gruppe sind nicht belastbar (die Unsicherheit ist groß, ein einziges Bild verändert Raten um mehr als 15 Prozentpunkte). Die Geräte unterscheiden sich in Klassenanteilen und Fallzahl.

### Ungestörte Bild-ROC-AUC je Gerät (alle sieben Gruppen aus den gespeicherten 5C-Scores; Seed-Mittel ± SD)

| Gruppe | GE Logiq 5 | GE Logiq 7 | Toshiba Aplio 300 | U-Systems | alle Bilder |
|---|---:|---:|---:|---:|---:|
| Dummy | 0,500 | 0,500 | 0,500 | 0,500 | 0,500 |
| LogReg L2 | 0,704 | 0,685 | 0,544 | 1,000 | 0,688 |
| Kleines CNN | 0,632 ± 0,115 | 0,686 ± 0,100 | 0,687 ± 0,052 | 0,500 ± 0,000 | 0,613 ± 0,107 |
| Vollbild frozen | 0,798 ± 0,014 | 0,833 ± 0,015 | 0,788 ± 0,033 | 1,000 ± 0,000 | 0,817 ± 0,010 |
| Vollbild feinjustiert | 0,896 ± 0,016 | 0,891 ± 0,015 | 0,886 ± 0,036 | 1,000 ± 0,000 | 0,895 ± 0,009 |
| Experten-ROI | 0,962 ± 0,006 | 0,918 ± 0,012 | 0,943 ± 0,017 | 1,000 ± 0,000 | 0,937 ± 0,008 |
| Automatische ROI | 0,939 ± 0,006 | 0,912 ± 0,011 | 0,806 ± 0,031 | 1,000 ± 0,000 | 0,915 ± 0,008 |

Die Spalte „alle Bilder“ stimmt mit den 5C-Werten überein. Bei U-Systems (2 maligne, 4 benigne Bilder) sind die AUC-Werte durch die Fallzahl bestimmt (die Werte 1,000 und 0,500 sind keine Leistungsunterschiede der Gruppen).

### Ungestörte Sensitivität und Spezifität am eingefrorenen Ziel-0,95-Punkt (Bildebene, Seed-Mittel ± SD)

| Gruppe | Gerät | Sensitivität | Spezifität |
|---|---|---:|---:|
| Vollbild feinjustiert | GE Logiq 5 | 0,989 ± 0,017 | 0,393 ± 0,100 |
| Vollbild feinjustiert | GE Logiq 7 | 0,945 ± 0,022 | 0,508 ± 0,100 |
| Vollbild feinjustiert | Toshiba Aplio 300 | 0,983 ± 0,037 | 0,533 ± 0,103 |
| Vollbild feinjustiert | U-Systems | 1,000 ± 0,000 | 0,800 ± 0,112 |
| Vollbild feinjustiert | alle | 0,969 ± 0,015 | 0,473 ± 0,093 |
| Experten-ROI | GE Logiq 5 | 0,992 ± 0,010 | 0,546 ± 0,102 |
| Experten-ROI | GE Logiq 7 | 0,967 ± 0,024 | 0,575 ± 0,051 |
| Experten-ROI | Toshiba Aplio 300 | 0,983 ± 0,037 | 0,448 ± 0,156 |
| Experten-ROI | U-Systems | 1,000 ± 0,000 | 0,050 ± 0,112 |
| Experten-ROI | alle | 0,980 ± 0,017 | 0,546 ± 0,063 |
| Automatische ROI | GE Logiq 5 | 0,974 ± 0,029 | 0,528 ± 0,067 |
| Automatische ROI | GE Logiq 7 | 0,989 ± 0,010 | 0,486 ± 0,070 |
| Automatische ROI | Toshiba Aplio 300 | 0,950 ± 0,075 | 0,314 ± 0,072 |
| Automatische ROI | U-Systems | 1,000 ± 0,000 | 0,100 ± 0,224 |
| Automatische ROI | alle | 0,979 ± 0,016 | 0,481 ± 0,053 |

### Unter Angriff: Bild-ROC-AUC und Fehlerzahlen je Gerät (Seed-Mittel)

Format je Zelle: Bild-ROC-AUC / FN / FP (Ziel-0,95-Punkt, Seed-Mittel; FN und FP sind Anzahlen im Gerät, keine Raten).

| Bedingung | Epsilon | GE Logiq 5 (145 Bilder) | GE Logiq 7 (192 Bilder) | Toshiba Aplio 300 (33 Bilder) | U-Systems (6 Bilder) |
|---|---|---|---|---|---|
| Vollbild (4B feinjustiert) | 0 | 0,896 / 0,6 / 55,8 | 0,891 / 3,0 / 67,4 | 0,886 / 0,2 / 9,8 | 1,000 / 0,0 / 0,8 |
| Vollbild (4B feinjustiert) | 1/255 | 0,050 / 29,4 / 91,4 | 0,015 / 38,4 / 137,0 | 0,007 / 9,6 / 21,0 | 0,000 / 1,8 / 4,0 |
| Vollbild (4B feinjustiert) | 2/255 | 0,010 / 37,2 / 92,0 | 0,004 / 44,2 / 137,0 | 0,004 / 10,0 / 21,0 | 0,000 / 2,0 / 4,0 |
| Vollbild (4B feinjustiert) | 4/255 | 0,004 / 41,2 / 92,0 | 0,005 / 45,6 / 136,8 | 0,005 / 10,0 / 21,0 | 0,000 / 2,0 / 4,0 |
| Automatische ROI (4E-Klassifikator) | 0 | 0,939 / 1,4 / 43,4 | 0,912 / 0,6 / 70,4 | 0,806 / 0,6 / 14,4 | 1,000 / 0,0 / 3,6 |
| Automatische ROI (4E-Klassifikator) | 1/255 | 0,187 / 21,2 / 89,6 | 0,150 / 23,4 / 136,2 | 0,065 / 5,6 / 21,0 | 0,225 / 0,0 / 4,0 |
| Automatische ROI (4E-Klassifikator) | 2/255 | 0,108 / 25,4 / 90,4 | 0,117 / 26,6 / 136,4 | 0,049 / 5,4 / 21,0 | 0,175 / 0,0 / 4,0 |
| Automatische ROI (4E-Klassifikator) | 4/255 | 0,088 / 26,2 / 91,0 | 0,130 / 24,0 / 135,6 | 0,056 / 4,8 / 21,0 | 0,525 / 0,0 / 4,0 |

Streuungen (SD), Case-Ebene, die übrigen Schwellen und Änderungen je Gerät stehen in `device_test_attacked_summary.csv` und `device_test_attacked_changes_summary.csv`.

### Automatische ROI nach Gerät und Zuschnittskategorie (5C-Diagnose; Seed-Mittel der Fehlerzahlen am Ziel-0,95-Punkt)

| Gerät | Zuschnittskategorie (5C) | Bilder | maligne / benigne | Fallbacks | mittl. Dice der rohen Vorhersage | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| GE Logiq 5 | covers_90_to_below_100_percent | 14 | 12 / 2 | 0 | 0,79 | 2,0 | 0,0 |
| GE Logiq 5 | covers_all_expert_pixels | 127 | 37 / 90 | 0 | 0,92 | 41,4 | 1,4 |
| GE Logiq 5 | covers_more_than_0_to_below_90_percent | 4 | 4 / 0 | 0 | 0,46 | 0,0 | 0,0 |
| GE Logiq 7 | covers_90_to_below_100_percent | 9 | 3 / 6 | 0 | 0,75 | 1,6 | 0,0 |
| GE Logiq 7 | covers_all_expert_pixels | 173 | 50 / 123 | 0 | 0,90 | 63,6 | 0,6 |
| GE Logiq 7 | covers_more_than_0_to_below_90_percent | 8 | 2 / 6 | 0 | 0,54 | 3,2 | 0,0 |
| GE Logiq 7 | covers_no_expert_pixel | 1 | 0 / 1 | 0 | 0,42 | 1,0 | 0,0 |
| GE Logiq 7 | fallback_full_image | 1 | 0 / 1 | 1 | 0,00 | 1,0 | 0,0 |
| Toshiba Aplio 300 | covers_90_to_below_100_percent | 7 | 4 / 3 | 0 | 0,78 | 1,4 | 0,0 |
| Toshiba Aplio 300 | covers_all_expert_pixels | 21 | 6 / 15 | 0 | 0,86 | 10,2 | 0,2 |
| Toshiba Aplio 300 | covers_more_than_0_to_below_90_percent | 3 | 1 / 2 | 0 | 0,25 | 1,8 | 0,4 |
| Toshiba Aplio 300 | fallback_full_image | 2 | 1 / 1 | 2 | 0,00 | 1,0 | 0,0 |
| U-Systems | covers_all_expert_pixels | 6 | 2 / 4 | 0 | 0,89 | 3,6 | 0,0 |

Die Tabelle fasst vorhandene 5C-Diagnosen mit den Entscheidungen der fünf automatischen Pipelines zusammen; es gibt keine neue Fallback- oder Zuschnittsregel. Kleine Zellen (wenige Bilder) sind nicht interpretierbar; Fallbacks und Abdeckungsfehler sind selten (siehe `docs/test_evaluation.md`).

## Einordnung (beschreibend; keine Kausal- und keine klinische Aussage)

- Unter dem Einzelschritt-Angriff mit wahren Labels verlieren beide Bedingungen schon bei ε = 1/255 fast die gesamte Trennschärfe; die automatische ROI behält in diesem Test mehr als das Vollbild (Bild-AUC 0,164 gegenüber 0,027). Das ist eine Beschreibung dieses Stresstests, kein Beleg, dass Zuschnitte robuster sind: ε entspricht in Vollbild und Zuschnitt nicht derselben Störung in Originalbildkoordinaten (ein Zuschnitt wird auf 160 × 160 vergrößert), die Angriffe nutzen den Gradienten des jeweiligen Modells, und fünf Seeds derselben Architektur sind keine unabhängigen Kohorten.
- Die AUC unter 0,5 und die fast vollständige Fehlerzahl am niedrigen Ziel-0,95-Schwellenpunkt (fast alle benignen Bilder werden positiv) sind Folge der Angriffsrichtung (Loss-Erhöhung mit wahren Labels); die Orientierung der Scores wurde nicht umgedreht. Eine Schwelle wurde nicht nachgezogen; ein Training gegen die Störung ist nicht Teil dieses Schritts.
- Die Geräteaufschlüsselung zeigt, dass die ungestörten Ergebnisse je Gerät streuen und die Gruppen sehr unterschiedlich groß sind; ein Geräteeffekt oder demografische Fairness lässt sich daraus nicht ableiten (kleine Fallzahlen, andere Klassenanteile, Einrichtungs- und Probandenunterschiede unbekannt).

## Grenzen

- Begrenzter digitaler Einzelschritt-Stresstest mit wahren Labels auf einem festen Holdout einer Einrichtung und vier Geräten; kein Robustheitsnachweis, keine Simulation klinischer Aufnahmefehler, keine Aussage zu anderen Angriffen oder zu einer Übertragung auf andere Daten.
- Gleiche Epsilon-Werte in Vollbild und ROI sind keine gleiche Störung in Originalbildkoordinaten; bei der automatischen ROI wird nur der Klassifikatoreingang angegriffen, nicht U-Net oder Zuschnitt.
- Der Gradient stammt aus einem Diagnosegraphen, dessen Vorwärtspfad unter TF32 geringfügig vom Inferenzpfad abweicht; die Richtung ist genutzt, ihre numerische Stabilität nicht belegt (ein zweiter Gradientenpfad im Probelauf lieferte für einzelne Bilder deutlich andere Angriffsscores bei gleicher mittlerer Wirkung).
- Die Seeds messen Trainingsstreuung auf demselben Holdout; mehrere Bilder eines Cases sind abhängig; die Mittel und Stichproben-SD sind deskriptiv, keine Signifikanzaussage.
- Gerätegruppen sind klein und unbalanciert; keine gerätespezifischen Schwellen, keine Fairness- oder Kausalaussage.
- Schlechte Werte lösen kein neues Training und keine Schwellenanpassung aus.

## Abweichungen, Prüfungen und offene Punkte

- **Produktionslauf** (`env -u LD_LIBRARY_PATH .venv/bin/python analysis/11_robustness_devices.py`): **erster Versuch (18:05:11) Exit 1 vor der Registrierung**: Die Hashprüfung des unveränderten 5B-Helfers suchte in der falschen Planstelle der 5D-Registrierung; es wurden nur ein Fehlerprotokoll (`reports/tables/robustness_test_status_failed_attempt_fgsm5e-20261021T180511-1f3165c4.json`) geschrieben, kein Plan, kein Bild, keine GPU. Das Skript wurde korrigiert (Suche in beiden Planfeldern), die Funktions- und Probeläufe wurden auf dem endgültigen Code wiederholt. **Zweiter Versuch** Exit 0, 18:07:39 bis 18:10:45 Uhr (Gesamtlauf 184.6 s), zehn Worker je etwa 18 s, kein Worker-Fehlschlag, kein Wiederanlauf. **Dritter Aufruf** Exit 0: schreibgeschützte Verifikation (kein Worker, kein Bild, keine GPU, nichts geschrieben, Code gleich der Registrierung). Abbildungen (`.venv/bin/python plots/plot_robustness_devices_results.py`): Exit 0, vier Abbildungen.
- **Funktions- und Probeläufe vor der Registrierung:** 43 Funktionsprüfungen (FGSM-Schritt an Arrays: Vorzeichen, Clipping an den Wertebereichsgrenzen, Padding, ε = 0, unabhängige Einzelschritte je ε, nicht endliche Gradienten; Entscheidungswechsel inklusive der Trennung richtig → falsch / falsch → richtig; Gerätezuordnung mit gemischtem Case und Ein-Klassen-Gruppe; Teilmengenkennzahlen gegen sklearn; undefinierte Raten; Seed-Zusammenfassung mit ddof = 1; gepaarte Änderungen) und 52 Prüfungen eines vollständigen Probelaufs mit echten Checkpoints und erfundenen Bildern in isolierten Projektkopien (neun Abbrüche vor der Registrierung: Interpreter, geänderte 5C-Scores, 5A-Schwellen, 5D-Pin, 5B-Diagnosehelfer, Manifest, 5C-Status, Checkpoint, fehlende Geometrie, abweichendes Gerät; Hauptlauf mit unabhängiger Neuberechnung aller Kennzahlen, Betriebspunkte, Entscheidungswechsel, Gerätekennzahlen und eines zweiten FGSM-Gradientenpfads; Wiederaufruf; elf Manipulationen am fertigen Zustand; Wiederanlauf unter derselben Registrierung; Code-Gate; Teilfortsetzung nur des fehlenden Workers; verwaistes Artefakt als recovery_required; Abbruch bei nicht reproduzierten Scores; Abbildungsskript). Es gab keine Tests auf echten Testbildern vor der Registrierung.
- **Probelauf, Toleranz und zweiter Gradientenpfad:** Der Probelauf nutzte für die Treue bei Epsilon 0 eine Toleranz von 1e-3 (Produktion 1e-5), weil die synthetisch gespeicherten Scores zwischen Prozessen um bis zu etwa 4e-4 variierten; im Produktionslauf war die Abweichung 0.0 (bitgleich). Ein unabhängiger zweiter Gradientenpfad (Keras-Loss auf der Modellausgabe) lieferte im Probelauf angegriffene Scores, die im Mittel um 0.014 abwichen (größter Einzelwert 0.097, Korrelation mindestens 0.996) bei vergleichbarer mittlerer Lossänderung; die Richtung eines TF32-Gradienten ist für einzelne Bilder nicht stabil, wie im Plan festgehalten.
- **5D-Nacharbeiten (getrennt protokolliert):** datierter Sichtungsnachtrag in `docs/gradcam_test_interpretation.md` (Beobachtungen und Interpretationen getrennt, präzisierte Registrierungsaussage, Provenienz der FP-Vorschau mit dem Upload-Hash als Übertragungsnachweis); lesbare Fassung der FN-Vorschau `data/processed/gradcam_test_preview_core_FN_readable.png` mit eigenem Darstellungsnachweis `reports/tables/gradcam_test_fn_readable_status.json` (Quellhashes, Codestand, Ausgabehash); die Layoutänderung wurde an synthetischen Daten geprüft, es wurde keine Karte neu berechnet, die sechs ursprünglichen Vorschauen, ihr Figurenstatus und alle 5D-Rechenartefakte sind byte-identisch. Daraus folgt keine globale Aussage „vor jedem bisherigen Bildzugriff“.
- **Unveränderter Bestand:** 1.751 Dateien des Ausgangsstandes (ohne README. Projektdefinition. privaten Status und die 5D-Dokumentation) sind nach dem Lauf byte-identisch. darunter alle 5A-. 5C- und 5D-Tabellen und -Registrierungen. Scores. Checkpoints. Split. Schwellen und die ursprünglichen Vorschauen; Rohdatenbaum und `pip freeze --all` unverändert; keine Datei entfernt.
- **Offen:** Abschlussbericht und Vortrag.

## Dateien und Wiederaufruf

- Skript: `analysis/11_robustness_devices.py` (Koordinator und zehn Worker, GPU), Modul `src/busbra/robustness_test.py` (Entscheidungswechsel, Gerätegruppen, Teilmengenkennzahlen, Seed-Zusammenfassungen), Abbildungen `plots/plot_robustness_devices_results.py` (CPU).
- Registrierung und Status: `reports/tables/robustness_test_plan.json`, `reports/tables/robustness_test_status.json`; Worker-Rohscores und -Status in `reports/tables/robustness_test/`; Tabellen `reports/tables/robustness_test_*.csv` (Quellen, vollständige Einzelbildscores, Kennzahlen, Betriebspunkte, Seed-Zusammenfassungen, Änderungen, Entscheidungswechsel, Epsilon-0-Vergleich, Störungsprüfungen, Vollständigkeit) und `reports/tables/device_test_*.csv` (Zuordnung, ungestört, gestört, Änderungen, automatische ROI nach Gerät).
- Abbildungen: `reports/figures/robustness_test_auc_ap.png`, `robustness_test_errors.png`, `device_test_clean.png`, `device_test_attacked.png` (aus den Tabellen; eigener Status `reports/tables/robustness_devices_figures_status.json`). Angriffs-Arrays werden nicht gespeichert.
- Aufruf: `env -u LD_LIBRARY_PATH .venv/bin/python analysis/11_robustness_devices.py`; ein fertiges Ergebnis wird nur schreibgeschützt geprüft (kein Worker, kein Bild, keine GPU, nichts geschrieben). Ergebnisse ohne gültigen Status werden nicht überschrieben (`recovery_required`); nur ein vollständig fehlender Worker wird unter dem unveränderten registrierten Code nachgeholt. Die Abbildungen entstehen mit `.venv/bin/python plots/plot_robustness_devices_results.py` ohne GPU.

## Nachtrag vom 21.10.2026 (Schritt 6): unabhängige Nachrechnung der Geräteauswertung

Aus dem unveränderten Manifest und der unveränderten Diagnosetabelle von 5C wurde eine reduzierte Zuordnung exportiert: ein Eintrag je Testbild mit `ID`, `Case`, `label`, `Device`, `category`, `fallback_full_image` und `dice_original` (keine Pfade, keine Pixel, keine Koordinaten; `analysis/23_export_device_review_table.py`, Ausgabe und Notiz mit Quellen und Hashes in der früheren Ablage `reports/final_review/`, nicht Teil des Repositories; die Zuordnung steht in `reports/tables/device_test_assignment.csv`). Aus dieser Zuordnung, den gespeicherten Scores (27 Kombinationen aus 5C, zehn Kombinationen × vier Stärken aus 5E) und den eingefrorenen 5A-Schwellen (aus dem exakten Hex-Text) wurde die Geräteauswertung mit eigenem Code ohne Module aus `src/busbra` neu berechnet (`analysis/24_recompute_device_results.py`): Gerätebelegung und Fallzuordnung (jeder der 212 Testfälle hat genau ein Gerät, die Gruppe „mixed devices“ ist leer), AUC, AP und Konfusionen ungestört und gestört, Seed-Zusammenfassungen, paarweise Änderungen gegenüber ε = 0, die Auto-ROI-Diagnostik je Gerät und Zuschnittskategorie mit Fehlerzahlen und gewichtetem Dice-Wert sowie Querprüfungen der Gruppe „all“ und der Dice-Werte gegen 5C. Alle 13 Vergleiche stimmen überein (größte Abweichung 7,1e-15; Zählwerte exakt gleich, undefinierte Werte auf beiden Seiten undefiniert); ein Selbsttest zeigt, dass absichtlich verfälschte Kopien als Widerspruch erkannt würden. Das Protokoll der Nachrechnung (`reports/final_review/device_recomputation_protocol.md`) ist nicht Teil des Repositories. Keine Ausgabe von 5E wurde verändert; die Studienwerte dieses Dokuments bleiben maßgeblich.

## Quellen

- Goodfellow, Shlens, Szegedy (2015), Explaining and Harnessing Adversarial Examples (FGSM); Kursunterlagen Tag 19/20 (Saliency und FGSM); Abschlussprotokoll `docs/final_evaluation_protocol.md`, Testauswertung `docs/test_evaluation.md`, Grad-CAM am Test `docs/gradcam_test_interpretation.md`.
