# Abschlussprotokoll (Schritt 5A): Forschungsfrage, Vergleichsgruppen und eingefrorene Validation-Betriebspunkte

**KI-gestützte Beurteilung von Brustultraschall: Vollbild oder Läsionsausschnitt?**

Stand: 21.10.2026. Schritt 5A ist fachlich und technisch geprüft; der Plan und die eingefrorenen Betriebspunkte sind unverändert. Dieses Dokument legt die verbindliche Forschungsfrage, die Vergleichsgruppen, die Auswertungsregeln und die auf Validation eingefrorenen Schwellen fest und beschreibt das Protokoll der späteren Testphase. **Testergebnisse liegen nicht vor; die Testdaten sind gesperrt.**

## Forschungsfrage

> Verbessert die Eingrenzung des Läsionsbereichs die Unterscheidung gutartiger und bösartiger Befunde im Brustultraschall?

Diese Arbeit untersucht, ob die Eingrenzung des Läsionsbereichs die KI-gestützte Unterscheidung gutartiger und bösartiger Brustultraschallbefunde verbessert. Auf dem BUS-BRA-Datensatz werden vollständige Bilder mit Läsionsausschnitten verglichen, deren Position entweder aus Expertenmasken oder aus einer automatischen U-Net-Segmentierung stammt. Bewertet werden die Klassifikationsleistung, die Streuung über Trainings-Seeds und die Fehlalarme bei festgelegten Sensitivitätszielen.

**Anwendungsbezug:** methodische Grundlage für computergestützte Befundunterstützung bei bereits sichtbaren Brustläsionen. Der Datensatz untersucht vorhandene Läsionen, keine gesunde Screening-Population. Die Arbeit misst keine tatsächliche Verbesserung ärztlicher Entscheidungen und begründet keine klinische Einsatzfähigkeit.

**Drei gleichwertig zu berichtende Vergleiche:**

1. Vollbild (4B) gegen Experten-ROI (4E): Nutzen des fachlich lokalisierten Bildausschnitts.
2. Experten-ROI (4E) gegen automatische ROI (4F-B): Unterschied durch die automatische Lokalisation.
3. Vollbild (4B) gegen automatische ROI (4F-B): Nutzen der automatisch anwendbaren Gesamtpipeline.

Baselines, CNN, Transfer Learning und Augmentation bilden die Modellierungsgrundlage. Die anfängliche Frage, ob ein CNN gegenüber einer einfachen lernenden Baseline einen Mehrwert bringt, bleibt als Unterfrage erhalten.

## Methodischer Status

Die Modellentwicklung erfolgte explorativ auf Train und Validation; die Validation wurde dabei wiederholt für Entwicklung und Checkpoint-Auswahl genutzt, ihre Ergebnisse sind daher bekannt und können optimistisch sein. Die in diesem Dokument festgelegten Abschlussregeln wurden **vor der neuen Schwellenberechnung dieses Schritts und vor jeder Auswertung der Testdaten** registriert (`reports/tables/final_evaluation_plan.json`). Es wird keine prospektive Registrierung vor der ersten Kenntnis der Validation-Ergebnisse behauptet. Der Holdout ist ein fester Split aus den äußeren Folds der 5-Fold-Datei (Folds 3–5 Training, Fold 2 Validation, Fold 1 Test), keine vollständig ausgeführte 5-Fold-Cross-Validation.

| Teilmenge | Bilder | Fälle | davon maligne |
|---|---:|---:|---|
| Training | 1.114 | 639 | 360 Bilder |
| Validation | 385 | 213 | 125 Bilder aus 69 Fällen (benigne: 260 Bilder aus 144 Fällen) |
| Test | 376 | 212 | gesperrt |

Trainings-Seeds: 7, 21, 42, 123, 2026. Fünf Seeds messen die Trainingsstreuung auf demselben Holdout, keine Streuung über neue Kohorten; mehrere Bilder eines Falls sind abhängig.

## Die sieben Vergleichsgruppen (27 Kombinationen)

| Gruppe | Quelle | Realisierungen | Eingabebedingung |
|---|---|---|---|
| Dummy (Klassenprior) | 2B | 1 (kein Seed) | no image information (constant class-prior score) |
| LogReg L2, 64×64 | 2B | 1 (kein Seed) | full image, 64 x 64 grayscale, flattened |
| Kleines CNN | 3F | 5 Seeds (7, 21, 42, 123, 2026) | full image, 160 x 160, flip augmentation |
| MobileNetV2 frozen, Vollbild (4B) | 4B | 5 Seeds (7, 21, 42, 123, 2026) | full image, 160 x 160, flip + rotation + zoom augmentation |
| MobileNetV2 feinjustiert, Vollbild (4B) | 4B | 5 Seeds (7, 21, 42, 123, 2026) | full image, 160 x 160, flip + rotation + zoom augmentation |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 4E | 5 Seeds (7, 21, 42, 123, 2026) | expert-mask ROI crop (roi-expert-mask-v1), 160 x 160; needs a manual mask at application time |
| U-Net → automatische ROI → feinjustiert (4F-B) | 4F-B | 5 Seeds (7, 21, 42, 123, 2026) | automatic ROI from the stored 4F-A U-Net (seed 7), largest 8-connected component, frozen 20 % box rule; empty mask → full image; image only at application time |

Das sind 27 Kombinationen aus Modell, Seed und Eingabebedingung. Experten-ROI und automatische ROI verwenden dieselben fünf 4E-Klassifikatoren unter verschiedenen Eingabebedingungen. Dummy und LogReg haben genau eine Realisierung ohne Seed (leeres Seed-Feld; kein künstlicher Seed 7). 4B ist die Vollbild-Hauptreferenz (160 Pixel, Spiegeln/Rotation/Zoom, erweitertes Fine-Tuning mit Adam 1e-4). Der Experten-ROI-Zweig benötigt bei der Anwendung eine manuelle Maske. Der automatische Zweig benötigt nur das Bild: gespeichertes U-Net aus 4F-A (ein Seed), Schwelle 0,5, größte 8er-Komponente, feste Box- und 20-%-Regel des ROI-Vertrags; eine leere Maske führt zum Vollbild mit demselben ROI-Klassifikator (kein Wechsel auf den 4B-Klassifikator). Kein Seed wird nach AUC, Schwelle oder Heatmap ausgewählt; es gibt kein Ensemble. Die Trainingsstreuung des U-Nets wurde nicht untersucht.

Das Quellenregister (`reports/tables/final_evaluation_sources.csv`) nennt je Kombination die Validation-Scoredatei, Status- und Konfigurationsdateien, den Modellpfad und die vollständigen SHA-256-Prüfsummen sowie die ROI- und U-Net-Abhängigkeiten; alle Pfade sind projektrelativ. Die Modelldateien werden nur als Bytes gehasht, nicht geladen.

**Provenienz von 4F-B.** Das Auswertungsskript der Studie 4F-B wurde nach dem Studienlauf durch eine Schutzkorrektur (Wiederanlauf- und Provenienzschutz) verändert und hat deshalb einen anderen Hash als der bei der historischen Planregistrierung gespeicherte. Der historische Berechnungscode (Skript `8e31e78a…`, Modul `auto_roi.py` `60ffdb23…`) und der heutige Stand der Schutzkorrektur (Skript `41599e55…`) sind im Plan getrennt festgehalten. Akzeptiert wird nur diese dokumentierte Situation: alle 64 hashbaren Ergebnisdateien des Studienstatus (zusätzlich drei Statistikabbildungen, die nicht geöffnet werden) sind unverändert, `auto_roi.py` ist unverändert, und die fünf Erfolgsstatus tragen den registrierten historischen Codestand. Die historische Registrierung wurde nicht angepasst, das Skript wurde nicht erneut ausgeführt.

## Auswertungsregeln

- **Primärmetrik:** ROC-AUC auf Bildebene; **Sekundärmetrik:** Average Precision. Beide aus den ungerundeten gespeicherten Validation-Scores. Der Score ist der Score für malignant (positive Klasse = 1) und keine kalibrierte Wahrscheinlichkeit.
- **Betriebspunkte:** Sensitivität 90 % und 95 % sind gleichberechtigte Forschungs-Betriebspunkte; die Schwelle 0,5 bleibt die konventionelle Referenz. Es gibt keine feste Accuracy-Mindestmarke und keinen Optimalitätsanspruch.
- **Schwellenregel:** Eine Einheit ist positiv, wenn `score >= Schwelle`. Für das Ziel q ist die erforderliche Trefferzahl `ceil(q × Anzahl Maligne)` (exakte Bruchrechnung): Bildebene 113 (0,90) und 119 (0,95) von 125, Case-Ebene 63 und 66 von 69. Gewählt wird die **höchste** Schwelle, die mindestens diese Trefferzahl erreicht (der k-te größte Score unter den malignen Einheiten); gleiche Scores werden gemeinsam klassifiziert. Keine ROC- oder Quantilinterpolation, keine Toleranz, keine Optimierung nach F1, Accuracy oder Youden. Bei konstanten Scores (Dummy) kann die Zielschwelle alle Einheiten positiv klassifizieren; das ist ein korrektes Ergebnis.
- **Speicherung:** Jede Schwelle steht als kürzester Dezimaltext, der exakt denselben IEEE-754-Wert zurückliefert (`threshold`), und zusätzlich in der exakten Hexdarstellung (`threshold_hex`).
- **Case-Ebene:** Case-Score = Maximum der Bildscores des Cases; eigene Case-Schwellen nach derselben Regel. Die Bildschwellen angewendet auf Case-Scores bilden eine **getrennte historische Vergleichssicht** (`final_case_metrics_at_image_thresholds.csv`, 81 Zeilen) und sind nicht mit den Case-Schwellen zu verwechseln.
- **Kennzahlen je Zeile:** TP/FN/FP/TN, Sensitivität, Spezifität, Precision, NPV, F1, Accuracy, Anteil positiver Entscheidungen; eine Rate mit Nenner null ist undefiniert (leere Zelle), nicht 0.
- **Zusammenfassung je Gruppe:** Anzahl der Realisierungen, Mittelwert, Stichproben-SD (ddof = 1), Minimum, Maximum; bei Einzelmodellen ist die SD undefiniert. Ein Mittelwert wie 57,2 Fehlalarme ist ausdrücklich ein **Seed-Mittelwert**. Keine Pooling-AUC, kein Ensemble, keine Behandlung von 5 × 385 Scores als unabhängige Bilder. Die Zusammenfassungstabelle enthält auch einen deskriptiven Mittelwert der Schwellen über die Seeds; er ist **keine gemeinsame Entscheidungsgrenze**. Für die Anwendung gelten ausschließlich die in `final_operating_points.csv` gespeicherten Einzel-Schwellen je Pipeline, Seed und Ebene (Bild/Case).

## Eingefrorene Validation-Ergebnisse

Alle Zahlen stammen aus den veröffentlichten Tabellen (`final_validation_metrics*.csv`, `final_operating_point*.csv`) und sind **explorativ**: Die Validation-Bilder wurden bereits für die Entwicklung genutzt, und die Schwellen stammen aus denselben 385 Bildern (213 Fälle), auf denen sie hier beschrieben werden. Sie sind Betriebspunkte einzelner trainierter Modelle, keine universell optimalen Schwellen; die Schwellen verschiedener Seeds sind nicht austauschbar, und es gibt keinen besten Seed.

### Trennschärfe (ROC-AUC, Average Precision)

| Gruppe | Realisierungen | Bild-ROC-AUC (Seed-Mittel ± SD) | Bild-AP (Seed-Mittel ± SD) | Case-ROC-AUC (Maximum der Bildscores) |
|---|---:|---:|---:|---:|
| Dummy (Klassenprior) | 1 | 0,500 | 0,325 | 0,500 |
| LogReg L2, 64×64 | 1 | 0,669 | 0,509 | 0,678 |
| Kleines CNN | 5 | 0,661 ± 0,060 | 0,504 ± 0,042 | 0,657 ± 0,066 |
| MobileNetV2 frozen, Vollbild (4B) | 5 | 0,797 ± 0,011 | 0,685 ± 0,011 | 0,829 ± 0,015 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 5 | 0,881 ± 0,011 | 0,803 ± 0,024 | 0,906 ± 0,010 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 5 | 0,921 ± 0,012 | 0,872 ± 0,014 | 0,937 ± 0,012 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 5 | 0,898 ± 0,011 | 0,820 ± 0,032 | 0,919 ± 0,012 |

### Benigne Fehlalarme bei den Sensitivitätszielen

Bildebene (260 benigne Bilder; erkannt werden mindestens 113 bzw. 119 der 125 malignen Bilder):

| Gruppe | Fehlalarme bei ≥ 113 von 125 (Ziel 0,90): Seed-Mittel ± SD (Min–Max) | Fehlalarme bei ≥ 119 von 125 (Ziel 0,95): Seed-Mittel ± SD (Min–Max) |
|---|---:|---:|
| Dummy (Klassenprior) | 260,0 | 260,0 |
| LogReg L2, 64×64 | 170,0 | 218,0 |
| Kleines CNN | 196,0 ± 18,3 (177–217) | 221,2 ± 16,2 (206–241) |
| MobileNetV2 frozen, Vollbild (4B) | 145,4 ± 10,0 (137–161) | 178,2 ± 18,5 (157–208) |
| MobileNetV2 feinjustiert, Vollbild (4B) | 90,0 ± 17,1 (74–110) | 146,8 ± 22,8 (122–180) |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 57,2 ± 20,0 (34–89) | 116,4 ± 19,1 (95–136) |
| U-Net → automatische ROI → feinjustiert (4F-B) | 88,8 ± 9,4 (76–98) | 126,6 ± 20,3 (109–160) |

Case-Ebene (144 benigne Cases; Case-Schwellen; erkannt werden mindestens 63 bzw. 66 der 69 malignen Cases):

| Gruppe | Fehlalarme bei ≥ 63 von 69 (Ziel 0,90): Seed-Mittel ± SD (Min–Max) | Fehlalarme bei ≥ 66 von 69 (Ziel 0,95): Seed-Mittel ± SD (Min–Max) |
|---|---:|---:|
| Dummy (Klassenprior) | 144,0 | 144,0 |
| LogReg L2, 64×64 | 98,0 | 110,0 |
| Kleines CNN | 109,6 ± 11,3 (97–122) | 126,8 ± 6,9 (119–135) |
| MobileNetV2 frozen, Vollbild (4B) | 69,6 ± 6,4 (61–78) | 83,6 ± 6,1 (76–92) |
| MobileNetV2 feinjustiert, Vollbild (4B) | 37,4 ± 7,8 (26–45) | 62,0 ± 19,1 (39–92) |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 26,2 ± 10,1 (16–38) | 53,2 ± 10,4 (41–64) |
| U-Net → automatische ROI → feinjustiert (4F-B) | 33,0 ± 5,6 (27–40) | 61,2 ± 6,5 (55–71) |

Das Sensitivitätsziel ist in allen 108 Zielzeilen (27 Kombinationen × 2 Ebenen × 2 Ziele) erreicht; bei Dummy sind dafür alle Einheiten positiv, und LogReg sowie das kleine CNN erzeugen dabei viele Fehlalarme. Die Unterschiede zwischen den Gruppen sind beschreibend. Auf Test werden die festgelegten Vergleiche mit Unsicherheitsintervallen ausgewertet (siehe unten).

### Schwellen und Fehlalarme der drei Hauptbedingungen je Seed

Schwellen (Anzeige auf vier Stellen gerundet; die exakten Werte stehen in `final_operating_points.csv`):

| Bedingung | Seed | Bild-Schwelle 0,90 | Bild-Schwelle 0,95 | Case-Schwelle 0,90 | Case-Schwelle 0,95 |
|---|---:|---:|---:|---:|---:|
| MobileNetV2 feinjustiert, Vollbild (4B) | 7 | 0,3202 | 0,1282 | 0,4173 | 0,3230 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 21 | 0,1660 | 0,0396 | 0,4569 | 0,1311 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 42 | 0,1551 | 0,0503 | 0,3088 | 0,2537 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 123 | 0,3008 | 0,1868 | 0,3617 | 0,3027 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 2026 | 0,2328 | 0,1833 | 0,4089 | 0,3551 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 7 | 0,2421 | 0,1082 | 0,3450 | 0,2208 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 21 | 0,2023 | 0,1176 | 0,3366 | 0,2023 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 42 | 0,1688 | 0,0443 | 0,1971 | 0,1326 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 123 | 0,2060 | 0,1639 | 0,2622 | 0,2060 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 2026 | 0,2190 | 0,0729 | 0,3229 | 0,0847 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 7 | 0,1852 | 0,1252 | 0,3107 | 0,1852 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 21 | 0,1234 | 0,0632 | 0,3050 | 0,1614 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 42 | 0,0976 | 0,0298 | 0,2362 | 0,1057 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 123 | 0,1911 | 0,1642 | 0,2767 | 0,1913 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 2026 | 0,1119 | 0,0494 | 0,2899 | 0,1119 |

Benigne Fehlalarme je Seed (absolute Zahlen):

| Bedingung | Seed | Bild: FP bei 0,90 | Bild: FP bei 0,95 | Case: FP bei 0,90 | Case: FP bei 0,95 |
|---|---:|---:|---:|---:|---:|
| MobileNetV2 feinjustiert, Vollbild (4B) | 7 | 79 | 153 | 41 | 61 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 21 | 107 | 180 | 26 | 92 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 42 | 80 | 150 | 33 | 39 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 123 | 74 | 122 | 42 | 57 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 2026 | 110 | 129 | 45 | 61 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 7 | 54 | 130 | 21 | 44 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 21 | 51 | 97 | 20 | 41 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 42 | 58 | 136 | 38 | 55 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 123 | 89 | 124 | 36 | 64 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 2026 | 34 | 95 | 16 | 62 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 7 | 83 | 109 | 33 | 60 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 21 | 90 | 130 | 27 | 56 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 42 | 97 | 160 | 37 | 64 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 123 | 98 | 121 | 40 | 71 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 2026 | 76 | 113 | 28 | 55 |

### Referenz bei Schwelle 0,5

| Gruppe | Bild: Sensitivität / Spezifität bei 0,5 (Seed-Mittel) | Bild: positive Entscheidungen bei 0,5 (Anteil) | Case: Sensitivität / Spezifität bei 0,5 (Seed-Mittel) |
|---|---:|---:|---:|
| Dummy (Klassenprior) | 0,000 / 1,000 | 0,000 | 0,000 / 1,000 |
| LogReg L2, 64×64 | 0,376 / 0,781 | 0,270 | 0,507 / 0,688 |
| Kleines CNN | 0,066 / 0,981 | 0,034 | 0,096 / 0,969 |
| MobileNetV2 frozen, Vollbild (4B) | 0,512 / 0,869 | 0,255 | 0,687 / 0,792 |
| MobileNetV2 feinjustiert, Vollbild (4B) | 0,682 / 0,896 | 0,291 | 0,820 / 0,844 |
| MobileNetV2 feinjustiert, Experten-ROI (4E) | 0,675 / 0,956 | 0,249 | 0,771 / 0,938 |
| U-Net → automatische ROI → feinjustiert (4F-B) | 0,656 / 0,932 | 0,259 | 0,745 / 0,907 |

Die Betriebspunkte liegen bei 0,5 weit auseinander (zum Beispiel arbeitet das kleine CNN mit nahezu nur negativen Entscheidungen); ein Vergleich bei 0,5 ist deshalb kein Vergleich bei gleicher Sensitivität.

## Prüfungen dieses Schritts

- Alle 27 Score-Tabellen enthalten genau die 385 Validation-IDs je einmal, 213 Cases, Labels und Klassenbelegung wie der eingefrorene Split, ein Label je Case, endliche numerische Scores in [0, 1]; kein Test-Datensatz; Train-Zeilen gemischter Dateien wurden nur zum Filtern gelesen. Dateien ohne Splitspalte (4F-B) wurden anhand der ID-Liste als reine Validation-Dateien bestätigt.
- ROC-AUC, AP und die Konfusion bei 0,5 aller 27 Kombinationen stimmen mit den von den Quellen selbst gespeicherten Kennzahlen überein (größte Abweichung 0). Die Bild-Schwellen, Konfusionen und die Case-Zahlen unter Bildschwellen von 60 historischen Betriebspunkten (4B frozen und feinjustiert, 4E feinjustiert, 4F-B feinjustiert; fünf Seeds × drei Punkte) stimmen exakt mit der Tabelle `auto_roi_operating_points.csv` überein.
- Die Schwellenregel wurde je Zielpunkt unabhängig geprüft (exakte Trefferzahl, kurvenbasierte Herleitung, die nächsthöhere Score-Stufe verfehlt das Ziel, Roundtrip des Schwellentexts).
- Der Schritt öffnet keine Bilder, Masken oder Vorschauen, lädt kein Modell, importiert weder TensorFlow noch Keras und liest keine Testvorhersagen. Das wurde im Lauf selbst und durch ein unabhängiges Zugriffsprotokoll belegt.
- Funktionstests mit synthetischen Scores liefen vor der Registrierung; Wiederanlauf- und Fehlerfälle wurden in isolierten Kopien mit Kopien statt Verknüpfungen auf Projektergebnisse geprüft.

## Späterer Testauftrag (festgelegt, in 5A nicht ausgeführt)

Dieses Protokoll gilt für die spätere Abschlussphase. Die technischen Nachweise aus Schritt 5B (Grad-CAM, FGSM) sind nicht Teil von 5A und werden hier nicht als bestanden dargestellt. Erzwingt ihre Prüfung eine inhaltliche Änderung, ist vor der Testfreigabe eine begründete neue Protokollversion nötig; der eingefrorene Plan wird nicht stillschweigend geändert.

1. **Umfang:** Die sieben Gruppen und alle 27 Kombinationen werden unverändert auf Test bewertet. Bild- und Case-Schwellen stammen ausschließlich aus 5A. Eine Unterschreitung des Sensitivitätsziels auf Test wird berichtet, nicht durch Nachjustieren korrigiert.
2. **Automatische Masken/ROIs** werden ausschließlich aus Testbildern erzeugt und eingefroren, bevor Test-Expertenmasken für den Expertenzweig oder für Diagnosen genutzt werden. U-Net-Dice/IoU und ROI-Abdeckung werden als getrennte Qualitätsgrößen berichtet.
3. **Statistik der drei Hauptvergleiche:** 2.000 gepaarte, nach Case-Label geschichtete Cluster-Bootstrap-Wiederholungen mit RNG-Seed 2026. Alle Bilder eines Cases werden zusammen gezogen; wiederholte Cases behalten ihre Multiplizität. Dieselben Ziehungen gelten für alle Pipelines und Seeds. Die Kennzahlen werden zuerst je Seed berechnet, dann werden die gepaarten Differenzen gemittelt (keine Score-Ensembles). Berichtet werden zweiseitige 95-%-Perzentilintervalle der AUC-Differenz sowie der Sensitivitäts- und Spezifitätsdifferenzen an den festgelegten Zielpunkten auf Bild- und Case-Ebene. Seed-SD und Stichprobenunsicherheit werden getrennt erklärt; die Intervalle berücksichtigen keine komplette neue Trainings- und Schwellenwahl und keine externe Klinik.
4. **Grad-CAM:** technische Vorbereitung in 5B, Hauptinterpretation nach den Testmetriken. Darstellungsseed 7, Zielschicht `Conv_1`, Malignant-Logit vor der Sigmoid-Funktion, korrektes Entfernen von Padding und Rückabbilden der ROI. Außerhalb des Ausschnitts gibt es keine berechnete Karte des ROI-Klassifikators. Keine Gewichts-Updates. Eine plausible Heatmap beweist keine fachliche Eignung.
5. **Grad-CAM-Beispiele:** unter der eingefrorenen Bild-95-%-Schwelle des automatischen Zweigs bis zu zwei Bilder aus unterschiedlichen Cases je TP/TN/FP/FN, deterministisch nach ID und insgesamt Case-distinkt; dieselben Bilder unter allen drei Hauptbedingungen; bis zu vier weitere deterministisch gewählte Lokalisationssonderfälle, Überschneidungen entfernt; fehlende Kategorien bleiben offen. Die 95-%-Schwelle fixiert nur die Bildauswahl; 90 % und 95 % bleiben statistisch gleichberechtigt.
6. **FGSM:** Vorbereitung in 5B, Hauptauswertung nach den sauberen Testmetriken. 4B und automatische ROI, je fünf Seeds; Epsilon 0, 1/255, 2/255, 4/255 im Modelleingang [0, 1]; BCE mit den echten Labels, ungezielter Angriff, Clipping; künstliches Padding bleibt konstant. Der Angriff der automatischen ROI setzt erst am festgelegten Klassifikatoreingang an und bewertet nicht die gesamte Segmentierungs- und Zuschnittskette. Keine erneute Schwellenwahl; Änderungen werden mit dem ungestörten Zustand derselben Pipeline verglichen.
7. **Geräteanalyse:** Bild- und Case-Zahlen, Klassenbelegung, Konfusionen und Kennzahlen bei den globalen Schwellen, keine gerätespezifischen Grenzwerte. Cases mit mehreren Geräten gelten als gemischt; undefinierte Kennzahlen werden offen ausgewiesen. Die Analyse ist deskriptiv und kein Nachweis demografischer Fairness.
8. **Phase:** Testmetriken, Grad-CAM und FGSM bilden eine vorher festgelegte Abschlussphase. Die geplante Mehrfachverarbeitung der Testbilder ist erlaubt; Anpassung von Modellen und Schwellen anhand ihrer Ergebnisse ist nicht vorgesehen. Rein technische Auswertungsfehler dürfen mit dokumentierter Korrektur behoben werden.

## Nachtrag vom 21.10.2026: exploratives Zusatzexperiment 5B-R (Testumfang unverändert)

Vor der Testphase wurde ein begrenztes, exploratives Zusatzexperiment zum Adversarial Training des Experten-ROI-Klassifikators ergänzt (Schritt 5B-R, Beschreibung und Ergebnisse: [adversarial_roi_comparison.md](adversarial_roi_comparison.md)). Anlass war der FGSM-Stresstest aus Schritt 5B; die Validation-Ergebnisse der Pipelines waren bei der Planung bekannt, das Experiment wurde nicht von Beginn an als Teil des Projektplans geplant.

Für dieses Protokoll gilt:

- **Daten:** Training und Auswertung des Zusatzexperiments nutzen ausschließlich Train und Validation. Es wurde keine Testdatei geöffnet.
- **Testumfang unverändert:** Die sieben Vergleichsgruppen und die 27 Kombinationen aus Modell, Seed und Eingabebedingung bleiben der Umfang der späteren Testphase. Die zehn neuen Fortsetzungsläufe (je Seed eine Fortsetzung mit sauberen Bildern und eine mit FGSM-Beispielen) und die fünf Referenzzustände sind Validation-Zusatzmodelle und **nicht** Teil dieser 27 Kombinationen. Ihre Aufnahme in die Testphase würde einen gesondert beauftragten Protokollnachtrag erfordern; sie ist nicht entschieden.
- **Schwellen:** Die in 5A eingefrorenen Schwellen gehören zu den ursprünglichen Checkpoints und werden für die neuen Zustände nicht wiederverwendet. Es wurde keine neue Schwellensuche für 90 % oder 95 % Sensitivität durchgeführt.
- **Unverändert:** Die 5A-Registrierung, die maschinenlesbaren 5A- und 5B-Dateien und alle bisherigen Checkpoints (auch die fünf ursprünglichen 4E-Klassifikatoren) wurden nicht verändert.
- **Numerik:** Für die Berechnungen der Zusatzstudie gilt dieselbe beobachtete Abhängigkeit wie in Schritt 5B: Der Vorwärtspfad innerhalb des Gradientengraphen weicht unter TF32 vom Inferenzpfad ab. Berichtet werden deshalb Scores und Losswerte aus dem Inferenzpfad; der Gradientengraph liefert die Richtung. Eine konkrete Ursache der Abweichung ist nicht nachgewiesen; es wurde keine Toleranz gelockert und der numerische Modus nicht geändert.

## Dateien und Wiederaufruf

| Datei | Inhalt |
|---|---|
| `analysis/07_operating_points.py` | Interpreterprüfung, Quellen- und Hashkette, Plan-Registrierung, Berechnung, Veröffentlichung, schreibgeschützte Prüfung |
| `src/busbra/final_evaluation.py` | CPU-Helfer: Adapter, Schwellenregel, Zusammenfassungen, Gegenprüfungen |
| `reports/tables/final_evaluation_plan.json` | unveränderliche Regeln und Provenienz (inklusive Quellenregister und Testprotokoll) |
| `reports/tables/final_evaluation_sources.csv` | die 27 Kombinationen mit Quellen und Prüfsummen |
| `reports/tables/final_operating_points.csv` | die 162 Kernzeilen (27 × 2 Ebenen × 3 Punkte) mit Schwellen und Konfusionen |
| `reports/tables/final_validation_metrics.csv`, `final_validation_metrics_summary.csv` | Bild- und Case-AUC/AP je Kombination und je Gruppe |
| `reports/tables/final_operating_point_summary.csv` | Zusammenfassung der Betriebspunkte über die Seeds |
| `reports/tables/final_case_metrics_at_image_thresholds.csv` | historische Vergleichssicht: Bildschwellen auf Case-Scores (81 Zeilen) |
| `reports/tables/final_evaluation_status.json` | Erfolgsstatus mit Ausgabe-Hashes und Zugriffsnachweis (zuletzt geschrieben) |

Aufruf (aus dem Projektordner): `env -u LD_LIBRARY_PATH .venv/bin/python analysis/07_operating_points.py`. Der Schritt braucht kein TensorFlow; der bereinigte `LD_LIBRARY_PATH` bleibt für die späteren TensorFlow-Schritte dokumentiert. Ein erneuter Aufruf eines vollständigen Erfolgs prüft Status, Plan, alle Ausgaben und alle Eingaben anhand der Hashes schreibgeschützt (Exit 0, keine neuen Dateien). Ergebnisse ohne gültigen Erfolgsstatus werden als `recovery_required` gemeldet und nie überschrieben; unter einer bestehenden Registrierung rechnet nur unveränderter Code (Skript, `final_evaluation.py`, `evaluation.py`, `thresholds.py`). Eine tatsächlich nötige neue Berechnung erfordert eine neue, ausdrücklich versionierte Planregistrierung.

## Grenzen

- Explorativ entwickelt auf Train/Validation; die Validation-Zahlen sind bekannt und die Schwellen tragen die Stichprobenunsicherheit von 385 Bildern beziehungsweise 213 Fällen (69 maligne Cases).
- Fester Holdout, eine Datenquelle (eine Einrichtung, vier Geräte), keine externe Validierung; Seeds messen Trainingsstreuung auf demselben Holdout.
- Der Experten-ROI-Zweig benötigt bei der Anwendung eine manuelle Maske und ist keine reine Bild-zu-Score-Pipeline; der automatische Zweig nutzt ein U-Net mit einem Seed, und seine Klassifikatoren wurden auf Experten-Zuschnitten trainiert.
- Eingebrannte Texte und Messmarker sind mögliche Shortcuts und nicht entfernt; Masken und Labels werden nicht klinisch nachbegutachtet.
- Nur Läsionen, keine Screening-Population; keine Messung ärztlicher Entscheidungen; keine klinische Einsatzfähigkeit.
