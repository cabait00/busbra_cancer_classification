# Projektdefinition – BUS-BRA

Stand: 21.10.2026 (Abschluss). Dieses Dokument legt Frage, Eingabebedingungen, Daten- und Auswertungsvertrag und die Grenzen des Projekts fest. Die Ergebnisse stehen in der [Projektdokumentation](BUSBRA_Projektdokumentation.md); die einzelnen Studien sind in den jeweiligen Dokumenten beschrieben (Übersicht: [README.md](../README.md)).

## Titel und Forschungsfrage

**KI-gestützte Beurteilung von Brustultraschall: Vollbild oder Läsionsausschnitt?**

> Verbessert die Eingrenzung des Läsionsbereichs die Unterscheidung gutartiger und bösartiger Befunde im Brustultraschall?

Auf dem BUS-BRA-Datensatz werden vollständige Bilder mit Läsionsausschnitten verglichen, deren Position entweder aus Expertenmasken oder aus einer automatischen U-Net-Segmentierung stammt. Bewertet werden die Klassifikationsleistung, die Streuung über Trainings-Seeds und die Fehlalarme bei festgelegten Sensitivitätszielen. Drei Vergleiche werden gleichwertig berichtet: Vollbild gegen Experten-ROI, Experten-ROI gegen automatische ROI, Vollbild gegen automatische ROI. Die ursprüngliche Unterfrage bleibt erhalten: Wie gut lassen sich gutartige und bösartige BUS-BRA-Läsionen allein anhand des Bildes auf ungesehenen Fällen unterscheiden, und bringt ein CNN einen Mehrwert gegenüber einer einfachen lernenden Baseline?

**Zielgruppe und unterstützte Entscheidung.** Ein technisches Team, das computergestützte Entscheidungsunterstützung bei **bereits sichtbaren Läsionen** untersucht. Es wird keine Behandlungs- oder Biopsieentscheidung automatisiert, keine Entwarnung für einzelne Patientinnen gegeben und kein Screening gesunder Personen untersucht. Der Datensatz enthält nur Läsionen; „gutartig“ bedeutet eine gutartige Läsion.

## Eingabebedingungen und Informationsregeln

| Zweig | Eingabe des Klassifikators | bei der Anwendung benötigt | Maske |
|---|---|---|---|
| **Vollbild** | das ganze Ultraschallbild (seitenverhältnistreu auf 160 × 160, Rand schwarz) | nur das Bild | keine Verwendung als Eingabe; nur Integritätsprüfung und Referenz in der Interpretation |
| **Experten-ROI** (Vertrag `roi-expert-mask-v1`) | unmaskierter Zuschnitt um die Box aller Maskenpixel, 20 % je Seite erweitert, an den Bildrändern begrenzt (feste Wahl, nicht optimiert; [masks_and_roi_definition.md](masks_and_roi_definition.md)) | Bild **und manuelle Maske** | lokalisiert den Zuschnitt (kein Kanal, keine Schwärzung); die Leistung ist keine reine Bild-zu-Score-Leistung |
| **Automatische ROI** | Zuschnitt nach derselben Regel aus der Vorhersage eines U-Net (nur Bild als Eingabe; leere Vorhersage: Vollbild); die Klassifikatoren sind die des Experten-ROI-Zweigs und wurden nicht auf automatischen Zuschnitten trainiert | nur das Bild | die Maske ist Trainingsziel und Messreferenz des U-Net, nicht Eingabe des Klassifikators |

Keine Modelleingaben sind in keinem Zweig: Fall (dient nur der Trennung der Teilmengen), Histologie, BI-RADS, Gerät (nur deskriptive Teilgruppenauswertung), Seite und Bounding-Box-Angaben. Es gibt keine Texterkennung und keine Entfernung eingebrannter Markierungen. Maske und Labels sind verschieden: Die Maske umreißt die Läsion (auch gutartige haben eine Maske); die Klassifikatoren lernen ausschließlich `Pathology`.

## Daten und Zielgröße

* **Datensatz:** BUS-BRA, Zenodo-Record 8231412, Version 1.0 (CC BY 4.0 laut Metadaten; die mitgelieferte `LICENSE.txt` verlangt die Zitierung der Veröffentlichung). Erwartet und beobachtet: 1.875 Bilder, 1.875 Masken, 1.064 Fälle; 1.268 gutartige und 607 bösartige Bilder; vier Geräte. Rohdatenprüfung und Abweichungen: [data_audit.md](data_audit.md).
* **Zielgröße:** `Pathology` (gutartig = 0, bösartig = 1); positive Klasse ist bösartig. Der Modell-Score ist der Score für bösartig und keine kalibrierte individuelle Wahrscheinlichkeit.
* **Kennungen:** `ID` und `Case` bleiben Zeichenketten; Bild, Maske und Metadaten werden ausdrücklich über die ID verknüpft. Die tatsächlichen Bildgrößen gelten; vier CSV-Höhen weichen ab und bleiben dokumentiert.

## Split und Auswertungsvertrag

* **Fester Holdout** aus der äußeren Spalte `kFold` der ursprünglichen `5-fold-cv.csv`: Folds 3–5 Training, Fold 2 Validation, Fold 1 Test. Kanonische Zuordnung: `data/processed/manifest_split.csv` (Status und Prüfsumme: `reports/tables/split_status.json`); sie wird nie überschrieben. Die inneren `valid_*`-Spalten werden nicht verwendet. Keine 5-fach-Kreuzvalidierung.

| Teilmenge | Bilder | Fälle | Bilder gutartig / bösartig | Fälle gutartig / bösartig |
|---|---:|---:|---:|---:|
| Training | 1.114 | 639 | 754 / 360 | 434 / 205 |
| Validation | 385 | 213 | 260 / 125 | 144 / 69 |
| Test | 376 | 212 | 254 / 122 | 144 / 68 |
| gesamt | 1.875 | 1.064 | 1.268 / 607 | 722 / 342 |

* **Auswertungseinheit** sind Bilder mit fallgetrennten Teilmengen; es werden stets Bild- und Fallzahlen genannt. Der Fall-Score ist das Maximum der Bild-Scores des Falls. Zwei Bilder eines Falls sind nicht unabhängig.
* **Metriken:** ROC-AUC (primär) und Average Precision aus kontinuierlichen Scores; ferner Sensitivität, Spezifität, Precision, F1, Konfusionszahlen und Accuracy. Eine Rate mit Nenner 0 ist undefiniert, nicht 0. Teilgruppen werden mit Nennern berichtet; Gerätevergleiche sind deskriptiv.
* **Datentrennung:** gelernte Vorverarbeitung und Klassengewichte nur auf dem Training (verwendet wird nur die feste Skalierung durch 255); Augmentation nur im Training; kein `validation_split` als Ersatz für die Falltrennung.
* **Auswahl:** Checkpoints, Modell- und Hyperparameterwahl und Schwellen nur auf der Validation. Konfiguration und Schwellen wurden vor der Testauswertung eingefroren; nach Einsicht in Testergebnisse wird nichts mehr optimiert. Der Test wurde in der Abschlussphase einmal ausgewertet und danach nur für die vorab festgelegte Interpretation (Grad-CAM) und Robustheitsprüfung (FGSM, Geräte) verwendet.
* **Betriebspunkte:** je Kombination aus Modell, Seed und Eingabebedingung Bild- und Fallschwellen für die Referenz 0,5 und die gleichberechtigten Sensitivitätsziele 0,90 und 0,95 (höchste Schwelle mit mindestens `⌈q × Positive⌉` Treffern; [final_evaluation_protocol.md](final_evaluation_protocol.md)). „Ziel“ bezeichnet die Auswahl auf der Validation; die auf dem Test erreichte Sensitivität wird gesondert berichtet.
* **Statistik der drei Hauptvergleiche:** gepaarte, nach Fall-Label geschichtete Cluster-Bootstrap-Ziehungen (2.000, Zufallsgenerator-Seed 2026) mit 95-%-Perzentilintervall; die Intervalle beschreiben die Stichprobenunsicherheit bei festen Modellen. Fünf Seeds (7, 21, 42, 123, 2026) messen die Trainingsstreuung auf demselben Holdout; kein bester Seed, kein Ensemble, keine gepoolte AUC.

## Modelle und Gruppen

Sieben Gruppen mit 27 Kombinationen aus Modell, Seed und Eingabebedingung:

1. **Dummy** (Klassenprior; ein Modell) und 2. **logistische Regression** (L2, 64 × 64 Pixel, abgeflacht; ein Modell) als Referenzen ([baselines.md](baselines.md)).
3. **Kleines CNN** (drei Faltungsblöcke, Global Average Pooling; 100.993 Parameter; fünf Seeds; [small_cnn.md](small_cnn.md)).
4. **MobileNetV2 frozen** (ImageNet-Vortraining, nur der Kopf trainiert; fünf Seeds) und 5. **MobileNetV2 feinjustiert** (Blöcke 14–16 und `Conv_1` samt Kopf, Adam 1e-4, alle 52 BatchNorm-Schichten fest; Spiegeln, Rotation ±10 °, Zoom 0,10; fünf Seeds): die Vollbild-Hauptreferenz ([transfer_learning.md](transfer_learning.md), [extended_fine_tuning.md](extended_fine_tuning.md), [augmentation_160_comparison.md](augmentation_160_comparison.md)).
6. **Experten-ROI** (gleiche Architektur auf Experten-Zuschnitten; fünf Seeds; [expert_roi_comparison.md](expert_roi_comparison.md)).
7. **Automatische ROI** (U-Net, ein Seed, mit denselben fünf Experten-ROI-Klassifikatoren; [segmentation_4fa.md](segmentation_4fa.md), [auto_roi_classification.md](auto_roi_classification.md)).

Auflösung (Klassifikatoren 160 × 160, U-Net 256 × 256) und Batchgröße (Klassifikatoren 16, U-Net 8) wurden nach der GPU-Prüfung festgelegt; Mixed Precision wurde nicht verwendet. Es gibt keine GAN-, VAE-, GNN-, ViT- oder LLM-Modelle und keine breite Hyperparametersuche.

## Interpretation und Robustheit

Eine Erklärungsmethode: Grad-CAM auf der letzten Faltung für den Malignant-Logit (elf nach festen Regeln gewählte Testbilder, ein Seed; [gradcam_test_interpretation.md](gradcam_test_interpretation.md)). Ein kleiner, vorab festgelegter FGSM-Stresstest (ε = 0, 1/255, 2/255, 4/255 auf der Klassifikatoreingabe [0, 1], echte Labels, Begrenzung auf [0, 1]) und eine beschreibende Aufschlüsselung nach Gerät ([robustness_and_devices.md](robustness_and_devices.md)). Ein begrenztes, ursprünglich nicht geplantes Zusatzexperiment zum Adversarial Training der Experten-ROI-Klassifikatoren (5B-R, nur Training und Validation, nicht Teil der 27 Testkombinationen) ist in der [Projektdokumentation](BUSBRA_Projektdokumentation.md) (Abschnitt 7.3) und in [adversarial_roi_comparison.md](adversarial_roi_comparison.md) beschrieben; es ist kein Präzedenzfall für weitere Angriffe oder Trainingsverfahren.

## Methodischer Status

Die Modellentwicklung war explorativ auf Training und Validation; die Validation wurde wiederholt genutzt, ihre Ergebnisse sind bekannt und möglicherweise optimistisch. Die Abschlussregeln (Gruppen, Schwellen, Statistik) wurden vor der Auswertung des Tests schriftlich und hashgesichert festgehalten; eine Registrierung der Hypothese vor der Entwicklung wird nicht behauptet. Die Segmentierung (4F-A) wurde umgesetzt, ausgeführt, lokal geprüft; das U-Net wurde danach in 4F-B, 5A und 5C unverändert und per Hash geprüft verwendet.

## Zulässige Aussagen und Grenzen

**Zulässig:** interne Klassifikationsleistung auf Läsionsbildern von BUS-BRA-Fällen, die nicht zum Training verwendet wurden, getrennt nach den drei Eingabebedingungen. **Nicht zulässig:** klinische Einsatzreife, Screening-Leistung bei gesunden Personen, Robustheit über neue Einrichtungen oder Geräte, eine Aussage über ärztliche Entscheidungen.

**Grenzen:** eine Einrichtung mit vier Geräten und keine externe Validierung; die Klassenanteile unterscheiden sich je Gerät (30–49 % bösartige Bilder), und eingebrannte Texte, Mess- und Gerätemarker sind mögliche Abkürzungen; die Duplikatprüfung schließt nur exakte Duplikate aus; Masken und Labels wurden nicht klinisch nachbegutachtet; beim Fall 856 unterscheiden sich die BI-RADS-Angaben (dokumentiert, nicht korrigiert); der Experten-ROI-Zweig braucht eine manuelle Maske; das U-Net wurde nur mit einem Seed trainiert.

**Erfolgskriterium.** Es gibt keinen numerischen Mindestwert. Die Frage wird mit reproduzierbaren, leckagefreien Vergleichen beantwortet: Bild-ROC-AUC als Primärmetrik, Average Precision, Streuung über fünf Seeds, gepaarte Differenzen mit Intervall und benigne Fehlalarme an den Sensitivitätszielen. Ein Ergebnis ist auch dann gültig, wenn die Eingrenzung oder das CNN keine überzeugende Verbesserung zeigt; es ist kein Anlass zur nachträglichen Optimierung auf Testdaten.

## Rahmen und Ergebnis

Das Projekt entstand als Kursprojekt. Ergebnisse sind lauffähiger Code, eine deutsche Projektdokumentation ([BUSBRA_Projektdokumentation.md](BUSBRA_Projektdokumentation.md)) und ein Vortrag von etwa zehn Minuten (die Folien enthalten Ultraschallbilder und sind nicht Teil des Repositories). Hinweise zu Reproduzierbarkeit und Dateibestand: [reproducibility_and_handover.md](reproducibility_and_handover.md).
