# Schritt 4F-A – erste feste Segmentierung der annotierten Läsion

Stand: 20.10.2026. Status: umgesetzt, ausgeführt, lokal geprüft; die Sichtprüfung der lokalen Vorschauen war zum Zeitpunkt dieses Berichts offen und ist nachträglich erfolgt (Nachtrag vom 20.10.2026 im Abschnitt „Vorschau und Sichtprüfung“). Dieser Bericht beschreibt die Methode, den Lauf und die Ergebnisse; Zahlen stammen aus den gespeicherten Tabellen unter `reports/tables/segmentation/unet_seed7/` und `reports/tables/segmentation_*`.

**Kernaussage in einem Absatz.** Ein kompaktes U-Net, von Grund auf mit Seed 7 trainiert, segmentiert die annotierte Läsion auf den Validation-Bildern mit einem mittleren Pixel-Dice von 0,860 (Median 0,913) auf der Originalbildfläche; auf den Trainingsbildern sind es 0,917 (Median 0,935). Die Verteilung hat einen langen linken Schwanz: 16 von 385 Validation-Bildern (4,2 %) haben einen Dice unter 0,5, darunter 6 leere Vorhersagen. Es handelt sich um **einen** Lauf einer festen Architektur; die Validation steuert den Checkpoint und ist **keine unabhängige Abschlussbewertung**. Diese Werte sind keine Klassifikationsleistung, und ob vorhergesagte Masken als Lokalisation für einen Klassifikator taugen, ist nicht untersucht.

## Zweck und Abgrenzung

Schritt 4F-A trainiert eine **Segmentierung** der annotierten Läsion auf vollständigen Ultraschallbildern. Das Netz sagt für jedes Pixel voraus, ob es zur von Expertinnen oder Experten annotierten Läsion gehört. Es ist **keine Klassifikation**: Es entsteht kein Score für benign oder malignant, es gibt keine ROI-Klassifikation mit vorhergesagten Masken, keine Modell- oder Klassifikationsschwellenwahl und keinerlei Testzugriff. Die Eingabe ist ausschließlich das bereitgestellte Ultraschallbild; `Pathology`, `Case`, `Device`, BI-RADS und Bounding Boxes sind keine Modelleingabe. Die Expertenmaske ist das **Trainingsziel** und die **Messreferenz**.

Informationsbedingung: Der Segmentierungszweig arbeitet **ohne manuelle Maske zur Anwendungszeit**. Er ist der bildbasierte Gegenpart zum Experten-ROI aus Schritt 4E, dessen Ergebnis wegen der nötigen manuellen Maske keine reine Bild-zu-Score-Leistung war ([expert_roi_comparison.md](expert_roi_comparison.md)). Ob vorhergesagte Masken als Lokalisation für einen Klassifikator taugen, wird erst in einem späteren Schritt (4F-B) untersucht; dieser Schritt misst nur die Segmentierungsqualität.

**Wichtige Einschränkung zur Validation.** Die Validation-Bilder und ihre Expertenmasken werden in diesem Schritt dreifach verwendet: für den Segmentierungs-Loss, der Early Stopping steuert, für die Wahl des Checkpoints und für die Qualitätsmessung. Diese Validation ist deshalb **keine unabhängige Abschlussbewertung**; die Werte sind explorativ und wahrscheinlich etwas optimistisch. Die Testdaten bleiben gesperrt; die 376 Testzeilen des Manifests wurden nur gezählt.

## Festes Verfahren (vor dem ersten Training im Studienplan registriert)

| Bestandteil | Festlegung |
|---|---|
| Eingabe / Ausgabe | 256 × 256 × 1 `float32` in [0, 1] → 256 × 256 × 1, Sigmoid (Wahrscheinlichkeit „Läsionspixel“, keine kalibrierte klinische Wahrscheinlichkeit) |
| Architektur | kompakter U-Net-artiger Encoder-Decoder mit Skip-Verbindungen, von Grund auf trainiert: vier Encoder-Blöcke (16, 32, 64, 128 Filter), Bottleneck (256), vier Decoder-Stufen mit transponierter Faltung (2 × 2, Stride 2) und Konkatenation; Block = Conv 3 × 3 (`he_normal`, ohne Bias) → BatchNormalization → ReLU, zweimal; Kopf Conv 1 × 1 mit Sigmoid; kein Dropout, keine Zufallsschicht im Netz |
| Parameter | 1.945.233 insgesamt, 1.942.289 trainierbar, 2.944 nicht trainierbar (BatchNorm-Statistiken); 19 Faltungen, 4 transponierte Faltungen, 18 BatchNorm-Schichten |
| Seed | 7 (Python, NumPy, TensorFlow, Keras, Initialisierung; Strom für das Mischen: 10 = Seed + 3, Strom für den Spiegel: 8 = Seed + 1; `PYTHONHASHSEED=7`) |
| Loss | je Bild Binary Crossentropy (Mittel über alle Pixel, Wahrscheinlichkeiten auf [1e-7, 1 − 1e-7] begrenzt) **plus** (1 − Soft-Dice), gleiche Gewichte 1 : 1, Mittel über die Bilder des Batches |
| Soft-Dice | je Bild `(2 · Σ(p · g) + s) / (Σp + Σg + s)` mit den Summen über alle 256 × 256 Pixel und der Glättung **s = 1,0** (in Zähler und Nenner) |
| Optimierer | Adam, Lernrate 1e-3, Keras-Standardwerte (β₁ 0,9, β₂ 0,999, ε 1e-7), neu |
| Batchgröße | 8; nur wenn die echte GPU-Probe einen sauberen Speicherfehler zeigt, genau ein Neubeginn mit 4 (als Abweichung dokumentiert); keine CPU-Ausweichung. Tatsächlich verwendet: **8** |
| Training | höchstens 40 Epochen; Early Stopping auf `val_loss` (min), Geduld 7, `min_delta` 0, beste Gewichte wiederherstellen; Checkpoint (vollständiges Modell) bei kleinstem Validation-Loss (bei Gleichstand die erste beste Epoche); `float32`, `jit_compile=False`, keine gemischte Präzision, keine Klassengewichte; der letzte, kleinere Batch jeder Epoche bleibt erhalten |
| Augmentation | **nur** gemeinsames horizontales Spiegeln von Trainingsbild und Trainingsmaske (Wahrscheinlichkeit 0,5 je Bild und Epoche, **eine** Entscheidung je Bild, auf beide Arrays angewandt); Validation ohne Augmentation; keine Rotation, kein Zoom, keine Transformation, die Bild und Maske unterschiedlich behandelt |
| Binärschwelle | **fest 0,5** (Wahrscheinlichkeit ≥ 0,5), vor dem Training im Plan, nicht optimiert |
| Kurven während des Trainings | `dice_soft`, `dice_hard_0_5`, `iou_hard_0_5` auf dem Modellraster (nur Kurven; der Checkpoint folgt ausschließlich dem Validation-Loss) |

Es gibt keine zusätzliche Modellfamilie und keine Hyperparametersuche. Der Plan (`reports/tables/segmentation_study_config.json`, Studie `seg4fa-d5456bec`, Plan-SHA-256 `d5456becca3ce3c8…`) enthält außerdem Split-Hash, Bild- und Maskengeometrie, die Auswertungsmetriken, die Regeln der Vorschau-Auswahl, die Quellprüfsummen, den Zugriffsrahmen und den Übergang zu 4F-B; er enthält keinen Zeitstempel.

## Geometrie und Datenzugriff

- **Bild:** das ganze Bild wird mit der vorhandenen Regel `resize_with_padding` seitenverhältnistreu in 256 × 256 eingepasst (Pillow `ImageOps.contain`, bilinear) und auf schwarzem Hintergrund zentriert (Versatz `(256 − eingepasst) // 2`). Alle 1.499 Bilder haben eine Seite über 256 Pixel; es wird nie vergrößert.
- **Maske:** dieselbe eingepasste Größe und dieselben Versätze, aber Nearest-Neighbor-Interpolation; die Werte bleiben {0, 1}, das Padding ist 0. Eine Maske, die auf dem 256er-Raster leer würde, wäre ein Fehler (kein Fall der Daten).
- **Aufzeichnung je Bild** (lokale Tabelle `reports/tables/segmentation_geometry_train_validation.csv`): dekodierte Größe, eingepasste Größe, Versätze, Skalierungsfaktoren, Zahl der Maskenpixel (Original und Raster) und die Rundreise der Expertenmaske. Die dekodierten Größen steuern die Geometrie, **nie** die CSV-Spalten Width/Height: Für die vier bekannten Abweichungen (`bus_0856-l`, `bus_0857-r`, `bus_0912-l`, `bus_0990-l`) liegen alle vier in Train oder Validation und die Geometrie entspricht der dekodierten Größe.
- **Rückabbildung** (Vorhersage → Originalpixel): Ausschnitt des eingepassten Bereichs der 256 × 256-Wahrscheinlichkeitskarte (das **Padding wird verworfen**), bilineare Vergrößerung dieser Karte (`float32`) auf die dekodierte Größe, danach die Schwelle ≥ 0,5. Pixelmitten: `x_Modell = pad_x + (x + 0,5) · scale_x − 0,5` (und umgekehrt). Die **Rundreise der Expertenmaske** (auf das Raster abbilden und zurück) dient als **Rasterungsreferenz** (keine strenge mathematische Obergrenze des erreichbaren Dice) und wird mit ausgewiesen.
- **Maskenprüfung wie in 4D/4E:** PNG-Modus 1, `bool`, Wahrheitswert = annotierte Läsion, nicht leer, dekodierte Bild- und Maskengröße gleich und gleich den auditierten Werten; Bild im Modus `L`. Keine Reparatur, keine Umkodierung, keine Ausschlüsse.
- **Zugriff:** nur der eingefrorene, fallgetrennte Split (1.114 Trainingspaare, 639 Fälle; 385 Validation-Paare, 213 Fälle; keine Fallüberschneidung). Bild und Maske werden über die ID im Manifest verknüpft; das Manifest wird zeilenweise mit Split-Filter zuerst gelesen, die Testzeilen werden nur gezählt. Jeder Prozess zeichnet seine PNG-Öffnungen auf: erlaubt sind genau die 2.998 Dateien (Bilder und Masken von Train und Validation).

## Reihenfolge und Registrierung

Vorbedingungen (Interpreter, eingefrorener Split, eingefrorener ROI-Vertrag `roi-expert-mask-v1`, Quellprüfsummen, Zustand von 4E) → der unveränderliche Studienplan wird registriert (2026-10-20T20:30:37+02:00) → **erst danach** öffnet ein eigener CPU-Prozess die Train- und Validation-Paare, prüft sie und friert den Daten-Fingerabdruck und die Geometrietabelle ein (2026-10-20T20:30:46+02:00) → ein Worker-Prozess berechnet seine Eingaben neu und muss den Fingerabdruck reproduzieren, führt die echte GPU-Probe aus, setzt zurück, trainiert und wertet aus (Start 2026-10-20T20:30:46+02:00) → der Koordinator prüft den Lauf, zeichnet die Statistikfiguren aus den gespeicherten Tabellen und schreibt den Studienstatus. Ein Start der Datenprüfung ohne registrierten Plan bricht ab, bevor irgendeine Bild- oder Maskendatei geöffnet wird (getestet).

**Echte GPU-Probe.** Vor dem Haupttraining läuft mit dem Segmentierungsmodell selbst und den ersten acht Trainingsbildern ein vollständiger Schritt (Vorwärtslauf, Loss, Gradienten, Update) als kompilierter Graph auf GPU:0: alle Variablen, die Ausgabe, der Loss und alle 64 Gradienten liegen auf der GPU, alle Werte sind endlich, die trainierbaren Gewichte und die BatchNorm-Statistiken ändern sich. Danach folgt ein **vollständiger Reset**: neues Keras-Session-Objekt, Seed 7, Neuaufbau; die Gewichte (trainierbare und BatchNorm-Statistiken) müssen **bitgleich** den geplanten Anfangsgewichten entsprechen, Adam ist neu (0 Iterationen). Memory Growth wird vor der GPU-Initialisierung gesetzt, Soft Placement ist aus, fremde CUDA-Bibliotheken werden geprüft (keine), der Lauf nutzt den Projektinterpreter mit `env -u LD_LIBRARY_PATH`.

**Sichere Wiederaufnahme.** Ein Lauf, dessen Training Checkpoint oder History hinterließ, dessen Auswertung oder Berichtserstellung aber nicht abschloss, gilt als `recovery_required`: Koordinator und direkter Worker trainieren ihn nie erneut und löschen nichts; erfolgreiche Artefakte bleiben erhalten. Nur die Auswertung lässt sich gezielt aus dem gespeicherten Checkpoint wiederaufnehmen (`--resume-evaluation`, kein Training, History und Checkpoint bleiben byte-identisch, der frühere Fehlerstatus bleibt als technischer Beleg erhalten).

## Übergang zu 4F-B (in 4F-A registriert, **nicht** in 4F-A ausgeführt)

*Nachtrag 20.10.2026: Der Übergang wurde in Schritt 4F-B unverändert umgesetzt, als Inferenz mit den zehn unveränderten 4E-Klassifikatoren; siehe [auto_roi_classification.md](auto_roi_classification.md). Der folgende Text ist der in 4F-A registrierte Wortlaut.*

Der Weg von der vorhergesagten Maske zur ROI des späteren Schritts 4F-B ist im Studienplan **vor** der ersten Auswertung eines nachfolgenden Klassifikators festgelegt. In 4F-A entsteht kein Klassifikationsscore, und die Validation-Expertenmasken werden nicht zur Erzeugung von Klassifikations-ROIs verwendet.

1. Wahrscheinlichkeitskarte des besten 4F-A-Checkpoints (nicht nachgetunt), auf das dekodierte Bildraster zurückgebildet (Regel oben).
2. Binärmaske: Wahrscheinlichkeit ≥ 0,5.
3. Mehrere 8er-zusammenhängende Komponenten: nur die **größte** bleibt; bei Gleichstand in der Pixelzahl gewinnt die Komponente, deren oberstes linkestes Pixel in Zeilenreihenfolge zuerst kommt (kleinste Zeile, dann kleinste Spalte).
4. Leere Maske: ein ausdrücklich protokollierter **Vollbild-Fallback** (4B-Vollbildeingabe dieses Bildes); Zahl und IDs der Fallback-Bilder werden berichtet.
5. Danach exakt die Box-/20-%-Zuschnittregel aus dem eingefrorenen Vertrag `roi-expert-mask-v1` (enge Box der gewählten Komponente, Erweiterung um `ceil(Kante / 5)` je Seite, Beschnitt am Bildrand, Zuschnitt des **unmaskierten** Originals, vorhandene Resize-Regel auf 160 × 160).

In 4F-B darf die Validation-Expertenmaske **keinesfalls** die vorhergesagte ROI-Box bestimmen. Der Vertrag `roi-expert-mask-v1` (Inhalts-SHA-256 `a0c72744776e92a4…`) bleibt unverändert; ebenso bleiben Split, Manifest und alle früheren Checkpoints und Ergebnisse unverändert.

## Ergebnisse

Alle Werte stammen aus **einem** Trainingslauf (Seed 7); sie sagen nichts über die Streuung zwischen Trainingsläufen aus. Die Validation-Werte sind explorativ (siehe oben). Mehrere Bilder eines Falls sind abhängig; wo Fälle gezählt werden, steht die Fallzahl neben der Bildzahl.

### Lauf

| Größe | Wert |
|---|---|
| Epochen (Limit 40) | 33, Stopp: Early Stopping (Geduld 7) |
| beste Epoche (kleinster Validation-Loss) | 26 |
| bester Validation-Loss | 0,2243 |
| Loss der besten Epoche, Train | 0,1388 |
| Dice (weich, Kurve, Modellraster) der besten Epoche, Train / Validation | 0,9018 / 0,8485 |
| Dice (binär 0,5, Kurve, Modellraster) der besten Epoche, Train / Validation | 0,9123 / 0,8597 |
| Batchgröße | 8 (geplant 8; die GPU-Probe zeigte keinen Speicherfehler, der Rückfall auf 4 war nicht nötig) |
| Trainingszeit / Gesamtlaufzeit des Workers | 802 s / 841 s (Messung dieses Laufs, keine allgemeine Schätzung) |
| Zeit je Epoche (Median, Minimum, Maximum) | 24,1 s, 23,6 s, 31,7 s |
| GPU-Speicherspitze (TensorFlow-Allokator) Probe / Training | 1763 MiB / 1512 MiB (Beobachtung) |
| Checkpoint | `models/segmentation/unet_seed7/unet_256_seed7_best.keras` (lokal, nicht versioniert), SHA-256 `b2b16c3de9034318…`, wiederhergestelltes und neu geladenes Modell: größte Abweichung der Validation-Wahrscheinlichkeiten 0.0e+00, zweites Laden 0.0e+00 |
| Validation-Loss unabhängig nachgerechnet | gespeichert 0,22432, aus den Vorhersagen mit NumPy 0,22432 (Toleranz 0,001) |

Der Verlauf ist in `reports/figures/segmentation_learning_curve.png` dargestellt (aus der History gezeichnet). Der Validation-Loss fiel bis etwa Epoche 5 steil ab und schwankt danach zwischen den Epochen: von Epoche 16 bis 33 liegt er zwischen 0,2243 und 0,2712, das Minimum liegt bei Epoche 26. Die genaue Wahl des Checkpoints ist deshalb zum Teil Rauschen der Epochen; Early Stopping griff nach sieben Epochen ohne neues Minimum. Die Trainingskurven (mit Spiegelung und BatchNorm-Batchstatistik) sind mit den Inferenzwerten der Validation nicht unmittelbar vergleichbar.

### Segmentierungsqualität (Pixel-Dice und IoU je Bild)

Auf der **Originalbildfläche** (Vorhersage nach der Rückabbildung, ohne Padding, gegen die Expertenmaske der Originalgröße; Schwelle 0,5), Mittel, Median, Stichproben-SD (ddof = 1) und Quantile über die Bilder; zusätzlich das diagnostische 256er-Raster (mit Padding):

| Teilmenge | Raster | Maß | Mittel | Median | SD | Min. | 5 % | 25 % | 75 % | 95 % | Max. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation (385 Bilder, 213 Fälle) | Originalbildfläche | Dice | 0,8601 | 0,9133 | 0,1762 | 0,0000 | 0,5830 | 0,8645 | 0,9468 | 0,9686 | 0,9810 |
| Validation (385 Bilder, 213 Fälle) | Originalbildfläche | IoU | 0,7832 | 0,8405 | 0,1905 | 0,0000 | 0,4114 | 0,7613 | 0,8990 | 0,9391 | 0,9627 |
| Validation (385 Bilder, 213 Fälle) | 256er-Raster (mit Padding) | Dice | 0,8597 | 0,9125 | 0,1761 | 0,0000 | 0,5831 | 0,8633 | 0,9468 | 0,9677 | 0,9801 |
| Validation (385 Bilder, 213 Fälle) | 256er-Raster (mit Padding) | IoU | 0,7826 | 0,8390 | 0,1904 | 0,0000 | 0,4115 | 0,7594 | 0,8990 | 0,9373 | 0,9610 |
| Train (1.114 Bilder, 639 Fälle) | Originalbildfläche | Dice | 0,9171 | 0,9345 | 0,0669 | 0,0379 | 0,8115 | 0,9024 | 0,9535 | 0,9692 | 0,9834 |
| Train (1.114 Bilder, 639 Fälle) | Originalbildfläche | IoU | 0,8524 | 0,8771 | 0,0932 | 0,0193 | 0,6827 | 0,8221 | 0,9111 | 0,9403 | 0,9673 |
| Train (1.114 Bilder, 639 Fälle) | 256er-Raster (mit Padding) | Dice | 0,9168 | 0,9346 | 0,0666 | 0,0453 | 0,8117 | 0,9028 | 0,9532 | 0,9690 | 0,9830 |
| Train (1.114 Bilder, 639 Fälle) | 256er-Raster (mit Padding) | IoU | 0,8519 | 0,8772 | 0,0930 | 0,0232 | 0,6831 | 0,8228 | 0,9105 | 0,9398 | 0,9665 |

- **Train gegen Validation:** der mittlere Dice liegt auf den Trainingsbildern 0,057 höher als auf den Validation-Bildern (0,917 gegen 0,860); der Median unterscheidet sich weniger (0,935 gegen 0,913). Auf der Validation haben 83,6 % der Bilder einen Dice von mindestens 0,8 und 57,1 % mindestens 0,9.
- **Ein Mittelwert, ein langer Schwanz:** der Mittelwert (0,860) liegt unter dem Median (0,913), weil wenige Bilder sehr schlecht segmentiert werden (siehe die Verteilung in `reports/figures/segmentation_dice_distribution.png` und die Fehleranalyse unten). Die Maxima (Validation 0,981) und die Minima (Validation 0,000, Train 0,038) zeigen die Spannweite.
- **Das 256er-Raster ist nicht der Engpass:** auf dem Modellraster ist der mittlere Dice mit 0,8597 praktisch gleich dem der Originalbildfläche (0,8601). Die **Rundreise der Expertenmaske** (auf das Raster abbilden und zurück; die Rasterungsreferenz) liegt bei:

| Teilmenge | Mittel | Median | Minimum | 25 % |
|---|---:|---:|---:|---:|
| Validation | 0,9917 | 0,9923 | 0,9645 | 0,9898 |
| Train | 0,9917 | 0,9925 | 0,9699 | 0,9894 |

  Der Rasterungsverlust beträgt also etwa einen Prozentpunkt Dice. Die Rundreise ist eine Referenz, keine strenge Obergrenze (ein Netz könnte die Originalmaske besser treffen als ihre rückgebildete Rastervariante); die Abweichungen der Vorhersage stammen jedenfalls nicht in erster Linie aus der Rasterung.

### Leere Vorhersagen, mehrere Komponenten, große Vorhersagen

Nenner sind die Bilder und Fälle der Teilmenge; die Box-Merkmale beziehen sich auf die Bilder mit nicht leerer Vorhersage.

| Merkmal | Validation | Train |
|---|---|---|
| leere vorhergesagte Maske | 6 von 385 (1,6 %; 5 von 213 Fällen) | 0 von 1.114 (0,0 %; 0 von 639 Fällen) |
| mehr als eine Komponente (8er-Nachbarschaft) | 70 von 385 (18,2 %; 62 von 213 Fällen) | 111 von 1.114 (10,0 %; 92 von 639 Fällen) |
| vorhergesagte Maske über 50 % der Bildfläche | 0 von 385 (0,0 %; 0 von 213 Fällen) | 2 von 1.114 (0,2 %; 1 von 639 Fällen) |
| Dice unter 0,5 | 16 von 385 (4,2 %; 12 von 213 Fällen) | 2 von 1.114 (0,2 %; 2 von 639 Fällen) |
| Dice unter 0,25 | 13 von 385 (3,4 %; 10 von 213 Fällen) | 2 von 1.114 (0,2 %; 2 von 639 Fällen) |
| Box ohne Überlappung mit der Expertenbox | 1 von 379 (0,3 %; 1 von 212 Fällen) | 0 von 1.114 (0,0 %; 0 von 639 Fällen) |
| Boxfläche unter der Hälfte der Expertenbox | 4 von 379 (1,1 %; 4 von 212 Fällen) | 3 von 1.114 (0,3 %; 3 von 639 Fällen) |
| Boxfläche über dem Doppelten der Expertenbox | 24 von 379 (6,3 %; 22 von 212 Fällen) | 19 von 1.114 (1,7 %; 18 von 639 Fällen) |
| unter 90 % der Expertenmaske in der vorhergesagten Box | 40 von 379 (10,6 %; 35 von 212 Fällen) | 76 von 1.114 (6,8 %; 67 von 639 Fällen) |

- **Leere Vorhersagen** (Validation): 6 Bilder aus 5 Fällen (`bus_0522-l`, `bus_0522-r`, `bus_0971-l`, `bus_1013-l`, `bus_1014-l`, `bus_1041-r`); in den Trainingsbildern gibt es keine. Zwei weitere Validation-Bilder haben eine nicht leere Vorhersage ohne Überlappung mit der Expertenmaske (`bus_0743-s`, `bus_0971-r`; Dice 0).
- **Mehrere Komponenten:** 70 von 385 Validation-Bildern (18,2 %) und 111 von 1.114 Trainingsbildern (10,0 %) haben eine Vorhersage aus mehr als einer 8er-zusammenhängenden Komponente (Expertenmasken haben in allen Trainingspaaren genau eine, siehe 4D). Für den registrierten Weg zu 4F-B (nur die größte Komponente) ist das relevant: die Boxmaße unten beziehen sich auf **alle** vorhergesagten Pixel, nicht auf die größte Komponente.
- **Über 50 % der Bildfläche:** in der Validation keines; bei den Trainingsbildern `bus_0666-l`, `bus_0666-r` (die Expertenmasken dieser Bilder bedecken selbst mehr als die Hälfte des Bildes, siehe 4D).

### Lage und Größe der vorhergesagten Box gegenüber der Expertenmaske

Die Box umfasst alle positiven Pixel der Vorhersage (halboffen, Originalkoordinaten) und wird mit der Box um alle Pixel der Expertenmaske verglichen.

Nenner: Validation 379 von 385 Bildern mit nicht leerer Vorhersage.

| Maß (Validation) | Mittel | Median | 25 % … 75 % | Median Train |
|---|---:|---:|---|---:|
| Box-IoU | 0,788 | 0,846 | 0,755 … 0,918 | 0,887 |
| Mittelpunktversatz horizontal (Einheiten der Expertenbox-Breite, vorhergesagt minus Experte) | -0,006 | -0,006 | -0,030 … 0,012 | -0,008 |
| Mittelpunktversatz vertikal (Einheiten der Expertenbox-Höhe; positiv = weiter unten) | 0,047 | 0,000 | -0,016 … 0,024 | 0,000 |
| Mittelpunktabstand relativ zur Diagonale der Expertenbox | 0,073 | 0,027 | 0,014 … 0,058 | 0,020 |
| Breitenverhältnis vorhergesagt / Experte | 0,986 | 0,969 | 0,908 … 1,019 | 0,975 |
| Höhenverhältnis vorhergesagt / Experte | 1,118 | 0,989 | 0,944 … 1,049 | 0,983 |
| Flächenverhältnis der Boxen vorhergesagt / Experte | 1,162 | 0,964 | 0,872 … 1,071 | 0,958 |
| Anteil der Expertenmaske innerhalb der vorhergesagten Box | 0,949 | 0,982 | 0,949 … 0,995 | 0,987 |

Der Median zeigt eine gut zentrierte und etwa gleich große Box (Box-IoU 0,846 im Median); der Mittelwert der Flächenverhältnisse liegt wegen einzelner stark vergrößerter Boxen (mehrere Komponenten, siehe oben: Boxfläche über dem Doppelten bei 24 Bildern) höher als der Median. Bei 40 von 379 Validation-Bildern mit Vorhersage (10,6 %) liegen weniger als 90 % der Expertenmaske in der vorhergesagten Box. Die Frage, wie viel der Läsion eine um 20 % erweiterte Box der **größten Komponente** erfasst, ist Gegenstand von 4F-B und hier nicht untersucht.

### Gruppen (deskriptiv, mit Nennern)

Klasse und Gerät, jeweils Bilder / Fälle. Das sind beschreibende Zahlen; sie belegen keine demografische Fairness, die Gruppen unterscheiden sich in vielen Merkmalen zugleich (Gerät, Klassenanteil, Maskengröße, eingebrannte Marker), und Gruppen unter zehn Bildern werden nicht interpretiert.

| Teilmenge | Art | Gruppe | Bilder / Fälle | Dice Mittel | Dice Median | Dice Min. | IoU Mittel | leere Vorhersagen | Hinweis |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Validation | gesamt | alle | 385 / 213 | 0,8601 | 0,9133 | 0,0000 | 0,7832 | 6 |  |
| Validation | Klasse | benigne | 260 / 144 | 0,8814 | 0,9276 | 0,0000 | 0,8128 | 4 |  |
| Validation | Klasse | maligne | 125 / 69 | 0,8156 | 0,8909 | 0,0000 | 0,7218 | 2 |  |
| Validation | Gerät | GE Logiq 5 @10-12MHz | 142 / 85 | 0,9002 | 0,9369 | 0,0000 | 0,8371 | 2 |  |
| Validation | Gerät | GE Logiq 7 @10-14MHz | 206 / 109 | 0,8620 | 0,9027 | 0,0000 | 0,7788 | 1 |  |
| Validation | Gerät | Toshiba Aplio 300 @12-14 MHz | 31 / 16 | 0,6817 | 0,7998 | 0,0000 | 0,5921 | 3 |  |
| Validation | Gerät | U-Systems @10-14MHz | 6 / 3 | 0,7664 | 0,7580 | 0,5635 | 0,6489 | 0 | weniger als zehn Bilder: nicht interpretiert |
| Train | gesamt | alle | 1.114 / 639 | 0,9171 | 0,9345 | 0,0379 | 0,8524 | 0 |  |
| Train | Klasse | benigne | 754 / 434 | 0,9249 | 0,9402 | 0,0379 | 0,8654 | 0 |  |
| Train | Klasse | maligne | 360 / 205 | 0,9007 | 0,9208 | 0,5361 | 0,8253 | 0 |  |
| Train | Gerät | GE Logiq 5 @10-12MHz | 522 / 315 | 0,9268 | 0,9389 | 0,1419 | 0,8676 | 0 |  |
| Train | Gerät | GE Logiq 7 @10-14MHz | 504 / 277 | 0,9111 | 0,9310 | 0,0379 | 0,8433 | 0 |  |
| Train | Gerät | Toshiba Aplio 300 @12-14 MHz | 75 / 39 | 0,8855 | 0,9071 | 0,5643 | 0,8034 | 0 |  |
| Train | Gerät | U-Systems @10-14MHz | 13 / 8 | 0,9360 | 0,9364 | 0,8629 | 0,8808 | 0 |  |

Beobachtung (beschreibend): Auf der Validation ist der mittlere Dice bei malignen Läsionen niedriger als bei benignen (0,816 gegen 0,881; 125 / 69 gegen 260 / 144 Bilder / Fälle) und beim Gerät Toshiba Aplio 300 am niedrigsten (0,682, 31 Bilder aus 16 Fällen; Dice unter 0,5 bei 7 von 31 Bildern, gegen 3 von 142 beim GE Logiq 5 und 6 von 206 beim GE Logiq 7). Die Toshiba-Gruppe ist klein (31 Validation-Bilder; 75 von 1.114 Trainingsbildern); ob Gerät, Bildmerkmale oder Zufall die Ursache sind, ist nicht untersucht.

### Fehleranalyse (nicht für Änderungen verwendet)

Die zwölf Validation-Bilder mit dem kleinsten Dice (Auswahlregel im Plan; ändert weder Schwelle noch Modell). Sie stammen aus 9 Fällen: Bilder desselben Falls scheitern teils gemeinsam (z. B. beide Seiten von Fall 522, 971 und 1014).

| Rang | ID | Klasse | Gerät | Dice | IoU | leere Vorhersage | Komponenten | Anteil der Expertenmaske an der Bildfläche |
|---:|---|---|---|---:|---:|---|---:|---:|
| 1 | `bus_0522-l` | benign | GE Logiq 5 @10-12MHz | 0,0000 | 0,0000 | ja | 0 | 3,9 % |
| 2 | `bus_0522-r` | benign | GE Logiq 5 @10-12MHz | 0,0000 | 0,0000 | ja | 0 | 4,5 % |
| 3 | `bus_0743-s` | benign | GE Logiq 5 @10-12MHz | 0,0000 | 0,0000 | nein | 4 | 2,1 % |
| 4 | `bus_0971-l` | benign | GE Logiq 7 @10-14MHz | 0,0000 | 0,0000 | ja | 0 | 0,8 % |
| 5 | `bus_0971-r` | benign | GE Logiq 7 @10-14MHz | 0,0000 | 0,0000 | nein | 3 | 0,7 % |
| 6 | `bus_1013-l` | malignant | Toshiba Aplio 300 @12-14 MHz | 0,0000 | 0,0000 | ja | 0 | 33,4 % |
| 7 | `bus_1014-l` | malignant | Toshiba Aplio 300 @12-14 MHz | 0,0000 | 0,0000 | ja | 0 | 12,0 % |
| 8 | `bus_1041-r` | benign | Toshiba Aplio 300 @12-14 MHz | 0,0000 | 0,0000 | ja | 0 | 0,9 % |
| 9 | `bus_1014-r` | malignant | Toshiba Aplio 300 @12-14 MHz | 0,0118 | 0,0060 | nein | 7 | 9,8 % |
| 10 | `bus_0287-l` | malignant | GE Logiq 7 @10-14MHz | 0,1617 | 0,0880 | nein | 1 | 15,5 % |
| 11 | `bus_1053-s` | benign | Toshiba Aplio 300 @12-14 MHz | 0,2035 | 0,1133 | nein | 4 | 18,7 % |
| 12 | `bus_0232-r` | benign | GE Logiq 7 @10-14MHz | 0,2351 | 0,1332 | nein | 10 | 1,4 % |

Warum diese Bilder scheitern, ist hier **nicht** beurteilt: die Sichtung der lokalen Fehleranalyse-Vorschau steht aus. Die Tabelle nennt nur IDs und Zahlen.

## Vorschau und Sichtprüfung

Zwei lokale Kontaktübersichten (je Beispiel: Originalbild, Expertenmaske, vorhergesagte Maske, Überlagerung mit beiden Konturen; Rückabbildung auf die Originalpixel). Sie werden vom Skript geschrieben, **nicht geöffnet**, nicht übertragen und nicht versioniert; die Sichtprüfung erfolgt lokal und war zum Zeitpunkt dieses Berichts offen (Nachtrag unten).

| Datei | SHA-256 | Inhalt |
|---|---|---|
| `data/processed/segmentation_preview_representative.png` | `76f352254d15f84fe47d5467625db1b7be347015d8c5be78dfe1eeb129c1fdf6` | je Teilmenge (Train, Validation) acht repräsentative Beispiele: je Gerät und Klasse das Bild, dessen Maskenanteil dem Median der Gruppe am nächsten liegt (Regel ohne Blick auf die Modellausgabe) |
| `data/processed/segmentation_preview_error_analysis_validation.png` | `4fc700c12625836b5a8bc81033ac8e24ed5de19a2151c92c73c6ff96bd24b9f8` | gesondert als **Fehleranalyse** gekennzeichnet: die zwölf Validation-Bilder mit dem kleinsten Dice |

IDs der repräsentativen Auswahl: train: `bus_0496-l`; train: `bus_0428-r`; train: `bus_0311-r`; train: `bus_0250-r`; train: `bus_1051-r`; train: `bus_1062-r`; train: `bus_0851-s`; train: `bus_0854-s`; validation: `bus_0636-l`; validation: `bus_0754-s`; validation: `bus_0169-l`; validation: `bus_0314-r`; validation: `bus_1030-l`; validation: `bus_1014-r`; validation: `bus_0850-l`; validation: `bus_0857-l`. IDs der Fehleranalyse: `bus_0522-l`, `bus_0522-r`, `bus_0743-s`, `bus_0971-l`, `bus_0971-r`, `bus_1013-l`, `bus_1014-l`, `bus_1041-r`, `bus_1014-r`, `bus_0287-l`, `bus_1053-s`, `bus_0232-r`. Die Auswahltabelle (mit den Dice-Werten) ist `reports/tables/segmentation/unet_seed7/preview_selection.csv`.

Zu beurteilen: ob die vorhergesagten Masken auf der sichtbaren Läsion liegen; typische Fehler in der Fehleranalyse (verfehlte Läsion, falsche Struktur, zerfallene oder verschmolzene Masken, Schatten, eingebrannte Marker); ob die Masken als Lokalisation für den späteren Schritt 4F-B brauchbar wirken. Die Sichtprüfung war erst nach dem Lauf möglich und ist unten mit Datum nachgetragen; sie ist keine Prüfung aller Vorhersagen.

### Nachtrag vom 20.10.2026: Sichtprüfung der beiden Vorschauen und Terminologie

**Sichtprüfung.** Die beiden lokalen Vorschauen wurden nach dem 4F-A-Lauf lokal gesichtet; vor dem Lauf hatte keine Sichtung stattgefunden, und die Auswahl der IDs war regelbasiert (siehe oben), nicht nach Ansicht. Die Dateien auf der Platte entsprechen den oben genannten SHA-256-Werten; ein eigener Hash der gesichteten Dateien wurde bei der Sichtung nicht protokolliert. **Befund (sachlich):** Die repräsentativen Beispiele sind meist passend. In der Fehleranalyse sind echte leere und falsch lokalisierte Vorhersagen zu sehen. Das ist **keine fachärztliche Prüfung**, bestätigt keine medizinische Richtigkeit der Masken und gilt nur für die in den Vorschauen gelisteten IDs. Die Zahlen der Fehleranalyse (sechs leere Vorhersagen bei 385 Validation-Bildern; zwei weitere Bilder mit nicht leerer Vorhersage ohne Überlappung) stimmen mit diesem Befund überein; im Schritt 4F-B wurden die Zuschnitte daraus zusätzlich gegen die Expertenmasken diagnostiziert ([auto_roi_classification.md](auto_roi_classification.md)).

**Terminologie.** Die „Rundreise der Expertenmaske“ (Maske auf das 256er-Raster abbilden und zurück) ist eine **Rasterungsreferenz**, keine strenge mathematische Obergrenze des erreichbaren Dice: Sie gibt an, wie viel Dice allein die Rasterung kostet, schließt aber nicht aus, dass eine Vorhersage die Originalmaske besser trifft. Wo dieser Bericht oben von einer „Obergrenze“ oder „Auflösungsgrenze“ sprach, ist das entsprechend gemeint; der unveränderlich registrierte Studienplan und die gespeicherten Skriptausgaben von 4F-A behalten ihre ursprüngliche Formulierung („resolution ceiling“) unverändert.

## Prüfungen

Gezielt geprüft (82 Prüfungen, alle wie erwartet; kein zusätzliches Training außer dem Lauf selbst):

- **Geometrie (synthetisch und an den 1.499 echten Größen):** eingepasste Größe und Versätze gleich der unabhängigen Neuherleitung der Pillow-Regel, dem echten Ausgabebild von `resize_with_padding` und dem Träger der transformierten Maske; Bild und Maske auf denselben Pixeln (nicht quadratische Bilder, kleine Masken, Masken am Bildrand); eine Maske, die auf dem Raster verschwindet, ist ein Fehler.
- **Rückabbildung:** Koordinaten Original ↔ Modell exakt umkehrbar; synthetische Masken (Rundreise: Dice über 0,95, Schwerpunkt, Randkontakt bleibt erhalten); Vorhersagen im Padding werden verworfen; ein Fleck an der Modellposition eines Originalpunkts landet dort.
- **Synchrone Augmentation:** gleiche Entscheidung auf Bild und Maske, andere Proben unverändert, Eingaben nicht verändert; **in einem echten `model.fit`** folgen die Trainingsbatches exakt dem geplanten Strom aus (Seed, Epoche): Reihenfolge, eine Spiegelentscheidung je Bild, Bild und Maske gemeinsam gespiegelt, letzter kleinerer Batch erhalten; die Validation ist unverändert und nicht augmentiert.
- **Loss und Metriken:** der Keras-Loss gleich dem NumPy-Zwilling (größte Abweichung 1,2e-07); die Dice-Glättung wie definiert; Kurvenmetriken gleich den Pixelmaßen der Auswertung; nullwertige Nenner sind undefiniert (leer), nicht 0.
- **GPU und Reset:** Parameterzahl, Ausgabeform, Probeschritt auf GPU:0 mit endlichem Loss und einem Gradienten je trainierbarem Gewicht; zwei Konstruktionen mit dem Studienseed geben identische Anfangsgewichte; der Reset stellt sie einschließlich BatchNorm-Statistiken bitgleich wieder her, Adam ist neu.
- **Checkpoint:** speichern und mit `compile=False` laden gibt identische Gewichte und Vorhersagen; ein zweites Laden reproduziert sie exakt; BatchNorm nutzt bei der Vorhersage seine gleitenden Statistiken (kein Einfluss anderer Bilder im Batch).
- **Zugriffssperre und Schranken in isolierten Kopien:** ein geänderter ROI-Vertrag, ein geänderter Split oder ein veränderter 4E-Vorgänger sowie der falsche Interpreter stoppen vor dem Plan und vor jeder PNG-Datei; die Datenprüfung ohne registrierten Plan öffnet keine Datei; eine Maske im Modus L, anderer Größe oder leer stoppt die Datenprüfung vor dem Einfrieren und nennt die ID; normaler Ablauf Plan → Datenprüfung → Worker; ein geänderter Plan stoppt. Beim echten Lauf öffnete jeder Prozess genau die 2.998 Train-/Validation-Dateien (aufgezeichnet).
- **Sichere Wiederaufnahme:** ein fertiger Lauf wird ohne Worker-Start validiert; ein manipuliertes Ergebnis oder ein geänderter Plan stoppt; ein Lauf mit Trainingsartefakten ohne Status ist `recovery_required` (kein Worker, nichts überschrieben, der Befehl zur Wiederaufnahme wird genannt); ein direkt gestarteter Worker wird vor jedem Import abgewiesen; die **Auswertungswiederaufnahme** aus dem gespeicherten Checkpoint (GPU, ohne Training) beendet einen Lauf ohne vollständige Tabellen, lässt History und Checkpoint byte-identisch und reproduziert den Dice je Bild; für fertige oder leere Verzeichnisse wird sie verweigert.
  Beim Funktionstest vor dem Lauf (in einer isolierten Kopie, auf zwei Epochen gekürzt, **ohne** berichtete Werte) schlug eine eigene Zusatzprüfung fehl (Vergleich der Vorhersagen bei verschiedenen Batchgrößen: TF32-Rundung auf der GPU, nicht BatchNorm); der entstandene `recovery_required`-Zustand wurde sofort zum Test der Auswertungswiederaufnahme genutzt. Die Prüfung wurde danach auf Batches gleicher Größe mit verschiedenen anderen Bildern umgestellt (das ist die eigentliche Frage nach der BatchNorm-Statistik).
- **Echtlauf (unabhängige Nachrechnung):** Reihenfolge Plan vor Datenprüfung vor Training, Probe, Reset, Batchgröße, Fingerabdrücke, Parameterzahlen; für 40 Bilder (24 Validation, 12 Train, die vier mit dem kleinsten Dice) der Dice auf der Originalbildfläche mit eigener Vorverarbeitung, eigener Rückabbildung und eigenem Dice aus dem gespeicherten Checkpoint nachgerechnet; Tabellen, Kennzahlen, Gruppen, Box-Zusammenfassung und Auswahlregeln aus der Einzelbildtabelle nachgerechnet; alle Ausgabe-Hashes; die Plot-Skript-Abbildungen byte-identisch ohne TensorFlow.
- Rohdaten, Manifest, Split, der Vertrag `roi-expert-mask-v1`, 4B bis 4E (Checkpoints, Vorhersagen, Tabellen, Studien), alle früheren Modelle sind per Hash unverändert (951 Dateien vor und nach); `py_compile` und `pip check` ohne Befund; keine Installation.

## Grenzen

- **Ein Lauf, ein Seed:** keine Aussage zur Streuung des Trainings und keine zu anderen Architekturen, Schwellen oder Augmentationen. Es gab keine Hyperparametersuche; ein anderes Ergebnis mit anderem Seed ist möglich.
- **Validation ist nicht unabhängig:** sie steuert Early Stopping und Checkpoint-Wahl und liefert die Qualitätsmessung; die Werte sind wahrscheinlich etwas optimistisch. Die Testdaten wurden nicht verwendet.
- **Rasterung und Interpolation:** das 256er-Raster und die bilineare Rückabbildung begrenzen den Dice nur gering (Rundreise, siehe oben); die Binärschwelle 0,5 ist fest und nicht optimiert.
- **Abhängige Bilder:** mehrere Bilder eines Falls sind nicht unabhängig; Fallzahlen stehen neben den Bildzahlen; keine Aussage über Patientinnen.
- **Gruppen:** deskriptiv, mit Nennern; kleine Gruppen (U-Systems, sechs Validation-Bilder) werden nicht interpretiert; keine Fairnessaussage.
- **Eingebrannte Texte und Messmarker** sind Teil der Bilder; das Netz kann sie nutzen; das ist nicht untersucht (kein OCR, kein Inpainting).
- **Sigmoid-Ausgabe** ist keine kalibrierte klinische Wahrscheinlichkeit. GPU-Training ist nicht bitgenau reproduzierbar; die Checkpoint-Datei ist lokal (nicht versioniert) und wird nicht als byte-gleich zwischen Läufen erwartet.
- **Keine Klassifikation:** dieser Schritt liefert keinen Klassifikationsscore, keine ROI-Klassifikation und keine Modell- oder Schwellenwahl. Die Sichtprüfung war zum Zeitpunkt des Berichts offen und ist im Nachtrag dokumentiert.

## Dateien und Reproduktion

- Skript und Module: `analysis/15_segmentation_unet.py` (Interpreterprüfung, Vorbedingungen, Plan, Datenprüfung, Koordinator, Worker mit Probe, Reset, Training und Auswertung, Auswertungswiederaufnahme), `analysis/plot_segmentation_results.py` (Statistikfiguren aus Tabellen, CPU, kein TensorFlow), `src/busbra/segmentation.py` (Geometrie, Rückabbildung, Metriken, Tabellen, Plan; ohne TensorFlow), `src/busbra/segmentation_model.py` (U-Net, Loss, Metriken, Trainingsstrom, Callbacks; Keras).
- Aufruf (aus dem Projektordner, der Interpreter wird geprüft): `env -u LD_LIBRARY_PATH .venv/bin/python analysis/15_segmentation_unet.py`; Auswertungswiederaufnahme: `env -u LD_LIBRARY_PATH .venv/bin/python analysis/15_segmentation_unet.py --resume-evaluation`. Gemessene Laufzeit dieses Laufs: Worker 841 s (davon Training 802 s) auf der RTX 3050 Laptop; das ist eine Messung eines Laufs.
- Ergebnisse: Plan `reports/tables/segmentation_study_config.json`, Datenprüfung `segmentation_data_check.json`, Studienstatus `segmentation_status.json` (+ `_last_success`), Geometrietabelle `segmentation_geometry_train_validation.csv` (**nur lokal**), im Lauf-Ordner `reports/tables/segmentation/unet_seed7/`: `history.csv`, `config.json`, `status.json`, `metrics_summary.csv`, `metrics_by_group.csv`, `quality_flags.csv`, `box_deviation_summary.csv`, `dice_histogram.csv`, `preview_selection.csv`, `per_image_train_validation.csv` (**nur lokal**), `run.log`; Checkpoint `models/segmentation/unet_seed7/unet_256_seed7_best.keras` (SHA-256 `b2b16c3de9034318cdcd52fb103a30dacf607bf4e2fb722ad6f0fb963e171947`, **nur lokal**) und `model_summary.txt`; Abbildungen `reports/figures/segmentation_learning_curve.png`, `segmentation_dice_distribution.png`; lokale Vorschauen in `data/processed/`.
- Der Studienstatus verweist auf alle Ausgabe-Hashes; ein wiederholter Aufruf validiert den fertigen Lauf ohne Worker-Start und reproduziert die Abbildungen byte-identisch.
