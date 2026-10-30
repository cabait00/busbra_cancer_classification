# Schritt 5B: Grad-CAM und FGSM technisch vorbereiten (vier Validation-Bilder)

> Hinweis: Das technische Vorbereitungsskript dieses Schritts (`18_prepare_gradcam_fgsm.py`) und seine Nachweisdateien gehören zum historischen Ablauf und liegen nicht mehr im Projektordner; die Methoden stehen in `src/busbra/model_diagnostics.py`.

Stand: 21.10.2026. Schritt 5B ist umgesetzt und **technisch geprüft**; die Sichtprüfung der hochgeladenen Präsentationen ist abgeschlossen (Nachtrag im Abschnitt „Lokale Vorschauen und Sichtprüfung“). Er prüft die Erklärung (Grad-CAM) und den digitalen Stresstest (FGSM) technisch an einer kleinen, festen Validation-Stichprobe. **Es ist keine Leistungsstudie:** aus vier Bildern folgt weder eine AUC noch eine Rangfolge der Bedingungen noch eine repräsentative Aussage zur Modellaufmerksamkeit oder zur Robustheit. Ein technisch bestandenes Ergebnis ist keine klinische Freigabe. Die Hauptinterpretation und die FGSM-Auswertung über fünf Seeds folgen in der festgelegten Abschlussphase auf Test (Protokoll: [final_evaluation_protocol.md](final_evaluation_protocol.md)); die Testdaten sind weiterhin gesperrt und wurden nicht berührt.

Forschungsfrage des Projekts: *Verbessert die Eingrenzung des Läsionsbereichs die Unterscheidung gutartiger und bösartiger Befunde im Brustultraschall?* Verglichen werden Vollbild (4B), Experten-ROI (4E) und automatische ROI (4F-B). Der automatische Zweig verwendet denselben auf Experten-ROIs trainierten Klassifikator wie 4E; ein leerer Segmentierungsausgang führt zum Vollbild mit diesem ROI-Klassifikator, nicht zum 4B-Modell.

## Umfang und Abgrenzung

Freigegeben und ausgeführt: Inferenz und Gradientenberechnung mit den zwei vorhandenen Klassifikatoren (Seed 7), das Lesen von 16 Validation-Bildern und der vier Expertenmasken der Zielbilder, lokale Vorschauen. **Nicht** ausgeführt: Training, Optimierungsschritt, erneute oder erstmalige U-Net-Inferenz (das U-Net wird nicht geladen), neue ROI-Regeln oder Schwellen, jeder Zugriff auf Testbilder, Testmasken oder Testvorhersagen. Die Validation-Ergebnisse der Pipelines sind bekannt (Entwicklungsbefunde). Plan, Quellen, Toleranzen und Code wurden **vor dem ersten Öffnen eines Bildes** registriert (`reports/tables/diagnostics_precheck_plan.json`, Analyse `prec5b-b8d9ce9f`, Plan-SHA-256 `b8d9ce9f64448b0a9e6d0c1959fc03d72ae58c7084b5af5617c5c996f2fb790a`); vorherige Funktionstests verwendeten ausschließlich synthetische Daten.

## Stichprobe

Zielbilder: je Klasse die lexikografisch ersten Validation-IDs aus verschiedenen Cases (global Case-distinkt); entschieden haben ausschließlich ID, Case und Label, nie Score, Fehlerkategorie, Segmentierungsqualität oder eine erwartete Heatmap.

| Zielbild | Case | Klasse | Position in der Validation-Reihenfolge |
|---|---|---|---:|
| bus_0008-l | 8 | benigne | 0 |
| bus_0019-l | 19 | benigne | 2 |
| bus_0021-l | 21 | maligne | 4 |
| bus_0032-l | 32 | maligne | 6 |

**Batch-Kontext (nur numerisch, ohne Erklärung, Störung oder Vorschau):** Die damaligen Auswertungen verarbeiteten die Validation in Blöcken zu 16 Bildern in aufsteigender ID-Reihenfolge. Alle vier Zielbilder liegen im ersten Block; zum numerisch vergleichbaren Aufruf wurden die zwölf übrigen Bilder dieses Blocks geladen: bus_0008-r, bus_0019-r, bus_0021-r, bus_0032-r, bus_0034-l, bus_0034-r, bus_0039-l, bus_0039-r, bus_0046-l, bus_0046-r, bus_0053-l, bus_0053-r. Insgesamt wurden damit 16 verschiedene Validation-Bilder gelesen (höchstens 64 waren zugelassen); der Teilbatch am Ende der Validation (ein Bild) war nicht betroffen. Expertenmasken wurden nur für die vier Zielbilder geöffnet.

## Modelle, Eingaben und Geometrie

| Bedingung | Klassifikator (Seed 7) | SHA-256 |
|---|---|---|
| Vollbild (4B) | `models/augmentation160/seed_7/mobilenetv2_higher_lr_160_aug/…_seed7_best.keras` | `adad4bc0560a4762a6fc8cfb68ebe87862768b0d5d4cebaa3ac88a6f1bdbe02b` |
| Experten-ROI (4E) und automatische ROI (4F-B) | `models/expert_roi/seed_7/mobilenetv2_higher_lr_160_roi/…_seed7_best.keras` (derselbe Klassifikator) | `eada8248abd3f5fa5e1b89006737e86766cda5e4ad7c90b3fcefdbfc773ebefd` |

Die Dateihashes stimmen mit den Vorgaben und mit dem 5A-Quellenregister überein; 5A-Plan (`final5a-937da596`), Erfolgsstatus und die sieben aufgezeichneten Ausgaben wurden anhand der Hashes geprüft. Externe Eingabe: NHWC (N, 160, 160, 1), Graustufen, float32 in [0, 1], bisherige Resize-/Padding-Regel; die drei Kanäle und die Abbildung auf [−1, 1] entstehen im Modell, es gibt keine zusätzliche Standardisierung. Alle Zuschnitte entstehen aus den **unmaskierten Originalpixeln**; dekodierte Bildgrößen steuern die Geometrie (die vier bekannten abweichenden CSV-Höhen werden nie verwendet).

- **Experten-ROI:** Box aus der geprüften Geometrietabelle von 4E; für die vier Zielbilder zusätzlich nach dem Vertrag `roi-expert-mask-v1` aus der Expertenmaske abgeleitet, in allen vier Fällen identisch.
- **Automatische ROI:** ausschließlich die in 4F-B eingefrorenen Koordinaten (`auto_roi_frozen.json`, `auto_roi_geometry_validation.csv`, Hashes anhand der historischen 4F-B-Ergebnisregistrierung geprüft). Keine neue Maskenschätzung, keine Neuwahl einer Komponente; eine Expertenmaske bestimmt oder korrigiert die automatische Box nie. In den 16 gelesenen Bildern gibt es keinen Vollbild-Fallback.

| Bedingung | Bild | Label | Fallback | Bild (B × H) | Box (x0, y0, x1, y1) | Zuschnitt (B × H) | eingepasst (B × H) | Padding oben/unten, links/rechts | Quelle der Box |
|---|---|---|---|---|---|---|---|---|---|
| Vollbild (4B) | bus_0008-l | benigne | – | 275 × 297 | (0, 0, 275, 297) | 275 × 297 | 148 × 160 | 0/0, 6/6 | ganzes Bild (4B-Aufbereitung) |
| Vollbild (4B) | bus_0019-l | benigne | – | 276 × 372 | (0, 0, 276, 372) | 276 × 372 | 119 × 160 | 0/0, 20/21 | ganzes Bild (4B-Aufbereitung) |
| Vollbild (4B) | bus_0021-l | maligne | – | 281 × 346 | (0, 0, 281, 346) | 281 × 346 | 130 × 160 | 0/0, 15/15 | ganzes Bild (4B-Aufbereitung) |
| Vollbild (4B) | bus_0032-l | maligne | – | 275 × 350 | (0, 0, 275, 350) | 275 × 350 | 126 × 160 | 0/0, 17/17 | ganzes Bild (4B-Aufbereitung) |
| Experten-ROI (4E) | bus_0008-l | benigne | – | 275 × 297 | (61, 117, 193, 187) | 132 × 70 | 160 × 85 | 37/38, 0/0 | Geometrietabelle 4E (aus der Maske bestätigt) |
| Experten-ROI (4E) | bus_0019-l | benigne | – | 276 × 372 | (0, 13, 276, 181) | 276 × 168 | 160 × 97 | 31/32, 0/0 | Geometrietabelle 4E (aus der Maske bestätigt) |
| Experten-ROI (4E) | bus_0021-l | maligne | – | 281 × 346 | (70, 29, 195, 131) | 125 × 102 | 160 × 131 | 14/15, 0/0 | Geometrietabelle 4E (aus der Maske bestätigt) |
| Experten-ROI (4E) | bus_0032-l | maligne | – | 275 × 350 | (0, 70, 261, 269) | 261 × 199 | 160 × 122 | 19/19, 0/0 | Geometrietabelle 4E (aus der Maske bestätigt) |
| automatische ROI (4F-B) | bus_0008-l | benigne | nein | 275 × 297 | (55, 119, 185, 186) | 130 × 67 | 160 × 82 | 39/39, 0/0 | eingefrorene 4F-B-Koordinaten |
| automatische ROI (4F-B) | bus_0019-l | benigne | nein | 276 × 372 | (0, 13, 276, 178) | 276 × 165 | 160 × 96 | 32/32, 0/0 | eingefrorene 4F-B-Koordinaten |
| automatische ROI (4F-B) | bus_0021-l | maligne | nein | 281 × 346 | (75, 29, 211, 132) | 136 × 103 | 160 × 121 | 19/20, 0/0 | eingefrorene 4F-B-Koordinaten |
| automatische ROI (4F-B) | bus_0032-l | maligne | nein | 275 × 350 | (0, 70, 248, 264) | 248 × 194 | 160 × 125 | 17/18, 0/0 | eingefrorene 4F-B-Koordinaten |

## Numerische Treue und GPU

Vor jeder Erklärung und jeder Störung wurden die sauberen Scores mit den gespeicherten Quellen aus 4B, 4E und 4F-B verglichen: gleiche Batchgröße (16), gleiche Reihenfolge, Inferenzmodus (`training=False`), gleiche Kompilierung, absolute Toleranz 1e-5 (nicht gelockert). Größte Abweichung über alle 16 Bilder: Vollbild 3,27e-08, Experten-ROI 2,77e-08, automatische ROI 4,90e-08 — jeweils für die kompilierte Vorhersagefunktion, den Logit-Graphen und den Grad-CAM-Graphen. Es wurde keine Validation neu durchgerechnet.

Alle Modellrechnungen und Gradienten liefen nachweisbar auf `GPU:0` (die Tensoren tragen dieses Gerät; weiche Geräteplatzierung aus, Memory Growth an, kein Speicherlimit, `jit_compile=False`, TF32 wie in den akzeptierten Läufen, keine Mixed Precision; GPU-Speicherspitze rund 341 MiB). Die Dateihashes beider Checkpoints und die Fingerabdrücke aller Modellvariablen (trainierbar und nicht trainierbar) sind vor und nach dem Diagnoseschritt identisch; es gab keinen Optimierungsschritt, nichts wurde gespeichert. Der Lauf dauerte 17,7 s.

**Dokumentierte numerische Beobachtung (vor der Registrierung mit synthetischem Rauschen diagnostiziert):** Der Vorwärtspfad innerhalb des Eingabe-Gradientengraphen weicht unter TF32 von der Inferenzfunktion ab (bei synthetischem Rauschen bis zu 6e-4 in der Wahrscheinlichkeit mit TF32 gegenüber 3e-8 ohne TF32; im Lauf bis zu 4,07e-03 im Logit). Deshalb stammen **alle berichteten Scores und Losswerte aus dem Inferenzpfad**; der Gradientengraph liefert nur das Vorzeichen des Eingabegradienten, seine Vorwärtsdifferenz ist als Diagnose je Zeile gespeichert (kein Prüfkriterium). Die Grad-CAM-Funktion (Aktivierung, Logit) ist davon nicht betroffen und reproduziert die gespeicherten Scores.

## Grad-CAM

Methode (Kursanker Tag 12, projektspezifisch angepasst): Ziel ist der **Malignant-Logit vor der Sigmoid-Funktion** (ein einzelner Sigmoid-Ausgang: kein Klassenvektor, kein `argmax`). Er wird über einen separaten, gewichtsgleichen Diagnosegraphen zugänglich gemacht, der die Schichtobjekte des geladenen Modells wiederverwendet (Logit = gepoolte Merkmale · Kernel + Bias der vorhandenen Ausgabeschicht); das geladene Modell wird nicht verändert. Aktivierung und Logit entstehen im **selben verbundenen Rechenweg** durch den verschachtelten MobileNetV2-Backbone (ein Gradient `None` oder nicht endlich wäre ein Fehler). Zielschicht: die tatsächlich geladene Schicht `mobilenetv2_backbone/Conv_1` (abschließende 1×1-Faltung vor BatchNorm und ReLU), ausgelesene Aktivierungsform **5 × 5 × 1280**. Gradient des Logits nach der Aktivierung, Kanalgewichte je Bild als Mittel über die räumlichen Achsen, gewichtete Summe der Kanäle, ReLU. Bei 160 Pixeln Eingabe ist die Karte grob (5 × 5); es wird keine Pixelgenauigkeit behauptet und keine andere Schicht gesucht. Sigmoid(Logit) reproduziert in allen 12 Fällen den gespeicherten Score (größte Abweichung unter 3e-8).

**Rückabbildung:** 5 × 5-Karte → 160 × 160-Eingabefläche (bilinear) → exakt das Resize-Padding entfernen → tatsächliche Zuschnittgröße → Einsetzen in die eingefrorene Box im Originalbild. **Außerhalb der ROI ist keine Erklärung berechnet** (die Vorschau blendet den Bereich aus; er ist nicht „unwichtig = 0“). Jede nichtleere Karte wird mit ihrem eigenen positiven Maximum normiert (Rohmaximum und Leerstatus stehen in der Tabelle); die Farben individuell normierter Karten sind kein absoluter Vergleich der Merkmalswichtigkeit zwischen Bedingungen oder Modellen. Expertenkonturen erscheinen nur als Referenz; es gibt keinen Dice-Vergleich von Grad-CAM und Expertenmaske.

| Bedingung | Bild | Label | Score (gespeichert) | Logit | Rohmaximum der Karte | positive Zellen (von 25) | Box / Zuschnitt (B × H) | Anteil der Bildfläche berechnet |
|---|---|---|---:|---:|---:|---:|---|---:|
| Vollbild (4B) | bus_0008-l | benigne | 0,303 | -0,83 | 6,18e-02 | 15 | 275 × 297 | 100,0 % |
| Vollbild (4B) | bus_0019-l | benigne | 0,048 | -2,99 | 3,69e-01 | 15 | 276 × 372 | 100,0 % |
| Vollbild (4B) | bus_0021-l | maligne | 0,458 | -0,17 | 1,66e-01 | 18 | 281 × 346 | 100,0 % |
| Vollbild (4B) | bus_0032-l | maligne | 0,624 | 0,51 | 8,48e-02 | 23 | 275 × 350 | 100,0 % |
| Experten-ROI (4E) | bus_0008-l | benigne | 0,009 | -4,68 | 1,28e-01 | 8 | 132 × 70 | 11,3 % |
| Experten-ROI (4E) | bus_0019-l | benigne | 0,142 | -1,80 | 5,33e-02 | 13 | 276 × 168 | 45,2 % |
| Experten-ROI (4E) | bus_0021-l | maligne | 0,543 | 0,17 | 3,73e-02 | 7 | 125 × 102 | 13,1 % |
| Experten-ROI (4E) | bus_0032-l | maligne | 0,756 | 1,13 | 1,15e-01 | 12 | 261 × 199 | 54,0 % |
| automatische ROI (4F-B) | bus_0008-l | benigne | 0,003 | -5,79 | 1,70e-01 | 8 | 130 × 67 | 10,7 % |
| automatische ROI (4F-B) | bus_0019-l | benigne | 0,136 | -1,85 | 5,64e-02 | 14 | 276 × 165 | 44,4 % |
| automatische ROI (4F-B) | bus_0021-l | maligne | 0,125 | -1,94 | 1,06e-01 | 8 | 136 × 103 | 14,4 % |
| automatische ROI (4F-B) | bus_0032-l | maligne | 0,723 | 0,96 | 9,34e-02 | 12 | 248 × 194 | 50,0 % |

Befunde: keine leere Karte nach ReLU, kein Bild mit vollständig verschwindendem Gradienten, alle Karten innerhalb der Box endlich und außerhalb exakt „nicht berechnet“. Die Zahl positiver Zellen und das Rohmaximum zeigen, dass die Karten grob und ungleich stark ausgeprägt sind; mehr wird aus vier Bildern nicht abgeleitet.

## FGSM

Fester Angriff auf den externen float32-Eingang in [0, 1] (Kursanker Tag 19 / 20_01, projektspezifisch angepasst): `x_adv = clip(x + ε · sign(∇ₓ BCE(y_true, Modell(x))), 0, 1)` mit ε = 0, 1/255, 2/255, 4/255, echten binären Labels, ungezielt, **ein Schritt** je Stärke vom ungestörten Eingang aus (keine Iteration, kein Verketten), ohne Quantisierung auf uint8. Die BCE wird stabil aus dem Logit berechnet (`sigmoid_cross_entropy_with_logits`); die Äquivalenz zu `BinaryCrossentropy(from_logits=False)` auf der Sigmoid-Ausgabe wurde je Zeile geprüft (Sigmoid wird nicht doppelt angewendet). Das künstliche Resize-Padding bleibt mit einer **geometrischen Gültigkeitsmaske** exakt konstant; sie folgt der Bildgeometrie, nie der Läsionsmaske, echte dunkle Pixel im Bild sind reguläre Bildpixel. Nur Vollbild und automatische ROI; im automatischen Zweig wird der festgelegte Klassifikatoreingang angegriffen, **nicht** die Segmentierungs- und Zuschnittskette (U-Net und Zuschnittsentscheidung bleiben unverändert). Gleiche ε-Werte entsprechen wegen der Zuschnitte nicht derselben Änderung in Originalbildkoordinaten.

Geprüft je Zeile: endliche Eingabegradienten, unverändertes Padding, Wertebereich [0, 1], größte tatsächliche Änderung höchstens ε + 1e-7, ε = 0 lässt das Eingabearray unverändert und reproduziert den sauberen Score; beide Scorepfade stimmen auf 1e-5 überein.

| Bedingung | Bild | Label | Score sauber | ε = 1/255 | ε = 2/255 | ε = 4/255 | BCE sauber → 4/255 | Entscheidung bei 0,5: sauber → 1 / 2 / 4 (/255) |
|---|---|---|---:|---:|---:|---:|---|---|
| Vollbild (4B) | bus_0008-l | benigne | 0,3030 | 0,9400 | 0,9607 | 0,9544 | 0,361 → 3,088 | negativ → positiv / positiv / positiv |
| Vollbild (4B) | bus_0019-l | benigne | 0,0478 | 0,6981 | 0,7086 | 0,7102 | 0,049 → 1,239 | negativ → positiv / positiv / positiv |
| Vollbild (4B) | bus_0021-l | maligne | 0,4585 | 0,0166 | 0,0206 | 0,0328 | 0,780 → 3,419 | negativ → negativ / negativ / negativ |
| Vollbild (4B) | bus_0032-l | maligne | 0,6242 | 0,0255 | 0,0048 | 0,0040 | 0,471 → 5,517 | positiv → negativ / negativ / negativ |
| automatische ROI (4F-B) | bus_0008-l | benigne | 0,0030 | 0,0865 | 0,1252 | 0,1061 | 0,003 → 0,112 | negativ → negativ / negativ / negativ |
| automatische ROI (4F-B) | bus_0019-l | benigne | 0,1359 | 0,6436 | 0,7225 | 0,6416 | 0,146 → 1,026 | negativ → positiv / positiv / positiv |
| automatische ROI (4F-B) | bus_0021-l | maligne | 0,1252 | 0,0010 | 0,0011 | 0,0037 | 2,077 → 5,604 | negativ → negativ / negativ / negativ |
| automatische ROI (4F-B) | bus_0032-l | maligne | 0,7231 | 0,1895 | 0,1298 | 0,0762 | 0,324 → 2,575 | positiv → negativ / negativ / negativ |

Befunde (offen berichtet): Der Loss stieg in allen 24 gestörten Zeilen; das war kein Prüfkriterium und wird bei echten Netzen nicht erzwungen. Die Entscheidung unter der Referenzschwelle 0,5 wechselte in 15 der 24 gestörten Zeilen, unter der eingefrorenen Bildschwelle für 90 % Sensitivität in 17 und für 95 % in 13 (Schwellen aus 5A, nicht neu bestimmt; die Zeilen stehen in `diagnostics_precheck_fgsm.csv`). Es gab keinen endlichen Nullgradienten. Diese kleine Stichprobe begründet keine Aussage zur Robustheit einer Bedingung, keine Case-Sensitivität und keinen neuen Betriebspunkt.

## Technische Prüfungen

- Funktionstests mit synthetischen Daten (69 Prüfungen): Geometrie (Padding entfernen, Rückabbildung mit Markierungen, ungerader Pixelrest, Randboxen, außerhalb der ROI „nicht berechnet“), Grad-CAM an einem verschachtelten synthetischen Modell mit analytischer Referenz (Aktivierung, Logit, Gradient, ReLU, Normierung, Leerfall, Ablehnung einer fehlenden Zielschicht und eines unverbundenen Graphen), FGSM (Vorzeichen, echtes Label, Lossrichtung, ε = 0, Einzelschritt, Padding, ε-Grenze, Finite-Differenzen-Vergleich).
- Probe des gesamten Ablaufs mit synthetischen Bildern und den zwei Checkpoints (16 Prüfungen); Vorbedingungs-, Wiederaufruf- und Schutzfälle in isolierten Kopien (30 Prüfungen: falscher Interpreter, geänderte Modell- oder 5A-Quelle, Test-ID in der Zielliste, geänderter Code unter der Registrierung, fehlender oder unlesbarer Status, beschädigte oder verwaiste Ausgaben; ein vollständiger Wiederaufruf prüft schreibgeschützt ohne Bild, GPU oder neue Dateien).
- Unabhängige Zugriffskontrolle des echten Laufs: genau die 16 registrierten Validation-Bilder und die vier registrierten Zielmasken wurden gelesen, kein weiteres PNG außer den eigenen Vorschauen (die zum Hashen zurückgelesen werden); Modelldateien: nur die zwei registrierten Klassifikatoren; kein U-Net, keine Train-, Test- oder Vorschaudatei als Eingabe.

## Lokale Vorschauen und Sichtprüfung

Die beiden Vorschauen bleiben im Projekt (nicht versioniert, nicht weitergeben):

| Datei | SHA-256 |
|---|---|
| `data/processed/gradcam_precheck_validation.png` | `a3d25c9f12e26a4f1f0b7c13e663deb950bb01e8611722c84d680a072fe28b7d` |
| `data/processed/fgsm_precheck_validation.png` | `7909c4b1ae38b9431a0e1e72c472aeefb5d31b066ad095ab1cf8f984be933fce` |

**Sichtprüfung bei Abschluss des Schritts (überholt durch den Nachtrag unten):** Die echten Vorschauen wurden **nicht geöffnet**; die Sichtung durch den Projektinhaber stand **offen**. Nur das Layout der Darstellung (Beschriftung, Ausblendung außerhalb der ROI, unverändertes Padding) wurde an Vorschauen aus synthetischem Rauschen geprüft, die keine Ultraschalldaten enthalten; die geometrische Korrektheit wurde nicht durch Ansehen, sondern durch die Tests oben belegt. Eine fachliche Beurteilung der Heatmaps liegt nicht vor und ist hier nicht Ziel.

### Nachtrag vom 21.10.2026: Sichtprüfung der hochgeladenen Präsentationen

Die beiden Präsentationen (Grad-CAM und FGSM) wurden nach dem Abschluss von Schritt 5B angesehen. Die Beschreibung betrifft ausschließlich diese vier Beispielbilder und die hochgeladenen Dateien; sie ist keine fachärztliche Prüfung und keine Leistungsaussage.

- **Darstellung:** Grün ist die Expertenkontur, Cyan die tatsächlich verwendete Box. Die Rückabbildung der Heatmap in das Originalbild ist optisch plausibel (Lage der Karte innerhalb der Box, kein erkennbarer Versatz).
- **bus_0032-l:** Die Aktivierung liegt im Bereich der Läsion.
- **bus_0021-l:** Der Score bei Experten-ROI und bei automatischer ROI unterscheidet sich deutlich (0,543 gegenüber 0,125), obwohl die sichtbare Läsionskontur in beiden Boxen liegt. Die Ursache ist aus dieser Sichtung nicht ableitbar; ein einzelnes Bild begründet keine Aussage zu einer Bedingung.
- **Vollbildkarten:** Teilweise zeigen sie starke Randaktivierungen in der Nähe der eingebrannten Angaben. Das ist kein Nachweis einer Shortcut-Nutzung und kein Gegenbeweis; die Karten sind grob (5 × 5 Zellen), und der Datensatz enthält eingebrannte Texte und Messmarker (siehe Grenzen).
- **Einordnung der Karten:** Sie erklären den Malignant-Logit und sind je Karte individuell normiert; Farben sind kein absoluter Vergleich zwischen Bildern, Bedingungen oder Modellen.
- **FGSM-Darstellung:** Technisch plausibel. Aus vier Bildern folgt keine belastbare Rangfolge der Bedingungen.

**Hashabweichung der Dateien:** Die hochgeladenen Dateien haben andere SHA-256-Werte als die beim Lauf protokollierten lokalen Originalvorschauen.

| Vorschau | SHA-256 der lokalen Originaldatei (unverändert gleich dem 5B-Status) | SHA-256 der hochgeladenen Datei |
|---|---|---|
| Grad-CAM | `a3d25c9f12e26a4f1f0b7c13e663deb950bb01e8611722c84d680a072fe28b7d` | `765fae16799a0f02bee017cc62d736f74890f42971c89bc584269c1d737e5ed8` |
| FGSM | `7909c4b1ae38b9431a0e1e72c472aeefb5d31b066ad095ab1cf8f984be933fce` | `4a345e8a9341dad64f7ae68845ba06767ae95aeafb7ba7bea34240168808eeb8` |

Die lokalen Originaldateien wurden am 21.10.2026 nur gehasht (nicht geöffnet, nicht neu erzeugt); ihre Werte entsprechen den Einträgen des unveränderten 5B-Status. Die Originalhashes wurden nicht ersetzt. Die Ursache der Abweichung ist nicht geklärt; die Pixelidentität der angesehenen Dateien mit den lokalen Originalen ist deshalb nicht belegt, und die Sichtprüfung gilt für die hochgeladenen Dateien.

## Abweichungen und Grenzen

- Der Vorwärtspfad im Gradientengraphen weicht unter TF32 numerisch vom Inferenzpfad ab (siehe oben); berichtet wird daher aus dem Inferenzpfad. Eine inhaltliche Änderung des 5A-Testprotokolls ist dafür nicht nötig (Epsilon-Werte, Seed, Zielschicht und Zielgröße sind unverändert).
- Vier Bilder aus vier Cases sind ein technischer Test, keine repräsentative Stichprobe; ein Seed; die Karte ist wegen der 5 × 5-Aktivierung grob.
- Fehlversuche vor dem produktiven Lauf: die Probe mit synthetischen Bildern deckte einen Fehler der Zugriffsprüfung auf (die eigenen Vorschauen wurden mitgezählt) und eine fehlende Umgebungsverknüpfung im Testprojekt auf; beides wurde vor der Plan-Registrierung behoben. Der produktive Lauf gelang beim ersten Versuch.

## Dateien und Wiederaufruf

| Datei | Inhalt |
|---|---|
| `analysis/18_prepare_gradcam_fgsm.py` | Interpreterprüfung, Vorbedingungen, Plan-Registrierung, Bildaufbereitung, GPU-Teil, Veröffentlichung, schreibgeschützte Prüfung (Skript-SHA-256 `1689301910383aeac3b0435b532c2e6e347f9bf727f4b894c468b518de33e5fb`) |
| `src/busbra/model_diagnostics.py` | Geometrie, Diagnosegraph, Grad-CAM, FGSM (SHA-256 `e2b95de1afaa7192754119e4a4b2ef5199ed0dd6d5657323534956a061c7652a`) |
| `reports/tables/diagnostics_precheck_plan.json` | unveränderlicher 5B-Plan |
| `reports/tables/diagnostics_precheck_sample.csv`, `_sources.csv` | Ziel- und Kontext-IDs getrennt; tatsächlich gelesene Quellen mit Hashes |
| `reports/tables/diagnostics_precheck_score_fidelity.csv`, `_geometry.csv`, `_gradcam.csv`, `_fgsm.csv` | Scoretreue, Geometrie, Grad-CAM- und FGSM-Nachweise |
| `reports/tables/diagnostics_precheck_status.json` | Erfolgsstatus mit Zugriffs-, GPU- und Gewichtsnachweis (zuletzt geschrieben) |

Aufruf: `env -u LD_LIBRARY_PATH .venv/bin/python analysis/18_prepare_gradcam_fgsm.py`. Ein erneuter Aufruf eines vollständigen Erfolgs prüft Status, Plan, alle Ausgaben und Eingaben schreibgeschützt (Exit 0, keine neue Datei, keine GPU). Ergebnisse ohne gültigen Erfolgsstatus werden als `recovery_required` gemeldet und nie ersetzt; unter einer bestehenden Registrierung rechnet nur unveränderter Code; ein Fehlerbericht nennt die Dateien, die dieser Versuch tatsächlich angelegt hat.
