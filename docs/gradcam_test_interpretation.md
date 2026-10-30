# Schritt 5D: Grad-CAM und qualitative Fehleranalyse am eingefrorenen Test

Stand: 21.10.2026. Schritt 5D ist umgesetzt, ausgeführt. Er untersucht an 11 nach festen Regeln ausgewählten Testbildern (7 Kernbilder, 4 Lokalisationssonderfälle) aus 11 verschiedenen Cases, wo die groben Grad-CAM-Karten der drei Eingabebedingungen (Vollbild, Experten-ROI, automatische ROI) liegen. Verwendet werden ausschließlich Inferenz und Gradienten der zwei ursprünglichen Seed-7-Klassifikatoren (4B feinjustiert, 4E feinjustiert); nichts wird trainiert, ausgewählt oder neu eingestellt, und die Testergebnisse von Schritt 5C bleiben unverändert. Die FGSM-Hauptauswertung und die Geräteaufschlüsselung (5E) sind nicht Teil dieses Schritts. Die Karten beweisen weder eine Ursache noch eine klinisch richtige Merkmalsnutzung.

## Kurzfassung

- **Registrierung:** Studie `gcam5d-83832257`, Plan-SHA-256 `838322578282820bbdaa43eee3bedb9bc316ac8158fbe3e577f2137f25a59d12`, registriert am 2026-10-21 15:52:43, nach der Hashprüfung aller Quellen und der Auswahl aus gespeicherten Tabellen und **vor dem ersten produktiven Testbildzugriff dieses Schritts und vor jeder Heatmap** (nach den separat dokumentierten Numerikproben, die auch reale Validation-Bilder verwendeten). Die Auswahlregel stammt aus dem vor dem Test eingefrorenen Abschlussprotokoll; die konkreten IDs wurden bei bekannten Testergebnissen, aber vor Kenntnis der Heatmaps festgelegt. Eine rückwirkende Registrierung vor der Modellentwicklung oder vor den Testergebnissen wird nicht behauptet.
- **Auswahl:** 7 Kernbilder (je höchstens zwei pro Kategorie TP, TN, FP, FN des automatischen Zweigs am eingefrorenen 0,95-Schwellenpunkt, höchstens ein Bild je Case) und 4 Sonderfälle; die Kernauswahl stimmt mit der vorab festgelegten Liste überein (bus_0009-l, bus_0012-r, bus_0007-l, bus_0033-l, bus_0020-l, bus_0024-l, bus_0810-s).
- **Technische Prüfung:** 216 Bildscores in 5 ursprünglichen Batchblöcken (Längen 16, 16, 16, 16, 8) werden mit dem Predict-Pfad, dem Logit-Graphen und dem Grad-CAM-Graphen auf höchstens 5,76e-08 an die gespeicherten 5C-Scores reproduziert (Toleranz 1e-5, nicht gelockert). 33 Karten, GPU belegt (Spitze 117 MiB), Gewichte und Checkpoint-Dateien vor und nach dem Lauf unverändert.
- **Karteneigenschaften:** 0 leere Karten nach ReLU, 0 Nullgradienten; Zielschicht `mobilenetv2_backbone/Conv_1`, Aktivierungsform 5x5x1280; außerhalb der ROI ist die Karte als nicht berechnet gespeichert (NaN), nie als Null.
- **Entscheidungen:** An den jeweils eigenen eingefrorenen 0,95-Schwellen kommen die drei Bedingungen bei 9 von 11 Beispielen zu verschiedenen Ergebniskategorien (TP / FP / TN / FN); die Kategorie eines Beispiels bezieht sich auf den automatischen Zweig.
- **Zugriffe:** Berechnung 72 Bilder (Ziele und Batchkontext), 0 Masken; Darstellung 11 Zielbilder und 11 Zielmasken (Referenzkontur) sowie die eingefrorenen 5C-Pixelvorhersagen als Referenz.
- **Sichtprüfung:** offen. Die lokalen Vorschauen wurden vom ausführenden Prozess nicht angesehen; belegt sind nur Tabellen und technische Eigenschaften, sichtbare Beobachtungen gibt es noch nicht (Abschnitt „Sichtprüfung (offen)").

## Frage, Einordnung und Registrierung

Forschungsfrage des Projekts: *Verbessert die Eingrenzung des Läsionsbereichs die Unterscheidung gutartiger und bösartiger Befunde im Brustultraschall?* Schritt 5D beantwortet sie nicht statistisch (das leistet 5C), sondern untersucht an wenigen Beispielen, **wo** die Karten der Klassifikatoren liegen und welche Rolle Lokalisation, Ausschnitt und sichtbare Artefakte spielen könnten. Die Beispiele illustrieren Fehler und schätzen keine Häufigkeiten; Häufigkeiten stammen aus Schritt 5C.

Der technische Plan (`reports/tables/gradcam_test_plan.json`) enthält die Auswahl mit allen Scores und Schwellen, die zusätzlich gelesenen Kontextbilder, die Zugriffsregeln, die Numerik, die Methode, die Darstellung und die Hashes aller Eingaben, Skripte und Module. Er wurde vor dem ersten produktiven Testbildzugriff dieses Schritts und vor jeder Karte registriert (die vorbereitenden Numerikproben verwendeten auch reale Validation-Bilder); ein unveränderter Wiederaufruf prüft ihn nur noch, anderer Code unter derselben Registrierung rechnet nicht.

## Auswahl der Beispiele

**Kernauswahl.** Maßgeblich sind die gespeicherten 5C-Scores der automatischen Bedingung (Seed 7) und ihre eingefrorene Bildschwelle zum Validation-Ziel 0,95 (Entscheidung: Score ≥ Schwelle = maligne). Kategorien in der Reihenfolge TP, TN, FP, FN; je Kategorie aufsteigend nach Bild-ID; höchstens zwei Bilder je Kategorie und höchstens ein Bild je Case über die gesamte Auswahl; bereits verwendete Cases werden übersprungen, fehlende Bilder werden nicht durch handverlesene ersetzt. Diese Schwelle dient hier nur der festen Beispielauswahl; die Ziele 0,90 und 0,95 bleiben im statistischen Bericht gleichberechtigt.

**Sonderfälle.** Bis zu vier Lokalisationsfehler aus der gespeicherten 5C-Segmentierungsdiagnose, je Kategorie höchstens ein Bild in aufsteigender ID, weder ein Kernbild noch aus einem bereits verwendeten Case, in der festen Reihenfolge: (1) leere Segmentierung, Vollbild-Fallback; (2) tatsächlicher Zuschnitt ohne Expertenmaskenpixel; (3) gewählte größte Komponente überlappt die Expertenmaske nicht, obwohl eine andere es tut; (4) tatsächlicher Zuschnitt enthält mehr als 0, aber weniger als 90 % der Expertenmaskenpixel. Eine leere Kategorie wird als „kein zusätzliches Beispiel verfügbar" dokumentiert.

| Nr. | Art | Kategorie | Bild | Case | Label | Gerät | Batchblock / Position |
|---|---|---|---|---|---|---|---|
| 1 | Kern | TP | `bus_0009-l` | 9 | maligne | GE Logiq 7 @10-14MHz | 0 / 2 |
| 2 | Kern | TP | `bus_0012-r` | 12 | maligne | GE Logiq 7 @10-14MHz | 0 / 5 |
| 3 | Kern | TN | `bus_0007-l` | 7 | benigne | GE Logiq 7 @10-14MHz | 0 / 0 |
| 4 | Kern | TN | `bus_0033-l` | 33 | benigne | GE Logiq 7 @10-14MHz | 0 / 12 |
| 5 | Kern | FP | `bus_0020-l` | 20 | benigne | GE Logiq 7 @10-14MHz | 0 / 8 |
| 6 | Kern | FP | `bus_0024-l` | 24 | benigne | GE Logiq 7 @10-14MHz | 0 / 10 |
| 7 | Kern | FN | `bus_0810-s` | 810 | maligne | GE Logiq 5 @10-12MHz | 17 / 6 |
| 8 | Sonderfall | leere Segmentierung, Vollbild-Fallback | `bus_0870-r` | 870 | benigne | GE Logiq 7 @10-14MHz | 18 / 6 |
| 9 | Sonderfall | Zuschnitt ohne Expertenmaskenpixel | `bus_0874-l` | 874 | benigne | GE Logiq 7 @10-14MHz | 18 / 9 |
| 10 | Sonderfall | gewählte Komponente verfehlt die Läsion, andere Komponente überlappt | `bus_1047-r` | 1047 | benigne | Toshiba Aplio 300 @12-14 MHz | 23 / 5 |
| 11 | Sonderfall | Zuschnitt enthält mehr als 0 und weniger als 90 % der Expertenmaskenpixel | `bus_0060-r` | 60 | benigne | GE Logiq 7 @10-14MHz | 1 / 12 |

**Übersprungene Kandidaten und fehlende Beispiele** (vollständig in `gradcam_test_selection_log.csv`):

- Kern TP: `bus_0009-r` übersprungen: Case 9 ist bereits durch `bus_0009-l` belegt (höchstens ein Bild je Case über die gesamte Auswahl).
- Kern FP: `bus_0007-r` übersprungen: Case 7 ist bereits durch `bus_0007-l` belegt (höchstens ein Bild je Case über die gesamte Auswahl).
- Kern FP: `bus_0020-r` übersprungen: Case 20 ist bereits durch `bus_0020-l` belegt (höchstens ein Bild je Case über die gesamte Auswahl).
- Kern FN: `bus_0012-l` übersprungen: Case 12 ist bereits durch `bus_0012-r` belegt (höchstens ein Bild je Case über die gesamte Auswahl).
- Sonderfall gewählte Komponente verfehlt die Läsion, andere Komponente überlappt: `bus_0874-l` übersprungen: das Bild ist bereits Teil der Auswahl.
- Sonderfall Zuschnitt enthält mehr als 0 und weniger als 90 % der Expertenmaskenpixel: `bus_0024-l` übersprungen: das Bild ist bereits Teil der Auswahl.
- Kern FN: nur ein Teil der vorgesehenen zwei Bilder auswählbar (2 Bild(er) dieser Kategorie existieren; die übrigen sind wie oben übersprungen); kein Ersatz durch handverlesene Bilder.

Hinweis zur Auswahl: Das zweite FN-Bild `bus_0012-l` gehört nicht zum Case des ausgewählten FN-Bildes `bus_0810-s`, sondern zu Case 12 und wird übersprungen, weil dieser Case bereits durch das TP-Bild `bus_0012-r` verwendet ist (die Kategorien werden in der Reihenfolge TP, TN, FP, FN abgearbeitet). Das Ergebnis ist dasselbe wie in der vorab festgelegten Liste: sieben Kernbilder.

## Verfahren

**Bedingungen und Modelle.** Nur Seed 7 und nur die zwei ursprünglichen Checkpoints: der feinjustierte 4B-Klassifikator auf dem ganzen Bild (Bedingung Vollbild) und der feinjustierte 4E-Klassifikator, der einmal auf den Experten-ROIs und einmal auf den automatischen ROIs (eingefrorene U-Net-Koordinaten aus 5C) angewendet wird. Bei leerer Segmentierung ist die automatische ROI das ganze Bild mit demselben 4E-Klassifikator (kein Wechsel auf das 4B-Modell). Es werden weder 5B-R-Modelle noch das U-Net geladen; es gibt keine neue Segmentierung.

**Eingaben.** Originalpixel ohne Maskierung, zugeschnitten an den eingefrorenen Koordinaten (Experten-Geometrietabelle beziehungsweise eingefrorene automatische Geometrie aus 5C), dann das unveränderte Resize mit zentriertem schwarzem Padding auf 160 × 160 × 1, float32 in [0, 1]; Kanalkopie und Skalierung auf [-1, 1] geschehen im Modell. Die Berechnung liest **keine** Maske; die Klassifikatoreingaben entstehen aus Bild und Geometrietabelle.

**Ursprüngliche Batchblöcke.** Die gespeicherten 5C-Scores entstanden in geordneten 16er-Batches (die letzte Gruppe hat 8 Bilder). Weil sich unter dem verwendeten numerischen Modus Scores bei anderer Batchzusammensetzung im Bereich von 1e-3 ändern können, werden genau die Batchblöcke der Zielbilder gelesen und als dieselben geordneten Blöcke berechnet: Blöcke 0, 1, 17, 18, 23 mit 72 Bildern (61 Kontextbilder ohne Heatmap oder Beispiel).

**Treue.** Je Block und Bedingung müssen der kompilierte Predict-Pfad, der Logit-Graph und der Grad-CAM-Graph den gespeicherten 5C-Score **jedes** Bildes des Blocks mit absoluter Toleranz 1e-5 reproduzieren. Ein Verstoß stoppt den Schritt vor jeder Karte; die Toleranz wird nicht gelockert.

**Grad-CAM.** Zielschicht `mobilenetv2_backbone/Conv_1` (bei 160 px 5 × 5 × 1280), Ziel immer der Malignant-Logit vor der Sigmoidfunktion (eine einzelne Ausgabeeinheit, kein Argmax, kein Wechsel der Zielklasse). Kanalgewichte = räumliches Mittel des Logit-Gradienten, gewichtete Summe, ReLU; jede nichtleere Karte wird durch ihr eigenes positives Maximum normalisiert. Eine leere Karte oder ein Nullgradient ist ein regulärer Befund und wird gekennzeichnet; ein fehlender, unverbundener oder nicht endlicher Gradient wäre ein technischer Fehler. Die grobe Karte wird auf das 160er-Raster abgebildet (bilinear), genau das künstliche Padding wird entfernt, auf die tatsächliche Ausschnittgröße zurückskaliert und an der eingefrorenen Box ins Originalbild eingebettet. Außerhalb der ROI gilt die Karte als **nicht berechnet**; es wird dort keine Aufmerksamkeit von null suggeriert. Individuell normalisierte Farben sind keine zwischen Modellen vergleichbare Wichtigkeit. Es wird kein Dice und keine andere Leistungskennzahl einer Karte berechnet.

**Darstellung.** Ein eigenes CPU-Skript zeichnet aus den lokal gespeicherten Karten die Vorschauen: Original mit Expertenkontur (Referenz), Expertenbox, automatischer Box und den eingefrorenen U-Net-Pixeln, daneben die drei Grad-CAM-Überlagerungen auf demselben Originalbild mit Score, eigener Schwelle und Entscheidung je Bedingung. Layoutkorrekturen brauchen keine GPU.

## Technische Ergebnisse (belegt)

**Treue zu den gespeicherten 5C-Scores** (alle Bilder der gelesenen Batchblöcke, auch die Kontextbilder):

| Bedingung | Bilder | Blöcke | max. Abw. Predict | max. Abw. Logit-Graph | max. Abw. Grad-CAM-Graph |
|---|---:|---:|---:|---:|---:|
| Vollbild (4B) | 72 | 5 | 0,00e+00 | 5,76e-08 | 5,76e-08 |
| Experten-ROI (4E) | 72 | 5 | 0,00e+00 | 5,39e-08 | 5,39e-08 |
| Automatische ROI (4E-Klassifikator) | 72 | 5 | 0,00e+00 | 4,42e-08 | 4,42e-08 |

**Karteneigenschaften** (nach Bedingung; Rohmaximum vor der Normalisierung):

| Bedingung | Karten | leer (ReLU) | Nullgradient | kleinstes Rohmax. | größtes Rohmax. | Aktivierung | Schicht |
|---|---:|---:|---:|---:|---:|---|---|
| Vollbild (4B) | 11 | 0 | 0 | 5,28e-02 | 7,21e-01 | 5x5x1280 | mobilenetv2_backbone/Conv_1 |
| Experten-ROI (4E) | 11 | 0 | 0 | 1,69e-02 | 1,81e-01 | 5x5x1280 | mobilenetv2_backbone/Conv_1 |
| Automatische ROI (4E-Klassifikator) | 11 | 0 | 0 | 2,53e-02 | 1,72e-01 | 5x5x1280 | mobilenetv2_backbone/Conv_1 |

Alle Karten sind innerhalb der eingefrorenen Box endlich und außerhalb als nicht berechnet (NaN) gespeichert; die Zahl der nicht berechneten Pixel entspricht in jedem Fall Bildfläche minus Boxfläche. Aktivierungen, Logits und Gradienten liegen auf `GPU:0` (weiche Platzierung aus, Memory Growth an, `jit_compile=False`, numerischer Modus der akzeptierten Läufe); die Gewichte und die Checkpoint-Dateien sind vor und nach dem Lauf identisch (Dateihash und Fingerabdruck aller Variablen). Laufzeit des Schritts 27,0 s.

**Lage der Spitzenzelle und sichtbares Maximum.** Die grobe Karte wird vor dem Entfernen des Paddings durch ihr Maximum normalisiert. Bei 5 (Vollbild), 8 (Experten-ROI) und 6 (automatische ROI) von 11 nichtleeren Karten liegt die Mitte der Spitzenzelle im künstlichen Padding des 160 × 160-Eingangs; bei 1, 7 beziehungsweise 4 Karten liegt das Maximum der im Originalbild dargestellten Karte unter 0,9. Das ist eine Eigenschaft der groben Zellen (32 Pixel im Eingangsraster, Rezeptive Felder reichen über die Zellgrenzen) und besonders bei stark nichtquadratischen Zuschnitten häufig; es sagt nicht, dass das Netz „auf das Padding schaut", markiert aber Karten, deren Spitze im sichtbaren Bild nur abgeschwächt erscheint.

**Numerische Genauigkeit der Karten.** Die Karten entstehen im numerischen Modus der akzeptierten Läufe (TF32 an). Im synthetischen Probelauf wich ein zweiter Gradientenpfad im selben Modus um höchstens 3,1e-2 der normalisierten Kartenspitze ab, eine Neuberechnung in reinem float32 um höchstens 9,0e-2 (Rohmaximum relativ 6,2e-2). Feine Strukturen einer 5 × 5-Karte sind deshalb nicht belastbar; nur grobe Aussagen („Spitze eher oben links") tragen. Die Treue der Scores (1e-5) und die Genauigkeit der Gradientenkarten sind verschiedene Eigenschaften.

**Variation zwischen Prozessen.** Im Probelauf auf kleinen synthetischen Batches lieferten frische Prozesse für dieselben Eingaben zwei verschiedene Scorevektoren (Differenz etwa 2,3e-4; vermutlich unterschiedliche GPU-Algorithmenauswahl). Im Produktionslauf trat das nicht auf: alle 216 Bildscores stimmten auf höchstens 5,76e-08 mit den gespeicherten 5C-Scores überein. Die Toleranz von 1e-5 wurde nicht gelockert; im Probelauf wurde sie nur dort auf 1e-3 gesetzt, um die Mechanik an der Prozessvariation vorbeizutesten.

## Die Beispiele im Überblick (belegt: gespeicherte Scores und Entscheidungen)

Score → Entscheidung (Ergebniskategorie) je Bedingung an der **eigenen** eingefrorenen 0,95-Schwelle der Bedingung; die Entscheidungen können sich zwischen den Bedingungen unterscheiden. Die Kategorie des Beispiels bezieht sich auf den automatischen Zweig.

| Bild | Label | Kategorie | Vollbild (Schwelle 0,1282) | Experten-ROI (0,1082) | Automatische ROI (0,1252) |
|---|---|---|---|---|---|
| `bus_0009-l` | maligne | TP | 0,526 → maligne (TP) | 0,854 → maligne (TP) | 0,817 → maligne (TP) |
| `bus_0012-r` | maligne | TP | 0,027 → benigne (FN) | 0,170 → maligne (TP) | 0,128 → maligne (TP) |
| `bus_0007-l` | benigne | TN | 0,323 → maligne (FP) | 0,084 → benigne (TN) | 0,100 → benigne (TN) |
| `bus_0033-l` | benigne | TN | 0,156 → maligne (FP) | 0,030 → benigne (TN) | 0,016 → benigne (TN) |
| `bus_0020-l` | benigne | FP | 0,044 → benigne (TN) | 0,500 → maligne (FP) | 0,322 → maligne (FP) |
| `bus_0024-l` | benigne | FP | 0,527 → maligne (FP) | 0,659 → maligne (FP) | 0,855 → maligne (FP) |
| `bus_0810-s` | maligne | FN | 0,487 → maligne (TP) | 0,362 → maligne (TP) | 0,115 → benigne (FN) |
| `bus_0870-r` | benigne | Sonderfall | 0,499 → maligne (FP) | 0,065 → benigne (TN) | 0,634 → maligne (FP) |
| `bus_0874-l` | benigne | Sonderfall | 0,001 → benigne (TN) | 0,010 → benigne (TN) | 0,315 → maligne (FP) |
| `bus_1047-r` | benigne | Sonderfall | 0,038 → benigne (TN) | 0,176 → maligne (FP) | 0,230 → maligne (FP) |
| `bus_0060-r` | benigne | Sonderfall | 0,173 → maligne (FP) | 0,039 → benigne (TN) | 0,050 → benigne (TN) |

## Zuschnitt und Lokalisation (belegt: Boxen aus den eingefrorenen Tabellen, Abdeckung aus der 5C-Diagnose)

| Bild | Originalgröße | Experten-Box (Anteil) | Automatische Box (Anteil) | Fallback | Zuschnittskategorie (5C) | Expertenpixel im Zuschnitt | Dice der rohen Vorhersage (5C) |
|---|---|---|---|---|---|---:|---:|
| `bus_0009-l` | 276 × 352 | 204 × 151 (31,7 %) | 180 × 144 (26,7 %) | nein | covers_all_expert_pixels | 100,0 % | 0,938 |
| `bus_0012-r` | 276 × 351 | 160 × 75 (12,4 %) | 144 × 77 (11,4 %) | nein | covers_all_expert_pixels | 100,0 % | 0,888 |
| `bus_0007-l` | 274 × 351 | 102 × 60 (6,4 %) | 90 × 59 (5,5 %) | nein | covers_all_expert_pixels | 100,0 % | 0,904 |
| `bus_0033-l` | 277 × 297 | 83 × 54 (5,4 %) | 82 × 49 (4,9 %) | nein | covers_all_expert_pixels | 100,0 % | 0,912 |
| `bus_0020-l` | 275 × 373 | 192 × 136 (25,5 %) | 203 × 126 (24,9 %) | nein | covers_all_expert_pixels | 100,0 % | 0,902 |
| `bus_0024-l` | 278 × 335 | 258 × 165 (45,7 %) | 160 × 123 (21,1 %) | nein | covers_more_than_0_to_below_90_percent | 80,6 % | 0,747 |
| `bus_0810-s` | 566 × 472 | 251 × 181 (17,0 %) | 252 × 157 (14,8 %) | nein | covers_all_expert_pixels | 100,0 % | 0,928 |
| `bus_0870-r` | 287 × 315 | 61 × 32 (2,2 %) | 287 × 315 (100,0 %) | ja | fallback_full_image | 100,0 % | 0,000 |
| `bus_0874-l` | 275 × 357 | 129 × 63 (8,3 %) | 179 × 116 (21,1 %) | nein | covers_no_expert_pixel | 0,0 % | 0,424 |
| `bus_1047-r` | 276 × 304 | 146 × 63 (11,0 %) | 94 × 101 (11,3 %) | nein | covers_more_than_0_to_below_90_percent | 0,4 % | 0,094 |
| `bus_0060-r` | 276 × 371 | 194 × 76 (14,4 %) | 87 × 66 (5,6 %) | nein | covers_more_than_0_to_below_90_percent | 76,3 % | 0,524 |

Die Spalten stammen aus den eingefrorenen Geometrietabellen und der 5C-Diagnose; keine Box wurde neu abgeleitet oder aus einer Expertenmaske korrigiert. Das Dice gilt für die rohe U-Net-Vorhersage (alle vorhergesagten Pixel) und ist keine Aussage über eine Grad-CAM-Karte.

## Kartenmerkmale (belegt: technische Eigenschaften, kein Lokalisationsmaß)

| Bild | Vollbild: Rohmax.; Zellen > 0; Spitzenzelle (Zeile, Spalte) | Experten-ROI | Automatische ROI |
|---|---|---|---|
| `bus_0009-l` | 9,20e-02; 25/25; Zelle (4, 4); sichtbar 0,99 (Spitzenzelle im Padding) | 1,81e-01; 12/25; Zelle (2, 2); sichtbar 0,99 | 1,53e-01; 12/25; Zelle (2, 2); sichtbar 0,99 |
| `bus_0012-r` | 4,79e-01; 16/25; Zelle (4, 3); sichtbar 1,00 | 6,95e-02; 8/25; Zelle (4, 3); sichtbar 0,79 (Spitzenzelle im Padding) | 8,67e-02; 7/25; Zelle (4, 3); sichtbar 0,71 (Spitzenzelle im Padding) |
| `bus_0007-l` | 1,36e-01; 17/25; Zelle (3, 4); sichtbar 0,97 (Spitzenzelle im Padding) | 7,83e-02; 8/25; Zelle (4, 0); sichtbar 0,44 (Spitzenzelle im Padding) | 6,54e-02; 8/25; Zelle (4, 0); sichtbar 0,59 (Spitzenzelle im Padding) |
| `bus_0033-l` | 3,44e-01; 10/25; Zelle (3, 4); sichtbar 1,00 | 1,25e-01; 8/25; Zelle (4, 4); sichtbar 0,81 (Spitzenzelle im Padding) | 1,28e-01; 8/25; Zelle (4, 4); sichtbar 0,73 (Spitzenzelle im Padding) |
| `bus_0020-l` | 4,42e-01; 12/25; Zelle (3, 4); sichtbar 0,90 (Spitzenzelle im Padding) | 1,69e-02; 12/25; Zelle (1, 1); sichtbar 0,98 | 2,53e-02; 11/25; Zelle (3, 4); sichtbar 1,00 |
| `bus_0024-l` | 5,28e-02; 24/25; Zelle (3, 3); sichtbar 0,99 | 6,81e-02; 9/25; Zelle (2, 2); sichtbar 0,99 | 1,72e-01; 14/25; Zelle (2, 2); sichtbar 0,99 |
| `bus_0810-s` | 9,68e-02; 17/25; Zelle (3, 4); sichtbar 1,00 | 4,16e-02; 13/25; Zelle (0, 4); sichtbar 0,92 (Spitzenzelle im Padding) | 9,87e-02; 13/25; Zelle (4, 4); sichtbar 0,75 (Spitzenzelle im Padding) |
| `bus_0870-r` | 5,90e-02; 25/25; Zelle (4, 4); sichtbar 1,00 | 1,00e-01; 8/25; Zelle (4, 4); sichtbar 0,64 (Spitzenzelle im Padding) | 5,94e-02; 16/25; Zelle (1, 2); sichtbar 0,99 |
| `bus_0874-l` | 7,21e-01; 16/25; Zelle (3, 4); sichtbar 0,98 (Spitzenzelle im Padding) | 1,55e-01; 9/25; Zelle (4, 4); sichtbar 0,41 (Spitzenzelle im Padding) | 5,55e-02; 12/25; Zelle (4, 3); sichtbar 0,95 (Spitzenzelle im Padding) |
| `bus_1047-r` | 4,50e-01; 15/25; Zelle (4, 3); sichtbar 1,00 | 8,09e-02; 10/25; Zelle (4, 3); sichtbar 0,89 (Spitzenzelle im Padding) | 1,02e-01; 14/25; Zelle (3, 4); sichtbar 1,00 |
| `bus_0060-r` | 2,82e-01; 18/25; Zelle (3, 4); sichtbar 0,96 (Spitzenzelle im Padding) | 1,14e-01; 8/25; Zelle (4, 4); sichtbar 0,18 (Spitzenzelle im Padding) | 1,49e-01; 12/25; Zelle (4, 4); sichtbar 0,98 (Spitzenzelle im Padding) |

Die Spitzenzelle bezeichnet die Zelle der groben 5 × 5-Karte des Klassifikatoreingangs mit dem größten Wert; sie ist ohne Bild nicht einer anatomischen Struktur zuzuordnen. Rohmaxima verschiedener Bedingungen sind nicht vergleichbar, weil jede Karte durch ihr eigenes Maximum normalisiert wird. Die grobe Karte wird vor dem Entfernen des Paddings normalisiert; „sichtbar" nennt das Maximum der im Originalbild dargestellten Karte, das unter 1 liegt, wenn der größte Wert im entfernten Padding oder am Rand liegt (Vermerk „Spitzenzelle im Padding", wenn die Mitte der Spitzenzelle im künstlichen Padding des Eingangs liegt).

## Sichtprüfung (offen)

Die Vorschauen liegen lokal und enthalten Testbilder; sie wurden von dem ausführenden Prozess nicht geöffnet. Die gemeinsame Sichtung mit dem Projektinhaber steht aus.

| Vorschau (lokal) | Bilder (Zeilen) | SHA-256 |
|---|---|---|
| `data/processed/gradcam_test_preview_core_TP.png` | `bus_0009-l`, `bus_0012-r` | `9517bd29e5e03a80aa35a272d8ee61337828676bc22bf6a2d947419259d096b5` |
| `data/processed/gradcam_test_preview_core_TN.png` | `bus_0007-l`, `bus_0033-l` | `23fc04e41dd9d1793e3c088972e5f058ed24487a76223d53b3c8c5e8ed697edb` |
| `data/processed/gradcam_test_preview_core_FP.png` | `bus_0020-l`, `bus_0024-l` | `4c6394a0e42543f6cadd1cea1e352e2bb00647a9d5fc347d10a78bc034bfbeea` |
| `data/processed/gradcam_test_preview_core_FN.png` | `bus_0810-s` | `99874d11948f0b7208d55070693a595871750e9fd7ff02edccd05f859cf8cac8` |
| `data/processed/gradcam_test_preview_special_1.png` | `bus_0870-r`, `bus_0874-l` | `8f4e605ada040006424d904892ae9b320547b549af28d11baaed0934a3e002a2` |
| `data/processed/gradcam_test_preview_special_2.png` | `bus_1047-r`, `bus_0060-r` | `c6b75821011c600fc1431432ad27a5ae62f9b8242ee5d12517bc1cd28721e77c` |

Prüfprotokoll für die gemeinsame Sichtung (leer; wird erst nach der Sichtung ausgefüllt, nur tatsächlich gesichtete Bilder werden dokumentiert):

| Bild | Bedingung | auffälliger Bereich im Läsionsbereich? | im Kontext (Gewebe, Schatten)? | bei Messmarke, Beschriftung oder Rand? | Bemerkung |
|---|---|---|---|---|---|
| `bus_0009-l` | Vollbild (4B) |  |  |  |  |
| `bus_0009-l` | Experten-ROI (4E) |  |  |  |  |
| `bus_0009-l` | Automatische ROI (4E-Klassifikator) |  |  |  |  |
| `bus_0012-r` | Vollbild (4B) |  |  |  |  |
| `bus_0012-r` | Experten-ROI (4E) |  |  |  |  |
| `bus_0012-r` | Automatische ROI (4E-Klassifikator) |  |  |  |  |
| `bus_0007-l` | Vollbild (4B) |  |  |  |  |
| `bus_0007-l` | Experten-ROI (4E) |  |  |  |  |
| `bus_0007-l` | Automatische ROI (4E-Klassifikator) |  |  |  |  |
| `bus_0033-l` | Vollbild (4B) |  |  |  |  |
| `bus_0033-l` | Experten-ROI (4E) |  |  |  |  |
| `bus_0033-l` | Automatische ROI (4E-Klassifikator) |  |  |  |  |
| `bus_0020-l` | Vollbild (4B) |  |  |  |  |
| `bus_0020-l` | Experten-ROI (4E) |  |  |  |  |
| `bus_0020-l` | Automatische ROI (4E-Klassifikator) |  |  |  |  |
| `bus_0024-l` | Vollbild (4B) |  |  |  |  |
| `bus_0024-l` | Experten-ROI (4E) |  |  |  |  |
| `bus_0024-l` | Automatische ROI (4E-Klassifikator) |  |  |  |  |
| `bus_0810-s` | Vollbild (4B) |  |  |  |  |
| `bus_0810-s` | Experten-ROI (4E) |  |  |  |  |
| `bus_0810-s` | Automatische ROI (4E-Klassifikator) |  |  |  |  |
| `bus_0870-r` | Vollbild (4B) |  |  |  |  |
| `bus_0870-r` | Experten-ROI (4E) |  |  |  |  |
| `bus_0870-r` | Automatische ROI (4E-Klassifikator) |  |  |  |  |
| `bus_0874-l` | Vollbild (4B) |  |  |  |  |
| `bus_0874-l` | Experten-ROI (4E) |  |  |  |  |
| `bus_0874-l` | Automatische ROI (4E-Klassifikator) |  |  |  |  |
| `bus_1047-r` | Vollbild (4B) |  |  |  |  |
| `bus_1047-r` | Experten-ROI (4E) |  |  |  |  |
| `bus_1047-r` | Automatische ROI (4E-Klassifikator) |  |  |  |  |
| `bus_0060-r` | Vollbild (4B) |  |  |  |  |
| `bus_0060-r` | Experten-ROI (4E) |  |  |  |  |
| `bus_0060-r` | Automatische ROI (4E-Klassifikator) |  |  |  |  |

Leitfragen: Liegen die auffälligen Bereiche im Läsionsbereich, im umgebenden Gewebe oder bei Messmarken und eingebranntem Text? Was verändert sich mit dem Ausschnitt (Vollbild gegen Experten-ROI gegen automatische ROI)? Scheitert bei einem Beispiel die Lokalisation (Box, Fallback), die Klassifikation oder beides? Eine plausibel aussehende Karte bei richtiger Klassifikation beweist keine fachlich richtige Merkmalsnutzung; die Karte des Klassifikators erklärt nicht das U-Net.

## Einordnung und Interpretation (vorläufig, ohne Sichtbefund)

Die folgenden Aussagen folgen ausschließlich aus den gespeicherten Zahlen. **Sichtbare Beobachtungen liegen noch nicht vor**; die Interpretation nennt mögliche Bedeutungen ohne kausalen oder klinischen Anspruch.


- **Belegt:** Bei 9 von 11 Beispielen unterscheiden sich die Ergebniskategorien der drei Bedingungen an ihren eigenen 0,95-Schwellen (`bus_0012-r`, `bus_0007-l`, `bus_0033-l`, `bus_0020-l`, `bus_0810-s`, `bus_0870-r`, `bus_0874-l`, `bus_1047-r`, `bus_0060-r`). Eingabeausschnitt und Modell verändern also die Entscheidung am selben Bild.
- **Belegt:** 1 ausgewähltes Bild hat eine leere Segmentierung und damit einen Vollbild-Fallback in der automatischen Bedingung (`bus_0870-r`); dort bewertet der auf Zuschnitten trainierte Klassifikator das ganze Bild.
- **Belegt:** Bei `bus_0874-l` enthält der tatsächliche automatische Zuschnitt keinen Expertenmaskenpixel; die automatische Bedingung bewertet dort einen Bildausschnitt ohne die annotierte Läsion. Ob das die Entscheidung verändert hat, zeigt der Vergleich der Spalten oben, nicht eine Karte.
- **Belegt:** Bei 6 Beispielen entscheiden die Experten-ROI und die automatische ROI gleich, das Vollbild aber anders (`bus_0012-r`, `bus_0007-l`, `bus_0033-l`, `bus_0020-l`, `bus_1047-r`, `bus_0060-r`); bei den übrigen Beispielen stimmen entweder alle drei überein oder die automatische ROI weicht von der Experten-ROI ab.
- **Interpretation (Hypothese):** Unterschiede zwischen den Karten derselben Bilder können vom Ausschnitt (Kontext fällt weg), von der Skalierung des Zuschnitts auf 160 × 160 und von den unterschiedlichen Trainingsdaten der Modelle kommen; die Karten allein trennen diese Erklärungen nicht.
- **Interpretation (Hypothese):** Eine Karte, deren Maximum nicht in der annotierten Läsion liegt, wäre ein Hinweis, Messmarken, Text oder Randstrukturen als mögliche Abkürzungen zu prüfen; belegt ist das erst durch Sichtung und gezielte Gegenversuche, die nicht Teil dieses Schritts sind.

## Sichtungsnachtrag vom 21.10.2026

Am 21.10.2026 wurden alle sechs lokalen Vorschauen gesichtet. Das ist eine technische und qualitative Beurteilung der vorhandenen Darstellungen, keine fachärztliche Prüfung der Masken oder Befunde. Der Status „offen" in den Abschnitten oben beschreibt den Stand zum Produktionslauf und bleibt als historischer Stand erhalten; die Registrierung, die Statusdateien und die Rechenartefakte sind unverändert. Die folgende Tabelle trennt die **sichtbare Beobachtung** (nur aus der Sichtung der Vorschauen) vom **gesicherten Kontext** (gespeicherte Scores, Entscheidungen und 5C-Diagnosen).

| Bild | Sichtbare Beobachtung | Gesicherter Kontext (Tabellen) |
|---|---|---|
| `bus_0009-l` | Im Vollbild liegt der stärkste rote Bereich unten rechts, außerhalb der grünen Expertenkontur; bei beiden ROI-Bedingungen liegt ein zentraler roter Bereich innerhalb der Kontur. | Alle drei Entscheidungen TP. |
| `bus_0012-r` | Im Vollbild dominiert der rechte untere Randbereich; auch bei den ROI-Karten liegt der sichtbare Schwerpunkt eher unten rechts als im Zentrum der Kontur. | Vollbild FN, beide ROI-Bedingungen TP an ihren jeweiligen Schwellen. |
| `bus_0007-l`, `bus_0033-l` | Wiederkehrendes Randmuster unten rechts im Vollbild; in den ROI-Karten ist innerhalb der Kontur keine ausgeprägte rote Spitze sichtbar. | Vollbild FP, beide ROI-Bedingungen TN. |
| `bus_0020-l` | Die ROI-Schwerpunkte liegen eher am Rand beziehungsweise um die Kontur. | Vollbild TN, beide ROI-Bedingungen FP; die automatische Box enthält laut 5C alle Expertenpixel (kein vollständiges Lokalisationsversagen). |
| `bus_0024-l` | Roter ROI-Schwerpunkt innerhalb der annotierten Region. | FP in allen drei Bedingungen; die automatische Box schneidet einen Teil der Kontur ab, Expertenmaskenabdeckung laut 5C rund 80,6 %. |
| `bus_0810-s` | (keine zusätzliche Beobachtung über den Kontext hinaus) | Die automatische Box enthält die gesamte Expertenmaske; dennoch Auto-ROI FN (Score 0,114871, Schwelle 0,125246), Experten-ROI und Vollbild TP. Kein vollständiges Verfehlen oder Abschneiden der Expertenmaske; Unterschiede von Box und Kontext können trotzdem eine Rolle spielen. |
| `bus_0870-r` | Leere Segmentierung und sichtbarer Vollbild-Fallback mit dem ROI-Klassifikator. | Experten-ROI TN, automatische ROI und Vollbild FP. |
| `bus_0874-l` | Die automatische Box liegt unterhalb der grünen Kontur. | Ohne Expertenpixel (5C); Auto-ROI FP, Vollbild und Experten-ROI TN. |
| `bus_1047-r` | Die ausgewählte Komponente beziehungsweise Box liegt links neben dem überwiegenden Teil der grünen Kontur. | Nur rund 0,38 % der Expertenpixel im Zuschnitt (5C); Auto-ROI und Experten-ROI FP, Vollbild TN. |
| `bus_0060-r` | Der automatische Ausschnitt erfasst nur einen Teil der Kontur. | Rund 76,3 % der Expertenpixel im Zuschnitt (5C), trotzdem TN; Experten-ROI TN, Vollbild FP. |

**Einordnung (Interpretation, keine Beobachtung):**

- Wiederkehrende Schwerpunkte am rechten unteren Vollbildrand verdienen Erwähnung als möglicher Hinweis auf Rand- oder Layoutartefakte. Die Karten beweisen keine Nutzung bestimmter Texte, Messmarker oder des Paddings.
- Die ROI-Karten zeigen nicht durchgehend einen Schwerpunkt innerhalb der Läsion. Eine passend liegende Karte garantiert keine korrekte Benign-/Malign-Entscheidung.
- Eine vollständige Maskenabdeckung garantiert keine korrekte Klassifikation; unvollständige Abdeckung führt nicht zwangsläufig zu einer falschen Entscheidung.
- Relative Farbwerte verschiedener Karten sind nicht direkt vergleichbar; die Karten haben nur 5 × 5 Zellen und eine dokumentierte numerische Empfindlichkeit. Es gibt keine kausale und keine klinische Aussage.
- Aus den Vorschauen wird kein anatomischer oder histologischer Befund neu diagnostiziert.

**Präzisierung zur Registrierung.** Die Plan-Registrierung lag vor dem ersten **produktiven** Testbildzugriff dieses Schrittes. Die separat dokumentierten vorbereitenden Numerikproben (ihr Protokoll ist nicht Teil des Repositories) verwendeten auch reale Validation-Bilder. Eine weitergehende Vorabregistrierung wird nicht behauptet; die historischen Plan- und Statusdateien sind unverändert.

**Provenienz der übermittelten FP-Vorschau.** Fünf übermittelte Vorschauen stimmen bytegleich mit dem Figurenstatus überein. Die übermittelte FP-Datei war anders kodiert (Upload-SHA-256 `86b208279c158394e4a075668a2373fdf38a303db8cb248e06c13a0b1127ec45`, Original laut Figurenstatus `4c6394a0e42543f6cadd1cea1e352e2bb00647a9d5fc347d10a78bc034bfbeea`). Es wurde belegt, dass eine reine PNG-Kodierung derselben RGBA-Pixel mit dem ursprünglichen Softwareeintrag und 105 dpi exakt die ursprünglichen Bytes und den Originalhash ergibt; die Pixel sind also gleich. Der Uploadhash ist ein datierter Übertragungsnachweis (21.10.2026); die Produktionsdatei und der Figurenstatus bleiben unverändert, es wurde keine Karte neu berechnet.

**Lesbare Fassung der FN-Vorschau.** In `gradcam_test_preview_core_FN.png` überlagern sich unten Legende und Fußtext. Aus der vorhandenen lokalen Karte wurde mit dem Layout `--readable` von `plots/plot_gradcam_test_interpretation.py` eine getrennte Fassung `data/processed/gradcam_test_preview_core_FN_readable.png` (SHA-256 `4800721ddd20c37c3f8cb7c4a9ea259f68797dc923a3a9d05f1960c0935160e0`) erzeugt; sie hat einen eigenen Darstellungsnachweis `reports/tables/gradcam_test_fn_readable_status.json` mit den Quellhashes und dem Codestand. Kein TensorFlow, kein Modell, keine neue Heatmap; die sechs ursprünglichen Vorschauen und ihr Figurenstatus sind byte-identisch. Die Layoutkorrektur wurde an synthetischen Daten geprüft, die echte Datei wurde vom ausführenden Prozess nicht angesehen.

## Grenzen

- Elf Beispiele sind eine Illustration, keine repräsentative Stichprobe; keine Häufigkeit, kein Ranking und keine Leistungskennzahl folgt aus ihnen. Häufigkeiten der Lokalisationsfehler stehen in Schritt 5C.
- Grad-CAM auf einer 5 × 5-Schicht ist sehr grob; die Karten werden bilinear auf die Bildgröße gedehnt und sind keine Segmentierung. Individuelle Normalisierung macht die Farben zwischen Karten unvergleichbar.
- Ein Seed (7) und ein Datensatz mit vier Geräten; die Experten-ROI-Bedingung benötigt zur Anwendung eine manuelle Maske, die automatische Bedingung füttert denselben Klassifikator mit automatischen Zuschnitten.
- Die Karte erklärt den Klassifikator, nicht das U-Net und nicht die klinische Bedeutung; eine Fachprüfung der Karten fand nicht statt.
- Eingebrannte Texte und Messmarker sind mögliche Abkürzungen und nicht entfernt; ob eine Karte sie nutzt, ist erst nach Sichtung beurteilbar.
- Die Auswahl beruht auf bekannten Testergebnissen (Scores und Diagnose aus 5C); sie ist deshalb gezielt und nicht zufällig. Die Testdaten wurden dadurch weder für Modell- noch für Schwellenentscheidungen verwendet.

## Abweichungen, Prüfungen und offene Punkte

- **Produktionslauf** (`env -u LD_LIBRARY_PATH .venv/bin/python analysis/10_interpretation.py`): erster Aufruf Exit 0, 15:52:42 bis 15:53:10 Uhr (Skriptlaufzeit 27,0 s), kein Fehlschlag, kein Wiederanlauf. Zweiter Aufruf: Exit 0, schreibgeschützte Verifikation (kein Bild, keine GPU, kein TensorFlow, nichts geschrieben, Code gleich der Registrierung). Darstellung (`.venv/bin/python plots/plot_gradcam_test_interpretation.py`): Exit 0, 11 Zielbilder und 11 Zielmasken gelesen, kein TensorFlow.
- **Unabhängiger Zugriffsmitschnitt des Produktionslaufs:** Beim ersten Öffnen eines Bildes existierte die Plan-Registrierung bereits (plan_existed True) und TensorFlow war noch nicht importiert (importiert False); 72 verschiedene PNG-Dateien (genau die registrierten Bilder), keine Maske, nur die zwei registrierten Checkpoints geöffnet (kein U-Net). Die Skriptauswertung des eigenen Zugriffs stimmt überein.
- **Vor der Registrierung geprüft:** 64 Funktionsprüfungen der Hilfsfunktionen mit synthetischen Daten (Auswahl, Sonderfälle und ihre strengen Konsistenzprüfungen, Batchblöcke, Vorschaugruppen, Kartenmerkmale, Rückabbildung bei Hoch- und Querformat, ungeradem Rest, extremem Seitenverhältnis, leere Karte, Nullgradient, nicht berechnete Pixel) sowie der Vergleich der Auswahl mit den gespeicherten 5C-Tabellen (nur Tabellen, keine Bilder). Zusätzlich 58 Prüfungen eines vollständigen Probelaufs mit echten Checkpoints und erfundenen Bildern in isolierten Projektkopien: acht Abbrüche vor der Registrierung (falscher Interpreter, geänderte 5C-Scores, 5A-Schwellentabelle, Manifest, 5C-Status, Checkpoint, abweichende Auswahl, fehlende Geometrie; jeweils ohne Bild, ohne TensorFlow), Hauptlauf mit unabhängiger Auswahl, unabhängigem zweitem Gradientenpfad und unabhängiger Rückabbildung, Mitschnitt der Registrierung vor dem ersten Bild, Wiederaufruf, elf Manipulationen am fertigen Zustand, Wiederanlauf unter derselben Registrierung nach einem Fehlschlag, Code-Gate (geändertes Skript oder Modul), Abbruch bei nicht reproduzierten Scores und die Darstellung. Es gab keine Funktionstests auf echten Testbildern vor der Registrierung.
- **Probelauf und Produktionscode:** Der Probelauf lief auf synthetischen Daten mit Batchgröße 6 und einer Toleranz von 1e-3 (Produktion: Batchgröße 16, Toleranz 1e-5), weil kleine synthetische Batches prozessabhängig um etwa 2,3e-4 variieren. Mehrere Fehlschläge des Probelaufs lagen im Prüfskript und in der Probe-Basis (Hash-Buchführung, Ausgabe-Liste des Status, Lesen boolescher Spalten, Annahme eines Kartenmaximums von 1); aus dem Probelauf entstanden zwei Ergänzungen des Produktionscodes vor der Registrierung: das sichtbare Kartenmaximum und die Padding-Kennzeichnung der Spitzenzelle sowie zwei Hinweise im Plan (Prozessvariation, TF32-Genauigkeit der Karten).
- **Abweichung von der Vorgabe:** Die Vorgabe nennt die zwei FN-Bilder als Bilder desselben Cases. Tatsächlich gehört `bus_0012-l` zu Case 12, der schon durch das TP-Bild `bus_0012-r` belegt ist, nicht zum Case von `bus_0810-s`; das Ergebnis (sieben Kernbilder, dieselbe Liste) ist unverändert.
- **Unveränderter Bestand:** 1.692 Dateien des Ausgangsstandes (ohne README, Projektdefinition und den privaten Status) sind nach dem Lauf byte-identisch, darunter alle 5C-Tabellen, Scores, Geometrien, die lokale Pixelvorhersage, Pläne, Modelle, Split und Schwellen; der Rohdatenbaum und `pip freeze --all` sind unverändert, keine Datei wurde entfernt; neu sind nur die Ausgaben dieses Schritts.
- **Befunde zur Numerik (siehe oben):** prozessabhängige Scoreunterschiede auf kleinen synthetischen Batches (nicht im Produktionslauf) und begrenzte Genauigkeit der Gradientenkarten unter TF32; beide ändern keine Auswahl und keine Entscheidung, begrenzen aber die Belastbarkeit feiner Kartenstrukturen.
- **Offen:** Sichtprüfung der Vorschauen durch den Projektinhaber (Abschnitt „Sichtprüfung (offen)"); FGSM-Hauptauswertung und Geräteaufschlüsselung (5E); Bericht und Vortrag.

## Dateien und Wiederaufruf

- Skript: `analysis/10_interpretation.py` (Berechnung, GPU), Modul `src/busbra/gradcam_test.py` (Auswahl, Batchblöcke, Kartenmerkmale), Darstellung `plots/plot_gradcam_test_interpretation.py` (CPU).
- Registrierung: `reports/tables/gradcam_test_plan.json`; Status: `reports/tables/gradcam_test_status.json`; Tabellen: `gradcam_test_selection.csv`, `gradcam_test_selection_log.csv`, `gradcam_test_images.csv`, `gradcam_test_sources.csv`, `gradcam_test_score_fidelity.csv`, `gradcam_test_geometry.csv`, `gradcam_test_cam.csv` (alle in `reports/tables/`); Figurenstatus: `reports/tables/gradcam_test_figures_status.json`.
- Lokal (nicht für ein Paket): `data/processed/gradcam_test_maps.npz` (Karten im Bildraster, NaN = nicht berechnet) und die Vorschauen `data/processed/gradcam_test_preview_*.png`.
- Aufruf: `env -u LD_LIBRARY_PATH .venv/bin/python analysis/10_interpretation.py`; ein fertiges Ergebnis wird nur schreibgeschützt geprüft (keine Bilder, keine GPU, nichts wird geschrieben). Ergebnisdateien ohne gültigen Status werden nicht überschrieben (`recovery_required`). Die Vorschauen werden mit `.venv/bin/python plots/plot_gradcam_test_interpretation.py` neu gezeichnet, ohne GPU.

## Quellen

- Abschlussprotokoll `docs/final_evaluation_protocol.md` und 5A-Registrierung (Auswahlregel, eingefrorene Schwellen), 5B-Vorbereitung `docs/interpretation_robustness_precheck.md` (Grad-CAM-Methode), Testauswertung `docs/test_evaluation.md` (5C).
- Selvaraju et al. (2017), Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (Methode); Kursunterlage Tag 12 (Grad-CAM).

