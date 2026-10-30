# Explorative Analyse des Trainingssplits

Stand: 19.10.2026. Ursprünglich mit den früheren Treibern erzeugt, heute entsprechend: `analysis/01_prepare_data.py`. Grundlage sind ausschließlich die **1.114 Trainingsbilder (639 Cases)** des eingefrorenen Splits (siehe `docs/split_definition.md`). Validation- und Testbilder wurden nicht untersucht. Alle Zahlen stammen aus den Tabellen `reports/tables/eda_train_*.csv`; die Übersichtsgrafik liegt in `reports/figures/eda_train_overview.png`.

## 1. Klassen und Bilder je Case

| Klasse | Bilder | Cases | Anteil Bilder (von 1.114) | Anteil Cases (von 639) |
|---|---:|---:|---:|---:|
| benign | 754 | 434 | 67,68 % | 67,92 % |
| malignant | 360 | 205 | 32,32 % | 32,08 % |

Von den 639 Cases haben 164 ein Bild (benign 114, malignant 50) und 475 zwei Bilder (benign 320, malignant 155). Die Klassenanteile auf Bild- und Case-Ebene sind nahezu gleich. Eine konstante Vorhersage „benign“ ergäbe im Training etwa 67,7 % Accuracy bei einer Sensitivität von 0; das ist eine rechnerische Referenz, kein Modellergebnis.

## 2. Geräte im Training

| Gerät | Bilder | Cases | Maligne Bilder | Maligne Cases | Maligner Anteil (Bilder) | Maligner Anteil (Cases) |
|---|---:|---:|---:|---:|---:|---:|
| GE Logiq 5 @10-12MHz | 522 | 315 | 155 | 88 | 29,7 % (155/522) | 27,9 % (88/315) |
| GE Logiq 7 @10-14MHz | 504 | 277 | 160 | 93 | 31,7 % (160/504) | 33,6 % (93/277) |
| Toshiba Aplio 300 @12-14 MHz | 75 | 39 | 40 | 21 | 53,3 % (40/75) | 53,8 % (21/39) |
| U-Systems @10-14MHz | 13 | 8 | 5 | 3 | 38,5 % (5/13) | 37,5 % (3/8) |

Die Klassenanteile unterscheiden sich zwischen den Geräten. Am auffälligsten ist Toshiba mit über 50 % malignen Bildern gegenüber etwa 30 % bei den beiden GE-Geräten. Das ist ein **mögliches Shortcut-Risiko**: Ein Modell könnte gerätespezifische Bildmerkmale nutzen. Die Tabelle zeigt weder, dass das passiert, noch eine Ursache der Unterschiede. Sie ist auch kein Fairnessbefund. Für Toshiba (39 Cases) und besonders U-Systems (8 Cases, 5 maligne Bilder) sind die Nenner klein; Anteile dort sind sehr unsicher.

## 3. Dekodierte Bildgrößen

Verwendet werden die tatsächlich dekodierten Größen aus dem geprüften Manifest. Die falschen CSV-Höhen werden nicht benutzt; drei der vier betroffenen Bilder liegen im Training. Es wurde kein Bild wegen seiner Größe entfernt.

| Größe | Min | 25 % | Median | 75 % | Max |
|---|---:|---:|---:|---:|---:|
| Breite (px) | 261 | 276 | 318 | 323 | 579 |
| Höhe (px) | 268 | 351 | 386 | 447 | 478 |
| Seitenverhältnis B/H | 0,67 | 0,73 | 0,75 | 0,83 | 2,11 |

Es gibt 521 verschiedene Breite-Höhe-Paare. Die Größen hängen stark vom Gerät ab: Die Mediane der Höhe betragen 450 px (GE Logiq 5), 352 px (GE Logiq 7), 305 px (Toshiba) und 405 px (U-Systems), und Toshiba-Bilder haben fast alle dieselbe Breite von 276 px bei einem Median-Seitenverhältnis von 0,91. Die Bildgröße ist damit selbst ein Gerätehinweis. Für eine spätere Bildpipeline folgt: Eine einheitliche Zielgröße muss mit einer Entscheidung über Seitenverhältnis bzw. Padding verbunden werden. Das ist nicht Teil dieses Schritts.

## 4. Bildstichprobe für die Sichtprüfung

Ausgewählt wurden nach stabiler ID-Sortierung mit Seed 42 ein Trainingsbild je Gerät und Klasse, jeweils aus verschiedenen Cases. Alle acht Kombinationen waren im Training vorhanden.

| Gerät | benign | malignant |
|---|---|---|
| GE Logiq 5 @10-12MHz | bus_0404-l (Case 404) | bus_0663-r (Case 663) |
| GE Logiq 7 @10-14MHz | bus_0317-l (Case 317) | bus_0250-l (Case 250) |
| Toshiba Aplio 300 @12-14 MHz | bus_1010-r (Case 1010) | bus_1057-r (Case 1057) |
| U-Systems @10-14MHz | bus_0851-s (Case 851) | bus_0858-l (Case 858) |

Die Bilder werden vollständig und unverändert gezeigt, ohne Maske, BBOX, Overlay, Zuschnitt oder Skalierung. Die Kontaktübersicht liegt nur lokal und ist nicht Teil des öffentlichen Bestands.

### Lokale Sichtprüfung (durchgeführt am 19.10.2026)

Die Sichtprüfung wurde am 19.10.2026 lokal durch den Projektinhaber anhand der acht oben genannten Trainingsbilder durchgeführt. Betrachtet wurde eine **Vorschau der Kontaktübersicht** (`data/processed/eda_train_contact_sheet.png`). Ein Hashvergleich dieser Vorschau mit der lokalen PNG-Datei fand nicht statt. Das Analyseskript führt selbst keine Sichtprüfung durch; es verweist im EDA-Status nur auf diesen Nachweis und prüft, ob die aktuelle Stichprobe dieselben acht IDs in derselben Reihenfolge enthält. Bei einer anderen Auswahl gilt dieser Nachweis nicht.

**Beobachtungen je Gerät:**

| Gerät | Sichtbare Zusatzelemente in den Beispielen |
|---|---|
| GE Logiq 5 @10-12MHz | Herstellerlogo, Messmarkierungen, Texte und Maßangaben |
| GE Logiq 7 @10-14MHz | Messkreuze sowie Seiten- und Untersuchungsbeschriftungen |
| Toshiba Aplio 300 @12-14 MHz | nummerierte Skalen am rechten Rand |
| U-Systems @10-14MHz | im malignen Beispiel eine größere Textbeschriftung am unteren Bildrand |

**Weitere Beobachtungen:**

- Unterschiede in Kontrast, Körnung und Schärfeeindruck sind zwischen den Beispielen sichtbar. Ihre Ursache ist aus acht Beispielen nicht zuzuordnen.
- Es wurden keine eindeutigen durchgängigen zusätzlichen schwarzen Rahmen erkannt.
- Die dunkelgrauen Flächen zwischen den Bildern sowie die Überschriften mit ID, Klasse und Gerät stammen nur von der Kontaktübersicht. Sie sind nicht Teil der Bilder und keine Trainingsinputs.

**Einordnung:** Sichtbare Zusatzmarkierungen könnten von einem Modell als Abkürzung genutzt werden. Dass diese Markierungen mit der Zielklasse korrelieren und dass ein Modell sie tatsächlich nutzt, ist nicht nachgewiesen. Die Stichprobe aus acht Bildern belegt keine Häufigkeiten. Es wurden weder alle Originalbilder geprüft noch Marker erkannt, gezählt oder entfernt.

| Feld | Stand |
|---|---|
| Datum der Sichtprüfung | 19.10.2026 |
| Betrachtetes Medium | Vorschau der Kontaktübersicht (kein Hashvergleich mit der lokalen PNG) |
| Geprüfte IDs | die acht IDs der Tabelle oben (auch in `reports/tables/eda_train_status.json`) |
| Beobachtungen je Gerät und zwischen den Klassen | siehe oben |

## Grenzen

- Beschreibende Statistik auf einem Trainingssplit; sie liefert keine Ursachen und keine Leistungsaussage.
- Unterschiede zwischen Geräten sind ein Hinweis auf mögliche Shortcuts, kein Nachweis.
- Die Bildstichprobe ersetzt keine systematische Prüfung aller Bilder; Marker wurden weder erkannt, gezählt noch entfernt.
