# Definition des Splits

Stand: 19.10.2026. Erzeugt und geprüft mit `analysis/01_prepare_data.py`.

## Verfahren

Der Split ist ein **fester Holdout auf Basis der offiziellen Folds** des Datensatzes. Verwendet wird ausschließlich die äußere Spalte `kFold` der ursprünglichen `data/raw/BUSBRA/5-fold-cv.csv`:

| Teilmenge | Äußere Folds |
|---|---|
| train | 3, 4, 5 |
| validation | 2 |
| test | 1 |

- Es wird **keine** 5-Fold-Cross-Validation durchgeführt, nicht neu zufällig aufgeteilt und kein Fold nach Modellergebnissen ausgewählt. Die Regel stand vor jeder Modellierung fest.
- Die Zuordnung ist deterministisch. Es wird kein Zufallsseed benötigt.
- Die inneren Spalten `valid_*` der Fold-Dateien werden nicht gelesen. Sie enthalten in allen 15 untersuchten Durchläufen Cases, die zugleich im inneren Training und in der inneren Validation liegen (siehe `docs/data_audit.md`).
- Aufgeteilt wird nach `Case`. Zwei Bilder desselben Case sind keine unabhängigen Patientinnen und liegen deshalb immer in derselben Teilmenge.

## Kanonische Zuordnung

`data/processed/manifest_split.csv` enthält alle Informationen des Audit-Manifests mit unveränderten Originalwerten und zusätzlich die Spalten `outer_kfold` und `split`. Sie ist nach `ID` sortiert und die einzige Zuordnungsdatei; es gibt keine parallelen Listen. Eine darin gespeicherte Maskenreferenz macht die Maske nicht zum Modellinput. Die Datei liegt im nicht versionierten Datenordner. Ihre Prüfsumme steht in `reports/tables/split_status.json`.

Der Split gilt als technisch eingefroren, wenn alle Prüfungen bestanden sind. Das Skript überschreibt eine vorhandene Zuordnung nie: Stimmt eine vorhandene Datei nicht mit dem neu berechneten Split überein, bricht es mit einer Konfliktmeldung ab.

## Beobachtete Zahlen

Alle Zahlen stammen aus den Daten (`reports/tables/split_summary.csv`) und stimmen mit den vorab festgelegten Referenzwerten überein.

| Teilmenge | Bilder | Cases | Benigne Bilder | Maligne Bilder | Benigne Cases | Maligne Cases | Maligner Anteil (Bilder) |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1.114 | 639 | 754 | 360 | 434 | 205 | 32,32 % |
| validation | 385 | 213 | 260 | 125 | 144 | 69 | 32,47 % |
| test | 376 | 212 | 254 | 122 | 144 | 68 | 32,45 % |
| gesamt | 1.875 | 1.064 | 1.268 | 607 | 722 | 342 | 32,37 % |

**Case-Überschneidungen:** train/validation 0, train/test 0, validation/test 0. Jeder Case hat genau ein Label, liegt in genau einem äußeren Fold und in genau einer Teilmenge.

## Durchgeführte Prüfungen

Die Prüfungen laufen vor dem Schreiben; jede einzelne ist mit Ergebnis in `reports/tables/split_status.json` aufgeführt.

- Der Audit-Status aus Schritt 1 ist `passed`, und die Quelldateien `bus_data.csv` und `5-fold-cv.csv` haben noch die dort erfassten Prüfsummen.
- Das Audit-Manifest stimmt zeilenweise mit den Originalmetadaten überein; das Label ist eindeutig codiert (benign 0, malignant 1).
- Die Verknüpfung mit der Fold-Datei läuft über die ID als Text, eins zu eins und mit vollständig gleicher ID-Menge. Das Label der Fold-Datei entspricht dem Manifest.
- Gültige Fold-Werte, keine fehlende Zuordnung, keine zusätzlichen oder fehlenden Zeilen.
- Alle Referenzzahlen aus der Tabelle oben werden reproduziert.
- Alle 1.875 Bildpfade verweisen auf vorhandene Dateien im Ordner `data/raw/BUSBRA/Images/`. Geprüft wird die Existenz; Validation- und Testbilder werden dafür nicht dekodiert.

Die Fehlerbedingungen wurden gezielt an einer isolierten Kopie geprüft (doppelte oder fehlende ID, Case in zwei Folds, abweichendes Label, ungültiger Fold, nicht bestandener Audit, abweichende Zählwerte, fehlende Bilddatei, Konflikt mit bestehender Zuordnung). Jeweils endet das Skript mit Fehler und lässt eine vorhandene Zuordnung unverändert.

## Methodische Sperre

**Ab dem Einfrieren sind die Testdaten für die Modellentwicklung zurückgestellt.** Sie dienen nicht der Modell-, Hyperparameter- oder Schwellenwahl, nicht dem Vergleich von Architekturen und nicht der Exploration. Validation ist für Checkpoints, Modellvergleich und optionale Schwellenentscheidungen vorgesehen. Die Testauswertung erfolgt einmal in einer vorab geplanten Abschlussphase, nachdem Konfiguration und Schwelle festgelegt sind.

Davon zu unterscheiden sind rein technische Prüfungen und Bestandszählungen: Die Integritätsprüfungen aus Schritt 1 (Lesbarkeit, Pfadzuordnung, Duplikate, Fold-Struktur) und die Zählung von Bildern, Cases und Klassen aller Teilmengen. Sie verwenden keine Modellvorhersagen, keine Leistungswerte und keine inhaltliche Betrachtung der Testbilder. Die Vorschau des Split-Vorschlags in `docs/data_audit.md` (Abschnitt 8) bleibt als historischer Stand unverändert.

Die bildbasierte EDA (`docs/train_eda.md`) verwendet ausschließlich Trainingsbilder.

## Grenzen

- Es gibt einen einzigen Holdout. Die Unsicherheit der Schätzung durch diese eine Aufteilung wird nicht quantifiziert.
- Alle Bilder stammen aus einer Einrichtung mit vier Ultraschallgeräten. Das Ergebnis erlaubt keine Aussage zu neuen Kliniken oder Geräten.
- Mehrere Bilder eines Case sind abhängig. Die Auswertung erfolgt auf Bildebene bei fallgetrenntem Split; Bild- und Case-Zahlen werden immer gemeinsam genannt.
- Die Gerätegruppen sind klein: Im Training stehen für U-Systems 13 Bilder (8 Cases) und für Toshiba 75 Bilder (39 Cases) zur Verfügung. Subgruppenaussagen bleiben deskriptiv.
