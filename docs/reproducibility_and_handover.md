# Reproduzierbarkeit und Dateibestand

Stand: 22.10.2026. Das Dokument beschreibt, wie die vorhandenen Ergebnisse gelesen und nachgerechnet werden, welche Eingaben nötig sind und was bei einem neuen Lauf ausdrücklich zu wählen ist. Es verspricht keine vollständige Reproduzierbarkeit des Trainings: GPU-Rechnungen sind nur im Rahmen der dokumentierten Toleranzen wiederholbar (Abschnitt 3).

## 1 Ergebnisse lesen

Jedes Skript in `analysis/` zeigt ohne Argument gespeicherte Ergebnisse und startet nichts. Wo eine Neuberechnung aus gespeicherten Scores möglich ist, wird sie im Arbeitsspeicher ausgeführt und mit der gespeicherten Tabelle verglichen (Ausgabe „identical to the stored table“); es wird nichts geschrieben.

```bash
cd <Pfad zu>/BUSBRA
.venv/bin/python analysis/01_prepare_data.py          # Split-Manifest gegen die Rohdaten-Metadaten, Datenbefunde
.venv/bin/python analysis/03_classification.py --preset single_runs  # die frühen Seed-42-Läufe (kleines CNN, MobileNetV2) mit Kennzahlen
.venv/bin/python analysis/03_classification.py --preset multiseed   # fünf Seeds, Referenzgruppen, Paarvergleiche, Vergleich mit den gespeicherten Tabellen (Presets: multiseed, resolution250, augmentation160, expert_roi)
.venv/bin/python analysis/07_operating_points.py      # eingefrorene Validation-Schwellen und Betriebspunkte der 27 Kombinationen
.venv/bin/python analysis/09_test_evaluation.py       # Testkennzahlen, Hauptvergleiche, Cluster-Bootstrap (2.000 Ziehungen), U-Net-Qualität
.venv/bin/python analysis/11_robustness_devices.py    # FGSM-Stresstest und Geräteaufschlüsselung aus den gespeicherten Angriffs-Scores
.venv/bin/python -m unittest discover -s tests         # Regressionstests (Split, Schwellen, ROI-Rundung, Bootstrap, gespeicherte Kennzahlen)
```

**Welche Standardaufrufe welche Eingaben brauchen.** (a) *Öffentliche Score-Auswertung*: `02_baselines`, `03_classification --preset …` (alle fünf Presets), `04`, `05`, `06`, `07`, `08`, `09`, `10`, `11` und `12` ohne Argument lesen nur Dateien des Repositories (`reports/tables/`, Manifest). Sie wurden in einer Kopie nur des öffentlichen Bestands ausgeführt (Exit 0, TensorFlow/Keras blockiert); Modellgewichte, Rohdaten und `.npz`-Arrays sind dafür nicht nötig. Zwei optionale Referenzdateien fehlen im öffentlichen Bestand: die historischen Bootstrap-Ziehungen (`reports/tables/test_evaluation/bootstrap_arrays.npz`; `09` meldet ausdrücklich, dass keine Gleichheit mit gespeicherten Ziehungen behauptet wird) und die Grad-CAM-Karten (`data/processed/gradcam_test_maps.npz`; nur ein Test, der dann übersprungen wird). (b) *Rohdaten-Metadatenprüfung*: `01_prepare_data` ohne Argument liest zusätzlich die Metadaten-CSV aus `data/raw/BUSBRA/` und bricht ohne Rohdaten mit `FileNotFoundError` ab. (c) *Ausdrücklich gewählte Aktionen* (Abschnitt 2, mit `--train`, `--run`, `--infer`, `--attack` usw.): brauchen je nach Aktion Rohdaten, Checkpoints oder Gewichtsdateien und eine GPU; sie sind nicht Teil der Standardaufrufe.

Die Skripte importieren weder TensorFlow noch öffnen sie Bilder, solange keine ausdrückliche Aktion gewählt wird. Die Zeichenskripte in `plots/` lesen gespeicherte Tabellen und schreiben Abbildungen nach `reports/figures/`; nur der Grad-CAM-Zeichner liest zusätzlich die wenigen ausgewählten Bilder und Masken und schreibt lokale Vorschauen nach `data/processed/` (mit Testbildern, nur lokal).

## 2 Ausdrücklich zu wählende Aktionen (neue Läufe)

| Aktion | Befehl | Hardware | Wirkung |
|---|---|---|---|
| Baselines neu anpassen | `analysis/02_baselines.py --fit` | CPU | schreibt Metriken, Vorhersagen und zwei Modelle |
| Klassifikation trainieren | `.venv/bin/python analysis/03_classification.py --preset <preset> --train --seeds <Seed> [<Seed> ...]` | GPU | `busbra.launcher` startet je Variante und Seed einen eigenen Prozess in der Reihenfolge von `src/busbra/config.py` (Seeds außen, Varianten innen; Fine-Tuning startet aus dem besten Frozen-Checkpoint desselben Seeds) und setzt `PYTHONHASHSEED` gleich dem Seed (`single_runs`: nur Seed 42, ohne Hash-Seed); prüft vorab alle Ausgabedateien des ganzen Plans und bei ROI-Varianten die gemeinsame Expertengeometrie |
| Expertengeometrie vorbereiten | `.venv/bin/python analysis/03_classification.py --preset expert_roi --prepare` | CPU | misst die Expertenmasken von Training und Validation **einmal** (nicht je Seed oder Variante) und schreibt `expert_roi_geometry_train_validation.csv` und die Häufigkeitstabelle; ROI-Training und Zusatzexperiment lesen sie und messen keine Masken; fehlt sie, nennt die Fehlermeldung diesen Befehl |
| Eine Variante direkt | `PYTHONHASHSEED=<Seed> env -u LD_LIBRARY_PATH .venv/bin/python analysis/03_classification.py --preset <preset> --train-one <Variante> --seeds <Seed>` | GPU | genau eine Variante und ein Seed in diesem Prozess (das, was der Launcher startet); ohne passenden `PYTHONHASHSEED` bricht sie vor dem Laden von Daten ab |
| Tabellen eines Presets schreiben | `... 03_classification.py --preset <preset> --aggregate` | CPU | schreibt die Studientabellen (Experten-ROI: zusätzlich Schwellenkurven, Betriebspunkte und gepaarte Unterschiede) |
| Masken-EDA | `analysis/04_mask_eda_roi.py --run` | CPU | liest nur Trainingsbilder und -masken |
| Segmentierung trainieren | `PYTHONHASHSEED=7 env -u LD_LIBRARY_PATH .venv/bin/python analysis/05_segmentation.py --train` | GPU | U-Net, Seed 7 (der Hash-Seed 7 gehört zur dokumentierten Ausführung) |
| Automatische ROI | `... 06_auto_roi.py --run` | GPU | Inferenz auf den Validation-Bildern: Geometrie, Pixelvorhersagen, Scores, Diagnosen |
| Automatische ROI auswerten | `... 06_auto_roi.py --aggregate` | CPU | schreibt die 15 Studien- und Diagnosetabellen aus den gespeicherten Scores und Diagnosen (öffnet keine Bilder, Masken oder Modelle); der Standardaufruf leitet sie im Arbeitsspeicher ab und vergleicht |
| Zusatzexperiment | `... 08_adversarial_addon.py --train` | GPU | 5 Referenzauswertungen und 10 Fortsetzungen (nur Train und Validation) |
| Testinferenz | `... 09_test_evaluation.py --infer` | GPU | **einmalig**: Scores der 27 Kombinationen, automatische ROIs, U-Net-Diagnose auf den Testbildern |
| Grad-CAM | `... 10_interpretation.py --run` | GPU | Karten für die 11 regelbasiert gewählten Testbilder |
| FGSM | `... 11_robustness_devices.py --attack` | GPU | zehn Angriffe auf dem Test |
| Tabellen schreiben | `--write` bei 07, 08, 09, 11, 12 und `--aggregate` bei 03 | CPU | schreibt fehlende Tabellen; alle Ziele werden vorab gemeinsam geprüft, bei einem Konflikt wird nichts geschrieben (09 schreibt nach einer neuen Testinferenz auch die lange Score-Tabelle, die die übrigen Analysen lesen) |

Reihenfolge eines vollständigen neuen ROI-Bestands (alle fünf Studien-Seeds; die vollständigen Vollbild-Referenzen 4B und die lokale ImageNet-Datei sind Voraussetzung): Geometrie vorbereiten (`--prepare`), U-Net trainieren, Experten-ROI für alle fünf Seeds trainieren, Experten-ROI aggregieren (verlangt alle fünf Seeds), danach `06 --run` und `06 --aggregate`; die Befehle stehen im README. Ein Teillauf mit einzelnen Seeds ist möglich; dessen lesende Standardansicht ist als Teilansicht gekennzeichnet und wird nie als Tabelle geschrieben. `--variants` gehört nur zu `--train`. Das Zusatzexperiment liest dieselbe Expertengeometrie.

Jede dieser Aktionen prüft alle ihre Ausgabedateien, bevor Bilder oder Modelle geladen werden, und bricht mit einem Fehler ab, wenn auch nur eine davon schon existiert (zum Beispiel blockiert die vorhandene zentrale Score-Tabelle eine erneute Testinferenz). Es gibt keinen versteckten Wiederanlauf, kein automatisches Löschen und keinen Start allein wegen einer fehlenden Statusdatei. Ein neues Training erzeugt andere Checkpoints und Scores als die gespeicherten; vorhandene Ergebnisse bleiben deshalb unverändert, und ein Lauf in einen anderen Ordner ist nur durch Anpassen der Pfade in `src/busbra/config.py` möglich.

## 3 Umgebung, Daten und Eingabe-Spezifikation

**Umgebung (tatsächlich verwendet).** Linux Mint 22.3, NVIDIA GeForce RTX 3050 Laptop GPU (4.096 MiB), Python 3.13.14 aus einer conda-Umgebung als Basis der projektlokalen `.venv`, TensorFlow 2.20.0 mit Keras 3.15.1 (`tensorflow[and-cuda]`), NumPy 2.5.4, pandas 3.0.6, Pillow 13.0.0, scikit-learn 1.9.1, Matplotlib 3.11.2; vollständiger Paketstand in [environment_freeze.txt](environment_freeze.txt). Die Installation aus `requirements.txt` wurde nur als `--dry-run` gegen die bestehende Umgebung geprüft, nicht in einer leeren Umgebung. TensorFlow-Schritte laufen mit `env -u LD_LIBRARY_PATH`, damit keine fremde CUDA-Umgebung geladen wird (Details: [environment_and_gpu.md](environment_and_gpu.md)); `src/busbra/training.py` aktiviert Memory Growth vor der Initialisierung und bricht ohne GPU ab (kein stiller CPU-Ersatz).

**Numerik.** Neuronale Scores wurden im TF32-Modus berechnet. Eine erneute GPU-Inferenz reproduziert die gespeicherten Scores nur im Rahmen der dokumentierten Toleranzen (zum Beispiel höchstens 6e-08 in der Vorkontrolle der Testauswertung, bei gleicher Batch-Zusammensetzung bitgleich); andere Batch-Zusammensetzungen verschieben Scores in der Größenordnung 1e-3, und die Richtung der Gradienten ist numerisch empfindlich (siehe [robustness_and_devices.md](robustness_and_devices.md)). Geladene Modelle werden vor der Vorhersage mit `jit_compile=False` kompiliert; ein unkompiliertes Modell würde mit automatischem XLA leicht andere Werte liefern. Ein erneutes Training liefert nicht notwendigerweise dieselben Gewichte.

**Daten beschaffen.** Original `BUSBRA.zip` aus dem Zenodo-Record [https://zenodo.org/records/8231412](https://zenodo.org/records/8231412) (DOI 10.5281/zenodo.8231412, Version 1.0, Lizenz CC BY 4.0 laut Metadaten; die mitgelieferte `LICENSE.txt` verlangt die Zitierung der Veröffentlichung). Erwartet: 133.917.740 Bytes, MD5 `1f8b2be6476d58fc97bfb5e5a1ea9bab`; entpackt nach `data/raw/BUSBRA/`. Die ZIP-Datei selbst wird zum Lesen und Nachrechnen nicht gebraucht. `data/processed/manifest_split.csv` ist die einzige maßgebliche Zuordnung der Bilder zu Training, Validation und Test (fester, nach `Case` getrennter Holdout aus den äußeren Folds von `5-fold-cv.csv`: Folds 3–5 Training, 2 Validation, 1 Test); `analysis/01_prepare_data.py --build` schreibt es nur, wenn es noch nicht existiert.

**Eingabe-Spezifikation (aus `src/busbra/image_processing.py`).** Graustufenbild (8 Bit) einlesen; das ganze Bild proportional in das Quadrat einpassen (`PIL.ImageOps.contain`, bilineare Interpolation, auch Vergrößerung kleiner Bilder), zentriert auf schwarzem Rand; `float32(uint8) / 255` nach [0, 1]; keine weitere Skalierung und nichts aus Daten geschätzt. Zielgrößen: 64 × 64 (logistische Regression, abgeflacht, 4.096 Merkmale), 160 × 160 × 1 (neuronale Klassifikatoren; 250 × 250 nur als Vergleich), 256 × 256 × 1 (U-Net). Die MobileNetV2-Klassifikatoren kopieren den Kanal im Modell dreifach und skalieren intern mit `Rescaling(2, −1)` auf [−1, 1]; das kleine CNN und das U-Net haben diese Skalierung nicht. Zuschnitt und Rückabbildung der ROI: `reports/tables/roi_contract.json` und `src/busbra/roi.py`.

## 4 Benötigte Dateien je Kombination

Die 27 Kombinationen der Testauswertung (sieben Gruppen) sind in `src/busbra/final_evaluation.py` (`GROUPS`, `combinations()`) festgelegt.

| Gruppe (Kombinationen) | Scores | Modell (nur lokal) |
|---|---|---|
| Dummy, logistische Regression (1 + 1) | `reports/tables/baseline_predictions.csv` | `models/baselines/` |
| kleines CNN (5) | `reports/tables/multiseed/seed_<s>/small_cnn_160/` | `models/multiseed/seed_<s>/small_cnn_160/` |
| Vollbild frozen (5) und feinjustiert (5) | `reports/tables/augmentation160/seed_<s>/<variante>/` | `models/augmentation160/seed_<s>/<variante>/` |
| Experten-ROI (5) | `reports/tables/expert_roi/seed_<s>/mobilenetv2_higher_lr_160_roi/` | `models/expert_roi/seed_<s>/mobilenetv2_higher_lr_160_roi/` |
| automatische ROI (5) | `reports/tables/auto_roi/seed_<s>/finetune/scores_auto.csv`; **dieselben fünf Experten-ROI-Checkpoints** | zusätzlich U-Net `models/segmentation/unet_seed7/unet_256_seed7_best.keras` |

Die Test-Scores, die Schwellen und alle Kennzahlen stehen in `reports/tables/test_evaluation_*.csv` und `final_operating_points.csv`; zum Nachrechnen sind keine Modelle nötig. Das Zusatzexperiment (Adversarial-Training-Fortsetzung) und die übrigen Checkpoints der Entwicklung gehören nicht zu den 27 Kombinationen.

## 5 Quellstände und Entwicklungsmaterial

Die Studien wurden mit früheren, umfangreicheren Treiberskripten ausgeführt (Registrierung der Pläne, Hashketten, Statusverwaltung, Wiederanlauf-Schutz, Umgebungsprüfung). Diese Treiber, die Hash- und Dateiprüfer, Prüfprotokolle, Kursmaterial, Entwicklungsprotokolle, rein technische Ergebnisdateien (Status, Pläne, Registrierungen, Laufprotokolle, Umgebungsberichte, Kalibrierungen), die Erstellungsquellen der Präsentation (Aufbauskripte, Notizen, Fragen und Antworten, Assets) und die frühere Zweitablage der Geräte-Nachrechnung (`final_review`) sind nicht Teil des Repositories. Die fachlichen Ergebnisse, Scores, Verläufe, Schwellen, Geometrien und Abbildungen sind unverändert; die Geräteergebnisse stehen in `reports/tables/device_test_*.csv`. Die gespeicherten Ergebnisse entstanden mit dem gesicherten damaligen Quellstand; die heutigen Skripte sind eine Zusammenführung derselben fachlichen Abläufe. Dass sie die Ergebnisse reproduzieren, ist für alle Rechnungen aus gespeicherten Scores durch die Vergleichsausgaben (Abschnitt 1) und die Tests belegt, auch für die Studientabellen mit ihren Referenzgruppen. Für Training und Inferenz wurde die Gleichheit der Konfiguration (Architektur, Seeds, Initialisierung, Trainierbarkeit, Loss, Optimizer, Batch- und Callback-Regeln) im Quelltext verglichen und im Abschlusscheck vom 22.10.2026 durch begrenzte GPU-Funktionsprüfungen ergänzt (Abschnitt 7). Ein vollständiger Neulauf der Studien liegt nicht vor.

Die Studiendokumente nennen die damaligen Skripte, Prüfprotokolle, Statusdateien und Ablagen (zum Beispiel `analysis/plot_final_report_figures.py`, heute `plots/plot_final_report_figures.py`, oder `reports/final_review/`); sie beschreiben die ursprüngliche Ausführung. Frühere Skriptpfade und nicht mehr vorhandene Status- und Registrierungsdateien sind keine heutigen Pflichteingaben; die historischen Ergebnisse und Zeitpunkte wurden nicht nachträglich geändert. Die Zuordnung alter zu neuen Skripten steht im README. Maßgeblicher Bericht ist die [Projektdokumentation](BUSBRA_Projektdokumentation.md) (die datierte PDF-Abgabefassung vom 22.10.2026 liegt nur lokal). Die Präsentationsfolien enthalten Ultraschallbilder und sind nicht Teil des Repositories.

## 6 Kursanforderungen (Kurzfassung)

Der ausführliche Abgleich mit der Evaluations-Checkliste des Kurses (Projektvorlage Tag 20) ist Entwicklungsmaterial und nicht Teil des Repositories. Die wesentlichen Punkte:

* **Reproduzierbarkeit:** Seeds 7, 21, 42, 123 und 2026 (der frühe Einzellauf des kleinen CNN in Schritt 3A verwendete Seed 42; im finalen Vergleich stammt das kleine CNN aus der Mehr-Seed-Studie 3F und verwendet ebenfalls alle fünf Seeds); `requirements.txt` und `docs/environment_freeze.txt` (die Neuinstallation wurde nur als Trockenlauf geprüft); jedes Skript besteht aus chronologischen `# %%`-Zellen und läuft einzeln von oben nach unten. Die Rohdaten sind nicht Teil des Repositories.
* **Methodik:** fester, nach `Case` getrennter Holdout; keine aus Daten geschätzte Skalierung; Baselines Dummy und logistische Regression; Auswahl und Schwellen nur auf der Validation; Test einmal ausgewertet und danach nur nach vorab festgelegten Regeln weiterverwendet.
* **Evaluation:** ROC-AUC (primär), AP, Sensitivität, Spezifität, Precision, F1 und Accuracy; Betriebspunkte mit Konfusionszahlen; Lernkurven und Baseline-Vergleich im Bericht.
* **Interpretation und Reflexion:** Grad-CAM (qualitativ, ohne Kausalanspruch), FGSM-Stresstest, beschreibender Gerätevergleich, Grenzen und vier nicht durchgeführte nächste Schritte im Bericht.
* **Teilweise erfüllt oder offen:** kein numerischer Mindestwert als Erfolgskriterium (Projektdokumentation, Abschnitt 2.3); keine Neuinstallation in einer leeren Umgebung getestet.

## 7 Prüfstand der heutigen Skriptstruktur

Die folgenden Angaben stammen aus dem dokumentierten Abschlusscheck vom 22.10.2026 (Protokolle und Prüfausgaben liegen außerhalb des Repositories) und aus der CPU-Prüfung für die Veröffentlichung am 30.10.2026. Für die Veröffentlichung wurden keine GPU-Läufe wiederholt.

**Ausgeführt:**

* *CPU:* Syntaxprüfung aller Python-Dateien, Unit-Tests (bei blockiertem TensorFlow/Keras-Import), Standardaufrufe aller Analyseskripte einschließlich der fünf Presets von `03` und Nachrechnungen gespeicherter Ergebnisse im Arbeitsspeicher (Tabellenvergleiche; Float-Unterschiede höchstens 1,1e-16). Am 30.10.2026 wurde dies wiederholt, einmal im vollständigen Projekt und einmal in einer Kopie nur des öffentlichen Bestands.
* *GPU-Funktionsprüfungen (22.10.2026, begrenzt):* Laden der Checkpoints und Vorhersage auf je 16 Validation-Bildern (kleines CNN, MobileNetV2 eingefroren, Vollbild feinjustiert, Experten-ROI; Seed 7; Scores gleich den gespeicherten), je ein echter Optimizer-Schritt in 14 Trainingsvarianten, U-Net-Inferenz und ein U-Net-Trainingsschritt, die automatische ROI mit frisch erzeugten Vorhersagen, Grad-CAM, FGSM und der Adversarial-Pfad. Das sind Funktionsprüfungen der Anschlüsse; sie belegen nicht die Gleichheit ganzer Experimente.
* *Vollständiger Neulauf des kleinen CNN (Seed 42, 22.10.2026):* mit dem echten Trainer und Ausgaben außerhalb des Projekts; 30 Epochen, beste Epoche 25, Validation-ROC-AUC ungefähr 0,6971 gegenüber 0,7026 im gespeicherten frühen Lauf. Das ist ein technischer Prüflauf, kein Ergebnis der Vergleichstabellen.

**Beobachtete Abweichungen:** Beim U-Net unterschieden sich die Pixelzahlen der Vorhersagen auf 8 Validation-Bildern um bis zu drei Pixel (höchstens 0,033 %) bei gleichen Boxen, Komponentenzahlen und Rückfallentscheidungen; zwei Läufe untereinander wichen in den Wahrscheinlichkeiten bis 4,4e-4 ab. Das spricht für GPU-Numerik (TF32), die Ursache wurde nicht abschließend bewiesen, und daraus folgt keine Gleichheit aller Vorhersagen.

**Nicht belegt:** ein vollständiger ROI-Neuaufbau, eine Wiederholung aller Studien, eine Neuinstallation in einer leeren Umgebung sowie neue Testinferenz, Test-Karten oder FGSM-Läufe.
