# BUS-BRA: Brustultraschall – Vollbild oder Läsionsausschnitt?

Projektarbeit im Modul Deep Learning der IfaDW-Weiterbildung „AI Engineer“. Untersucht wird, ob die Eingrenzung des Läsionsbereichs die Unterscheidung gutartiger und bösartiger Befunde im Brustultraschall verbessert.

## Fragestellung und Daten

> Verbessert die Eingrenzung des Läsionsbereichs die Unterscheidung gutartiger und bösartiger Befunde im Brustultraschall?

Grundlage ist der öffentliche Datensatz **BUS-BRA** (1.875 Bilder von 1.064 Fällen, vier Geräte, eine Einrichtung; Läsionsmasken liegen bei). Verglichen werden drei Eingaben für denselben Modelltyp (MobileNetV2, feinjustiert, fünf Trainings-Seeds):

| Eingabe | Was der Klassifikator sieht | Bei der Anwendung benötigt |
|---|---|---|
| **Vollbild** | das ganze Ultraschallbild | nur das Bild |
| **Experten-Zuschnitt** | Ausschnitt um die Läsion (Box der Expertenmaske plus 20 % je Seite, unmaskiertes Original) | Bild und manuelle Läsionsmaske |
| **Automatischer Zuschnitt** | Ausschnitt aus einer U-Net-Segmentierung; derselbe Klassifikator wie beim Experten-Zuschnitt, nicht neu trainiert | nur das Bild |

Als Vergleichsmodelle dienen ein Dummy-Klassifikator (Klassenprior), eine logistische Regression auf 64 × 64 Pixeln, ein selbst entwickeltes kleines CNN und MobileNetV2 mit eingefrorenem Backbone.

Der Datensatz enthält keine gesunde Population; „gutartig“ bedeutet gutartige Läsion. Die Arbeit misst keine Verbesserung ärztlicher Entscheidungen und begründet keine klinische Einsatzfähigkeit.

## Zentrale Testergebnisse

Test: 376 Bilder aus 212 Fällen, einmalige Auswertung mit auf der Validation festgelegten Schwellen. Angegeben ist die Bild-ROC-AUC (Mittel ± Standardabweichung über fünf Seeds; ein Rangmaß, keine Trefferquote):

| Gruppe | Bild-ROC-AUC |
|---|---:|
| Dummy (Klassenprior) | 0,500 |
| Logistische Regression (64 × 64, ein Modell) | 0,688 |
| Kleines CNN | 0,613 ± 0,107 |
| MobileNetV2, eingefroren, Vollbild | 0,817 ± 0,010 |
| **Vollbild** (feinjustiert) | **0,895 ± 0,009** |
| **Experten-Zuschnitt** | **0,937 ± 0,008** |
| **Automatischer Zuschnitt** | **0,915 ± 0,008** |

Gepaarte Differenzen der Bild-ROC-AUC mit 95-%-Perzentilintervall (Cluster-Bootstrap über Fälle, 2.000 Ziehungen):

* Experten-Zuschnitt minus Vollbild: +0,042 [+0,015; +0,073]
* Automatischer Zuschnitt minus Experten-Zuschnitt: −0,022 [−0,039; −0,007]
* Automatischer Zuschnitt minus Vollbild: +0,020 [−0,009; +0,050]

Ein von Expert:innen lokalisierter Läsionsbereich verbessert in diesem Test die Trennschärfe gegenüber dem ganzen Bild. Die automatische Eingrenzung liegt hinter dem Experten-Zuschnitt, und ihr Vorteil gegenüber dem ganzen Bild ist **statistisch nicht gesichert** (das Intervall schließt 0 ein). Die Pipelines unterscheiden sich in der Information bei der Anwendung; der Unterschied ist daher kein reiner Masken-Effekt.

Weitere Befunde, Einordnung und Grenzen stehen in der [Projektdokumentation](docs/BUSBRA_Projektdokumentation.md). Zahlen aus der Entwicklung auf der Validation sind explorativ und stehen getrennt in den Studiendokumenten.

## Vorgehen

* **Split:** fester, nach Fall getrennter Holdout aus den äußeren Folds von `5-fold-cv.csv` des Datensatzes (Training 1.114 Bilder / 639 Fälle, Validation 385 / 213, Test 376 / 212). Es wurde keine Kreuzvalidierung ausgeführt.
* **Seeds:** fünf Trainings-Seeds (7, 21, 42, 123, 2026) auf demselben Holdout; sie messen die Trainingsstreuung, nicht die Streuung über neue Kohorten. Es gibt keinen „besten Seed“, kein Ensemble und keine gepoolte AUC.
* **Auswahl und Schwellen** nur auf der Validation; der Test wurde einmal ausgewertet, danach folgten nur Interpretation und Robustheit nach vorab festgelegten Regeln ([Protokoll](docs/final_evaluation_protocol.md)).
* **Eingabe:** allein das Ultraschallbild (beim Zuschnitt der daraus geschnittene Ausschnitt). Fall, Histologie, BI-RADS, Gerät, Seite und Masken sind keine zusätzlichen Eingaben; Masken lokalisieren nur im Experten-Zweig und sind Trainingsziel der Segmentierung.
* **Ebenen:** Bild- und Fallebene getrennt (Fall-Score = Maximum der Bild-Scores).

## Ordnerübersicht

| Pfad | Inhalt |
|---|---|
| `analysis/` | die fachlichen Abläufe in Reihenfolge (`# %%`-Zellen): `01_prepare_data`, `02_baselines`, `03_classification` (alle Klassifikationsstudien als Presets), `04_mask_eda_roi`, `05_segmentation`, `06_auto_roi`, `07_operating_points`, `08_adversarial_addon`, `09_test_evaluation`, `10_interpretation`, `11_robustness_devices`, `12_threshold_analysis` |
| `src/busbra/` | gemeinsame Implementierung: Konfiguration, Datenzugriff, Modelle (kleines CNN, MobileNetV2), Training und Inferenz, Segmentierung und Zuschnitt, Metriken, Schwellen, Bootstrap, Interpretation und Robustheit |
| `plots/` | Zeichenskripte für die Statistikabbildungen und der PDF-Export des Berichts |
| `tests/` | Regressionstests an den fachlichen Grenzen (Split, Schwellen, Zuschnittsrundung, Bootstrap, gespeicherte Kennzahlen) |
| `docs/` | Methoden- und Studiendokumente je Schritt, Projektdokumentation, Quellen ([Übersicht](#dokumente)) |
| `reports/tables/`, `reports/figures/` | gespeicherte Ergebnisse (Scores, Verläufe, Kennzahlen, Schwellen, Geometrien) und Statistikabbildungen |
| `data/processed/manifest_split.csv` | Split-Manifest: Zuordnung der Bilder zu Training, Validation und Test mit Metadaten, relativen Pfaden und Bild-Hashes (enthält keine Bilder) |

Nicht im Repository enthalten sind Rohbilder und Masken, Modellgewichte, abgeleitete Arrays, lokale Bildvorschauen und die Präsentationsfolien, weil sie Ultraschallbilder oder Masken enthalten bzw. zu groß sind.

## Installation

Entwickelt und ausgeführt wurde mit Python 3.13 unter Linux (NVIDIA GeForce RTX 3050 Laptop GPU, 4.096 MiB), TensorFlow 2.20.0 (Keras 3.15.1), NumPy, pandas, scikit-learn, Pillow und Matplotlib. Die direkten Abhängigkeiten stehen in [requirements.txt](requirements.txt), der vollständige Paketstand der Entwicklungsumgebung in [docs/environment_freeze.txt](docs/environment_freeze.txt).

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -c docs/environment_freeze.txt
```

Diese Installation wurde nur als Trockenlauf gegen die bestehende Umgebung geprüft, nicht in einer leeren Umgebung. Die TensorFlow-Schritte brauchen eine GPU und laufen ohne fremde CUDA-Umgebung (`env -u LD_LIBRARY_PATH`; der Prozessstart des Trainings setzt das selbst). Hinweise zu GPU und Umgebung: [docs/environment_and_gpu.md](docs/environment_and_gpu.md).

## Daten bereitstellen

Die Rohdaten gehören nicht zum Repository. `BUSBRA.zip` stammt aus [Zenodo 8231412](https://zenodo.org/records/8231412) (DOI 10.5281/zenodo.8231412; 133.917.740 Bytes; MD5 `1f8b2be6476d58fc97bfb5e5a1ea9bab`) und wird nach `data/raw/BUSBRA/` entpackt (Unterordner `Images/` und `Masks/`). Das Split-Manifest `data/processed/manifest_split.csv` liegt im Repository und ist die maßgebliche Zuordnung der Bilder zu den Splits.

Lizenz des Datensatzes laut Zenodo: CC BY 4.0. Bei Verwendung ist die Veröffentlichung zu zitieren: Gómez-Flores, Gregorio-Calas, Coelho de Albuquerque Pereira, *BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis systems*, Medical Physics 51, 3110–3123, 2024, [doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812). Weitere Quellen: [docs/literature_and_sources.md](docs/literature_and_sources.md).

## Gespeicherte Ergebnisse lesen

Die Ergebnisse liegen als CSV- und JSON-Dateien unter `reports/tables/` und als Abbildungen unter `reports/figures/`. Sie entstanden in den früheren, umfangreicheren Ausführungsskripten des Projekts; die Skripte in `analysis/` sind eine Zusammenführung derselben fachlichen Abläufe. Dass sie die gespeicherten Kennzahlen aus den gespeicherten Scores reproduzieren, ist durch Vergleichsausgaben und Tests belegt. Die GPU-Pfade der heutigen Struktur wurden nur als begrenzte Funktionsprüfungen und mit einem Neulauf des kleinen CNN (Seed 42) geprüft; ein vollständiger Neuaufbau der Zuschnittszweige, eine Wiederholung aller Studien und eine Neuinstallation in leerer Umgebung sind nicht belegt. Einzelheiten: [docs/reproducibility_and_handover.md](docs/reproducibility_and_handover.md), Abschnitt 7.

Jedes Skript in `analysis/` liest ohne Argument nur gespeicherte Ergebnisse und vergleicht sie, wo möglich, mit einer Neuberechnung im Arbeitsspeicher. Rechenintensive oder schreibende Aktionen müssen ausdrücklich gewählt werden (`--train`, `--fit`, `--run`, `--infer`, `--attack`, `--write`, `--aggregate`). Sie brechen vorab mit einem Fehler ab, wenn eine ihrer Ausgabedateien schon existiert; fehlende Ergebnisse lösen kein Training aus. Beispiele für lesende Aufrufe:

```bash
.venv/bin/python analysis/03_classification.py --preset multiseed   # fünf Seeds, Referenzgruppen und Paarvergleiche, nachgerechnet und verglichen
.venv/bin/python analysis/06_auto_roi.py                            # automatischer Zuschnitt: Kennzahlen aus gespeicherten Scores und Diagnosen
.venv/bin/python analysis/09_test_evaluation.py                     # Testkennzahlen und Bootstrap aus den gespeicherten Scores (ein bis zwei Minuten)
```

Diese drei Aufrufe (und die übrigen Standardaufrufe von `analysis/`) wurden in einer Kopie nur des öffentlichen Bestands ohne Fehler ausgeführt; sie brauchen weder Modellgewichte noch Rohdaten. Ausnahme ist `analysis/01_prepare_data.py`: Die Metadatenprüfung liest `data/raw/BUSBRA/` und bricht ohne Rohdaten ab. Training, Inferenz und der Neuaufbau der Zuschnittszweige sind ausdrücklich zu wählende GPU-Aktionen, setzen Rohdaten, Checkpoints beziehungsweise die lokale ImageNet-Gewichtsdatei voraus und gehören nicht zu den hier beschriebenen Aufrufen; die Optionen zeigt `--help`, die Abläufe stehen in [docs/reproducibility_and_handover.md](docs/reproducibility_and_handover.md).

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

Die 68 vorhandenen Tests laufen ohne TensorFlow auf der CPU. Im vollständigen lokalen Projekt laufen alle 68 durch. In einer Kopie nur des öffentlichen Bestands werden 66 Tests ausgeführt und bestanden, 2 werden übersprungen, weil optionale lokale Referenzdateien fehlen: `data/processed/gradcam_test_maps.npz` (Grad-CAM-Tabellen gegen den Renderer) und `reports/tables/test_evaluation/bootstrap_arrays.npz` (historische Bootstrap-Ziehungen). Übersprungene Tests sind keine bestandenen fachlichen Prüfungen; die synthetischen Grad-CAM-Format- und Schnittstellentests laufen auch ohne diese Dateien.

## Dokumente

* Abschluss: [BUSBRA_Projektdokumentation.md](docs/BUSBRA_Projektdokumentation.md) (maßgeblicher Bericht), [final_evaluation_protocol.md](docs/final_evaluation_protocol.md), [test_evaluation.md](docs/test_evaluation.md), [gradcam_test_interpretation.md](docs/gradcam_test_interpretation.md), [robustness_and_devices.md](docs/robustness_and_devices.md), [adversarial_roi_comparison.md](docs/adversarial_roi_comparison.md)
* Projekt: [project_definition.md](docs/project_definition.md), [reproducibility_and_handover.md](docs/reproducibility_and_handover.md), [literature_and_sources.md](docs/literature_and_sources.md), [environment_and_gpu.md](docs/environment_and_gpu.md)
* Daten und Split: [data_audit.md](docs/data_audit.md), [split_definition.md](docs/split_definition.md), [train_eda.md](docs/train_eda.md)
* Modelle: [baselines.md](docs/baselines.md), [small_cnn.md](docs/small_cnn.md), [transfer_learning.md](docs/transfer_learning.md), [fine_tuning.md](docs/fine_tuning.md), [extended_fine_tuning.md](docs/extended_fine_tuning.md), [learning_rate_comparison.md](docs/learning_rate_comparison.md), [multiseed_comparison.md](docs/multiseed_comparison.md), [resolution_250_comparison.md](docs/resolution_250_comparison.md), [augmentation_160_comparison.md](docs/augmentation_160_comparison.md)
* Schwellen, Masken, Zuschnitt: [threshold_analysis.md](docs/threshold_analysis.md), [masks_and_roi_definition.md](docs/masks_and_roi_definition.md), [expert_roi_comparison.md](docs/expert_roi_comparison.md), [segmentation_4fa.md](docs/segmentation_4fa.md), [auto_roi_classification.md](docs/auto_roi_classification.md), [interpretation_robustness_precheck.md](docs/interpretation_robustness_precheck.md)

Die Studiendokumente beschreiben die ursprüngliche Ausführung mit den damaligen Skripten (frühere Nummerierung `01`–`22`); frühere Skriptpfade und nicht mehr vorhandene Status- und Registrierungsdateien sind keine heutigen Pflichteingaben. Die Skripte sind in `analysis/` neu zugeordnet: Rohdaten-Audit und Split → `01_prepare_data`; Baselines → `02_baselines`; kleines CNN, MobileNetV2-Varianten, Mehr-Seed, Auflösung, Augmentation und Experten-Zuschnitt → `03_classification`; Masken-EDA → `04_mask_eda_roi`; U-Net → `05_segmentation`; automatischer Zuschnitt → `06_auto_roi`; Abschlussprotokoll und Betriebspunkte → `07_operating_points`; Zusatzexperiment → `08_adversarial_addon`; Testauswertung → `09_test_evaluation`; Grad-CAM → `10_interpretation`; FGSM und Geräte → `11_robustness_devices`. Weitere Hinweise: [docs/reproducibility_and_handover.md](docs/reproducibility_and_handover.md).

## Grenzen

* Eine Datenquelle mit vier Geräten und **keine externe Validierung**; die Ergebnisse sind interne Forschungsergebnisse auf Läsionsbildern, keine validierte klinische Diagnostik und kein Screening gesunder Personen.
* Fester Holdout und fünf Seeds; keine Kreuzvalidierung. Die Validation wurde wiederholt für die Entwicklung genutzt, ihre Ergebnisse sind möglicherweise optimistisch.
* Der Vorteil des automatischen Zuschnitts gegenüber dem Vollbild ist nicht gesichert; der automatische Zuschnitt nutzt Klassifikatoren, die auf Experten-Zuschnitten trainiert wurden, und ein einzelnes U-Net (ein Seed).
* Mögliche Abkürzungen durch eingebrannte Markierungen und Gerätemerkmale; der Experten-Zuschnitt setzt eine manuelle Maske voraus.
* Ein einstufiger FGSM-Angriff senkt die Bild-ROC-AUC beider geprüften Pipelines weit unter 0,5; die Modelle sind nicht robust gegen solche Störungen.

## Lizenz

Der eigene Quellcode dieses Repositories steht unter der [MIT-Lizenz](LICENSE) (Copyright (c) 2026 Carlo Babo). Sie gilt nur für diesen Quellcode. Sie räumt keine Rechte an fremden Daten, Modellgewichten, Bibliotheken oder Publikationen ein, und die Berichte und Dokumente unter `docs/` werden dadurch nicht stillschweigend zusätzlich lizenziert.

Der Datensatz BUS-BRA und die daraus übernommenen Metadaten im Split-Manifest behalten ihre eigene Lizenz und Quellenangabe (CC BY 4.0 laut Datensatzquelle, siehe [Daten bereitstellen](#daten-bereitstellen) und [docs/literature_and_sources.md](docs/literature_and_sources.md)).
