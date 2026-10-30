# Rohdaten-Audit BUS-BRA

Stand: 17.10.2026. Ursprünglich mit den früheren Treibern erzeugt, heute entsprechend: `analysis/01_prepare_data.py`. Python 3.13.14, NumPy 2.5.4, pandas 3.0.6, Pillow 13.0.0; kein TensorFlow.

Alle Zahlen in diesem Dokument stammen aus den erzeugten Artefakten:

- `reports/tables/audit_status.json` (alle Prüfungen mit Detailtext)
- die CSV-Tabellen unter `reports/tables/audit_*.csv`
- `data/processed/manifest_audit.csv`

Referenzwerte aus der Projektvorbereitung (erwartete Eigenschaften der Datensatzversion, im Skript hinterlegt) dienten nur zum Vergleich.

**Ergebnis:**

- Status `passed`: 92 Prüfungen, 0 Fehler, 3 Warnungen. Die drei Warnungen sind bekannte, dokumentierte Abweichungen (Abschnitte 5–7).
- Alle 26 Vergleichswerte der Projektvorbereitung wurden reproduziert (`audit_reference_comparison.csv`).
- Zwei Läufe ergaben bytegleiche Manifest- und Tabellendateien. Die Rohdaten blieben unverändert (Abschnitt 9).

## 1. Quelle, Integrität und Lizenz

| Angabe | Wert |
|---|---|
| Quelle | Zenodo-Record https://zenodo.org/records/8231412, DOI 10.5281/zenodo.8231412, Version 1.0 |
| Datei | `data/raw/BUSBRA.zip`, unverändert |
| Größe | 133.917.740 Bytes (erwartet: 133.917.740) |
| MD5 | `1f8b2be6476d58fc97bfb5e5a1ea9bab` (erwartet laut Zenodo: identisch) |
| SHA-256 | `ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff` |
| Publikation | Gómez-Flores, Gregorio-Calas, Coelho de Albuquerque Pereira, *Medical Physics* 51, 3110–3123 (2024), https://doi.org/10.1002/mp.16812 |

**Lizenz, getrennt dokumentiert:**

- **Zenodo-Metadaten:** CC BY 4.0 (`cc-by-4.0`). Am 17.10.2026 über die Zenodo-API eingesehen; das Skript selbst arbeitet offline.
- **Mitgelieferte `data/raw/BUSBRA/LICENSE.txt`:** „Copyright (c) 2023“ mit einem permissiven, MIT-ähnlichen Erlaubnistext. Er verlangt das Zitieren der Publikation und den Erhalt von Copyright- und Erlaubnishinweis. Der Text ist nicht wortgleich mit CC BY 4.0. Er nennt die Publikation noch als „Medical Physics, 2023. (In revision)“; zitiert wird die veröffentlichte Fassung von 2024.
- SHA-256 der Lizenzdatei: `e9e3ea607a53575d9d17c07cdb35972d9506a19dcafde49383fb0f875f01b92d`.

**SHA-256 der Original-CSV-Dateien:**

| Datei | SHA-256 |
|---|---|
| `bus_data.csv` | `f218d0f10b0aeeb14ff4b2cfed8631b2f4a7cf608e564548b35bddd55f3aafb3` |
| `5-fold-cv.csv` | `8bed2d2efdf6ea370ab82a3cbda4c213f498274a68b434828ceeb636ce43a68d` |
| `10-fold-cv.csv` | `e4b788a1073b51179e2d1e501a1e87ebbff8c743248787b8f1641aea51e9e242` |

## 2. Sicheres Entpacken

- **Mitgliederprüfung vor dem Schreiben:** 3.757 Archiveinträge (3.754 Dateien, 3 Ordner). Keine absoluten Pfade, kein `..`, keine Backslashes, keine Symlinks, keine Duplikate, nur das dokumentierte Layout unter `BUSBRA/`.
- **Erster Lauf:** Die Einträge werden einzeln in den temporären Ordner `data/raw/.BUSBRA_extracting/` geschrieben. zipfile prüft dabei die CRC jedes Eintrags. Danach wird der Ordner nach `data/raw/BUSBRA/` umbenannt.
- **Wiederholungslauf:** Es wird nichts überschrieben. Alle 3.754 Dateien werden per Größe und CRC32 gegen das Archiv validiert; zusätzliche Dateien würden gemeldet, nicht gelöscht. Weicht etwas ab, bricht das Skript ohne Änderung ab.
- **Ergebnis:**
  - Keine fehlenden Dateien, keine Größen- oder CRC-Abweichungen, keine zusätzlichen Dateien.
  - Kein `BUSBRA/BUSBRA`, kein verbliebener Staging-Ordner.
  - Inhalts-Fingerprint: `3c449221bce1d8ed7de60fb00b7e311428e83074c25a413c04721022d0739610`. Das ist ein SHA-256 über die sortierten Zeilen „Pfad, Größe, CRC32“.

Struktur: `data/raw/BUSBRA/` mit `Images/`, `Masks/`, `bus_data.csv`, `5-fold-cv.csv`, `10-fold-cv.csv` und `LICENSE.txt`.

## 3. Metadaten

Alle CSV-Spalten werden als Text gelesen, ohne automatische NA-Erkennung. Dadurch bleiben ID und Case exakt erhalten (Case z. B. `"856"`, nie `856.0`), und das literale `NaN` der Fold-Dateien bleibt sichtbar.

- **`bus_data.csv`:** 1.875 Zeilen, 10 Spalten (`ID, Case, Histology, Pathology, BIRADS, Device, Width, Height, Side, BBOX`).
- **Kennungen:** 1.875 eindeutige Bild-IDs und 1.064 eindeutige Cases. 253 Cases haben 1 Bild, 811 Cases haben 2 Bilder.
- **Fehlende Werte:** keine. Es gibt keine leeren Zellen und keine NA-artigen Zeichenketten.
- **Kategorien:** keine unerwarteten Werte.
  - Pathology: benign/malignant
  - BIRADS: 2–5
  - Device: die vier dokumentierten Geräte
  - Side: left/right/single
- **ID-Struktur:** Alle IDs folgen `bus_NNNN-[lrs]`. Der numerische Stamm entspricht bei allen 1.875 Zeilen dem Case. Der Suffix entspricht eins zu eins der Spalte Side: l = left (811), r = right (811), s = single (253).

### Klassen und Geräte

| Gerät (laut CSV) | Bilder | Fälle | Benigne Bilder | Maligne Bilder | Maligner Anteil (Bilder) |
|---|---:|---:|---:|---:|---:|
| GE Logiq 5 @10-12MHz | 809 | 490 | 566 | 243 | 30,04 % |
| GE Logiq 7 @10-14MHz | 902 | 488 | 617 | 285 | 31,60 % |
| Toshiba Aplio 300 @12-14 MHz | 139 | 72 | 71 | 68 | 48,92 % |
| U-Systems @10-14MHz | 25 | 14 | 14 | 11 | 44,00 % |
| **Gesamt** | **1.875** | **1.064** | **1.268** | **607** | 32,37 % |

Auf Fallebene sind es 722 benigne und 342 maligne Fälle. Die Klassenanteile unterscheiden sich je Gerät. Gerätemerkmale sind deshalb ein möglicher Shortcut; das ist ein begründetes Risiko, noch kein Nachweis. Quelle: `audit_class_device_counts.csv`.

### BI-RADS × Pathology (Bilder, nur beschreibend)

| BI-RADS | benign | malignant | gesamt |
|---|---:|---:|---:|
| 2 | 552 | 10 | 562 |
| 3 | 432 | 31 | 463 |
| 4 | 280 | 413 | 693 |
| 5 | 4 | 153 | 157 |

BI-RADS ist weder Input noch Ersatzlabel. Maligne Labels bei BI-RADS 2/3 werden nicht umetikettiert. Quelle: `audit_birads_by_pathology.csv`.

## 4. Datei-Zuordnung, Bilder und Masken

- **Zuordnung:** Sie wird pro Metadatenzeile explizit gebildet, nicht durch Zusammenführen sortierter Dateilisten. Bild: `Images/{ID}.png`. Maske: `Masks/mask_{NNNN-x}.png`, z. B. `bus_0001-l` → `mask_0001-l.png`.
  - 1.875 von 1.875 Bildern und 1.875 von 1.875 Masken vorhanden.
  - Keine fehlenden und keine unreferenzierten Dateien, Zuordnung eins zu eins.
- **Dekodierung:** Alle 1.875 Bilder und 1.875 Masken lassen sich mit Pillow ohne Konvertierung dekodieren.
  - Bilder: PNG, Modus `L` (8-Bit-Graustufen, `uint8`), Pixelwerte über alle Bilder 0–255.
  - Masken: PNG, Modus `1`, als NumPy-Array boolesch mit den Werten {0, 1}.
- **Qualität:** kein konstantes Bild, keine leere Maske. Bild und Maske haben bei allen 1.875 Paaren dieselbe tatsächliche Größe.
- **Bildgrößen:** 713 verschiedene Größen.
  - Breite 240–579 px, Median 286 px.
  - Höhe 268–478 px, Median 383 px.
  - Die Spannen sind komponentenweise Minima und Maxima, keine konkreten Eckformate.

## 5. CSV-Abmessungen gegenüber dekodierten Bildern (Warnung)

Bei genau vier IDs weicht die CSV-Höhe von der tatsächlichen Bild- und Maskenhöhe ab. Die Breiten stimmen.

| Bild-ID | Case | Gerät | CSV B × H | Bild B × H | Maske B × H |
|---|---|---|---|---|---|
| bus_0856-l | 856 | U-Systems @10-14MHz | 387 × 405 | 387 × 353 | 387 × 353 |
| bus_0857-r | 857 | U-Systems @10-14MHz | 384 × 409 | 384 × 393 | 384 × 393 |
| bus_0912-l | 912 | GE Logiq 7 @10-14MHz | 360 × 354 | 360 × 352 | 360 × 352 |
| bus_0990-l | 990 | GE Logiq 7 @10-14MHz | 540 × 354 | 540 × 352 | 540 × 352 |

Das sind genau die vier erwarteten IDs; weitere Abweichungen gibt es nicht. Die Originalwerte bleiben im Manifest unverändert in `Width`/`Height`. Die tatsächlichen Größen stehen in `image_width`/`image_height`, gekennzeichnet über `csv_size_mismatch`. Für Laden, Resize und Overlays gelten die tatsächlichen Abmessungen. Die vier Bilder werden deshalb nicht entfernt. Quelle: `audit_csv_dimension_mismatches.csv`.

## 6. Konsistenz innerhalb eines Case

- Pathology, Histology und Device sind innerhalb jedes Case einheitlich.
- **BI-RADS (Warnung):** Bei Case 856 tragen `bus_0856-l` und `bus_0856-r` BI-RADS 4 bzw. 5, beide mit malignant/invasive ductal carcinoma. Das ist eine dokumentierte Annotationsdifferenz und wird nicht korrigiert. Referenzlabel bleibt Pathology. Quelle: `audit_case_consistency.csv`.

## 7. Fold-Dateien

**Beide Dateien** (`5-fold-cv.csv`, `10-fold-cv.csv`):

- Die ID-Mengen stimmen vollständig mit `bus_data.csv` überein. Der Join über ID ist eins zu eins.
- Pathology ist bei allen 1.875 Bildern identisch.
- Jeder der 1.064 Cases liegt in genau einem äußeren Fold. Zwischen den äußeren Folds gibt es keine Case-Überschneidung.

**Innere `valid_j`-Spalten:**

- Sie enthalten nur `1`, `0` und `NaN`.
- `NaN` steht in jedem Durchlauf exakt bei den Zeilen des äußeren Folds j. Das bestätigt die Bedeutung 1 = Training, 0 = Validation, NaN = äußerer Testfold.
- Die äußeren Testfälle überschneiden sich in keinem Durchlauf mit innerem Training oder innerer Validation.
- **Warnung:** In allen 15 inneren Durchläufen liegen Fälle zugleich in Training und Validation.

| Schema | Fälle zugleich in innerem Train und Validation |
|---|---|
| 5-fold | valid_1: 210, valid_2: 201, valid_3: 212, valid_4: 216, valid_5: 211 |
| 10-fold | 221–245 je Durchlauf |

Folgerung: Die äußere Fold-Zuordnung ist fallgetrennt und brauchbar. Die inneren `valid_*`-Spalten erfüllen die geforderte Gruppentrennung nicht und werden **nicht** als Validation-Split verwendet. Quellen: `audit_inner_validation_overlap.csv`, `audit_outer_folds.csv`.

## 8. Technische Vorschau des Split-Vorschlags (nicht eingefroren)

Regel: `5-fold-cv.csv`, äußerer kFold 1 = Test, 2 = Validation, 3–5 = Training.

| Teilmenge | Folds | Bilder | Fälle | Benigne Bilder | Maligne Bilder | Benigne Fälle | Maligne Fälle | Maligner Anteil |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 3, 4, 5 | 1.114 | 639 | 754 | 360 | 434 | 205 | 32,32 % |
| Validation | 2 | 385 | 213 | 260 | 125 | 144 | 69 | 32,47 % |
| Test | 1 | 376 | 212 | 254 | 122 | 144 | 68 | 32,45 % |

- Jedes Bild ist genau einer Teilmenge zugeordnet. Die paarweisen Case-Überschneidungen (Train/Validation, Train/Test, Validation/Test) sind alle 0.
- Es ist ein fixer Holdout auf Basis der offiziellen Folds, keine Cross-Validation.
- Erzeugt wurden nur Zählwerte (`audit_split_proposal_preview.csv`): keine Zuordnungsdatei je Bild, keine Splitspalte im Manifest, keine Modelldateien.
- Der verbindliche Split entsteht erst im freigegebenen Vorbereitungsschritt.

## 9. Duplikate und Unveränderlichkeit der Rohdaten

- **Exakte Duplikate:** keine. Weder identische Bilddateien (SHA-256 der Datei) noch identische dekodierte Pixelarrays (SHA-256 über Form, dtype und Pixel) kommen vor. Quelle: `audit_exact_duplicates.csv`; sie enthält nur die Kopfzeile, weil keine Duplikate gefunden wurden.
- **Grenze dieser Aussage:** Nur exakte Duplikate sind ausgeschlossen. Sehr ähnliche Ansichten, leicht veränderte oder neu kodierte Kopien und nicht dokumentierte Mehrfachzuordnungen derselben Person zu unterschiedlichen Case-IDs sind damit **nicht** ausgeschlossen. Zwei Bilder desselben Case sind erwartete Abhängigkeiten und werden über die Fallgruppierung berücksichtigt.
- **Unveränderte Rohdaten:**
  - Größe und Änderungszeit von Archiv und allen 3.754 entpackten Dateien sind vor und nach der Auditphase identisch; beim Wiederholungslauf auch vor und nach der Validierung.
  - Archiv-MD5, Inhalts-Fingerprint, Manifest und alle CSV-Tabellen waren zwischen Lauf 1 (Entpacken) und Lauf 2 (nur Validierung) identisch.
  - Bilder werden nur lesend geöffnet.

## 10. Manifest `data/processed/manifest_audit.csv`

Das Manifest enthält genau eine Zeile pro Bild (1.875 Zeilen), sortiert nach ID. **Beim Einlesen `dtype={"ID": str, "Case": str}` setzen.** Ein Round-Trip-Test hat bestätigt, dass IDs und Cases als Text erhalten bleiben.

| Spaltengruppe | Spalten | Bedeutung |
|---|---|---|
| Kennung und Target | `ID`, `Case`, `Pathology`, `label` | `label`: benign = 0, malignant = 1 (explizites Mapping; 1.268 × 0, 607 × 1) |
| Original-Metadaten, unverändert | `Device`, `Histology`, `BIRADS`, `Width`, `Height`, `Side`, `BBOX` | nur beschreibend; **keine Modellinputs** |
| Dateizuordnung | `image_relpath`, `mask_relpath` | projektrelative Pfade |
| Dekodierte Eigenschaften | `image_format`, `image_mode`, `image_width`, `image_height`, `image_min`, `image_max`, `mask_format`, `mask_mode`, `mask_width`, `mask_height`, `mask_values`, `mask_foreground_pixels` | tatsächlich gelesene Werte |
| Prüf- und Abweichungsflags | `image_is_constant`, `mask_is_empty`, `image_mask_size_match`, `csv_width_mismatch`, `csv_height_mismatch`, `csv_size_mismatch` | Integritätsbefunde |
| Hashes | `image_file_sha256`, `image_pixel_sha256`, `mask_file_sha256` | Integrität und exakte Duplikate |

**Inputvertrag:** Modellinput ist ausschließlich das Bild unter `image_relpath`. Case dient nur der Gruppentrennung. Device dient nur der späteren deskriptiven Subgruppenanalyse. Masken dienen nur der Integritätsprüfung und als Referenz bei der Interpretation, nicht der Modellauswahl. BBOX und Masken werden nie zum Zuschneiden verwendet. Das Manifest enthält keine Bildtransformationen, keine aus Masken erzeugten Inputs und keine Splitspalte.

## 11. Nicht geprüft bzw. offene Risiken

- **Eingebrannte Texte, Mess- und Gerätemarker:** Aus der Projektvorbereitung ist ein mögliches Shortcut-Risiko übernommen, das hier nicht selbst überprüft wurde. In diesem Schritt fand **keine Sichtprüfung** statt; das Skript bewertet oder entfernt die Marker nicht. Es gibt keine OCR und kein Inpainting. Eine eigene Sichtprüfung an einer Trainingsstichprobe ist in Schritt 2A vorbereitet und steht noch aus (siehe `docs/train_eda.md`).
- **Keine klinische Nachbegutachtung** von Labels oder Masken.
- **Gerätegruppen** stammen aus der CSV und wurden nicht unabhängig verifiziert.
- **Lizenzangabe CC BY 4.0** stammt aus den Zenodo-Metadaten (online eingesehen), nicht aus dem Archiv selbst.

## 12. Wiederholung

```bash
.venv/bin/python analysis/01_prepare_data.py   # im Projektordner BUSBRA; Exit-Code 0 = keine Fehler
```

Fehlt `data/raw/BUSBRA.zip`, nennt das Skript den erwarteten Pfad und den Downloadlink (https://zenodo.org/records/8231412/files/BUSBRA.zip?download=1) und bricht ab. Bei abweichender Größe oder MD5 bricht es vor dem Entpacken ab.
