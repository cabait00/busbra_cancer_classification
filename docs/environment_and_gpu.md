# Umgebung, GPU und Projektstart

> Hinweis: Das beschriebene Umgebungsprüfskript (`00_environment_check.py`) gehört zum historischen Ablauf und liegt nicht mehr im Projektordner; die Anforderungen an Pakete und GPU stehen in `requirements.txt` und in diesem Dokument.

Stand der Angaben: 17.–19.10.2026 (Prüfungen der Umgebung); dieses Dokument bewahrt die Beschreibung der Umgebung auf, die früher im README stand. Kurzfassung und Standardeinstieg: [README.md](../README.md) und [reproducibility_and_handover.md](reproducibility_and_handover.md).

## Projektstart (Terminal und VS Code)

Projektwurzel ist der Ordner `BUSBRA`; alle Pfade und Befehle beziehen sich auf ihn.

**Terminal:**

```bash
cd <Pfad zu>/3_Projekt/BUSBRA
echo "${LD_LIBRARY_PATH:-<nicht gesetzt>}"    # darf nicht auf andere Umgebungen zeigen
env -u LD_LIBRARY_PATH .venv/bin/python analysis/00_environment_check.py
```

Kontrolle nach dem Lauf: Exit-Code 0, und `reports/tables/environment_check.txt` enthält

- `Status: PASSED`,
- `Interpreter: <project>/.venv/bin/python (...), project .venv: True`,
- `GPU discovery: ['/physical_device:GPU:0']`,
- `CUDA libraries loaded from outside .venv: none`.

**VS Code:**

- Ordner `BUSBRA` öffnen. Ist stattdessen `3_Projekt` geöffnet, liegt das Projekt im Unterordner `BUSBRA`.
- Interpreter: „Python: Select Interpreter“ → „Enter interpreter path“ → `.venv/bin/python`. Bei geöffnetem `3_Projekt` lautet der Pfad `BUSBRA/.venv/bin/python`.
- Skripte im integrierten Terminal mit den Befehlen oben ausführen. Das integrierte Terminal lädt das Shell-Profil und kann dadurch `LD_LIBRARY_PATH` aus `dl-keras` erben; `env -u LD_LIBRARY_PATH` bleibt dort nötig.
- Interactive Window (`# %%`-Zellen): Die Skripte bestimmen die Projektwurzel über ihren Dateipfad. Ist dieser nicht verfügbar, suchen sie ab dem Arbeitsverzeichnis aufwärts nach `analysis/`, `README.md` und `requirements.txt`. Das Arbeitsverzeichnis muss dann in `BUSBRA` oder `BUSBRA/analysis` liegen; von `3_Projekt` aus wird das Projekt nicht gefunden, und das Skript bricht mit einer Meldung ab.

**Umgebung des Interactive-Window-Kernels – erledigt am 19.10.2026:** Die Interpreterwahl allein belegt nicht, dass der Kernel ohne die geerbte Variable `LD_LIBRARY_PATH` startet; ebenso belegt eine sichtbare GPU keine GPU-Rechenoperationen. Deshalb führt der vollständige Umgebungs-Check in einer Zelle des tatsächlich verwendeten Kernels aus (Arbeitsverzeichnis `BUSBRA`); seine Kindprozesse erben die Kernel-Umgebung:

```python
%run analysis/00_environment_check.py
```

Carlo hat das am 19.10.2026 um 19:01:54 ausgeführt. Der lokale Bericht `reports/tables/environment_check.txt` zeigt `Status: PASSED`, den Projektinterpreter, die bestandenen GPU-Operationen und `CUDA libraries loaded from outside .venv: none`. Das ist ein Nachweis für die Umgebung dieses Kernels zu diesem Zeitpunkt, kein Nachweis eines erfolgreichen Trainings. Für das Training im Interactive Window gilt zusätzlich: ganzes Skript in einem **frischen** Kernel ausführen; ein Kernel, der die GPU schon initialisiert hat, kann Memory Growth nicht mehr setzen, und `04_train_small_cnn.py` bricht dann mit einem Hinweis ab. Der Terminal-Aufruf (`env -u LD_LIBRARY_PATH ...`) ersetzt weder den Check noch umgekehrt. Nach Änderungen an Umgebung oder Kernel den Check wiederholen und auf einen aktuellen PASSED-Bericht achten.

## Umgebung und GPU

**Geprüftes System:** Linux Mint 22.3 (Ubuntu-Noble-Basis), nativ, also kein WSL, Container oder Remote-Zugriff. NVIDIA GeForce RTX 3050 Laptop GPU mit 4096 MiB, Treiber 595.91.07, Compute Capability 8.6. Es gibt kein systemweites CUDA-Toolkit (`/usr/local/cuda` fehlt). Der Treiber stellt `libcuda` bereit; alle übrigen CUDA/cuDNN-Laufzeitbibliotheken kommen aus den `nvidia-*`-Paketen in der `.venv`. Das zeigt im finalen Lauf die Liste der geladenen Bibliotheken.

**Versionen – Build-Angabe gegenüber installierter Laufzeit:**

- *Build-Angabe:* TensorFlow 2.20.0 meldet über `tf.sysconfig.get_build_info()` CUDA 12.5.1 und cuDNN 9. Die offizielle Tabelle getesteter Builds nennt für 2.20.0 CUDA 12.5 und cuDNN 9.3.
- *Installiert und geladen:* die von pip für `tensorflow[and-cuda]==2.20.0` aufgelösten Laufzeitpakete, u. a. `nvidia-cuda-runtime-cu12` 12.9.79, `nvidia-cublas-cu12` 12.9.2.10, `nvidia-cudnn-cu12` 9.27.0.42 und `nvidia-cusolver-cu12` 11.7.5.82. Die vollständige Liste steht in `docs/environment_freeze.txt` und `environment_check.json`.
- Diese Laufzeit weicht in den Nebenversionen von der getesteten Build-Konfiguration ab. Ihre lokale Funktion ist nur durch den Smoke-Test belegt.

**Projektumgebung:** `.venv/` im Projektordner.

- Basisinterpreter: Python 3.13.14 aus der conda-Umgebung `py`. Diese conda-Umgebung wurde nicht verändert. Die `.venv` funktioniert aber nur, solange dieser Interpreter dort existiert. Wird `py` gelöscht oder auf eine andere Python-Minorversion umgestellt, muss die `.venv` neu erstellt werden.
- Warum nicht der System-Python 3.12: Ihm fehlt `ensurepip` (Paket `python3.12-venv`), dessen Installation sudo erfordert. Die conda-base-Umgebung hat Python 3.14, wofür kein TensorFlow-Wheel existiert.
- **VS-Code-Interpreter:** `.venv/bin/python`, siehe „Projektstart“.

**TensorFlow 2.20.0 statt 2.21.0:**

- *Beobachtung:* 2.21.0 war am 17.10.2026 die aktuelle Version und wurde zuerst installiert. Ohne `LD_LIBRARY_PATH` konnte TensorFlow `libcusolver.so.11` nicht laden („cannot open shared object file“). Diese Meldung stammt aus einem nicht archivierten Diagnoselauf mit `TF_CPP_MAX_VLOG_LEVEL=1`. TensorFlow registrierte daraufhin keine GPU; das ist archiviert in `environment_check_attempts/run2_tf2.21.0_without_ld_library_path.*`. Mit 2.20.0 besteht der Smoke-Test.
- *Lokale Diagnose der Ursache* (Protokoll mit Befehlen, Dateinamen und SHA-256: früherer Bestand, heute extern gesichert): Die Bibliothek mit dem cuSOLVER-Loader ist `libtensorflow_framework.so.2`.
  - Im geprüften Wheel `tensorflow-2.21.0-cp313-cp313-manylinux_2_27_x86_64.whl` enthält ihr RUNPATH kein `nvidia/cusolver`; dieser Eintrag steht dort nur in `libtensorflow_cc.so.2`.
  - Im installierten 2.20.0 enthält der RUNPATH von `libtensorflow_framework.so.2` `nvidia/cusolver`.
- *Grenzen:* Die Aussage gilt für diese beiden Dateien (CPython 3.13, Linux x86_64); andere 2.21.0-Builds wurden nicht geprüft. Dass das am 18.10.2026 geprüfte Wheel dieselbe Datei ist, die am 17.10. installiert war, ist zu erwarten, aber nicht belegt; damals wurde kein Hash erfasst.
- 2.20.0 ist eine stabile Release. Die offizielle Tabelle getesteter Builds nennt dafür Python 3.9–3.13, CUDA 12.5 und cuDNN 9.3; die pip-Installationsanleitung nennt Treiber ≥ 525.60.13.

**Wichtig – `LD_LIBRARY_PATH`:** In der bisher verwendeten Shell (Stand 17./18.10.2026) ist die conda-Umgebung `dl-keras` aktiv. Sie setzt `LD_LIBRARY_PATH=~/.conda/envs/dl-keras/.../nvidia/cusolver/lib`. Dann lädt TensorFlow `libcusolver`, `libcusparse` und `libnvJitLink` aus dieser fremden Umgebung, und die GPU-Prüfung schlägt bewusst fehl. Für Projektbefehle deshalb:

```bash
echo "$LD_LIBRARY_PATH"     # sollte leer sein
# andernfalls dem Befehl voranstellen:
env -u LD_LIBRARY_PATH .venv/bin/python analysis/00_environment_check.py
```

Die Umgebung `dl-keras` und das Shell-Profil bleiben unverändert. Für den Interactive-Window-Kernel siehe „Projektstart“.

**Speicherbeobachtung:** Beim Anlegen von GPU:0 schreibt TensorFlow die Logzeile „Created device … with 2142 MB memory“. `nvidia-smi` meldete am 17.10.2026 vor dem Start 4096 MiB gesamt und 3756 MiB frei. Die 2142 MB sind eine beobachtete TensorFlow-Logangabe. Das Projekt setzt keine Speichergrenze, sondern nur Memory Growth. Der Wert ist weder ein Laufzeitmaß noch eine Garantie für eine bestimmte Batchgröße.

## Installation (tatsächlich ausgeführt)

Im Projektordner `BUSBRA`:

```bash
~/miniconda3/envs/py/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "tensorflow[and-cuda]==2.21.0" pandas pillow ipykernel   # erster Versuch
.venv/bin/python -m pip install "tensorflow[and-cuda]==2.20.0"                            # Korrektur, siehe oben
```

Für eine Neuinstallation mit identischem Paketstand ist dieser Befehl vorgesehen. Er wurde nur als `--dry-run` gegen die bestehende `.venv` geprüft (alle Anforderungen erfüllt, keine Konflikte), nicht als Neuinstallation in einer leeren Umgebung:

```bash
.venv/bin/python -m pip install -r requirements.txt -c docs/environment_freeze.txt
```

Ein Aktivieren der Umgebung ist nicht nötig, alle Befehle rufen `.venv/bin/python` direkt auf. Wer aktivieren möchte: `source .venv/bin/activate`.

## Daten

**Quelle:** Originalarchiv `BUSBRA.zip` aus dem Zenodo-Record https://zenodo.org/records/8231412 (DOI 10.5281/zenodo.8231412, Version 1.0).

- Datei: 133.917.740 Bytes, MD5 `1f8b2be6476d58fc97bfb5e5a1ea9bab`.
- Ablage: unverändert unter `data/raw/BUSBRA.zip`. Das Skript `01_project_foundation.py` prüft Größe und MD5 und entpackt erst danach nach `data/raw/BUSBRA/`.

**Lizenz:**

- Laut Zenodo-Metadaten gilt CC BY 4.0.
- Das Archiv enthält zusätzlich eine eigene `LICENSE.txt` mit einem permissiven, MIT-ähnlichen Erlaubnistext und Zitierpflicht. Sie ist nicht wortgleich mit CC BY 4.0.
- Beide Hinweise bleiben erhalten und werden getrennt dokumentiert.

**Zu zitieren:** W. Gómez-Flores, M. J. Gregorio-Calas, W. Coelho de Albuquerque Pereira: „BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems“, *Medical Physics* 51, 3110–3123 (2024), https://doi.org/10.1002/mp.16812. Die `LICENSE.txt` im Archiv nennt noch den älteren Stand „Medical Physics, 2023 (In revision)“.

Rohdaten, Manifeste, die Kontaktübersicht, Kursarchive und die `.venv` sind per `.gitignore` von einer Versionierung ausgeschlossen. Medizinische Bilder und daraus erzeugte Übersichten werden nicht an externe Dienste übertragen.


## Quellen

- BUS-BRA, Zenodo: https://zenodo.org/records/8231412 (Metadaten: https://zenodo.org/api/records/8231412)
- Publikation: https://doi.org/10.1002/mp.16812
- Autoren-Repository: https://github.com/wgomezf/BUS-BRA
- TensorFlow-Installation (pip): https://www.tensorflow.org/install/pip
- Getestete TensorFlow-Build-Konfigurationen: https://www.tensorflow.org/install/source#gpu
- TensorFlow GPU-Leitfaden: https://www.tensorflow.org/guide/gpu
