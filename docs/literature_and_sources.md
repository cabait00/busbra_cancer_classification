# Literatur, Quellen, Masken und Rechte

Stand: 21.10.2026. Dieses Dokument belegt, welche Angaben zu Datensatz, Masken, Lizenzen und Vergleichsstudien an der Quelle geprüft wurden, und trennt sie von Angaben, die nicht geprüft werden konnten. Die Projektdokumentation ([BUSBRA_Projektdokumentation.md](BUSBRA_Projektdokumentation.md)) verweist im Anhang hierher; einschlägig sind ihre Abschnitte 3 (Daten) und 8.3 (Einordnung in die Literatur). Es gab keine systematische Literatursuche; die Auswahl umfasst die Datensatzbeschreibung, die Methodenarbeiten der verwendeten Verfahren und zwei weitere Studien mit BUS-BRA.

## 1 Was an der Quelle geprüft wurde

| Quelle | Geprüft am 21.10.2026 | Ergebnis | Nicht geprüft |
|---|---|---|---|
| Zenodo-Record 8231412 (Programmierschnittstelle des Records) | Titel, Version, DOI, Lizenz, Autorinnen und Autoren, Dateiliste mit Größe und MD5 | Version 1.0 (Record-Revision 5), veröffentlicht 13.03.2023, DOI 10.5281/zenodo.8231412, Lizenz `cc-by-4.0`, Datei `BUSBRA.zip` mit 133.917.740 Bytes und MD5 `1f8b2be6476d58fc97bfb5e5a1ea9bab` (identisch mit dem lokal geprüften Archiv); die Beschreibung nennt 1.064 Patientinnen, BI-RADS 2 bis 5, biopsiegesicherte Tumoren und Grundwahrheits-Umrisse; sie nennt keine Aufteilung | – |
| Gómez-Flores, Gregorio-Calas, Coelho de Albuquerque Pereira, *Medical Physics* 51(4), 3110–3123 (2024), [doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812) | Metadaten und Kurzfassung über die Literaturdatenbank Europe PMC (PMID 37937827) | 1.875 anonymisierte Bilder von 1.064 Frauen, vier Ultraschallgeräte am Nationalen Krebsinstitut in Rio de Janeiro, 722 gutartige und 342 bösartige biopsiegesicherte Fälle, BI-RADS 2–5 durch eine erfahrene Fachkraft („senior ultrasonographer“), manuelle Läsionsumrisse, 5- und 10-fach-Partitionen sowie ein Auswertungsrahmen mit tiefen Netzen für Segmentierung und Klassifikation | Volltext (nicht abrufbar); deshalb werden Ergebniswerte der Arbeit nicht wiedergegeben |
| Autorencode [github.com/wgomezf/BUS-BRA](https://github.com/wgomezf/BUS-BRA) (`BUSBRA-Codes.zip`, SHA-256 `6d42515063fea715a5d9599ade45d4a97b3d4eaa09e58e9c73d4287301e4e1c2`, am 21.10.2026 heruntergeladen, nur gelesen, nicht ausgeführt) | `Classification/cls_busdataset.m`, `cls_partitions.m`, `cls_parameters.m`, `cls_evaluation.m`; Beschreibung des Repositorys | ResNet18/ResNet50, Eingabe 224 × 224, 5- oder 10-fach-Validierung nach den mitgelieferten Fold-Dateien (`kFold` als Testfold, `valid_k` für Training und Validation); **für alle Bilder** wird aus der Grundwahrheitsmaske die Box (+10 Pixel je Seite) bestimmt, der Ausschnitt kontrastgestreckt und aus Graubild, maskiertem Graubild und Maske ein Dreikanalbild gebildet; Klassengewichte werden aus den Häufigkeiten über den gesamten Datensatz berechnet; Repository-Lizenz GPL-3.0 | Ergebnisse des Codes (im Repository keine Zahlen), Auswertung je Bild oder je Fall (nicht untersucht) |
| Abbadi et al. 2025, Preprint [arXiv:2509.05004](https://arxiv.org/abs/2509.05004) | Zusammenfassung und HTML-Fassung | BUS-BRA wird als „1.875 Bilder von 1.064 Patienten“ beschrieben; Quelle und Version nicht genannt; Aufteilung 80/10/10, „sofern der Datensatz offizielle Aufteilungen mitbringt (z. B. BUS-BRA)“ nicht aufgeschlüsselt; ROI-Extraktion „wenn Masken verfügbar“ (genannt: BUSI); Tabelle I fasst BUSI und BUS-BRA zusammen (ResNet-18: Accuracy 99,7 %, Precision 0,99, Recall 1,00, F1 0,995, AUC 0,998) | ob Fälle getrennt wurden, Auswertungseinheit, Klassenzuordnung für BUS-BRA (Hauptaufgabe ist dreiklassig), Begutachtungsstatus des Preprints |
| Nastase, Moldovanu, Biswas, Moraru, *Scientific Reports* 14, 22754 (2024), [doi:10.1038/s41598-024-74316-5](https://doi.org/10.1038/s41598-024-74316-5) | Metadaten und Kurzfassung (Europe PMC), Volltext (PubMed Central, PMC11448494; Lizenz CC BY-NC-ND) | BUS-BRA für Training und Test (zwei Klassen), BUSI als externer Test; Ausgangsbilder, mit den mitgelieferten Masken erzeugte ROI-Eingaben sowie erodierte und dilatierte Varianten (10 Pixel); MobileNetV2, VGG16, EfficientNetB7; **zufällige Aufteilung 70 % / 30 % auf Bildebene** („randomly assigned“), keine Aussage zur Trennung von Patientinnen; AUC bis 1,00 für einzelne Varianten (zum Beispiel MobileNetV2, erodiert); Rohbilder „am schlechtesten“ ohne Zahlen; die Einstellungen (Optimierer, Lernrate, Epochen) sind an mehreren Stellen des Textes uneinheitlich angegeben | ob die Maske bei der Anwendung (insbesondere auf BUSI) ausdrücklich vorausgesetzt wird (im Text nicht eindeutig); Auswertungseinheit |
| Methodenarbeiten: MobileNetV2 ([arXiv:1801.04381](https://arxiv.org/abs/1801.04381)), U-Net ([arXiv:1505.04597](https://arxiv.org/abs/1505.04597)), Grad-CAM ([arXiv:1610.02391](https://arxiv.org/abs/1610.02391)), FGSM ([arXiv:1412.6572](https://arxiv.org/abs/1412.6572)), Adam ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980)) | Titel, Autoren, Veröffentlichungsort und Kennung | stimmen mit den Angaben im Bericht überein | Volltexte nicht im Einzelnen nachgelesen; die Verfahren werden nur zitiert, nicht reproduziert |

**Nicht übernommene Fundstellen.** Eine Suchmaschine lieferte weitere Treffer, die an der Quelle nicht bestätigt werden konnten (zum Beispiel ein Preprint zu einem Ultraschall-Benchmark, dessen Seite BUS-BRA nicht erwähnt, sowie eine Bestenliste ohne Protokollangaben). Sie werden nicht zitiert. Weitere Seiten (die Verlagsseite der Datensatzbeschreibung, PubMed und eine MDPI-Seite) waren für den Abruf gesperrt; ihre Inhalte werden weder wiedergegeben noch bewertet.

## 2 Vergleichstabelle (BUS-BRA-Studien)

Die Tabelle steht auch im Bericht (Abschnitt 8.3). „Nicht berichtet / unklar“ bedeutet, dass die Angabe in den eingesehenen Quellen fehlt oder nicht einsehbar war. Es gibt kein Ranking; ein Zahlenvergleich ist wegen der unterschiedlichen Informationsbedingungen und Aufteilungen nicht sinnvoll.

| Merkmal | Gómez-Flores et al. 2024 (Datensatz) und Autorencode | Abbadi et al. 2025 (Preprint) | Nastase et al. 2024 | Diese Arbeit |
|---|---|---|---|---|
| **Quelle** | *Med. Phys.* 51:3110, [doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812) | [arXiv:2509.05004](https://arxiv.org/abs/2509.05004) | *Sci. Rep.* 14:22754, [doi:10.1038/s41598-024-74316-5](https://doi.org/10.1038/s41598-024-74316-5) | dieser Bericht |
| **Aufgabe** | Segmentierung; Klassifikation (Pathologie, BI-RADS) | Klassifikation über BUSI, BUS-BRA und BrEaST-Lesions USG (Hauptaufgabe laut Text dreiklassig; Klassenzuordnung für BUS-BRA nicht bestätigt) | zweiklassige Klassifikation; Vergleich Vollbild, ROI, erodierte und dilatierte ROI; externer Test auf BUSI | Pathologie gutartig / bösartig auf Bild- und Fallebene |
| **Datenversion** | BUS-BRA (1.875 Bilder, 1.064 Patientinnen) | laut Text 1.875 Bilder von 1.064 Patientinnen; Quelle und Version nicht angegeben | BUS-BRA (Training und Test), BUSI (extern) | BUS-BRA Version 1.0 (Zenodo) |
| **Aufteilung** | 5- und 10-fach-Folds mitgeliefert, vom Code verwendet | „offizielle Splits, wo vorhanden (z. B. BUS-BRA)“, sonst 80/10/10 | zufällig 70 % / 30 % auf Bildebene | fester Holdout aus äußeren Folds (3–5 / 2 / 1) |
| **Trennung Bild / Patientin / Fall** | äußere Folds fallgetrennt, innere `valid_*`-Spalten mit Überlappung (eigene Prüfung) | nicht berichtet | **nicht berichtet**; zufällige Bildzuordnung, Patientinnen können in beiden Teilen vorkommen | Fälle strikt getrennt |
| **Masken- bzw. ROI-Zugriff bei der Anwendung** | Autorencode (`cls_busdataset.m`): für alle Bilder Zuschnitt um die Maskenbox (+10 Pixel) und Kanäle aus Graubild, maskiertem Graubild und Maske; die Maske wird auch für Testbilder benötigt | ROI-Extraktion, wo Masken vorhanden (BUSI); für BUS-BRA unklar | ROI-Eingaben aus den mitgelieferten Masken; Anwendung im Text nicht ausdrücklich geklärt | Vollbild und automatische ROI: nur Bild; Experten-ROI: manuelle Maske |
| **Auswertungseinheit** | im Code je Bild; weitere Details nicht geprüft | nicht berichtet | Bild (nicht ausdrücklich genannt) | Bild und Fall (Maximum der Bild-Scores) |
| **Metrik** | im Volltext berichtet, hier nicht eingesehen | Accuracy, Precision, Recall, F1, AUC | Accuracy, AUC | ROC-AUC, AP, Sensitivität, Spezifität, Fehlalarme |
| **Ergebnis** | nicht verifiziert (Volltext nicht abrufbar), daher nicht wiedergegeben | BUSI und BUS-BRA zusammengefasst (ResNet-18: Accuracy 99,7 %, AUC 0,998); nicht getrennt, nicht vergleichbar | AUC bis 1,00 je Variante; Rohbilder am schlechtesten (ohne Zahlen); Einstellungen im Text uneinheitlich | Bericht, Abschnitt 6 |

## 3 Zweck der Masken und ihre Verwendung

**Offizieller Zweck.** Die Masken sind manuell gezeichnete Läsionsumrisse („Grundwahrheit“, die Bilder in tumoröse und normale Regionen teilt). Der Datensatz ist laut Kurzfassung der Datensatzbeschreibung für Läsionserkennung und Segmentierung, Pathologie- und BI-RADS-Klassifikation, Entrauschen (Speckle), Kontrastverbesserung und Merkmalsentwicklung gedacht. Auch gutartige Läsionen haben eine Maske. Eine Maske ist deshalb weder ein Krebslabel noch eine Markierung gesunden Gewebes.

**Verwendung in diesem Projekt.**

| Verwendung | Zweig | Einfluss auf die Vorhersage |
|---|---|---|
| Integritätsprüfung (Kodierung, Größe, nicht leer, Zuordnung zur ID) | alle | keiner |
| Trainingsziel und Messreferenz der Segmentierung (U-Net); die Validation-Masken steuern Validation-Loss und Checkpoint | automatische ROI | die Maske wird gelernt, nicht eingegeben |
| Grundlage des Zuschnitts (Box + 20 %) | Experten-ROI | **ja, in Training, Validation und Test** (die Maske lokalisiert; sie ist kein Kanal und blendet nichts aus) |
| Referenzkontur, Dice und Zuschnitts-Diagnose nach der Auswertung (Test) | Interpretation | keiner |
| Eingabe in den Vollbildzweig | Vollbild | keine |

Die Klassifikatoren lernen ausschließlich die Pathologie-Labels; das U-Net lernt die Läsionsmarkierung. Die Metadaten Fall, Histologie, BI-RADS, Gerät, Seite und Bounding-Box-Angaben sind keine Modelleingaben.

## 4 Rechte und Lizenzen

| Gegenstand | Befund | Folge im Projekt |
|---|---|---|
| Datensatz (Bilder, Masken, Tabellen) | Zenodo-Metadaten: CC BY 4.0. Das Archiv enthält zusätzlich `LICENSE.txt` (Copyright 2023, SHA-256 `e9e3ea607a53575d9d17c07cdb35972d9506a19dcafde49383fb0f875f01b92d`): permissiver, MIT-ähnlicher Erlaubnistext, der die Nennung der Publikation und den Erhalt des Hinweises verlangt; er ist nicht wortgleich mit CC BY 4.0 und nennt die Publikation noch als „in revision“ | Beide Texte bleiben erhalten; es wird die veröffentlichte Fassung (2024) zitiert; die Rohdaten sind nicht Teil des Repositories |
| Ultraschallbilder und Masken | werden in Bericht und Dokumentation nicht abgebildet | Im öffentlichen Repository gibt es nur Tabellen und Statistikdarstellungen; eine Veröffentlichung von Bildern bräuchte eine eigene Prüfung der Rechtsgrundlage |
| Abbildungen des Berichts | eigene Statistikdarstellungen und ein Schema aus den Projekttabellen; keine Abbildungen aus Veröffentlichungen | keine Rechte Dritter berührt |
| Text der Veröffentlichungen | nur zitiert; die Angaben in den Tabellen sind eigene Zusammenfassungen | – |
| Autorencode | GPL-3.0 laut Repository | nur gelesen, nicht verwendet und nicht in das Repository aufgenommen |
| Vortrainierte MobileNetV2-Gewichte (ImageNet) | lokale Datei, per SHA-256 identifiziert; **Lizenz der Gewichtsdatei nicht geprüft** | Die Modell-Checkpoints (die die Gewichte enthalten) sind nicht Teil des Repositories; ihre Weitergabe wäre nach einer Lizenzprüfung zu entscheiden |
| Eigene Modelle (Checkpoints) | aus CC-BY-Daten und ImageNet-Gewichten abgeleitet | siehe oben |
| Kursmaterial | lokal vorhanden, nicht Teil des Repositories | – |
| Eigener Quellcode | MIT-Lizenz, Copyright (c) 2026 Carlo Babo, siehe [LICENSE](../LICENSE) | gilt nur für den eigenen Quellcode im Repository; keine Rechte an Daten, Gewichten, Bibliotheken oder Publikationen Dritter; die Berichte und Dokumente werden dadurch nicht zusätzlich lizenziert |
