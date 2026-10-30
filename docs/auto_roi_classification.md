# Schritt 4F-B – automatische Segmentierung → ROI → Klassifikation (Inferenz auf Validation)

Stand: 20.10.2026. Status: umgesetzt, ausgeführt, lokal geprüft; explorativ. Zahlen stammen aus den gespeicherten Tabellen `reports/tables/auto_roi_*.csv` und den Statusdateien unter `reports/tables/auto_roi/`.

**Kernaussage.** Der gespeicherte beste U-Net-Checkpoint aus 4F-A erzeugt auf den 385 Validation-Bildern automatische Zuschnitte, die die zehn **unveränderten** 4E-Klassifikatoren (fünf Seeds × Frozen-Kopf / erweitertes Fine-Tuning 1e-4) ohne manuelle Maske verwenden können. Im Mittel über die fünf Seeds liegt die Validation-ROC-AUC beim erweiterten Fine-Tuning bei 0,881 (Vollbild, 4B), 0,921 (Experten-ROI, 4E) und 0,898 (automatische ROI); beim Frozen-Kopf bei 0,797, 0,892 und 0,870. Die automatische ROI liegt gepaart in **allen** fünf Seeds beider Stufen unter der Experten-ROI (Differenz −0,0229 beim Fine-Tuning, −0,0220 beim Frozen-Kopf) und über dem Vollbild (Frozen +0,0723 in 5 von 5 Seeds positiv; Fine-Tuning +0,0170, 4 von 5 positiv). Bei den Sensitivitätszielen ist das Bild uneinheitlich (siehe unten). Von 385 Zuschnitten sind 6 ein Vollbild-Fallback (leere Vorhersage) und bei 33 von 379 der übrigen liegt die Expertenmaske nicht vollständig im Zuschnitt (4 davon ohne ein einziges Expertenmaskenpixel). Das sind beschreibende Ergebnisse eines explorativen Vergleichs auf wiederholt verwendeten Validation-Daten: keine Signifikanz-, Überlegenheits- oder Eignungsaussage, keine Modell- und keine Schwellenwahl, kein Testzugriff.

## Zweck, Einordnung und Abgrenzung

- **Was geprüft wird.** Funktioniert die Kette „Bild → gespeichertes U-Net (4F-A) → größte vorhergesagte Komponente → Zuschnitt nach der festen 20-%-Regel → 4E-Klassifikator“ ohne manuelle Maske, und wie nahe kommt sie dem Experten-ROI-Ergebnis (4E) und dem Vollbildergebnis (4B)? Verglichen werden auf **denselben** 385 Validation-Bildern (213 Fälle; 125 maligne und 260 benigne Bilder) je Seed und Stufe: (1) Vollbild (4B, **nur aus gespeicherten Vorhersagen gelesen**), (2) Experten-ROI (4E, **nur aus gespeicherten Vorhersagen gelesen**), (3) automatische ROI (**neu inferiert** mit demselben gespeicherten 4E-Checkpoint).
- **Kein neues Training, keine Veränderung.** Kein Checkpoint wird verändert, neu trainiert oder ersetzt; die zehn 4E-Checkpoints und der 4F-A-Checkpoint werden ausschließlich zur Inferenz geladen (Hashes vor dem Laden und nach der Inferenz geprüft, Gewichte vor und nach der Inferenz bitgleich). Der 4F-A-Plan und der eingefrorene Vertrag `roi-expert-mask-v1` bleiben unverändert; es gibt keine neue Segmentierungsschwelle, Randbreite, Architektur, Augmentation, Seed-Auswahl oder Klassifikationsschwelle. Testbilder, Testmasken und Testlabels bleiben gesperrt (die 376 Testzeilen des Manifests werden nur gezählt und verworfen).
- **Wichtige Einordnung.** Die 4E-Klassifikatoren wurden auf **Experten-ROIs trainiert**. 4F-B ändert ihre **Eingabe bei der Anwendung**; es ist **kein** auf automatisch erzeugten ROIs neu trainierter Klassifikator. Das U-Net wurde mit Expertenmasken trainiert und mithilfe der Validation-Bilder ausgewählt; die automatische Pipeline braucht bei der Inferenz keine manuelle Maske, ihre Validation-Ergebnisse sind aber wegen der wiederholten Nutzung dieser Daten **explorativ**. Fünf Seeds der Klassifikatoren sind keine fünf unabhängigen Datensätze (derselbe Holdout); für die Segmentierung gibt es nur einen Seed. Keine klinische Eignung wird behauptet.
- **Nachträgliche Sichtprüfung der 4F-A-Vorschauen** (Dokumentation der Sichtprüfung, die erst nach dem 4F-A-Lauf erfolgte; vor dem Lauf gab es keine Sichtung): Die repräsentativen Beispiele waren meist passend; die gesondert markierte Fehleranalyse zeigt echte leere und falsch lokalisierte Vorhersagen. Das ist keine fachärztliche Prüfung und gilt nur für die in den Vorschauen gelisteten IDs. Die hier gemessenen Fallbacks und Fehllokalisationen (unten) bestätigen diesen Befund zahlenmäßig, ersetzen die Sichtung aber nicht. Die Rundreise der Expertenmaske aus 4F-A ist eine **Rasterungsreferenz**, keine strenge mathematische Obergrenze des erreichbaren Dice.

## Festes Verfahren (vor dem ersten Validation-Bild im Studienplan registriert)

Studie `auto4fb-8e64dc98`, Plan-SHA-256 `8e64dc984e7dedbd…` (`reports/tables/auto_roi_study_config.json`; registriert 2026-10-20T21:49:33+02:00, nach der Prüfung aller Quellen per Hash und **bevor** ein Validation-Bild geöffnet oder ein neuer Score berechnet wurde). Der Plan enthält die Hashes aller zehn 4E- und zehn 4B-Läufe (Status, Vorhersagen, Config, Checkpoint), der 4F-A-Kette (Plan, Datenprüfung, Lauf-Status, Config, History, Einzelbildergebnisse, Checkpoint), des ROI-Vertrags und der Code-Module, die Vergleichsgruppen, die Geometrieregel, die Zielgrößen, die Diagnosedefinitionen, die Toleranz und die Zugriffsgrenzen; er enthält keinen Zeitstempel.

| Gruppe (je Seed und Stufe) | Quelle | Klassifikator-Checkpoint |
|---|---|---|
| Vollbild | 4B, gespeicherte Validation-Vorhersagen (nicht neu berechnet) | 4B-Läufe (nicht geladen) |
| Experten-ROI | 4E, gespeicherte Validation-Vorhersagen (nicht neu berechnet) | der 4E-Lauf |
| automatische ROI | **neu inferiert** | **derselbe** unveränderte 4E-Lauf |

**Automatische ROI** (exakt der in 4F-A registrierte Übergang):

1. Bild mit der unveränderten 4F-A-Vorverarbeitung (256 × 256, schwarzes Padding, Batchgröße 8) in den gespeicherten besten U-Net-Checkpoint; Wahrscheinlichkeitskarte.
2. Rückabbildung auf die **dekodierte** Originalgröße (Ausschnitt des eingepassten Bereichs, Padding verworfen, bilineare Vergrößerung der Karte); **erst dort** Binarisierung bei Wahrscheinlichkeit ≥ 0,5.
3. 8er-zusammenhängende Komponenten: nur die mit den meisten Pixeln bleibt; bei Gleichstand entscheidet das oberste linke Pixel in Zeilenreihenfolge (kleinste Zeile, dann kleinste Spalte). In den 385 Bildern kam kein Gleichstand vor; die Regel ist an synthetischen Fällen geprüft.
4. Leere Maske: das vollständige, unveränderte Bild ist der Zuschnitt (**protokollierter Vollbild-Fallback** für denselben 4E-Klassifikator, kein Wechsel zum 4B-Modell); die vorhandene Resize-Regel gilt wie bei jedem Bild.
5. Sonst enge Box der gewählten Komponente, Erweiterung nach der festen Regel des Vertrags `roi-expert-mask-v1` (je Seite `ceil(Kante / 5)`, am Bildrand begrenzt), Zuschnitt des **unmaskierten** Originalbildes (nichts geschwärzt, kein Maskenkanal), danach die vorhandene Resize-/Padding-Regel auf 160 × 160 × 1 `float32` in [0, 1].

**Die Validation-Expertenmaske bestimmt keinen Pixel des automatischen Zuschnitts.** Keine Funktion, die einen Zuschnitt erzeugt, hat einen Maskenparameter (geprüft); die ROI-Phase öffnet nur die 385 Validation-**Bilder**. Die automatischen ROIs (IDs, Geometrietabelle, vorhergesagte Pixel, Fingerabdruck) wurden 2026-10-20T21:49:44+02:00 **eingefroren**, bevor irgendeine Validation-Expertenmaske gelesen wurde (erste Maskenzugriffe: Kontrollphase 2026-10-20T21:49:45+02:00, Diagnosephase 2026-10-20T21:50:47+02:00). Ein Lauf **ohne** Validation-Expertenmasken erzeugt dieselben Zuschnitte: in isolierten Kopien ohne `Masks`-Ordner, mit unbrauchbaren Masken und mit den echten Masken wurden bei deterministischen GPU-Kernen (`TF_DETERMINISTIC_OPS=1`, nur in den Testkopien) bitgleiche ROIs eingefroren (gleicher Array-Fingerabdruck, gleiche Geometrietabelle, gleiche vorhergesagte Pixel); ohne diese Einstellung stimmten die abgeleiteten Zuschnitte (Boxen, Fallbacks, Array-Fingerabdruck `573864fa15c7906d…`) in allen beobachteten Wiederholungen überein, einzelne vorhergesagte Pixel konnten sich aber verschieben. Einschränkung: GPU-Inferenz ist über Prozesse hinweg nicht in jedem Fall bitgleich (siehe Grenzen); deshalb sind die ROIs eingefroren, und die späteren Phasen verwenden ausschließlich die eingefrorenen Boxen.

**Phasen (je ein eigener Prozess mit eigener Zugriffsprüfung):**

| Phase | geöffnete PNG-Dateien (aufgezeichnet) | Inhalt |
|---|---|---|
| ROI (GPU) | 385 Validation-Bilder, **keine** Maske | gespeichertes U-Net, automatische ROIs, Einfrieren |
| Kontrolle (GPU) | 385 Validation-Bilder und 385 Validation-Masken | jeder 4E-Checkpoint auf seinen ursprünglichen Experten-ROI-Eingaben muss die gespeicherten 4E-Scores reproduzieren |
| Klassifikation (GPU) | 385 Validation-Bilder, **keine** Maske | zehn 4E-Checkpoints, einzeln geladen, auf den eingefrorenen automatischen ROIs (nur Bilder und eingefrorene Boxen, kein U-Net) |
| Diagnose (CPU) | 385 Validation-Masken | Expertenmasken gegen die eingefrorenen Zuschnitte (nur Diagnose) |

Alle Phasen: nur Validation-Dateien, keine Train-, Test-, Vorschau- oder Kontaktübersichtsdatei. Die GPU wird wie im Projekt konfiguriert (Memory Growth vor der Initialisierung, Soft Placement aus, keine fremden CUDA-Bibliotheken, `env -u LD_LIBRARY_PATH`); eine fehlende GPU wäre ein Fehler, kein stiller CPU-Ausweichweg.

**Kontrolle der Lade- und Vorverarbeitungskette.** Vor dem ersten neuen Score (Toleranz vorab im Plan: größte absolute Score-Abweichung höchstens 1e-5, bei Abweichung Stopp und Ursachensuche, keine Lockerung): Jeder der zehn Checkpoints reproduziert auf seinen ursprünglichen Experten-ROI-Validation-Eingaben die gespeicherten 4E-Scores mit einer größten Abweichung von 0,0e+00 (alle zehn Läufe); die neu berechneten Experten-ROI-Eingaben entsprechen dem in 4E eingefrorenen Fingerabdruck. Ebenso reproduziert die erneute U-Net-Inferenz die gespeicherten 4F-A-Vorhersagen für alle 385 Validation-Bilder exakt (vorhergesagte Pixel, Komponentenzahl, Box um alle Pixel).

**Vergleiche.** Je Seed und Stufe: ROC-AUC, Average Precision, TP/FP/FN/TN, Sensitivität und Spezifität bei 0,5; zusätzlich die bestehende 4C-Regel für die explorativen Bild-Sensitivitätsziele ≥ 113 von 125 (0,90) und ≥ 119 von 125 (0,95): die **größte** Schwelle (Regel Score ≥ Schwelle), die das Ziel auf diesen Bildern erreicht; berichtet werden Schwelle und Zahl benigner Fehlalarme von 260. Case-Ebene: Maximum der Bild-Scores bei denselben bildbasierten Schwellen (erkannte beziehungsweise verpasste maligne Cases von 69, Fehlalarm-Cases von 144). Gepaarte Differenzen (automatisch minus Experten-ROI, automatisch minus Vollbild) je Seed, Mittel und Stichproben-SD (ddof = 1) der fünf Differenzen, Zahl negativer / positiver / gleicher Seeds – nur beschreibend: keine gepoolte AUC über Seeds, keine Signifikanz- oder Überlegenheitsaussage, kein „bester Seed“, die Schwellen werden **nicht** eingefroren.

## Ergebnisse der automatischen ROIs (Diagnose, nach dem Einfrieren)

Die Validation-Expertenmasken dienen hier ausschließlich der Diagnose der tatsächlichen Zuschnitte; die Zuschnittregel wurde deswegen nicht verändert. „Alle vorhergesagten Pixel“ (das Maß aus 4F-A) und „größte Komponente plus 20 %“ (der Zuschnitt dieses Schritts) sind verschiedene Boxen und werden getrennt ausgewiesen.

| Merkmal | Anzahl (Bilder; Fälle) | Nenner |
|---|---|---|
| leere Vorhersage: Vollbild-Fallback | 6 von 385 (1,6 %; 5 von 213 Fällen) | alle Bilder |
| mehr als eine Komponente (nur die größte wird behalten) | 70 von 385 (18,2 %; 62 von 213 Fällen) | alle Bilder |
| Gleichstand zwischen den größten Komponenten | 0 von 385 (0,0 %; 0 von 213 Fällen) | alle Bilder |
| Expertenmaske nicht vollständig im Zuschnitt | 33 von 379 (8,7 %; 30 von 212 Fällen) | Bilder ohne Fallback |
| weniger als 90 % der Expertenmaske im Zuschnitt | 14 von 379 (3,7 %; 14 von 212 Fällen) | Bilder ohne Fallback |
| kein Expertenmaskenpixel im Zuschnitt | 4 von 379 (1,1 %; 4 von 212 Fällen) | Bilder ohne Fallback |
| Dice der gewählten Komponente unter 0,5 | 12 von 379 (3,2 %; 12 von 212 Fällen) | Bilder ohne Fallback |
| Zuschnitt um mindestens Faktor 2 vergrößert | 22 von 385 (5,7 %; 14 von 213 Fällen) | alle Bilder |
| Zuschnitt um mindestens Faktor 3 vergrößert | 1 von 385 (0,3 %; 1 von 213 Fällen) | alle Bilder |
| 20-%-Erweiterung durch den Bildrand gekürzt | 60 von 379 (15,8 %; 44 von 212 Fällen) | Bilder ohne Fallback |
| enge Box berührt den Bildrand | 0 von 379 (0,0 %; 0 von 212 Fällen) | Bilder ohne Fallback |
| Zuschnitt umfasst mindestens 90 % der Bildfläche | 0 von 379 (0,0 %; 0 von 212 Fällen) | Bilder ohne Fallback |

Je Klasse und Gerät (Anzahl / Nenner; Nenner der Abdeckungsmerkmale sind die Bilder **ohne** Fallback der Gruppe):

| Art | Gruppe | Bilder / Fälle | Fallback (Anzahl / Nenner) | mehrere Komponenten | Abdeckung < 100 % (ohne Fallback) | Abdeckung < 90 % | Abdeckung 0 |
|---|---|---|---|---|---|---|---|
| Klasse | benigne | 260 / 144 | 4 / 260 | 35 / 260 | 10 / 256 | 5 / 256 | 2 / 256 |
| Klasse | maligne | 125 / 69 | 2 / 125 | 35 / 125 | 23 / 123 | 9 / 123 | 2 / 123 |
| Gerät | GE Logiq 5 @10-12MHz | 142 / 85 | 2 / 142 | 15 / 142 | 5 / 140 | 1 / 140 | 1 / 140 |
| Gerät | GE Logiq 7 @10-14MHz | 206 / 109 | 1 / 206 | 41 / 206 | 19 / 205 | 8 / 205 | 2 / 205 |
| Gerät | Toshiba Aplio 300 @12-14 MHz | 31 / 16 | 3 / 31 | 9 / 31 | 7 / 28 | 4 / 28 | 1 / 28 |
| Gerät | U-Systems @10-14MHz | 6 / 3 | 0 / 6 | 5 / 6 | 2 / 6 | 1 / 6 | 0 / 6 |

**Fallback-Bilder** (6 von 385, 5 Fälle): `bus_0522-l`, `bus_0522-r`, `bus_0971-l`, `bus_1013-l`, `bus_1014-l`, `bus_1041-r`. Sie sind in keiner Abdeckungsquote als Erfolg enthalten; die Quote wird deshalb zweimal berichtet (ohne Fallback = Lokalisationsqualität; mit Fallback als Vollabdeckung gezählt = trivial hoch und keine Lokalisationsqualität):

| Gruppe | Berechnungsbasis | Bilder / Fälle | Anteil der Expertenmaske im Zuschnitt: Mittel | Median | 5-%-Quantil | Zahl unter 100 % / unter 90 % / null |
|---|---|---|---:|---:|---:|---|
| alle | **ohne** Fallback-Bilder (Lokalisationsqualität) | 379 / 212 | 0,974 | 1,000 | 0,944 | 33 / 14 / 4 |
| alle | Fallback als Vollabdeckung gezählt (**keine** Lokalisationsqualität) | 385 / 213 | 0,974 | 1,000 | 0,947 | 33 / 14 / 4 |
| benigne | **ohne** Fallback-Bilder (Lokalisationsqualität) | 256 / 143 | 0,984 | 1,000 | 1,000 | 10 / 5 / 2 |
| benigne | Fallback als Vollabdeckung gezählt (**keine** Lokalisationsqualität) | 260 / 144 | 0,985 | 1,000 | 1,000 | 10 / 5 / 2 |
| maligne | **ohne** Fallback-Bilder (Lokalisationsqualität) | 123 / 69 | 0,952 | 1,000 | 0,829 | 23 / 9 / 2 |
| maligne | Fallback als Vollabdeckung gezählt (**keine** Lokalisationsqualität) | 125 / 69 | 0,953 | 1,000 | 0,829 | 23 / 9 / 2 |

Kategorien des tatsächlichen Zuschnitts (Ganzzahlarithmetik auf Pixelzahlen, keine Gleitkommaschwelle):

| Kategorie des tatsächlichen Zuschnitts | alle 385 Bilder (213 Fälle) | benigne 260 (144) | maligne 125 (69) |
|---|---|---|---|
| Fallback (ganzes Bild) | 6 (5 Fälle) | 4 (3 Fälle) | 2 (2 Fälle) |
| deckt alle Expertenmaskenpixel | 346 (203 Fälle) | 246 (140 Fälle) | 100 (63 Fälle) |
| deckt 90 % bis < 100 % | 19 (17 Fälle) | 5 (5 Fälle) | 14 (12 Fälle) |
| deckt > 0 bis < 90 % | 10 (10 Fälle) | 3 (3 Fälle) | 7 (7 Fälle) |
| deckt kein Expertenmaskenpixel | 4 (4 Fälle) | 2 (2 Fälle) | 2 (2 Fälle) |

Vier Boxdefinitionen auf denselben Bildern (zeigt, warum die Zahl „Expertenmaske nicht vollständig in der Box“ von der Definition abhängt):

Nenner: die 379 Validation-Bilder mit nicht leerer Vorhersage (ohne die Fallback-Bilder).

| Definition der Box | Expertenmaske nicht vollständig drin | weniger als 90 % drin | keine drin | mittlerer Anteil | Median |
|---|---:|---:|---:|---:|---:|
| Box um **alle** vorhergesagten Pixel, ohne Erweiterung (das Maß aus 4F-A) | 341 | 40 | 1 | 0,949 | 0,982 |
| Box um alle vorhergesagten Pixel, um 20 % erweitert (nicht für einen Zuschnitt verwendet) | 27 | 10 | 1 | 0,986 | 1,000 |
| enge Box der **größten Komponente**, ohne Erweiterung | 344 | 48 | 6 | 0,936 | 0,982 |
| größte Komponente **plus 20 %**: der tatsächliche Zuschnitt dieses Schritts | 33 | 14 | 4 | 0,974 | 1,000 |

- **Erweiterung um 20 % wirkt stark:** ohne Erweiterung liegen bei etwa neun von zehn Bildern einzelne Expertenmaskenpixel außerhalb der engen Box (341 von 379 bei der 4F-A-Box, 344 bei der größten Komponente), mit der 20-%-Erweiterung und der größten Komponente bleibt bei 33 von 379 Bildern (8,7 %) ein Rest außerhalb; bei 14 Bildern (3,7 %) fehlen mehr als 10 %, bei 4 fehlt die Läsion ganz.
- **Maligne Bilder trifft es häufiger:** Abdeckung unter 100 % bei 23 von 123 malignen (ohne Fallback; 20 Fälle) gegenüber 10 von 256 benignen Bildern (10 Fälle). Beschreibend; die Gruppen unterscheiden sich in Größe, Gerät und Läsionsgröße.

**Wahl der Komponente.** Von den 70 Bildern mit mehreren Komponenten (62 Fälle) ist die größte in 66 zugleich die mit der größten Überlappung mit der Expertenmaske; in 4 Bildern hat die gewählte keine Überlappung, eine andere schon (die Regel „größte Komponente“ wählt dort die falsche), in 2 überlappt keine Komponente die Expertenmaske; Gleichstände gab es keine.

| Art | Gruppe | Bilder mit mehreren Komponenten | größte Komponente ist zugleich die mit der größten Überlappung mit der Expertenmaske | gewählte überlappt, eine andere mehr | gewählte ohne Überlappung, eine andere überlappt | keine Komponente überlappt | Gleichstände |
|---|---|---|---:|---:|---:|---:|---:|
| alle | alle | 70 (62 Fälle) | 66 | 0 | 4 | 2 | 0 |
| Klasse | benigne | 35 (34 Fälle) | 34 | 0 | 1 | 2 | 0 |
| Klasse | maligne | 35 (28 Fälle) | 32 | 0 | 3 | 0 | 0 |
| Gerät | GE Logiq 5 @10-12MHz | 15 (13 Fälle) | 15 | 0 | 0 | 1 | 0 |
| Gerät | GE Logiq 7 @10-14MHz | 41 (38 Fälle) | 38 | 0 | 3 | 1 | 0 |
| Gerät | Toshiba Aplio 300 @12-14 MHz | 9 (8 Fälle) | 8 | 0 | 1 | 0 | 0 |
| Gerät | U-Systems @10-14MHz | 5 (3 Fälle) | 5 | 0 | 0 | 0 | 0 |

**Auffällige Beispiele** (die drei aus der 4F-A-Sichtprüfung hervorgehobenen und alle zwölf Bilder der 4F-A-Fehleranalyse):

★ = in der Sichtprüfung von 4F-A besonders erwähnt. „Seeds ≥ 0,5“ = Zahl der fünf Seeds, in denen das Bild bei 0,5 als maligne klassifiziert wird, in der Reihenfolge Vollbild / Experten-ROI / automatische ROI.

| ID | Klasse | Gerät | Dice 4F-A (alle Pixel) | vorhergesagte Pixel / Komponenten / größte Komponente | Zuschnitt gegenüber der Expertenmaske (Anteil drin) | Zuschnitt (Anteil der Bildfläche, Skalierung auf 160) | Frozen: Seeds ≥ 0,5 | Fine-Tuning: Seeds ≥ 0,5 |
|---|---|---|---:|---|---|---|---|---|
| `bus_1014-r` ★ | maligne | Toshiba Aplio 300 | 0,012 | 3.588 / 7 / 3.229 | deckt kein Expertenmaskenpixel (0,00) | 122 × 84 (12,2 %, ×1,31) | 0/5/0 | 0/5/0 |
| `bus_1013-l` ★ | maligne | Toshiba Aplio 300 | 0,000 | 0 / 0 / 0 | Fallback | 276 × 304 (100,0 %, ×0,53) | 0/5/5 | 5/5/5 |
| `bus_1014-l` ★ | maligne | Toshiba Aplio 300 | 0,000 | 0 / 0 / 0 | Fallback | 276 × 305 (100,0 %, ×0,52) | 0/0/1 | 0/0/1 |
| `bus_0522-l` | benigne | GE Logiq 5 | 0,000 | 0 / 0 / 0 | Fallback | 319 × 431 (100,0 %, ×0,37) | 1/0/5 | 0/0/5 |
| `bus_0522-r` | benigne | GE Logiq 5 | 0,000 | 0 / 0 / 0 | Fallback | 319 × 431 (100,0 %, ×0,37) | 1/0/5 | 0/0/5 |
| `bus_0743-s` | benigne | GE Logiq 5 | 0,000 | 4.464 / 4 / 4.339 | deckt kein Expertenmaskenpixel (0,00) | 130 × 90 (5,7 %, ×1,23) | 5/0/4 | 4/0/0 |
| `bus_0971-l` | benigne | GE Logiq 7 | 0,000 | 0 / 0 / 0 | Fallback | 279 × 362 (100,0 %, ×0,44) | 0/0/5 | 0/0/1 |
| `bus_0971-r` | benigne | GE Logiq 7 | 0,000 | 534 / 3 / 376 | deckt > 0 bis < 90 % (0,01) | 25 × 54 (1,3 %, ×2,96) | 0/0/5 | 0/0/3 |
| `bus_1041-r` | benigne | Toshiba Aplio 300 | 0,000 | 0 / 0 / 0 | Fallback | 276 × 303 (100,0 %, ×0,53) | 0/0/5 | 0/0/5 |
| `bus_0287-l` | maligne | GE Logiq 7 | 0,162 | 1.451 / 1 / 1.451 | deckt > 0 bis < 90 % (0,16) | 70 × 55 (3,6 %, ×2,29) | 0/1/0 | 0/1/0 |
| `bus_1053-s` | benigne | Toshiba Aplio 300 | 0,203 | 2.985 / 4 / 1.498 | deckt > 0 bis < 90 % (0,35) | 118 × 82 (6,9 %, ×1,36) | 0/0/0 | 0/1/0 |
| `bus_0232-r` | benigne | GE Logiq 7 | 0,235 | 9.524 / 10 / 8.067 | deckt kein Expertenmaskenpixel (0,00) | 175 × 139 (23,9 %, ×0,91) | 0/0/5 | 3/0/2 |

- `bus_1014-r` (maligne, Toshiba): sieben Komponenten; die größte (3.229 Pixel) wird gewählt und liegt **neben** der Läsion – der Zuschnitt enthält kein Expertenmaskenpixel, in beiden Stufen erkennt ihn bei 0,5 kein Seed (die Experten-ROI: alle fünf). Das ist eine echte Fehllokalisation der Segmentierung, die die Zuschnittregel nicht auffängt.
- `bus_1013-l` und `bus_1014-l` (maligne, Toshiba): leere Vorhersagen, also Vollbild-Fallback. `bus_1013-l` wird dabei von allen fünf Seeds beider Stufen als maligne klassifiziert, `bus_1014-l` nur von je einem der fünf Seeds (in beiden Stufen); die Experten-ROI trifft `bus_1013-l` in allen Seeds, `bus_1014-l` in keinem. Die Fallback-Eingabe ist für den ROI-Klassifikator eine andere Eingabeart (kein Zuschnitt), auf der er nicht trainiert wurde.

## Ergebnisse der Klassifikation (Validation, 385 Bilder, 213 Fälle; fünf Seeds je Stufe)

Mittel ± Stichproben-SD (ddof = 1) und Spannweite über die fünf Seeds; Trainingsstreuung auf demselben Holdout, keine unabhängigen Datensätze:

| Stufe | Eingabe | ROC-AUC, Mittel ± SD (Minimum–Maximum) | Average Precision, Mittel ± SD (Minimum–Maximum) |
|---|---|---|---|
| Frozen-Kopf | Vollbild (4B, gespeichert) | 0,797 ± 0,011 (0,781–0,809) | 0,685 ± 0,011 (0,668–0,696) |
| Frozen-Kopf | Experten-ROI (4E, gespeichert) | 0,892 ± 0,010 (0,874–0,901) | 0,832 ± 0,018 (0,804–0,852) |
| Frozen-Kopf | automatische ROI (4F-B) | 0,870 ± 0,010 (0,854–0,881) | 0,781 ± 0,015 (0,766–0,806) |
| erweitertes Fine-Tuning 1e-4 | Vollbild (4B, gespeichert) | 0,881 ± 0,011 (0,862–0,890) | 0,803 ± 0,024 (0,774–0,841) |
| erweitertes Fine-Tuning 1e-4 | Experten-ROI (4E, gespeichert) | 0,921 ± 0,012 (0,905–0,934) | 0,872 ± 0,014 (0,852–0,888) |
| erweitertes Fine-Tuning 1e-4 | automatische ROI (4F-B) | 0,898 ± 0,011 (0,885–0,909) | 0,820 ± 0,032 (0,770–0,853) |

Gepaarte Differenzen innerhalb desselben Seeds (nur beschreibend):

| Stufe | Vergleich (je Seed gepaart) | Maß | Differenzen je Seed (7 ; 21 ; 42 ; 123 ; 2026) | Mittel | SD (ddof = 1) | negativ / positiv / gleich |
|---|---|---|---|---:|---:|---|
| Frozen-Kopf | automatisch minus Experten-ROI | ROC-AUC | −0,0196 ; −0,0238 ; −0,0203 ; −0,0196 ; −0,0270 | −0,0220 | 0,0033 | 5 / 0 / 0 |
| Frozen-Kopf | automatisch minus Experten-ROI | Average Precision | −0,0314 ; −0,0597 ; −0,0591 ; −0,0464 ; −0,0581 | −0,0509 | 0,0122 | 5 / 0 / 0 |
| Frozen-Kopf | automatisch minus Vollbild | ROC-AUC | +0,0448 ; +0,0720 ; +0,0791 ; +0,1000 ; +0,0655 | +0,0723 | 0,0201 | 0 / 5 / 0 |
| Frozen-Kopf | automatisch minus Vollbild | Average Precision | +0,0768 ; +0,0950 ; +0,0828 ; +0,1374 ; +0,0862 | +0,0957 | 0,0243 | 0 / 5 / 0 |
| erweitertes Fine-Tuning 1e-4 | automatisch minus Experten-ROI | ROC-AUC | −0,0206 ; −0,0218 ; −0,0271 ; −0,0196 ; −0,0252 | −0,0229 | 0,0032 | 5 / 0 / 0 |
| erweitertes Fine-Tuning 1e-4 | automatisch minus Experten-ROI | Average Precision | −0,0444 ; −0,0357 ; −0,0558 ; −0,0820 ; −0,0432 | −0,0522 | 0,0181 | 5 / 0 / 0 |
| erweitertes Fine-Tuning 1e-4 | automatisch minus Vollbild | ROC-AUC | +0,0207 ; +0,0176 ; +0,0013 ; −0,0013 ; +0,0465 | +0,0170 | 0,0191 | 1 / 4 / 0 |
| erweitertes Fine-Tuning 1e-4 | automatisch minus Vollbild | Average Precision | +0,0309 ; +0,0118 ; +0,0086 ; −0,0278 ; +0,0609 | +0,0169 | 0,0325 | 1 / 4 / 0 |

**Bei der festen Schwelle 0,5** (der Betriebspunkt hängt von den unterschiedlichen Scoreverteilungen der Eingaben ab; Zähler sind je Seed und werden nicht über Seeds summiert, hier als Mittel):

| Stufe | Eingabe | TP / FN (Mittel über die Seeds) | FP / TN (Mittel über die Seeds) | Sensitivität | Spezifität |
|---|---|---|---|---:|---:|
| Frozen-Kopf | Vollbild (4B, gespeichert) | 64,0 / 61,0 | 34,0 / 226,0 | 0,512 | 0,869 |
| Frozen-Kopf | Experten-ROI (4E, gespeichert) | 89,2 / 35,8 | 20,2 / 239,8 | 0,714 | 0,922 |
| Frozen-Kopf | automatische ROI (4F-B) | 88,2 / 36,8 | 28,8 / 231,2 | 0,706 | 0,889 |
| erweitertes Fine-Tuning 1e-4 | Vollbild (4B, gespeichert) | 85,2 / 39,8 | 27,0 / 233,0 | 0,682 | 0,896 |
| erweitertes Fine-Tuning 1e-4 | Experten-ROI (4E, gespeichert) | 84,4 / 40,6 | 11,4 / 248,6 | 0,675 | 0,956 |
| erweitertes Fine-Tuning 1e-4 | automatische ROI (4F-B) | 82,0 / 43,0 | 17,8 / 242,2 | 0,656 | 0,932 |

| Stufe | Vergleich | Zähler bei 0,5 | Differenzen je Seed | Mittel | SD | negativ / positiv / null |
|---|---|---|---|---:|---:|---|
| Frozen-Kopf | automatisch minus Experten-ROI | TP | +1 ; −3 ; −2 ; +3 ; −4 | −1,0 | 2,9 | 3 / 2 / 0 |
| Frozen-Kopf | automatisch minus Experten-ROI | FP | +7 ; +9 ; +9 ; +9 ; +9 | +8,6 | 0,9 | 0 / 5 / 0 |
| Frozen-Kopf | automatisch minus Vollbild | TP | +14 ; +45 ; +17 ; +25 ; +20 | +24,2 | 12,3 | 0 / 5 / 0 |
| Frozen-Kopf | automatisch minus Vollbild | FP | −13 ; +19 ; −6 ; −13 ; −13 | −5,2 | 13,9 | 4 / 1 / 0 |
| erweitertes Fine-Tuning 1e-4 | automatisch minus Experten-ROI | TP | −3 ; −6 ; +2 ; −1 ; −4 | −2,4 | 3,0 | 4 / 1 / 0 |
| erweitertes Fine-Tuning 1e-4 | automatisch minus Experten-ROI | FP | +9 ; +2 ; +6 ; +10 ; +5 | +6,4 | 3,2 | 0 / 5 / 0 |
| erweitertes Fine-Tuning 1e-4 | automatisch minus Vollbild | TP | −1 ; −11 ; +14 ; −9 ; −9 | −3,2 | 10,4 | 4 / 1 / 0 |
| erweitertes Fine-Tuning 1e-4 | automatisch minus Vollbild | FP | −9 ; −7 ; +0 ; −4 ; −26 | −9,2 | 10,0 | 4 / 0 / 1 |

**Bei den Sensitivitätszielen** (4C-Regel, explorativ, keine Schwelle wird festgelegt; die nötige Schwelle gehört zu jedem Seed und jeder Eingabe und ist nicht austauschbar):

| Stufe | Ziel (erkannte maligne Bilder) | Eingabe | nötige Schwelle, Mittel (Minimum–Maximum) | benigne Fehlalarme von 260, Mittel ± SD (Minimum–Maximum) | Cases: Fehlalarm-Cases von 144 / verpasste maligne Cases von 69 (Mittel) |
|---|---|---|---|---|---|
| Frozen-Kopf | ≥ 113 von 125 (0,90) | Vollbild (4B, gespeichert) | 0,252 (0,184–0,290) | 145,4 ± 10,0 (137–161) | 101,8 / 2,2 |
| Frozen-Kopf | ≥ 113 von 125 (0,90) | Experten-ROI (4E, gespeichert) | 0,197 (0,167–0,223) | 74,6 ± 12,3 (63–94) | 54,4 / 3,4 |
| Frozen-Kopf | ≥ 113 von 125 (0,90) | automatische ROI (4F-B) | 0,156 (0,100–0,186) | 102,8 ± 10,5 (92–119) | 72,4 / 4,0 |
| Frozen-Kopf | ≥ 119 von 125 (0,95) | Vollbild (4B, gespeichert) | 0,195 (0,129–0,250) | 178,2 ± 18,5 (157–208) | 116,0 / 0,8 |
| Frozen-Kopf | ≥ 119 von 125 (0,95) | Experten-ROI (4E, gespeichert) | 0,049 (0,036–0,063) | 180,8 ± 13,0 (165–200) | 112,8 / 0,8 |
| Frozen-Kopf | ≥ 119 von 125 (0,95) | automatische ROI (4F-B) | 0,077 (0,063–0,102) | 155,0 ± 14,7 (135–174) | 100,2 / 1,2 |
| erweitertes Fine-Tuning 1e-4 | ≥ 113 von 125 (0,90) | Vollbild (4B, gespeichert) | 0,235 (0,155–0,320) | 90,0 ± 17,1 (74–110) | 69,8 / 3,0 |
| erweitertes Fine-Tuning 1e-4 | ≥ 113 von 125 (0,90) | Experten-ROI (4E, gespeichert) | 0,208 (0,169–0,242) | 57,2 ± 20,0 (34–89) | 43,0 / 3,6 |
| erweitertes Fine-Tuning 1e-4 | ≥ 113 von 125 (0,90) | automatische ROI (4F-B) | 0,142 (0,098–0,191) | 88,8 ± 9,4 (76–98) | 64,4 / 2,6 |
| erweitertes Fine-Tuning 1e-4 | ≥ 119 von 125 (0,95) | Vollbild (4B, gespeichert) | 0,118 (0,040–0,187) | 146,8 ± 22,8 (122–180) | 103,0 / 1,0 |
| erweitertes Fine-Tuning 1e-4 | ≥ 119 von 125 (0,95) | Experten-ROI (4E, gespeichert) | 0,101 (0,044–0,164) | 116,4 ± 19,1 (95–136) | 81,6 / 0,4 |
| erweitertes Fine-Tuning 1e-4 | ≥ 119 von 125 (0,95) | automatische ROI (4F-B) | 0,086 (0,030–0,164) | 126,6 ± 20,3 (109–160) | 87,0 / 0,6 |

Gepaarte Differenzen der benignen Fehlalarme (negativ = weniger Fehlalarme bei gleichem Sensitivitätsziel):

| Stufe | Ziel | Vergleich | Differenz der benignen Fehlalarme je Seed (7 ; 21 ; 42 ; 123 ; 2026) | Mittel | SD | negativ / positiv / null | Mittel der Fehlalarm-Cases (Case-Ebene) |
|---|---|---|---|---:|---:|---|---:|
| Frozen-Kopf | 0,90 | automatisch minus Experten-ROI | +10 ; +25 ; +27 ; +56 ; +23 | +28,2 | 16,9 | 0 / 5 / 0 | +18,0 |
| Frozen-Kopf | 0,90 | automatisch minus Vollbild | −39 ; −33 ; −66 ; −30 ; −45 | −42,6 | 14,3 | 5 / 0 / 0 | −29,4 |
| Frozen-Kopf | 0,95 | automatisch minus Experten-ROI | −26 ; −34 ; −7 ; −39 ; −23 | −25,8 | 12,3 | 5 / 0 / 0 | −12,6 |
| Frozen-Kopf | 0,95 | automatisch minus Vollbild | +17 ; −26 ; −50 ; −43 ; −14 | −23,2 | 26,5 | 4 / 1 / 0 | −15,8 |
| erweitertes Fine-Tuning 1e-4 | 0,90 | automatisch minus Experten-ROI | +29 ; +39 ; +39 ; +9 ; +42 | +31,6 | 13,6 | 0 / 5 / 0 | +21,4 |
| erweitertes Fine-Tuning 1e-4 | 0,90 | automatisch minus Vollbild | +4 ; −17 ; +17 ; +24 ; −34 | −1,2 | 24,1 | 2 / 3 / 0 | −5,4 |
| erweitertes Fine-Tuning 1e-4 | 0,95 | automatisch minus Experten-ROI | −21 ; +33 ; +24 ; −3 ; +18 | +10,2 | 21,9 | 2 / 3 / 0 | +5,4 |
| erweitertes Fine-Tuning 1e-4 | 0,95 | automatisch minus Vollbild | −44 ; −50 ; +10 ; −1 ; −16 | −20,2 | 26,2 | 4 / 1 / 0 | −16,0 |

### Beschreibende Beobachtungen

- **ROC-AUC und Average Precision:** Die automatische ROI liegt in allen zehn Seed-Stufen-Kombinationen unter der Experten-ROI und über dem Vollbild – mit einer Ausnahme (Fine-Tuning, Seed 123: −0,0013 gegenüber dem Vollbild). Die Spannweite zwischen den Seeds ist beim Fine-Tuning gegenüber dem Vollbild groß (SD der Differenzen 0,0191), deshalb ist das Vorzeichen dort nicht einheitlich.
- **Fehlalarme bei ≥ 113 und ≥ 119 von 125 (Fine-Tuning):** Vollbild 90,0 ± 17,1 / 146,8 ± 22,8, Experten-ROI 57,2 ± 20,0 / 116,4 ± 19,1, automatische ROI 88,8 ± 9,4 / 126,6 ± 20,3. Beim Ziel 0,90 liegt die automatische ROI damit nahe am Vollbild (Differenz −1,2, Vorzeichen uneinheitlich: 2 negativ, 3 positiv) und in allen fünf Seeds über der Experten-ROI (+31,6); beim Ziel 0,95 liegt sie dazwischen, die Vorzeichen gegenüber der Experten-ROI sind uneinheitlich. Beim Frozen-Kopf liegt die automatische ROI beim Ziel 0,90 zwischen Vollbild und Experten-ROI und beim Ziel 0,95 unter beiden. Wegen der kleinen Zahl benigner Bilder und fünf Seeds auf demselben Holdout sind das Tendenzen, keine Nachweise.
- **Case-Ebene:** Das Maximum der Bild-Scores bei denselben bildbasierten Schwellen verschiebt die Fehlalarme stark in Richtung vieler Fehlalarm-Cases (ein einziges falsch positives Bild genügt); die Reihenfolge der Eingaben entspricht im Wesentlichen der auf Bildebene (Tabelle oben, letzte Spalte).

### Fallen schlechte Zuschnitte, Fallback oder Fehlsegmentierung mit Klassifikationsfehlern zusammen? (beschreibend)

Je Kategorie des tatsächlichen Zuschnitts: Mittel (Minimum–Maximum über die fünf Seeds) der bei 0,5 falsch klassifizierten Bilder (FN bei malignen, FP bei benignen) je Eingabe. Es wird keine neue Fallback- oder Zuschnittregel daraus abgeleitet.

**Frozen-Kopf**, Frozen-Kopf

| Kategorie | Bilder (Fälle) | Vollbild, Mittel (Minimum–Maximum über die Seeds) | Experten-ROI | automatische ROI |
|---|---|---|---|---|
| Zuschnitt deckt alle Expertenmaskenpixel | 100 (63) | 46,2 (40–61) | 29,0 (23–34) | 28,6 (26–34) |
| deckt 90 % bis < 100 % | 14 (12) | 6,0 (4–10) | 2,0 (2–2) | 4,4 (4–5) |
| deckt > 0 bis < 90 % | 7 (7) | 4,8 (4–5) | 2,8 (2–3) | 2,0 (2–2) |
| deckt kein Expertenmaskenpixel | 2 (2) | 2,0 (2–2) | 1,0 (1–1) | 1,0 (1–1) |
| Fallback (ganzes Bild) | 2 (2) | 2,0 (2–2) | 1,0 (1–1) | 0,8 (0–1) |
| Dice der gewählten Komponente < 0,5 (zusätzliche Sicht) | 8 (8) | 5,0 (5–5) | 3,8 (3–4) | 3,0 (3–3) |
| alle Bilder der Klasse | 125 (69) | 61,0 (53–80) | 35,8 (30–41) | 36,8 (33–43) |

**Frozen-Kopf**, Frozen-Kopf

| Kategorie | Bilder (Fälle) | Vollbild, Mittel (Minimum–Maximum über die Seeds) | Experten-ROI | automatische ROI |
|---|---|---|---|---|
| Zuschnitt deckt alle Expertenmaskenpixel | 246 (140) | 31,2 (8–41) | 18,2 (14–21) | 20,0 (17–23) |
| deckt 90 % bis < 100 % | 5 (5) | 1,4 (1–2) | 2,0 (2–2) | 2,0 (2–2) |
| deckt > 0 bis < 90 % | 3 (3) | 0,0 (0–0) | 0,0 (0–0) | 1,0 (1–1) |
| deckt kein Expertenmaskenpixel | 2 (2) | 1,0 (1–1) | 0,0 (0–0) | 1,8 (1–2) |
| Fallback (ganzes Bild) | 4 (3) | 0,4 (0–1) | 0,0 (0–0) | 4,0 (4–4) |
| Dice der gewählten Komponente < 0,5 (zusätzliche Sicht) | 4 (4) | 1,0 (1–1) | 0,0 (0–0) | 2,8 (2–3) |
| alle Bilder der Klasse | 260 (144) | 34,0 (10–45) | 20,2 (16–23) | 28,8 (25–32) |

**erweitertes Fine-Tuning 1e-4**, erweitertes Fine-Tuning 1e-4

| Kategorie | Bilder (Fälle) | Vollbild, Mittel (Minimum–Maximum über die Seeds) | Experten-ROI | automatische ROI |
|---|---|---|---|---|
| Zuschnitt deckt alle Expertenmaskenpixel | 100 (63) | 29,8 (24–37) | 33,8 (28–46) | 34,4 (28–48) |
| deckt 90 % bis < 100 % | 14 (12) | 4,0 (2–7) | 2,6 (2–5) | 4,2 (4–5) |
| deckt > 0 bis < 90 % | 7 (7) | 3,6 (3–4) | 2,2 (1–3) | 2,4 (2–3) |
| deckt kein Expertenmaskenpixel | 2 (2) | 1,4 (1–2) | 1,0 (1–1) | 1,2 (1–2) |
| Fallback (ganzes Bild) | 2 (2) | 1,0 (1–1) | 1,0 (1–1) | 0,8 (0–1) |
| Dice der gewählten Komponente < 0,5 (zusätzliche Sicht) | 8 (8) | 4,0 (3–5) | 3,4 (2–5) | 3,6 (3–4) |
| alle Bilder der Klasse | 125 (69) | 39,8 (31–50) | 40,6 (34–56) | 43,0 (36–57) |

**erweitertes Fine-Tuning 1e-4**, erweitertes Fine-Tuning 1e-4

| Kategorie | Bilder (Fälle) | Vollbild, Mittel (Minimum–Maximum über die Seeds) | Experten-ROI | automatische ROI |
|---|---|---|---|---|
| Zuschnitt deckt alle Expertenmaskenpixel | 246 (140) | 23,2 (16–36) | 10,0 (6–13) | 12,4 (10–18) |
| deckt 90 % bis < 100 % | 5 (5) | 2,4 (2–4) | 1,2 (1–2) | 1,2 (1–2) |
| deckt > 0 bis < 90 % | 3 (3) | 0,0 (0–0) | 0,2 (0–1) | 0,6 (0–1) |
| deckt kein Expertenmaskenpixel | 2 (2) | 1,4 (0–2) | 0,0 (0–0) | 0,4 (0–1) |
| Fallback (ganzes Bild) | 4 (3) | 0,0 (0–0) | 0,0 (0–0) | 3,2 (3–4) |
| Dice der gewählten Komponente < 0,5 (zusätzliche Sicht) | 4 (4) | 1,4 (0–2) | 0,2 (0–1) | 1,0 (0–2) |
| alle Bilder der Klasse | 260 (144) | 27,0 (20–40) | 11,4 (7–14) | 17,8 (14–23) |

- **Fallback erklärt einen Teil der zusätzlichen Fehlalarme bei 0,5:** Von den vier benignen Fallback-Bildern klassifiziert die automatische Pipeline im Mittel 3,2 (Fine-Tuning) beziehungsweise 4,0 (Frozen) als maligne, die Experten-ROI 0,0 und das Vollbildmodell 0,0 beziehungsweise 0,4. Der Mehrbetrag der automatischen gegenüber der Experten-ROI an Fehlalarmen bei 0,5 (+6,4 beim Fine-Tuning, +8,6 beim Frozen-Kopf) kommt also zu einem erheblichen Teil von diesen vier Bildern; der Rest verteilt sich auf Bilder mit vollständigem Zuschnitt.
- **Verpasste maligne Bilder:** Die meisten falsch negativen malignen Bilder haben einen Zuschnitt, der die Expertenmaske vollständig enthält (Fine-Tuning, automatische ROI: 34,4 von 43,0 im Mittel; in allen drei Eingaben ähnlich). Die Zahl der malignen Bilder mit schlechter Abdeckung ist klein (zwei Bilder ganz ohne Abdeckung, sieben mit Abdeckung unter 90 %, vierzehn mit 90 % bis unter 100 %). In der Kategorie 90 % bis unter 100 % (14 Bilder) verpasst die automatische ROI im Mittel 4,2 (Fine-Tuning) beziehungsweise 4,4 (Frozen), die Experten-ROI 2,6 beziehungsweise 2,0; in den übrigen schlechten Kategorien sind die Zahlen zu klein für eine Aussage. Schlechte Zuschnittabdeckung ist damit nur ein kleiner Teil der Klassifikationsfehler; der größere Teil entsteht auch bei vollständiger Abdeckung.

## Prüfungen

Gezielt geprüft (74 Prüfungen, alle wie erwartet; kein Training):

- **Komponentenwahl und Zuschnitt (synthetisch):** größte Komponente (8er-Nachbarschaft, auch diagonal), Gleichstand nach der Zeilenreihenfolge-Regel (Zeilen-, Spalten- und Dreifach-Gleichstand), 60 Zufallsmasken gegen eine unabhängig geschriebene Flood-Fill-Implementierung; leere Vorhersage → Vollbild-Fallback; die unveränderte 20-%-Regel gegen eine unabhängige Ganzzahlherleitung (120 Zufallsboxen, auch am Rand beschnitten); Zuschnitt des **unmaskierten** Originals (Pixel neben der Läsion bleiben erhalten, ein Kanal), Resize-Regel; Fallback gleich der vorhandenen Ganzbild-Aufbereitung; Rückabbildung auf Originalkoordinaten, Padding verworfen, Schwelle genau ≥ 0,5; keine Funktion, die einen Zuschnitt erzeugt, hat einen Expertenmaskenparameter.
- **Tabellen (synthetisch):** 4C-Regel (größte Schwelle, die das Ziel erreicht) und Fehlalarme gegen eine unabhängige Herleitung; Case-Maximum; gepaarte Differenzen als unabhängige Subtraktionen; Diagnosekategorien in Ganzzahlarithmetik; Abdeckung **mit und ohne** Fallback getrennt; Flags mit Nennern.
- **Schranken in isolierten Kopien:** geänderter ROI-Vertrag, Split, 4F-A-Checkpoint, 4F-A-Status, gespeicherte 4E- oder 4B-Vorhersagen oder falscher Interpreter stoppen vor dem Plan und vor jeder PNG-Datei; jede Phase ohne registrierten Plan öffnet nichts; ein geänderter Plan stoppt vor jeder Phase.
- **Maskenunabhängigkeit:** die ROI-Phase friert ohne `Masks`-Ordner, mit unbrauchbaren und mit echten Masken bitgleiche ROIs ein (echte GPU-Inferenz mit deterministischen Kernen in den Testkopien) und öffnet in allen drei Fällen genau die 385 Bilder und keine Maske. Ein beim ersten Versuch ohne deterministische Kerne beobachteter Unterschied von einem Pixel bei 39 Bildern trat zwischen zwei Läufen mit gleichen Eingaben auf (GPU-Rundung), nicht durch Masken; die abgeleiteten Zuschnitte waren auch dort gleich. Daraufhin wurde die Prüfung auf deterministische Kerne umgestellt; die Studie selbst läuft wie im Projekt üblich mit der Standardeinstellung.
- **Echtlauf (unabhängige Nachrechnung):** Reihenfolge Plan < Einfrieren < Maskenzugriffe, Zugriff je Phase, Fingerabdrücke, Kontrolle aller zehn Läufe; die vorhergesagten Pixel mit eigener Inferenz (eigene Vorverarbeitung, eigene Rückabbildung) für **alle** 385 Bilder exakt reproduziert; Komponenten, Boxen und alle 385 Zuschnitte mit eigener Implementierung gleich dem eingefrorenen Array-Fingerabdruck; die Diagnose (Pixel im Zuschnitt, Kategorie) für alle 385 Bilder aus den Maskendateien nachgerechnet; ROC-AUC, AP, Konfusion, Schwellen, Fehlalarme, Case-Ebene und gepaarte Differenzen aller 30 Vorhersagemengen unabhängig nachgerechnet; gleiche IDs, Fälle und Labels in allen Vergleichsgruppen.
- **Sicherer Wiederaufruf:** ein vollständiger Lauf wird anhand der Hashes validiert und ohne neue Inferenz übersprungen; eine fehlende Klassifikation wird gezielt ergänzt (nur der fehlende Lauf, die übrigen neun bleiben byte-identisch); ein manipuliertes Ergebnis stoppt; schlägt eine Phase fehl, bleiben frühere Ergebnisse unberührt und der Fehler steht neben dem früheren Erfolgsstatus.
- Rohdaten, Manifest, Split, Verträge, alle Checkpoints und Ergebnisse aus 4B, 4D, 4E und 4F-A sind per Hash unverändert; `py_compile` und `pip check` ohne Befund; keine Installation.

## Abweichungen, Grenzen und offene Punkte

- **Funktionstest vor der Plan-Registrierung.** Weil die Inferenz nur Sekunden dauert, hat der Funktionstest des Skripts in einer **isolierten Kopie** (mit eigenem Plan in der Kopie) die Studie bereits einmal vollständig durchgerechnet, bevor der Plan im Projekt registriert wurde; die Ergebnisse dieses Testlaufs waren beim Registrieren des Projektplans bekannt. Der Projektplan, die Zuschnittregel und alle Auswertungen waren zu diesem Zeitpunkt durch die vorab festgelegte Planung bestimmt; nach dem Testlauf wurde nichts an der Methode geändert, nur Darstellungen (Abbildungen) und Prüfungen verbessert und dem Plan eine Beschreibung der Arbeitsreihenfolge hinzugefügt (vor seiner Registrierung). Der Studienlauf im Projekt reproduzierte die Ergebnisse des Testlaufs vollständig: alle 16 Ergebnistabellen (einschließlich der lokalen Einzelbildtabellen) sind byte-identisch, der ROI-Fingerabdruck ist gleich. Die Reihenfolge „Plan vor erstem Validation-Bild im Projekt“ ist belegt; die Reihenfolge „Plan vor jeder Berechnung überhaupt“ gilt wegen des Funktionstests nur für die Projektdateien.
- **Keine Neutrainings, keine Anpassung nach der Diagnose:** Fallback und Zuschnittregel wurden nach den Diagnosen nicht verändert; eine mögliche Verbesserung (zum Beispiel eine andere Behandlung leerer Vorhersagen oder mehrerer Komponenten) wäre ein eigener, neu zu planender Schritt.
- **Explorativ:** Die Validation-Bilder dienten wiederholt der Entwicklung, auch der Wahl des U-Net-Checkpoints; die Größe einer möglichen Überschätzung ist nicht bestimmt. Die Schwellen der Sensitivitätsziele stammen aus denselben 385 Bildern, auf denen sie ausgewertet werden, und sind keine Schwellenwahl.
- **Ein Segmentierungsseed, kein neu trainierter Klassifikator:** keine Aussage zur Streuung der Segmentierung und keine zu einem auf automatischen Zuschnitten trainierten Klassifikator. Fünf Klassifikator-Seeds auf demselben Holdout sind Trainingsstreuung, keine fünf Kohorten; mehrere Bilder eines Falls sind abhängig.
- **Gruppen:** deskriptiv, mit Nennern; die Gruppen unterscheiden sich in vielen Merkmalen (Gerät, Klasse, Läsionsgröße); kleine Gruppen (U-Systems, sechs Bilder; Toshiba, 31 Bilder) werden nicht interpretiert; keine Fairnessaussage.
- **Eingabeverschiebung beim Fallback:** Der Klassifikator sieht bei einem Fallback ein ganzes Bild, auf dem er nicht trainiert wurde; die Scores solcher Bilder sind nicht mit denen der Zuschnitte vergleichbar und werden getrennt ausgewiesen.
- **Eingebrannte Texte und Messmarker** bleiben in den Zuschnitten; das ist nicht untersucht. Die Scores sind Sigmoid-Ausgaben, keine kalibrierten Wahrscheinlichkeiten. **GPU-Inferenz ist über Prozesse hinweg nicht in jedem Fall bitgleich:** in einer unabhängigen Neu-Inferenz des U-Nets (eigene Vorverarbeitung und Rückabbildung) waren bei einem Aufruf alle 385 vorhergesagten Masken identisch, bei einem anderen 332 von 385; die übrigen 53 Bilder wichen um höchstens drei Pixel ab (Dice ≥ 0,9994), und in keinem Fall änderte sich dadurch ein Zuschnitt (Box, Fallback). Die Ursache ist die Auswahl der cuDNN-Algorithmen und die TF32-Rundung; deshalb sind die Zuschnitte eingefroren und werden nicht neu berechnet, und die Prüfung verlangt die Gleichheit des abgeleiteten Zuschnitts, nicht die Bitgleichheit der Masken.
- **Keine endgültige Modell- oder Klassifikationsschwellenwahl, kein Testzugriff, keine Segmentierungs- oder Klassifikationsänderung.**

## Dateien und Reproduktion

- Skript und Module: `analysis/06_auto_roi.py` (Interpreterprüfung, Quellenprüfung, Plan, Phasen `--roi`, `--control`, `--classify`, `--diagnose`, Aggregation, Wiederaufruf), `plots/plot_auto_roi_results.py` (Abbildungen aus Tabellen, CPU, kein TensorFlow), `src/busbra/auto_roi.py` (Komponentenwahl, Zuschnitt, Diagnose, Tabellen, Plan; ohne TensorFlow).
- Aufruf (aus dem Projektordner): `env -u LD_LIBRARY_PATH .venv/bin/python analysis/06_auto_roi.py`; ein erneuter Aufruf validiert alle Phasen anhand der Hashes und führt keine Inferenz aus (Messung: wenige Sekunden). Laufzeiten dieses Laufs: ROI-Phase 11 s, Kontrolle, Klassifikation und Diagnose zusammen etwa eine Minute (Messung, keine allgemeine Schätzung).
- Ergebnisse: Plan `reports/tables/auto_roi_study_config.json`, Einfrierung `auto_roi_frozen.json`, Studienstatus `auto_roi_status.json` (+ `_last_success`), Phasenstatus und je Lauf `scores_auto.csv`, `scores_control.csv` und Status unter `reports/tables/auto_roi/`, Tabellen `auto_roi_run_metrics.csv`, `_summary_validation.csv`, `_paired_differences.csv`, `_paired_summary.csv`, `_threshold_curves.csv`, `_operating_points.csv`, `_operating_point_differences.csv`, `_operating_point_summary.csv`, `_crop_flags.csv`, `_crop_categories.csv`, `_coverage_summary.csv`, `_box_definitions.csv`, `_component_selection.csv`, `_notable_examples.csv`, `_concurrence.csv`; **nur lokal** (aus den Masken beziehungsweise Vorhersagen je Bild abgeleitet): `auto_roi_geometry_validation.csv`, `auto_roi_diagnostics_validation.csv`, `auto_roi/predicted_masks_validation.npz`; Abbildungen `reports/figures/auto_roi_validation_auc.png`, `auto_roi_false_alarms_at_targets.png`, `auto_roi_crop_coverage.png`.

## Nachtrag vom 20.10.2026: Wiederanlauf- und Provenienzschutz

**Anlass.** Nach Abschluss der Studie wurden zwei Schutzlücken des Wiederanlaufs in `analysis/06_auto_roi.py` festgestellt. (a) Fehlte der Status einer Phase oder eines Laufs, obwohl Ergebnisdateien vorhanden waren (eingefrorene ROIs, Score-Dateien, Diagnosetabellen), galt das Ergebnis als „nicht vorhanden“ und hätte bei einem erneuten Aufruf neu berechnet und überschrieben werden können. (b) Die bei der Planregistrierung neben dem Plan gespeicherten Hashes von Auswerteskript und `src/busbra/auto_roi.py` (`script_sha256_at_registration`, `new_modules_sha256_at_registration` in `reports/tables/auto_roi_study_config.json`) wurden später nicht mehr verglichen; ein geändertes Modul hätte unter demselben Plan-Hash einen anderen Zuschnitt erzeugen können. Die Studie selbst ist davon nicht berührt: Plan, Statuswerte, Modelle und alle Ergebnisdateien sind unverändert (Hashvergleich vor und nach der Korrektur), und die in der Planregistrierung gespeicherten Hashes stimmen mit den Codeständen überein, die in den vier Phasenstatus und im Studienstatus aufgezeichnet sind. Es wurden keine Zuschnittsregel, Maske, Schwelle, Komponentenwahl, Vorverarbeitung, Architektur, Gewichte, Seeds, Klassifikationsschwelle oder Metrik geändert; 4F-B bleibt explorativ. Die oben dokumentierte Offenlegung (isolierter Funktionstest vor der formalen Planregistrierung) bleibt unverändert gültig.

**Zustandsregeln des korrigierten Skripts.** Die Regeln gelten für Aufrufe ab diesem Nachtrag; der Studienlauf vom 20.10.2026 (21:49–21:51 Uhr) lief noch ohne sie.

| Gegenstand | Zustand | Verhalten |
|---|---|---|
| Studie mit vollständigem Erfolgsstatus | alle Ausgaben, Phasenzustände und aufgezeichneten Codeversionen stimmen | nur schreibgeschützte Prüfung (Hashes aller im Studienstatus aufgeführten 67 Ausgaben, Phasenzustände, Codeversionen); es wird nichts geschrieben, kein Bild geöffnet, TensorFlow nicht importiert; Exit 0 |
| | Abweichung (geänderte oder fehlende Ausgabe, verlorener Lauf) | Abbruch mit Diagnose; keine Reparatur, keine Neuberechnung |
| ROI-Phase | gültiger Erfolgsstatus | fertig, wird nicht neu berechnet |
| | kein gültiger Status, aber eine der drei Einfrierdateien (`auto_roi_frozen.json`, `auto_roi_geometry_validation.csv`, `auto_roi/predicted_masks_validation.npz`) oder ein späteres Ergebnis (Scores, Laufstatus, Diagnose- oder Aggregattabellen, Erfolgsstatus einer späteren Phase, Studienstatus mit Einfrierinformation) vorhanden | `recovery_required`: Abbruch vor dem Laden der Bilder und vor TensorFlow; nichts wird neu abgeleitet oder überschrieben |
| | kein gültiger Status und keine solche Datei (zum Beispiel abgebrochener erster Versuch mit Fehlerstatus) | darf nach den bisherigen Regeln neu starten; ein früherer Fehlerstatus bleibt als `roi_status_previous_<run>.json` erhalten |
| Lauf (Kontrolle, Klassifikation) | Score-Datei vorhanden, Laufstatus fehlt, ist unlesbar oder kein Erfolg | `recovery_required`: Abbruch, die Score-Datei wird nicht neu berechnet oder überschrieben |
| | Score-Datei und Laufstatus fehlen beide (gültiges Einfrieren und die übrigen Voraussetzungen vorhanden) | gezielter Wiederanlauf genau dieses Laufs; die übrigen Läufe bleiben byteidentisch |
| Diagnose | Diagnosedateien vorhanden, Status fehlt, unlesbar oder kein Erfolg | `recovery_required` |
| | eine aufgeführte Diagnosedatei fehlt oder ist verändert | `inconsistent` |
| | fertige Diagnose | wird auch bei direktem Aufruf (`--diagnose`) nicht neu berechnet; entsprechend wird eine fertige Kontroll- oder Klassifikationsphase bei direktem Aufruf nicht neu geschrieben |
| Aggregation | Ergebnisdateien vorhanden, Studienstatus fehlt oder ist unlesbar | `recovery_required` |
| Planregistrierung | fehlt, obwohl Ergebnisse oder Status der Studie existieren | Abbruch; es wird kein Plan (mit einem neuen Codehash) nachträglich registriert |

Eine unlesbare Statusdatei wird nie überschrieben. Die Fehlerprotokolle solcher Schutzabbrüche liegen als `<status>_failed_attempt_<run>.json` neben den Statusdateien (Feld `protective_stop: true`); die Statusdateien selbst und alle Ergebnisdateien bleiben unberührt. Ein Status, den eine fortgesetzte Phase ersetzen muss, bleibt als `<name>_previous_<run>.json` daneben erhalten.

**Codeversion.** Vor jeder Datenaufbereitung, Inferenz oder Ergebniserzeugung (ROI-Phase, Kontrolle, Klassifikation, Diagnose, Aggregation, auch bei direktem Phasenaufruf) werden das Auswerteskript und `src/busbra/auto_roi.py` mit den bei der Planregistrierung gespeicherten Hashes verglichen. Zusätzlich müssen die Codeversionen, die in den fertigen Erfolgsstatus aufgezeichnet sind, mit den registrierten übereinstimmen (eine nachträglich geänderte Registrierung fällt dadurch auf). Bei Abweichung bricht der Aufruf ab, bevor ein Bild gelesen oder etwas berechnet wird; das Protokoll nennt die Hashes. Die registrierten Hashes eines Plans werden nie aktualisiert. Eine tatsächlich nötige neue Berechnung setzt eine neue, ausdrücklich versionierte Planregistrierung voraus (eigener Schritt); sie ist nicht Teil dieser Korrektur.

**Was ein Aufruf des korrigierten Skripts mit dem historischen Plan noch darf.** Das korrigierte Skript hat den Hash `7dd47f8943b8eb5ba1d85a60c17ca5ae6925d93f179fdfb7a5537ce516c4e3c0`; der im historischen Plan registrierte Hash des Auswerteskripts ist `8e31e78a894fc66a3cb6563d50e93515e8279926a71942a9a46849f1115ec176`, der von `src/busbra/auto_roi.py` (unverändert) `60ffdb234593c24be3a843a693cc8498e85231583ec14e9387a2e877247c1c82`. Das Skript unterscheidet sich also vom registrierten Stand, und für den historischen Plan gilt: (1) Ein Aufruf ohne Argumente prüft den vollständigen historischen Lauf schreibgeschützt und meldet, dass der Code vom registrierten Stand abweicht (Exit 0, nichts geschrieben). (2) Jede Berechnung, auch ein direkter Aufruf einer Phase, stoppt am Codehash. (3) Ein Schutzabbruch schreibt nur das Fehlerprotokoll neben die Statusdateien. Die neuen Prüfungen liefen nicht im historischen Studienlauf, sondern nachträglich in isolierten Kopien.

**Nachweis.** 176 Prüfungen der neuen Testreihe sowie die 48 früheren Prüfungen der Teile A–C (Komponentenwahl, Tabellen, Schranken, Maskenunabhängigkeit) Prüfungen in isolierten Kopien (kein Produktivlauf, keine Änderung an Plan, Statuswerten, Modellen oder Ergebnissen): historischer Zustand schreibgeschützt geprüft, geänderte oder fehlende Ergebnisse erkannt; ROI-Status fehlt oder ist fehlgeschlagen bei vorhandenen oder teilweise vorhandenen Einfrierdateien; Laufstatus fehlt, ist unlesbar oder fehlgeschlagen bei vorhandener Score-Datei; Score-Datei und Status fehlen beide (gezielter Wiederanlauf zweier Läufe); Diagnosestatus fehlt bei vorhandenen Diagnosedateien; geänderter Modul- oder Skriptstand nach der Planregistrierung, nachträglich angepasste Registrierung, Registrierung ohne Hashes; ein vollständiger Lauf des korrigierten Skripts mit passender Registrierung und dessen schreibgeschützter Wiederaufruf. Zum Vergleich überschrieb das vorherige Skript in denselben Ausgangszuständen eingefrorene ROIs, Score-Dateien und Diagnosedateien, und es fror mit einem funktional geänderten Modul bei unverändertem Plan-Hash ROIs ein, die in 70 von 385 Bildern vom registrierten Stand abweichen.

**Grenzen.** Der Schutz bezieht sich auf versehentliches Überschreiben und unbeabsichtigte Codeänderungen; wer Schreibzugriff auf das Projekt hat, kann Dateien einschließlich Plan und Statuswerten absichtlich ändern (die aufgezeichneten Hashes machen Änderungen an fertigen Ergebnissen sichtbar). `plots/plot_auto_roi_results.py` ist nicht Teil der bei der Registrierung gespeicherten Hashes; die erzeugten Abbildungen sind aber in den Ausgabe-Hashes des Studienstatus erfasst. Die übrigen Module sind im Skript mit festen Hashes angebunden und deshalb im Skript-Hash enthalten.

**Ergänzung zum Nachtrag (20.10.2026): Aggregation mit fehlgeschlagenem Status.** In der Tabelle oben gilt für die Aggregation die genauere Regel: Sind Aggregationsergebnisse (die zehn Aggregattabellen oder die drei Abbildungen) vorhanden und ist der Studienstatus kein gültiger Erfolg, ist der Zustand `recovery_required` – ob der Status fehlt, unlesbar ist oder als lesbarer Status `failed` vorliegt. Der Aufruf bricht ohne Bildzugriff und ohne TensorFlow ab; Tabellen und Status bleiben bytegleich, nur das Fehlerprotokoll `auto_roi_status_failed_attempt_<run>.json` kommt hinzu. Ein fehlgeschlagener Status **ohne** Aggregationsergebnisse darf nach den bisherigen Regeln wiederaufgenommen werden (der frühere Status bleibt als `auto_roi_status_previous_<run>.json` erhalten). Beide Fälle wurden in isolierten Kopien geprüft. Der Hash des Skripts ist durch diese Ergänzung `41599e554596f2347ab1f1be7693d97f2a5b9561f7dd1c2e59217409a144df16` (statt des oben genannten); für den historischen Plan gilt unverändert, dass er vom registrierten Stand abweicht und nur schreibgeschützt geprüft wird.
