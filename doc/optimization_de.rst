Julia-Optimierung in eDisGo mit PowerModels
=======================================================================

Inhaltsverzeichnis
------------------

1. `Überblick <#überblick>`__
2. `Notation und Meta-Variablen <#notation-und-meta-variablen>`__
3. `Alle Julia-Variablen
   (Tabellarisch) <#alle-julia-variablen-tabellarisch>`__
4. `Zeitliche Einordnung der
   Optimierung <#zeitliche-einordnung-der-optimierung>`__
5. `Die analyze-Funktion <#die-analyze-funktion>`__
6. `Die reinforce-Funktion <#die-reinforce-funktion>`__
7. `Die §14a EnWG Optimierung <#die-14a-enwg-optimierung>`__
8. `Zeitreihen-Nutzung <#zeitreihen-nutzung>`__
9. `Dateipfade und Referenzen <#dateipfade-und-referenzen>`__

--------------

Überblick
---------

Die Julia-Optimierung in eDisGo verwendet **PowerModels.jl** zur Lösung
von Optimal Power Flow (OPF) Problemen. Der Workflow erfolgt über eine
Python-Julia-Schnittstelle:

-  **Python (eDisGo)**: Netzmodellierung, Zeitreihen,
   Ergebnisverarbeitung
-  **Julia (PowerModels)**: Mathematische Optimierung, Solver-Interface
-  **Kommunikation**: JSON über stdin/stdout

**Optimierungsziele:** - Minimierung von Netzverlusten - Einhaltung von
Spannungs- und Stromgrenzen - Flexibilitätsnutzung (Speicher,
Wärmepumpen, E-Autos, DSM) - Optional: §14a EnWG Abregelung mit
Zeitbudget-Constraints

--------------

Notation und Meta-Variablen
---------------------------

Bevor wir die konkreten Optimierungsvariablen betrachten, hier eine
Übersicht über die **allgemeinen Variablen und Notation**, die im
Julia-Code verwendet werden:

Meta-Variablen (nicht Teil des Optimierungsproblems)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------+-------+-----------------------+--------------------+
| Variable       | Typ   | Beschreibung          | Verwendung         |
+================+=======+=======================+====================+
| ``pm``         | ``    | PowerModels-Objekt    | Enthält das        |
|                | Abstr |                       | gesamte            |
|                | actPo |                       | O                  |
|                | werMo |                       | ptimierungsproblem |
|                | del`` |                       | (Netz, Variablen,  |
|                |       |                       | Constraints)       |
+----------------+-------+-----------------------+--------------------+
| ``nw`` oder    | ``    | Network-ID            | Identifiziert      |
| ``n``          | Int`` | (Zeitschritt-Index)   | einen Zeitschritt  |
|                |       |                       | im                 |
|                |       |                       | Mu                 |
|                |       |                       | lti-Period-Problem |
|                |       |                       | (0, 1, 2, …, T-1)  |
+----------------+-------+-----------------------+--------------------+
| ``nw_ids(pm)`` | ``Ar  | Alle Network-IDs      | Gibt alle          |
|                | ray{I |                       | Z                  |
|                | nt}`` |                       | eitschritt-Indizes |
|                |       |                       | zurück, z.B.       |
|                |       |                       | ``[0,              |
|                |       |                       | 1, 2, ..., 8759]`` |
|                |       |                       | für 8760h          |
+----------------+-------+-----------------------+--------------------+
| `              | ``D   | Referenzdaten für     | Zugriff auf        |
| `ref(pm, nw)`` | ict`` | Zeitschritt           | Netzdaten eines    |
|                |       |                       | bestimmten         |
|                |       |                       | Zeitschritts       |
+----------------+-------+-----------------------+--------------------+
| `              | ``D   | Variablen-Dictionary  | Zugriff auf        |
| `var(pm, nw)`` | ict`` |                       | Opt                |
|                |       |                       | imierungsvariablen |
|                |       |                       | eines Zeitschritts |
+----------------+-------+-----------------------+--------------------+
| ``model`` oder | ``Ju  | Ju                    | Das                |
| ``pm.model``   | MP.Mo | MP-Optimierungsmodell | zugrundeliegende   |
|                | del`` |                       | mathematische      |
|                |       |                       | Optimierungsmodell |
+----------------+-------+-----------------------+--------------------+

Index-Variablen
~~~~~~~~~~~~~~~

+---------------+----------------+---------------------+---------------+
| Variable      | Bedeutung      | Beschreibung        | Beispiel      |
+===============+================+=====================+===============+
| ``i``, ``j``  | Bus-Index      | Identifiziert       | ``i=1`` = Bus |
|               |                | Knoten im Netzwerk  | “Bus_MV_123”  |
+---------------+----------------+---------------------+---------------+
| ``l``         | Branch-Index   | Identifiziert       | ``l=5`` =     |
|               | (Lin           | Leitungen und       | Leitung       |
|               | e/Transformer) | Transformatoren     | “Line_LV_456” |
+---------------+----------------+---------------------+---------------+
| ``g``         | G              | Identifiziert       | ``g=3`` =     |
|               | enerator-Index | Generatoren (PV,    | “PV_001”      |
|               |                | Wind, BHKW, Slack)  |               |
+---------------+----------------+---------------------+---------------+
| ``s``         | Storage-Index  | Identifiziert       | ``s=1`` =     |
|               |                | Batteriespeicher    | “Storage_1”   |
+---------------+----------------+---------------------+---------------+
| ``h``         | Heat           | Identifiziert       | ``h=2`` =     |
|               | Pump-Index     | Wärmepumpen         | “HP_LV_789”   |
+---------------+----------------+---------------------+---------------+
| ``c``         | Charging       | Identifiziert       | ``c=4`` =     |
|               | Point-Index    | Ladepunkte für      | “CP_LV_101”   |
|               |                | E-Autos             |               |
+---------------+----------------+---------------------+---------------+
| ``d``         | DSM-Index      | Identifiziert       | ``d=1`` =     |
|               |                | DSM-Lasten          | “DSM_Load_1”  |
+---------------+----------------+---------------------+---------------+
| ``t`` oder    | Zei            | Zeitpunkt im        | ``t=0`` =     |
| ``n``         | tschritt-Index | O                   | 2035-01-01    |
|               |                | ptimierungshorizont | 00:00,        |
|               |                |                     | ``t=1`` =     |
|               |                |                     | 01:00, …      |
+---------------+----------------+---------------------+---------------+

PowerModels-Funktionen
~~~~~~~~~~~~~~~~~~~~~~

+--------------+--------------------+--------------------+--------------+
| Funktion     | Rückgabewert       | Beschreibung       | Beispiel     |
+==============+====================+====================+==============+
| ``ids(pm, :  | ``Array{Int}``     | Gibt alle Bus-IDs  | ``[1, 2, 3,  |
| bus, nw=n)`` |                    | für Zeitschritt n  |  ..., 150]`` |
|              |                    | zurück             |              |
+--------------+--------------------+--------------------+--------------+
| ``           | ``Array{Int}``     | Gibt alle          | ``[1, 2, 3,  |
| ids(pm, :bra |                    | Branch-IDs         |  ..., 200]`` |
| nch, nw=n)`` |                    | (Leitungen/Trafos) |              |
|              |                    | zurück             |              |
+--------------+--------------------+--------------------+--------------+
| ``ids(pm, :  | ``Array{Int}``     | Gibt alle          | ``[1, 2, 3   |
| gen, nw=n)`` |                    | Generator-IDs      | , ..., 50]`` |
|              |                    | zurück             |              |
+--------------+--------------------+--------------------+--------------+
| ``i          | ``Array{Int}``     | Gibt alle          | `            |
| ds(pm, :stor |                    | Storage-IDs zurück | `[1, 2, 3]`` |
| age, nw=n)`` |                    |                    |              |
+--------------+--------------------+--------------------+--------------+
| ``ref(pm, nw | ``Dict``           | Gibt Daten für Bus | ``{"vmin":   |
| , :bus, i)`` |                    | i in Zeitschritt   | 0.9, "vmax": |
|              |                    | nw                 |  1.1, ...}`` |
+--------------+--------------------+--------------------+--------------+
| ``r          | ``Dict``           | Gibt Daten für     | ``           |
| ef(pm, nw, : |                    | Branch l in        | {"rate_a": 0 |
| branch, l)`` |                    | Zeitschritt nw     | .5, "br_r":  |
|              |                    |                    | 0.01, ...}`` |
+--------------+--------------------+--------------------+--------------+
| ``var(pm,    | ``JuMP.Variable``  | Gibt               | JuMP-Var     |
| nw, :p, l)`` |                    | Wir                | iable-Objekt |
|              |                    | kleistungsvariable |              |
|              |                    | für Branch l       |              |
|              |                    | zurück             |              |
+--------------+--------------------+--------------------+--------------+
| ``var(pm,    | ``JuMP.Variable``  | Gibt               | JuMP-Var     |
| nw, :w, i)`` |                    | Spannungsvariable  | iable-Objekt |
|              |                    | für Bus i zurück   |              |
+--------------+--------------------+--------------------+--------------+

Typische Code-Muster
~~~~~~~~~~~~~~~~~~~~

1. Iteration über alle Zeitschritte
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   for n in nw_ids(pm)
       # Code für Zeitschritt n
       println("Verarbeite Zeitschritt $n")
   end

2. Iteration über alle Busse in einem Zeitschritt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   for i in ids(pm, :bus, nw=n)
       # Code für Bus i in Zeitschritt n
       bus_data = ref(pm, n, :bus, i)
       println("Bus $i: Vmin = $(bus_data["vmin"])")
   end

3. Zugriff auf Variablen
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   # Variable abrufen
   w_i = var(pm, n, :w, i)  # Spannungsvariable für Bus i, Zeitschritt n

   # Variable in Constraint verwenden
   JuMP.@constraint(pm.model, w_i >= 0.9^2)  # Untere Spannungsgrenze

4. Variable erstellen und im Dictionary speichern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   # Variablen-Dictionary für Zeitschritt n initialisieren
   var(pm, n)[:p_hp14a] = JuMP.@variable(
       pm.model,
       [h in ids(pm, :gen_hp_14a, nw=n)],
       base_name = "p_hp14a_$(n)",
       lower_bound = 0.0
   )

   # Später darauf zugreifen
   for h in ids(pm, :gen_hp_14a, nw=n)
       p_hp14a_h = var(pm, n, :p_hp14a, h)
   end

Multi-Network-Struktur
~~~~~~~~~~~~~~~~~~~~~~

PowerModels verwendet eine **Multi-Network-Struktur** für zeitabhängige
Optimierung:

::

   pm (PowerModel)
   ├─ nw["0"]  (Zeitschritt 0: 2035-01-01 00:00)
   │  ├─ :bus → {1: {...}, 2: {...}, ...}          ← Alle 150 Busse
   │  ├─ :branch → {1: {...}, 2: {...}, ...}       ← Alle 200 Leitungen/Trafos
   │  ├─ :gen → {1: {...}, 2: {...}, ...}          ← Alle 50 Generatoren
   │  ├─ :load → {1: {...}, 2: {...}, ...}         ← Alle 120 Lasten
   │  └─ :storage → {1: {...}, 2: {...}, ...}      ← Alle 5 Speicher
   │
   ├─ nw["1"]  (Zeitschritt 1: 2035-01-01 01:00)
   │  ├─ :bus → {1: {...}, 2: {...}, ...}          ← WIEDER alle 150 Busse
   │  ├─ :branch → {1: {...}, 2: {...}, ...}       ← WIEDER alle 200 Leitungen
   │  └─ ...                                        ← usw.
   │
   ├─ nw["2"]  (Zeitschritt 2: 2035-01-01 02:00)
   │  ├─ :bus → {1: {...}, 2: {...}, ...}          ← WIEDER alle 150 Busse
   │  └─ ...
   │
   ├─ ...  (8757 weitere Zeitschritte)
   │
   └─ nw["8759"]  (Zeitschritt 8759: 2035-12-31 23:00)
      └─ Komplettes Netz nochmal

**WICHTIG: Das Netz existiert T-mal!**

Für einen Optimierungshorizont von **8760 Stunden** (1 Jahr) bedeutet
das: - Das gesamte Netz wird **8760-mal dupliziert** - Jeder Zeitschritt
hat seine eigene vollständige Netz-Kopie - Alle Busse, Leitungen,
Trafos, Generatoren, Lasten existieren **8760-mal** - Jeder Zeitschritt
hat **eigene Optimierungsvariablen**

**Was unterscheidet die Zeitschritte?**

+--------+----------------------+--------------------------------------+
| Aspekt | Zeitschritte         | Unterschiedlich pro Zeitschritt      |
+========+======================+======================================+
| **Net  | Identisch            | Gleiche Busse, Leitungen, Trafos     |
| ztopol |                      |                                      |
| ogie** |                      |                                      |
+--------+----------------------+--------------------------------------+
| **Net  | Identisch            | Gleiche Widerstände, Kapazitäten     |
| zparam |                      |                                      |
| eter** |                      |                                      |
+--------+----------------------+--------------------------------------+
| **     | Unterschiedlich      | Generator-Einspeisung, Lasten, COP   |
| Zeitre |                      |                                      |
| ihen-W |                      |                                      |
| erte** |                      |                                      |
+--------+----------------------+--------------------------------------+
| *      | Unterschiedlich      | Spannungen, Leistungsflüsse,         |
| *Varia |                      | Speicher-Leistung                    |
| blen** |                      |                                      |
+--------+----------------------+--------------------------------------+
| **Sp   | Gekoppelt            | SOC[t+1] hängt von SOC[t] ab         |
| eicher |                      |                                      |
| -SOC** |                      |                                      |
+--------+----------------------+--------------------------------------+

**Beispiel: Wirkleistungsvariable p[l,i,j]**

Für eine Leitung ``l=5`` zwischen Bus ``i=10`` und ``j=11``: -
``var(pm, 0, :p)[(5,10,11)]`` = Wirkleistung in Zeitschritt 0 (00:00
Uhr) - ``var(pm, 1, :p)[(5,10,11)]`` = Wirkleistung in Zeitschritt 1
(01:00 Uhr) - ``var(pm, 2, :p)[(5,10,11)]`` = Wirkleistung in
Zeitschritt 2 (02:00 Uhr) - … - ``var(pm, 8759, :p)[(5,10,11)]`` =
Wirkleistung in Zeitschritt 8759 (23:00 Uhr)

→ **8760 verschiedene Variablen** für dieselbe Leitung!

**Optimierungsproblem-Größe:**

Für ein Netz mit: - 150 Busse - 200 Leitungen/Trafos - 50 Generatoren -
5 Batteriespeicher - 20 Wärmepumpen - 10 Ladepunkte - 8760 Zeitschritte
(1 Jahr, 1h-Auflösung)

**Anzahl Variablen (grob):** - Spannungen: 150 Busse × 8760 Zeitschritte
= **1,314,000 Variablen** - Leitungsflüsse: 200 × 2 (p,q) × 8760 =
**3,504,000 Variablen** - Generatoren: 50 × 2 (p,q) × 8760 = **876,000
Variablen** - Speicher: 5 × 2 (Leistung + SOC) × 8760 = **87,600
Variablen** - …

→ **Mehrere Millionen Variablen** für Jahressimulation!

**Warum dieser Ansatz?**

**Vorteile:** - Erlaubt zeitgekoppelte Optimierung (Speicher,
Wärmepumpen) - PowerModels-Syntax bleibt einfach (jeder Zeitschritt wie
Einzelproblem) - Flexible Zeitreihen (unterschiedliche Werte pro
Zeitschritt)

**Nachteile:** - Sehr großes Optimierungsproblem (Millionen Variablen) -
Hoher Speicherbedarf - Lange Lösungszeiten (Minuten bis Stunden)

**Inter-Zeitschritt Constraints:**

Bestimmte Constraints koppeln die Zeitschritte:

.. code:: julia

   # Speicher-Energiekopplung
   for n in 0:8758  # Alle Zeitschritte außer letzter
       for s in storage_ids
           # SOC in t+1 hängt von SOC in t und Leistung in t ab
           @constraint(pm.model,
               var(pm, n+1, :se, s) ==
               var(pm, n, :se, s) + var(pm, n, :ps, s) × Δt × η
           )
       end
   end

→ Diese Constraints verbinden die sonst unabhängigen Zeitschritte!

**Zusammenfassung:** - Jeder Zeitschritt hat eine **eigene vollständige
Kopie** des Netzes - Zeitreihen-Werte (Lasten, Einspeisung)
unterscheiden sich zwischen Zeitschritten - Variablen existieren **pro
Zeitschritt** (8760-mal für jede physikalische Variable!) -
Inter-zeitschritt Constraints (Speicher-SOC, Wärmespeicher) koppeln die
Zeitschritte - **Für 8760 Zeitschritte:** Das Netz existiert 8760-mal →
Millionen von Variablen

--------------

Alle Julia-Variablen (Tabellarisch)
-----------------------------------

Netz-Variablen (Grid Variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------+---------------+------------+--------------------+
| Variable         | Dimension     | Einheit    | Beschreibung       |
+==================+===============+============+====================+
| ``p[l,i,j]``     | ℝ             | MW         | Wirkleistungsfluss |
|                  |               |            | auf Leitung/Trafo  |
|                  |               |            | von Bus i zu Bus j |
+------------------+---------------+------------+--------------------+
| ``q[l,i,j]``     | ℝ             | MVAr       | B                  |
|                  |               |            | lindleistungsfluss |
|                  |               |            | auf Leitung/Trafo  |
|                  |               |            | von Bus i zu Bus j |
+------------------+---------------+------------+--------------------+
| ``w[i]``         | ℝ⁺            | p.u.²      | Quadrierte         |
|                  |               |            | Spannungsamplitude |
|                  |               |            | an Bus i           |
+------------------+---------------+------------+--------------------+
| ``ccm[l,i,j]``   | ℝ⁺            | kA²        | Quadrierte         |
|                  |               |            | Stromstärke auf    |
|                  |               |            | Leitung/Trafo      |
+------------------+---------------+------------+--------------------+
| ``ll[l,i,j]``    | [0,1]         | -          | Leitungsauslastung |
|                  |               |            | (nur OPF Version 1 |
|                  |               |            | & 3)               |
+------------------+---------------+------------+--------------------+

**Hinweise:** - ``l`` = Leitungs-/Trafo-ID - ``i,j`` = Bus-IDs
(from_bus, to_bus) - Quadrierte Variablen vermeiden nichtkonvexe
Wurzelfunktionen

--------------

Generator-Variablen (Generation Variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------+-----------------+-------------+---------------------+
| Variable      | Dimension       | Einheit     | Beschreibung        |
+===============+=================+=============+=====================+
| ``pg[g]``     | ℝ               | MW          | Wirkl               |
|               |                 |             | eistungseinspeisung |
|               |                 |             | Generator g         |
+---------------+-----------------+-------------+---------------------+
| ``qg[g]``     | ℝ               | MVAr        | Blindl              |
|               |                 |             | eistungseinspeisung |
|               |                 |             | Generator g         |
+---------------+-----------------+-------------+---------------------+
| ``pgc[g]``    | ℝ⁺              | MW          | Abregelung          |
|               |                 |             | nicht-regelbarer    |
|               |                 |             | Generatoren         |
|               |                 |             | (Curtailment)       |
+---------------+-----------------+-------------+---------------------+
| ``pgs``       | ℝ               | MW          | Slack-Generator     |
|               |                 |             | Wirkleistung        |
|               |                 |             | (Netzanschluss)     |
+---------------+-----------------+-------------+---------------------+
| ``qgs``       | ℝ               | MVAr        | Slack-Generator     |
|               |                 |             | Blindleistung       |
+---------------+-----------------+-------------+---------------------+

**Hinweise:** - Slack-Generator repräsentiert Übertragungsnetz-Anschluss
- Curtailment nur für EE-Anlagen (PV, Wind)

--------------

Batteriespeicher-Variablen (Battery Storage Variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------+-----------------+-------------+---------------------+
| Variable      | Dimension       | Einheit     | Beschreibung        |
+===============+=================+=============+=====================+
| ``ps[s,t]``   | ℝ               | MW          | Wirkleistung        |
|               |                 |             | Batteriespeicher s  |
|               |                 |             | zum Zeitpunkt t (+  |
|               |                 |             | = Entladung, - =    |
|               |                 |             | Ladung)             |
+---------------+-----------------+-------------+---------------------+
| ``qs[s,t]``   | ℝ               | MVAr        | Blindleistung       |
|               |                 |             | Batteriespeicher s  |
+---------------+-----------------+-------------+---------------------+
| ``se[s,t]``   | ℝ⁺              | MWh         | Energieinhalt       |
|               |                 |             | (State of Energy)   |
|               |                 |             | Batteriespeicher s  |
+---------------+-----------------+-------------+---------------------+

**Constraints:** - SOC-Kopplung zwischen Zeitschritten:
``se[t+1] = se[t] + ps[t] × Δt × η`` - Kapazitätsgrenzen:
``se_min ≤ se[t] ≤ se_max`` - Leistungsgrenzen:
``ps_min ≤ ps[t] ≤ ps_max``

--------------

Wärmepumpen-Variablen (Heat Pump Variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------+-----------------+-------------+---------------------+
| Variable      | Dimension       | Einheit     | Beschreibung        |
+===============+=================+=============+=====================+
| ``php[h,t]``  | ℝ⁺              | MW          | Elektrische         |
|               |                 |             | Leistungsaufnahme   |
|               |                 |             | Wärmepumpe h        |
+---------------+-----------------+-------------+---------------------+
| ``qhp[h,t]``  | ℝ               | MVAr        | Blindleistung       |
|               |                 |             | Wärmepumpe h        |
+---------------+-----------------+-------------+---------------------+
| ``phs[h,t]``  | ℝ               | MW          | Leistung            |
|               |                 |             | Wärmespeicher h (+  |
|               |                 |             | = Beladung, - =     |
|               |                 |             | Entladung)          |
+---------------+-----------------+-------------+---------------------+
| ``hse[h,t]``  | ℝ⁺              | MWh         | Energieinhalt       |
|               |                 |             | Wärmespeicher h     |
+---------------+-----------------+-------------+---------------------+

**Hinweise:** - Wärmepumpen mit thermischem Speicher können zeitlich
verschoben werden - Wärmebedarf muss über Optimierungshorizont gedeckt
werden - COP (Coefficient of Performance) verknüpft elektrische und
thermische Leistung

--------------

Ladepunkt-Variablen (Charging Point / EV Variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------+----------------+-------------+---------------------+
| Variable       | Dimension      | Einheit     | Beschreibung        |
+================+================+=============+=====================+
| ``pcp[c,t]``   | ℝ⁺             | MW          | Ladeleistung        |
|                |                |             | Ladepunkt c zum     |
|                |                |             | Zeitpunkt t         |
+----------------+----------------+-------------+---------------------+
| ``qcp[c,t]``   | ℝ              | MVAr        | Blindleistung       |
|                |                |             | Ladepunkt c         |
+----------------+----------------+-------------+---------------------+
| ``cpe[c,t]``   | ℝ⁺             | MWh         | Energieinhalt       |
|                |                |             | Fahrzeugbatterie am |
|                |                |             | Ladepunkt c         |
+----------------+----------------+-------------+---------------------+

**Constraints:** - Energiekopplung:
``cpe[t+1] = cpe[t] + pcp[t] × Δt × η`` - Kapazität:
``cpe_min ≤ cpe[t] ≤ cpe_max`` - Ladeleistung: ``0 ≤ pcp[t] ≤ pcp_max``

--------------

DSM-Variablen (Demand Side Management Variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============= ========= ======= =====================================
Variable      Dimension Einheit Beschreibung
============= ========= ======= =====================================
``pdsm[d,t]`` ℝ⁺        MW      Verschiebbare Last d zum Zeitpunkt t
``qdsm[d,t]`` ℝ         MVAr    Blindleistung DSM-Last d
``dsme[d,t]`` ℝ⁺        MWh     Virtueller Energieinhalt DSM-Speicher
============= ========= ======= =====================================

**Hinweise:** - DSM modelliert verschiebbare Lasten (z.B.
Industrieprozesse) - Gesamtenergie über Horizont bleibt konstant

--------------

Slack-Variablen für Netzrestriktionen (Slack Variables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nur in **OPF Version 2 & 4** (mit Netzrestriktionen):

+----------------+-----------+---------+-------------------------------------------+
| Variable       | Dimension | Einheit | Beschreibung                              |
+================+===========+=========+===========================================+
| ``phps[h,t]``  | ℝ⁺        | MW      | Slack für Wärmepumpen-Restriktion         |
+----------------+-----------+---------+-------------------------------------------+
| ``phps2[h,t]`` | ℝ⁺        | MW      | Slack für Wärmepumpen-Betriebsrestriktion |
+----------------+-----------+---------+-------------------------------------------+
| ``phss[h,t]``  | ℝ⁺        | MW      | Slack für Wärmespeicher-Restriktion       |
+----------------+-----------+---------+-------------------------------------------+
| ``pds[d,t]``   | ℝ⁺        | MW      | Lastabwurf (Load Shedding)                |
+----------------+-----------+---------+-------------------------------------------+
| ``pgens[g,t]`` | ℝ⁺        | MW      | Slack für Generator-Abregelung            |
+----------------+-----------+---------+-------------------------------------------+
| ``pcps[c,t]``  | ℝ⁺        | MW      | Slack für Ladepunkt-Restriktion           |
+----------------+-----------+---------+-------------------------------------------+
| ``phvs[t]``    | ℝ⁺        | MW      | Slack für Hochspannungs-Anforderungen     |
+----------------+-----------+---------+-------------------------------------------+

**Zweck:** - Gewährleisten Lösbarkeit des Optimierungsproblems - Hohe
Kosten im Zielfunktional → werden minimiert - Zeigen an, wo
Netzrestriktionen nicht eingehalten werden können

--------------

§14a EnWG Variablen (NEU)
~~~~~~~~~~~~~~~~~~~~~~~~~

Nur wenn ``curtailment_14a=True``:

Wärmepumpen §14a
^^^^^^^^^^^^^^^^

+----------------+----------------+-------------+---------------------+
| Variable       | Dimension      | Einheit     | Beschreibung        |
+================+================+=============+=====================+
| ``             | ℝ⁺             | MW          | Virtueller          |
| p_hp14a[h,t]`` |                |             | Generator für       |
|                |                |             | WP-Abregelung (0    |
|                |                |             | bis pmax)           |
+----------------+----------------+-------------+---------------------+
| ``             | Binär          | {0,1}       | -                   |
| z_hp14a[h,t]`` |                |             |                     |
+----------------+----------------+-------------+---------------------+

Ladepunkte §14a
^^^^^^^^^^^^^^^

+----------------+----------------+-------------+---------------------+
| Variable       | Dimension      | Einheit     | Beschreibung        |
+================+================+=============+=====================+
| ``             | ℝ⁺             | MW          | Virtueller          |
| p_cp14a[c,t]`` |                |             | Generator für       |
|                |                |             | CP-Abregelung (0    |
|                |                |             | bis pmax)           |
+----------------+----------------+-------------+---------------------+
| ``             | Binär          | {0,1}       | -                   |
| z_cp14a[c,t]`` |                |             |                     |
+----------------+----------------+-------------+---------------------+

**Wichtige Parameter:** - ``pmax = P_nominal - P_min_14a`` (maximale
Abregelleistung) - ``P_min_14a = 0.0042 MW`` (4.2 kW Mindestleistung
gemäß §14a) - ``max_hours_per_day`` (z.B. 2h/Tag Zeitbudget)

**Funktionsweise:** - Virtueller Generator “erzeugt” Leistung am
WP/CP-Bus - Effekt: Nettolast = Original-Last - p_hp14a - Simuliert
Abregelung ohne komplexe Lastanpassung

--------------

Zeitliche Einordnung der Optimierung
------------------------------------

Gesamter Workflow
~~~~~~~~~~~~~~~~~

**WICHTIGER HINWEIS zum Workflow:** - **Reinforce VOR der Optimierung:**
Nur optional und sinnvoll für das Basisnetz (z.B. ohne
Wärmepumpen/E-Autos). Wenn man vor der Optimierung bereits das komplette
Netz ausbaut, gibt es keine Überlastungen mehr und die Optimierung zur
Flexibilitätsnutzung macht keinen Sinn. - **Reinforce NACH der
Optimierung:** In der Regel erforderlich! Die Optimierung nutzt
Flexibilität um Netzausbau zu minimieren, kann aber nicht alle Probleme
lösen. Verbleibende Überlastungen und Spannungsverletzungen müssen durch
konventionellen Netzausbau behoben werden.

**Typischer Anwendungsfall:** 1. Basisnetz laden (z.B. Ist-Zustand ohne
neue Wärmepumpen) 2. Optional: reinforce auf Basisnetz 3. (Neue)
Komponenten hinzufügen (Wärmepumpen, E-Autos für Zukunftsszenario) 4.
Optimierung durchführen → nutzt Flexibilität statt Netzausbau 5.
**Zwingend:** reinforce mit optimierten Zeitreihen → behebt verbleibende
Probleme

::

   ┌─────────────────────────────────────────────────────────────────────┐
   │ 1. INITIALISIERUNG                                                  │
   ├─────────────────────────────────────────────────────────────────────┤
   │ - Netzladung (ding0-Netz oder Datenbank)                           │
   │ - Import Zeitreihen (Generatoren, Lasten ohne neue Komponenten)    │
   │ - Konfiguration Optimierungsparameter                              │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 2. BASISNETZ-VERSTÄRKUNG (optional)                                 │
   ├─────────────────────────────────────────────────────────────────────┤
   │ edisgo.reinforce()                                                  │
   │ - Verstärkung des Basisnetzes (OHNE neue WP/CP)                    │
   │ - Sinnvoll als Referenzszenario                                    │
   │ - Erstellt Ausgangsbasis für Szenariovergleich                     │
   │                                                                     │
   │ WICHTIG: Dies ist NICHT der Hauptverstärkungsschritt!              │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 3. NEUE KOMPONENTEN HINZUFÜGEN                                      │
   ├─────────────────────────────────────────────────────────────────────┤
   │ - Wärmepumpen hinzufügen (mit thermischen Speichern)               │
   │ - E-Auto Ladepunkte hinzufügen (mit Flexibilitätsbändern)          │
   │ - Batteriespeicher hinzufügen                                      │
   │ - Zeitreihen für neue Komponenten setzen                           │
   │                                                                     │
   │ → Netz ist jetzt wahrscheinlich überlastet                         │
   │ → KEIN reinforce an dieser Stelle!                                 │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 4. JULIA-OPTIMIERUNG                                                │
   ├─────────────────────────────────────────────────────────────────────┤
   │ edisgo.pm_optimize(opf_version=2, curtailment_14a=True)            │
   │                                                                     │
   │ ZIEL: Flexibilität nutzen um Netzausbau zu VERMEIDEN               │
   │ - Batteriespeicher optimal laden/entladen                          │
   │ - Wärmepumpen zeitlich verschieben (thermischer Speicher)          │
   │ - E-Auto-Ladung optimieren (innerhalb Flexibilitätsband)           │
   │ - §14a Abregelung bei Engpässen (max. 2h/Tag)                     │
   │                                                                     │
   │ 4.1 PYTHON → POWERMODELS KONVERTIERUNG                             │
   │     ├─ to_powermodels(): Netz → PowerModels-Dictionary            │
   │     ├─ Zeitreihen für alle Komponenten                            │
   │     ├─ Falls 14a: Virtuelle Generatoren für WP/CP erstellen       │
   │     └─ Serialisierung zu JSON                                     │
   │                                                                     │
   │ 4.2 PYTHON → JULIA KOMMUNIKATION                                   │
   │     ├─ Starte Julia-Subprozess: julia Main.jl [args]              │
   │     ├─ Übergabe JSON via stdin                                    │
   │     └─ Args: grid_name, results_path, method (soc/nc), etc.       │
   │                                                                     │
   │ 4.3 JULIA-OPTIMIERUNG                                              │
   │     ├─ Parse JSON → PowerModels Multinetwork                      │
   │     ├─ Solver-Auswahl: Gurobi (SOC) oder IPOPT (NC)               │
   │     ├─ build_mn_opf_bf_flex():                                    │
   │     │   ├─ Variablen erstellen (alle aus Tabellen oben)           │
   │     │   ├─ Constraints pro Zeitschritt:                           │
   │     │   │   ├─ Leistungsbilanz an Knoten                          │
   │     │   │   ├─ Spannungsfallgleichungen                           │
   │     │   │   ├─ Stromgleichungen                                   │
   │     │   │   ├─ Speicher-/WP-/CP-Zustandsgleichungen               │
   │     │   │   ├─ §14a Binär-Kopplung (falls aktiviert)              │
   │     │   │   └─ §14a Mindest-Nettolast (falls aktiviert)           │
   │     │   ├─ Inter-Zeitschritt Constraints:                         │
   │     │   │   ├─ Energiekopplung Speicher/WP/CP                     │
   │     │   │   └─ §14a Tages-Zeitbudget (falls aktiviert)            │
   │     │   └─ Zielfunktion setzen (versionsabhängig)                 │
   │     ├─ Optimierung lösen                                           │
   │     ├─ Ergebnisse zu JSON serialisieren                           │
   │     └─ Output via stdout                                          │
   │                                                                     │
   │ 4.4 JULIA → PYTHON KOMMUNIKATION                                   │
   │     ├─ Python liest stdout zeilenweise                            │
   │     ├─ Erfasse JSON-Ergebnis (beginnt mit {"name")                │
   │     └─ Parse JSON zu Dictionary                                   │
   │                                                                     │
   │ 4.5 POWERMODELS → EDISGO KONVERTIERUNG                             │
   │     ├─ from_powermodels(): Extrahiere optimierte Zeitreihen       │
   │     ├─ Schreibe zu edisgo.timeseries:                             │
   │     │   ├─ generators_active_power, generators_reactive_power     │
   │     │   ├─ storage_units_active_power (optimiert)                 │
   │     │   ├─ heat_pump_loads (zeitlich verschoben)                  │
   │     │   ├─ charging_point_loads (optimiert)                       │
   │     │   └─ §14a Abregelung als virtuelle Generatoren:             │
   │     │       ├─ hp_14a_support_{name}                              │
   │     │       └─ cp_14a_support_{name}                              │
   │     └─ Abregelung = Virtuelle Generator-Leistung                  │
   │                                                                     │
   │ ERGEBNIS: Optimierte Zeitreihen mit minimiertem Netzausbaubedarf   │
   │          Aber: Evtl. verbleibende Überlastungen (Slacks > 0)      │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 5. NETZAUSBAU MIT OPTIMIERTEN ZEITREIHEN (ZWINGEND!)               │
   ├─────────────────────────────────────────────────────────────────────┤
   │ edisgo.reinforce()                                                  │
   │                                                                     │
   │ WICHTIG: Dieser Schritt ist ZWINGEND erforderlich!                 │
   │                                                                     │
   │ Warum?                                                              │
   │ - Optimierung nutzt Flexibilität, kann aber nicht alle Probleme    │
   │   lösen (z.B. Netzrestriktionen, zu geringe Flexibilität)          │
   │ - Slack-Variablen > 0 zeigen verbleibende Verletzungen             │
   │ - Verbleibende Überlastungen müssen durch Netzausbau behoben       │
   │   werden                                                            │
   │                                                                     │
   │ Ablauf:                                                             │
   │ - Iterative Verstärkungsmaßnahmen                                  │
   │ - Leitungsausbau, Trafoausbau                                      │
   │ - Berechnung Netzausbaukosten                                      │
   │                                                                     │
   │ ERGEBNIS: Netzausbaukosten NACH Flexibilitätsnutzung               │
   │          (deutlich geringer als ohne Optimierung!)                 │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 6. AUSWERTUNG                                                       │
   ├─────────────────────────────────────────────────────────────────────┤
   │ - Analyse optimierter Zeitreihen                                   │
   │ - Berechnung §14a Statistiken (Abregelenergie, Zeitbudget-Nutzung) │
   │ - Vergleich Netzausbaukosten (mit vs. ohne Optimierung)            │
   │ - Flexibilitätsnutzung analysieren                                 │
   │ - Visualisierung, Export                                           │
   └─────────────────────────────────────────────────────────────────────┘

--------------

Workflow-Varianten im Vergleich
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die folgende Tabelle zeigt die wichtigsten Workflow-Varianten und deren
Anwendungsfälle:

+--------------+--------------+------------------------+--------------+
| Workflow     | Schritte     | Wann sinnvoll?         | Ergebnis     |
+==============+==============+========================+==============+
| **A: Nur     | 1. Netz      | - Keine Flexibilitäten | Hohe         |
| Netzausbau   | laden2.      | vorhanden- Schnelle    | Ne           |
| (ohne        | Komponenten  | konservative Planung-  | tzausbaukost |
| Op           | hinzufügen3. | Referenzszenario       | enFlexibilit |
| timierung)** | ``r          |                        | ätspotenzial |
|              | einforce()`` |                        | ungenutzt    |
+--------------+--------------+------------------------+--------------+
| **B: Mit     | 1. Netz      | - Flexibilitäten       | Minimale     |
| Optimierung  | laden2.      | vorhanden (Speicher,   | Netzausbauko |
| (            | Optional:    | WP, CP)- §14a-Nutzung  | stenOptimale |
| EMPFOHLEN)** | ``r          | gewünscht- Minimierung | Flexibilität |
|              | einforce()`` | Netzausbaukosten       | snutzungBetr |
|              | auf          |                        | iebssicheres |
|              | Basisnetz3.  |                        | Netz         |
|              | Komponenten  |                        |              |
|              | hinzufügen4. |                        |              |
|              | ``pm_opti    |                        |              |
|              | mize()``\ 5. |                        |              |
|              | *            |                        |              |
|              | *Zwingend:** |                        |              |
|              | ``r          |                        |              |
|              | einforce()`` |                        |              |
+--------------+--------------+------------------------+--------------+
| **C:         | 1. Netz      | - Kostenvergleich      | Kostent      |
| Basisn       | laden        | mit/ohne neue          | ransparenzAt |
| etz-Referenz | (            | Komponenten- Analyse   | tributierung |
| +            | Basisnetz)2. | Zusatzkosten durch     | auf neue     |
| O            | ``r          | WP/CP- Bewertung       | Ko           |
| ptimierung** | einforce()`` | §14a-Nutzen            | mponentenQua |
|              | → Kosten₁3.  |                        | ntifizierung |
|              | Neue         |                        | Flexibi      |
|              | Komponenten  |                        | litätsnutzen |
|              | hinzufügen4. |                        |              |
|              | ``pm_opti    |                        |              |
|              | mize()``\ 5. |                        |              |
|              | ``r          |                        |              |
|              | einforce()`` |                        |              |
|              | → Kosten₂6.  |                        |              |
|              | Vergleich:   |                        |              |
|              | Kosten₂ -    |                        |              |
|              | Kosten₁      |                        |              |
+--------------+--------------+------------------------+--------------+
| **D: Mehrere | 1. Netz      | - Bewertung            | V            |
| Optimierung  | laden +      | verschiedener          | ollständiger |
| sszenarien** | Komponenten  | Flexibilitätsoptionen- | S            |
|              | h            | Kosten-Nutzen-Analyse  | zenariovergl |
|              | inzufügen2a. | §14a-                  | eichOptimale |
|              | ``r          | Sensitivitätsanalyse   | Strategiew   |
|              | einforce()`` |                        | ahlFundierte |
|              | →            |                        | Entscheidu   |
|              | Referenz2b.  |                        | ngsgrundlage |
|              | ``           |                        |              |
|              | pm_optimize( |                        |              |
|              | 14a=False)`` |                        |              |
|              | +            |                        |              |
|              | ``reinfo     |                        |              |
|              | rce()``\ 2c. |                        |              |
|              | `            |                        |              |
|              | `pm_optimize |                        |              |
|              | (14a=True)`` |                        |              |
|              | +            |                        |              |
|              | ``reinf      |                        |              |
|              | orce()``\ 3. |                        |              |
|              | Vergleich    |                        |              |
+--------------+--------------+------------------------+--------------+

**Wichtige Erkenntnisse:**

1. **Reinforce VOR Optimierung macht NUR Sinn für:**

   -  Basisnetz ohne neue Komponenten (Referenzszenario)
   -  Dokumentation des Ausgangszustands
   -  Status quo soll ermittelt werden
   -  **NICHT nach Hinzufügen der (neuen) Komponenten, deren
      Flexibilitätseinsatz untersucht werden soll** → Würde
      Flexibilitätspotenzial zunichtemachen

2. **Reinforce NACH Optimierung ist ZWINGEND:**

   -  Optimierung reduziert Netzausbau, löst aber nicht alle Probleme
   -  Slack-Variablen zeigen verbleibende Verletzungen
   -  Ohne finales ``reinforce()`` ist das Netz **NICHT betriebssicher**

3. **Beispielhafte Kostenreduktion:**

   -  Ohne Optimierung: 100% Netzausbaukosten (Referenz)
   -  Mit Optimierung ohne §14a: 60-80% der Referenzkosten
   -  Mit Optimierung mit §14a: 40-60% der Referenzkosten
   -  Abhängig von: Flexibilitätsgrad, Netzstruktur, Lastprofile

**Beispiel-Code für Workflow B (empfohlen):**

.. code:: python

   # Workflow B: Mit Optimierung (BESTE PRAXIS)

   # 1. Netz laden
   edisgo = EDisGo(ding0_grid="path/to/grid")

   # Zeitreihen laden etc.

   # 2. Optional: Basisnetz verstärken (für Vergleich)
   # edisgo.reinforce()  # Nur wenn Referenzkosten gewünscht oder auf status quo ausgebaut werden soll

   # 3. Neue Komponenten für Zukunftsszenario hinzufügen
   edisgo.add_heat_pumps(
       scenario="eGon2035",
       with_thermal_storage=True  # Flexibilität!
   )
   edisgo.add_charging_points(
       scenario="eGon2035"
   )

   # 4. Optimierung durchführen
   edisgo.pm_optimize(
       opf_version=2,              # Mit Netzrestriktionen
       curtailment_14a=True,       # §14a-Abregelung nutzen
       max_hours_per_day=2.0,      # 2h/Tag Zeitbudget
       solver="gurobi"
   )

   # 5. ZWINGEND: Netzausbau für verbleibende Probleme
   edisgo.reinforce()

   # 6. Ergebnisse analysieren
   costs = edisgo.results.grid_expansion_costs
   curtailment = edisgo.timeseries.generators_active_power[
       [c for c in edisgo.timeseries.generators_active_power.columns
        if '14a_support' in c]
   ]

   print(f"Netzausbaukosten (nach Optimierung): {costs:,.0f} €")
   print(f"§14a-Abregelung gesamt: {curtailment.sum().sum():.2f} MWh")

--------------

Detaillierter Zeitablauf der Julia-Optimierung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 1: Problemaufbau (build_mn_opf_bf_flex)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Für jeden Zeitschritt n im Optimierungshorizont:**

.. code:: julia

   for n in nw_ids(pm)
       # 1. VARIABLEN ERSTELLEN
       PowerModels.variable_bus_voltage(pm, nw=n)           # w[i]
       PowerModels.variable_gen_power(pm, nw=n)             # pg, qg
       PowerModels.variable_branch_power(pm, nw=n)          # p, q
       eDisGo_OPF.variable_branch_current(pm, nw=n)         # ccm

       # Flexibilitäten
       eDisGo_OPF.variable_storage_power(pm, nw=n)          # ps
       eDisGo_OPF.variable_heat_pump_power(pm, nw=n)        # php
       eDisGo_OPF.variable_heat_storage_power(pm, nw=n)     # phs
       eDisGo_OPF.variable_charging_point_power(pm, nw=n)   # pcp

       # Falls OPF Version 1 oder 3: Line Loading
       if opf_version in [1, 3]
           eDisGo_OPF.variable_line_loading(pm, nw=n)       # ll
       end

       # Falls OPF Version 2 oder 4: Slack-Variablen
       if opf_version in [2, 4]
           eDisGo_OPF.variable_slack_heatpumps(pm, nw=n)    # phps, phps2
           eDisGo_OPF.variable_slack_heat_storage(pm, nw=n) # phss
           eDisGo_OPF.variable_slack_loads(pm, nw=n)        # pds
           eDisGo_OPF.variable_slack_gens(pm, nw=n)         # pgens
           eDisGo_OPF.variable_slack_cps(pm, nw=n)          # pcps
       end

       # Falls §14a aktiviert: Virtuelle Generatoren + Binärvariablen
       if curtailment_14a
           eDisGo_OPF.variable_gen_hp_14a_power(pm, nw=n)   # p_hp14a
           eDisGo_OPF.variable_gen_hp_14a_binary(pm, nw=n)  # z_hp14a
           eDisGo_OPF.variable_gen_cp_14a_power(pm, nw=n)   # p_cp14a
           eDisGo_OPF.variable_gen_cp_14a_binary(pm, nw=n)  # z_cp14a
       end

       # 2. CONSTRAINTS PRO ZEITSCHRITT
       for i in ids(pm, :bus, nw=n)
           constraint_power_balance(pm, i, n)               # Eq 3.3, 3.4
       end

       for l in ids(pm, :branch, nw=n)
           constraint_voltage_drop(pm, l, n)                # Eq 3.5
           constraint_current_limit(pm, l, n)               # Eq 3.6
           if opf_version in [1, 3]
               constraint_line_loading(pm, l, n)            # ll definition
           end
       end

       for s in ids(pm, :storage, nw=n)
           constraint_storage_state(pm, s, n)               # Eq 3.9
           constraint_storage_complementarity(pm, s, n)     # Eq 3.10
       end

       for h in ids(pm, :heat_pump, nw=n)
           constraint_heat_pump_operation(pm, h, n)         # Eq 3.19
           constraint_heat_storage_state(pm, h, n)          # Eq 3.22
           constraint_heat_storage_complementarity(pm, h, n)# Eq 3.23
       end

       for c in ids(pm, :charging_point, nw=n)
           constraint_cp_state(pm, c, n)                    # Eq 3.25
           constraint_cp_complementarity(pm, c, n)          # Eq 3.26
       end

       for d in ids(pm, :dsm, nw=n)
           constraint_dsm_state(pm, d, n)                   # Eq 3.32
           constraint_dsm_complementarity(pm, d, n)         # Eq 3.33
       end

       # §14a Constraints pro Zeitschritt
       if curtailment_14a
           for h in ids(pm, :gen_hp_14a, nw=n)
               constraint_hp_14a_binary_coupling(pm, h, n)  # p_hp14a ≤ pmax × z
               constraint_hp_14a_min_net_load(pm, h, n)     # Nettolast ≥ min(Last, 4.2kW)
           end
           for c in ids(pm, :gen_cp_14a, nw=n)
               constraint_cp_14a_binary_coupling(pm, c, n)
               constraint_cp_14a_min_net_load(pm, c, n)
           end
       end
   end

Phase 2: Inter-Zeitschritt Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   # Speicher-Energiekopplung zwischen Zeitschritten
   for s in ids(pm, :storage)
       for t in 1:(T-1)
           se[t+1] == se[t] + ps[t] × Δt × η
       end
   end

   # Wärmespeicher-Kopplung
   for h in ids(pm, :heat_pump)
       for t in 1:(T-1)
           hse[t+1] == hse[t] + phs[t] × Δt × η
       end
   end

   # EV-Batterie-Kopplung
   for c in ids(pm, :charging_point)
       for t in 1:(T-1)
           cpe[t+1] == cpe[t] + pcp[t] × Δt × η
       end
   end

   # §14a Tages-Zeitbudget
   if curtailment_14a
       # Gruppiere Zeitschritte in 24h-Tage
       day_groups = group_timesteps_by_day(timesteps)

       for day in day_groups
           for h in ids(pm, :gen_hp_14a)
               sum(z_hp14a[h,t] for t in day) ≤ max_hours_per_day / Δt
           end
           for c in ids(pm, :gen_cp_14a)
               sum(z_cp14a[c,t] for t in day) ≤ max_hours_per_day / Δt
           end
       end
   end

Phase 3: Zielfunktion
^^^^^^^^^^^^^^^^^^^^^

**OPF Version 1** (geliftete Restriktionen, ohne Slacks):

.. code:: julia

   minimize: 0.9 × sum(Verluste) + 0.1 × max(ll) + 0.05 × sum(p_hp14a) + 0.05 × sum(p_cp14a)

**OPF Version 2** (mit Netzrestriktionen, mit Slacks):

.. code:: julia

   minimize: 0.4 × sum(Verluste) + 0.6 × sum(Slacks) + 0.5 × sum(p_hp14a) + 0.5 × sum(p_cp14a)

**OPF Version 3** (mit HV-Anforderungen, geliftete Restriktionen):

.. code:: julia

   minimize: 0.9 × sum(Verluste) + 0.1 × max(ll) + 50 × sum(phvs) + 0.05 × sum(p_hp14a) + 0.05 × sum(p_cp14a)

**OPF Version 4** (mit HV-Anforderungen und Restriktionen):

.. code:: julia

   minimize: 0.4 × sum(Verluste) + 0.6 × sum(Slacks) + 50 × sum(phvs) + 0.5 × sum(p_hp14a) + 0.5 × sum(p_cp14a)

**Wichtig:** - §14a-Terme haben moderate Gewichte → Abregelung wird
genutzt, aber minimiert - Slack-Variablen haben hohe implizite Kosten →
nur wenn unvermeidbar - HV-Slack hat sehr hohes Gewicht → Einhaltung
prioritär

Phase 4: Lösen
^^^^^^^^^^^^^^

.. code:: julia

   # Solver-Auswahl
   if method == "soc"
       solver = Gurobi.Optimizer
       # SOC-Relaxation: ccm-Constraints als Second-Order-Cone
   elseif method == "nc"
       solver = Ipopt.Optimizer
       # Non-Convex: ccm-Constraints als quadratische Gleichungen
   end

   # Optimierung durchführen
   result = optimize_model!(pm, solver)

   # Optional: Warm-Start NC mit SOC-Lösung
   if warm_start
       result_soc = optimize_model!(pm, Gurobi.Optimizer)
       initialize_from_soc!(pm, result_soc)
       result = optimize_model!(pm, Ipopt.Optimizer)
   end

--------------

Die analyze-Funktion
--------------------

Funktionsdefinition
~~~~~~~~~~~~~~~~~~~

**Datei:** ``edisgo/edisgo.py`` (Zeile ~1038)

**Signatur:**

.. code:: python

   def analyze(
       self,
       mode: str | None = None,
       timesteps: pd.DatetimeIndex | None = None,
       troubleshooting_mode: str | None = None,
       scale_timeseries: float | None = None,
       **kwargs
   ) -> None

Was macht analyze?
~~~~~~~~~~~~~~~~~~

Die ``analyze``-Funktion führt eine **statische, nicht-lineare
Leistungsflussberechnung** (Power Flow Analysis, PFA) mit PyPSA durch.
Sie berechnet:

1. **Spannungen** an allen Knoten (``v_res``)
2. **Ströme** auf allen Leitungen und Trafos (``i_res``)
3. **Wirkleistungsflüsse** auf Betriebsmitteln (``pfa_p``)
4. **Blindleistungsflüsse** auf Betriebsmitteln (``pfa_q``)

Die Ergebnisse werden in ``edisgo.results`` gespeichert.

Parameter
~~~~~~~~~

+--------------------+------------------+-----------------------------+
| Parameter          | Default          | Beschreibung                |
+====================+==================+=============================+
| ``mode``           | str \| None      | Analyseebene: ``'mv'``      |
|                    |                  | (MS-Netz), ``'mvlv'`` (MS   |
|                    |                  | mit NS an Sekundärseite),   |
|                    |                  | ``'lv'`` (einzelnes         |
|                    |                  | NS-Netz), ``None``          |
|                    |                  | (gesamtes Netz)             |
+--------------------+------------------+-----------------------------+
| ``timesteps``      | DatetimeIndex \| | Zeitschritte für Analyse.   |
|                    | None             | ``None`` = alle in          |
|                    |                  | ``timeseries.timeindex``    |
+--------------------+------------------+-----------------------------+
| ``trou             | str \| None      | ``'lpf'`` = Linear PF       |
| bleshooting_mode`` |                  | seeding, ``'iteration'`` =  |
|                    |                  | schrittweise Lasterhöhung   |
+--------------------+------------------+-----------------------------+
| ``                 | float \| None    | Skalierungsfaktor für       |
| scale_timeseries`` |                  | Zeitreihen (z.B. 0.5 für    |
|                    |                  | 50% Last)                   |
+--------------------+------------------+-----------------------------+

Zeitreihen-Nutzung
~~~~~~~~~~~~~~~~~~

``analyze`` verwendet **alle** Zeitreihen aus ``edisgo.timeseries``:

Generatoren
^^^^^^^^^^^

-  **Quelle:** ``edisgo.timeseries.generators_active_power``
-  **Quelle:** ``edisgo.timeseries.generators_reactive_power``
-  **Inhalt:** Einspeisung aller Generatoren (PV, Wind, BHKW, etc.) in
   MW/MVAr
-  **Zeitauflösung:** Typisch 1h oder 15min
-  **Herkunft:** Datenbank (eGon), WorstCase-Profil, oder optimierte
   Zeitreihen

Lasten
^^^^^^

-  **Quelle:** ``edisgo.timeseries.loads_active_power``
-  **Quelle:** ``edisgo.timeseries.loads_reactive_power``
-  **Inhalt:** Haushaltslast, Gewerbe, Industrie in MW/MVAr
-  **Zeitauflösung:** Typisch 1h oder 15min
-  **Herkunft:** Datenbank, Standardlastprofile, oder gemessene Daten

Speicher
^^^^^^^^

-  **Quelle:** ``edisgo.timeseries.storage_units_active_power``
-  **Quelle:** ``edisgo.timeseries.storage_units_reactive_power``
-  **Inhalt:** Batteriespeicher Ladung/Entladung in MW/MVAr
-  **Zeitauflösung:** Wie Zeitreihenindex
-  **Herkunft:** Optimierung oder vorgegebene Fahrpläne

Wärmepumpen
^^^^^^^^^^^

-  **Quelle:** Indirekt aus ``heat_demand_df`` und ``cop_df``
-  **Berechnung:** ``P_el = heat_demand / COP``
-  **Zeitauflösung:** Wie Zeitreihenindex
-  **Herkunft:** Wärmebedarfsprofile (z.B. BDEW), COP-Profile
   (temperaturabhängig)
-  **Nach Optimierung:** Aus optimierten Zeitreihen
   ``timeseries.heat_pumps_active_power``

Ladepunkte (E-Autos)
^^^^^^^^^^^^^^^^^^^^

-  **Quelle:** ``edisgo.timeseries.charging_points_active_power``
-  **Zeitauflösung:** Wie Zeitreihenindex
-  **Herkunft:** Ladeprofile (z.B. SimBEV), Flexibilitätsbänder, oder
   Optimierung

Prozessablauf
~~~~~~~~~~~~~

.. code:: python

   # 1. Zeitschritte bestimmen
   if timesteps is None:
       timesteps = self.timeseries.timeindex
   else:
       timesteps = pd.DatetimeIndex(timesteps)

   # 2. In PyPSA-Netzwerk konvertieren
   pypsa_network = self.to_pypsa(
       mode=mode,
       timesteps=timesteps
   )

   # 3. Optional: Zeitreihen skalieren
   if scale_timeseries is not None:
       pypsa_network.loads_t.p_set *= scale_timeseries
       pypsa_network.generators_t.p_set *= scale_timeseries
       # ... weitere Zeitreihen skalieren

   # 4. Leistungsflussberechnung durchführen
   pypsa_network.pf(
       timesteps,
       use_seed=(troubleshooting_mode == 'lpf')
   )

   # 5. Konvergenz prüfen
   converged_ts = timesteps[pypsa_network.converged]
   not_converged_ts = timesteps[~pypsa_network.converged]

   if len(not_converged_ts) > 0:
       logger.warning(f"Power flow did not converge for {len(not_converged_ts)} timesteps")

   # 6. Ergebnisse verarbeiten
   pypsa_io.process_pfa_results(
       edisgo_obj=self,
       pypsa_network=pypsa_network,
       timesteps=timesteps
   )

   # 7. Ergebnisse in edisgo.results speichern
   # self.results.v_res       -> Spannungen an Knoten
   # self.results.i_res       -> Ströme auf Leitungen
   # self.results.pfa_p       -> Wirkleistungsflüsse
   # self.results.pfa_q       -> Blindleistungsflüsse

Wann wird analyze aufgerufen?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Manuell vom Benutzer:**

   .. code:: python

      edisgo.analyze()  # Analysiere gesamtes Netz, alle Zeitschritte

2. **Von reinforce-Funktion:**

   -  Initial: Identifiziere Netzprobleme
   -  Nach jeder Verstärkung: Prüfe ob Probleme gelöst
   -  Iterativ bis keine Verletzungen mehr

3. **Nach Optimierung (optional):**

   .. code:: python

      edisgo.pm_optimize(...)
      edisgo.analyze()  # Analysiere mit optimierten Zeitreihen

4. **Bei Worst-Case-Analyse:**

   .. code:: python

      # Nur zwei kritische Zeitpunkte
      worst_case_ts = edisgo.get_worst_case_timesteps()
      edisgo.analyze(timesteps=worst_case_ts)

Troubleshooting-Modi
~~~~~~~~~~~~~~~~~~~~

Linear Power Flow Seeding (``'lpf'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Problem: Nicht-linearer PF konvergiert nicht
-  Lösung: Starte mit linearer PF-Lösung (Winkel) als Startwert
-  Nutzen: Stabilisiert Konvergenz bei schwierigen Netzen

Iteration (``'iteration'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Problem: Konvergenz bei hoher Last nicht möglich
-  Lösung: Beginne mit 10% Last, erhöhe schrittweise bis 100%
-  Nutzen: Findet Lösung bei extremen Betriebspunkten

Ausgabe
~~~~~~~

**Erfolgreiche Analyse:**

::

   Info: Power flow analysis completed for 8760 timesteps
   Info: 8760 timesteps converged, 0 did not converge

**Konvergenzprobleme:**

::

   Warning: Power flow did not converge for 15 timesteps
   Warning: Non-converged timesteps: ['2035-01-15 18:00', '2035-07-21 12:00', ...]

--------------

Die reinforce-Funktion
----------------------

.. _funktionsdefinition-1:

Funktionsdefinition
~~~~~~~~~~~~~~~~~~~

**Datei:** ``edisgo/edisgo.py`` (Zeile ~1243) **Implementierung:**
``edisgo/flex_opt/reinforce_grid.py`` (Zeile ~25)

**Signatur:**

.. code:: python

   def reinforce(
       self,
       timesteps_pfa: str | pd.DatetimeIndex | None = None,
       reduced_analysis: bool = False,
       max_while_iterations: int = 20,
       split_voltage_band: bool = True,
       mode: str | None = None,
       without_generator_import: bool = False,
       n_minus_one: bool = False,
       **kwargs
   ) -> None

Was macht reinforce?
~~~~~~~~~~~~~~~~~~~~

Die ``reinforce``-Funktion **identifiziert Netzprobleme** (Überlastung,
Spannungsverletzungen) und **führt Verstärkungsmaßnahmen** durch:

1. **Leitungsverstärkung:** Parallele Leitungen oder Ersatz durch
   größeren Querschnitt
2. **Transformatorverstärkung:** Paralleltrafos oder größere Leistung
3. **Spannungsebenen-Trennung:** LV-Netze bei Bedarf aufteilen
4. **Kostenberechnung:** Netzausbaukosten (€)

.. _parameter-1:

Parameter
~~~~~~~~~

+----------------+--------+----------------+--------------------------+
| Parameter      | Typ    | Default        | Beschreibung             |
+================+========+================+==========================+
| ``t            | ``     | ``'snapsh      | ``'snapshot_analysis'``  |
| imesteps_pfa`` | str \| | ot_analysis'`` | = 2                      |
|                |  Datet |                | Worst-Case-Zeitschritte, |
|                | imeInd |                | ``DatetimeIndex`` =      |
|                | ex \|  |                | benutzerdefiniert,       |
|                | None`` |                | ``None`` = alle          |
|                |        |                | Zeitschritte             |
+----------------+--------+----------------+--------------------------+
| ``redu         | ``     | ``False``      | Nutzt nur die            |
| ced_analysis`` | bool`` |                | kritischsten             |
|                |        |                | Zeitschritte (höchste    |
|                |        |                | Überlast oder            |
|                |        |                | Spannungsabweichung)     |
+----------------+--------+----------------+--------------------------+
| ``max_whil     | `      | ``20``         | Maximale Anzahl der      |
| e_iterations`` | `int`` |                | Verstärkungsiterationen  |
+----------------+--------+----------------+--------------------------+
| ``split_       | ``     | ``True``       | Getrennte                |
| voltage_band`` | bool`` |                | Spannungsbänder für      |
|                |        |                | NS/MS (z.B. NS ±3 %, MS  |
|                |        |                | ±7 %)                    |
+----------------+--------+----------------+--------------------------+
| ``mode``       | ``s    | ``None``       | Netzebene: ``'mv'``,     |
|                | tr \|  |                | ``'mvlv'``, ``'lv'``     |
|                | None`` |                | oder ``None`` (=         |
|                |        |                | automatisch)             |
+----------------+--------+----------------+--------------------------+
| ``without_gene | ``     | ``False``      | Ignoriert                |
| rator_import`` | bool`` |                | Generatoreinspeisung     |
|                |        |                | (nur für                 |
|                |        |                | Planungsanalysen         |
|                |        |                | sinnvoll)                |
+----------------+--------+----------------+--------------------------+
| `              | ``     | ``False``      | Berücksichtigt das       |
| `n_minus_one`` | bool`` |                | (n-1)-Kriterium          |
+----------------+--------+----------------+--------------------------+

.. _zeitreihen-nutzung-1:

Zeitreihen-Nutzung
~~~~~~~~~~~~~~~~~~

Nutzt **dieselben Zeitreihen wie analyze**:

-  ``generators_active_power``, ``generators_reactive_power``
-  ``loads_active_power``, ``loads_reactive_power``
-  ``storage_units_active_power``, ``storage_units_reactive_power``
-  Wärmepumpen-Lasten (aus heat_demand/COP)
-  Ladepunkt-Lasten

**Zeitreihen-Auswahl:**

Option 1: Snapshot Analysis (``timesteps_pfa='snapshot_analysis'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Nur 2 kritische Zeitpunkte
   ts1 = timestep_max_residual_load    # Max. Residuallast (hohe Last, wenig Einspeisung)
   ts2 = timestep_min_residual_load    # Min. Residuallast (niedrige Last, viel Einspeisung)
   timesteps = [ts1, ts2]

**Vorteil:** Sehr schnell (nur 2 PFA statt 8760) **Nachteil:** Kann
seltene Probleme übersehen

Option 2: Reduzierte Analyse (``reduced_analysis=True``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # 1. Initiale PFA mit allen Zeitschritten
   edisgo.analyze(timesteps=all_timesteps)

   # 2. Identifiziere kritischste Zeitschritte
   critical_timesteps = get_most_critical_timesteps(
       overloading_factor=1.0,  # Nur Zeitschritte mit Überlast
       voltage_deviation=0.03   # Nur Zeitschritte mit >3% Spannungsabweichung
   )

   # 3. Verstärkung nur auf Basis dieser Zeitschritte
   timesteps = critical_timesteps  # z.B. 50 statt 8760

**Vorteil:** Deutlich schneller als volle Analyse, genauer als Snapshot
**Nachteil:** Initial-PFA mit allen Zeitschritten nötig

Option 3: Alle Zeitschritte (``timesteps_pfa=None``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   timesteps = edisgo.timeseries.timeindex  # z.B. alle 8760h eines Jahres

**Vorteil:** Maximale Genauigkeit **Nachteil:** Sehr rechenintensiv
(viele PFA)

Option 4: Custom (``timesteps_pfa=custom_datetimeindex``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Z.B. nur Wintermonate
   timesteps = pd.date_range('2035-01-01', '2035-03-31', freq='H')

reinforce Algorithmus
~~~~~~~~~~~~~~~~~~~~~

Schritt 1: Überlastungen beseitigen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   iteration = 0
   while has_overloading() and iteration < max_while_iterations:
       iteration += 1

       # 1.1 HV/MV-Station prüfen
       if hv_mv_station_max_overload() > 0:
           reinforce_hv_mv_station()

       # 1.2 MV/LV-Stationen prüfen
       for station in mv_lv_stations:
           if station.max_overload > 0:
               reinforce_mv_lv_station(station)

       # 1.3 MV-Leitungen prüfen
       for line in mv_lines:
           if line.max_relative_overload > 0:
               reinforce_line(line)

       # 1.4 LV-Leitungen prüfen
       for line in lv_lines:
           if line.max_relative_overload > 0:
               reinforce_line(line)

       # 1.5 Erneute Analyse
       edisgo.analyze(timesteps=timesteps)

       # 1.6 Konvergenz prüfen
       if not has_overloading():
           break

**Verstärkungsmaßnahmen:** - **Parallelleitungen:** Identischer Typ
parallel schalten - **Leitungsersatz:** Größerer Querschnitt (z.B.
150mm² → 240mm²) - **Paralleltrafos:** Identischer Trafo parallel -
**Trafoersatz:** Größere Leistung (z.B. 630kVA → 1000kVA)

Schritt 2: MV-Spannungsprobleme lösen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   iteration = 0
   while has_voltage_issues_mv() and iteration < max_while_iterations:
       iteration += 1

       # Identifiziere kritische Leitungen
       critical_lines = get_lines_voltage_issues(voltage_level='mv')

       for line in critical_lines:
           reinforce_line(line)

       # Erneute Analyse
       edisgo.analyze(timesteps=timesteps)

Schritt 3: MV/LV-Stations-Spannungsprobleme
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   for station in mv_lv_stations:
       if has_voltage_issues_at_secondary_side(station):
           # Trafo-Kapazität erhöhen
           reinforce_mv_lv_station(station)

   edisgo.analyze(timesteps=timesteps)

Schritt 4: LV-Spannungsprobleme lösen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   for lv_grid in lv_grids:
       while has_voltage_issues(lv_grid) and iteration < max_while_iterations:
           iteration += 1

           # Kritische Leitungen verstärken
           critical_lines = get_lines_voltage_issues(
               grid=lv_grid,
               voltage_level='lv'
           )

           for line in critical_lines:
               reinforce_line(line)

           edisgo.analyze(timesteps=timesteps, mode='lv')

Schritt 5: Finale Überprüfung
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Prüfe ob Spannungsverstärkungen neue Überlastungen verursacht haben
   edisgo.analyze(timesteps=timesteps)

   if has_overloading():
       # Zurück zu Schritt 1
       goto_step_1()

Schritt 6: Kostenberechnung
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Berechne Netzausbaukosten
   costs = calculate_grid_expansion_costs(edisgo)

   # Kosten pro Komponente
   line_costs = costs['lines']          # €
   trafo_costs = costs['transformers']  # €
   total_costs = costs['total']         # €

   # Speichere in edisgo.results
   edisgo.results.grid_expansion_costs = costs

Verstärkungslogik
~~~~~~~~~~~~~~~~~

Leitungsverstärkung
^^^^^^^^^^^^^^^^^^^

.. code:: python

   def reinforce_line(line):
       # 1. Berechne benötigte Kapazität
       required_capacity = line.s_nom * (1 + max_relative_overload)

       # 2. Option A: Parallelleitungen
       num_parallel = ceil(required_capacity / line.s_nom)
       cost_parallel = num_parallel * line_cost(line.type)

       # 3. Option B: Größerer Querschnitt
       new_type = get_next_larger_type(line.type)
       if new_type is not None:
           cost_replacement = line_cost(new_type)
       else:
           cost_replacement = inf

       # 4. Wähle günstigere Option
       if cost_parallel < cost_replacement:
           add_parallel_lines(line, num_parallel - 1)
       else:
           replace_line(line, new_type)

Transformatorverstärkung
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   def reinforce_transformer(trafo):
       # 1. Berechne benötigte Leistung
       required_power = trafo.s_nom * (1 + max_relative_overload)

       # 2. Option A: Paralleltrafos
       num_parallel = ceil(required_power / trafo.s_nom)
       cost_parallel = num_parallel * trafo_cost(trafo.type)

       # 3. Option B: Größerer Trafo
       new_type = get_next_larger_trafo(trafo.s_nom)
       cost_replacement = trafo_cost(new_type)

       # 4. Wähle günstigere Option
       if cost_parallel < cost_replacement:
           add_parallel_trafos(trafo, num_parallel - 1)
       else:
           replace_trafo(trafo, new_type)

.. _ausgabe-1:

Ausgabe
~~~~~~~

**Erfolgreiche Verstärkung:**

::

   Info: ==> Checking stations.
   Info:   MV station is not overloaded.
   Info:   All MV/LV stations are within allowed load range.
   Info: ==> Checking lines.
   Info:   Reinforcing 15 overloaded MV lines.
   Info:   Reinforcing 42 overloaded LV lines.
   Info: ==> Voltage issues in MV grid.
   Info:   Reinforcing 8 lines due to voltage issues.
   Info: ==> Voltage issues in LV grids.
   Info:   Reinforcing 23 lines in LV grids.
   Info: Grid reinforcement finished. Total costs: 145,320 €

**Iterations-Limit erreicht:**

::

   Warning: Maximum number of iterations (20) reached. Grid issues may remain.

Wann wird reinforce aufgerufen?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Manuell vom Benutzer:**

   .. code:: python

      edisgo.reinforce()

2. **Basisnetz-Verstärkung VOR neuen Komponenten (optional):**

   .. code:: python

      # Laden eines Basisnetzes (z.B. Ist-Zustand 2024)
      edisgo = EDisGo(ding0_grid="path/to/grid")
      edisgo.import_generators(scenario="status_quo")
      edisgo.import_loads(scenario="status_quo")

      # Optional: Verstärkung des Basisnetzes (als Referenz)
      edisgo.reinforce()  # Kosten für Basisnetz dokumentieren

      # DANN: Neue Komponenten für Zukunftsszenario hinzufügen
      edisgo.add_heat_pumps(scenario="2035_high")
      edisgo.add_charging_points(scenario="2035_high")

      # WICHTIG: KEIN reinforce an dieser Stelle!
      # Stattdessen: Optimierung nutzen → siehe Punkt 5

   **Sinnvoll für:** Referenzszenario, Vergleich Netzausbaukosten
   mit/ohne Flexibilität

3. **Nach Szenario-Simulation OHNE Optimierung:**

   .. code:: python

      # Wenn man KEINE Optimierung nutzen möchte (rein konventioneller Netzausbau)
      edisgo.add_charging_points(scenario='high_ev')
      edisgo.reinforce()  # Verstärke Netz für neue Last (ohne Flexibilität)

   **Nachteil:** Hohe Netzausbaukosten, Flexibilitätspotenzial wird
   nicht genutzt

4. **Nach Optimierung (ZWINGEND ERFORDERLICH!):**

   .. code:: python

      # Korrekte Reihenfolge:
      # 1. Neue Komponenten hinzufügen (WP, CP, Speicher)
      edisgo.add_heat_pumps(...)
      edisgo.add_charging_points(...)

      # 2. Optimierung durchführen (nutzt Flexibilität)
      edisgo.pm_optimize(opf_version=2, curtailment_14a=True)

      # 3. ZWINGEND: Netzausbau für verbleibende Probleme
      edisgo.reinforce()  # Behebt Überlastungen die Optimierung nicht lösen konnte

      # Ergebnis: Minimierte Netzausbaukosten durch Flexibilitätsnutzung

   **Warum zwingend?**

   -  Optimierung minimiert Netzausbau, kann aber nicht alle Probleme
      lösen
   -  Slack-Variablen > 0 zeigen verbleibende
      Netzrestriktionsverletzungen
   -  Verbleibende Überlastungen müssen durch konventionellen Ausbau
      behoben werden
   -  **Ohne diesen Schritt:** Netz ist NICHT betriebssicher!

5. **Iterativer Workflow für mehrere Szenarien:**

   .. code:: python

      # Vergleich verschiedener Flexibilitätsszenarien

      # Szenario 1: Ohne Optimierung (Referenz)
      edisgo_ref = edisgo.copy()
      edisgo_ref.reinforce()
      costs_ref = edisgo_ref.results.grid_expansion_costs

      # Szenario 2: Mit Optimierung aber ohne §14a
      edisgo_opt = edisgo.copy()
      edisgo_opt.pm_optimize(opf_version=2, curtailment_14a=False)
      edisgo_opt.reinforce()  
      costs_opt = edisgo_opt.results.grid_expansion_costs

      # Szenario 3: Mit Optimierung und §14a
      edisgo_14a = edisgo.copy()
      edisgo_14a.pm_optimize(opf_version=2, curtailment_14a=True)
      edisgo_14a.reinforce()  
      costs_14a = edisgo_14a.results.grid_expansion_costs

      # Vergleich
      print(f"Ohne Optimierung: {costs_ref:,.0f} €")
      print(f"Mit Optimierung: {costs_opt:,.0f} € (-{100*(1-costs_opt/costs_ref):.1f}%)")
      print(f"Mit §14a: {costs_14a:,.0f} € (-{100*(1-costs_14a/costs_ref):.1f}%)")

--------------

Die §14a EnWG Optimierung
-------------------------

Was ist §14a EnWG?
~~~~~~~~~~~~~~~~~~

**Gesetzesgrundlage:** § 14a Energiewirtschaftsgesetz (EnWG)

**Inhalt:** Netzbetreiber dürfen **steuerbare Verbrauchseinrichtungen**
(Wärmepumpen, Ladeeinrichtungen für E-Autos) bei Netzengpässen
**abregelnd** bis auf eine **Mindestleistung** (4,2 kW).

**Bedingungen:** - Maximales **Zeitbudget**: Typisch 2 Stunden pro Tag -
**Mindestleistung**: 4,2 kW (0,0042 MW) muss gewährleistet bleiben -
**Vergütung**: Reduziertes Netzentgelt für Kunden

**Ziel:** Netzausbau reduzieren durch gezielte Spitzenlast-Kappung

Wie unterscheidet sich §14a von Standard-Optimierung?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard-Optimierung (OHNE §14a):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Wärmepumpen mit thermischem Speicher:** Zeitliche Lastverschiebung
-  **E-Autos:** Ladesteuerung innerhalb Flexibilitätsband
-  **Inflexible WP/CP:** Können NICHT abgeregelt werden

§14a-Optimierung (MIT §14a):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **ALLE Wärmepumpen > 4,2 kW:** Können bis auf 4,2 kW abgeregelt
   werden
-  **ALLE Ladepunkte > 4,2 kW:** Können bis auf 4,2 kW abgeregelt werden
-  **Auch ohne Speicher:** Abregelung möglich
-  **Zeitbudget-Constraints:** Max. 2h/Tag Abregelung
-  **Binäre Entscheidung:** Abregelung aktiv JA/NEIN

Mathematische Modellierung
~~~~~~~~~~~~~~~~~~~~~~~~~~

Virtuelle Generatoren
^^^^^^^^^^^^^^^^^^^^^

§14a-Abregelung wird durch **virtuelle Generatoren** modelliert:

::

   Nettolast_WP = Original_Last_WP - p_hp14a

   Beispiel:
   - WP-Last: 8 kW
   - Abregelung: 3,8 kW (virtueller Generator erzeugt 3,8 kW)
   - Nettolast: 8 - 3,8 = 4,2 kW (Mindestlast)

**Vorteile dieser Modellierung:** - Keine Änderung der Last-Zeitreihen
nötig - Kompatibel mit PowerModels-Struktur - Einfache Implementierung
in Optimierungsproblem

Variablen (pro Wärmepumpe h, Zeitschritt t)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   # Kontinuierliche Variable: Abregelleistung
   @variable(model, 0 <= p_hp14a[h,t] <= pmax[h])

   # Binäre Variable: Abregelung aktiv?
   @variable(model, z_hp14a[h,t], Bin)

**Parameter:** - ``pmax[h] = P_nominal[h] - P_min_14a`` -
``P_nominal[h]``: Nennleistung Wärmepumpe h (z.B. 8 kW) -
``P_min_14a = 4,2 kW``: Gesetzliche Mindestlast -
``pmax[h] = 8 - 4,2 = 3,8 kW``: Maximale Abregelleistung

-  ``max_hours_per_day = 2``: Zeitbudget in Stunden pro Tag

Constraints
^^^^^^^^^^^

1. Binäre Kopplung (Binary Coupling)
''''''''''''''''''''''''''''''''''''

.. code:: julia

   @constraint(model, p_hp14a[h,t] <= pmax[h] × z_hp14a[h,t])

**Bedeutung:** - Wenn ``z_hp14a[h,t] = 0`` (keine Abregelung):
``p_hp14a[h,t] = 0`` - Wenn ``z_hp14a[h,t] = 1`` (Abregelung aktiv):
``0 ≤ p_hp14a[h,t] ≤ pmax[h]``

**Zweck:** Verhindert “Teilabregelung” ohne binäre Aktivierung

2. Mindest-Nettolast (Minimum Net Load)
'''''''''''''''''''''''''''''''''''''''

.. code:: julia

   @constraint(model,
       p_hp_load[h,t] - p_hp14a[h,t] >= min(p_hp_load[h,t], p_min_14a)
   )

**Bedeutung:** - Nettolast muss mindestens so groß wie aktuelle Last
ODER 4,2 kW sein - Verhindert “negative Last” (Generator größer als
Last)

**Spezialfälle:**

**Fall A: WP ist aus** (``p_hp_load[h,t] < 1e-6 MW``):

.. code:: julia

   @constraint(model, p_hp14a[h,t] == 0)

Abregelung macht keinen Sinn, wenn WP eh aus ist.

**Fall B: WP zu klein** (``pmax[h] < 1e-6 MW``):

.. code:: julia

   @constraint(model, p_hp14a[h,t] == 0)

WP-Nennleistung < 4,2 kW → keine Abregelung möglich.

**Fall C: Normalbetrieb** (``p_hp_load[h,t] >= p_min_14a``):

.. code:: julia

   @constraint(model, p_hp_load[h,t] - p_hp14a[h,t] >= p_min_14a)

Nettolast muss mindestens 4,2 kW bleiben.

3. Tages-Zeitbudget (Daily Time Budget)
'''''''''''''''''''''''''''''''''''''''

.. code:: julia

   for day in days
       @constraint(model,
           sum(z_hp14a[h,t] for t in timesteps_in_day(day))
           <= max_hours_per_day / time_elapsed_per_timestep
       )
   end

**Beispiel:** - Zeitauflösung: 1h - Zeitbudget: 2h/Tag - Constraint:
``sum(z_hp14a[h,t] for t in 0..23) <= 2``

**Beispiel mit 15min-Auflösung:** - Zeitauflösung: 0,25h - Zeitbudget:
2h/Tag - Constraint:
``sum(z_hp14a[h,t] for t in 0..95) <= 2 / 0.25 = 8``

**Alternative:** Total Budget Constraint (über gesamten Horizont):

.. code:: julia

   @constraint(model,
       sum(z_hp14a[h,t] for t in all_timesteps)
       <= max_hours_total / time_elapsed_per_timestep
   )

Zielfunktions-Integration
^^^^^^^^^^^^^^^^^^^^^^^^^

§14a-Variablen werden mit **moderaten Gewichten** ins Zielfunktional
aufgenommen:

**OPF Version 2 (mit §14a):**

.. code:: julia

   minimize:
       0.4 × sum(line_losses[t] for t in timesteps)
     + 0.6 × sum(all_slacks[t] for t in timesteps)
     + 0.5 × sum(p_hp14a[h,t] for h,t)
     + 0.5 × sum(p_cp14a[c,t] for c,t)

**Interpretation der Gewichte:** - ``0.4`` für Verluste: Basiskosten
Netzbetrieb - ``0.6`` für Slacks: Hohe Priorität Netzrestriktionen
einhalten - ``0.5`` für §14a: Moderate Kosten → Abregelung wird genutzt,
aber minimiert - Präferenz: Erst andere Flexibilitäten (Speicher,
zeitliche Verschiebung) - Wenn andere Flexibilitäten nicht ausreichen:
§14a als “letzte Reserve”

**OPF Version 1 (mit §14a):**

.. code:: julia

   minimize:
       0.9 × sum(line_losses[t] for t in timesteps)
     + 0.1 × max(line_loading[l,t] for l,t)
     + 0.05 × sum(p_hp14a[h,t] for h,t)
     + 0.05 × sum(p_cp14a[c,t] for c,t)

**Niedrigeres Gewicht (0.05):** §14a wird bevorzugt gegenüber hoher
Leitungsauslastung.

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

Datei-Struktur
^^^^^^^^^^^^^^

**Python-Seite:** - **Datei:** ``edisgo/io/powermodels_io.py`` -
**Funktionen:** - ``_build_gen_hp_14a_support()``: Erstellt virtuelle
Generatoren für WP - ``_build_gen_cp_14a_support()``: Erstellt virtuelle
Generatoren für CP - ``to_powermodels(..., curtailment_14a=True)``: Ruft
obige Funktionen auf

**Julia-Seite:** - **Variablen:**
``edisgo/opf/eDisGo_OPF.jl/src/core/variables.jl`` -
``variable_gen_hp_14a_power()`` - ``variable_gen_hp_14a_binary()`` -
``variable_gen_cp_14a_power()`` - ``variable_gen_cp_14a_binary()``

-  **Constraints:**
   ``edisgo/opf/eDisGo_OPF.jl/src/core/constraint_hp_14a.jl``

   -  ``constraint_hp_14a_binary_coupling()``
   -  ``constraint_hp_14a_min_net_load()``

-  **Constraints:**
   ``edisgo/opf/eDisGo_OPF.jl/src/core/constraint_cp_14a.jl``

   -  ``constraint_cp_14a_binary_coupling()``
   -  ``constraint_cp_14a_min_net_load()``

-  **Problemdefinition:**
   ``edisgo/opf/eDisGo_OPF.jl/src/prob/opf_bf.jl``

   -  Integration in ``build_mn_opf_bf_flex()``

Python: Virtuelle Generatoren erstellen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   def _build_gen_hp_14a_support(
       edisgo_obj,
       pm_dict,
       timesteps,
       max_hours_per_day=2.0
   ):
       """
       Erstellt virtuelle Generatoren für §14a-Abregelung von Wärmepumpen.
       """
       # 1. Alle Wärmepumpen identifizieren
       heat_pumps = edisgo_obj.topology.loads_df[
           edisgo_obj.topology.loads_df.type == "heat_pump"
       ]

       gen_hp_14a_dict = {}

       for idx, (hp_name, hp_row) in enumerate(heat_pumps.iterrows()):
           # 2. Parameter berechnen
           p_nominal = hp_row["p_set"]  # Nennleistung in MW
           p_min_14a = 0.0042           # 4.2 kW
           pmax = p_nominal - p_min_14a

           # Nur wenn WP groß genug (> 4.2 kW)
           if pmax > 1e-6:
               # 3. Virtuellen Generator definieren
               gen_hp_14a_dict[idx] = {
                   "name": f"hp_14a_support_{hp_name}",
                   "gen_bus": hp_row["bus"],  # Gleicher Bus wie WP
                   "pmax": pmax,              # Maximale Abregelleistung
                   "pmin": 0.0,
                   "qmax": 0.0,               # Nur Wirkleistung
                   "qmin": 0.0,
                   "max_hours_per_day": max_hours_per_day,
                   "p_min_14a": p_min_14a,
                   "hp_name": hp_name,        # Referenz zur Original-WP
                   "index": idx,
                   "source_id": f"gen_hp_14a_{idx}"
               }

       # 4. Zeitreihen erstellen (zunächst Nullen, Optimierung setzt Werte)
       gen_hp_14a_p = pd.DataFrame(
           0.0,
           index=timesteps,
           columns=[f"gen_hp_14a_{i}" for i in gen_hp_14a_dict.keys()]
       )

       gen_hp_14a_q = gen_hp_14a_p.copy()

       # 5. In PowerModels-Dictionary einfügen
       for n, ts in enumerate(timesteps):
           pm_dict["nw"][str(n)]["gen_hp_14a"] = gen_hp_14a_dict

       return pm_dict

**Analoger Code für Ladepunkte (``_build_gen_cp_14a_support``).**

Julia: Variablen definieren
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   function variable_gen_hp_14a_power(pm::AbstractPowerModel; nw::Int=nw_id_default)
       """
       Erstellt kontinuierliche Variable für §14a WP-Abregelung.
       """
       gen_hp_14a = PowerModels.ref(pm, nw, :gen_hp_14a)

       # Variable p_hp14a[i] für jeden virtuellen Generator i
       PowerModels.var(pm, nw)[:p_hp14a] = JuMP.@variable(
           pm.model,
           [i in keys(gen_hp_14a)],
           base_name = "p_hp14a_$(nw)",
           lower_bound = 0.0,
           upper_bound = gen_hp_14a[i]["pmax"],
           start = 0.0
       )
   end

   function variable_gen_hp_14a_binary(pm::AbstractPowerModel; nw::Int=nw_id_default)
       """
       Erstellt binäre Variable für §14a WP-Abregelung.
       """
       gen_hp_14a = PowerModels.ref(pm, nw, :gen_hp_14a)

       # Binärvariable z_hp14a[i]
       PowerModels.var(pm, nw)[:z_hp14a] = JuMP.@variable(
           pm.model,
           [i in keys(gen_hp_14a)],
           base_name = "z_hp14a_$(nw)",
           binary = true,
           start = 0
       )
   end

Julia: Constraints implementieren
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   function constraint_hp_14a_binary_coupling(
       pm::AbstractPowerModel,
       i::Int,
       nw::Int=nw_id_default
   )
       """
       Binäre Kopplung: p_hp14a <= pmax × z_hp14a
       """
       p_hp14a = PowerModels.var(pm, nw, :p_hp14a, i)
       z_hp14a = PowerModels.var(pm, nw, :z_hp14a, i)

       gen_hp_14a = PowerModels.ref(pm, nw, :gen_hp_14a, i)
       pmax = gen_hp_14a["pmax"]

       # Constraint
       JuMP.@constraint(pm.model, p_hp14a <= pmax * z_hp14a)
   end

   function constraint_hp_14a_min_net_load(
       pm::AbstractPowerModel,
       i::Int,
       nw::Int=nw_id_default
   )
       """
       Mindest-Nettolast: Last - p_hp14a >= min(Last, 4.2kW)
       """
       p_hp14a = PowerModels.var(pm, nw, :p_hp14a, i)
       gen_hp_14a = PowerModels.ref(pm, nw, :gen_hp_14a, i)

       # Finde zugehörige WP-Last
       hp_name = gen_hp_14a["hp_name"]
       hp_bus = gen_hp_14a["gen_bus"]
       p_hp_load = get_hp_load_at_bus(pm, hp_bus, hp_name, nw)

       pmax = gen_hp_14a["pmax"]
       p_min_14a = gen_hp_14a["p_min_14a"]

       # Spezialfälle
       if pmax < 1e-6
           # WP zu klein
           JuMP.@constraint(pm.model, p_hp14a == 0.0)
       elseif p_hp_load < 1e-6
           # WP ist aus
           JuMP.@constraint(pm.model, p_hp14a == 0.0)
       else
           # Normalbetrieb: Nettolast >= min(Last, Mindestlast)
           min_net_load = min(p_hp_load, p_min_14a)
           JuMP.@constraint(pm.model, p_hp_load - p_hp14a >= min_net_load)
       end
   end

Julia: Zeitbudget-Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   function constraint_hp_14a_time_budget_daily(
       pm::AbstractPowerModel,
       gen_hp_14a_ids,
       timesteps_per_day,
       max_hours_per_day,
       time_elapsed
   )
       """
       Tages-Zeitbudget: Σ(z_hp14a über Tag) <= max_hours / Δt
       """
       # Gruppiere Zeitschritte nach Tagen
       day_groups = group_timesteps_by_day(timesteps_per_day)

       for (day_idx, day_timesteps) in enumerate(day_groups)
           for i in gen_hp_14a_ids
               # Summe binärer Variablen über Tag
               z_sum = sum(
                   PowerModels.var(pm, nw, :z_hp14a, i)
                   for nw in day_timesteps
               )

               # Max. erlaubte Zeitschritte
               max_timesteps = max_hours_per_day / time_elapsed

               # Constraint
               JuMP.@constraint(pm.model, z_sum <= max_timesteps)
           end
       end
   end

**Hilfsfunktion:**

.. code:: julia

   function group_timesteps_by_day(timesteps)
       """
       Gruppiert Zeitschritte in 24h-Tage.

       Beispiel:
       - Input: [2035-01-01 00:00, 2035-01-01 01:00, ..., 2035-01-02 00:00, ...]
       - Output: [[0..23], [24..47], [48..71], ...]
       """
       days = []
       current_day = []
       current_date = Date(timesteps[0])

       for (idx, ts) in enumerate(timesteps)
           ts_date = Date(ts)

           if ts_date == current_date
               push!(current_day, idx)
           else
               push!(days, current_day)
               current_day = [idx]
               current_date = ts_date
           end
       end

       # Letzter Tag
       if !isempty(current_day)
           push!(days, current_day)
       end

       return days
   end

Integration in Optimierungsproblem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Datei:** ``edisgo/opf/eDisGo_OPF.jl/src/prob/opf_bf.jl``

.. code:: julia

   function build_mn_opf_bf_flex(pm::AbstractPowerModel; kwargs...)
       # ... Standard-Variablen ...

       # §14a Variablen (falls aktiviert)
       for n in PowerModels.nw_ids(pm)
           if haskey(PowerModels.ref(pm, n), :gen_hp_14a) &&
              !isempty(PowerModels.ref(pm, n, :gen_hp_14a))

               # Variablen erstellen
               variable_gen_hp_14a_power(pm, nw=n)
               variable_gen_hp_14a_binary(pm, nw=n)
           end

           if haskey(PowerModels.ref(pm, n), :gen_cp_14a) &&
              !isempty(PowerModels.ref(pm, n, :gen_cp_14a))

               variable_gen_cp_14a_power(pm, nw=n)
               variable_gen_cp_14a_binary(pm, nw=n)
           end
       end

       # ... Standard-Constraints ...

       # §14a Constraints pro Zeitschritt
       for n in PowerModels.nw_ids(pm)
           # WP §14a
           for i in PowerModels.ids(pm, :gen_hp_14a, nw=n)
               constraint_hp_14a_binary_coupling(pm, i, n)
               constraint_hp_14a_min_net_load(pm, i, n)
           end

           # CP §14a
           for i in PowerModels.ids(pm, :gen_cp_14a, nw=n)
               constraint_cp_14a_binary_coupling(pm, i, n)
               constraint_cp_14a_min_net_load(pm, i, n)
           end
       end

       # §14a Zeitbudget-Constraints
       if haskey(PowerModels.ref(pm, first(nw_ids)), :gen_hp_14a)
           gen_hp_14a_ids = PowerModels.ids(pm, :gen_hp_14a, nw=first(nw_ids))
           constraint_hp_14a_time_budget_daily(
               pm,
               gen_hp_14a_ids,
               timesteps_per_day,
               max_hours_per_day,
               time_elapsed
           )
       end

       # Analog für CP

       # ... Zielfunktion (mit §14a-Termen) ...
   end

Ergebnisinterpretation
~~~~~~~~~~~~~~~~~~~~~~

Nach der Optimierung sind die §14a-Abregelungen in den Zeitreihen
enthalten:

.. code:: python

   # Optimierung durchführen
   edisgo.pm_optimize(
       opf_version=2,
       curtailment_14a=True,
       max_hours_per_day=2.0
   )

   # §14a-Abregelungen extrahieren
   hp_14a_gens = [
       col for col in edisgo.timeseries.generators_active_power.columns
       if 'hp_14a_support' in col
   ]

   curtailment_hp = edisgo.timeseries.generators_active_power[hp_14a_gens]

   # Beispiel: WP "HP_1234"
   hp_support_gen = "hp_14a_support_HP_1234"
   curtailment_ts = curtailment_hp[hp_support_gen]  # MW

   # Original-Last
   hp_load_ts = edisgo.timeseries.loads_active_power["HP_1234"]  # MW

   # Nettolast (nach Abregelung)
   net_load_ts = hp_load_ts - curtailment_ts

   # Statistiken
   total_curtailment_mwh = curtailment_ts.sum()  # MWh (bei 1h-Auflösung)
   total_load_mwh = hp_load_ts.sum()
   curtailment_percentage = (total_curtailment_mwh / total_load_mwh) * 100

   # Zeitbudget-Nutzung
   hours_curtailed = (curtailment_ts > 0).sum()  # Anzahl Stunden mit Abregelung
   days_in_horizon = len(curtailment_ts) / 24
   avg_hours_per_day = hours_curtailed / days_in_horizon

   print(f"Abregelung HP_1234:")
   print(f"  Total: {total_curtailment_mwh:.2f} MWh ({curtailment_percentage:.1f}%)")
   print(f"  Stunden mit Abregelung: {hours_curtailed}")
   print(f"  Ø pro Tag: {avg_hours_per_day:.2f}h (Limit: 2h)")

**Beispiel-Ausgabe:**

::

   Abregelung HP_1234:
     Total: 15.32 MWh (3.2%)
     Stunden mit Abregelung: 487
     Ø pro Tag: 1.33h (Limit: 2h)

Beispiel-Workflow mit §14a
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # 1. Netz laden
   edisgo = EDisGo(ding0_grid="path/to/grid")

   # 2. Wärmepumpen hinzufügen (alle > 4.2 kW können abgeregelt werden)
   edisgo.add_component(
       comp_type="heat_pump",
       comp_name="HP_001",
       bus="Bus_LV_123",
       p_set=0.008,  # 8 kW Nennleistung
       ...
   )

   # 3. Zeitreihen setzen
   edisgo.set_time_series_manual(
       generators_p=...,
       loads_p=...,
       heat_pump_cop_df=...,
       heat_demand_df=...
   )

   # 4. Optimierung MIT §14a
   edisgo.pm_optimize(
       opf_version=2,              # Mit Netzrestriktionen + Slacks
       curtailment_14a=True,       # §14a aktivieren
       max_hours_per_day=2.0,      # Zeitbudget 2h/Tag
       curtailment_14a_total_hours=None,  # Optional: Gesamtbudget statt täglich
       solver="gurobi",            # Binäre Variablen → MILP-Solver nötig
       warm_start=False            # Kein Warm-Start bei binären Variablen
   )

   # 5. ZWINGEND: Netzausbau für verbleibende Probleme
   edisgo.reinforce()  # Behebt Überlastungen die Optimierung nicht lösen konnte

   # 6. Ergebnisse analysieren
   # §14a Abregelung ist nun in edisgo.timeseries.generators_active_power
   # als virtuelle Generatoren "hp_14a_support_..." und "cp_14a_support_..."

   # Netzausbaukosten (nach Flexibilitätsnutzung)
   costs = edisgo.results.grid_expansion_costs
   print(f"Netzausbaukosten: {costs:,.0f} €")

   # §14a Statistiken
   hp_14a_curtailment = edisgo.timeseries.generators_active_power[
       [c for c in edisgo.timeseries.generators_active_power.columns
        if 'hp_14a_support' in c]
   ]
   cp_14a_curtailment = edisgo.timeseries.generators_active_power[
       [c for c in edisgo.timeseries.generators_active_power.columns
        if 'cp_14a_support' in c]
   ]

   total_curtailment = hp_14a_curtailment.sum().sum() + cp_14a_curtailment.sum().sum()
   print(f"§14a Abregelung gesamt: {total_curtailment:.2f} MWh")

--------------

.. _zeitreihen-nutzung-2:

Zeitreihen-Nutzung
------------------

Überblick Zeitreihen-Datenstruktur
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alle Zeitreihen werden in ``edisgo.timeseries`` gespeichert:

.. code:: python

   edisgo.timeseries
   ├── timeindex: pd.DatetimeIndex          # z.B. 8760 Stunden eines Jahres
   ├── generators_active_power: pd.DataFrame  # MW, Spalten = Generator-Namen
   ├── generators_reactive_power: pd.DataFrame  # MVAr
   ├── loads_active_power: pd.DataFrame      # MW, Spalten = Last-Namen
   ├── loads_reactive_power: pd.DataFrame    # MVAr
   ├── storage_units_active_power: pd.DataFrame  # MW (+ = Entladung, - = Ladung)
   ├── storage_units_reactive_power: pd.DataFrame  # MVAr
   └── ... weitere Komponenten

Zeitreihen-Quellen
~~~~~~~~~~~~~~~~~~

1. Datenbank-Import (eGon-Datenbank)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   edisgo.import_generators(
       generator_scenario="eGon2035",
       engine=db_engine
   )

**Inhalt:** - PV-Anlagen: Zeitreihen aus Wetterdaten (Globalstrahlung) -
Windkraftanlagen: Zeitreihen aus Windgeschwindigkeiten - BHKW:
Wärmegeführter Betrieb oder Grundlast

**Auflösung:** Typisch 1h für ein Jahr (8760 Zeitschritte)

2. Worst-Case-Profile
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   edisgo.set_time_series_worst_case(
       cases=["feed-in_case", "load_case"]
   )

**feed-in_case:** - PV: Maximale Einstrahlung (z.B. Mittag im Sommer) -
Wind: Maximaler Wind - Lasten: Minimale Last

**load_case:** - Generatoren: Minimale Einspeisung - Lasten: Maximale
Last (z.B. Winterabend)

**Nutzung:** Schnelle Netzplanung ohne vollständige Zeitreihen

3. Manuelle Zeitreihen
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Eigene Zeitreihen erstellen
   timesteps = pd.date_range("2035-01-01", periods=8760, freq="H")
   gen_p = pd.DataFrame({
       "PV_001": pv_timeseries,
       "Wind_002": wind_timeseries
   }, index=timesteps)

   edisgo.set_time_series_manual(
       generators_p=gen_p,
       loads_p=load_p,
       ...
   )

4. Optimierte Zeitreihen (nach pm_optimize)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nach ``edisgo.pm_optimize()`` werden Zeitreihen überschrieben mit
optimierten Werten:

.. code:: python

   # VOR Optimierung: Original-Lasten
   hp_load_original = edisgo.timeseries.loads_active_power["HP_123"]

   # Optimierung durchführen
   edisgo.pm_optimize(opf_version=2)

   # NACH Optimierung: Optimierte Lasten
   hp_load_optimized = edisgo.timeseries.loads_active_power["HP_123"]

   # Unterschied: Zeitliche Verschiebung durch thermischen Speicher
   # oder Abregelung durch §14a

**Zusätzlich:** Virtuelle Generatoren für §14a:

.. code:: python

   # NEU nach Optimierung mit curtailment_14a=True
   curtailment = edisgo.timeseries.generators_active_power["hp_14a_support_HP_123"]

Zeitreihen in analyze
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   edisgo.analyze(timesteps=None)

**Verwendete Zeitreihen:** 1. ``generators_active_power``,
``generators_reactive_power`` - Für alle PV, Wind, BHKW, etc. -
Einspeisung ins Netz

2. ``loads_active_power``, ``loads_reactive_power``

   -  Für alle Haushalte, Gewerbe, Industrie
   -  Entnahme aus dem Netz

3. ``storage_units_active_power``, ``storage_units_reactive_power``

   -  Batteriespeicher Ladung/Entladung
   -  Positiv = Entladung (wie Generator)
   -  Negativ = Ladung (wie Last)

4. **Wärmepumpen** (speziell):

   -  NICHT direkt aus ``loads_active_power``
   -  Berechnung: ``P_el(t) = heat_demand(t) / COP(t)``
   -  ``heat_demand_df``: Wärmebedarf in MW_th
   -  ``cop_df``: Coefficient of Performance (temperaturabhängig)

5. **Ladepunkte**:

   -  ``charging_points_active_power`` (falls vorhanden)
   -  Oder aus Flexibilitätsbändern berechnet

**Ablauf:**

.. code:: python

   def to_pypsa(self, mode=None, timesteps=None):
       # 1. Zeitreihen extrahieren
       gen_p = self.timeseries.generators_active_power.loc[timesteps]
       load_p = self.timeseries.loads_active_power.loc[timesteps]

       # 2. Wärmepumpen elektrische Last berechnen
       hp_loads = []
       for hp_name in heat_pumps:
           heat_demand = self.heat_pump.heat_demand_df.loc[timesteps, hp_name]
           cop = self.heat_pump.cop_df.loc[timesteps, hp_name]
           hp_load_p = heat_demand / cop
           hp_loads.append(hp_load_p)

       # 3. Zu PyPSA-Zeitreihen konvertieren
       pypsa.loads_t.p_set = load_p
       pypsa.generators_t.p_set = gen_p
       pypsa.storage_units_t.p_set = storage_p

       return pypsa

Zeitreihen in reinforce
~~~~~~~~~~~~~~~~~~~~~~~

``reinforce`` nutzt dieselben Zeitreihen wie ``analyze``, aber mit
verschiedenen Auswahlmodi:

Modus 1: Snapshot Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   edisgo.reinforce(timesteps_pfa="snapshot_analysis")

**Zeitreihen:** - Nur 2 Zeitschritte: 1. **Max. Residuallast:**
``t_max = argmax(load - generation)`` 2. **Min. Residuallast:**
``t_min = argmin(load - generation)``

**Berechnung:**

.. code:: python

   residual_load = (
       edisgo.timeseries.loads_active_power.sum(axis=1)
       - edisgo.timeseries.generators_active_power.sum(axis=1)
   )

   t_max = residual_load.idxmax()  # Kritischer Zeitpunkt für Überlastung
   t_min = residual_load.idxmin()  # Kritischer Zeitpunkt für Rückspeisung

   timesteps = pd.DatetimeIndex([t_max, t_min])

**Vorteil:** Sehr schnell (nur 2 Power Flow Analysen) **Risiko:**
Übersieht seltene Extremereignisse

Modus 2: Reduzierte Analyse
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   edisgo.reinforce(reduced_analysis=True)

**Ablauf:** 1. Initiale PFA mit allen Zeitschritten 2. Identifiziere
kritische Zeitschritte: - Überlastung > 0% auf irgendeiner Komponente -
Spannungsabweichung > 3% an irgendeinem Knoten 3. Wähle nur diese
Zeitschritte für Verstärkung

**Zeitreihen:**

.. code:: python

   # 1. Initiale Analyse
   edisgo.analyze(timesteps=edisgo.timeseries.timeindex)

   # 2. Kritische Zeitschritte identifizieren
   overloaded_ts = edisgo.results.i_res[
       (edisgo.results.i_res / s_nom).max(axis=1) > 1.0
   ].index

   voltage_issues_ts = edisgo.results.v_res[
       ((edisgo.results.v_res < 0.97) | (edisgo.results.v_res > 1.03)).any(axis=1)
   ].index

   critical_ts = overloaded_ts.union(voltage_issues_ts).unique()

   # 3. Nur diese Zeitschritte für Verstärkung
   timesteps = critical_ts  # z.B. 50 statt 8760

**Vorteil:** Viel schneller als volle Analyse, genauer als Snapshot
**Nachteil:** Initial-PFA mit allen Zeitschritten notwendig

Modus 3: Alle Zeitschritte
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   edisgo.reinforce(timesteps_pfa=None)

**Zeitreihen:** - Alle Zeitschritte in ``edisgo.timeseries.timeindex`` -
Typisch 8760 Stunden für ein Jahr

**Vorteil:** Höchste Genauigkeit **Nachteil:** Sehr rechenintensiv

Modus 4: Custom
^^^^^^^^^^^^^^^

.. code:: python

   # Nur Wintermonate
   winter_ts = edisgo.timeseries.timeindex[
       edisgo.timeseries.timeindex.month.isin([11, 12, 1, 2])
   ]

   edisgo.reinforce(timesteps_pfa=winter_ts)

Zeitreihen in pm_optimize
~~~~~~~~~~~~~~~~~~~~~~~~~

Input-Zeitreihen
^^^^^^^^^^^^^^^^

.. code:: python

   edisgo.pm_optimize(
       opf_version=2,
       curtailment_14a=True
   )

**Verwendete Zeitreihen (vor Optimierung):**

1. **Generatoren:**

   -  ``generators_active_power``: Einspeise-Zeitreihen
   -  ``generators_reactive_power``: Blindleistung (oder aus cos φ
      berechnet)

2. **Inflexible Lasten:**

   -  ``loads_active_power``: Haushalte, Gewerbe ohne Flexibilität
   -  ``loads_reactive_power``: Blindleistung

3. **Batteriespeicher:**

   -  Initial State of Charge (SOC)
   -  Kapazität, Ladeleistung, Entladeleistung
   -  **Keine Input-Zeitreihen** (werden optimiert)

4. **Wärmepumpen:**

   -  ``heat_demand_df``: Wärmebedarf in MW_th
   -  ``cop_df``: COP-Zeitreihen
   -  Flexibilitätsband (falls flexible WP):

      -  ``p_min``, ``p_max`` pro Zeitschritt
      -  Thermischer Speicher: Kapazität, Anfangs-SOC

5. **Ladepunkte:**

   -  Flexibilitätsbänder:

      -  ``p_min``, ``p_max`` pro Zeitschritt
      -  ``energy_min``, ``energy_max`` (SOC-Grenzen)

   -  Ladeeffizienz

6. **§14a-Komponenten (falls aktiviert):**

   -  Alle WP/CP > 4,2 kW werden identifiziert
   -  Virtuelle Generatoren mit ``pmax = P_nom - 4,2kW`` erstellt

Optimierungs-Prozess
^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Python → Julia Datenübergabe
   pm_dict = to_powermodels(
       edisgo_obj,
       timesteps=timesteps,
       curtailment_14a=True
   )

   # pm_dict enthält:
   {
       "multinetwork": true,
       "nw": {
           "0": {  # Zeitschritt 0
               "bus": {...},
               "load": {...},  # Inflexible Lasten (P, Q vorgegeben)
               "gen": {...},   # Generatoren (P, Q vorgegeben)
               "storage": {...},  # Batterie (SOC optimiert)
               "heat_pump": {...},  # WP (P optimiert, Wärmebedarf vorgegeben)
               "charging_point": {...},  # CP (P optimiert, Flexband vorgegeben)
               "gen_hp_14a": {...},  # §14a virtuelle Generatoren (P optimiert)
               "gen_cp_14a": {...}
           },
           "1": {...},  # Zeitschritt 1
           ...
       }
   }

**Julia-Optimierung:** - Löst für alle Zeitschritte gleichzeitig
(Multi-Period OPF) - Bestimmt optimale Fahrpläne für: - Batteriespeicher
(Ladung/Entladung) - Wärmepumpen (elektrische Leistung) - Wärmespeicher
(Beladung/Entladung) - Ladepunkte (Ladeleistung) - §14a-Abregelung
(Curtailment)

Output-Zeitreihen
^^^^^^^^^^^^^^^^^

.. code:: python

   # Julia → Python Ergebnisrückgabe
   result = pm_optimize(...)

   # from_powermodels schreibt optimierte Zeitreihen zurück
   from_powermodels(edisgo_obj, result)

**Aktualisierte Zeitreihen:**

1. **Generatoren:**

   -  ``generators_active_power``:

      -  Original-Generatoren (ggf. mit Curtailment)
      -  **NEU:** Virtuelle §14a-Generatoren (``hp_14a_support_...``,
         ``cp_14a_support_...``)

   -  ``generators_reactive_power``: Optimierte Blindleistung

2. **Speicher:**

   -  ``storage_units_active_power``: Optimierte Ladung/Entladung
   -  ``storage_units_reactive_power``: Optimierte Blindleistung

3. **Wärmepumpen:**

   -  ``loads_active_power`` (WP-Spalten): Optimierte elektrische
      Leistung
   -  Berücksichtigt thermischen Speicher (zeitliche Verschiebung)

4. **Ladepunkte:**

   -  ``loads_active_power`` (CP-Spalten): Optimierte Ladeleistung
   -  Innerhalb Flexibilitätsbänder

5. **§14a-Abregelung (NEU):**

   -  Als virtuelle Generatoren in ``generators_active_power``:
   -  Spalten: ``hp_14a_support_{hp_name}``,
      ``cp_14a_support_{cp_name}``
   -  Werte: Abregelleistung in MW
   -  **Nettolast = Original-Last - Abregelung**

**Beispiel:**

.. code:: python

   # VOR Optimierung
   print(edisgo.timeseries.generators_active_power.columns)
   # ['PV_001', 'Wind_002', 'PV_003', ...]

   print(edisgo.timeseries.loads_active_power.columns)
   # ['Load_HH_001', 'HP_LV_123', 'CP_LV_456', ...]

   # Optimierung MIT §14a
   edisgo.pm_optimize(opf_version=2, curtailment_14a=True)

   # NACH Optimierung
   print(edisgo.timeseries.generators_active_power.columns)
   # ['PV_001', 'Wind_002', 'PV_003', ...,
   #  'hp_14a_support_HP_LV_123',          # NEU!
   #  'cp_14a_support_CP_LV_456']          # NEU!

   # WP-Last optimiert (zeitlich verschoben durch thermischen Speicher)
   hp_load_opt = edisgo.timeseries.loads_active_power["HP_LV_123"]

   # §14a-Abregelung
   hp_curtailment = edisgo.timeseries.generators_active_power["hp_14a_support_HP_LV_123"]

   # Effektive Nettolast
   hp_net_load = hp_load_opt - hp_curtailment

Zeitreihen-Auflösung
~~~~~~~~~~~~~~~~~~~~

Die Zeitauflösung beeinflusst Optimierung und Genauigkeit:

+------------+----------------------+----------------------+-----------+
| Auflösung  | Zeitschritte/Jahr    | Optimierungsgröße    | Use Case  |
+============+======================+======================+===========+
| 1 Stunde   | 8760                 | Groß (langsam)       | Jahressi  |
|            |                      |                      | mulation, |
|            |                      |                      | det       |
|            |                      |                      | aillierte |
|            |                      |                      | Planung   |
+------------+----------------------+----------------------+-----------+
| 15 Minuten | 35040                | Sehr groß            | Intr      |
|            |                      |                      | aday-Flex |
|            |                      |                      | ibilität, |
|            |                      |                      | genaue    |
|            |                      |                      | Ne        |
|            |                      |                      | tzanalyse |
+------------+----------------------+----------------------+-----------+
| 1 Tag      | 365                  | Klein (schnell)      | Grobe     |
|            |                      |                      | Planung,  |
|            |                      |                      | Se        |
|            |                      |                      | nsitivitä |
|            |                      |                      | tsstudien |
+------------+----------------------+----------------------+-----------+
| Worst-Case | 2-10                 | Sehr klein           | Schnelle  |
|            |                      |                      | Net       |
|            |                      |                      | zplanung, |
|            |                      |                      | Screening |
+------------+----------------------+----------------------+-----------+

**Wichtig für §14a:** - Zeitbudget skaliert mit Auflösung: -
1h-Auflösung: ``max_timesteps_per_day = 2h / 1h = 2`` - 15min-Auflösung:
``max_timesteps_per_day = 2h / 0.25h = 8``

Zeitreihen-Persistenz
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Speichern
   edisgo.save(directory="results/grid_001")

   # Lädt:
   # - Netztopologie (Leitungen, Trafos, Busse)
   # - Zeitreihen (alle DataFrames)
   # - Results (PFA-Ergebnisse)
   # - Optimierungsergebnisse (§14a-Curtailment)

   # Laden
   edisgo = EDisGo(directory="results/grid_001")

   # Zeitreihen direkt verfügbar
   timesteps = edisgo.timeseries.timeindex
   gen_p = edisgo.timeseries.generators_active_power

--------------

Dateipfade und Referenzen
-------------------------

Python-Dateien
~~~~~~~~~~~~~~

+-----------------+---------------+------------------------------------+
| Datei           | Pfad          | Beschreibung                       |
+=================+===============+====================================+
| **EDisGo        | ``edisg       | ``analyze()``, ``reinforce()``,    |
| Hauptklasse**   | o/edisgo.py`` | ``pm_optimize()`` Methoden         |
+-----------------+---------------+------------------------------------+
| **PowerModels   | ``edi         | ``to_powermodels()``,              |
| I/O**           | sgo/io/powerm | ``from_powermodels()``,            |
|                 | odels_io.py`` | §14a-Generatoren                   |
+-----------------+---------------+------------------------------------+
| **PowerModels   | ``edisg       | Julia-Subprozess,                  |
| OPF**           | o/opf/powermo | JSON-Kommunikation                 |
|                 | dels_opf.py`` |                                    |
+-----------------+---------------+------------------------------------+
| **Reinforce     | ``edisgo/fl   | Verstärkungsalgorithmus            |
| I               | ex_opt/reinfo |                                    |
| mplementation** | rce_grid.py`` |                                    |
+-----------------+---------------+------------------------------------+
| **Reinforce     | ``            | Leitungs-/Trafo-Verstärkung        |
| Measures**      | edisgo/flex_o |                                    |
|                 | pt/reinforce_ |                                    |
|                 | measures.py`` |                                    |
+-----------------+---------------+------------------------------------+
| **Timeseries**  | ``edisg       | Zeitreihen-Verwaltung              |
|                 | o/edisgo.py`` |                                    |
|                 | (Klasse       |                                    |
|                 | ``            |                                    |
|                 | TimeSeries``) |                                    |
+-----------------+---------------+------------------------------------+

Julia-Dateien
~~~~~~~~~~~~~

+-----------------+---------------+------------------------------------+
| Datei           | Pfad          | Beschreibung                       |
+=================+===============+====================================+
| **Main Entry**  | ``edisgo/o    | Solver-Setup, JSON I/O             |
|                 | pf/eDisGo_OPF |                                    |
|                 | .jl/Main.jl`` |                                    |
+-----------------+---------------+------------------------------------+
| **OPF Problem** | ``edisgo      | ``build_mn_opf_bf_flex()``         |
|                 | /opf/eDisGo_O |                                    |
|                 | PF.jl/src/pro |                                    |
|                 | b/opf_bf.jl`` |                                    |
+-----------------+---------------+------------------------------------+
| **Variables**   | ``edisgo/op   | Alle Variablendefinitionen         |
|                 | f/eDisGo_OPF. |                                    |
|                 | jl/src/core/v |                                    |
|                 | ariables.jl`` |                                    |
+-----------------+---------------+------------------------------------+
| **Constraints** | ``edisgo/opf  | Standard-Constraints               |
|                 | /eDisGo_OPF.j |                                    |
|                 | l/src/core/co |                                    |
|                 | nstraint.jl`` |                                    |
+-----------------+---------------+------------------------------------+
| **§14a HP       | ``edis        | WP-§14a-Constraints                |
| Constraints**   | go/opf/eDisGo |                                    |
|                 | _OPF.jl/src/c |                                    |
|                 | ore/constrain |                                    |
|                 | t_hp_14a.jl`` |                                    |
+-----------------+---------------+------------------------------------+
| **§14a CP       | ``edis        | CP-§14a-Constraints                |
| Constraints**   | go/opf/eDisGo |                                    |
|                 | _OPF.jl/src/c |                                    |
|                 | ore/constrain |                                    |
|                 | t_cp_14a.jl`` |                                    |
+-----------------+---------------+------------------------------------+
| **Objective**   | ``edisgo/op   | Zielfunktionen (4 Versionen)       |
|                 | f/eDisGo_OPF. |                                    |
|                 | jl/src/core/o |                                    |
|                 | bjective.jl`` |                                    |
+-----------------+---------------+------------------------------------+
| **Branch Flow** | ``ed          | Branch-Flow-Formulierung           |
|                 | isgo/opf/eDis |                                    |
|                 | Go_OPF.jl/src |                                    |
|                 | /form/bf.jl`` |                                    |
+-----------------+---------------+------------------------------------+
| **Data          | ``edis        | Datenstrukturen                    |
| Handling**      | go/opf/eDisGo |                                    |
|                 | _OPF.jl/src/c |                                    |
|                 | ore/data.jl`` |                                    |
+-----------------+---------------+------------------------------------+
| **Solution      | ``edisgo/o    | Ergebnis-Extraktion                |
| Processing**    | pf/eDisGo_OPF |                                    |
|                 | .jl/src/core/ |                                    |
|                 | solution.jl`` |                                    |
+-----------------+---------------+------------------------------------+

Beispiele
~~~~~~~~~

+-----------------+---------------+------------------------------------+
| Datei           | Pfad          | Beschreibung                       |
+=================+===============+====================================+
| **§14a          | ``example     | Beispiel für §14a-Optimierung und  |
| Analyse**       | s/example_ana | Auswertung                         |
|                 | lyze_14a.py`` |                                    |
+-----------------+---------------+------------------------------------+
| **Standard      | ``exam        | Standard-OPF ohne §14a             |
| Optimierung**   | ples/example_ |                                    |
|                 | optimize.py`` |                                    |
+-----------------+---------------+------------------------------------+
| **              | ``examp       | Netzausbau-Beispiel                |
| Reinforcement** | les/example_r |                                    |
|                 | einforce.py`` |                                    |
+-----------------+---------------+------------------------------------+

Konfigurationsdateien
~~~~~~~~~~~~~~~~~~~~~

+-----------------+---------------+------------------------------------+
| Datei           | Pfad          | Beschreibung                       |
+=================+===============+====================================+
| **Config**      | ``edisgo/c    | Default-Einstellungen              |
|                 | onfig/config_ | (Spannungsgrenzen,                 |
|                 | default.cfg`` | Kostenparameter)                   |
+-----------------+---------------+------------------------------------+
| **Equipment     | ``edisgo/c    | Leitungs-/Trafo-Typen mit techn.   |
| Data**          | onfig/equipme | Daten                              |
|                 | nt_data.csv`` |                                    |
+-----------------+---------------+------------------------------------+

--------------