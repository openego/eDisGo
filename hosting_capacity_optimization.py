"""
AP 5.4 - Methode 2: Optimierung mit §14a EnWG
==============================================

Dieser Code implementiert die §14a-basierte Optimierungsmethode zur Bestimmung der
Hosting Capacity mit INDIVIDUELLEN Leistungen pro Ladesäule durch:
- Szenario S1: Eine Ladesäule am HV/MV Trafo
- Szenario S2: Je eine Ladesäule an jedem MV Bus + jedem LV ONS (oberes Ende)
- Szenario S3: Gleichmäßige Verteilung (Referenz)
- Szenario S4: Je eine Ladesäule an weitestmöglichen Punkten (unteres Ende)
- Szenario S5: Je eine Ladesäule am weitesten LV Bus pro LV Grid

Methodik:
- Keine E-Mobilität aus Datenbank importiert
- Ladesäulen werden pro Szenario hinzugefügt
- Nur Worst-Case Zeitschritte werden analysiert (standardmäßig 4)

Optimierungsansatz (§14a EnWG):
- Alle CPs werden auf unrealistisch hohe Leistung gesetzt
- Virtuelle Generatoren werden an jedem CP-Bus hinzugefügt
- PowerModels.jl Optimierung bestimmt optimale "Curtailment" pro CP
- Optimale CP-Leistung = Ausgangsleistung - Curtailment
- Erlaubt INDIVIDUELLE Leistungen pro CP
"""

import json
import logging
import os

from datetime import datetime

import numpy as np
import pandas as pd

from edisgo import EDisGo


def setup_logging(script_name):
    """
    Richtet Logging mit Timestamp für Console und File ein.
    Erfasst auch Logs von externen Bibliotheken (PyPSA, eDisGo, etc.)

    Parameters
    ----------
    script_name : str
        Name des Skripts (z.B. 'ap54_optimization')

    Returns
    -------
    logging.Logger
        Konfigurierter Logger
    """
    # Erstelle logs Verzeichnis falls nicht vorhanden
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Erstelle Timestamp für Dateiname
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    # Formatter für strukturierte Logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Konfiguriere Root Logger um ALLE Logs zu erfassen (inkl. externe Bibliotheken)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Entferne existierende Handler um Duplikate zu vermeiden
    root_logger.handlers.clear()

    # File Handler für alle Logs
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console Handler für alle Logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Erstelle spezifischen Logger für dieses Script
    logger = logging.getLogger(script_name)
    logger.info(f"Logging gestartet - Log-Datei: {log_filename}")

    return logger


class HostingCapacityOptimization:
    """
    Klasse zur Berechnung der Hosting Capacity durch Optimierungsszenarien.

    Die 5 Szenarien (S1-S5) repräsentieren verschiedene extreme Anordnungen
    der Ladeinfrastruktur, um obere und untere Grenzen der Hosting Capacity
    zu bestimmen.
    """

    def __init__(self, edisgo_obj, logger=None):
        """
        Parameters
        ----------
        edisgo_obj : EDisGo
            Basis eDisGo Objekt (ohne Elektromobilität)
        logger : logging.Logger, optional
            Logger für Ausgaben
        """
        self.edisgo_base = edisgo_obj
        self.logger = logger if logger else logging.getLogger(__name__)
        self.worst_case_timesteps = None

    def create_scenario_s1(self, p_set=50000000000000000):
        """
        S1: Eine Ladesäule am HV/MV Trafo.

        Parameters
        ----------
        p_set : float
            Anfangsleistung pro Ladesäule in kW

        Returns
        -------
        EDisGo
            eDisGo Objekt mit S1 Konfiguration
        """
        edisgo_s1 = self.edisgo_base.copy(deep=True)

        # Entferne alle bisherigen Ladepunkte (falls vorhanden)
        self._remove_all_charging_points(edisgo_s1)

        # Füge eine Ladesäule am HV/MV Trafo hinzu
        hvmv_bus = edisgo_s1.topology.transformers_hvmv_df.iloc[0]["bus1"]

        edisgo_s1.add_component(
            comp_type="load",
            type="charging_point",
            bus=hvmv_bus,
            p_set=p_set,
            sector="charging_point_s1",
        )

        return edisgo_s1

    def create_scenario_s2(self, p_set=500000000000):
        """
        S2: Je eine Ladesäule an jedem MV Bus + je eine an jedem LV ONS.
        Ergibt oberes Ende der Aufnahmefähigkeit.

        Parameters
        ----------
        p_set : float
            Leistung pro Ladesäule in kW

        Returns
        -------
        EDisGo
            eDisGo Objekt mit S2 Konfiguration
        """
        edisgo_s2 = self.edisgo_base.copy(deep=True)

        # Entferne alle bisherigen Ladepunkte
        self._remove_all_charging_points(edisgo_s2)

        # Füge eine Ladesäule an jedem MV Bus hinzu
        mv_buses = edisgo_s2.topology.mv_grid.buses_df.index[1:]  # Ohne HV/MV Station
        for i, bus in enumerate(mv_buses):
            edisgo_s2.add_component(
                comp_type="load",
                type="charging_point",
                bus=bus,
                p_set=p_set,
                sector=f"mv_cp_{i}_s2",
            )

        # Füge eine Ladesäule an jedem LV ONS hinzu
        lv_grids = list(edisgo_s2.topology.mv_grid.lv_grids)
        for lv_grid in lv_grids:
            # ONS ist bus1 des ersten Transformers
            ons_bus = lv_grid.transformers_df.iloc[0]["bus1"]

            edisgo_s2.add_component(
                comp_type="load",
                type="charging_point",
                bus=ons_bus,
                p_set=p_set,
                sector=f"lv_{lv_grid.id}_ons_s2",
            )

        return edisgo_s2

    def create_scenario_s3(self, p_set=1240):
        """
        S3: Referenzszenario - gleichmäßige Verteilung.
        Je eine Ladesäule an ausgewählten MV und LV Buses.

        Parameters
        ----------
        p_set : float
            Leistung pro Ladesäule in kW

        Returns
        -------
        EDisGo
            eDisGo Objekt mit S3 Konfiguration
        """
        edisgo_s3 = self.edisgo_base.copy(deep=True)

        # Entferne alle bisherigen Ladepunkte
        self._remove_all_charging_points(edisgo_s3)

        # Füge Ladesäulen an 50% der MV Buses hinzu (gleichmäßig verteilt)
        mv_buses = edisgo_s3.topology.mv_grid.buses_df.index[1:]  # Ohne HV/MV Station
        mv_sample = mv_buses[::2]  # Jeder 2. Bus
        for i, bus in enumerate(mv_sample):
            edisgo_s3.add_component(
                comp_type="load",
                type="charging_point",
                bus=bus,
                p_set=p_set,
                sector=f"mv_cp_{i}_s3",
            )

        # Füge Ladesäulen an 50% der LV Buses hinzu (pro LV Grid)
        lv_grids = list(edisgo_s3.topology.mv_grid.lv_grids)
        for lv_grid in lv_grids:
            lv_buses = lv_grid.buses_df.index
            lv_sample = lv_buses[::2]  # Jeder 2. Bus
            for i, bus in enumerate(lv_sample):
                edisgo_s3.add_component(
                    comp_type="load",
                    type="charging_point",
                    bus=bus,
                    p_set=p_set,
                    sector=f"lv_{lv_grid.id}_cp_{i}_s3",
                )

        return edisgo_s3

    def create_scenario_s4(self, p_set=1240):
        """
        S4: Je eine Ladesäule an weitestmöglichen Punkten im Netz.
        Ergibt unteres Ende der Aufnahmefähigkeit.

        Parameters
        ----------
        p_set : float
            Leistung pro Ladesäule in kW

        Returns
        -------
        EDisGo
            eDisGo Objekt mit S4 Konfiguration
        """
        edisgo_s4 = self.edisgo_base.copy(deep=True)

        # Entferne alle bisherigen Ladepunkte
        self._remove_all_charging_points(edisgo_s4)

        # Füge eine Ladesäule am weitesten MV Bus hinzu
        furthest_mv_bus = self._find_furthest_bus_mv(edisgo_s4)
        edisgo_s4.add_component(
            comp_type="load",
            type="charging_point",
            bus=furthest_mv_bus,
            p_set=p_set,
            sector="mv_furthest_s4",
        )

        # Füge eine Ladesäule am weitesten LV Bus pro LV Grid hinzu
        lv_grids = list(edisgo_s4.topology.mv_grid.lv_grids)
        for lv_grid in lv_grids:
            furthest_lv_bus = self._find_furthest_bus_lv(lv_grid)

            edisgo_s4.add_component(
                comp_type="load",
                type="charging_point",
                bus=furthest_lv_bus,
                p_set=p_set,
                sector=f"lv_{lv_grid.id}_furthest_s4",
            )

        return edisgo_s4

    def create_scenario_s5(self, p_set=1240):
        """
        S5: Je eine Ladesäule an jedem LV Bus (am weitesten Punkt jedes LV Grids).
        Keine CPs im MV Netz.

        Parameters
        ----------
        p_set : float
            Leistung pro Ladesäule in kW

        Returns
        -------
        EDisGo
            eDisGo Objekt mit S5 Konfiguration
        """
        edisgo_s5 = self.edisgo_base.copy(deep=True)

        # Entferne alle bisherigen Ladepunkte
        self._remove_all_charging_points(edisgo_s5)

        # Füge je eine Ladesäule am weitesten LV Bus pro LV Grid hinzu
        lv_grids = list(edisgo_s5.topology.mv_grid.lv_grids)
        for lv_grid in lv_grids:
            furthest_lv_bus = self._find_furthest_bus_lv(lv_grid)

            edisgo_s5.add_component(
                comp_type="load",
                type="charging_point",
                bus=furthest_lv_bus,
                p_set=p_set,
                sector=f"lv_{lv_grid.id}_furthest_s5",
            )

        return edisgo_s5

    def run_hosting_capacity_optimization(
        self,
        max_cp_power_kw=500.0,
    ):
        """
        Führt die Hosting Capacity Optimierung für alle Szenarien durch.

        Parameters
        ----------
        max_cp_power_kw : float
            Maximale CP-Leistung in kW für Optimierung (default: 500 kW)
            - nur für optimization

        Returns
        -------
        pd.DataFrame
            Ergebnisse für alle Szenarien
        """
        results_list = []

        # Identifiziere Worst-Case Zeitschritte (einmal für alle Szenarien)
        self.logger.info("\nIdentifiziere Worst-Case Zeitschritte...")
        self.edisgo_base.set_time_series_worst_case_analysis()
        self.worst_case_timesteps = self.edisgo_base.timeseries.timeindex

        self.logger.info(
            f"\nVerwende §14a OPTIMIERUNG "
            f"(individuelle CP-Leistungen, max {max_cp_power_kw:.0f} kW)"
        )
        # WICHTIG: Erstelle alle Szenarien VOR jeglichem weiteren Reinforcement
        # (deepcopy des edisgo_base muss funktionieren)
        self.logger.info("\nErstelle alle Szenarien (deepcopy von Basis-Grid)...")
        scenarios = {
            "S1_all_at_hvmv": self.create_scenario_s1(),
            "S2_aggregated_optimal": self.create_scenario_s2(),
            "S3_reference": self.create_scenario_s3(),
            "S4_furthest": self.create_scenario_s4(),
            "S5_all_lv_furthest": self.create_scenario_s5(),
        }
        self.logger.info(f"✓ Alle {len(scenarios)} Szenarien erstellt")

        for scenario_name, edisgo_obj in scenarios.items():
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Analysiere Szenario: {scenario_name}")
            self.logger.info(f"{'='*70}")

            # Komplettes Szenario in try-except, um alle Fehler zu loggen
            try:
                # Zähle Ladepunkte
                n_charging_points = len(
                    edisgo_obj.topology.loads_df[
                        edisgo_obj.topology.loads_df["type"] == "charging_point"
                    ]
                )
                self.logger.info(f"Anzahl Ladepunkte: {n_charging_points}")
                # §14a Optimierung - individuelle CP-Leistungen
                hc_result = self._calculate_hosting_capacity_optimization(
                    edisgo_obj, max_cp_power_kw=max_cp_power_kw
                )

                results_list.append(
                    {
                        "scenario": scenario_name,
                        "n_charging_points": n_charging_points,
                        **hc_result,
                    }
                )

            except Exception as e:
                self.logger.error(f"✗ Fehler bei Szenario {scenario_name}: {e}")
                import traceback

                self.logger.error("Vollständiger Traceback:")
                self.logger.error(traceback.format_exc())
                results_list.append({"scenario": scenario_name, "error": str(e)})

        results_df = pd.DataFrame(results_list)
        self.results = results_df

        # Bestimme oberes und unteres Ende
        if not results_df.empty and "max_power_per_cp_kw" in results_df.columns:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("HOSTING CAPACITY GRENZEN:")
            self.logger.info("=" * 60)

            # Prüfe ob S2 und S4 Ergebnisse haben
            s2_rows = results_df[results_df["scenario"] == "S2_aggregated_optimal"]
            s4_rows = results_df[results_df["scenario"] == "S4_furthest"]

            if not s2_rows.empty and "max_power_per_cp_kw" in s2_rows.columns:
                s2_hc = s2_rows["max_power_per_cp_kw"].values
                if len(s2_hc) > 0:
                    self.logger.info(
                        f"Oberes Ende (S2): {s2_hc[0]:.2f} kW pro Ladesäule"
                    )
            else:
                self.logger.warning("S2 Szenario hat keine Ergebnisse (fehlgeschlagen)")

            if not s4_rows.empty and "max_power_per_cp_kw" in s4_rows.columns:
                s4_hc = s4_rows["max_power_per_cp_kw"].values
                if len(s4_hc) > 0:
                    self.logger.info(
                        f"Unteres Ende (S4): {s4_hc[0]:.2f} kW pro Ladesäule"
                    )
            else:
                self.logger.warning("S4 Szenario hat keine Ergebnisse (fehlgeschlagen)")

            # Berechne Spanne nur wenn beide vorhanden
            if (
                not s2_rows.empty
                and not s4_rows.empty
                and "max_power_per_cp_kw" in s2_rows.columns
                and "max_power_per_cp_kw" in s4_rows.columns
            ):
                s2_hc = s2_rows["max_power_per_cp_kw"].values
                s4_hc = s4_rows["max_power_per_cp_kw"].values
                if len(s2_hc) > 0 and len(s4_hc) > 0:
                    self.logger.info(f"Spanne: {s2_hc[0] - s4_hc[0]:.2f} kW")

        return results_df

    # ===== Hilfsfunktionen =====

    def _remove_all_charging_points(self, edisgo_obj):
        """Entfernt alle Ladepunkte aus dem Netz."""
        charging_points = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df["type"] == "charging_point"
        ].index.tolist()

        for cp in charging_points:
            edisgo_obj.remove_component(comp_type="load", comp_name=cp)

    def _find_furthest_bus_mv(self, edisgo_obj):
        """Findet weitesten Bus im MV Netz (von HV/MV Station).

        Berechnet die Summe der Leitungslaengen (km) entlang des
        kuerzesten Pfades von der HV/MV Station zu jedem MV Bus
        und gibt den Bus mit der groessten Distanz zurueck.
        """
        import networkx as nx

        mv_grid = edisgo_obj.topology.mv_grid
        mv_buses = mv_grid.buses_df

        if mv_buses.empty:
            return None

        graph = mv_grid.graph
        station = mv_grid.station.index[0]

        max_dist = -1
        furthest_bus = mv_buses.index[0]

        for bus in mv_buses.index:
            try:
                dist = nx.shortest_path_length(
                    graph, source=station, target=bus,
                    weight="length",
                )
                if dist > max_dist:
                    max_dist = dist
                    furthest_bus = bus
            except nx.NetworkXNoPath:
                continue

        return furthest_bus

    def _find_furthest_bus_lv(self, lv_grid):
        """Findet weitesten Bus im LV Netz (von ONS).

        Berechnet die Summe der Leitungslaengen (km) entlang des
        kuerzesten Pfades von der ONS zum jeweiligen LV Bus und
        gibt den Bus mit der groessten Distanz zurueck.
        """
        import networkx as nx

        lv_buses = lv_grid.buses_df

        if lv_buses.empty:
            return None

        graph = lv_grid.graph
        station = lv_grid.station.index[0]

        max_dist = -1
        furthest_bus = lv_buses.index[0]

        for bus in lv_buses.index:
            try:
                dist = nx.shortest_path_length(
                    graph, source=station, target=bus,
                    weight="length",
                )
                if dist > max_dist:
                    max_dist = dist
                    furthest_bus = bus
            except nx.NetworkXNoPath:
                continue

        return furthest_bus

    def _check_network_violations(self, edisgo_obj):
        """
        Prüft ob Netzrestriktionen verletzt werden.

        Parameters
        ----------
        edisgo_obj : EDisGo
            eDisGo Objekt (muss bereits analyze() ausgeführt haben)

        Returns
        -------
        dict
            Dictionary mit Violation-Informationen
        """
        from edisgo.flex_opt.check_tech_constraints import (
            lines_relative_load,
            voltage_issues,
        )

        # Prüfe Leitungsüberlastungen
        rel_load = lines_relative_load(edisgo_obj)
        max_rel_load = rel_load.max().max() if not rel_load.empty else 0
        has_overloading = (rel_load > 1.0).any().any() if not rel_load.empty else False

        # Prüfe Spannungsprobleme
        v_issues = voltage_issues(edisgo_obj, voltage_level=None)
        has_voltage_issues = not v_issues.empty
        max_voltage_dev = (
            v_issues["abs_max_voltage_dev"].max() if not v_issues.empty else 0
        )

        # Netzprobleme vorhanden?
        has_violations = has_overloading or has_voltage_issues

        return {
            "has_violations": has_violations,
            "has_overloading": has_overloading,
            "has_voltage_issues": has_voltage_issues,
            "max_line_loading": max_rel_load,
            "max_voltage_deviation": max_voltage_dev,
            "n_overloaded_lines": (
                int((rel_load > 1.0).any().sum()) if not rel_load.empty else 0
            ),
            "n_voltage_issues": len(v_issues) if not v_issues.empty else 0,
        }

    def _ensure_complete_timeseries(self, edisgo_obj):
        """
        Stellt sicher dass alle Komponenten in der Topologie Zeitreihen haben.
        Fehlende Zeitreihen werden mit sinnvollen Defaults aufgefüllt.

        Parameters
        ----------
        edisgo_obj : EDisGo
            eDisGo Objekt
        """
        # Prüfe Loads (Active Power)
        all_loads = edisgo_obj.topology.loads_df.index
        existing_load_ts = edisgo_obj.timeseries.loads_active_power.columns
        missing_loads = set(all_loads) - set(existing_load_ts)

        if missing_loads:
            self.logger.info(
                f"Füge fehlende Active Power Zeitreihen für "
                f"{len(missing_loads)} Loads hinzu"
            )
            # Kopiere bestehende Zeitreihen
            loads_p = edisgo_obj.timeseries.loads_active_power.copy()
            for load in missing_loads:
                p_set = edisgo_obj.topology.loads_df.loc[load, "p_set"]
                # Setze auf nominale Leistung
                loads_p[load] = p_set
            edisgo_obj.set_time_series_manual(loads_p=loads_p)

        # Prüfe Loads (Reactive Power)
        existing_load_q_ts = edisgo_obj.timeseries.loads_reactive_power.columns
        missing_loads_q = set(all_loads) - set(existing_load_q_ts)

        if missing_loads_q:
            self.logger.info(
                f"Füge fehlende Reactive Power Zeitreihen für "
                f"{len(missing_loads_q)} Loads hinzu"
            )
            # Kopiere bestehende Zeitreihen
            loads_q = edisgo_obj.timeseries.loads_reactive_power.copy()
            for load in missing_loads_q:
                # Setze auf 0 (Blindleistung erstmal vernachlässigen)
                loads_q[load] = 0.0
            edisgo_obj.set_time_series_manual(loads_q=loads_q)

        # Prüfe Generatoren (Active Power)
        all_gens = edisgo_obj.topology.generators_df.index
        existing_gen_ts = edisgo_obj.timeseries.generators_active_power.columns
        missing_gens = set(all_gens) - set(existing_gen_ts)

        if missing_gens:
            self.logger.info(
                f"Füge fehlende Active Power Zeitreihen für "
                f"{len(missing_gens)} Generatoren hinzu"
            )
            # Kopiere bestehende Zeitreihen
            gens_p = edisgo_obj.timeseries.generators_active_power.copy()
            for gen in missing_gens:
                # Setze auf 0 (konservativ, da wir nicht wissen ob der Generator läuft)
                gens_p[gen] = 0.0
            edisgo_obj.set_time_series_manual(generators_p=gens_p)

        # Prüfe Generatoren (Reactive Power)
        existing_gen_q_ts = edisgo_obj.timeseries.generators_reactive_power.columns
        missing_gens_q = set(all_gens) - set(existing_gen_q_ts)

        if missing_gens_q:
            self.logger.info(
                f"Füge fehlende Reactive Power Zeitreihen für "
                f"{len(missing_gens_q)} Generatoren hinzu"
            )
            # Kopiere bestehende Zeitreihen
            gens_q = edisgo_obj.timeseries.generators_reactive_power.copy()
            for gen in missing_gens_q:
                gens_q[gen] = 0.0
            edisgo_obj.set_time_series_manual(generators_q=gens_q)

    def _set_charging_timeseries(self, edisgo_obj, power_per_cp_mw):
        """
        Aktualisiert Zeitreihen für Ladepunkte: Volle Last bei Worst-Case
        Zeitschritten, sonst 0. Behält alle anderen Last-Zeitreihen bei.

        Parameters
        ----------
        edisgo_obj : EDisGo
            eDisGo Objekt mit Ladepunkten
        power_per_cp_mw : float
            Leistung pro Ladesäule in MW
        """
        # Hole alle Ladepunkte
        charging_points = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df["type"] == "charging_point"
        ].index.tolist()

        if not charging_points:
            return

        # WICHTIG: Nimm bestehende Zeitreihen und update nur Charging Points
        # Damit gehen Haushalts-Lasten etc. nicht verloren
        timeindex = edisgo_obj.timeseries.timeindex

        # Wenn keine Zeitreihen existieren, erstelle neue
        if edisgo_obj.timeseries.loads_active_power.empty:
            all_loads_ts = pd.DataFrame(
                0.0, index=timeindex, columns=edisgo_obj.topology.loads_df.index
            )
        else:
            # Kopiere bestehende Zeitreihen
            all_loads_ts = edisgo_obj.timeseries.loads_active_power.copy()

        # Füge Charging Point Spalten hinzu falls sie noch nicht existieren
        missing_cps = [
            cp for cp in charging_points if cp not in all_loads_ts.columns
        ]
        if missing_cps:
            missing_df = pd.DataFrame(
                0.0, index=all_loads_ts.index, columns=missing_cps
            )
            all_loads_ts = pd.concat([all_loads_ts, missing_df], axis=1)

        # Setze Charging Points auf 0 für alle Zeitschritte
        all_loads_ts.loc[:, charging_points] = 0.0

        # Setze Worst-Case Zeitschritte auf volle Leistung (nur für CPs)
        if self.worst_case_timesteps is not None:
            for ts in self.worst_case_timesteps:
                if ts in all_loads_ts.index:
                    all_loads_ts.loc[ts, charging_points] = power_per_cp_mw

        # Setze komplette Zeitreihen (mit allen Lasten)
        edisgo_obj.set_time_series_manual(loads_p=all_loads_ts)

    def _calculate_hosting_capacity_optimization(
        self, edisgo_obj, max_cp_power_kw=500.0
    ):
        """
        Berechnet Hosting Capacity durch §14a Optimierung mit virtuellen
        Generatoren.

        Diese Methode ermöglicht es, für jede Ladesäule eine INDIVIDUELLE
        optimale Leistung zu finden.

        Methodik:
        1. Setze alle CPs auf unrealistisch hohe Leistung
        2. Füge virtuelle Generatoren an jedem CP-Bus hinzu
        3. Optimierung "curtailed" die CPs durch die virtuellen Generatoren
        4. Optimale CP-Leistung = Ausgangsleistung - Curtailment

        Parameters
        ----------
        edisgo_obj : EDisGo
            eDisGo Objekt mit Ladepunkten
        max_cp_power_kw : float
            Maximale unrealistische CP-Leistung in kW (default: 500 kW)

        Returns
        -------
        dict
            Dictionary mit Hosting Capacity Ergebnissen inkl. individueller
            CP-Leistungen
        """
        # Zähle Ladepunkte
        charging_points = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df["type"] == "charging_point"
        ].index.tolist()

        n_charging_points = len(charging_points)

        if n_charging_points == 0:
            self.logger.warning("Keine Ladepunkte vorhanden")
            return {
                "max_power_per_cp_kw": 0,
                "max_power_per_cp_mw": 0,
                "total_power_mw": 0,
                "max_line_loading": 0,
                "has_overloading": False,
                "has_voltage_issues": False,
                "cp_powers": {},
            }

        self.logger.info(
            f"\nVerwende §14a Optimierung für {n_charging_points} Ladesäulen"
        )
        self.logger.info(
            f"Setze alle CPs auf {max_cp_power_kw:.1f} kW (unrealistisch hoch)"
        )

        # Erstelle Kopie
        edisgo_opt = edisgo_obj.copy(deep=True)

        # Setze alle CPs auf hohe Leistung
        max_cp_power_mw = max_cp_power_kw / 1000.0
        edisgo_opt.topology.loads_df.loc[charging_points, "p_set"] = max_cp_power_mw

        # Erstelle Zeitreihen (nur Worst-Case Zeitschritte mit Last)
        self._set_charging_timeseries(edisgo_opt, max_cp_power_mw)

        # Stelle sicher dass alle Komponenten Zeitreihen haben
        # (wichtig nach .copy() da manche Zeitreihen fehlen können)
        self._ensure_complete_timeseries(edisgo_opt)

        # Konfiguriere §14a Curtailment
        # - max_power_mw: 0.0 = keine minimale Leistung
        #   (CPs können auf 0 reduziert werden)
        # - max_hours_per_day: 24.0 = kein Zeitbudget-Limit
        #   (kann ganzen Tag curtailen)
        curtailment_14a_config = {
            "max_power_mw": 0.0,  # Keine minimale Leistung
            "max_hours_per_day": 24.0,  # Kein Zeitbudget
            "components": [],  # Leer = alle CPs
        }

        self.logger.info("Konfiguration §14a Curtailment:")
        self.logger.info(
            f"  - Minimale Leistung: "
            f"{curtailment_14a_config['max_power_mw']*1000:.1f} kW "
            f"(keine Grenze)"
        )
        self.logger.info(
            f"  - Zeitbudget: "
            f"{curtailment_14a_config['max_hours_per_day']:.1f} h/Tag "
            f"(kein Limit)"
        )

        # Führe Optimierung aus
        self.logger.info("\nStarte PowerModels Optimierung...")
        try:
            edisgo_opt.pm_optimize(
                opf_version=2,  # Version 2 nutzt §14a support
                curtailment_14a=curtailment_14a_config,
            )
            self.logger.info("✓ Optimierung erfolgreich")
        except Exception as e:
            self.logger.error(f"✗ Optimierung fehlgeschlagen: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return {
                "max_power_per_cp_kw": 0,
                "max_power_per_cp_mw": 0,
                "total_power_mw": 0,
                "max_line_loading": 0,
                "has_overloading": True,
                "has_voltage_issues": True,
                "cp_powers": {},
                "error": str(e),
            }

        # Extrahiere Curtailment-Ergebnisse aus den virtuellen Generatoren
        self.logger.info("\nExtrahiere individuelle CP-Leistungen...")

        # Die virtuellen Generatoren haben Namen wie
        # "cp_14a_support_Load_charging_point_..."
        cp_14a_generators = [
            col
            for col in edisgo_opt.timeseries.generators_active_power.columns
            if "cp_14a_support" in col
        ]

        self.logger.info(
            f"Gefunden: {len(cp_14a_generators)} virtuelle §14a Generatoren"
        )

        if len(cp_14a_generators) == 0:
            self.logger.warning(
                "⚠ Keine §14a Generatoren gefunden - "
                "Optimierung hat keine Curtailment durchgeführt"
            )
            # Alle CPs bleiben bei voller Leistung
            cp_powers = {cp: max_cp_power_kw for cp in charging_points}
        else:
            # Berechne Curtailment für jeden CP
            cp_powers = {}
            for cp_name in charging_points:
                # Finde den zugehörigen virtuellen Generator
                # Name-Pattern: "cp_14a_support_{cp_name}"
                gen_prefix = f"cp_14a_support_{cp_name}"
                matching_gens = [
                    gen for gen in cp_14a_generators
                    if gen == gen_prefix or gen.startswith(gen_prefix)
                ]

                if matching_gens:
                    gen_name = matching_gens[0]
                    # Curtailment = maximale Leistung des virtuellen
                    # Generators über Worst-Case Zeitschritte (konservativ)
                    curtailment_mw = edisgo_opt.timeseries.generators_active_power.loc[
                        self.worst_case_timesteps, gen_name
                    ].max()

                    # Optimale CP-Leistung = Initial-Leistung - Curtailment
                    optimal_power_mw = max_cp_power_mw - curtailment_mw
                    optimal_power_kw = optimal_power_mw * 1000.0

                    # Sicherstellen dass Leistung nicht negativ wird
                    optimal_power_kw = max(0.0, optimal_power_kw)

                    cp_powers[cp_name] = optimal_power_kw

                    self.logger.info(
                        f"  {cp_name}: {optimal_power_kw:.1f} kW "
                        f"(Curtailment: {curtailment_mw*1000:.1f} kW)"
                    )
                else:
                    # Kein Generator gefunden - CP bleibt bei voller Leistung
                    cp_powers[cp_name] = max_cp_power_kw
                    self.logger.info(
                        f"  {cp_name}: {max_cp_power_kw:.1f} kW "
                        f"(kein Curtailment nötig)"
                    )

        # WICHTIG: Setze CP-Lasten auf optimierte Werte für finale Analyse
        self.logger.info("\nSetze CP-Lasten auf optimierte Werte...")
        loads_p = edisgo_opt.timeseries.loads_active_power.copy()
        for cp_name, optimal_power_kw in cp_powers.items():
            optimal_power_mw = optimal_power_kw / 1000.0
            # Update p_set in topology
            edisgo_opt.topology.loads_df.loc[cp_name, "p_set"] = optimal_power_mw
            # Update timeseries for all worst-case timesteps
            loads_p.loc[self.worst_case_timesteps, cp_name] = optimal_power_mw

        # Update reactive power proportionally (CPs have pf=1.0 -> Q=0)
        loads_q = edisgo_opt.timeseries.loads_reactive_power.copy()
        for cp_name in cp_powers:
            if cp_name in loads_q.columns:
                loads_q.loc[self.worst_case_timesteps, cp_name] = 0.0

        edisgo_opt.set_time_series_manual(loads_p=loads_p, loads_q=loads_q)
        self.logger.info("✓ Lasten auf optimierte Werte gesetzt (P und Q)")

        # Führe finale Power Flow Analyse durch um Verletzungen zu prüfen
        self.logger.info("\nFühre finale Power Flow Analyse durch...")
        try:
            edisgo_opt.analyze(timesteps=self.worst_case_timesteps)
            violations = self._check_network_violations(edisgo_opt)

            self.logger.info(
                f"Finale Netzanalyse: "
                f"{violations['n_overloaded_lines']} Überlastungen, "
                f"{violations['n_voltage_issues']} Spannungsprobleme"
            )
        except Exception as e:
            self.logger.warning(f"Finale Analyse fehlgeschlagen: {e}")
            violations = {
                "has_violations": True,
                "has_overloading": False,
                "has_voltage_issues": False,
                "max_line_loading": 0,
                "max_voltage_deviation": 0,
                "n_overloaded_lines": 0,
                "n_voltage_issues": 0,
            }

        # Berechne Statistiken
        avg_power_kw = np.mean(list(cp_powers.values()))
        min_power_kw = np.min(list(cp_powers.values()))
        max_power_kw = np.max(list(cp_powers.values()))
        total_power_mw = sum(cp_powers.values()) / 1000.0

        self.logger.info("\n✓ Optimierung abgeschlossen:")
        self.logger.info(f"  Durchschnitt: {avg_power_kw:.1f} kW pro CP")
        self.logger.info(f"  Minimum: {min_power_kw:.1f} kW")
        self.logger.info(f"  Maximum: {max_power_kw:.1f} kW")
        self.logger.info(f"  Gesamt: {total_power_mw:.3f} MW ({n_charging_points} CPs)")

        return {
            "max_power_per_cp_kw": avg_power_kw,  # Durchschnitt für Vergleich
            "max_power_per_cp_mw": avg_power_kw / 1000.0,
            "min_power_per_cp_kw": min_power_kw,
            "max_power_individual_kw": max_power_kw,
            "total_power_mw": total_power_mw,
            "max_line_loading": violations["max_line_loading"],
            "has_overloading": violations["has_overloading"],
            "has_voltage_issues": violations["has_voltage_issues"],
            "n_overloaded_lines": violations["n_overloaded_lines"],
            "n_voltage_issues": violations["n_voltage_issues"],
            "max_voltage_deviation": violations["max_voltage_deviation"],
            "cp_powers": cp_powers,  # Individuelle Leistungen pro CP
            "method": "optimization",
        }


def run_analysis(grid_path, output_dir=None):
    """
    Führt die Hosting Capacity Optimierung für ein gegebenes Grid durch.

    Parameters
    ----------
    grid_path : str
        Pfad zum Ding0 Grid
    output_dir : str, optional
        Pfad zum Output-Verzeichnis. Wenn None, wird im aktuellen
        Verzeichnis gespeichert.

    Returns
    -------
    pd.DataFrame
        Ergebnisse der Hosting Capacity Optimierung
    """
    # Setup Logging
    logger = setup_logging("ap54_optimization")

    if output_dir is None:
        output_dir = "."

    logger.info("=" * 70)
    logger.info("AP 5.4 - Methode 2: Hosting Capacity Optimierung")
    logger.info("=" * 70)
    logger.info(f"\nLade Grid: {grid_path}")

    # 1. Initialisiere eDisGo mit Ding0 Grid
    # Setze Zeitindex - nutze stündliche Auflösung
    timeindex = pd.date_range(
        "2023-01-01", periods=24, freq="H"
    )  # 1 Tag, stündliche Auflösung
    edisgo = EDisGo(ding0_grid=grid_path, legacy_ding0_grids=False, timeindex=timeindex)

    # 2. Setze Worst-Case Zeitreihen fuer alle Lasten und Generatoren
    logger.info("\nSetze Worst-Case Zeitreihen...")
    edisgo.set_time_series_worst_case_analysis()
    logger.info("✓ Zeitreihen gesetzt")

    # 3. WICHTIG: Erstelle Hosting Capacity Optimierungs Objekt VOR dem Reinforcement
    #    (deepcopy funktioniert nicht nach reinforce wegen Thread-lokaler Objekte)
    logger.info("\nErstelle Hosting Capacity Optimierung (vor Reinforcement)...")
    hc_opt = HostingCapacityOptimization(edisgo, logger=logger)

    # 4. Verstärke das Netz um Basisprobleme zu beheben
    logger.info("\nVerstärke Netz um Basisprobleme zu beheben...")
    edisgo.reinforce()
    logger.info("✓ Netz verstärkt - Basisprobleme behoben")

    # 5. Führe Optimierung durch
    logger.info("\nFühre Optimierung für alle Szenarien durch...")
    logger.info("Verwende §14a Optimierung mit virtuellen Generatoren")
    logger.info("-" * 70)
    results = hc_opt.run_hosting_capacity_optimization(
        max_cp_power_kw=500000000000.0,  # Maximale unrealistische CP-Leistung
    )

    # 6. Speichere Ergebnisse
    output_file = os.path.join(output_dir, "hosting_capacity_optimization_results.csv")

    # Speichere Haupt-CSV (ohne cp_powers dict, da das nicht CSV-kompatibel ist)
    results_for_csv = results.copy()
    if "cp_powers" in results_for_csv.columns:
        # Entferne cp_powers dict für CSV
        cp_powers_data = results_for_csv["cp_powers"].to_dict()
        results_for_csv = results_for_csv.drop(columns=["cp_powers"])
    else:
        cp_powers_data = None

    results_for_csv.to_csv(output_file, index=False)
    logger.info(f"\n✓ Ergebnisse gespeichert: {output_file}")

    # Speichere individuelle CP-Leistungen pro Szenario als CSV
    if cp_powers_data is not None:
        cp_powers_dir = os.path.join(output_dir, "cp_powers")
        os.makedirs(cp_powers_dir, exist_ok=True)
        for idx, cp_dict in cp_powers_data.items():
            scenario_name = (
                results.loc[idx, "scenario"]
                if idx in results.index
                else f"scenario_{idx}"
            )
            if isinstance(cp_dict, dict) and cp_dict:
                cp_df = pd.DataFrame(
                    list(cp_dict.items()),
                    columns=["charging_point", "power_kw"],
                )
                cp_df = cp_df.sort_values("power_kw", ascending=False)
                cp_csv = os.path.join(cp_powers_dir, f"{scenario_name}.csv")
                cp_df.to_csv(cp_csv, index=False)
                logger.info(f"✓ CP-Leistungen gespeichert: {cp_csv}")

    # 7. Zeige Zusammenfassung
    logger.info("\n" + "=" * 70)
    logger.info("ZUSAMMENFASSUNG")
    logger.info("=" * 70)
    logger.info("\n" + results.to_string(index=False))

    return results


if __name__ == "__main__":
    import sys

    # Prüfe Kommandozeilenargumente
    if len(sys.argv) > 1:
        grid_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    else:
        # Default: Beispielgrid
        grid_path = "/home/gurobi/.ding0/2024-07-25T17:38:34_new_planning_new_edisgo/ding0_grids/30879"  # noqa
        output_dir = "."

    # Führe Analyse aus (setup_logging wird in run_analysis aufgerufen)
    results = run_analysis(
        grid_path=grid_path, output_dir=output_dir
    )
