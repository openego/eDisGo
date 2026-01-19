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
- Alle CPs werden auf unrealistisch hohe Leistung gesetzt (500 kW)
- Virtuelle Generatoren werden an jedem CP-Bus hinzugefügt
- PowerModels.jl Optimierung bestimmt optimale "Curtailment" pro CP
- Optimale CP-Leistung = 500 kW - Curtailment
- Erlaubt INDIVIDUELLE Leistungen pro CP (im Gegensatz zu Binary Search)

Alternative Methode (Binary Search):
- Findet UNIFORME maximale Leistung pro Ladesäule
- Kann durch method='binary_search' aktiviert werden
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
        self.scenarios = {}
        self.logger = logger if logger else logging.getLogger(__name__)
        self.worst_case_timesteps = None

    def create_scenario_s1(self, p_set=0.011):
        """
        S1: Eine Ladesäule am HV/MV Trafo.

        Parameters
        ----------
        p_set : float
            Anfangsleistung pro Ladesäule in MW (default: 11 kW)

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

    def create_scenario_s2(self, p_set=0.011):
        """
        S2: Je eine Ladesäule an jedem MV Bus + je eine an jedem LV ONS.
        Ergibt oberes Ende der Aufnahmefähigkeit.

        Parameters
        ----------
        p_set : float
            Leistung pro Ladesäule in MW (default: 11 kW)

        Returns
        -------
        EDisGo
            eDisGo Objekt mit S2 Konfiguration
        """
        edisgo_s2 = self.edisgo_base.copy(deep=True)

        # Entferne alle bisherigen Ladepunkte
        self._remove_all_charging_points(edisgo_s2)

        # Füge eine Ladesäule an jedem MV Bus hinzu
        mv_buses = edisgo_s2.topology.mv_grid.buses_df.index
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

    def create_scenario_s3(self, p_set=0.011):
        """
        S3: Referenzszenario - gleichmäßige Verteilung.
        Je eine Ladesäule an ausgewählten MV und LV Buses.

        Parameters
        ----------
        p_set : float
            Leistung pro Ladesäule in MW (default: 11 kW)

        Returns
        -------
        EDisGo
            eDisGo Objekt mit S3 Konfiguration
        """
        edisgo_s3 = self.edisgo_base.copy(deep=True)

        # Entferne alle bisherigen Ladepunkte
        self._remove_all_charging_points(edisgo_s3)

        # Füge Ladesäulen an 50% der MV Buses hinzu (gleichmäßig verteilt)
        mv_buses = edisgo_s3.topology.mv_grid.buses_df.index
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

    def create_scenario_s4(self, p_set=0.011):
        """
        S4: Je eine Ladesäule an weitestmöglichen Punkten im Netz.
        Ergibt unteres Ende der Aufnahmefähigkeit.

        Parameters
        ----------
        p_set : float
            Leistung pro Ladesäule in MW (default: 11 kW)

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

    def create_scenario_s5(self, p_set=0.011):
        """
        S5: Je eine Ladesäule an jedem LV Bus (am weitesten Punkt jedes LV Grids).
        Keine CPs im MV Netz.

        Parameters
        ----------
        p_set : float
            Leistung pro Ladesäule in MW (default: 11 kW)

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

    def _identify_worst_case_timesteps(self, edisgo_obj, n_timesteps=4):
        """
        Identifiziert die n Worst-Case Zeitschritte basierend auf höchster Gesamtlast.

        Parameters
        ----------
        edisgo_obj : EDisGo
            eDisGo Objekt
        n_timesteps : int
            Anzahl der Worst-Case Zeitschritte (default: 4)

        Returns
        -------
        list
            Liste der Worst-Case Zeitstempel
        """
        # Berechne Gesamtlast pro Zeitschritt
        total_load = edisgo_obj.timeseries.loads_active_power.sum(axis=1)

        # Finde die n höchsten Zeitschritte
        worst_case_indices = total_load.nlargest(n_timesteps).index.tolist()

        self.logger.info(f"Worst-Case Zeitschritte identifiziert: {worst_case_indices}")
        return worst_case_indices

    def run_hosting_capacity_optimization(
        self,
        n_worst_case_timesteps=4,
        tolerance=0.01,
        max_iterations=15,
        method="optimization",
        max_cp_power_kw=500.0,
    ):
        """
        Führt die Hosting Capacity Optimierung für alle Szenarien durch.

        Parameters
        ----------
        n_worst_case_timesteps : int
            Anzahl der Worst-Case Zeitschritte (default: 4)
        tolerance : float
            Konvergenz-Toleranz für iterative Methode (default: 1%)
            - nur für binary_search
        max_iterations : int
            Maximale Iterationen für Binary Search (default: 15)
            - nur für binary_search
        method : str
            Optimierungsmethode: 'optimization' (§14a mit individuellen
            CP-Leistungen) oder 'binary_search' (uniforme CP-Leistung).
            Default: 'optimization'
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
        self.worst_case_timesteps = self._identify_worst_case_timesteps(
            self.edisgo_base, n_timesteps=n_worst_case_timesteps
        )

        # Log welche Methode verwendet wird
        if method == "optimization":
            self.logger.info(
                f"\nVerwende §14a OPTIMIERUNG "
                f"(individuelle CP-Leistungen, max {max_cp_power_kw:.0f} kW)"
            )
        else:
            self.logger.info("\nVerwende BINARY SEARCH (uniforme CP-Leistung)")

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

                # Wähle Methode
                if method == "optimization":
                    # §14a Optimierung - individuelle CP-Leistungen
                    hc_result = self._calculate_hosting_capacity_optimization(
                        edisgo_obj, max_cp_power_kw=max_cp_power_kw
                    )
                else:
                    # Binary Search - uniforme CP-Leistung
                    hc_result = self._calculate_hosting_capacity_iterative(
                        edisgo_obj, tolerance=tolerance, max_iterations=max_iterations
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

    def _calculate_total_charging_power(self, edisgo_obj):
        """Berechnet Gesamt-Ladeleistung."""
        charging_points = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df["type"] == "charging_point"
        ]
        return {
            "p_set": charging_points["p_set"].sum() if not charging_points.empty else 0
        }

    def _remove_all_charging_points(self, edisgo_obj):
        """Entfernt alle Ladepunkte aus dem Netz."""
        charging_points = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df["type"] == "charging_point"
        ].index.tolist()

        for cp in charging_points:
            edisgo_obj.remove_component(comp_type="load", comp_name=cp)

    def _classify_charging_points_by_voltage(self, edisgo_obj):
        """Klassifiziert Ladepunkte nach Spannungsebene (MV/LV)."""
        charging_points = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df["type"] == "charging_point"
        ]

        mv_buses = edisgo_obj.topology.mv_grid.buses_df.index
        mv_cps = charging_points[charging_points["bus"].isin(mv_buses)].index.tolist()
        lv_cps = charging_points[~charging_points["bus"].isin(mv_buses)].index.tolist()

        return mv_cps, lv_cps

    def _calculate_charging_power_subset(self, edisgo_obj, cp_list):
        """Berechnet Gesamtleistung einer Teilmenge von Ladepunkten."""
        if not cp_list:
            return 0

        charging_points = edisgo_obj.topology.loads_df.loc[cp_list]
        return charging_points["p_set"].sum()

    def _is_cp_in_lv_grid(self, edisgo_obj, cp_name, lv_grid):
        """Prüft ob Ladepunkt in bestimmtem LV Grid liegt."""
        cp_bus = edisgo_obj.topology.loads_df.loc[cp_name, "bus"]
        return cp_bus in lv_grid.buses_df.index

    def _find_furthest_bus_mv(self, edisgo_obj):
        """Findet weitesten Bus im MV Netz (von HV/MV Station)."""
        # Nutze Entfernung von Station als Proxy
        # Vereinfachte Implementierung: Bus mit höchstem Index
        mv_buses = edisgo_obj.topology.mv_grid.buses_df
        # TODO: Implementiere tatsächliche Distanzberechnung
        return mv_buses.index[-1] if not mv_buses.empty else None

    def _find_furthest_bus_lv(self, lv_grid):
        """Findet weitesten Bus im LV Netz (von ONS)."""
        # Vereinfachte Implementierung
        lv_buses = lv_grid.buses_df
        # TODO: Implementiere tatsächliche Distanzberechnung
        return lv_buses.index[-1] if not lv_buses.empty else None

    def _aggregate_charging_timeseries(self, edisgo_obj):
        """Aggregiert alle Lade-Zeitreihen."""
        charging_ts = edisgo_obj.timeseries.loads_active_power.filter(like="Charging")
        if charging_ts.empty:
            return pd.DataFrame()
        return charging_ts.sum(axis=1).to_frame("aggregated_charging")

    def _set_worst_case_loading(self, edisgo_obj, load_factor=1.0):
        """
        Setzt nur die Ladepunkte auf Worst-Case (z.B. 100%),
        nicht die Basislasten.
        """
        # Identifiziere Ladepunkte
        charging_points = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df["type"] == "charging_point"
        ].index.tolist()

        # Multipliziere nur die Ladepunkt-Zeitreihen mit load_factor
        if hasattr(edisgo_obj.timeseries, "loads_active_power"):
            charging_ts_cols = [
                col
                for col in edisgo_obj.timeseries.loads_active_power.columns
                if col in charging_points
            ]
            if charging_ts_cols:
                edisgo_obj.timeseries._loads_active_power[
                    charging_ts_cols
                ] *= load_factor

    def _scale_charging_power(self, edisgo_obj, scaling_factor):
        """
        Skaliert die Ladeleistung aller Ladepunkte.

        Parameters
        ----------
        edisgo_obj : EDisGo
            eDisGo Objekt
        scaling_factor : float
            Skalierungsfaktor (z.B. 0.5 = 50%, 1.0 = 100%)

        Returns
        -------
        EDisGo
            eDisGo Objekt mit skalierter Ladeleistung
        """
        # Skaliere p_set für alle Ladepunkte
        charging_points = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df["type"] == "charging_point"
        ].index

        if not charging_points.empty:
            edisgo_obj.topology.loads_df.loc[charging_points, "p_set"] *= scaling_factor

            # Skaliere auch Zeitreihen
            if hasattr(edisgo_obj.timeseries, "loads_active_power"):
                charging_ts_cols = [
                    col
                    for col in edisgo_obj.timeseries.loads_active_power.columns
                    if col in charging_points
                ]
                if charging_ts_cols:
                    edisgo_obj.timeseries._loads_active_power[
                        charging_ts_cols
                    ] *= scaling_factor

        return edisgo_obj

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
                len(rel_load[rel_load > 1.0].dropna()) if not rel_load.empty else 0
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
        for cp in charging_points:
            if cp not in all_loads_ts.columns:
                all_loads_ts[cp] = 0.0

        # Setze Charging Points auf 0 für alle Zeitschritte
        all_loads_ts.loc[:, charging_points] = 0.0

        # Setze Worst-Case Zeitschritte auf volle Leistung (nur für CPs)
        if self.worst_case_timesteps:
            for ts in self.worst_case_timesteps:
                if ts in all_loads_ts.index:
                    all_loads_ts.loc[ts, charging_points] = power_per_cp_mw

        # Setze komplette Zeitreihen (mit allen Lasten)
        edisgo_obj.set_time_series_manual(loads_p=all_loads_ts)

    def _calculate_hosting_capacity_iterative(
        self, edisgo_obj, tolerance=0.01, max_iterations=15
    ):
        """
        Berechnet Hosting Capacity durch iterative Binary Search.

        Findet die maximale Leistung PRO Ladesäule, die ohne Netzprobleme
        integriert werden kann (nur für Worst-Case Zeitschritte).

        Parameters
        ----------
        edisgo_obj : EDisGo
            eDisGo Objekt mit Ladepunkten
        tolerance : float
            Konvergenz-Toleranz in MW (default: 0.01 MW = 10 kW)
        max_iterations : int
            Maximale Anzahl Iterationen (default: 15)

        Returns
        -------
        dict
            Dictionary mit Hosting Capacity Ergebnissen
        """
        # Zähle Ladepunkte
        n_charging_points = len(
            edisgo_obj.topology.loads_df[
                edisgo_obj.topology.loads_df["type"] == "charging_point"
            ]
        )

        if n_charging_points == 0:
            self.logger.warning("Keine Ladepunkte vorhanden")
            return {
                "max_power_per_cp_kw": 0,
                "max_power_per_cp_mw": 0,
                "total_power_mw": 0,
                "iterations": 0,
                "max_line_loading": 0,
                "has_overloading": False,
                "has_voltage_issues": False,
            }

        # Binary Search Grenzen (in MW pro Ladesäule)
        lower_bound = 0.0  # Untere Grenze (immer feasible)
        upper_bound = 0.5  # Obere Grenze (initial: 500 kW pro Säule)

        best_feasible_power = 0.0
        best_violations = None

        # Teste zuerst das Basisnetz (0 kW) um zu prüfen, ob es bereits Probleme hat
        self.logger.info("\nPrüfe Basisnetz ohne Ladesäulen...")
        edisgo_base = edisgo_obj.copy(deep=True)
        charging_points = edisgo_base.topology.loads_df[
            edisgo_base.topology.loads_df["type"] == "charging_point"
        ].index
        edisgo_base.topology.loads_df.loc[charging_points, "p_set"] = 0.0
        self._set_charging_timeseries(edisgo_base, 0.0)

        try:
            edisgo_base.analyze(timesteps=self.worst_case_timesteps)
            base_violations = self._check_network_violations(edisgo_base)
            self.logger.info(
                f"  Basisnetz: {base_violations['n_overloaded_lines']} Überlastungen, "
                f"{base_violations['n_voltage_issues']} Spannungsprobleme"
            )

            if base_violations["has_violations"]:
                self.logger.warning(
                    "⚠ Das Basisnetz hat bereits Probleme OHNE Ladesäulen!"
                )
                self.logger.warning("  Hosting Capacity wird 0 kW sein.")
                # Setze best_violations auf base violations für korrekte Ausgabe
                best_violations = base_violations
        except Exception as e:
            self.logger.warning(f"  Fehler bei Basisnetz-Analyse: {e}")

        self.logger.info(
            f"\nStarte iterative HC-Berechnung "
            f"(max {max_iterations} Iterationen, "
            f"Toleranz {tolerance*1000:.1f} kW)"
        )

        for iteration in range(max_iterations):
            # Berechne Testleistung pro Ladesäule (Mittelwert)
            test_power_mw = (lower_bound + upper_bound) / 2.0

            self.logger.info(
                f"  Iteration {iteration+1}: "
                f"Teste {test_power_mw*1000:.1f} kW pro Ladesäule"
            )

            # Erstelle Kopie
            edisgo_test = edisgo_obj.copy(deep=True)

            # Setze p_set für alle Ladepunkte
            charging_points = edisgo_test.topology.loads_df[
                edisgo_test.topology.loads_df["type"] == "charging_point"
            ].index
            edisgo_test.topology.loads_df.loc[charging_points, "p_set"] = test_power_mw

            # Erstelle Zeitreihen (nur Worst-Case Zeitschritte mit Last)
            self._set_charging_timeseries(edisgo_test, test_power_mw)

            # Stelle sicher dass alle Komponenten Zeitreihen haben
            self._ensure_complete_timeseries(edisgo_test)

            # Führe Power Flow aus (nur für Worst-Case Zeitschritte)
            try:
                edisgo_test.analyze(timesteps=self.worst_case_timesteps)

                # Prüfe Netzrestriktionen
                violations = self._check_network_violations(edisgo_test)

                if violations["has_violations"]:
                    # Zu viel Leistung - reduziere upper bound
                    upper_bound = test_power_mw
                    self.logger.info(
                        f"    → Netzprobleme: "
                        f"{violations['n_overloaded_lines']} Überlastungen, "
                        f"{violations['n_voltage_issues']} Spannungsprobleme"
                    )
                else:
                    # Feasible - erhöhe lower bound
                    lower_bound = test_power_mw
                    best_feasible_power = test_power_mw
                    best_violations = violations
                    self.logger.info(
                        f"    ✓ Feasible: "
                        f"Max Line Loading {violations['max_line_loading']:.2f}"
                    )

            except Exception as e:
                # Power Flow gescheitert - zu viel Leistung
                self.logger.warning(
                    f"    Power Flow fehlgeschlagen bei "
                    f"{test_power_mw*1000:.1f} kW: {e}"
                )
                upper_bound = test_power_mw

            # Konvergenz-Check
            if (upper_bound - lower_bound) < tolerance:
                self.logger.info(f"  Konvergiert nach {iteration+1} Iterationen")
                break

        # Berechne finale Hosting Capacity
        total_power = best_feasible_power * n_charging_points

        self.logger.info(
            f"✓ Hosting Capacity gefunden: "
            f"{best_feasible_power*1000:.1f} kW pro Ladesäule"
        )
        self.logger.info(
            f"  ({n_charging_points} Ladesäulen × "
            f"{best_feasible_power*1000:.1f} kW = "
            f"{total_power*1000:.1f} kW gesamt)"
        )

        return {
            "max_power_per_cp_kw": best_feasible_power * 1000,  # In kW
            "max_power_per_cp_mw": best_feasible_power,
            "total_power_mw": total_power,
            "iterations": iteration + 1,
            "max_line_loading": (
                best_violations["max_line_loading"] if best_violations else 0
            ),
            "has_overloading": (
                best_violations["has_overloading"] if best_violations else False
            ),
            "has_voltage_issues": (
                best_violations["has_voltage_issues"] if best_violations else False
            ),
            "n_overloaded_lines": (
                best_violations["n_overloaded_lines"] if best_violations else 0
            ),
            "n_voltage_issues": (
                best_violations["n_voltage_issues"] if best_violations else 0
            ),
            "max_voltage_deviation": (
                best_violations["max_voltage_deviation"] if best_violations else 0
            ),
        }

    def _calculate_hosting_capacity_optimization(
        self, edisgo_obj, max_cp_power_kw=500.0
    ):
        """
        Berechnet Hosting Capacity durch §14a Optimierung mit virtuellen
        Generatoren.

        Diese Methode ermöglicht es, für jede Ladesäule eine INDIVIDUELLE
        optimale Leistung zu finden (im Gegensatz zur Binary Search, die
        eine uniforme Leistung findet).

        Methodik:
        1. Setze alle CPs auf unrealistisch hohe Leistung (z.B. 500 kW)
        2. Füge virtuelle Generatoren an jedem CP-Bus hinzu (max = 500 kW, sehr günstig)
        3. Optimierung "curtailed" die CPs durch die virtuellen Generatoren
        4. Optimale CP-Leistung = 500 kW - Curtailment

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
                matching_gens = [gen for gen in cp_14a_generators if cp_name in gen]

                if matching_gens:
                    gen_name = matching_gens[0]
                    # Curtailment = durchschnittliche Leistung des virtuellen
                    # Generators über Worst-Case Zeitschritte
                    curtailment_mw = edisgo_opt.timeseries.generators_active_power.loc[
                        self.worst_case_timesteps, gen_name
                    ].mean()

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

        edisgo_opt.set_time_series_manual(loads_p=loads_p)
        self.logger.info("✓ Lasten auf optimierte Werte gesetzt")

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


def example_usage():
    """Beispiel zur Verwendung."""
    # edisgo = EDisGo(ding0_grid="path/to/grid")
    # edisgo.import_electromobility(...)
    #
    # hc_opt = HostingCapacityOptimization(edisgo)
    # results = hc_opt.run_hosting_capacity_optimization()
    # print(results)
    pass


def run_analysis(grid_path, output_dir=None, n_worst_case_timesteps=4):
    """
    Führt die Hosting Capacity Optimierung für ein gegebenes Grid durch.

    Parameters
    ----------
    grid_path : str
        Pfad zum Ding0 Grid
    output_dir : str, optional
        Pfad zum Output-Verzeichnis. Wenn None, wird im aktuellen
        Verzeichnis gespeichert.
    n_worst_case_timesteps : int, optional
        Anzahl der Worst-Case Zeitschritte (default: 4)

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

    # 2. Setze Zeitreihen für Basislasten
    logger.info("\nSetze Zeitreihen für Basislasten...")
    try:
        edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts="oedb", conventional_loads_ts="demandlib"
        )
        logger.info("✓ Zeitreihen gesetzt")
    except Exception as e:
        logger.warning(f"Warnung beim Setzen der Zeitreihen: {e}")

    # 3. WICHTIG: Erstelle Hosting Capacity Optimierung VOR dem Reinforcement
    #    (deepcopy funktioniert nicht nach reinforce wegen Thread-lokaler Objekte)
    logger.info("\nErstelle Hosting Capacity Optimierung (vor Reinforcement)...")
    hc_opt = HostingCapacityOptimization(edisgo, logger=logger)

    # 4. Verstärke das Netz um Basisprobleme zu beheben
    #    Dies geschieht jetzt NACH dem Erstellen der Szenarien
    logger.info("\nVerstärke Netz um Basisprobleme zu beheben...")
    try:
        edisgo.reinforce()
        logger.info("✓ Netz verstärkt - Basisprobleme behoben")
    except Exception as e:
        logger.warning(f"Warnung beim Verstärken des Netzes: {e}")

    # 5. Führe Optimierung durch
    logger.info("\nFühre Optimierung für alle Szenarien durch...")
    logger.info(f"Nutze {n_worst_case_timesteps} Worst-Case Zeitschritte")
    logger.info("Verwende §14a Optimierung mit virtuellen Generatoren")
    logger.info("-" * 70)
    results = hc_opt.run_hosting_capacity_optimization(
        n_worst_case_timesteps=n_worst_case_timesteps,
        method="optimization",  # §14a Optimierung mit individuellen CP-Leistungen
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

    # Speichere individuelle CP-Leistungen als JSON (falls vorhanden)
    if cp_powers_data is not None:
        cp_powers_file = os.path.join(output_dir, "hosting_capacity_cp_powers.json")
        # Konvertiere zu serialisierbarem Format
        cp_powers_serializable = {}
        for idx, cp_dict in cp_powers_data.items():
            scenario_name = (
                results.loc[idx, "scenario"]
                if idx in results.index
                else f"scenario_{idx}"
            )
            if isinstance(cp_dict, dict):
                # Konvertiere CP-Namen zu Strings für JSON
                cp_powers_serializable[scenario_name] = {
                    str(k): float(v) for k, v in cp_dict.items()
                }

        with open(cp_powers_file, "w") as f:
            json.dump(cp_powers_serializable, f, indent=2)
        logger.info(f"✓ Individuelle CP-Leistungen gespeichert: {cp_powers_file}")

    # 7. Zeige Zusammenfassung
    logger.info("\n" + "=" * 70)
    logger.info("ZUSAMMENFASSUNG")
    logger.info("=" * 70)
    logger.info("\n" + results.to_string(index=False))

    return results


if __name__ == "__main__":
    import sys

    # Setup Logging für main
    logger = setup_logging("ap54_optimization")

    logger.info("AP 5.4 - Methode 2: Optimierung")
    logger.info("=" * 50)
    logger.info(
        "Implementiert 5 Szenarien zur Bestimmung der Hosting Capacity Grenzen."
    )
    logger.info("")

    # Prüfe Kommandozeilenargumente
    if len(sys.argv) > 1:
        grid_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        n_worst_case = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    else:
        # Default: Beispielgrid
        grid_path = "/home/gurobi/.ding0/2024-07-25T17:38:34_new_planning_new_edisgo/ding0_grids/30879"  # noqa
        output_dir = "."
        n_worst_case = 4
        logger.info(f"Verwende Beispielgrid: {grid_path}")
        logger.info(f"Ausgabe in: {output_dir}")
        logger.info(f"Worst-Case Zeitschritte: {n_worst_case}")
        logger.info("")

    # Führe Analyse aus
    results = run_analysis(
        grid_path=grid_path, output_dir=output_dir, n_worst_case_timesteps=n_worst_case
    )
