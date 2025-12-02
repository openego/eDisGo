# Engine-Handling Änderungen

## Zusammenfassung

Das Engine-Handling in eDisGo wurde grundlegend umstrukturiert, um eine einheitlichere und logischere Nutzung zu ermöglichen.

## Hauptänderungen

### 1. Zentrales Engine-Objekt in EDisGo-Klasse

Die `EDisGo`-Klasse verfügt nun über ein zentrales `engine`-Attribut, das standardmäßig die TOEP (The Open Energy Platform) Engine enthält:

```python
# Neu in edisgo/edisgo.py
class EDisGo:
    def __init__(self, **kwargs):
        # Engine wird beim Initialisieren gesetzt (TOEP by default)
        self._engine = kwargs.get("engine", None)
        if self._engine is None:
            self._engine = egon_engine()
        # ...

    @property
    def engine(self):
        """Database engine für OEDB-Zugriff."""
        return self._engine

    @engine.setter
    def engine(self, engine):
        self._engine = engine
```

### 2. Entfernte Engine-Parameter

Die folgenden Methoden benötigen **keinen** `engine` Parameter mehr:

**edisgo/edisgo.py:**
- `set_time_series_active_power_predefined()`
- `import_generators()`
- `import_electromobility()`
- `import_heat_pumps()`
- `import_dsm()`
- `import_home_batteries()`

**edisgo/network/heat.py:**
- `HeatPump.set_cop()`
- `HeatPump.set_heat_demand()`

**edisgo/network/timeseries.py:**
- `TimeSeries.predefined_fluctuating_generators_by_technology()`

### 3. Verwendung von self.engine

Alle oben genannten Methoden verwenden jetzt intern `self.engine` bzw. `edisgo_object.engine`:

```python
# Vorher
def import_generators(self, generator_scenario, **kwargs):
    engine = kwargs["engine"] if "engine" in kwargs else egon_engine()
    generators_import.oedb(
        edisgo_object=self,
        engine=engine,
        scenario=generator_scenario,
    )

# Nachher
def import_generators(self, generator_scenario, **kwargs):
    generators_import.oedb(
        edisgo_object=self,
        engine=self.engine,
        scenario=generator_scenario,
    )
```

### 4. IO-Module bleiben unverändert

Die Funktionen in `edisgo/io/*.py` behalten ihre `engine` Parameter, da sie Utility-Funktionen sind, die von verschiedenen Stellen aufgerufen werden können. Sie werden nun mit `edisgo_object.engine` von den aufrufenden Funktionen versorgt.

## Verwendung

### Standard-Verwendung (TOEP)

```python
from edisgo import EDisGo

# Engine wird automatisch als TOEP initialisiert
edisgo = EDisGo(ding0_grid="path/to/grid")

# Alle Funktionen verwenden automatisch self.engine
edisgo.import_generators(generator_scenario="eGon2035")
edisgo.import_heat_pumps(scenario="eGon2035")
edisgo.import_electromobility(data_source="oedb", scenario="eGon2035")
edisgo.import_dsm(scenario="eGon2035")
edisgo.import_home_batteries(scenario="eGon2035")

# Auf Engine zugreifen
print(edisgo.engine)
```

### Verwendung mit Custom Engine

```python
from edisgo import EDisGo
from sqlalchemy import create_engine

# Option 1: Engine beim Initialisieren übergeben
custom_engine = create_engine("postgresql://user:password@host:port/database")
edisgo = EDisGo(ding0_grid="path/to/grid", engine=custom_engine)

# Option 2: Engine nachträglich setzen
edisgo = EDisGo(ding0_grid="path/to/grid")
edisgo.engine = create_engine("postgresql://user:password@host:port/database")

# Alle Funktionen verwenden die custom engine
edisgo.import_generators(generator_scenario="eGon2035")
```

## Vorteile

✅ **Einheitlich**: Engine wird nur einmal beim Erstellen des EDisGo-Objekts definiert  
✅ **Logischer**: Kein wiederholtes Übergeben der Engine in jeder Funktion  
✅ **Weniger fehleranfällig**: Keine Gefahr, verschiedene Engines in verschiedenen Funktionsaufrufen zu verwenden  
✅ **Abwärtskompatibel**: Beim Initialisieren kann weiterhin eine custom Engine übergeben werden  
✅ **Standard TOEP**: Die TOEP-Engine wird standardmäßig verwendet  

## Breaking Changes

⚠️ **WICHTIG**: Die folgenden Funktionen akzeptieren **keinen** `engine` Parameter mehr:

```python
# FALSCH - funktioniert nicht mehr:
edisgo.import_generators(generator_scenario="eGon2035", engine=my_engine)

# RICHTIG - engine wird von edisgo.engine genommen:
edisgo.engine = my_engine  # optional, falls nicht TOEP
edisgo.import_generators(generator_scenario="eGon2035")
```

## Migration Guide

### Alt (vor den Änderungen):

```python
from edisgo import EDisGo
from edisgo.io.db import engine as egon_engine

# Engine muss überall übergeben werden
my_engine = egon_engine()
edisgo = EDisGo(ding0_grid="path/to/grid")

edisgo.set_time_series_active_power_predefined(
    fluctuating_generators_ts="oedb",
    engine=my_engine
)
edisgo.import_generators(generator_scenario="eGon2035", engine=my_engine)
edisgo.import_heat_pumps(scenario="eGon2035", engine=my_engine)
```

### Neu (nach den Änderungen):

```python
from edisgo import EDisGo

# Engine wird automatisch initialisiert (TOEP)
edisgo = EDisGo(ding0_grid="path/to/grid")

# Kein engine Parameter mehr nötig
edisgo.set_time_series_active_power_predefined(
    fluctuating_generators_ts="oedb"
)
edisgo.import_generators(generator_scenario="eGon2035")
edisgo.import_heat_pumps(scenario="eGon2035")

# Optional: Custom Engine verwenden
# edisgo.engine = my_custom_engine
```

## Geänderte Dateien

- `edisgo/edisgo.py`
- `edisgo/network/heat.py`
- `edisgo/network/timeseries.py`
- `doc/usage_details.rst`
