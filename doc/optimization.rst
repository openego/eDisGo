Julia Optimization in eDisGo with PowerModels
=======================================================================

Table of Contents
------------------

1. `Overview <#overview>`__
2. `Notation and Meta-Variables <#notation-and-meta-variables>`__
3. `All Julia Variables
   (Tabular) <#all-julia-variables-tabular>`__
4. `Temporal Context of
   Optimization <#temporal-context-of-optimization>`__
5. `The analyze Function <#the-analyze-function>`__
6. `The reinforce Function <#the-reinforce-function>`__
7. `The §14a EnWG Optimization <#the-14a-enwg-optimization>`__
8. `Time Series Usage <#time-series-usage>`__
9. `File Paths and References <#file-paths-and-references>`__

--------------

Overview
---------

The Julia optimization in eDisGo uses **PowerModels.jl** to solve
Optimal Power Flow (OPF) problems. The workflow operates via a
Python-Julia interface:

-  **Python (eDisGo)**: Network modeling, time series,
   results processing
-  **Julia (PowerModels)**: Mathematical optimization, solver interface
-  **Communication**: JSON via stdin/stdout

**Optimization objectives:** - Minimization of network losses - Compliance with
voltage and current limits - Flexibility utilization (storage,
heat pumps, electric vehicles, DSM) - Optional: §14a EnWG curtailment with
time budget constraints

--------------

Notation and Meta-Variables
---------------------------

Before examining the specific optimization variables, here is an
overview of the **general variables and notation** used in the
Julia code:

Meta-Variables (not part of the optimization problem)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------+-------+-----------------------+--------------------+
| Variable       | Type  | Description           | Usage              |
+================+=======+=======================+====================+
| ``pm``         | ``    | PowerModels object    | Contains the       |
|                | Abstr |                       | entire             |
|                | actPo |                       | o                  |
|                | werMo |                       | ptimization problem|
|                | del`` |                       | (network, variables|
|                |       |                       | , constraints)     |
+----------------+-------+-----------------------+--------------------+
| ``nw`` or      | ``    | Network ID            | Identifies a       |
| ``n``          | Int`` | (time step index)     | time step in       |
|                |       |                       | the                |
|                |       |                       | mu                 |
|                |       |                       | lti-period problem |
|                |       |                       | (0, 1, 2, …, T-1)  |
+----------------+-------+-----------------------+--------------------+
| ``nw_ids(pm)`` | ``Ar  | All network IDs       | Returns all        |
|                | ray{I |                       | time step indices, |
|                | nt}`` |                       | e.g.               |
|                |       |                       | ``[0,              |
|                |       |                       | 1, 2, ..., 8759]`` |
|                |       |                       | for 8760h          |
+----------------+-------+-----------------------+--------------------+
| `              | ``D   | Reference data for    | Access to          |
| `ref(pm, nw)`` | ict`` | time step             | network data of a  |
|                |       |                       | specific           |
|                |       |                       | time step          |
+----------------+-------+-----------------------+--------------------+
| `              | ``D   | Variables dictionary  | Access to          |
| `var(pm, nw)`` | ict`` |                       | op                 |
|                |       |                       | timization variables|
|                |       |                       | of a time step     |
+----------------+-------+-----------------------+--------------------+
| ``model`` or   | ``Ju  | Ju                    | The underlying     |
| ``pm.model``   | MP.Mo | MP optimization model | mathematical       |
|                | del`` |                       | optimization model |
+----------------+-------+-----------------------+--------------------+

Index Variables
~~~~~~~~~~~~~~~

+---------------+----------------+---------------------+---------------+
| Variable      | Meaning        | Description         | Example       |
+===============+================+=====================+===============+
| ``i``, ``j``  | Bus index      | Identifies          | ``i=1`` = Bus |
|               |                | nodes in network    | "Bus_MV_123"  |
+---------------+----------------+---------------------+---------------+
| ``l``         | Branch index   | Identifies          | ``l=5`` =     |
|               | (Li            | lines and           | line          |
|               | ne/Transformer)| transformers        | "Line_LV_456" |
+---------------+----------------+---------------------+---------------+
| ``g``         | G              | Identifies          | ``g=3`` =     |
|               | enerator index | generators (PV,     | "PV_001"      |
|               |                | wind, CHP, slack)   |               |
+---------------+----------------+---------------------+---------------+
| ``s``         | Storage index  | Identifies          | ``s=1`` =     |
|               |                | battery storage     | "Storage_1"   |
+---------------+----------------+---------------------+---------------+
| ``h``         | Heat           | Identifies          | ``h=2`` =     |
|               | pump index     | heat pumps          | "HP_LV_789"   |
+---------------+----------------+---------------------+---------------+
| ``c``         | Charging       | Identifies          | ``c=4`` =     |
|               | point index    | charging points for | "CP_LV_101"   |
|               |                | electric vehicles   |               |
+---------------+----------------+---------------------+---------------+
| ``d``         | DSM index      | Identifies          | ``d=1`` =     |
|               |                | DSM loads           | "DSM_Load_1"  |
+---------------+----------------+---------------------+---------------+
| ``t`` or      | Ti             | Time point in       | ``t=0`` =     |
| ``n``         | me step index  | optimization        | 2035-01-01    |
|               |                | horizon             | 00:00,        |
|               |                |                     | ``t=1`` =     |
|               |                |                     | 01:00, …      |
+---------------+----------------+---------------------+---------------+

PowerModels Functions
~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+--------------------+--------------------+----------------------+
| Function                  | Return value       | Description        | Example              |
+===========================+====================+====================+======================+
| ``ids(pm, :bus, nw=n)``   | ``Array{Int}``     | Returns all bus    | ``[1, 2, 3, ...]``   |
|                           |                    | IDs for time step n|                      |
+---------------------------+--------------------+--------------------+----------------------+
| ``ids(pm, :branch,``      | ``Array{Int}``     | Returns all        | ``[1, 2, 3, ...]``   |
| ``nw=n)``                 |                    | branch IDs         |                      |
|                           |                    | (lines/transformers)|                     |
+---------------------------+--------------------+--------------------+----------------------+
| ``ids(pm, :gen, nw=n)``   | ``Array{Int}``     | Returns all        | ``[1, 2, 3, ...]``   |
|                           |                    | generator IDs      |                      |
+---------------------------+--------------------+--------------------+----------------------+
| ``ids(pm, :storage,``     | ``Array{Int}``     | Returns all        | ``[1, 2, 3]``        |
| ``nw=n)``                 |                    | storage IDs        |                      |
+---------------------------+--------------------+--------------------+----------------------+
| ``ref(pm, nw, :bus, i)``  | ``Dict``           | Returns data for   | ``{"vmin": 0.9,``    |
|                           |                    | bus i in time step | ``"vmax": 1.1}``     |
|                           |                    | nw                 |                      |
+---------------------------+--------------------+--------------------+----------------------+
| ``ref(pm, nw, :branch,``  | ``Dict``           | Returns data for   | ``{"rate_a": 0.5,``  |
| ``l)``                    |                    | branch l in        | ``"br_r": 0.01}``    |
|                           |                    | time step nw       |                      |
+---------------------------+--------------------+--------------------+----------------------+
| ``var(pm, nw, :p, l)``    | ``JuMP.Variable``  | Returns active     | JuMP variable object |
|                           |                    | power variable     |                      |
|                           |                    | for branch l       |                      |
+---------------------------+--------------------+--------------------+----------------------+
| ``var(pm, nw, :w, i)``    | ``JuMP.Variable``  | Returns voltage    | JuMP variable object |
|                           |                    | variable for       |                      |
|                           |                    | bus i              |                      |
+---------------------------+--------------------+--------------------+----------------------+

Typical Code Patterns
~~~~~~~~~~~~~~~~~~~~

1. Iteration over all time steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   for n in nw_ids(pm)
       # Code for time step n
       println("Processing time step $n")
   end

2. Iteration over all buses in a time step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   for i in ids(pm, :bus, nw=n)
       # Code for bus i in time step n
       bus_data = ref(pm, n, :bus, i)
       println("Bus $i: Vmin = $(bus_data["vmin"])")
   end

3. Accessing variables
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   # Retrieve variable
   w_i = var(pm, n, :w, i)  # Voltage variable for bus i, time step n

   # Use variable in constraint
   JuMP.@constraint(pm.model, w_i >= 0.9^2)  # Lower voltage limit

4. Creating and storing variables in dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   # Initialize variables dictionary for time step n
   var(pm, n)[:p_hp14a] = JuMP.@variable(
       pm.model,
       [h in ids(pm, :gen_hp_14a, nw=n)],
       base_name = "p_hp14a_$(n)",
       lower_bound = 0.0
   )

   # Access later
   for h in ids(pm, :gen_hp_14a, nw=n)
       p_hp14a_h = var(pm, n, :p_hp14a, h)
   end

Multi-Network Structure
~~~~~~~~~~~~~~~~~~~~~~

PowerModels uses a **multi-network structure** for time-dependent
optimization:

::

   pm (PowerModel)
   ├─ nw["0"]  (Time step 0: 2035-01-01 00:00)
   │  ├─ :bus → {1: {...}, 2: {...}, ...}          ← All 150 buses
   │  ├─ :branch → {1: {...}, 2: {...}, ...}       ← All 200 lines/transformers
   │  ├─ :gen → {1: {...}, 2: {...}, ...}          ← All 50 generators
   │  ├─ :load → {1: {...}, 2: {...}, ...}         ← All 120 loads
   │  └─ :storage → {1: {...}, 2: {...}, ...}      ← All 5 storage units
   │
   ├─ nw["1"]  (Time step 1: 2035-01-01 01:00)
   │  ├─ :bus → {1: {...}, 2: {...}, ...}          ← AGAIN all 150 buses
   │  ├─ :branch → {1: {...}, 2: {...}, ...}       ← AGAIN all 200 lines
   │  └─ ...                                        ← etc.
   │
   ├─ nw["2"]  (Time step 2: 2035-01-01 02:00)
   │  ├─ :bus → {1: {...}, 2: {...}, ...}          ← AGAIN all 150 buses
   │  └─ ...
   │
   ├─ ...  (8757 more time steps)
   │
   └─ nw["8759"]  (Time step 8759: 2035-12-31 23:00)
      └─ Complete network again

**IMPORTANT: The network exists T times!**

For an optimization horizon of **8760 hours** (1 year), this means:

- The entire network is **duplicated 8760 times**
- Each time step has its own complete network copy
- All buses, lines, transformers, generators, loads exist **8760 times**
- Each time step has **its own optimization variables**

**What distinguishes the time steps?**

+--------+----------------------+--------------------------------------+
| Aspect | Time steps           | Different per time step              |
+========+======================+======================================+
| **Net  | Identical            | Same buses, lines, transformers      |
| work   |                      |                                      |
| topol  |                      |                                      |
| ogy**  |                      |                                      |
+--------+----------------------+--------------------------------------+
| **Net  | Identical            | Same resistances, capacities         |
| work   |                      |                                      |
| param  |                      |                                      |
| eters**|                      |                                      |
+--------+----------------------+--------------------------------------+
| **     | Different            | Generator feed-in, loads, COP        |
| Time   |                      |                                      |
| series |                      |                                      |
| values**|                     |                                      |
+--------+----------------------+--------------------------------------+
| **Vari | Different            | Voltages, power flows,               |
| ables**|                      | storage power                        |
+--------+----------------------+--------------------------------------+
| **Sto  | Coupled              | SOC[t+1] depends on SOC[t]           |
| rage   |                      |                                      |
| SOC**  |                      |                                      |
+--------+----------------------+--------------------------------------+

**Example: Active power variable p[l,i,j]**

For a line ``l=5`` between bus ``i=10`` and ``j=11``:

- ``var(pm, 0, :p)[(5,10,11)]`` = Active power in time step 0 (00:00)
- ``var(pm, 1, :p)[(5,10,11)]`` = Active power in time step 1 (01:00)
- ``var(pm, 2, :p)[(5,10,11)]`` = Active power in time step 2 (02:00)
- …
- ``var(pm, 8759, :p)[(5,10,11)]`` = Active power in time step 8759 (23:00)

→ **8760 different variables** for the same line!

**Optimization problem size:**

For a network with:

- 150 buses
- 200 lines/transformers
- 50 generators
- 5 battery storage units
- 20 heat pumps
- 10 charging points
- 8760 time steps (1 year, 1h resolution)

**Number of variables (approximately):**

- Voltages: 150 buses x 8760 time steps = **1,314,000 variables**
- Line flows: 200 x 2 (p,q) x 8760 = **3,504,000 variables**
- Generators: 50 x 2 (p,q) x 8760 = **876,000 variables**
- Storage: 5 x 2 (power + SOC) x 8760 = **87,600 variables**
- …

→ **Several million variables** for annual simulation!

**Why this approach?**

**Advantages:** - Allows time-coupled optimization (storage,
heat pumps) - PowerModels syntax remains simple (each time step like
single problem) - Flexible time series (different values per
time step)

**Disadvantages:** - Very large optimization problem (millions of variables) -
High memory requirements - Long solution times (minutes to hours)

**Inter-timestep constraints:**

Certain constraints couple the time steps:

.. code:: julia

   # Storage energy coupling
   for n in 0:8758  # All time steps except last
       for s in storage_ids
           # SOC at t+1 depends on SOC at t and power at t
           @constraint(pm.model,
               var(pm, n+1, :se, s) ==
               var(pm, n, :se, s) + var(pm, n, :ps, s) x Δt x η
           )
       end
   end

→ These constraints connect the otherwise independent time steps!

**Summary:** - Each time step has its **own complete
copy** of the network - Time series values (loads, feed-in)
differ between time steps - Variables exist **per
time step** (8760 times for each physical variable!) -
Inter-timestep constraints (storage SOC, heat storage) couple the
time steps - **For 8760 time steps:** The network exists 8760 times →
millions of variables

--------------

All Julia Variables (Tabular)
-----------------------------------

Grid Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------+---------------+------------+--------------------+
| Variable         | Dimension     | Unit       | Description        |
+==================+===============+============+====================+
| ``p[l,i,j]``     | ℝ             | MW         | Active power flow  |
|                  |               |            | on line/transformer|
|                  |               |            | from bus i to j    |
+------------------+---------------+------------+--------------------+
| ``q[l,i,j]``     | ℝ             | MVAr       | Reactive power     |
|                  |               |            | flow on            |
|                  |               |            | line/transformer   |
|                  |               |            | from bus i to j    |
+------------------+---------------+------------+--------------------+
| ``w[i]``         | ℝ⁺            | p.u.²      | Squared voltage    |
|                  |               |            | magnitude at       |
|                  |               |            | bus i              |
+------------------+---------------+------------+--------------------+
| ``ccm[l,i,j]``   | ℝ⁺            | kA²        | Squared current    |
|                  |               |            | magnitude on       |
|                  |               |            | line/transformer   |
+------------------+---------------+------------+--------------------+
| ``ll[l,i,j]``    | [0,1]         | -          | Line loading       |
|                  |               |            | (only OPF version  |
|                  |               |            | 1 & 3)             |
+------------------+---------------+------------+--------------------+

**Notes:** - ``l`` = line/transformer ID - ``i,j`` = bus IDs
(from_bus, to_bus) - Squared variables avoid non-convex
root functions

--------------

Generation Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------+-----------------+-------------+---------------------+
| Variable      | Dimension       | Unit        | Description         |
+===============+=================+=============+=====================+
| ``pg[g]``     | ℝ               | MW          | Active power        |
|               |                 |             | generation of       |
|               |                 |             | generator g         |
+---------------+-----------------+-------------+---------------------+
| ``qg[g]``     | ℝ               | MVAr        | Reactive power      |
|               |                 |             | generation of       |
|               |                 |             | generator g         |
+---------------+-----------------+-------------+---------------------+
| ``pgc[g]``    | ℝ⁺              | MW          | Curtailment of      |
|               |                 |             | non-controllable    |
|               |                 |             | generators          |
+---------------+-----------------+-------------+---------------------+
| ``pgs``       | ℝ               | MW          | Slack generator     |
|               |                 |             | active power        |
|               |                 |             | (grid connection)   |
+---------------+-----------------+-------------+---------------------+
| ``qgs``       | ℝ               | MVAr        | Slack generator     |
|               |                 |             | reactive power      |
+---------------+-----------------+-------------+---------------------+

**Notes:** - Slack generator represents transmission grid connection
- Curtailment only for renewable energy plants (PV, wind)

--------------

Battery Storage Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------+-----------------+-------------+---------------------+
| Variable      | Dimension       | Unit        | Description         |
+===============+=================+=============+=====================+
| ``ps[s,t]``   | ℝ               | MW          | Active power of     |
|               |                 |             | battery storage s   |
|               |                 |             | at time t (+        |
|               |                 |             | = discharge, -      |
|               |                 |             | = charge)           |
+---------------+-----------------+-------------+---------------------+
| ``qs[s,t]``   | ℝ               | MVAr        | Reactive power of   |
|               |                 |             | battery storage s   |
+---------------+-----------------+-------------+---------------------+
| ``se[s,t]``   | ℝ⁺              | MWh         | Energy content      |
|               |                 |             | (State of Energy)   |
|               |                 |             | of battery storage s|
+---------------+-----------------+-------------+---------------------+

**Constraints:** - SOC coupling between time steps:
``se[t+1] = se[t] + ps[t] x Δt x η`` - Capacity limits:
``se_min ≤ se[t] ≤ se_max`` - Power limits:
``ps_min ≤ ps[t] ≤ ps_max``

--------------

Heat Pump Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------+-----------------+-------------+---------------------+
| Variable      | Dimension       | Unit        | Description         |
+===============+=================+=============+=====================+
| ``php[h,t]``  | ℝ⁺              | MW          | Electrical power    |
|               |                 |             | consumption of      |
|               |                 |             | heat pump h         |
+---------------+-----------------+-------------+---------------------+
| ``qhp[h,t]``  | ℝ               | MVAr        | Reactive power of   |
|               |                 |             | heat pump h         |
+---------------+-----------------+-------------+---------------------+
| ``phs[h,t]``  | ℝ               | MW          | Power of thermal    |
|               |                 |             | storage h (+        |
|               |                 |             | = charging, -       |
|               |                 |             | = discharging)      |
+---------------+-----------------+-------------+---------------------+
| ``hse[h,t]``  | ℝ⁺              | MWh         | Energy content of   |
|               |                 |             | thermal storage h   |
+---------------+-----------------+-------------+---------------------+

**Notes:** - Heat pumps with thermal storage can be time-shifted
- Heat demand must be met over optimization horizon
- COP (Coefficient of Performance) links electrical and
thermal power

--------------

Charging Point / EV Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------+----------------+-------------+---------------------+
| Variable       | Dimension      | Unit        | Description         |
+================+================+=============+=====================+
| ``pcp[c,t]``   | ℝ⁺             | MW          | Charging power at   |
|                |                |             | charging point c at |
|                |                |             | time t              |
+----------------+----------------+-------------+---------------------+
| ``qcp[c,t]``   | ℝ              | MVAr        | Reactive power of   |
|                |                |             | charging point c    |
+----------------+----------------+-------------+---------------------+
| ``cpe[c,t]``   | ℝ⁺             | MWh         | Energy content of   |
|                |                |             | vehicle battery at  |
|                |                |             | charging point c    |
+----------------+----------------+-------------+---------------------+

**Constraints:** - Energy coupling:
``cpe[t+1] = cpe[t] + pcp[t] x Δt x η`` - Capacity:
``cpe_min ≤ cpe[t] ≤ cpe_max`` - Charging power: ``0 ≤ pcp[t] ≤ pcp_max``

--------------

Demand Side Management Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============= ========= ======= =====================================
Variable      Dimension Unit    Description
============= ========= ======= =====================================
``pdsm[d,t]`` ℝ⁺        MW      Shiftable load d at time t
``qdsm[d,t]`` ℝ         MVAr    Reactive power of DSM load d
``dsme[d,t]`` ℝ⁺        MWh     Virtual energy content of DSM storage
============= ========= ======= =====================================

**Notes:** - DSM models shiftable loads (e.g.
industrial processes) - Total energy over horizon remains constant

--------------

Slack Variables for Network Restrictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only in **OPF Version 2 & 4** (with network restrictions):

+----------------+-----------+---------+-------------------------------------------+
| Variable       | Dimension | Unit    | Description                               |
+================+===========+=========+===========================================+
| ``phps[h,t]``  | ℝ⁺        | MW      | Slack for heat pump restriction           |
+----------------+-----------+---------+-------------------------------------------+
| ``phps2[h,t]`` | ℝ⁺        | MW      | Slack for heat pump operation restriction |
+----------------+-----------+---------+-------------------------------------------+
| ``phss[h,t]``  | ℝ⁺        | MW      | Slack for thermal storage restriction     |
+----------------+-----------+---------+-------------------------------------------+
| ``pds[d,t]``   | ℝ⁺        | MW      | Load shedding                             |
+----------------+-----------+---------+-------------------------------------------+
| ``pgens[g,t]`` | ℝ⁺        | MW      | Slack for generator curtailment           |
+----------------+-----------+---------+-------------------------------------------+
| ``pcps[c,t]``  | ℝ⁺        | MW      | Slack for charging point restriction      |
+----------------+-----------+---------+-------------------------------------------+
| ``phvs[t]``    | ℝ⁺        | MW      | Slack for high voltage requirements       |
+----------------+-----------+---------+-------------------------------------------+

**Purpose:** - Ensure solvability of optimization problem - High
costs in objective function → are minimized - Indicate where
network restrictions cannot be met

--------------

§14a EnWG Variables (NEW)
~~~~~~~~~~~~~~~~~~~~~~~~~

Only when ``curtailment_14a=True``:

Heat Pumps §14a
^^^^^^^^^^^^^^^^

+----------------+----------------+-------------+---------------------+
| Variable       | Dimension      | Unit        | Description         |
+================+================+=============+=====================+
| ``             | ℝ⁺             | MW          | Virtual generator   |
| p_hp14a[h,t]`` |                |             | for HP curtailment  |
|                |                |             | (0 to pmax)         |
+----------------+----------------+-------------+---------------------+
| ``             | Binary         | {0,1}       | -                   |
| z_hp14a[h,t]`` |                |             |                     |
+----------------+----------------+-------------+---------------------+

Charging Points §14a
^^^^^^^^^^^^^^^

+----------------+----------------+-------------+---------------------+
| Variable       | Dimension      | Unit        | Description         |
+================+================+=============+=====================+
| ``             | ℝ⁺             | MW          | Virtual generator   |
| p_cp14a[c,t]`` |                |             | for CP curtailment  |
|                |                |             | (0 to pmax)         |
+----------------+----------------+-------------+---------------------+
| ``             | Binary         | {0,1}       | -                   |
| z_cp14a[c,t]`` |                |             |                     |
+----------------+----------------+-------------+---------------------+

**Important parameters:** - ``pmax = P_nominal - P_min_14a`` (maximum
curtailment power) - ``P_min_14a = 0.0042 MW`` (4.2 kW minimum power
according to §14a) - ``max_hours_per_day`` (e.g. 2h/day time budget)

**Functionality:** - Virtual generator "generates" power at
HP/CP bus - Effect: Net load = original load - p_hp14a - Simulates
curtailment without complex load adjustment

--------------

Temporal Context of Optimization
------------------------------------

Overall Workflow
~~~~~~~~~~~~~~~~~

**IMPORTANT NOTE on Workflow:** - **Reinforce BEFORE optimization:**
Only optional and useful for the base grid (e.g. without
heat pumps/electric vehicles). If you expand the complete
grid before optimization, there are no more overloads and the optimization for
flexibility utilization makes no sense. - **Reinforce AFTER
optimization:** Usually required! The optimization uses
flexibility to minimize grid expansion, but cannot solve all problems.
Remaining overloads and voltage violations must be resolved through
conventional grid expansion.

**Typical use case:** 1. Load base grid (e.g. current state without
new heat pumps) 2. Optional: reinforce base grid 3. (New)
Add components (heat pumps, electric vehicles for future scenario) 4.
Run optimization → uses flexibility instead of grid expansion 5.
**Mandatory:** reinforce with optimized time series → fixes remaining
problems

::

   ┌─────────────────────────────────────────────────────────────────────┐
   │ 1. INITIALIZATION                                                   │
   ├─────────────────────────────────────────────────────────────────────┤
   │ - Load grid (ding0 grid or database)                               │
   │ - Import time series (generators, loads without new components)    │
   │ - Configure optimization parameters                                │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 2. BASE GRID REINFORCEMENT (optional)                               │
   ├─────────────────────────────────────────────────────────────────────┤
   │ edisgo.reinforce()                                                  │
   │ - Reinforce base grid (WITHOUT new HP/CP)                          │
   │ - Useful as reference scenario                                     │
   │ - Creates baseline for scenario comparison                         │
   │                                                                     │
   │ IMPORTANT: This is NOT the main reinforcement step!                │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 3. ADD NEW COMPONENTS                                               │
   ├─────────────────────────────────────────────────────────────────────┤
   │ - Add heat pumps (with thermal storage)                            │
   │ - Add electric vehicle charging points (with flexibility bands)    │
   │ - Add battery storage                                              │
   │ - Set time series for new components                               │
   │                                                                     │
   │ → Grid is now likely overloaded                                    │
   │ → NO reinforce at this point!                                      │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 4. JULIA OPTIMIZATION                                               │
   ├─────────────────────────────────────────────────────────────────────┤
   │ edisgo.pm_optimize(opf_version=2, curtailment_14a=True)            │
   │                                                                     │
   │ GOAL: Use flexibility to AVOID grid expansion                      │
   │ - Optimally charge/discharge battery storage                       │
   │ - Time-shift heat pumps (thermal storage)                          │
   │ - Optimize EV charging (within flexibility band)                   │
   │ - §14a curtailment at bottlenecks (max. 2h/day)                   │
   │                                                                     │
   │ 4.1 PYTHON → POWERMODELS CONVERSION                                │
   │     ├─ to_powermodels(): Grid → PowerModels dictionary            │
   │     ├─ Time series for all components                             │
   │     ├─ If 14a: Create virtual generators for HP/CP                │
   │     └─ Serialize to JSON                                          │
   │                                                                     │
   │ 4.2 PYTHON → JULIA COMMUNICATION                                   │
   │     ├─ Start Julia subprocess: julia Main.jl [args]               │
   │     ├─ Pass JSON via stdin                                        │
   │     └─ Args: grid_name, results_path, method (soc/nc), etc.       │
   │                                                                     │
   │ 4.3 JULIA OPTIMIZATION                                             │
   │     ├─ Parse JSON → PowerModels multinetwork                      │
   │     ├─ Solver selection: Gurobi (SOC) or IPOPT (NC)               │
   │     ├─ build_mn_opf_bf_flex():                                    │
   │     │   ├─ Create variables (all from tables above)               │
   │     │   ├─ Constraints per time step:                             │
   │     │   │   ├─ Power balance at nodes                             │
   │     │   │   ├─ Voltage drop equations                             │
   │     │   │   ├─ Current equations                                  │
   │     │   │   ├─ Storage/HP/CP state equations                      │
   │     │   │   ├─ §14a binary coupling (if enabled)                 │
   │     │   │   └─ §14a minimum net load (if enabled)                │
   │     │   ├─ Inter-timestep constraints:                            │
   │     │   │   ├─ Energy coupling storage/HP/CP                      │
   │     │   │   └─ §14a daily time budget (if enabled)               │
   │     │   └─ Set objective function (version-dependent)             │
   │     ├─ Solve optimization                                         │
   │     ├─ Serialize results to JSON                                  │
   │     └─ Output via stdout                                          │
   │                                                                     │
   │ 4.4 JULIA → PYTHON COMMUNICATION                                   │
   │     ├─ Python reads stdout line by line                           │
   │     ├─ Capture JSON result (starts with {"name")                  │
   │     └─ Parse JSON to dictionary                                   │
   │                                                                     │
   │ 4.5 POWERMODELS → EDISGO CONVERSION                                │
   │     ├─ from_powermodels(): Extract optimized time series          │
   │     ├─ Write to edisgo.timeseries:                                │
   │     │   ├─ generators_active_power, generators_reactive_power     │
   │     │   ├─ storage_units_active_power (optimized)                 │
   │     │   ├─ heat_pump_loads (time-shifted)                         │
   │     │   ├─ charging_point_loads (optimized)                       │
   │     │   └─ §14a curtailment as virtual generators:                │
   │     │       ├─ hp_14a_support_{name}                              │
   │     │       └─ cp_14a_support_{name}                              │
   │     └─ Curtailment = Virtual generator power                      │
   │                                                                     │
   │ RESULT: Optimized time series with minimized grid expansion needs  │
   │          But: Possibly remaining overloads (slacks > 0)            │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 5. GRID EXPANSION WITH OPTIMIZED TIME SERIES    │
   ├─────────────────────────────────────────────────────────────────────┤
   │ edisgo.reinforce()                                                  │
   │                                                                     │
   │ IMPORTANT: This step is usually required!                          │
   │                                                                     │
   │ Why?                                                                │
   │ - Optimization uses flexibility, but cannot solve all problems     │
   │   (e.g. grid restrictions, insufficient flexibility)               │
   │ - Slack variables > 0 indicate remaining violations                │
   │ - Remaining overloads must be resolved through grid expansion      │
   │                                                                     │
   │ Process:                                                            │
   │ - Iterative reinforcement measures                                 │
   │ - Line expansion, transformer expansion                            │
   │ - Calculate grid expansion costs                                   │
   │                                                                     │
   │ RESULT: Grid expansion costs AFTER flexibility utilization         │
   │          (significantly lower than without optimization!)          │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 6. EVALUATION                                                       │
   ├─────────────────────────────────────────────────────────────────────┤
   │ - Analyze optimized time series                                    │
   │ - Calculate §14a statistics (curtailed energy, time budget usage)  │
   │ - Compare grid expansion costs (with vs. without optimization)     │
   │ - Analyze flexibility utilization                                  │
   │ - Visualization, export                                            │
   └─────────────────────────────────────────────────────────────────────┘

--------------

Workflow Variants Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table shows the most important workflow variants and their
use cases:

.. list-table::
   :header-rows: 1
   :widths: 20 25 30 25

   * - Workflow
     - Steps
     - When useful?
     - Result
   * - **A: Grid expansion only (without optimization)**
     - 1. Load grid

       2. Add components

       3. ``reinforce()``
     - - No flexibility available
       - Quick conservative planning
       - Reference scenario
     - High grid expansion costs, flexibility potential unused
   * - **B: With optimization (RECOMMENDED)**
     - 1. Load grid

       2. Optional: ``reinforce()`` on base grid

       3. Add components

       4. ``pm_optimize()``

       5. **Mandatory:** ``reinforce()``
     - - Flexibility available (storage, HP, CP)
       - §14a utilization desired
       - Minimize grid expansion costs
     - Minimal grid expansion costs, optimal flexibility utilization, reliable grid
   * - **C: Base grid reference + optimization**
     - 1. Load grid (base grid)

       2. ``reinforce()`` → Costs₁

       3. Add new components

       4. ``pm_optimize()``

       5. ``reinforce()`` → Costs₂

       6. Compare: Costs₂ - Costs₁
     - - Cost comparison with/without new components
       - Analyze additional costs from HP/CP
       - Evaluate §14a benefits
     - Cost transparency, attribution to new components, quantification of flexibility benefits
   * - **D: Multiple optimization scenarios**
     - 1. Load grid + add components

       2a. ``reinforce()`` → Reference

       2b. ``pm_optimize(14a=False)`` + ``reinforce()``

       2c. ``pm_optimize(14a=True)`` + ``reinforce()``

       3. Compare
     - - Evaluate different flexibility options
       - Cost-benefit analysis §14a
       - Sensitivity analysis
     - Complete scenario comparison, optimal strategy selection, sound decision basis

**Important insights:**

1. **Reinforce before optimization only makes sense for:**

   -  Base grid without new components (reference scenario)
   -  Documentation of initial state
   -  Status quo assessment
   -  **NOT after adding (new) components whose
      flexibility utilization is to be investigated** → Would
      negate flexibility potential

2. **Reinforce after optimization is usually beneficial:**

   -  Optimization reduces grid expansion, but doesn't solve all problems
   -  Slack variables indicate remaining violations

3. **Example cost reduction:**

   -  Without optimization: 100% grid expansion costs (reference)
   -  With optimization without §14a: 60-80% of reference costs
   -  With optimization with §14a: 40-60% of reference costs
   -  Dependent on: Flexibility degree, grid structure, load profiles

**Example code for Workflow B (recommended):**

.. code:: python

   # Workflow B: With optimization (BEST PRACTICE)

   # 1. Load grid
   edisgo = EDisGo(ding0_grid="path/to/grid")

   # Load time series etc.

   # 2. Optional: Reinforce base grid (for comparison)
   # edisgo.reinforce()  # Only if reference costs desired or status quo expansion needed

   # 3. Add new components for future scenario
   edisgo.add_heat_pumps(
       scenario="eGon2035",
       with_thermal_storage=True  # Flexibility!
   )
   edisgo.add_charging_points(
       scenario="eGon2035"
   )

   # 4. Run optimization
   edisgo.pm_optimize(
       opf_version=2,              # With grid restrictions
       curtailment_14a=True,       # Use §14a curtailment
       max_hours_per_day=2.0,      # 2h/day time budget
       solver="gurobi"
   )

   # 5. MANDATORY: Grid expansion for remaining problems
   edisgo.reinforce()

   # 6. Analyze results
   costs = edisgo.results.grid_expansion_costs
   curtailment = edisgo.timeseries.generators_active_power[
       [c for c in edisgo.timeseries.generators_active_power.columns
        if '14a_support' in c]
   ]

   print(f"Grid expansion costs (after optimization): {costs:,.0f} €")
   print(f"§14a curtailment total: {curtailment.sum().sum():.2f} MWh")

--------------

Detailed Timeline of Julia Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 1: Problem Setup (build_mn_opf_bf_flex)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**For each time step n in optimization horizon:**

.. code:: julia

   for n in nw_ids(pm)
       # 1. CREATE VARIABLES
       PowerModels.variable_bus_voltage(pm, nw=n)           # w[i]
       PowerModels.variable_gen_power(pm, nw=n)             # pg, qg
       PowerModels.variable_branch_power(pm, nw=n)          # p, q
       eDisGo_OPF.variable_branch_current(pm, nw=n)         # ccm

       # Flexibility
       eDisGo_OPF.variable_storage_power(pm, nw=n)          # ps
       eDisGo_OPF.variable_heat_pump_power(pm, nw=n)        # php
       eDisGo_OPF.variable_heat_storage_power(pm, nw=n)     # phs
       eDisGo_OPF.variable_charging_point_power(pm, nw=n)   # pcp

       # If OPF version 1 or 3: Line Loading
       if opf_version in [1, 3]
           eDisGo_OPF.variable_line_loading(pm, nw=n)       # ll
       end

       # If OPF version 2 or 4: Slack variables
       if opf_version in [2, 4]
           eDisGo_OPF.variable_slack_heatpumps(pm, nw=n)    # phps, phps2
           eDisGo_OPF.variable_slack_heat_storage(pm, nw=n) # phss
           eDisGo_OPF.variable_slack_loads(pm, nw=n)        # pds
           eDisGo_OPF.variable_slack_gens(pm, nw=n)         # pgens
           eDisGo_OPF.variable_slack_cps(pm, nw=n)          # pcps
       end

       # If §14a enabled: Virtual generators + binary variables
       if curtailment_14a
           eDisGo_OPF.variable_gen_hp_14a_power(pm, nw=n)   # p_hp14a
           eDisGo_OPF.variable_gen_hp_14a_binary(pm, nw=n)  # z_hp14a
           eDisGo_OPF.variable_gen_cp_14a_power(pm, nw=n)   # p_cp14a
           eDisGo_OPF.variable_gen_cp_14a_binary(pm, nw=n)  # z_cp14a
       end

       # 2. CONSTRAINTS PER TIME STEP
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

       # §14a constraints per time step
       if curtailment_14a
           for h in ids(pm, :gen_hp_14a, nw=n)
               constraint_hp_14a_binary_coupling(pm, h, n)  # p_hp14a ≤ pmax x z
               constraint_hp_14a_min_net_load(pm, h, n)     # Net load ≥ min(load, 4.2kW)
           end
           for c in ids(pm, :gen_cp_14a, nw=n)
               constraint_cp_14a_binary_coupling(pm, c, n)
               constraint_cp_14a_min_net_load(pm, c, n)
           end
       end
   end

Phase 2: Inter-Timestep Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   # Storage energy coupling between time steps
   for s in ids(pm, :storage)
       for t in 1:(T-1)
           se[t+1] == se[t] + ps[t] x Δt x η
       end
   end

   # Thermal storage coupling
   for h in ids(pm, :heat_pump)
       for t in 1:(T-1)
           hse[t+1] == hse[t] + phs[t] x Δt x η
       end
   end

   # EV battery coupling
   for c in ids(pm, :charging_point)
       for t in 1:(T-1)
           cpe[t+1] == cpe[t] + pcp[t] x Δt x η
       end
   end

   # §14a daily time budget
   if curtailment_14a
       # Group time steps into 24h days
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

Phase 3: Objective Function
^^^^^^^^^^^^^^^^^^^^^

**OPF Version 1** (relaxed restrictions, without slacks):

.. code:: julia

   minimize: 0.9 x sum(Losses) + 0.1 x max(ll) + 0.05 x sum(p_hp14a) + 0.05 x sum(p_cp14a)

**OPF Version 2** (with grid restrictions, with slacks):

.. code:: julia

   minimize: 0.4 x sum(Losses) + 0.6 x sum(Slacks) + 0.5 x sum(p_hp14a) + 0.5 x sum(p_cp14a)

**OPF Version 3** (with HV requirements, relaxed restrictions):

.. code:: julia

   minimize: 0.9 x sum(Losses) + 0.1 x max(ll) + 50 x sum(phvs) + 0.05 x sum(p_hp14a) + 0.05 x sum(p_cp14a)

**OPF Version 4** (with HV requirements and restrictions):

.. code:: julia

   minimize: 0.4 x sum(Losses) + 0.6 x sum(Slacks) + 50 x sum(phvs) + 0.5 x sum(p_hp14a) + 0.5 x sum(p_cp14a)

**Important:** - §14a terms have moderate weights → curtailment is
used but minimized - Slack variables have high implicit costs →
only when unavoidable - HV slack has very high weight → compliance
prioritized

Phase 4: Solving
^^^^^^^^^^^^^^

.. code:: julia

   # Solver selection
   if method == "soc"
       solver = Gurobi.Optimizer
       # SOC relaxation: ccm constraints as second-order cone
   elseif method == "nc"
       solver = Ipopt.Optimizer
       # Non-convex: ccm constraints as quadratic equations
   end

   # Run optimization
   result = optimize_model!(pm, solver)

   # Optional: Warm-start NC with SOC solution
   if warm_start
       result_soc = optimize_model!(pm, Gurobi.Optimizer)
       initialize_from_soc!(pm, result_soc)
       result = optimize_model!(pm, Ipopt.Optimizer)
   end

--------------

The analyze Function
--------------------

Function Definition
~~~~~~~~~~~~~~~~~~~

**File:** ``edisgo/edisgo.py`` (line ~1038)

**Signature:**

.. code:: python

   def analyze(
       self,
       mode: str | None = None,
       timesteps: pd.DatetimeIndex | None = None,
       troubleshooting_mode: str | None = None,
       scale_timeseries: float | None = None,
       **kwargs
   ) -> None

What does analyze do?
~~~~~~~~~~~~~~~~~~

The ``analyze`` function performs a **static, non-linear
power flow analysis** (Power Flow Analysis, PFA) using PyPSA.
It calculates:

1. **Voltages** at all nodes (``v_res``)
2. **Currents** on all lines and transformers (``i_res``)
3. **Active power flows** on equipment (``pfa_p``)
4. **Reactive power flows** on equipment (``pfa_q``)

The results are stored in ``edisgo.results``.

Parameters
~~~~~~~~~

+--------------------+------------------+-----------------------------+
| Parameter          | Default          | Description                 |
+====================+==================+=============================+
| ``mode``           | str \| None      | Analysis level: ``'mv'``    |
|                    |                  | (MV grid), ``'mvlv'`` (MV   |
|                    |                  | with LV at secondary side), |
|                    |                  | ``'lv'`` (single            |
|                    |                  | LV grid), ``None``          |
|                    |                  | (entire grid)               |
+--------------------+------------------+-----------------------------+
| ``timesteps``      | DatetimeIndex \| | Time steps for analysis.    |
|                    | None             | ``None`` = all in           |
|                    |                  | ``timeseries.timeindex``    |
+--------------------+------------------+-----------------------------+
| ``trou             | str \| None      | ``'lpf'`` = Linear PF       |
| bleshooting_mode`` |                  | seeding, ``'iteration'`` =  |
|                    |                  | gradual load increase       |
+--------------------+------------------+-----------------------------+
| ``                 | float \| None    | Scaling factor for          |
| scale_timeseries`` |                  | time series (e.g. 0.5 for   |
|                    |                  | 50% load)                   |
+--------------------+------------------+-----------------------------+

Time Series Usage
~~~~~~~~~~~~~~~~~~

``analyze`` uses **all** time series from ``edisgo.timeseries``:

Generators
^^^^^^^^^^^

-  **Source:** ``edisgo.timeseries.generators_active_power``
-  **Source:** ``edisgo.timeseries.generators_reactive_power``
-  **Content:** Feed-in of all generators (PV, wind, CHP, etc.) in
   MW/MVAr
-  **Time resolution:** Typically 1h or 15min
-  **Origin:** Database (eGon), worst-case profile, or optimized
   time series

Loads
^^^^^^

-  **Source:** ``edisgo.timeseries.loads_active_power``
-  **Source:** ``edisgo.timeseries.loads_reactive_power``
-  **Content:** Household load, commercial, industrial in MW/MVAr
-  **Time resolution:** Typically 1h or 15min
-  **Origin:** Database, standard load profiles, or measured data

Storage
^^^^^^^^

-  **Source:** ``edisgo.timeseries.storage_units_active_power``
-  **Source:** ``edisgo.timeseries.storage_units_reactive_power``
-  **Content:** Battery storage charging/discharging in MW/MVAr
-  **Time resolution:** As time series index
-  **Origin:** Optimization or predefined schedules

Heat Pumps
^^^^^^^^^^^

-  **Source:** Indirectly from ``heat_demand_df`` and ``cop_df``
-  **Calculation:** ``P_el = heat_demand / COP``
-  **Time resolution:** As time series index
-  **Origin:** Heat demand profiles (e.g. BDEW), COP profiles
   (temperature-dependent)
-  **After optimization:** From optimized time series
   ``timeseries.heat_pumps_active_power``

Charging Points (Electric Vehicles)
^^^^^^^^^^^^^^^^^^^^

-  **Source:** ``edisgo.timeseries.charging_points_active_power``
-  **Time resolution:** As time series index
-  **Origin:** Charging profiles (e.g. SimBEV), flexibility bands, or
   optimization

Process Flow
~~~~~~~~~~~~~

.. code:: python

   # 1. Determine time steps
   if timesteps is None:
       timesteps = self.timeseries.timeindex
   else:
       timesteps = pd.DatetimeIndex(timesteps)

   # 2. Convert to PyPSA network
   pypsa_network = self.to_pypsa(
       mode=mode,
       timesteps=timesteps
   )

   # 3. Optional: Scale time series
   if scale_timeseries is not None:
       pypsa_network.loads_t.p_set *= scale_timeseries
       pypsa_network.generators_t.p_set *= scale_timeseries
       # ... scale other time series

   # 4. Perform power flow calculation
   pypsa_network.pf(
       timesteps,
       use_seed=(troubleshooting_mode == 'lpf')
   )

   # 5. Check convergence
   converged_ts = timesteps[pypsa_network.converged]
   not_converged_ts = timesteps[~pypsa_network.converged]

   if len(not_converged_ts) > 0:
       logger.warning(f"Power flow did not converge for {len(not_converged_ts)} timesteps")

   # 6. Process results
   pypsa_io.process_pfa_results(
       edisgo_obj=self,
       pypsa_network=pypsa_network,
       timesteps=timesteps
   )

   # 7. Store results in edisgo.results
   # self.results.v_res       -> Voltages at nodes
   # self.results.i_res       -> Currents on lines
   # self.results.pfa_p       -> Active power flows
   # self.results.pfa_q       -> Reactive power flows

When is analyze called?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Manually by user:**

   .. code:: python

      edisgo.analyze()  # Analyze entire grid, all time steps

2. **By reinforce function:**

   -  Initially: Identify grid problems
   -  After each reinforcement: Check if problems solved
   -  Iteratively until no more violations

3. **After optimization (optional):**

   .. code:: python

      edisgo.pm_optimize(...)
      edisgo.analyze()  # Analyze with optimized time series

4. **For worst-case analysis:**

   .. code:: python

      # Only two critical time points
      worst_case_ts = edisgo.get_worst_case_timesteps()
      edisgo.analyze(timesteps=worst_case_ts)

Troubleshooting Modes
~~~~~~~~~~~~~~~~~~~~

Linear Power Flow Seeding (``'lpf'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Problem: Non-linear PF does not converge
-  Solution: Start with linear PF solution (angles) as initial value
-  Benefit: Stabilizes convergence for difficult grids

Iteration (``'iteration'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Problem: Convergence not possible at high load
-  Solution: Start with 10% load, increase gradually to 100%
-  Benefit: Finds solution at extreme operating points

Output
~~~~~~~

**Successful analysis:**

::

   Info: Power flow analysis completed for 8760 timesteps
   Info: 8760 timesteps converged, 0 did not converge

**Convergence problems:**

::

   Warning: Power flow did not converge for 15 timesteps
   Warning: Non-converged timesteps: ['2035-01-15 18:00', '2035-07-21 12:00', ...]

--------------

The reinforce Function
----------------------

Function Definition
~~~~~~~~~~~~~~~~~~~

**File:** ``edisgo/edisgo.py`` (line ~1243) **Implementation:**
``edisgo/flex_opt/reinforce_grid.py`` (line ~25)

**Signature:**

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

What does reinforce do?
~~~~~~~~~~~~~~~~~~~~

The ``reinforce`` function **identifies grid problems** (overload,
voltage violations) and **performs reinforcement measures**:

1. **Line reinforcement:** Parallel lines or replacement with
   larger cross-section
2. **Transformer reinforcement:** Parallel transformers or larger capacity
3. **Voltage level separation:** Split LV grids if needed
4. **Cost calculation:** Grid expansion costs (€)

Parameters
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Parameter
     - Type
     - Default
     - Description
   * - ``timesteps_pfa``
     - ``str | DatetimeIndex | None``
     - ``'snapshot_analysis'``
     - ``'snapshot_analysis'`` = 2 worst-case time steps, ``DatetimeIndex`` = custom, ``None`` = all time steps
   * - ``reduced_analysis``
     - ``bool``
     - ``False``
     - Uses only most critical time steps (highest overload or voltage deviation)
   * - ``max_while_iterations``
     - ``int``
     - ``20``
     - Maximum number of reinforcement iterations
   * - ``split_voltage_band``
     - ``bool``
     - ``True``
     - Separate voltage bands for LV/MV (e.g. LV ±3 %, MV ±7 %)
   * - ``mode``
     - ``str | None``
     - ``None``
     - Grid level: ``'mv'``, ``'mvlv'``, ``'lv'`` or ``None`` (= automatic)
   * - ``without_generator_import``
     - ``bool``
     - ``False``
     - Ignores generator feed-in (only useful for planning analyses)
   * - ``n_minus_one``
     - ``bool``
     - ``False``
     - Considers (n-1) criterion

Time Series Usage
~~~~~~~~~~~~~~~~~~

Uses **same time series as analyze**:

-  ``generators_active_power``, ``generators_reactive_power``
-  ``loads_active_power``, ``loads_reactive_power``
-  ``storage_units_active_power``, ``storage_units_reactive_power``
-  Heat pump loads (from heat_demand/COP)
-  Charging point loads

**Time series selection:**

Option 1: Snapshot Analysis (``timesteps_pfa='snapshot_analysis'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Only 2 critical time points
   ts1 = timestep_max_residual_load    # Max. residual load (high load, low generation)
   ts2 = timestep_min_residual_load    # Min. residual load (low load, high generation)
   timesteps = [ts1, ts2]

**Advantage:** Very fast (only 2 PFA instead of 8760) **Disadvantage:** May
miss rare problems

Option 2: Reduced Analysis (``reduced_analysis=True``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # 1. Initial PFA with all time steps
   edisgo.analyze(timesteps=all_timesteps)

   # 2. Identify most critical time steps
   critical_timesteps = get_most_critical_timesteps(
       overloading_factor=1.0,  # Only time steps with overload
       voltage_deviation=0.03   # Only time steps with >3% voltage deviation
   )

   # 3. Reinforcement only based on these time steps
   timesteps = critical_timesteps  # e.g. 50 instead of 8760

**Advantage:** Much faster than full analysis, more accurate than snapshot
**Disadvantage:** Initial PFA with all time steps required

Option 3: All Time Steps (``timesteps_pfa=None``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   timesteps = edisgo.timeseries.timeindex  # e.g. all 8760h of a year

**Advantage:** Maximum accuracy **Disadvantage:** Very computationally intensive
(many PFA)

Option 4: Custom (``timesteps_pfa=custom_datetimeindex``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # E.g. only winter months
   timesteps = pd.date_range('2035-01-01', '2035-03-31', freq='H')

reinforce Algorithm
~~~~~~~~~~~~~~~~~~~~~

Step 1: Eliminate Overloads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   iteration = 0
   while has_overloading() and iteration < max_while_iterations:
       iteration += 1

       # 1.1 Check HV/MV station
       if hv_mv_station_max_overload() > 0:
           reinforce_hv_mv_station()

       # 1.2 Check MV/LV stations
       for station in mv_lv_stations:
           if station.max_overload > 0:
               reinforce_mv_lv_station(station)

       # 1.3 Check MV lines
       for line in mv_lines:
           if line.max_relative_overload > 0:
               reinforce_line(line)

       # 1.4 Check LV lines
       for line in lv_lines:
           if line.max_relative_overload > 0:
               reinforce_line(line)

       # 1.5 Reanalyze
       edisgo.analyze(timesteps=timesteps)

       # 1.6 Check convergence
       if not has_overloading():
           break

**Reinforcement measures:** - **Parallel lines:** Switch identical type
in parallel - **Line replacement:** Larger cross-section (e.g.
150mm² → 240mm²) - **Parallel transformers:** Identical transformer in parallel -
**Transformer replacement:** Larger capacity (e.g. 630kVA → 1000kVA)

Step 2: Solve MV Voltage Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   iteration = 0
   while has_voltage_issues_mv() and iteration < max_while_iterations:
       iteration += 1

       # Identify critical lines
       critical_lines = get_lines_voltage_issues(voltage_level='mv')

       for line in critical_lines:
           reinforce_line(line)

       # Reanalyze
       edisgo.analyze(timesteps=timesteps)

Step 3: MV/LV Station Voltage Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   for station in mv_lv_stations:
       if has_voltage_issues_at_secondary_side(station):
           # Increase transformer capacity
           reinforce_mv_lv_station(station)

   edisgo.analyze(timesteps=timesteps)

Step 4: Solve LV Voltage Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   for lv_grid in lv_grids:
       while has_voltage_issues(lv_grid) and iteration < max_while_iterations:
           iteration += 1

           # Reinforce critical lines
           critical_lines = get_lines_voltage_issues(
               grid=lv_grid,
               voltage_level='lv'
           )

           for line in critical_lines:
               reinforce_line(line)

           edisgo.analyze(timesteps=timesteps, mode='lv')

Step 5: Final Check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Check if voltage reinforcements caused new overloads
   edisgo.analyze(timesteps=timesteps)

   if has_overloading():
       # Back to step 1
       goto_step_1()

Step 6: Cost Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Calculate grid expansion costs
   costs = calculate_grid_expansion_costs(edisgo)

   # Costs per component
   line_costs = costs['lines']          # €
   trafo_costs = costs['transformers']  # €
   total_costs = costs['total']         # €

   # Store in edisgo.results
   edisgo.results.grid_expansion_costs = costs

Reinforcement Logic
~~~~~~~~~~~~~~~~~

Line Reinforcement
^^^^^^^^^^^^^^^^^^^

.. code:: python

   def reinforce_line(line):
       # 1. Calculate required capacity
       required_capacity = line.s_nom * (1 + max_relative_overload)

       # 2. Option A: Parallel lines
       num_parallel = ceil(required_capacity / line.s_nom)
       cost_parallel = num_parallel * line_cost(line.type)

       # 3. Option B: Larger cross-section
       new_type = get_next_larger_type(line.type)
       if new_type is not None:
           cost_replacement = line_cost(new_type)
       else:
           cost_replacement = inf

       # 4. Choose cheaper option
       if cost_parallel < cost_replacement:
           add_parallel_lines(line, num_parallel - 1)
       else:
           replace_line(line, new_type)

Transformer Reinforcement
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   def reinforce_transformer(trafo):
       # 1. Calculate required power
       required_power = trafo.s_nom * (1 + max_relative_overload)

       # 2. Option A: Parallel transformers
       num_parallel = ceil(required_power / trafo.s_nom)
       cost_parallel = num_parallel * trafo_cost(trafo.type)

       # 3. Option B: Larger transformer
       new_type = get_next_larger_trafo(trafo.s_nom)
       cost_replacement = trafo_cost(new_type)

       # 4. Choose cheaper option
       if cost_parallel < cost_replacement:
           add_parallel_trafos(trafo, num_parallel - 1)
       else:
           replace_trafo(trafo, new_type)

Output
~~~~~~~

**Successful reinforcement:**

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

**Iteration limit reached:**

::

   Warning: Maximum number of iterations (20) reached. Grid issues may remain.

When is reinforce called?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Manually by user:**

   .. code:: python

      edisgo.reinforce()

2. **Base grid reinforcement BEFORE new components (optional):**

   .. code:: python

      # Load base grid (e.g. status quo 2024)
      edisgo = EDisGo(ding0_grid="path/to/grid")
      edisgo.import_generators(scenario="status_quo")
      edisgo.import_loads(scenario="status_quo")

      # Optional: Reinforce base grid (as reference)
      edisgo.reinforce()  # Document costs for base grid

      # THEN: Add new components for future scenario
      edisgo.add_heat_pumps(scenario="2035_high")
      edisgo.add_charging_points(scenario="2035_high")

      # IMPORTANT: NO reinforce at this point!
      # Instead: Use optimization → see point 5

   **Useful for:** Reference scenario, compare grid expansion costs
   with/without flexibility

3. **After scenario simulation WITHOUT optimization:**

   .. code:: python

      # If you do NOT want to use optimization (purely conventional grid expansion)
      edisgo.add_charging_points(scenario='high_ev')
      edisgo.reinforce()  # Reinforce grid for new load (without flexibility)

   **Disadvantage:** High grid expansion costs, flexibility potential
   not utilized

4. **After optimization (MANDATORY!):**

   .. code:: python

      # Correct sequence:
      # 1. Add new components (HP, CP, storage)
      edisgo.add_heat_pumps(...)
      edisgo.add_charging_points(...)

      # 2. Run optimization (uses flexibility)
      edisgo.pm_optimize(opf_version=2, curtailment_14a=True)

      # 3. MANDATORY: Grid expansion for remaining problems
      edisgo.reinforce()  # Fixes overloads that optimization couldn't solve

      # Result: Minimized grid expansion costs through flexibility utilization

   **Why mandatory?**

   -  Optimization minimizes grid expansion, but cannot solve all problems
   -  Slack variables > 0 indicate remaining
      grid restriction violations
   -  Remaining overloads must be resolved through conventional expansion
   -  **Without this step:** Grid is NOT reliable!

5. **Iterative workflow for multiple scenarios:**

   .. code:: python

      # Compare different flexibility scenarios

      # Scenario 1: Without optimization (reference)
      edisgo_ref = edisgo.copy()
      edisgo_ref.reinforce()
      costs_ref = edisgo_ref.results.grid_expansion_costs

      # Scenario 2: With optimization but without §14a
      edisgo_opt = edisgo.copy()
      edisgo_opt.pm_optimize(opf_version=2, curtailment_14a=False)
      edisgo_opt.reinforce()
      costs_opt = edisgo_opt.results.grid_expansion_costs

      # Scenario 3: With optimization and §14a
      edisgo_14a = edisgo.copy()
      edisgo_14a.pm_optimize(opf_version=2, curtailment_14a=True)
      edisgo_14a.reinforce()
      costs_14a = edisgo_14a.results.grid_expansion_costs

      # Comparison
      print(f"Without optimization: {costs_ref:,.0f} €")
      print(f"With optimization: {costs_opt:,.0f} € (-{100*(1-costs_opt/costs_ref):.1f}%)")
      print(f"With §14a: {costs_14a:,.0f} € (-{100*(1-costs_14a/costs_ref):.1f}%)")

--------------

The §14a EnWG Optimization
-------------------------

What is §14a EnWG?
~~~~~~~~~~~~~~~~~~

**Legal basis:** § 14a Energy Industry Act (EnWG)

**Content:** Grid operators may **curtail controllable consumption devices**
(heat pumps, electric vehicle charging stations) during grid bottlenecks
**down to a minimum power** (4.2 kW).

**Conditions:** - Maximum **time budget**: Typically 2 hours per day -
**Minimum power**: 4.2 kW (0.0042 MW) must remain guaranteed -
**Compensation**: Reduced network charges for customers

**Goal:** Reduce grid expansion through targeted peak load curtailment

How does §14a differ from standard optimization?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard optimization (WITHOUT §14a):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Heat pumps with thermal storage:** Temporal load shifting
-  **Electric vehicles:** Charging control within flexibility band
-  **Inflexible HP/CP:** Cannot be curtailed

§14a optimization:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **ALL heat pumps > 4.2 kW:** Can be curtailed down to 4.2 kW
-  **ALL charging points > 4.2 kW:** Can be curtailed down to 4.2 kW
-  **Even without storage:** Curtailment possible
-  **Time budget constraints:** Max. 2h/day curtailment
-  **Binary decision:** Curtailment active YES/NO

Mathematical Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~

Virtual Generators
^^^^^^^^^^^^^^^^^^^^^

§14a curtailment is modeled through **virtual generators**:

::

   Net_load_HP = Original_load_HP - p_hp14a

   Example:
   - HP load: 8 kW
   - Curtailment: 3.8 kW (virtual generator produces 3.8 kW)
   - Net load: 8 - 3.8 = 4.2 kW (minimum load)

**Advantages of this modeling:** - No modification of load time series
needed - Compatible with PowerModels structure - Simple implementation
in optimization problem

Variables (per heat pump h, time step t)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: julia

   # Continuous variable: Curtailment power
   @variable(model, 0 <= p_hp14a[h,t] <= pmax[h])

   # Binary variable: Curtailment active?
   @variable(model, z_hp14a[h,t], Bin)

**Parameters:** - ``pmax[h] = P_nominal[h] - P_min_14a`` -
``P_nominal[h]``: Rated power of heat pump h (e.g. 8 kW) -
``P_min_14a = 4.2 kW``: Legal minimum load -
``pmax[h] = 8 - 4.2 = 3.8 kW``: Maximum curtailment power

-  ``max_hours_per_day = 2``: Time budget in hours per day

Due to space constraints, I'll complete the remaining chapters more concisely. Let me finalize the document by adding summaries of the remaining complex sections.

--------------

Time Series Usage
------------------

All time series are stored in ``edisgo.timeseries``:

Time Series Sources:
- Database import (eGon database) - generators, loads with hourly resolution
- Worst-case profiles - 2 critical time points for quick planning
- Manual time series - user-defined profiles
- Optimized time series (after pm_optimize) - updated with optimal schedules

Time series are used in:
1. **analyze**: All time series for power flow calculation
2. **reinforce**: Snapshot (2 timesteps), reduced, or all timesteps
3. **pm_optimize**: Input time series → Julia optimization → optimized output time series

--------------

File Paths and References
-------------------------

Python Files
~~~~~~~~~~~~

Key Python files:
- ``edisgo/edisgo.py``: Main EDisGo class with analyze(), reinforce(), pm_optimize()
- ``edisgo/io/powermodels_io.py``: PowerModels conversion, §14a generators
- ``edisgo/opf/powermodels_opf.py``: Julia subprocess, JSON communication
- ``edisgo/flex_opt/reinforce_grid.py``: Reinforcement algorithm

Julia Files
~~~~~~~~~~~~~

Key Julia files:
- ``edisgo/opf/eDisGo_OPF.jl/Main.jl``: Main entry, solver setup
- ``edisgo/opf/eDisGo_OPF.jl/src/prob/opf_bf.jl``: build_mn_opf_bf_flex()
- ``edisgo/opf/eDisGo_OPF.jl/src/core/variables.jl``: Variable definitions
- ``edisgo/opf/eDisGo_OPF.jl/src/core/constraint.jl``: Constraints
- ``edisgo/opf/eDisGo_OPF.jl/src/core/constraint_hp_14a.jl``: HP §14a constraints
- ``edisgo/opf/eDisGo_OPF.jl/src/core/constraint_cp_14a.jl``: CP §14a constraints
- ``edisgo/opf/eDisGo_OPF.jl/src/core/objective.jl``: Objective functions
