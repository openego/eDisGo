%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                  %%%%%
%%%%      NICTA Energy System Test Case Archive (NESTA) - v0.7.0      %%%%%
%%%%               Optimal Power Flow - Radial Topology               %%%%%
%%%%                         05 - June - 2017                         %%%%%
%%%%                                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   A modified version of the WSCC 9 bus network with a quadratic objective used in,
%
%   Kocuk, B. & Dey, S. & Sun, X.A.,
%   "Inexactness of SDP Relaxation and Valid Inequalities for Optimal Power Flow",
%   IEEE Transactions on Power Systems, March, 2015
%
function mpc = nesta_case9_kds__rad
mpc.version = '2';
mpc.baseMVA = 100.0;

%% area data
%	area	refbus
mpc.areas = [
	1	 5;
];

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	 3	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.10000	    0.90000;
	2	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.10000	    0.90000;
	3	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.10000	    0.90000;
	4	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.10000	    0.90000;
	5	 1	 90.0	 30.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.10000	    0.90000;
	6	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.10000	    0.90000;
	7	 1	 100.0	 35.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.10000	    0.90000;
	8	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.10000	    0.90000;
	9	 1	 125.0	 50.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.10000	    0.90000;
];

%% generator data Pg[3] =
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
%mpc.gen = [
%	1	 0.0	 0.0	 125.0	 100.0	 1.0	 100.0	 1	 250.0	 10.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0; % NUC
%	2	 163.0	 0.0	 150.0	 100.0	 1.0	 100.0	 1	 300.0	 10.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0; % COW
%	3	 85.0	 0.0	 135.0	 100.0	 1.0	 100.0	 1	 270.0	 10.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0; % NG
%];
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
	1	 0.0	 0.0	 125.0	 0.0	 1.0	 100.0	 1	 250.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0; % NUC
	2	 163.0	 0.0	 150.0	 0.0	 1.0	 100.0	 1	 300.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0; % COW
	3	 85.0	 0.0	 135.0	 0.0	 1.0	 100.0	 1	 270.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0	 0.0; % NG
];

%% generator cost data
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	 1500.0	 0.0	 3	   0.110000	   5.000000	 150.000000; % NUC
	2	 2000.0	 0.0	 3	   0.085000	   1.200000	 600.000000; % COW
	2	 3000.0	 0.0	 3	   0.122500	   1.000000	 335.000000; % NG
];





%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	 4	 0.0	 0.0576	 0.0	 910	 250.0	 250.0	 0.0	 0.0	 1	 -30.0	 30.0;
	4	 5	 0.017	 0.092	 0.158	 560	 250.0	 250.0	 0.0	 0.0	 1	 -30.0	 30.0;
	5	 6	 0.039	 0.17	 0.358	 301	 150.0	 150.0	 0.0	 0.0	 1	 -30.0	 30.0;
	3	 6	 0.0	 0.0586	 0.0	 894	 300.0	 300.0	 0.0	 0.0	 1	 -30.0	 30.0;
	7	 8	 0.0085	 0.072	 0.149	 723	 250.0	 250.0	 0.0	 0.0	 1	 -30.0	 30.0;
	8	 2	 0.0	 0.0625	 0.0	 839	 250.0	 250.0	 0.0	 0.0	 1	 -30.0	 30.0;
	8	 9	 0.032	 0.161	 0.306	 320	 250.0	 250.0	 0.0	 0.0	 1	 -30.0	 30.0;
	9	 4	 0.01	 0.085	 0.176	 612	 250.0	 250.0	 0.0	 0.0	 1	 -30.0	 30.0;
];
%% storage data
% hours
mpc.time_elapsed = 1.0
%   storage_bus ps qs energy  energy_rating charge_rating  discharge_rating  charge_efficiency  discharge_efficiency  thermal_rating  qmin  qmax  r  x  standby_loss  status
%mpc.storage = [
%	2	0.0	0.0	 20.0	 100.0	 50.0	 70.0	 0.8	 0.9	 100.0	 -50.0	 70.0	 0.1	 0.0	 0.0	 1;
%	3	0.0	0.0	 30.0	 100.0	 50.0	 70.0	 0.9	 0.8	 100.0	 -50.0	 70.0	 0.1	 0.0	 0.0	 1;
%];
%   storage_bus energy  energy_rating charge_rating  discharge_rating  charge_efficiency  discharge_efficiency  thermal_rating  qmin  qmax  r  x  standby_loss  status
mpc.storage = [
	2	100.0	 150.0	 100.0	 100.0	 0.9	 0.9	 100.0	 -50.0	 70.0	 0.1	 0.0	 0.0	 1;
%	3	30.0	 100.0	 50.0	 70.0	 0.9	 0.8	 100.0	 -50.0	 70.0	 0.1	 0.0	 0.0	 1;
];

% INFO    : === Translation Options ===
% INFO    : Phase Angle Bound:           30.0 (deg.)
% INFO    : Line Capacity Model:         ub
% INFO    : Gen Reactive Capacity Model: am50ag
% INFO    : Line Capacity PAB:           25.0 (deg.)
% WARNING : No active generation at the slack bus, assigning type - NUC
% INFO    : 
% INFO    : === Generator Classification Notes ===
% INFO    : NUC    1   -     0.00
% INFO    : COW    1   -    65.73
% INFO    : NG     1   -    34.27
% INFO    : 
% INFO    : === Generator Reactive Capacity Atmost Max 50 Percent Active Model Notes ===
% INFO    : Gen at bus 1 - NUC	: Pmax 250.0, Qmin 100.0, Qmax 300.0 -> Qmin 100.0, Qmax 125.0
% INFO    : Gen at bus 2 - COW	: Pmax 300.0, Qmin 100.0, Qmax 300.0 -> Qmin 100.0, Qmax 150.0
% INFO    : Gen at bus 3 - NG	: Pmax 270.0, Qmin 100.0, Qmax 300.0 -> Qmin 100.0, Qmax 135.0
% INFO    : 
% INFO    : === Line Capacity UB Model Notes ===
% INFO    : Updated Thermal Rating: on line 1-4 : Rate A , 9900.0 -> 910
% INFO    : Updated Thermal Rating: on line 4-5 : Rate A , 9900.0 -> 560
% INFO    : Updated Thermal Rating: on line 5-6 : Rate A , 9900.0 -> 301
% INFO    : Updated Thermal Rating: on line 3-6 : Rate A , 9900.0 -> 894
% INFO    : Updated Thermal Rating: on line 7-8 : Rate A , 9900.0 -> 723
% INFO    : Updated Thermal Rating: on line 8-2 : Rate A , 9900.0 -> 839
% INFO    : Updated Thermal Rating: on line 8-9 : Rate A , 9900.0 -> 320
% INFO    : Updated Thermal Rating: on line 9-4 : Rate A , 9900.0 -> 612
% INFO    : 
% INFO    : === Writing Matpower Case File Notes ===
