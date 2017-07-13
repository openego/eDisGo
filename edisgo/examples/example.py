from edisgo.grid.network import Network
import os

nd = Network.import_from_dingo(os.path.join('data', 'dingo_grids__3545.pkl'))

# TODO: provide select by attribute example how access generators, loads, etc.