import os


def test_path():

    opf_dir = os.path.dirname(os.path.abspath(__file__))
    julia_env_dir = os.path.join(opf_dir, "edisgoOPF/")

    scenario_data_dir = os.path.join(opf_dir, "edisgo_scenario_data/")
    solution_dir = os.path.join(opf_dir, "opf_solutions/")

    print(opf_dir)
    print(scenario_data_dir)
    print(solution_dir)
    print(julia_env_dir)
    return
