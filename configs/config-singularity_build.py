from mrunner.helpers.specification_helper import create_experiments_helper
import os

EXP_NAME = "example_eval"
EXP_TAGS = ["nethack_eval"]
assert "NEPTUNE_API_TOKEN" in os.environ, "Please set NEPTUNE_API_TOKEN environment variable"


experiments_list = create_experiments_helper(
    experiment_name=EXP_NAME,
    base_config={},
    params_grid={},
    # script="python3 nethack_experiments/fulltext_history_rollout_pm.py",
    script="bash nethack.sh",
    exclude=[
        ".idea",
        ".git",
        "__pychache__",
        "new_test",
        "env",
    ],
    python_path="",
    tags=[globals()["script"][:-3]] + EXP_TAGS,
    with_neptune=True,
    project_name="pmtest/bison-pl",
)