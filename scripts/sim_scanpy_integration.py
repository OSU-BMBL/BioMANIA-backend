import logging

logging.basicConfig(
    level=logging.INFO,
    # filename=f'exp/dep_graph/logs/scanpy_api_dep/{api_type}_{date}.log',
    format='%(asctime)s - %(name)s - %(levelname)s \n%(message)s\n',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from argparse import ArgumentParser
from biochatter.api_agent.dep_graph import DependencyGraph, ExecutionGraph
from biochatter.api_agent.python.scanpy.info_hub import api_names, dependencies
from biochatter.api_agent.python.scanpy.api_hub import TOOLS_DICT
from biochatter.api_agent.python.scanpy.base import ScanpyDependency
from biochatter.api_agent.dep_discovery import DependencyFinder

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--stage", type=str, choices=['construct_api', 'dependency_discovery', 'load_dep_graph'])
    args = parser.parse_args()

    if args.stage == 'construct_api':
        # Jiahang (TODO): unify API style.
        """
        python biochatter/api_agent/gen_data_model.py --package_name scanpy --api_dict_name pp
        """

    if args.stage == 'dependency_discovery':
        dep_finder = DependencyFinder(
            package_name='scanpy',
            api_sel='all',
            debug=False
        )

        dep_finder()

    ### Require developers to manually filter out inactive args. ###

    elif args.stage == 'load_dep_graph':
        dep_graph = DependencyGraph(api_names=api_names, 
                        dependencies=dependencies, 
                        api_class_dict=TOOLS_DICT,
                        dep_class=ScanpyDependency)

        execution_graph = ExecutionGraph()