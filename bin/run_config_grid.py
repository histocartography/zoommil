import argparse
import json

from zoommil.config.config_grid import ConfigGridRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters to run LSF jobs')
    parser.add_argument('--repo_path', type=str, default='./',
                        help='path to repository root')
    parser.add_argument('--config_path', type=str, default='./zoommil/config/config_bright.json',
                        help='path to config (grid) file')
    args = parser.parse_args()

    # load config grid
    with open(args.config_path, 'r') as c_file:
        config = json.load(c_file)

    grid_runner = ConfigGridRunner(config=config, repo_path=args.repo_path)

    # execute scripts
    grid_runner.run()
    print('Jobs submitted!')
