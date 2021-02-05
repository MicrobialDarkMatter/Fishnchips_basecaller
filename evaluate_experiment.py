import argparse
import src.api as api

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=False, help='Name of the experiment. If none provided, name with a timestamp will be assigned.')
    args = parser.parse_args()
    assert type(args.name) == str, 'Experiment name must be a string.'    
    return args  

def main(experiment_name):
    api.discard_existing_evaluation(experiment_name)
    api.evaluate(experiment_name)

if __name__ == "__main__":
    args = parse_args()
    main(args.name)
    print(' - Script has finished successfully.')