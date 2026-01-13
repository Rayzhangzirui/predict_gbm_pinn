#!/usr/bin/env python
# unified logger for mlflow, stdout, csv
import sys
import os
from util import *
try:
    import mlflow
    from MlflowHelper import *
    MLFLOW_AVAILABLE = True
    
except ImportError:
    MLFLOW_AVAILABLE = False

from torchtnt.utils.loggers import StdoutLogger
import socket


import shutil
import tempfile

class Logger:
    def __init__(self, opts):
        self.opts = opts
        
        if self.opts.get('use_mlflow') and not MLFLOW_AVAILABLE:
            print("Warning: MLflow not installed. Disabling MLflow logging.")
            self.opts['use_mlflow'] = False
        
        #
        if self.opts['use_mlflow']:
             # ...existing code...
            runname = self.opts['run_name']
            expname = self.opts['experiment_name']

            # # if activerun exist, end it
            # if mlflow.active_run() is not None:
            #     mlflow.end_run()

            # check if experiment exist. If not, create experiment
            if mlflow.get_experiment_by_name(expname) is None:
                mlflow.create_experiment(expname)
                print(f"Create experiment {expname}")

            # check if run_name already exist. If exist, raise warning
            if mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(expname).experiment_id, filter_string=f"tags.mlflow.runName = '{runname}'").shape[0] > 0:
                Warning(f"Run name {runname} already exist!")
                # Warning ValueError(f"Run name {runname} already exist!")


            mlflow.set_experiment(expname)
            self.mlrun = mlflow.start_run(run_name=self.opts['run_name'])
            mlflow.set_tag("host", socket.gethostname())

            print("tracking_uri:", mlflow.get_tracking_uri())
            print("artifact_uri:", mlflow.get_artifact_uri())

        else:
            self.stdout_logger = StdoutLogger(precision=6)

        # Determine save directory
        self.save_dir = self._determine_save_dir()
        print(f"Artifacts will be saved to: {self.save_dir}")

        

        
        
        
    def set_tags(self, key:str, value:str):
        if self.opts['use_mlflow']:
            mlflow.set_tags({key:value})
        else:
            print(key, value)

    def log_metrics(self, metric_dict:dict, step=None, prefix=''):
        # remove key with None value
        metric_dict = {prefix+k:v for k,v in metric_dict.items()}

        if self.opts['use_mlflow']:
            mlflow.log_metrics(to_double(metric_dict), step=step)
        else:
            self.stdout_logger.log_dict(payload = metric_dict, step=step)


    def close(self):
        if self.opts['use_mlflow']:
            mlflow.end_run()
        if hasattr(self, 'tmp_dir_obj'):
             self.tmp_dir_obj.cleanup()
             print("Cleaned up temporary directory.")

    def log_options(self, options: dict):
        # save optioins as json
        if self.opts['use_mlflow']:
            mlflow.log_params(flatten(options))
        else:
            print('\n')
            print(options)
    
    def log_params(self, params: dict):
        if self.opts['use_mlflow']:
            mlflow.log_params(flatten(params))
        else:
            print('\n')
            print(params)

    def get_dir(self):
        # Public method to get the determined directory
        return self.save_dir

    def _determine_save_dir(self):
        # Logic to determine where to save files
        
        if self.opts.get('use_tmp', False):
             self.tmp_dir_obj = tempfile.TemporaryDirectory()
             return self.tmp_dir_obj.name

        if self.opts['use_mlflow']:
            # MLflow manages its own artifact location, but we often need a local path to write to first
            # get_active_artifact_dir usually returns a local temp path managed by MLflow or the artifact URI
            return get_active_artifact_dir()
        else:
            # Default local structure
            dpath = os.path.join('./runs', self.opts['experiment_name'], self.opts['run_name'])
            os.makedirs(dpath, exist_ok=True)
            return dpath

    def gen_path(self, filename: str):
        return os.path.join(self.save_dir, filename)
    
    def load_artifact(self, exp_name=None, run_name=None, name_str=None):
        # return all {filename: path} in artifact directory
        # load from mlflow or local directory
        source = 'mlflow' if self.opts['use_mlflow'] else 'local'

        if name_str is not None:
            try:
                parts = name_str.split(':')
                if len(parts) == 2:
                    # exp_name:run_name
                    source = 'mlflow' if self.opts['use_mlflow'] else 'local'
                    exp_name = parts[0]
                    run_name = parts[1]
                elif len(parts) == 3:
                    # source:exp_name:run_name, source = local or mlflow
                    source = parts[0]
                    assert source in ['local', 'mlflow'], "source must be 'local' or 'mlflow'"
                    exp_name = parts[1]
                    run_name = parts[2]
            except ValueError:
                raise ValueError("name_str must be in the format '[source:]exp_name:run_name'")

        if source == 'mlflow':
            if not MLFLOW_AVAILABLE:
                raise ImportError("Cannot load artifact from MLflow: mlflow is not installed.")
            # get artifact from mlflow
            helper = MlflowHelper()
            run_id = helper.get_id_by_name(exp_name, run_name)
            artifact_dict = helper.get_artifact_dict_by_id(run_id)
            
        else:
            # get files in directory
            dpath = os.path.join(RUNS, exp_name, run_name)
            print(f"Load artifact from {dpath}")
            artifact_dict = {fname: os.path.join(dpath, fname) for fname in os.listdir(dpath)}
            artifact_dict['artifacts_dir'] = dpath
        
        return artifact_dict



if __name__ == "__main__":
    # simple test of logger

    opts  = {'use_mlflow':False, 'use_stdout':True, 'experiment_name':'tmp', 'run_name':'testlogger', 'save_dir':'./test'}

    # read options from command line key value pairs
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        key = args[i]
        val = args[i+1]
        opts[key] = val

    logger = Logger(opts)
    for i in range(10):
        logger.log_metrics({'loss':i}, step=i)
        logger.log_metrics({'param':i}, step=i)
    
    logger.log_options(opts)
    print('')
    print(logger.get_dir())
    print(logger.gen_path('test.txt'))

    logger.close()
    

