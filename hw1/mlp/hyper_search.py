import nni
from nni.experiment import Experiment
search_space = {
    'hidden_dim': {'_type': 'choice', '_value': [64, 128, 256, 512, 1024]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'batch_size': {'_type': 'choice', '_value': [16, 32, 64]},
    'epochs':{'_type': 'choice', '_value': [20,40,60,80,100]}
}

experiment = Experiment('local')
experiment.config.trial_command = 'python mlp.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 200
experiment.config.trial_concurrency = 2
experiment.run(8081)
