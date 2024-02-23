############### Configuration file for Bayesian ###############
layer_type = 'lrt'  # 'bbb' or 'lrt'
activation_type = 'softplus'  # 'softplus' or 'relu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

n_epochs = 200
lr_start = 0.001
num_workers = 8
valid_size = 0.2
batch_size = 128
train_ens = 1
valid_ens = 1
beta_type = 0.1  # 'Blundell', 'Standard', etc. Use float for const value
# n_weight_sampling_inference is basically test_ens
n_weight_sampling_inference = 100 # weight could also be implemented by repeating the test image itself (as done in uncertainty_estimation.py)
