from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config

# export OPENPI_DATA_HOME="/home/charles/workspace/models/openpi/"

# model_name = "pi0_fast_libero"
# model_link = "gs://openpi-assets/checkpoints/pi0_fast_libero"
model_name = "pi0_libero"
model_link = "gs://openpi-assets/checkpoints/pi0_libero"

config = config.get_config(model_name)
checkpoint_dir = download.maybe_download(model_link)

print(checkpoint_dir)

policy = policy_config.create_trained_policy(config, checkpoint_dir)
