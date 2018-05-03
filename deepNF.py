'''
Run deepNF

To run:
    python deepNF.py

    Legion:

    qsub -b y -N deepNF_GPU -pe smp 8 -l h_rt=2:0:0,mem=15G,gpu=1 -ac allow=P \
        python $HOME/git/deepNF/deepNF.py --architecture 2

    qsub -b y -N deepNF -pe smp 8 -l h_rt=2:0:0,mem=15G \
        python $HOME/git/deepNF/deepNF.py --architecture 2

    CS:

    qsub -b y -N deepNF_GPU
        -pe smp 8 -l h_rt=2:0:0,h_vmem=7.8G,tmem=7.8G,gpu=1 \
        -ac allow=P \
        python $HOME/git/deepNF/deepNF.py --architecture 2

    qsub -b y -N deepNF -pe smp 2 -l h_rt=2:0:0,h_vmem=7.8G,tmem=7.8G \
        python $HOME/git/deepNF/deepNF.py --architecture 2
'''


import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from pathlib import Path
from keras.models import Model, load_model
from sklearn.preprocessing import minmax_scale
from validation import cross_validation, temporal_holdout
from autoencoders import build_MDA, build_AE
from keras.callbacks import EarlyStopping
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
import sys
import os
import argparse


##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--organism',     default='yeast',      type=str)
parser.add_argument('-m', '--model-type',   default='mda',        type=str)
parser.add_argument('-p', '--models-path',  default="./models",   type=str)
parser.add_argument('-r', '--results-path', default="./results",  type=str)
parser.add_argument('-d', '--data-path',    default="$AGAPEDATA", type=str)
parser.add_argument('-a', '--architecture', default="2",          type=str)
parser.add_argument('-e', '--epochs',       default=10,           type=int)
parser.add_argument('-b', '--batch-size',   default=128,          type=int)
parser.add_argument('-n', '--n-trials',     default=10,           type=int)
parser.add_argument(      '--K',            default=3,            type=int)
parser.add_argument(      '--alpha',        default=.98,          type=float)
parser.add_argument(      '--outfile-tags', default="",           type=str)
args = parser.parse_args()
print("ARGUMENTS\n", args)

org          = args.organism
model_type   = args.model_type
models_path  = args.models_path
results_path = args.results_path
data_path    = os.path.expandvars(args.data_path)
select_arch  = [int(i) for i in args.architecture.split(",")]
epochs       = args.epochs
batch_size   = args.batch_size
n_trials     = args.n_trials
K            = args.K
alpha        = args.alpha
ofile_tags   = args.outfile_tags


architectures = {
    1: [600],
    2: [6*2000, 600, 6*2000],
    3: [6*2000, 6*1000, 600, 6*1000, 6*2000],
    4: [6*2000, 6*1000, 6*500, 600, 6*500, 6*1000, 6*2000],
    5: [6*2000, 6*1200, 6*800, 600, 6*800, 6*1200, 6*2000],
    6: [6*2000, 6*1200, 6*800, 6*400, 600, 6*400, 6*800, 6*1200, 6*2000]}

architecture = dict((key, a) for key, a in architectures.items() if key in select_arch)


######################
# Prepare filesystem #
######################

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


mkdir("models")
mkdir("results")


#################
# Load networks #
#################

def load_network(i):
    filepath = os.path.join(
        data_path, "deepNF",
        f"{org}_net_{str(i)}_K{str(K)}_alpha{str(alpha)}.mat")

    if not os.path.exists(filepath):
        raise OSError("Network not found at:", filepath)

    print(f"Loading network {i} from {filepath}")
    N = sio.loadmat(filepath, squeeze_me=True)['Net'].toarray()
    return N


Nets = []
input_dims = []

for i in range(1, 7):
    Net = load_network(i)
    Nets.append(minmax_scale(Net))
    input_dims.append(Net.shape[1])


#########################
# Train the autoencoder #
#########################

def plot_loss(history):
    plt.plot(history.history['loss'], 'o-')
    plt.plot(history.history['val_loss'], 'o-')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(str(Path(models_path, model_name + '_loss.png')),
                bbox_inches='tight')


model_names = []

for a in architecture:
    print("### [Model] Running for architecture: ", architecture[a])
    model_name = f"{org}_{model_type.upper()}_arch_{str(a)}_{ofile_tags}"
    model_names.append(model_name)

    # Build model
    model = build_MDA(input_dims, architecture[a])

    # Train model
    history = model.fit(
        Nets,
        Nets,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=2)])

    plot_loss(history)

    with open(Path(models_path, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    # Extract middle layer
    mid_model = Model(
        inputs=model.input,
        outputs=model.get_layer('middle_layer').output)

    mid_model.save(Path(models_path, f"{model_name}.h5"))


###############################
# Generate network embeddings #
###############################

for model_name in model_names:
    print(f"Running for: {model_name}")

    my_file = Path(models_path, f"{model_name}.h5")

    if my_file.exists():
        mid_model = load_model(my_file)
    else:
        raise OSError("Model does not exist", my_file)

    embeddings = minmax_scale(mid_model.predict(Nets))

    sio.savemat(
        str(Path(results_path, model_name + '_features.mat')),
        {'embeddings': embeddings})
