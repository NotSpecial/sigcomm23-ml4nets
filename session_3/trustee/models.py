"""Evaluate model predictions and prediction errors."""

# Importing torch etc. is quite slow, so we want to avoid it unless necessary.
# Otherwise any CLI script takes forever to start, even if it does not need
# the torch functions. We only import it in the relevant functions.
# pylint: disable=import-outside-toplevel

import logging

import experiment_helpers as eh
import numpy as np
import pandas as pd
from experiment_helpers.data import Path
from experiment_helpers.typing import (Dict, Iterable, Optional, PathType,
                                       Sequence, Union)
from numpy.typing import ArrayLike

#from ..config import PufferExperimentConfig
from .data import InOut, InOutDict, to_ttp_inout

InoutOrFrame = Union[InOut, pd.DataFrame]
InOutLike = Union[Dict[int, InOutDict], InOut, pd.DataFrame]


def load_model(modelpath: PathType):
    """Load a ttp model from a pytorch savefile (`.pt`) given by path.

    Alternatively, this can also load a custom model.
    If a `{modelpath}.meta` file exists on the same path, this file is loaded
    first to instantiate the custom model, and then the modelpath is loaded.
    """
    from .src_puffer import ttp
    metapath = Path(str(modelpath) + ".meta")
    if metapath.is_file():
        layers, units = eh.data.read_pickle(metapath)
        model = custom_model(layers=layers, units=units)
    else:
        model = ttp.Model()
    model.load(modelpath)
    return model


def custom_model(layers=4, units=128):
    """Return a custom model.

    This is quite hacky, and meant to create a model that is compatible with
    all our interfaces to the Puffer models, but that has different layers
    and units.

    The custom model can be identified by our `load_model` method by the
    presence of of `.meta` file in the same folder, and we modify the save
    function to automatically create this file.
    """
    import torch

    from .src_puffer import ttp

    class CustomModel(ttp.Model):
        """Override model layers"""

        def __init__(self):
            super().__init__()

            _layers = [
                torch.nn.Linear(self.DIM_IN, units),
                torch.nn.ReLU(),
            ]
            for _ in range(layers-2):
                _layers += [
                    torch.nn.Linear(units, units),
                    torch.nn.ReLU(),
                ]
            _layers += [torch.nn.Linear(units, self.DIM_OUT)]
            self.model = torch.nn.Sequential(
                *_layers
            ).double().to(device=ttp.DEVICE)
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.LEARNING_RATE,
                                              weight_decay=self.WEIGHT_DECAY)

        def save(self, model_path):
            """Save custom meta info, then normal save."""
            meta_path = str(model_path) + ".meta"
            eh.data.to_pickle((layers, units), meta_path)
            super().save(model_path)

    return CustomModel()


def evaluate(predictions: pd.DataFrame,
             modelpath: PathType,
             index: int,
             aggregate=(0.01, 0.5, 0.99)) -> pd.DataFrame:
    """Evaluate predictions for a single index, aggregating result.

    The aggregate always includes mean and std, and optionally all specified
    quantiles.

    Currently only a single model (index) can be evaluated at once.
    """
    assert predictions.columns.get_level_values("model_index").nunique() == 1, \
        "Only one index can be evaluated!"
    # Only one index, simplify frame
    predictions = predictions.droplevel("model_index", axis=1)

    # Load model and germine number of bins.
    model = load_model(Path(modelpath) / f"py-{index}.pt")
    bins = np.arange(model.DIM_OUT)

    # The actual transmission time and which bin it belongs to.
    times = predictions['y']
    y_bin = ttp_discretize(times, model)

    probabilities = (
        predictions
        # Ignore zero probability, otherwise we get -Inf score.
        .loc[predictions['p'] > 0, ['p']]
        .assign(
            logscore=lambda df: np.log(df['p']),
        )
    )

    # Compute squared error per bin and weight by probability
    probs = predictions[[f"y_{ind}" for ind in bins]]
    bin_centers = ttp_bin_centers(model)
    # Errors for each bin, then weight bei probability and sum.
    err = predictions[['y']].to_numpy() - bin_centers
    weighted_err = (err * probs).sum(axis=1).to_frame("err")
    weighted_abs = (np.abs(err) * probs).sum(axis=1).to_frame("err_abs")
    weighted_sqr = ((err**2) * probs).sum(axis=1).to_frame("err_sqr")

    # Aggregate each set of results
    results = [times, probabilities, weighted_err, weighted_abs, weighted_sqr]

    # Also group by bin for detailed results.
    for _bin in bins:
        results.append(weighted_err[y_bin == _bin].add_suffix(f"_{_bin}"))
        results.append(weighted_abs[y_bin == _bin].add_suffix(f"_{_bin}"))
        results.append(weighted_sqr[y_bin == _bin].add_suffix(f"_{_bin}"))

    aggregated = [
        pd.concat([
            res.agg(['mean', 'std']),
            res.quantile(list(aggregate))
        ], axis=0)
        for res in results
    ]
    # Put all together into one frame.
    return (
        pd.concat(aggregated, axis=1)
        .reset_index()
        .rename(columns={'index': "metric"})
    )


def predict(data: InOutLike,
            models: Union[PathType, Sequence[PathType]],
            indices: Optional[Iterable[int]] = None,
            workers: Optional[int] = None) -> pd.DataFrame:
    """Make predicitons (per session and sending time) for the whole dataframe.

    Models may be provided in several ways:
    - a directory containing model files named `py-{index}.pt`
    - a list of model files.

    If a directory is provided, the default indices are range(5).
    If files are provided, the default indices are range(len(models)).
    You can provide custom indices to override.

    Inout can also be a dataframe that will be converted.

    The resulting frame is indexed by session and sending time, and also has
    two-level columns: one for the model index, and one for the results.

    The results consist of:
        - y (true results)
        - y_hat (average prediction)
        - p (probability of correct prediction)
        - y_0, ..., y_20 (softmax prediction vector): *ONLY* probabilities, not
          the bin values!
    """
    from .src_puffer import ttp

    try:
        # Modeldir is a path.
        modeldir = Path(models)  # type: ignore
        if indices is None:
            indices = range(ttp.Model.FUTURE_CHUNKS)
        models = [modeldir / f"py-{ind}.pt" for ind in indices]
    except (AttributeError, TypeError):
        # Modeldir is not a path, we assume it's a sequence of files.
        models = [Path(modelpath) for modelpath in models]  # type: ignore
        if indices is None:
            indices = range(len(models))

    if isinstance(data, pd.DataFrame):
        inout = to_ttp_inout(data)
    else:
        inout = data  # type: ignore

    model_frames, keys = [], []
    for ind, modelfile in zip(indices, models):
        _in, _out, _session, _ts = [
            inout[ind][key]  # type: ignore
            for key in ('in', 'out', 'session', 'ts')
        ]
        if len(_in) == 0:  # No data, but that can happen
            continue

        model = load_model(modelfile)
        # samples x probs.
        probabilities = predict_probabilities(model, _in, workers=workers)
        predictions = prediction_average(model, probabilities)

        discrete = model.discretize_output(_out)
        # For each row, select the element of the softmax vector that would
        # have been the correct prediction.
        correct_prob = probabilities[np.arange(len(probabilities)),
                                     discrete]

        columns = {'y': _out, 'y_hat': predictions, 'p': correct_prob}
        for col in range(ttp.Model.DIM_OUT):
            columns[f"y_{col}"] = probabilities[:, col]

        model_frames.append(pd.DataFrame(
            columns,
            index=pd.MultiIndex.from_arrays(
                [_session, _ts], names=['session', 'sent time (ns GMT)']
            ),
        ))

        keys.append(str(ind))

    return pd.concat(model_frames, axis=1, keys=keys, names=["model_index"])


def predict_and_compare(model,
                        data_in: ArrayLike, data_out: ArrayLike,
                        workers: Optional[int] = None) -> np.ndarray:
    """Make predictions and compare to labels, returning True if correct."""
    predicted_bin = predict_probabilities(
        model, data_in, workers=workers).argmax(axis=1)
    true_bin = model.discretize_output(data_out)
    return predicted_bin == true_bin


def prediction_average(model, probabilities: ArrayLike) -> np.ndarray:
    """Compute the average predicted transmit time.

    Output has shape
    """
    # Weight each bin by it's predicted probability and take mean.
    # Note: predict_probabilities normalizes input etc.
    bins = ttp_bin_centers(model)
    average_transmit_times = (bins * probabilities).sum(axis=1)
    return average_transmit_times


def predict_probabilities(model, data_in: ArrayLike,
                          workers: Optional[int] = None) -> np.ndarray:
    """Get model predictions and use softmax to get probabilities."""
    import torch
    if workers is not None:
        torch.set_num_threads(workers)

    # Prepare model and input data.
    model.set_model_eval()
    data_in = model.normalize_input(np.array(data_in), update_obs=False)

    with torch.no_grad():
        prediction = model.model(torch.from_numpy(data_in))
        # Each row is one probability distribution
        # (if we apply softmax to the output) -> along axis 1
        distributions = torch.softmax(prediction, 1).numpy()

    return distributions


def predict_logscores(model,
                      data_in: ArrayLike, data_out: ArrayLike,
                      workers: Optional[int] = None):
    """Compute the logscore of the prediction."""
    predicted_probabilities = predict_probabilities(
        model, data_in, workers=workers)
    correct_bin = model.discretize_output(data_out)
    # For each row (index), select the probability of the correct bin.
    _index = np.arange(len(predicted_probabilities))
    correct_probabilities = predicted_probabilities[_index, correct_bin]
    return np.log(correct_probabilities)


def predict_loss(model,
                 data_in: ArrayLike, data_out: ArrayLike,
                 workers: Optional[int] = None) -> np.ndarray:
    """Predict the loss (for the whole data)."""
    return predict_losses(model, data_in, data_out, workers=workers).mean()


def predict_losses(model,
                   data_in: ArrayLike, data_out: ArrayLike,
                   workers: Optional[int] = None) -> np.ndarray:
    """Predict the loss (for each item separately)."""
    import torch
    if workers is not None:
        torch.set_num_threads(workers)

    # Prepare model and data.
    model.set_model_eval()
    data_in = torch.from_numpy(
        model.normalize_input(np.array(data_in), update_obs=False))
    data_out = torch.from_numpy(model.discretize_output(np.array(data_out)))

    with torch.no_grad():
        prediction = model.model(data_in)
        return model.loss_fn(prediction, data_out).numpy()


# Helpers to use model.
# =====================

def ttp_discretize(values, model=None) -> np.ndarray:
    """Discretize output using provided model or default."""
    if model is None:
        from .src_puffer import ttp  # pylint: disable=import-outside-toplevel
        model = ttp.Model()
    return model.discretize_output(values)


def ttp_bin_centers(model=None) -> np.ndarray:
    """Compute the centers of discrete ttp output bins.

    There is an unusual discretization going on: the first bin is smaller.
    """
    if model is None:
        from .src_puffer import ttp  # pylint: disable=import-outside-toplevel
        model = ttp.Model
    n_bins = model.DIM_OUT
    bin_size = float(model.BIN_SIZE)
    bins = np.arange(0, n_bins * bin_size, bin_size)
    bins[0] = 0.25 * bin_size
    return bins


def ttp_bin_edges(model=None) -> np.ndarray:
    """Compute the right edges of the discrete ttp output bins.

    Note that this is not 100% correct as the bins have open right edges,
    but is close enough for the purpose of computing the CDF based on edges.
    """
    if model is None:
        from .src_puffer import ttp  # pylint: disable=import-outside-toplevel
        model = ttp.Model
    n_bins = model.DIM_OUT
    bin_size = float(model.BIN_SIZE)
    return np.arange(0, n_bins * bin_size, bin_size) + 0.5 * bin_size
