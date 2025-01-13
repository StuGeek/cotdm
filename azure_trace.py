import math
import os.path
import csv
import pickle
import time
import copy
import warnings
from typing import List, Dict
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma, pareto
import numpy as np
from global_data.global_class import SingleInferenceRequest
from workload import Workload, PoissonProcess, GammaProcess

DEBUG = False

def preprocess_azure_v1_trace(trace_dir, n_day=14):
    if not os.path.exists(trace_dir):
        raise RuntimeError(f"{trace_dir}")
    tracelines = OrderedDict()
    print(f"Reading azure v1 trace in 14 days; it might take a while...")
    tic = time.time()
    for i in range(1, n_day + 1):
        day_str = str(i) if i >= 10 else "0" + str(i)
        filename = os.path.join(
            trace_dir, f"invocations_per_function_md.anon.d{day_str}.csv"
        )
        print(f"Read file: {filename}")
        with open(filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                function_name = row["HashFunction"]
                histogram_1min = np.array(
                    [int(row[str(j)]) for j in range(1, 1441)], dtype=np.int32
                )
                if i == 1:
                    assert function_name not in tracelines
                    tracelines[function_name] = histogram_1min
                else:
                    expected_size = 1440 * (i - 1)
                    if function_name in tracelines:
                        cur_size = tracelines[function_name].size
                        if cur_size != expected_size:
                            diff = expected_size - cur_size
                            assert diff % 1440 == 0
                            tracelines[function_name] = np.concatenate(
                                (
                                    tracelines[function_name],
                                    np.zeros((diff,), dtype=np.int32),
                                    histogram_1min,
                                )
                            )
                        else:
                            tracelines[function_name] = np.concatenate(
                                (tracelines[function_name], histogram_1min)
                            )
                    else:
                        tracelines[function_name] = np.concatenate(
                            (np.zeros((expected_size,), dtype=np.int32), histogram_1min)
                        )
    for function_name, histogram_1min in tracelines.items():
        if histogram_1min.size != n_day * 1440:
            diff = n_day * 1440 - histogram_1min.size
            assert diff % 1440 == 0
            tracelines[function_name] = np.concatenate(
                (tracelines[function_name], np.zeros((diff,), dtype=np.int32))
            )
    print(f"Reading takes: {time.time() - tic}s.")

    # report the stats.
    num_function_invocations = []
    for function_name, histogram_1min in tracelines.items():
        assert histogram_1min.size == 1440 * n_day, f"length: {histogram_1min.size}"
        num_function_invocations.append(np.sum(histogram_1min))
    num_functions = len(tracelines.keys())
    print(
        f"Azure trace v1, stats: #days: {n_day}, #functions: {num_functions}, "
        f"total invocations: {sum(num_function_invocations)}, "
        f"max: {max(num_function_invocations)}, min: {min(num_function_invocations)}, "
        f"avg: {np.mean(num_function_invocations):.2f}"
    )

    # pickle it to disk
    save_path = os.path.join(trace_dir, "azure_v1.pkl")
    with open(save_path, "wb") as handle:
        pickle.dump(tracelines, handle)
    print(
        f"Dump the data into {save_path}, file size: {os.path.getsize(save_path) // 1e6} MB."
    )


def preprocess_azure_v1_trace_sparse(trace_dir, n_day=1, max_invo_num=1):
    if not os.path.exists(trace_dir):
        raise RuntimeError(f"{trace_dir}")
    tracelines = OrderedDict()
    # print(f"Reading azure v1 trace in 14 days; it might take a while...")
    print(f"Reading azure v1 trace in 1 days; it might take a while...")
    tic = time.time()
    for i in range(1, n_day + 1):
        day_str = str(i) if i >= 10 else "0" + str(i)
        filename = os.path.join(
            trace_dir, f"invocations_per_function_md.anon.d{day_str}.csv"
        )
        print(f"Read file: {filename}")
        with open(filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                function_name = row["HashFunction"]
                histogram_1min = np.array(
                    [int(row[str(j)]) for j in range(1, 1441)], dtype=np.int32
                )
                if i == 1:
                    assert function_name not in tracelines
                    tracelines[function_name] = histogram_1min
                else:
                    expected_size = 1440 * (i - 1)
                    if function_name in tracelines:
                        cur_size = tracelines[function_name].size
                        if cur_size != expected_size:
                            diff = expected_size - cur_size
                            assert diff % 1440 == 0
                            tracelines[function_name] = np.concatenate(
                                (
                                    tracelines[function_name],
                                    np.zeros((diff,), dtype=np.int32),
                                    histogram_1min,
                                )
                            )
                        else:
                            tracelines[function_name] = np.concatenate(
                                (tracelines[function_name], histogram_1min)
                            )
                    else:
                        tracelines[function_name] = np.concatenate(
                            (np.zeros((expected_size,), dtype=np.int32), histogram_1min)
                        )

    for function_name, histogram_1min in tracelines.items():
        if histogram_1min.size != n_day * 1440:
            diff = n_day * 1440 - histogram_1min.size
            assert diff % 1440 == 0
            tracelines[function_name] = np.concatenate(
                (tracelines[function_name], np.zeros((diff,), dtype=np.int32))
            )
    print(f"Reading takes: {time.time() - tic}s.")

    # report the stats.
    del_key_list = []
    num_function_invocations = []
    for function_name, histogram_1min in tracelines.items():
        assert histogram_1min.size == 1440 * n_day, f"length: {histogram_1min.size}"
        invo_num = np.sum(histogram_1min)
        if invo_num != 0 and invo_num <= max_invo_num:
            num_function_invocations.append(invo_num)
        else:
            del_key_list.append(function_name)
    for key in del_key_list:
        del tracelines[key]

    num_functions = len(tracelines.keys())
    print(
        f"Azure trace v1, stats: #days: {n_day}, #functions: {num_functions}, "
        f"total invocations: {sum(num_function_invocations)}, "
        f"max: {max(num_function_invocations)}, min: {min(num_function_invocations)}, "
        f"avg: {np.mean(num_function_invocations):.2f}"
    )

    # pickle it to disk
    save_path = os.path.join(trace_dir, "azure_v1.pkl")
    with open(save_path, "wb") as handle:
        pickle.dump(tracelines, handle)
    print(
        f"Dump the data into {save_path}, file size: {os.path.getsize(save_path) // 1e6} MB."
    )


def load_trace(path: str) -> OrderedDict:
    assert path.endswith(".pkl")
    tic = time.time()
    with open(path, "rb") as handle:
        tracelines = pickle.load(handle)
    # print(f"Reading takes: {time.time() - tic}s.")

    # Do some check and report stats:
    num_functions = len(tracelines.keys())
    num_function_invocations = []
    for function_name, trace in tracelines.items():
        if trace.dtype == np.int32:
            num_function_invocations.append(np.sum(trace))
        else:
            num_function_invocations.append(trace.size)
    if DEBUG:
        print(
            f"Trace: {path[:-4]}, stats: #days: 14, #functions: {num_functions}, "
            f"total invocations: {sum(num_function_invocations)}, "
            f"max: {max(num_function_invocations)}, min: {min(num_function_invocations)}, "
            f"avg: {np.mean(num_functions):.2f}"
        )
    return tracelines


class TraceReplay:
    def __init__(
        self,
        model,
        arrivals,
        trace_name,
        start_time,
        end_time,
        interval_seconds,
        arrival_distribution,
        arrival_distribution_params=None,
        rate_scale_factor=1.0,
        cv_scale_factor=1.0,
        time_scale_factor=1.0,
        replication_factor=1,
    ):
        """A TraceReplay specifies the traffic arrival pattern of a model."""
        self.model = model
        self.arrivals = arrivals

        # other generation-time information
        self.trace_name = trace_name
        self.start_time = start_time
        self.end_time = end_time
        self.arrival_distribution = arrival_distribution
        self.interval_seconds = interval_seconds
        self.arrival_distribution_params = arrival_distribution_params

        # scale factors
        self.rate_scale_factor = rate_scale_factor
        self.cv_scale_factor = cv_scale_factor
        self.time_scale_factor = time_scale_factor
        self.replication_factor = replication_factor

        # stats
        if len(self.arrivals) > 1:
            self._rate = len(self.arrivals) / (
                (self.end_seconds - self.start_seconds) // self.time_scale_factor
            )
            intervals = self.arrivals[1:] - self.arrivals[:-1]
            self._cv = np.std(intervals) / (np.mean(intervals) + 1e-5)
        else:
            self._rate = 0
            self._cv = 0

    def to_workload(self, slo: float = 1.0):
        return Workload(
            self.arrivals,
            [SingleInferenceRequest(self.model, None, slo) for i in range(len(self.arrivals))],
        )

    def rate(self):
        return self._rate

    def cv(self):
        return self._cv

    @property
    def duration_seconds(self):
        duration_seconds = self.end_seconds - self.start_seconds
        return duration_seconds

    @property
    def duration(self):
        duration_mins = self.duration_seconds // 60
        duration_remained_seconds = self.duration_seconds % 60
        duration_hours = duration_mins // 60
        duration_remained_mins = duration_mins % 60
        duration_day = duration_hours // 24
        duration_remained_hours = duration_hours % 24
        return (
            duration_day,
            duration_remained_hours,
            duration_remained_mins,
            duration_remained_seconds,
        )

    @property
    def start_seconds(self):
        start_d, start_h, start_m = Trace.timestr_to_dhm(self.start_time)
        start_timestamp_seconds = (
            start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        )
        return start_timestamp_seconds

    @property
    def end_seconds(self):
        end_d, end_h, end_m = Trace.timestr_to_dhm(self.end_time)
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60
        return end_timestamp_seconds


class Trace:
    def __init__(self, trace_name, trace_dir):
        self.trace_name: str = trace_name
        self.trace_dir: str = trace_dir
        self.have_timestamp: bool = False
        self.function_histogram = None
        self.n_day = 14

        if trace_name == "azure_v1":
            self.function_histogram = load_trace(trace_dir)
        else:
            raise RuntimeError("Choose trace `azure_v1`")

    @staticmethod
    def timestr_to_dhm(time_str):
        dhm = time_str.split(sep=".")
        if len(dhm) != 3:
            raise RuntimeError("Wrong format for `start_time`.")
        day = int(dhm[0])
        hour = int(dhm[1])
        min = int(dhm[2])
        return day, hour, min

    def slice(self, start_time: str = "0.0.0", end_time: str = "13.23.60"):
        """Slice the trace given start time string and end_time string."""
        start_d, start_h, start_m = self.timestr_to_dhm(start_time)
        end_d, end_h, end_m = self.timestr_to_dhm(end_time)
        if start_d >= self.n_day or end_d >= self.n_day or start_d > end_d:
            raise RuntimeError("start day or end day must be within the trace range.")
        if start_h >= 24 or end_h >= 24:
            raise RuntimeError("start hour or end hour must be < 24.")
        if start_m > 60 or end_m > 60:
            raise RuntimeError("start min or end minute must be <= 60.")
        if self.trace_name == "azure_v1":
            ret = self.slice_histogram(start_d, start_h, start_m, end_d, end_h, end_m)
        else:
            raise NotImplementedError()
        return ret

    def slice_histogram(self, start_d, start_h, start_m, end_d, end_h, end_m):
        """Slice the histogram."""
        assert self.function_histogram is not None
        start_slot = start_d * 24 * 60 + start_h * 60 + start_m
        end_slot = end_d * 24 * 60 + end_h * 60 + end_m
        sliced_histogram = OrderedDict()
        for function_name, histogram in self.function_histogram.items():
            sliced_histogram[function_name] = histogram[start_slot:end_slot]
        
        return sliced_histogram

    def replay(
        self,
        models: List[str],
        model_mapping_strategy: str = "stripe",
        start_time: str = "0.0.0",
        end_time: str = "13.23.60",
        arrival_distribution: str = "exponential",
        interval_seconds: int = 600,
        rate_scale_factor: float = 1.0,
        cv_scale_factor: float = 1.0,
        time_scale_factor: float = 1.0,
        replication_factor: int = 1,
        seed: int = 0,
    ) -> Dict[str, TraceReplay]:
        """Return a workload that replays a given slice of the trace.

        The method replays the trace by mapping functions in the trace to models provided by
        the input `models`.

        Args:
            models (List[str]): a list of model names.
            model_mapping_strategy (str): `round_robin` or `stripe`.
            start_time (str): in the form of `{day}.{hour}.{minu7te}`.
            end_time (str): in the form of `{day}.{hour}.{minute}`.
            arrival_distribution (str): `vanilla`, `exponential`, or `gamma`.
            interval_seconds (int): the length of the interval in seconds to estimate a generation process.
            rate_scale_factor (float): scale the estimated rate give this factor.
            cv_scale_factor (float): scale the cv given this factor. Only works when distribution = `gamma`.
            time_scale_factor (float): downscale the time, e.g., when it is 2,
                a 1-hour trace will be used as if it were 30 mins.
            replication_factor (int): simply replicate each arrival given a factor.
            seed (int): random seed for the generation process.

        Returns:
            replays (Dict[str, TraceReplay]): the TraceReplay for each model.
        """
        # Do some checks
        if replication_factor < 1:
            warnings.warn(
                "`replication factor` should not be less than 1. Reset it to 1."
            )
        if replication_factor > 1:
            if not (
                self.trace_name == "azure_v2" and arrival_distribution == "vanilla"
            ):
                raise RuntimeError(
                    f"We can only replicate vanilla azure v2 trace, "
                    f"got: {self.trace_name}, {arrival_distribution}"
                )
        if time_scale_factor != 1.0:
            if self.trace_name != "azure_v2":
                raise RuntimeError("Cannot do time-scaling on azure_v1.")
            if arrival_distribution != "vanilla":
                raise RuntimeError("Can only do time-scaling on vanilla distributions.")
        if arrival_distribution != "gamma" and cv_scale_factor != 1.0:
            raise RuntimeError("No CV for exponential distributions.")
        if time_scale_factor != 1.0 and (
            rate_scale_factor != 1.0 or cv_scale_factor != 1.0
        ):
            raise RuntimeError("Choose one: scale rate/cv, or scale time.")

        replays = OrderedDict()
        start_d, start_h, start_m = self.timestr_to_dhm(start_time)
        end_d, end_h, end_m = self.timestr_to_dhm(end_time)
        start_timestamp_seconds = (
            start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        )
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60

        if self.trace_name == "azure_v1":
            # Trace are 1-min histograms
            # 1. Convert function trace to model trace
            model_histogram = OrderedDict()
            function_histogram = self.slice(start_time, end_time)
            # filter out all functions that have zero arrivals:
            functions_to_remove = [
                f for f in function_histogram if np.sum(function_histogram[f]) == 0
            ]
            for f in functions_to_remove:
                del function_histogram[f]

            # generate function model mapping.
            function_model_mapping = self.map_model(
                models, function_histogram.keys(), model_mapping_strategy
            )
            for f, m in function_model_mapping.items():
                if m not in model_histogram:
                    model_histogram[m] = copy.deepcopy(function_histogram[f])
                else:
                    model_histogram[m] += function_histogram[f]

            # 2. re-histogram based on `interval_seconds`
            histogram_dataset = OrderedDict()
            assert (
                interval_seconds % 60 == 0
            ), "Please set `interval_seconds` as a multiple of 60"
            n_min_per_interval = interval_seconds // 60
            for model, histogram in model_histogram.items():
                n_total_min = histogram.size
                n_interval = (
                    n_total_min + n_min_per_interval - 1
                ) // n_min_per_interval
                accumulated = np.zeros((n_interval,), dtype=np.int32)
                for i in range(accumulated.size):
                    start = i * n_min_per_interval
                    end = (
                        (i + 1) * n_min_per_interval
                        if (i + 1) * n_min_per_interval <= n_total_min
                        else n_total_min
                    )
                    accumulated[i] = np.sum(histogram[start:end])
                histogram_dataset[model] = accumulated

            # Estimate distribution parameters with histogram dataset
            distributions = self.estimate_parameters_with_histogram(
                histogram_dataset,
                interval_seconds,
                arrival_distribution,
                rate_scale_factor,
                cv_scale_factor,
            )
        else:
            raise NotImplementedError("Other trace ")

        # Sample from the distributions and generate the arrivals
        for m in distributions:
            arrivals = []
            arrival_distribution_params = []
            for i, distribution in enumerate(distributions[m]):
                if distribution is None:
                    arrival_distribution_params.append(None)
                    continue
                start = i * interval_seconds + start_timestamp_seconds
                arrivals.extend(
                    # distribution.generate_arrivals(start, interval_seconds, seed)
                    distribution.generate_arrivals(start, interval_seconds, seed)
                )
                # if DEBUG:
                #     arrivals.extend(distribution.generate_arrivals(0, 1.0e9, seed))
                #     self.visualize_inter_arrival(np.array(arrivals), "test")
                arrival_distribution_params.append(distribution.params())
                seed += 1
            replays[m] = TraceReplay(
                m,
                np.array(arrivals),
                self.trace_name,
                start_time,
                end_time,
                interval_seconds,
                arrival_distribution,
                arrival_distribution_params=arrival_distribution_params,
                rate_scale_factor=rate_scale_factor,
                cv_scale_factor=cv_scale_factor,
                time_scale_factor=time_scale_factor,
            )

        return replays

        # sort models
        # keys = list(replays.keys())
        # num_models = len(models)
        # indices = list(range(num_models))
        # indices.sort(key=lambda i: -len(replays[keys[i]].arrivals))

        # new_replay = OrderedDict()
        # for i in range(num_models):
        #     new_replay[models[i]] = replays[keys[indices[i]]]
        #     new_replay[models[i]].model = models[i]

        # return new_replay

    def replay_by_seed(
        self,
        models: List[str],
        model_mapping_strategy: str = "stripe",
        start_time: str = "0.0.0",
        end_time: str = "13.23.60",
        arrival_distribution: str = "exponential",
        interval_seconds: int = 600,
        rate_scale_factor: float = 1.0,
        cv_scale_factor: float = 1.0,
        time_scale_factor: float = 1.0,
        replication_factor: int = 1,
        seed: int = 0,
    ) -> Dict[str, TraceReplay]:
        replays = OrderedDict()
        start_d, start_h, start_m = self.timestr_to_dhm(start_time)
        end_d, end_h, end_m = self.timestr_to_dhm(end_time)
        start_timestamp_seconds = (
            start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        )
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60

        if self.trace_name == "azure_v1":
            # Trace are 1-min histograms
            # 1. Convert function trace to model trace
            model_histogram = OrderedDict()
            function_histogram = self.slice(start_time, end_time)
            # filter out all functions that have zero arrivals:
            functions_to_remove = [
                f for f in function_histogram if np.sum(function_histogram[f]) == 0
            ]
            for f in functions_to_remove:
                del function_histogram[f]

            # generate function model mapping.
            function_model_mapping = self.map_model(
                models, function_histogram.keys(), model_mapping_strategy
            )
            for f, m in function_model_mapping.items():
                if m not in model_histogram:
                    model_histogram[m] = copy.deepcopy(function_histogram[f])
                else:
                    model_histogram[m] += function_histogram[f]

            # 2. re-histogram based on `interval_seconds`
            histogram_dataset = OrderedDict()
            assert (
                interval_seconds % 60 == 0
            ), "Please set `interval_seconds` as a multiple of 60"
            n_min_per_interval = interval_seconds // 60
            for model, histogram in model_histogram.items():
                n_total_min = histogram.size
                n_interval = (
                    n_total_min + n_min_per_interval - 1
                ) // n_min_per_interval
                accumulated = np.zeros((n_interval,), dtype=np.int32)
                for i in range(accumulated.size):
                    start = i * n_min_per_interval
                    end = (
                        (i + 1) * n_min_per_interval
                        if (i + 1) * n_min_per_interval <= n_total_min
                        else n_total_min
                    )
                    accumulated[i] = np.sum(histogram[start:end])
                histogram_dataset[model] = accumulated

            # Estimate distribution parameters with histogram dataset
            distributions = self.estimate_parameters_with_histogram(
                histogram_dataset,
                interval_seconds,
                arrival_distribution,
                rate_scale_factor,
                cv_scale_factor,
            )
        else:
            raise NotImplementedError("Other trace ")

        if self.trace_name == "azure_v1":
            total_h = 0
            for histogram in histogram_dataset.values():
                total_h += sum(histogram)

        arri_rate = total_h / interval_seconds * rate_scale_factor
        arri_cv = cv_scale_factor
        print(total_h, total_h / interval_seconds, arri_rate, arri_cv)
        model_name_list = list(distributions.keys())
        model_name_list_len = len(model_name_list)
        seed = 0

        while True:
            for i in range(model_name_list_len):
                m = model_name_list[i]
                arrivals = []
                arrival_distribution_params = []
                for j, distribution in enumerate(distributions[m]):
                    if distribution is None:
                        arrival_distribution_params.append(None)
                        continue
                    start = j * interval_seconds + start_timestamp_seconds
                    arrivals.extend(
                        distribution.generate_distribution_arrivals(start, interval_seconds, seed + j)
                        # distribution.generate_arrivals_not_random(start, interval_seconds)
                    )
                    # if DEBUG:
                    #     arrivals.extend(distribution.generate_arrivals(0, 1.0e9, seed))
                    #     self.visualize_inter_arrival(np.array(arrivals), "test")
                    arrival_distribution_params.append(distribution.params())
                replays[m] = TraceReplay(
                    m,
                    np.array(arrivals),
                    self.trace_name,
                    start_time,
                    end_time,
                    interval_seconds,
                    arrival_distribution,
                    arrival_distribution_params=arrival_distribution_params,
                    rate_scale_factor=rate_scale_factor,
                    cv_scale_factor=cv_scale_factor,
                    time_scale_factor=time_scale_factor,
                )
                seed += 1
            
            replay_workload = replays[model_name_list[0]].to_workload()
            for i in range(1, model_name_list_len):
                replay_workload += replays[model_name_list[i]].to_workload()
            
            if abs(replay_workload.rate - arri_rate) < 1e-3 and abs(replay_workload.cv - arri_cv) < 1e-3:
                break
        
        print(replay_workload, arri_rate, arri_cv, "seed={:d}".format(seed))
        return replays

        # sort models
        # keys = list(replays.keys())
        # num_models = len(models)
        # indices = list(range(num_models))
        # indices.sort(key=lambda i: -len(replays[keys[i]].arrivals))

        # new_replay = OrderedDict()
        # for i in range(num_models):
        #     new_replay[models[i]] = replays[keys[indices[i]]]
        #     new_replay[models[i]].model = models[i]

        # return new_replay

    def map_model(self, models, function_names, strategy="stripe"):
        mapping = OrderedDict()
        n_model = len(models)
        n_function = len(function_names)
        assert n_function >= n_model, f"#function {n_function} < #models {n_model}"
        if strategy not in ["round_robin", "stripe"]:
            raise NotImplementedError(f"Unimplemented strategy: {strategy}")
        for i, f in enumerate(function_names):
             
            if strategy == "round_robin":
                mapping[f] = models[n_model * i // n_function]
            else:
                mapping[f] = models[i % n_model]
        return mapping

    def estimate_parameters_with_histogram(
        self,
        dataset,
        interval_seconds,
        arrival_distribution="exponential",
        rate_scale_factor=1.0,
        cv_scale_factor=1.0,
    ):
        if arrival_distribution not in ["exponential", "gamma"]:
            raise NotImplementedError(
                f"We can only use histogram data for exponential or gamma distribution, "
                f"got {arrival_distribution}"
            )
        distributions = OrderedDict()
        for model, histogram in dataset.items():
            distributions[model] = []
            for h in histogram:
                if h == 0:
                    distributions[model].append(None)
                else:
                    arrival_rate = h / interval_seconds
                    arrival_rate = arrival_rate * rate_scale_factor
                    if arrival_distribution == "exponential":
                        distributions[model].append(PoissonProcess(arrival_rate))
                    else:
                        distributions[model].append(
                            GammaProcess(arrival_rate, cv_scale_factor)
                        )
        return distributions

    @staticmethod
    def visualize_inter_arrival(inter_arrival, name, n_interval=300):
        count, bins, _ = plt.hist(inter_arrival, bins=np.linspace(0, 300, n_interval))
        plt.show()
        plt.ylabel("#reqs")
        plt.xlabel("#seconds")
        fig = plt.gcf()
        figure_size = (8, 4)
        fig.set_size_inches(figure_size)
        fig.savefig(f"plots/{name}.png", bbox_inches="tight")
        plt.close()

    @staticmethod
    def estimate_exponential(inter_arrivals):
        """Take inter-arrivals and return the rate parameters."""
        _, scale = expon.fit(inter_arrivals, floc=0)
        return 1.0 / scale

    @staticmethod
    def estimate_gamma(inter_arrivals):
        shape, _, scale = gamma.fit(inter_arrivals, floc=0)
        cv = math.sqrt(1.0 / shape)
        arrival_rate = 1.0 / (shape * scale)
        return arrival_rate, cv

    @staticmethod
    def estimate_pareto(inter_arrivals):
        shape, loc, scale = pareto.fit(inter_arrivals, floc=0.0, fscale=1.0)
        return shape, scale, loc

    @property
    def function_names(self):
        if self.trace_name == "azure_v1":
            return list(self.function_histogram.keys())
        else:
            raise NotImplementedError()

if __name__ == "__main__":
    from global_data import global_config

    if not os.path.exists(global_config.AZURE_V1_DIR):
        preprocess_azure_v1_trace_sparse(os.path.dirname(global_config.AZURE_V1_NAME), n_day=1)

    model_name_list = ["model 1", "model 2", "model 3", "model 4"]

    # 1.775 1.893 2.011 2.13 2.248
    # gen_rate_list = [1.77, 1.89, 2.01, 2.13, 2.25]
    gen_rate_scale_list = [0.15, 0.16, 0.17, 0.18, 0.19, 0.17, 0.17, 0.17, 0.17]
    gen_cv_scale_list = [1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 1.5, 1.9, 2.3]
    gen_seed_list = [29224, 37620, 680, 204380, 92836, 139100, 42988, 3179748, 2770340]
    gen_seed_list = [i - 4 for i in gen_seed_list]

    trace_azure_v1 = Trace(global_config.AZURE_V1_NAME, global_config.AZURE_V1_DIR)

    for i in range(len(gen_rate_scale_list)):
        replays = trace_azure_v1.replay(model_name_list,
                        model_mapping_strategy="stripe",
                        start_time="0.0.0",
                        end_time="0.0.1",
                        arrival_distribution="gamma",
                        interval_seconds=60,
                        rate_scale_factor = gen_rate_scale_list[i],
                        cv_scale_factor = gen_cv_scale_list[i],
                        seed=gen_seed_list[i])
        replay_workload = replays[model_name_list[0]].to_workload()
        model_name_list_len = len(model_name_list)
        for i in range(1, model_name_list_len):
            temp_workload = replays[model_name_list[i]].to_workload()
            replay_workload += temp_workload
        print(replay_workload)
