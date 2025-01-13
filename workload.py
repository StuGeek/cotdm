import numpy as np
from typing import List, Optional
from global_data.global_class import SingleInferenceRequest
from global_data.global_config import DEFAULT_INPUT_TEXT_BERT, DEFAULT_INPUT_TEXT_GPT2

eps = 1e-6

class Workload:
    def __init__(self, arrivals: List[float], requests: List[SingleInferenceRequest]):
        assert len(arrivals) == len(requests)

        self.arrivals = np.array(arrivals)
        self.requests = requests

        self.enable_simulator_cache = False
        self.cached_data = None

        if len(self.arrivals) > 1:
            intervals = self.arrivals[1:] - self.arrivals[:-1]
            self.rate = 1 / (np.mean(intervals) + eps)
            self.cv = np.std(intervals) * self.rate
        else:
            self.rate = 0
            self.cv = 0

    def split_round_robin(self, number: int):
        rets = []
        for i in range(number):
            rets.append(self[i::number])
        return rets

    def split_time_interval(self, interval: float):
        if len(self.arrivals) < 1:
            return []

        ws = []
        start_i = 0
        start_time = self.arrivals[start_i]
        for i in range(len(self.arrivals)):
            if self.arrivals[i] > start_time + interval:
                ws.append(self[start_i:i])
                start_i = i
                start_time = self.arrivals[i]

        ws.append(self[start_i:])
        return ws

    @classmethod
    def empty(cls):
        return cls([], [])

    def __getitem__(self, key):
        if isinstance(key, slice):
            arrivals = self.arrivals.__getitem__(key)
            requests = self.requests.__getitem__(key)
            return Workload(arrivals, requests)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        return Workload.merge(self, other)

    def __len__(self):
        return len(self.arrivals)

    def __str__(self):
        return (
            f"Workload(len={len(self)}, "
            f"rate={self.rate:.2f}, "
            f"CV={self.cv:.2f}) "
        )

    @classmethod
    def merge(cls, *args):
        if len(args) == 1:
            return args[0]

        number = sum(len(x) for x in args)

        merged_arrivals = np.concatenate(tuple(x.arrivals for x in args))
        merged_requests = sum((x.requests for x in args), [])

        sorted_indices = np.argsort(merged_arrivals)

        arrivals = [None] * number
        requests = [None] * number

        for i, j in enumerate(sorted_indices):
            arrivals[i] = merged_arrivals[j]
            requests[i] = merged_requests[j]
            requests[i].idx = i

        return cls(arrivals, requests)


class GammaProcess:
    def __init__(self, arrival_rate: float, cv: float):
        self.rate_ = arrival_rate
        self.cv_ = cv
        self.shape = 1 / (cv * cv)
        self.scale = cv * cv / arrival_rate

    def rate(self):
        return self.rate_

    def cv(self):
        return self.cv_

    def params(self):
        return self.rate(), self.cv()

    def generate_arrivals(self, start: float, duration: float, seed: int = 0):
        np.random.seed(seed)

        batch_size = max(int(self.rate_ * duration * 1.2), 1)
        intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
        pt = 0

        ticks = []
        cur = start + intervals[0]
        end = start + duration
        while cur < end:
            ticks.append(cur)

            pt += 1
            if pt >= batch_size:
                intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
                pt = 0

            cur += intervals[pt]

        arrivals = np.array(ticks)
        intervals = arrivals[1:] - arrivals[:-1]
        arri_rate = 1 / (np.mean(intervals) + eps)
        arri_cv = np.std(intervals) * arri_rate

        # print(f"Generate Workload(len={len(ticks)}, "
        #             f"rate={arri_rate:.2f}, "
        #             f"cv={arri_cv:.2f}), "
        #             f"seed={seed:d}")

        return ticks
    

    def generate_distribution_arrivals(self, start: float, duration: float, seed: int = 0):
        gen_rate = self.rate_
        # gen_cv = self.cv_ * 0.6
        gen_cv = self.cv_
        gen_shape = 1 / (gen_cv * gen_cv)
        gen_scale = gen_cv * gen_cv / gen_rate
    
        # seed = 0
        # while True:
        np.random.seed(seed)

        batch_size = max(int(gen_rate * duration * 1.2), 1)
        intervals = np.random.gamma(gen_shape, gen_scale, size=batch_size)
        pt = 0

        ticks = []
        cur = start + intervals[0]
        end = start + duration
        while cur < end:
            ticks.append(cur)

            pt += 1
            if pt >= batch_size:
                intervals = np.random.gamma(gen_shape, gen_scale, size=batch_size)
                pt = 0

            cur += intervals[pt]

        # arrivals = np.array(ticks)
        # intervals = arrivals[1:] - arrivals[:-1]
        # arri_rate = 1 / (np.mean(intervals) + eps)
        # arri_cv = np.std(intervals) * arri_rate

            # if abs(arri_rate - self.rate_) < 1e-3 and abs(arri_cv - self.cv_) < 1e-3:
            #     # print(f"Generate Workload(len={len(ticks)}, "
            #     #       f"rate={arri_rate:.2f}, "
            #     #       f"cv={arri_cv:.2f}), "
            #     #       f"seed={seed:d}")
            #     break

            # seed += 1

        return ticks

    
    def generate_arrivals_not_random(self, start: float, duration: float):
        seed = 0
        while True:
            np.random.seed(seed)

            batch_size = max(int(self.rate_ * duration * 1.2), 1)
            intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
            pt = 0

            ticks = []
            cur = start + intervals[0]
            end = start + duration
            while cur < end:
                ticks.append(cur)

                pt += 1
                if pt >= batch_size:
                    intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
                    pt = 0

                cur += intervals[pt]

            arrivals = np.array(ticks)
            intervals = arrivals[1:] - arrivals[:-1]
            arri_rate = 1 / (np.mean(intervals) + eps)
            arri_cv = np.std(intervals) * arri_rate

            if abs(arri_rate - self.rate_) < 1e-3 and abs(arri_cv - self.cv_) < 1e-3:
                print(f"Generate Workload(len={len(ticks)}, "
                      f"rate={arri_rate:.2f}, "
                      f"cv={arri_cv:.2f}), "
                      f"seed={seed:d}")
                break

            seed += 1

        return ticks


    def generate_workload(
        self,
        model_name: str,
        start: float,
        duration: float,
        slo: Optional[float] = None,
        seed: int = 0,
    ):
        ticks = self.generate_arrivals(start, duration, seed)
        print("ticks:", ticks)
        # print(len(ticks))
        return Workload(
            ticks,
            [
                SingleInferenceRequest(model_name, None, slo)
                for i in range(len(ticks))
            ],
        )

    def generate_workload_by_model_name_list(
        self,
        model_name_list: list,
        start: float,
        duration: float,
        slo: Optional[float] = None,
        seed: int = 0,
    ):
        ticks = self.generate_arrivals(start, duration, seed)
        # ticks = self.generate_arrivals_not_random(start, duration)
        # print("ticks:", ticks)
        arrivals = ticks

        req_list = []
        model_name_list_len = len(model_name_list)

        for i in range(len(arrivals)):
            model_name = model_name_list[i % model_name_list_len]
            input_text = ""
            if model_name.find("bert") != -1:
                input_text = DEFAULT_INPUT_TEXT_BERT
            elif model_name.find("gpt2") != -1:
                input_text = DEFAULT_INPUT_TEXT_GPT2

            req_list.append(SingleInferenceRequest(model_name, input_text, slo))

        return Workload(arrivals, req_list)
    

class PoissonProcess(GammaProcess):
    def __init__(self, arrival_rate: float):
        super().__init__(arrival_rate, 1)