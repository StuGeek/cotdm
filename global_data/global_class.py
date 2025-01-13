import threading
from threading import Thread


class ModelStage:
    def __init__(self, stage, type):
        self.stage = stage
        self.type = type


class ModelNameAndSlo:
    def __init__(self, name, slo):
        self.name = name
        self.slo = slo

    def __hash__(self):
        return hash((self.name, self.slo))

    def __eq__(self, other):
        return self.name == other.name and self.slo == other.slo
    
    def __str__(self):
        return "{{name: {}, slo: {}}}".format(self.name, self.slo)
 
    def __repr__(self):
        return "{{name: {}, slo: {}}}".format(self.name, self.slo)


class SingleLayerTime:
    def __init__(self, cal_time, trans_time, load_time):
        self.cal_time = cal_time
        self.trans_time = trans_time
        self.load_time = load_time


class SingleInferenceRequest:
    def __init__(self, model_name, input_text, slo=1.0):
        self.model_name = model_name
        self.input_text = input_text
        self.slo = slo


class WorkloadInferenceRequest:
    def __init__(self, inf_req_list, arri_time_list):
        self.inf_req_list = inf_req_list
        self.arri_time_list = arri_time_list


class ThreadWithReturnValue(Thread):
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return
    

class AtomicInteger:
    def __init__(self, initial=0):
        self.value = initial
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.value

    def set(self, new_value):
        with self.lock:
            self.value = new_value

    def incrementAndGet(self):
        with self.lock:
            self.value += 1
            return self.value

    def getAndIncrement(self):
        with self.lock:
            old_value = self.value
            self.value += 1
            return old_value

    def decrementAndGet(self):
        with self.lock:
            self.value -= 1
            return self.value

    def getAndDecrement(self):
        with self.lock:
            old_value = self.value
            self.value -= 1
            return old_value

    def addAndGet(self, delta):
        with self.lock:
            self.value += delta
            return self.value

    def getAndAdd(self, delta):
        with self.lock:
            old_value = self.value
            self.value += delta
            return old_value


# class ConvertItem:
#     def __init__(self, from_model_idx, to_model_idx, begin_idx, end_idx, convert_index):
#         self.from_model_idx = from_model_idx
#         self.to_model_idx = to_model_idx
#         self.begin_idx = begin_idx
#         self.end_idx = end_idx
#         self.convert_index = convert_index
