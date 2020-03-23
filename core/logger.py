
import pynvml
import os
import glob
import inspect

import tensorflow as tf
import Queue

import time
import pickle
import json
import threading

import numpy as np

if "RUN_LOG_PATH" in os.environ:
    path = os.environ["RUN_LOG_PATH"]
else:
    path = "./logs/"

if "RUN_CHECKPOINT_PATH" in os.environ:
    model_path = os.environ["RUN_CHECKPOINT_PATH"]
else:
    model_path = "./models/"

#gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

class logger:
    def __init__(self, name, construct_log, struct, restore=True, run_to_restore=-1):
        runs = glob.glob(model_path + name + "_*/")
        runs_by_numbers = {}
        run_number = 0
        for r in runs:
            number = int(r.split("/")[-2].split("_")[-1])
            runs_by_numbers[number] = r
            run_number = max(run_number, number+1)
        if run_to_restore == -1:
            if restore and run_number > 0:
                if run_to_restore == -1 or run_to_restore >= run_number:
                    run_number = run_number - 1
                else:
                    run_number = run_to_restore
                self.restore = True
            else:
                self.restore = False
        else:
            run_number = run_to_restore
            self.restore = True
        self.run_number = run_number
        self.log_path = path+name+"_"+str(run_number)+"/"
        self.model_path = model_path+name+"_"+str(run_number)+"/"
        try:
            os.makedirs(self.log_path)#, exist_ok=True)
        except OSError:
            pass
        try:
            os.makedirs(self.model_path)#, exist_ok=True)
        except OSError:
            pass

        self.tags = {}
        self.frames = Queue.Queue()
        self.current_frame = {}
        self.registered_opp = []

        self.Running_thread = False

        self.launch_saving_thread()

        self.structure_blueprint = {}
        self.structure_blueprint["MAIN"] = struct
        self.data = {}


        # pynvml.nvmlInit()
        # self.name


    def register_value(self, name, tensor):
        def log_val(val):
            self.current_frame[name] = val
            return np.zeros([1], dtype=np.float32)

        logging_op = tf.py_func(log_val, [tensor], np.float32)
        self.tags[name] = logging_op

    def register_opp(self, opp):
        if opp not in self.registered_opp:
            try:
                self.structure_blueprint["operation_"+opp.func_name]=inspect.getsource(opp)
                self.registered_opp.append(opp)
            except:
                pass


    def finalize_frame(self, input):
        def frame_handler():
            self.frames.put(self.current_frame)
            self.current_frame = {}
            return np.zeros([1], dtype=np.float32)
        with tf.control_dependencies(self.tags.values()):
            handler = tf.py_func(frame_handler, [], np.float32)
        with tf.control_dependencies([handler]):
            return tf.identity(input)

    def addData(self, **kwargs):
        for key in list(kwargs.keys()):
            self.data[key] = kwargs[key]

    def launch_saving_thread(self):
        def saver():
            while self.Running_thread:
                try:
                    frame = self.frames.get(timeout=1.0)
                    with open(self.log_path+"frame_"+str(time.time())+".pkl", "wb") as f:
                        pickle.dump(frame, f)
                except Queue.Empty:
                    pass #time.sleep(0.1)

        self.Running_thread = True
        t = threading.Thread(target=saver)
        t.daemon = True
        t.start()

    def save_header(self):
        with open(self.log_path + "header_"+str(time.time()) + ".pkl", "wb") as f:
            pickle.dump({ "structure_blueprint" : self.structure_blueprint, "data" : self.data}, f)
        with open(self.log_path + "header_"+str(time.time()) + ".json", "wb") as f:
            json.dump({ "structure_blueprint" : self.structure_blueprint}, f, indent=4)




# pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu

