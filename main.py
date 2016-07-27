from __future__ import print_function

import sys
import time

import matplotlib.pyplot as plt
import pandas as pd

NVIDIA_DUMP_FILE = 'nvidia_dump.txt'

COLORS = ['g', 'm', 'c', 'r']


# nvidia-smi --query-gpu=index,timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,clocks.sm,clocks.mem --format=csv -l 1 & python main.py power.draw

def follow_mock(file_handle):
    for line in file_handle.readlines():
        yield line


def follow(file_handle):
    """
    fileObject.seek(offset[, whence])
    offset -- This is the position of the read/write pointer within the file.
    whence -- This is optional and defaults to 0 which means absolute file positioning, other values are 1 which means seek relative to the current position and 2 means seek relative to the file's end.
    """
    file_handle.seek(0, 2)
    while True:
        line = file_handle.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line


def atof(x):
    if isinstance(x, float):
        return x
    if isinstance(x, int):
        return float(x)
    if '%' in x:
        return float(x.replace('%', ''))
    if ' ' in x:
        return x.split()[0]
    return x


# index,timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,clocks.sm,clocks.mem

if __name__ == '__main__':

    arguments = sys.argv
    if len(arguments) < 2:
        print('Specify the parameter of interest: temperature.gpu, utilization.gpu, '
              'utilization.memory, memory.used, power.draw, clocks.sm, clocks.mem')
        exit(0)

    df = pd.DataFrame(columns=('index', 'timestamp', 'name', 'pci.bus_id', 'driver_version',
                               'pstate', 'pcie.link.gen.max', 'pcie.link.gen.current', 'temperature.gpu',
                               'utilization.gpu', 'utilization.memory', 'memory.total', 'memory.free',
                               'memory.used', 'power.draw', 'clocks.sm', 'clocks.mem'))
    cuda_log_file = open(NVIDIA_DUMP_FILE, 'r')
    log_lines = follow_mock(cuda_log_file)
    i = 0
    plt.ion()
    for line in log_lines:
        df.loc[i] = line.split(',')
        i += 1
        column_of_interest = arguments[1]
        # pre-processing
        df[column_of_interest] = df[column_of_interest].apply(atof)
        df['index'] = df['index'].apply(int)
        look_back_window = 40
        dates = df.loc[df['index'] == 0]
        timestamps = [v.split()[1].split('.')[0] for v in dates['timestamp'].values[-look_back_window:]]
        plt.xticks(range(len(timestamps)), timestamps, rotation='vertical')

        for gpu_id in range(len(set(df['index'].values))):
            plt.plot(df.loc[df['index'] == gpu_id][column_of_interest].values[-look_back_window:],
                     color=COLORS[gpu_id],
                     linewidth=2,
                     label='GPU {}'.format(gpu_id))
        plt.legend(loc=6)
        plt.margins(0.1)
        plt.subplots_adjust(bottom=0.3)
        plt.grid()
        plt.title(column_of_interest)
        plt.pause(0.02)
        plt.clf()

    while True:
        plt.pause(0.02)
