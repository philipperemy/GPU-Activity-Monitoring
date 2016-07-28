from __future__ import print_function

import os
import sys
from subprocess import Popen
from sys import stderr
from time import sleep

import matplotlib.pyplot as plt
import pandas as pd

NVIDIA_DUMP_FILE = 'nvidia_dump.txt'

COLORS = ['g', 'm', 'c', 'r']


def run_command_and_read_output():
    command = 'nvidia-smi --query-gpu=index,timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,' \
              'pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,' \
              'memory.free,memory.used,power.draw,clocks.sm,clocks.mem --format=csv -f {}'.format(NVIDIA_DUMP_FILE)
    p = Popen(command, stdout=stderr, stderr=open(os.devnull, 'w'), shell=True)
    p.wait()
    assert not p.returncode, 'ERROR: Call to stanford_ner exited with a non-zero code status.'
    output = []
    sleep(1)
    with open(NVIDIA_DUMP_FILE, 'r') as f:
        output.extend(f.readlines()[1:])  # no headers
    os.remove(NVIDIA_DUMP_FILE)
    gpu_count = len(output)
    return output, gpu_count


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
    i = 0
    plt.ion()
    while True:

        log_lines, number_of_gpus = run_command_and_read_output()
        for line in log_lines:

            if 'index' in line:
                continue
                # header

            # print(line.strip())  # debug
            df.loc[i] = line.split(',')
            i += 1
            column_of_interest = arguments[1]
            # pre-processing
            df[column_of_interest] = df[column_of_interest].apply(atof)
            df['index'] = pd.to_numeric(df['index'])
            look_back_window = 40
            dates = df.loc[df['index'] == 0]
            timestamps = [v.split()[1].split('.')[0] for v in dates['timestamp'].values[-look_back_window:]]
            plt.xticks(range(len(timestamps)), timestamps, rotation='vertical')
            for gpu_id in range(number_of_gpus):
                plt.plot(df.loc[df['index'] == gpu_id][column_of_interest].values[-look_back_window:],
                         color=COLORS[gpu_id],
                         linewidth=2,
                         label='GPU {}'.format(gpu_id))
            plt.legend(loc=6)
            plt.margins(0.1)
            plt.subplots_adjust(bottom=0.3)
            plt.grid()
            plt.title(column_of_interest)
            plt.pause(0.01)
            plt.clf()
