import time


_global_times = []


def print_time(time, *, name=None):
    h = int(time // 3600)
    time -= h * 3600
    m = int((time) // 60)
    time -= m * 60
    s = time
    if h > 0:
        s = round(s)
        time_fmt = "{0:d}h:{1:02d}m:{2:02d}s"
    elif m > 0:
        s = round(s)
        time_fmt = "{1:d}m:{2:02d}s"
    else:
        time_fmt = "{2:.6f} seconds"

    if name is None:
        name_fmt = "done in {1}"
    else:
        name_fmt = "{0} done in {1}"

    print(name_fmt.format(name, time_fmt.format(h, m, s)))


def chrono(fun):
    def wrapper(*args, **kwargs):
        start(fun.__name__)
        res = fun(*args, **kwargs)
        stop()
        return res

    return wrapper


def start(chunck_name=None):
    if chunck_name is not None:
        print("Start of", chunck_name)
    _global_times.append([chunck_name, time.time()])


def stop():
    end = time.time()
    chunck_name, start = _global_times.pop()
    print_time(end - start, name=chunck_name)
