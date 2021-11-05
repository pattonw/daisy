from .cl_monitor import CLMonitor
from .server import Server
from .tcp import IOLooper
from threading import Thread, Event


def run_blockwise(tasks):
    task_ids = set()
    all_tasks = []
    while len(tasks) > 0:
        task, tasks = tasks[0], tasks[1:]
        if task.task_id not in task_ids:
            task_ids.add(task.task_id)
            all_tasks.append(task)
        tasks.extend(task.upstream_tasks)

    tasks = all_tasks
    stop_event = Event()
    return_value = []

    IOLooper.clear()
    thread = Thread(target=_run_blockwise, args=(tasks, stop_event, return_value))
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        thread.join()

    return return_value[0]


def _run_blockwise(tasks, stop_event, return_value):
    server = Server(stop_event=stop_event)
    cl_monitor = CLMonitor(server)  # noqa
    result = server.run_blockwise(tasks)
    return_value.append(result)
