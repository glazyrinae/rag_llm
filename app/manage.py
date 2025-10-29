#!/usr/bin/env python

import asyncio
import typer
import click
from api.queue import queue

SCHEDULER = {
    "get_predict": {
        "period": "* * * * *",
        "function_path": "api.endpoints:get_predict",
        "kwargs": {"file_link": "gt"},
    },
}

# queue.run_worker(queue_name="default", worker_idx=0)
asyncio.run(queue.run(queue_worker_cnt={"default": 1}))  # , scheduler=SCHEDULER))

# manager = typer.Typer(pretty_exceptions_enable=False)
# #queue:queue:default
# #redis-cli
# #keys *
# #lrange queue:queue:default 0 1

# @manager.command(name="queue_run")
# #@click.command()
# #@click.argument("cmd")
# def queue_run(cmd):
#      """Запустить обработку задач в очереди"""
#      queue.run_worker(queue_name="default", worker_idx=0)
#      #asyncio.run(queue.run(queue_worker_cnt={"default": 1})) #, scheduler=SCHEDULER))


# if __name__ == "__main__":
#     manager()
