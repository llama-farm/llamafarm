import threading

from celery import Celery  # type: ignore

app = Celery(
    "LlamaFarm",
    broker="memory://localhost",  # in-process broker
    backend="cache+memory://localhost",  # in-process result backend
)

# Create a thread and run the worker in it. This is not a long-term solution.
# Eventually we should use a proper broker and backend for this like Redis and
# we can remove this code.


# Code to start the worker


def run_worker():
    app.worker_main(argv=["worker", "-P", "solo"])


t = threading.Thread(target=run_worker, daemon=True)

t.start()
