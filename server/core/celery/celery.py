import os
import threading

from celery import Celery, signals

from core.settings import settings

app = Celery("LlamaFarm")


_folders = [
    f"{settings.lf_data_dir}/broker/in",
    f"{settings.lf_data_dir}/broker/processed",
    f"{settings.lf_data_dir}/broker/results",
]

for folder in _folders:
    os.makedirs(folder, exist_ok=True)

app.conf.update(
    {
        "broker_url": "filesystem://",
        "broker_transport_options": {
            "data_folder_in": f"{settings.lf_data_dir}/broker/in",
            "data_folder_out": f"{settings.lf_data_dir}/broker/in",  # has to be the same as 'data_folder_in'  # noqa: E501
            "data_folder_processed": f"{settings.lf_data_dir}/broker/processed",
        },
        "result_backend": f"file://{settings.lf_data_dir}/broker/results",
        "result_persistent": True,
    }
)


# Intentionally empty function to prevent Celery from overriding root logger config
@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    pass


# app.log.setup()

# Create a thread and run the worker in it. This is not a long-term solution.
# Eventually we should use a proper broker and backend for this like Redis and
# we can remove this code.


# Code to start the worker


def run_worker():
    app.worker_main(argv=["worker", "-P", "solo"])


t = threading.Thread(target=run_worker, daemon=True)

t.start()
