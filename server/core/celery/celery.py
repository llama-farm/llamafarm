import os
import threading

from celery import Celery, signals

from core.settings import settings

app = Celery("LlamaFarm")


def get_celery_config():
    config = {}

    redis_host = os.getenv("CELERY_REDIS_HOST")
    redis_port = os.getenv("CELERY_REDIS_PORT", "6379")
    redis_db = os.getenv("CELERY_REDIS_DB", "0")
    redis_password = os.getenv("CELERY_REDIS_PASSWORD")

    if redis_host:
        broker_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}" if redis_password else f"redis://{redis_host}:{redis_port}/{redis_db}"
        config["broker_url"] = broker_url
        config["result_backend"] = broker_url
        return config

    rabbitmq_host = os.getenv("CELERY_RABBITMQ_HOST")
    rabbitmq_port = os.getenv("CELERY_RABBITMQ_PORT", "5672")
    rabbitmq_user = os.getenv("CELERY_RABBITMQ_USER", "guest")
    rabbitmq_password = os.getenv("CELERY_RABBITMQ_PASSWORD", "guest")
    rabbitmq_vhost = os.getenv("CELERY_RABBITMQ_VHOST", "/")

    if rabbitmq_host:
        broker_url = f"amqp://{rabbitmq_user}:{rabbitmq_password}@{rabbitmq_host}:{rabbitmq_port}/{rabbitmq_vhost}"
        config["broker_url"] = broker_url

        if redis_host:
            result_backend = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}" if redis_password else f"redis://{redis_host}:{redis_port}/{redis_db}"
            config["result_backend"] = result_backend
        else:
            _folders = [
                f"{settings.lf_data_dir}/broker/in",
                f"{settings.lf_data_dir}/broker/processed",
                f"{settings.lf_data_dir}/broker/results",
            ]
            for folder in _folders:
                os.makedirs(folder, exist_ok=True)

            config["result_backend"] = f"file://{settings.lf_data_dir}/broker/results"
            config["result_persistent"] = True

        return config

    _folders = [
        f"{settings.lf_data_dir}/broker/in",
        f"{settings.lf_data_dir}/broker/processed",
        f"{settings.lf_data_dir}/broker/results",
    ]

    for folder in _folders:
        os.makedirs(folder, exist_ok=True)

    config.update(
        {
            "broker_url": "filesystem://",
            "broker_transport_options": {
                "data_folder_in": f"{settings.lf_data_dir}/broker/in",
                "data_folder_out": f"{settings.lf_data_dir}/broker/in",
                "data_folder_processed": f"{settings.lf_data_dir}/broker/processed",
            },
            "result_backend": f"file://{settings.lf_data_dir}/broker/results",
            "result_persistent": True,
        }
    )

    return config

celery_config = get_celery_config()
app.conf.update(celery_config)

@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    pass

def run_worker():
    app.worker_main(argv=["worker", "-P", "solo", "--uid", "0"])

t = threading.Thread(target=run_worker, daemon=True)

t.start()
