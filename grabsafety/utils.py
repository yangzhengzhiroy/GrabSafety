import logging
import logging.config


log_config = {

    "version": 1,

    "disable_existing_loggers": False,

    "formatters": {
        "simple": {
            "format": "[%(levelname)s] %(message)s"
        },
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s : %(message)s"
        },
        "verbose": {
            "format": "%(asctime)s [%(levelname)s] %(module)s %(process)d : %(message)s"
        }
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        }
    },

    "loggers": {
        "endpoint_api": {
            "level": "ERROR",
            "handlers": ["console"],
            "propagate": "no"
        }
    },

    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}


def setup_logging(default_conf=None, default_level=logging.INFO):
    """Setup logging configuration
    """
    if default_conf:
        logging.config.dictConfig(default_conf)
    else:
        logging.basicConfig(level=default_level)
