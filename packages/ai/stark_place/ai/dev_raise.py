import inspect
import logging
import os

logger = logging.getLogger(__name__)


def dev_raise(exc: Exception | str, cause: Exception | None = None):

    if isinstance(exc, str):
        exc = Exception(exc)

    frame = inspect.stack()[1]
    func = frame.function
    cls = frame.frame.f_locals.get("self", None)
    cls_name = cls.__class__.__name__ if cls else ""
    caller = f"{cls_name}.{func}" if cls else func
    error_msg = f"at {caller} -> {exc}. "
    if cause:
        error_msg += f"Cause: {cause}. "
    logger.error(error_msg, stacklevel=2)

    if os.environ.get("ENV") in {"DEV", "DEVELOPMENT", "TEST", "TESTS", "TESTING"}:
        if cause:
            raise exc from cause
        else:
            raise exc
