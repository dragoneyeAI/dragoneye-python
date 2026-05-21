import os
from typing import Callable, Optional

import backoff

from dragoneye.classification import Classification
from dragoneye.types.common import BASE_API_URL


class Dragoneye:
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 10,
        max_backoff_time: int = 120,
        backoff_jitter: Callable[[float], float] = backoff.full_jitter,
        base_api_url: str = BASE_API_URL,
    ):
        if api_key is None:
            api_key = os.getenv("DRAGONEYE_API_KEY")

        assert api_key is not None, (
            "API key is required - set the DRAGONEYE_API_KEY environment variable or pass it to the [Dragoneye] constructor"
        )

        self.api_key = api_key
        self.max_retries = max_retries
        self.max_backoff_time = max_backoff_time
        self.backoff_jitter = backoff_jitter
        self.base_api_url = base_api_url

        self.classification = Classification(self)
