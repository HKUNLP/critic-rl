# Copyright (2025) critic-rl Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
from typing import Optional, Union

import numpy as np
from sandbox_fusion import (
    EvalResult,
    RunCodeRequest,
    RunCodeResponse,
    SubmitRequest,
    TestConfig,
    run_code_async,
    submit_async,
)
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from ctrl.eval.code_utils import desanitize

logger = logging.getLogger(__name__)
TIMEOUT = 320
RUN_TIMEOUT = 30
NUM_RETRIES = 5


def get_submit_fn(dataset_name: str):
    def get_run_timeout(info):
        run_timeout = info.get("time_limit", RUN_TIMEOUT)
        if np.isnan(run_timeout):
            run_timeout = RUN_TIMEOUT
        return run_timeout

    if dataset_name == "code_contests":

        def submit_fn(response, info):
            if info["test"].startswith("assert"):
                return RunCodeRequest(
                    **{
                        "code": desanitize(response).strip()
                        + "\n"
                        + "\n".join(info["test"]),
                        "language": "python",
                        "run_timeout": get_run_timeout(info),
                    }
                )

            provided_data = {
                "test": info["test"],
            }

            req = SubmitRequest(
                dataset="code_contests",
                id=0,
                config=TestConfig(
                    language="python",
                    dataset_type="CommonOJDataset",
                    provided_data=provided_data,
                    extra={"run_all_cases": True},
                    run_timeout=get_run_timeout(info),
                ),
                completion=response,
            )

            return req

    elif dataset_name == "livecodebench":

        def submit_fn(response, info):
            provided_data = {k: info[k] for k in ["id", "content", "labels", "test"]}
            provided_data
            req = SubmitRequest(
                dataset="dataset_id",
                id=0,
                config=TestConfig(
                    dataset_type="LiveCodeBenchDataset",
                    provided_data=provided_data,
                    run_timeout=get_run_timeout(info),
                ),
                completion=response,
            )

            return req

    elif dataset_name == "mbppplus":

        def submit_fn(response, info):
            if isinstance(info["test"], list):
                ut = "\n".join(info["test"])
            else:
                ut = info["test"]
            req = RunCodeRequest(
                **{
                    "code": desanitize(response).strip() + "\n" + ut,
                    "language": "python",
                    "run_timeout": get_run_timeout(info),
                }
            )

            return req

    else:
        raise NotImplementedError(f"dataset {dataset_name} not supported")
    return submit_fn


def on_retry_error(s):
    e = s.outcome.exception()
    logger.error(f"give up requesting sandbox. error: {e}")
    raise e


def before_retry_sleep(s):
    logger.warning(
        f"error requesting sandbox for {s.attempt_number} time(s), will retry... error: {s.outcome.exception()}"
    )


@retry(
    wait=wait_exponential_jitter(),
    stop=stop_after_attempt(NUM_RETRIES),
    before_sleep=before_retry_sleep,
    retry_error_callback=on_retry_error,
)
async def submit_to_sandbox(
    request: Union[RunCodeRequest, SubmitRequest],
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Union[RunCodeResponse, EvalResult]:
    fn = run_code_async if isinstance(request, RunCodeRequest) else submit_async
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)
    async with semaphore:
        resp = await fn(request, client_timeout=TIMEOUT)
    return resp
