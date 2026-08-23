"""worker 进程装配：连接 Temporal（namespace: genpipe）并注册本仓全部 activities。

genpipe workflow 按 activity 归属把任务派到本仓专属 task queue；重试/心跳/取消/
背压沿用 Temporal activity 原生语义，不引入服务间 HTTP 调用（对齐文档 §3.1）。
"""

from __future__ import annotations

import asyncio
import os

from temporalio.client import Client
from temporalio.worker import Worker

from render3d_worker.activities import ACTIVITY_REGISTRY

GENPIPE_NAMESPACE = "genpipe"
RENDER3D_TASK_QUEUE = "render3d-activities"


async def run_worker(temporal_address: str) -> None:
    client = await Client.connect(temporal_address, namespace=GENPIPE_NAMESPACE)
    worker = Worker(
        client,
        task_queue=RENDER3D_TASK_QUEUE,
        activities=list(ACTIVITY_REGISTRY.values()),
    )
    await worker.run()


def main() -> None:
    asyncio.run(run_worker(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")))
