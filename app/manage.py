#!/usr/bin/env python

import asyncio
from api.queue import queue

asyncio.run(queue.run(queue_worker_cnt={"rag_questions": 1}))
