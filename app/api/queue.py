from redis import ConnectionPool
from lib.queue import QueueManager

queue = QueueManager(redis_pool=ConnectionPool.from_url("redis://redis"))
