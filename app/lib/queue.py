"""
Класс очереди QueueManager.
"""

import asyncio
import logging
import pickle
import signal
from datetime import datetime, timedelta
from typing import Any, Callable, Literal
import pytz
import croniter
import import_string
from uuid import uuid4
from redis import ConnectionPool, StrictRedis

#from .utils import get_now_tz, get_uuid

logger = logging.getLogger(__name__)
QUEUE_ACTION = Literal[
    "queue_deferred",
    "queue",
    "worker",
    "result",
    "schedule_last_work",
]

DEFAULT_TIMEZONE = "Europe/Moscow"

def get_uuid() -> str:
    """
    Получить уникальный токен. Возвращает строку со случайно сгенерированным
    уникальным идентификатором.
    """
    return uuid4().hex

def get_current_tz(tz_name: str | None = None) -> "tzinfo":
    """
    Получить временную зону. Возвращает временную зону, соответствующую
    переданному названию, или временную зону по умолчанию, если tz_name
    не передано.

    tz_name:
        Название временной зоны.
    """
    return pytz.timezone(tz_name) or DEFAULT_TIMEZONE

def get_now_tz(tz_name: str | None = None) -> datetime:
    """
    Получить текущее время с временной зоной.
    """
    return get_current_tz(tz_name).fromutc(datetime.utcnow())

class QueueManager:
    """
    Менеджер запуска цикла обработки фоновых операций.

    Реализован в виде класса, чтобы перехватить событие выхода и попытаться
    завершить работу без потери данных.

    _redis_pool:
        Пул соединений Redis.
    _exit_requested:
        Запрошен ли выход из цикла выполнения задач.
    _prefix:
        Префикс, используемый при формировании ключей для очереди.
    """

    def __init__(self, redis_pool: ConnectionPool, prefix: str | None = None):
        """
        Инициализирует объект класса QueueManager.

        redis_pool:
            Пул соединений Redis.
        prefix:
            Префикс, используемый при формировании ключей для очереди.
        """
        self._redis_pool = redis_pool
        self._exit_requested = False
        self._prefix = prefix

    def get_key(self, action: QUEUE_ACTION, **kwargs: str) -> str:
        """
        Получить ключ, в котором что-то храниться для queue. Возвращает строку
        со сформированным в зависимости от action, kwargs и _prefix ключом.

        action:
            Тип формируемого ключа. Для некоторых типов ключ формируется из
            данных в kwargs.

            "queue_deferred" - список задач для очереди отложенных задач.

            "queue" - список задач для конкретной очереди name, которые нужно
            брать в работу.

            "worker" - обработчик worker_idx конкретной очереди queue_name хранит
            задачу, которую он выполняет.

            "result" - результат выполнения задачи task_id. Хранится какое-то
            время.

            "schedule_last_work" - информация о последнем запуске периодической
            задачи name.
        """
        res = ""

        if action == "queue_deferred":
            res = "queue:queue_deferred"

        if action == "queue":
            name = kwargs["name"]
            res = f"queue:queue:{name}"

        if action == "worker":
            queue_name = kwargs["queue_name"]
            worker_idx = kwargs["worker_idx"]
            res = f"queue:worker:{queue_name}:{worker_idx}"

        if action == "result":
            task_id = kwargs["task_id"]
            res = f"queue:result:{task_id}"

        if action == "schedule_last_work":
            name = kwargs["name"]
            res = f"queue:schedule_last_work:{name}"

        if self._prefix:
            res = f"{self._prefix}_{res}"

        return res

    def get_redis(self) -> StrictRedis:
        return StrictRedis(connection_pool=self._redis_pool)

    def add(
        self,
        function: Callable | None = None,
        function_path: str | None = None,
        kwargs: dict = {},
        queue: str = "default",
        schedule_name: str | None = None,
        execute_time: datetime | None = None,
        save_result_sec: int | None = None,
        retry_countdown: int = 0,
        retry_delay_sec: int = 60 * 5,
    ) -> str:
        """
        Добавить в очередь задачу с аргументами для отложенного запуска. Если
        задачу нужно запустить не сейчас, а через какое-то время, то задача
        ставится в отдельную очередь, из которой через какое-то время ее
        извлекают, проверяют, подошло ли время и уже потом перекладывают в
        рабочую очередь. Возвращает идентификатор задачи.

        function:
            Функция, добавляемая в очередь.
        function_path:
            Полный путь к функции, добавляемой в очередь.
        kwargs:
            Именованные аргументы для функции, добавляемой в очередь.
        queue:
            Имя очереди. По умолчанию используется default.
        schedule_name:
            Имя планировщика для отложенных задач.
        execute_time:
            Время выполнения для тложенных задач.
        save_result_sec:
            Время протухания в кэше результат выполнения в функции. Если не
            переданно, то результат не сохраняется.
        retry_countdown:
            Число повторных попыток выполнения в случае исключения.
        retry_delay_sec:
            Время между повторными попытками при исключении.
        """
        redis = self.get_redis()
        task_id = get_uuid()

        if function:
            function_path = f"{function.__module__}:{function.__name__}"
            print(function_path)

        if execute_time:
            queue_key = self.get_key("queue_deferred")
        else:
            queue_key = self.get_key("queue", name=queue)

        redis.rpush(
            queue_key,
            pickle.dumps(
                {
                    "task_id": task_id,
                    "queue": queue,
                    "function_path": function_path,
                    "schedule_name": schedule_name,
                    "kwargs": kwargs or {},
                    "task_id": task_id,
                    "execute_time": execute_time,
                    "save_result_sec": save_result_sec,
                    "retry_countdown": retry_countdown,
                    "retry_delay_sec": retry_delay_sec,
                },
            ),
        )

        return task_id

    def get_result(self, task_id: str) -> Any | None:
        """
        Получить результат выполнения задачи. Возвращает объект, возвращаемый
        задачей из очереди или None, если задача ничего не вернула.

        task_id:
            Идентификатор задачи.
        """
        redis = self.get_redis()
        if data := redis.get(self.get_key("result", task_id=task_id)):
            return pickle.loads(data)

    def has_working_schedule(
        self, redis: StrictRedis, queue_name: str, schedule_name: str
    ) -> bool:
        """
        Проверим, есть ли в работе уже такая задача. Поищем по обработчикам,
        которые выполняют задачи запрошенной очереди. Возвращает булево
        значение, соответствующее результату поиска задачи в запрошенной
        очереди.

        redis:
            Объект реализации протокола Redis.
        queue_name:
            Имя очереди.
        schedule_name:
            Имя планировщика.
        """
        workers_key = self.get_key("worker", queue_name=queue_name, worker_idx="*")

        for worker_key in list(redis.scan_iter(match=workers_key)):
            if data := redis.get(worker_key):
                worker_info: dict = pickle.loads(data)
                if worker_info["schedule_name"] == schedule_name:
                    return True

        return False

    async def run_worker(self, queue_name: str, worker_idx: int) -> None:
        """
        Извлекаем задания из очереди и выполняем в бесконечном цикле.

        queue_name:
            Имя очереди.
        worker_idx:
            Идентификатор обработчика.
        """
        logger.info(f"queue_worker {queue_name} {worker_idx} init")
        worker_key = self.get_key("worker", queue_name=queue_name, worker_idx=worker_idx)

        async def run_task(redis: StrictRedis, item: dict) -> None:
            """
            Выполнить задачу в экземпляре redis с данными из item.

            redis:
                Объект реализации протокола Redis.
            item:
                Словарь с данными для выполнения задачи.
            """
            function_path = item["function_path"]
            logger.info(f"queue_worker {queue_name} {worker_idx} run_task {function_path}")

            # Если это задача по расписанию, смотрим, а не выполняется ли уже такая же.
            # Если выполняется и так, то ничего не делаем, потому что это гарантировано та же
            if schedule_name := item["schedule_name"]:
                if self.has_working_schedule(redis, queue_name, schedule_name):
                    redis.delete(worker_key)  # Очистка на случай, если это задача на доделку
                    return

            # Запомним что взяли задачу в работу
            item["start_at"] = get_now_tz(DEFAULT_TIMEZONE)
            redis.set(worker_key, pickle.dumps(item))

            try:
                function: Callable = import_string(function_path)  # type: ignore

                logger.info(f"queue_worker {queue_name} {worker_idx} start {function_path}")
                result = await function(**item["kwargs"])
                logger.info(
                    f"queue_worker {queue_name} {worker_idx} done {function_path} {result}"
                )

                # Если нужно сохранить результат на какое-то время
                if item["save_result_sec"]:
                    result_key = self.get_key("result", task_id=item["task_id"])
                    redis.set(result_key, pickle.dumps(result), ex=item["save_result_sec"])

            except Exception as exc:
                logger.error(str(exc), exc_info=True, extra={"task": item})

                # Если предусмотрен перезапуск по ошибке, перевыставим задачу в очередь отложенно,
                # вдруг ситуация исправиться (проблемы сети и прочее) или разработчик починит
                if item.get("retry_countdown"):
                    item_retry = dict(item)
                    item_retry["retry_countdown"] = item_retry["retry_countdown"] - 1
                    item_retry["execute_time"] = get_now_tz(DEFAULT_TIMEZONE) + timedelta(
                        seconds=item_retry["retry_delay_sec"]
                    )
                    self.add(**item_retry)

            redis.delete(worker_key)  # Задача этим воркером выполнена, очистка состояния

            logger.info(f"queue_worker {queue_name} {worker_idx} run_task {function_path} DONE")

        redis = self.get_redis()

        # Если на момент запуска в этом worker есть что-то недоделанное, доделываем
        if redis_item := redis.get(worker_key):
            await run_task(redis, pickle.loads(redis_item))

        queue_key = self.get_key("queue", name=queue_name)

        while True:
            while redis_item := redis.lpop(queue_key):
                await run_task(redis, pickle.loads(redis_item))

                # Выход по завершению обработки очередной задачи
                if self._exit_requested:
                    logger.info(f"queue_worker {queue_name} {worker_idx} exit")
                    return

            await asyncio.sleep(2)

            # Выход в процессе ожидания задач в очереди
            if self._exit_requested:
                logger.info(f"queue_worker {queue_name} {worker_idx} exit")
                return

    async def run_scheduler(self, scheduler: dict) -> None:
        """
        Выставляем задачи в работу по расписанию и отложенных.
        dc exec queue manage.py queue_sheduler

        scheduler:
            Данные планировщика.
        """
        logger.info("scheduler init")
        redis = self.get_redis()

        while True:
            current = get_now_tz()

            # Проверить расписания запуска
            for schedule_name, data in scheduler.items():
                queue_name = data.get("queue", "default")

                # Если сейчас такое расписание уже работает, пропустим
                if self.has_working_schedule(
                    redis=redis, queue_name=queue_name, schedule_name=schedule_name
                ):
                    logger.info(f"scheduler has_working {schedule_name}, skip")
                    continue

                last_run_key = self.get_key("schedule_last_work", name=schedule_name)
                last_run = redis.get(last_run_key)  # Когда запускалось в последний раз

                # Еще не запускалась ни разу - запомнить текущее время и пропустим,
                # чтобы получить точку начального отсчета
                if last_run is None:
                    redis.set(last_run_key, pickle.dumps(current))
                    continue

                last_run = pickle.loads(last_run)
                cron = croniter.croniter(data["period"], last_run)
                next_run = cron.get_next(datetime)

                # Время следующего запуска еще не подошло
                if next_run > current:
                    continue

                task_id = self.add(
                    function_path=data.get("function_path"),
                    kwargs=data.get("kwargs"),
                    queue=data.get("queue", "default"),
                    schedule_name=schedule_name,
                )
                redis.set(last_run_key, pickle.dumps(current))
                logger.info(f"queue_sheduler {schedule_name} add to queue {task_id}")

            # Просмотреть очередь отложенных задач и если время подошло,
            # переложить задачу в очередь на выполнение
            deferred_key = self.get_key("queue_deferred")
            idx = 0
            while redis_item := redis.lindex(deferred_key, idx):
                data = pickle.loads(redis_item)
                if data["execute_time"] <= current:
                    queue_name = data["queue"]
                    queue_key = self.get_key("queue", name=queue_name)
                    redis.rpush(queue_key, redis_item)
                    redis.lrem(name=deferred_key, count=1, value=redis_item)

                else:
                    idx += 1

            if self._exit_requested:
                logger.info("scheduler exit")
                return

            await asyncio.sleep(30)

    async def run(self, queue_worker_cnt: dict[str, int], scheduler: dict | None = None) -> None:
        """
        Запустить бесконечный цикл обработки событий. Для очередей создаем
        несколько обработчиков, которые будут работать в бесконечном цикле и
        также, запускаем отправку по расписанию задач в очередь.

        queue_worker_cnt:
            Словарь с количествами обработчиков для очередей.
        scheduler:
            Словарь с данными планировщика.
        """

        def gracefully_exit(sig, frame) -> None:
            """
            Пометим что запрошен выход и будем до последнего ждать завершения задач.
            """
            self._exit_requested = True
            logger.info("Request exit")

        signal.signal(signal.SIGHUP, gracefully_exit)
        signal.signal(signal.SIGINT, gracefully_exit)

        tasks = []

        for queue_name, worker_count in queue_worker_cnt.items():
            for worker_idx in range(worker_count):
                tasks.append(
                    asyncio.create_task(
                        self.run_worker(queue_name=queue_name, worker_idx=worker_idx)
                    )
                )

        if scheduler:
            tasks.append(asyncio.create_task(self.run_scheduler(scheduler=scheduler)))

        for task in tasks:
            await task
