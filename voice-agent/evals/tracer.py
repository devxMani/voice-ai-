import time
from evals.trace import ToolTrace


class ToolTracer:
    def __init__(self):
        self.tools: list[ToolTrace] = []

    async def run(self, name, func, *args, **kwargs):
        start = time.perf_counter()

        trace = ToolTrace(
            name=name,
            arguments=kwargs,
        )

        try:
            result = await func(*args, **kwargs)

            trace.output = str(result)
            trace.success = True

            return result

        except Exception as exc:
            trace.success = False
            trace.error = str(exc)
            raise

        finally:
            trace.latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            self.tools.append(trace)

            print(
                f"[TRACE] tool={trace.name} "
                f"latency={trace.latency_ms}ms "
                f"success={trace.success}"
            )