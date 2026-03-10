import math
import random
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from prometheus_client import (
    Counter, Gauge, Histogram,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
)

registry = CollectorRegistry()

http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests",
    ["service", "method", "handler", "status_code"],
    registry=registry,
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds", "HTTP request latency in seconds",
    ["service", "method", "handler"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry,
)
http_requests_in_flight = Gauge(
    "http_requests_in_flight", "Current number of in-flight HTTP requests",
    ["service"], registry=registry,
)
http_response_size_bytes = Histogram(
    "http_response_size_bytes", "HTTP response size in bytes",
    ["service", "handler"],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
    registry=registry,
)

payment_transactions_total = Counter(
    "payment_transactions_total", "Total payment transactions",
    ["service", "status", "payment_method"], registry=registry,
)
payment_processing_duration_seconds = Histogram(
    "payment_processing_duration_seconds", "Payment processing duration",
    ["service", "payment_method"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    registry=registry,
)
payment_queue_depth = Gauge(
    "payment_queue_depth", "Number of payments waiting to be processed",
    ["service"], registry=registry,
)
payment_amount_dollars = Histogram(
    "payment_amount_dollars", "Payment transaction amount in dollars",
    ["service", "payment_method"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
    registry=registry,
)
payment_errors_total = Counter(
    "payment_errors_total", "Total payment processing errors",
    ["service", "error_type"], registry=registry,
)

db_connections_active = Gauge(
    "db_connections_active", "Number of active database connections",
    ["service", "database"], registry=registry,
)
db_connections_idle = Gauge(
    "db_connections_idle", "Number of idle database connections",
    ["service", "database"], registry=registry,
)
db_connections_max = Gauge(
    "db_connections_max", "Maximum allowed database connections",
    ["service", "database"], registry=registry,
)
db_query_duration_seconds = Histogram(
    "db_query_duration_seconds", "Database query duration",
    ["service", "database", "operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=registry,
)
db_queries_total = Counter(
    "db_queries_total", "Total database queries",
    ["service", "database", "operation", "status"], registry=registry,
)
db_cache_hits_total = Counter(
    "db_cache_hits_total", "Total database cache hits",
    ["service", "database"], registry=registry,
)
db_cache_misses_total = Counter(
    "db_cache_misses_total", "Total database cache misses",
    ["service", "database"], registry=registry,
)
db_replication_lag_seconds = Gauge(
    "db_replication_lag_seconds", "Database replication lag in seconds",
    ["service", "database", "replica"], registry=registry,
)

app_info = Gauge(
    "app_info", "Application build information",
    ["service", "version", "environment"], registry=registry,
)
app_uptime_seconds = Gauge(
    "app_uptime_seconds", "Application uptime in seconds",
    ["service"], registry=registry,
)

START_TIME = time.time()


def noisy(base, pct=0.15):
    return max(0, base + random.gauss(0, base * pct))

def spike(base, t, period=300, amplitude=0.4):
    return base * (1 + amplitude * abs(math.sin(2 * math.pi * t / period)))


def simulate():
    for svc, ver in [("api-gateway", "1.4.2"), ("payment", "2.1.0"), ("database", "3.0.1")]:
        app_info.labels(service=svc, version=ver, environment="production").set(1)

    db_connections_max.labels(service="api-gateway", database="postgres").set(50)
    db_connections_max.labels(service="payment",     database="postgres").set(100)
    db_connections_max.labels(service="payment",     database="redis").set(200)

    for replica in ["replica-1", "replica-2"]:
        db_replication_lag_seconds.labels(
            service="database", database="postgres", replica=replica
        ).set(0)

    while True:
        t = time.time()

        for svc in ["api-gateway", "payment", "database"]:
            app_uptime_seconds.labels(service=svc).set(t - START_TIME)

        rps = spike(50, t)
        handlers = [
            ("/api/v1/users",    0.30, 0.05),
            ("/api/v1/orders",   0.25, 0.12),
            ("/api/v1/products", 0.25, 0.04),
            ("/api/v1/payments", 0.15, 0.35),
            ("/healthz",         0.05, 0.002),
        ]
        for handler, weight, latency_base in handlers:
            count = int(noisy(rps * weight))
            for _ in range(count):
                method     = random.choice(["GET", "POST", "PUT"]) if handler != "/healthz" else "GET"
                error_rate = 0.02 if "payment" in handler else 0.005
                code       = "500" if random.random() < error_rate else "200"
                http_requests_total.labels(
                    service="api-gateway", method=method, handler=handler, status_code=code,
                ).inc()
                http_request_duration_seconds.labels(
                    service="api-gateway", method=method, handler=handler,
                ).observe(noisy(latency_base))
                http_response_size_bytes.labels(
                    service="api-gateway", handler=handler,
                ).observe(noisy(2000))

        http_requests_in_flight.labels(service="api-gateway").set(noisy(rps * 0.1))

        for method, weight in [("card", 0.6), ("bank_transfer", 0.25), ("crypto", 0.15)]:
            for _ in range(int(noisy(rps * 0.15 * weight))):
                success = random.random() > 0.03
                payment_transactions_total.labels(
                    service="payment",
                    status="success" if success else "failed",
                    payment_method=method,
                ).inc()
                payment_processing_duration_seconds.labels(
                    service="payment", payment_method=method,
                ).observe(noisy(0.5 if method == "card" else 2.0))
                payment_amount_dollars.labels(
                    service="payment", payment_method=method,
                ).observe(random.expovariate(1/80))

        if random.random() < 0.05:
            payment_errors_total.labels(
                service="payment",
                error_type=random.choice(["timeout", "gateway_error", "insufficient_funds", "fraud_detected"]),
            ).inc()

        payment_queue_depth.labels(service="payment").set(noisy(spike(5, t, period=120)))

        qps = spike(200, t, period=180)
        for db, ops in [
            ("postgres", [("select", 0.6, 0.008), ("insert", 0.2, 0.005), ("update", 0.15, 0.006), ("delete", 0.05, 0.007)]),
            ("redis",    [("get", 0.7, 0.001), ("set", 0.2, 0.001), ("del", 0.1, 0.001)]),
        ]:
            svc = "payment" if db == "redis" else "api-gateway"
            for op, weight, latency_base in ops:
                for _ in range(int(noisy(qps * weight))):
                    ok = random.random() > 0.002
                    db_queries_total.labels(
                        service=svc, database=db, operation=op,
                        status="ok" if ok else "error",
                    ).inc()
                    db_query_duration_seconds.labels(
                        service=svc, database=db, operation=op,
                    ).observe(noisy(latency_base))
                    if db == "redis" and op == "get":
                        if random.random() < 0.85:
                            db_cache_hits_total.labels(service=svc, database=db).inc()
                        else:
                            db_cache_misses_total.labels(service=svc, database=db).inc()

        active = noisy(spike(20, t, period=240))
        db_connections_active.labels(service="api-gateway", database="postgres").set(active)
        db_connections_idle.labels(service="api-gateway",   database="postgres").set(noisy(50 - active))

        active_pay = noisy(spike(35, t, period=180))
        db_connections_active.labels(service="payment", database="postgres").set(active_pay)
        db_connections_idle.labels(service="payment",   database="postgres").set(noisy(100 - active_pay))
        db_connections_active.labels(service="payment", database="redis").set(noisy(spike(15, t)))
        db_connections_idle.labels(service="payment",   database="redis").set(noisy(20))

        for replica in ["replica-1", "replica-2"]:
            lag = noisy(0.05) if random.random() > 0.01 else noisy(2.0)
            db_replication_lag_seconds.labels(
                service="database", database="postgres", replica=replica,
            ).set(lag)

        time.sleep(1)


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            output = generate_latest(registry)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
        elif self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=simulate, daemon=True).start()
    print("app_metrics running on :8000")
    HTTPServer(("0.0.0.0", 8000), MetricsHandler).serve_forever()