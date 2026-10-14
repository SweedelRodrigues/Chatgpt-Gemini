"""
Evaluation and Benchmarking Metrics Calculation Module.
Provides reproducible calculation of benchmark statistics, latency aggregations,
and token accounting from test suites and CSV exports.
"""

import csv
import io
import statistics
from typing import Any, Optional
from pydantic import BaseModel, Field


class BenchmarkFailureDetail(BaseModel):
    """Details of a failed benchmark request."""
    id: Optional[Any] = None
    category: Optional[str] = None
    prompt: Optional[str] = None
    error: str


class BenchmarkSummary(BaseModel):
    """
    Standardized benchmark evaluation summary schema.
    """
    total_requests: int = Field(..., description="Total number of evaluated requests.")
    successful_requests: int = Field(..., description="Count of requests that completed with HTTP 200.")
    failed_requests: int = Field(..., description="Count of requests that failed or timed out.")
    success_rate: float = Field(..., description="Success rate percentage (0.0 to 100.0%).")
    mean_latency_ms: Optional[float] = Field(default=None, description="Average latency of successful requests in ms.")
    median_latency_ms: Optional[float] = Field(default=None, description="Median latency of successful requests in ms.")
    min_latency_ms: Optional[float] = Field(default=None, description="Minimum latency observed among successful requests.")
    max_latency_ms: Optional[float] = Field(default=None, description="Maximum latency observed among successful requests.")
    total_input_tokens: int = Field(default=0, description="Sum of input/prompt tokens across successful runs.")
    total_output_tokens: int = Field(default=0, description="Sum of output text tokens across successful runs.")
    total_tokens: int = Field(default=0, description="Total tokens consumed including thinking tokens.")
    failures: list[BenchmarkFailureDetail] = Field(default_factory=list, description="List of recorded failure reasons.")


def calculate_benchmark_metrics(results: list[dict[str, Any]]) -> BenchmarkSummary:
    """
    Calculates reproducible aggregate metrics across benchmark test items.

    Args:
        results: List of result dictionaries containing status, latency_ms, usage, and error.

    Returns:
        BenchmarkSummary model instance.
    """
    total = len(results)
    if total == 0:
        return BenchmarkSummary(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            success_rate=0.0,
            mean_latency_ms=None,
            median_latency_ms=None,
            min_latency_ms=None,
            max_latency_ms=None,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            failures=[],
        )

    successful = 0
    failed = 0
    latencies: list[float] = []
    input_tokens_list: list[int] = []
    output_tokens_list: list[int] = []
    total_tokens_list: list[int] = []
    failures: list[BenchmarkFailureDetail] = []

    for item in results:
        status_val = str(item.get("status") or item.get("Status") or "").strip().lower()

        if status_val == "success":
            successful += 1
            lat = item.get("latency_ms") if item.get("latency_ms") is not None else item.get("Latency_ms")
            if lat is not None and lat != "":
                try:
                    lat_num = float(lat)
                    latencies.append(lat_num)
                except (ValueError, TypeError):
                    pass

            # Extract token usage if available
            usage = item.get("usage") or item.get("Usage")
            if isinstance(usage, dict):
                in_t = usage.get("input_tokens") or usage.get("Input_Tokens")
                out_t = usage.get("output_tokens") or usage.get("Output_Tokens")
                tot_t = usage.get("total_tokens") or usage.get("Total_Tokens")
            else:
                in_t = item.get("input_tokens") or item.get("Input_Tokens")
                out_t = item.get("output_tokens") or item.get("Output_Tokens")
                tot_t = item.get("total_tokens") or item.get("Total_Tokens")

            if in_t is not None and in_t != "":
                try:
                    input_tokens_list.append(int(in_t))
                except (ValueError, TypeError):
                    pass

            if out_t is not None and out_t != "":
                try:
                    output_tokens_list.append(int(out_t))
                except (ValueError, TypeError):
                    pass

            if tot_t is not None and tot_t != "":
                try:
                    total_tokens_list.append(int(tot_t))
                except (ValueError, TypeError):
                    pass
        else:
            failed += 1
            err_msg = str(item.get("error") or item.get("Error") or "Request failed")
            failures.append(
                BenchmarkFailureDetail(
                    id=item.get("id") or item.get("ID"),
                    category=item.get("category") or item.get("Category"),
                    prompt=item.get("prompt") or item.get("Prompt"),
                    error=err_msg,
                )
            )

    success_rate = round((successful / total) * 100.0, 2)
    mean_lat = round(statistics.mean(latencies), 2) if latencies else None
    median_lat = round(float(statistics.median(latencies)), 2) if latencies else None
    min_lat = round(min(latencies), 2) if latencies else None
    max_lat = round(max(latencies), 2) if latencies else None

    return BenchmarkSummary(
        total_requests=total,
        successful_requests=successful,
        failed_requests=failed,
        success_rate=success_rate,
        mean_latency_ms=mean_lat,
        median_latency_ms=median_lat,
        min_latency_ms=min_lat,
        max_latency_ms=max_lat,
        total_input_tokens=sum(input_tokens_list),
        total_output_tokens=sum(output_tokens_list),
        total_tokens=sum(total_tokens_list),
        failures=failures,
    )


def parse_benchmark_csv_content(csv_content: str) -> BenchmarkSummary:
    """
    Parses CSV text representing benchmark runs and calculates the summary metrics.

    Args:
        csv_content: Raw CSV string with headers.

    Returns:
        BenchmarkSummary.
    """
    stream = io.StringIO(csv_content.strip())
    reader = csv.DictReader(stream)
    rows = list(reader)
    return calculate_benchmark_metrics(rows)
