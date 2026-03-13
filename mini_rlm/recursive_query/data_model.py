from pydantic import BaseModel


class RecursiveQueryConfig(BaseModel):
    max_depth: int = 2
    child_token_limit: int = 100_000
    child_iteration_limit: int = 20
    child_timeout_seconds: float = 180.0
    child_error_threshold: int = 5
    child_compacting_threshold_rate: float = 0.85
    inherit_parent_files: bool = True


class RecursiveQueryRuntime(BaseModel):
    remaining_depth: int


class RecursiveQueryRequest(BaseModel):
    prompt: str


class RecursiveQueryResult(BaseModel):
    termination_reason: str
    final_answer: str | None
    total_iterations: int
    total_tokens: int
    total_time_seconds: float
