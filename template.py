"""
K4 — Ngày 1: Khám Phá LLM API (14h00–18h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},
}

# Luồng chính: OpenAI (mặc định, không cần đặt gì trong .env).
# Không có key OpenAI? Dùng luồng thay thế Google Gemini (Phụ lục B
# trong LAB_GUIDE.md) — tên model đổi qua .env. NVIDIA NIM: Phụ lục C.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 15h00–15h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency_seconds = time.perf_counter() - start_time
    response_text = response.choices[0].message.content
    return response_text, latency_seconds


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)
    gpt4o_cost_estimate = (
        (len(gpt4o_response.split()) / 0.75)
        / 1000
        * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    )

    return {
        "gpt4o_answer": gpt4o_response,
        "mini_answer": mini_response,
        "gpt4o_time": gpt4o_latency,
        "mini_time": mini_latency,
        "gpt4o_cost": gpt4o_cost_estimate,
    }

# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 15h40–16h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_seconds = time.perf_counter() - start_time
    response_text = response.choices[0].message.content
    return response_text, latency_seconds


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken.

    Args:
        text:  Đoạn text cần đếm.
        model: Model dùng để chọn bộ mã hóa (encoding).

    Returns:
        Số token (int).

    Gợi ý:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

        tiktoken cần tải bộ mã hóa từ mạng ở lần chạy đầu. Hãy bọc trong
        try/except — nếu lỗi (offline, model lạ), dùng ước lượng dự phòng:
        max(1, len(text) // 4)   (trung bình 1 token ≈ 4 ký tự)
    """
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token THẬT (đếm bằng
    count_tokens) và bảng giá PRICING_PER_1K_TOKENS — tách riêng chi phí
    input (prompt) và output (response).

    Returns:
        Dict với các key:
            - "prompt_tokens":  int
            - "completion_tokens": int
            - "prompt_cost":    float  (USD)
            - "completion_cost":   float  (USD)
            - "total_cost":    float  (USD)

    Gợi ý:
        pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
        prompt_cost = prompt_tokens / 1000 * pricing["input"]
        (.get với fallback: model không có trong bảng giá — ví dụ model NIM
         miễn phí — thì lấy giá gpt-4o làm tham chiếu học tập)
    """
    prompt_tokens = count_tokens(prompt, model=model)
    completion_tokens = count_tokens(response, model=model)

    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
    prompt_cost = prompt_tokens / 1000 * pricing["input"]
    completion_cost = completion_tokens / 1000 * pricing["output"]

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": prompt_cost + completion_cost,
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 16h30–17h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": user_input})
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            stream=True,
        )

        reply_parts = []
        print("Assistant: ", end="")
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()

        history.append({"role": "assistant", "content": "".join(reply_parts)})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2**attempt)


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 17h10–17h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh — ghép mọi thứ bạn đã xây trong Part 1–3.

    Hành vi:
        1. Dùng `persona` làm system prompt cho TOÀN BỘ phiên chat.
        2. Mỗi lượt: đọc tin nhắn qua get_input(); nếu là 'quit'/'exit'/'bye'
           (không phân biệt hoa thường) → kết thúc phiên.
        3. Gọi API với stream=True, messages = system + history + tin nhắn mới.
           Bọc lời gọi API trong retry_with_backoff để chịu lỗi tạm thời.
        4. In từng chunk khi stream về, ghép lại thành reply hoàn chỉnh.
        5. Cập nhật history (user + assistant), giữ tối đa 4 lượt cuối
           (8 message): history = history[-8:]
        6. Cộng dồn thống kê bằng count_tokens và estimate_cost.
        7. Dừng khi đạt max_turns (nếu được đặt).

    Args:
        persona:   Mô tả vai trò, dùng làm system prompt.
        get_input: Hàm đọc input (mặc định: input). Tham số này giúp
                   test tự động không cần bàn phím thật.
        max_turns: Số lượt tối đa (None = không giới hạn).

    Returns:
        Dict thống kê phiên chat:
            - "turns":    int   (số lượt hỏi–đáp đã thực hiện)
            - "tokens_used": int   (tổng token user + assistant)
            - "total_cost":   float (tổng USD ước tính)
            - "history":      list  (history còn lại sau khi cắt, ≤ 8 message)

    Gợi ý khung sườn:
        if get_input is None:
            get_input = input
        history, turns, tokens_used, total_cost = [], 0, 0, 0.0
        while True:
            if max_turns is not None and turns >= max_turns:
                break
            user_msg = get_input()
            if user_msg.strip().lower() in ("quit", "exit", "bye"):
                break
            messages = [{"role": "system", "content": persona}] + history \\
                       + [{"role": "user", "content": user_msg}]
            # stream = retry_with_backoff(lambda: client.chat...create(
            #              model=..., messages=messages, stream=True))
            # reply = ghép các chunk...
            ...
        return {"turns": turns, "tokens_used": tokens_used,
                "total_cost": total_cost, "history": history}
    """
    from openai import OpenAI

    if get_input is None:
        get_input = input

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history, turns, tokens_used, total_cost = [], 0, 0, 0.0

    while True:
        if max_turns is not None and turns >= max_turns:
            break

        user_msg = get_input()
        if user_msg.strip().lower() in ("quit", "exit", "bye"):
            break

        messages = (
            [{"role": "system", "content": persona}]
            + history
            + [{"role": "user", "content": user_msg}]
        )

        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
        )

        reply_parts = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        reply = "".join(reply_parts)

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-8:]

        turns += 1
        tokens_used += count_tokens(user_msg) + count_tokens(reply)
        total_cost += estimate_cost(user_msg, reply)["total_cost"]

    return {
        "turns": turns,
        "tokens_used": tokens_used,
        "total_cost": total_cost,
        "history": history,
    }


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        comparison = compare_models(prompt)
        results.append({**comparison, "prompt": prompt})
    return results


def format_comparison_table(results: list[dict]) -> str:
    def shorten(text: str) -> str:
        return text if len(text) <= 40 else f"{text[:37]}..."

    rows = ["Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency"]
    for result in results:
        rows.append(
            " | ".join(
                [
                    shorten(result["prompt"]),
                    shorten(result["gpt4o_response"]),
                    shorten(result["mini_response"]),
                    f"{result['gpt4o_latency']:.2f}s",
                    f"{result['mini_latency']:.2f}s",
                ]
            )
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
