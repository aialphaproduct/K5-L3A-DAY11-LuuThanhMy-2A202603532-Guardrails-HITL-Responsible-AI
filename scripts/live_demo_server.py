"""
Live demo server — gọi THẬT Blue (OpenRouter liquid/lfm-3-1.7b:free) qua từng
lớp guardrail thật, hiển thị lớp nào đã áp dụng + JSON chi tiết từng request.

Đây KHÔNG phải Artifact tĩnh — là 1 server cục bộ. API key nằm server-side
(đọc từ .env), không bao giờ gửi xuống trình duyệt, nên an toàn để demo trực
tiếp mà không lộ key.

Chạy:
    python scripts/live_demo_server.py
Mở:
    http://127.0.0.1:8787
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from guardrails.input_guardrails import validate_input, detect_injection, topic_filter, MAX_INPUT_LENGTH
from guardrails.output_guardrails import content_filter, check_grounding
from core.openai_runtime import create_blue_pair, create_openai_pair
from core.config import (
    get_openrouter_api_key,
    blue_provider_label,
    red_provider_label,
    red_uses_openai_sdk,
    get_red_model,
)
from agents.agent import BLUE_INSTRUCTION
from agents.guards_agent import (
    detect_injection_strong,
    topic_filter_strong,
    content_filter_strong,
    RED_ADVANCE_INSTRUCTION,
    REGEX_INJECTION_MESSAGE,
    REGEX_TOPIC_MESSAGE,
    REGEX_OUTPUT_MESSAGE,
)
from core.utils import chat_with_agent

app = FastAPI(title="Blue Live Demo")

agent, runner = create_blue_pair(
    name="live_demo_blue",
    instruction=BLUE_INSTRUCTION,
    app_name="live_demo_blue",
)

# Red Advance (regex + NeMo) — khởi tạo lười (lazy), chỉ khi lần đầu người
# dùng chọn mục tiêu này, vì cần thêm key Red (OpenAI/Gemini) và cấu hình NeMo.
_red_advance_state: dict = {"ready": False, "agent": None, "runner": None, "nemo": None, "error": None}


def _init_red_advance():
    if _red_advance_state["ready"] or _red_advance_state["error"]:
        return
    try:
        from agents.red_advance_nemo import NemoRailsGuard

        nemo_guard = NemoRailsGuard()
        if red_uses_openai_sdk():
            red_agent, red_runner = create_openai_pair(
                name="live_demo_red_advance",
                instruction=RED_ADVANCE_INSTRUCTION,
                app_name="live_demo_red_advance",
                model=get_red_model(),
            )
        else:
            from google.adk.agents import llm_agent
            from google.adk import runners

            red_agent = llm_agent.LlmAgent(
                model=get_red_model(), name="live_demo_red_advance",
                instruction=RED_ADVANCE_INSTRUCTION,
            )
            red_runner = runners.InMemoryRunner(agent=red_agent, app_name="live_demo_red_advance")
        _red_advance_state.update(ready=True, agent=red_agent, runner=red_runner, nemo=nemo_guard)
    except Exception as e:  # fail-closed: báo lỗi rõ, không crash server
        _red_advance_state["error"] = f"{type(e).__name__}: {e}"


class AskBody(BaseModel):
    question: str
    target: str = "blue"


def _result(**kwargs) -> JSONResponse:
    return JSONResponse(kwargs)


@app.post("/api/ask")
async def ask(body: AskBody):
    text = (body.question or "").strip()
    if body.target == "red_advance":
        return await ask_red_advance(text)
    return await ask_blue(text)


async def ask_blue(text: str):
    t0 = time.time()
    trace: list[dict] = []

    def step(layer: str, status: str, **extra) -> dict:
        row = {"layer": layer, "status": status, **extra}
        trace.append(row)
        return row

    # --- Input Rails ---
    v = validate_input(text)
    step(
        "input_rails.validation", v,
        detail=f"độ dài {len(text)}/{MAX_INPUT_LENGTH} ký tự, không rỗng" if v == "ALLOW"
        else "chuỗi rỗng hoặc vượt quá độ dài cho phép",
    )
    if v == "BLOCK":
        answer = "I'm a VinBank assistant and can only help with banking-related questions."
        return _result(**_finalize("blue", text, trace, True, "input_rails.validation", answer, t0, model=blue_provider_label()))

    inj = detect_injection(text)
    step(
        "input_rails.injection_detection", inj,
        detail="khớp 1 trong 21 pattern injection (EN/VI/encoding)" if inj == "BLOCK"
        else "không khớp pattern injection nào",
    )
    if inj == "BLOCK":
        answer = "I cannot process that request. I can only help with VinBank banking questions."
        return _result(**_finalize("blue", text, trace, True, "input_rails.injection_detection", answer, t0, model=blue_provider_label()))

    topic = topic_filter(text)
    step(
        "input_rails.topic_filter", topic,
        detail="không chứa từ khoá ngân hàng cho phép, hoặc chứa từ khoá cấm" if topic == "BLOCK"
        else "thuộc chủ đề ngân hàng hợp lệ",
    )
    if topic == "BLOCK":
        answer = "I'm a VinBank assistant and can only help with banking-related questions."
        return _result(**_finalize("blue", text, trace, True, "input_rails.topic_filter", answer, t0, model=blue_provider_label()))

    # --- LLM Rails ---
    step(
        "llm_rails.system_prompt_hardening", "APPLIED",
        detail="BLUE_INSTRUCTION — quy tắc bảo mật outrank mọi hướng dẫn sau, "
        "tách dữ liệu không tin cậy khỏi lệnh thật",
    )
    raw_answer, _ = await chat_with_agent(agent, runner, text)
    step("llm_call", "CALLED", detail=f"gọi thật — {blue_provider_label()}")

    # --- Output Rails ---
    cf = content_filter(raw_answer)
    step(
        "output_rails.content_filter", "SAFE" if cf["safe"] else "REDACTED",
        detail=", ".join(cf["issues"]) if cf["issues"] else "không có PII/secret",
    )
    final_answer = cf["redacted"] if not cf["safe"] else raw_answer

    gr = check_grounding(final_answer)
    step(
        "output_rails.grounding", "GROUNDED" if gr["grounded"] else "MISMATCH",
        detail=str(gr["mismatches"]) if gr["mismatches"] else "số liệu (nếu có) khớp ground truth",
    )

    needs_review = (not cf["safe"]) or (not gr["grounded"])
    step(
        "output_rails.human_review", "FLAGGED" if needs_review else "NOT_NEEDED",
        detail="redact hoặc lệch số liệu → cần người duyệt" if needs_review else "không cần duyệt tay",
    )

    return _result(**_finalize(
        "blue", text, trace, False, None, final_answer, t0,
        raw_answer=raw_answer, needs_review=needs_review, model=blue_provider_label(),
    ))


async def ask_red_advance(text: str):
    t0 = time.time()
    trace: list[dict] = []

    def step(layer: str, status: str, **extra) -> dict:
        row = {"layer": layer, "status": status, **extra}
        trace.append(row)
        return row

    _init_red_advance()
    if _red_advance_state["error"]:
        step("setup_error", "ERROR", detail=_red_advance_state["error"])
        return _result(**_finalize(
            "red_advance", text, trace, True, "setup_error",
            f"Red Advance chưa sẵn sàng: {_red_advance_state['error']}", t0,
            model=red_provider_label(),
        ))

    red_agent = _red_advance_state["agent"]
    red_runner = _red_advance_state["runner"]
    nemo_guard = _red_advance_state["nemo"]

    # --- L1 Regex input ---
    inj = detect_injection_strong(text)
    step(
        "input_rails.l1_regex_injection", "BLOCK" if inj else "ALLOW",
        detail="detect_injection_strong() khớp pattern" if inj else "không khớp pattern injection",
    )
    if inj:
        return _result(**_finalize(
            "red_advance", text, trace, True, "input_rails.l1_regex_injection",
            REGEX_INJECTION_MESSAGE, t0, model=red_provider_label(),
        ))

    topic = topic_filter_strong(text)
    step(
        "input_rails.l1_regex_topic", "BLOCK" if topic else "ALLOW",
        detail="topic_filter_strong() — lạc chủ đề hoặc cố moi thông tin" if topic else "thuộc chủ đề ngân hàng hợp lệ",
    )
    if topic:
        return _result(**_finalize(
            "red_advance", text, trace, True, "input_rails.l1_regex_topic",
            REGEX_TOPIC_MESSAGE, t0, model=red_provider_label(),
        ))

    # --- L2 NeMo input rails ---
    verdict_in = await nemo_guard.check_input(text)
    step(
        "input_rails.l2_nemo", "BLOCK" if not verdict_in.allowed else "ALLOW",
        detail=(f"NeMo rail: {verdict_in.rail}" if not verdict_in.allowed else "self-check input: an toàn")
        + (f" (lỗi: {verdict_in.error})" if verdict_in.error else ""),
    )
    if not verdict_in.allowed:
        return _result(**_finalize(
            "red_advance", text, trace, True, "input_rails.l2_nemo",
            verdict_in.message, t0, model=red_provider_label(),
        ))

    # --- LLM call (Red Advance chính) ---
    raw_answer, _ = await chat_with_agent(red_agent, red_runner, text)
    step("llm_call", "CALLED", detail=f"gọi thật — {red_provider_label()}")

    # --- L3 Regex output ---
    cf = content_filter_strong(raw_answer)
    step(
        "output_rails.l3_regex", "BLOCK" if not cf["safe"] else "ALLOW",
        detail=f"content_filter_strong() khớp: {', '.join(cf['issues'])}" if not cf["safe"] else "không có secret nguyên văn",
    )
    if not cf["safe"]:
        return _result(**_finalize(
            "red_advance", text, trace, True, "output_rails.l3_regex",
            REGEX_OUTPUT_MESSAGE, t0, raw_answer=raw_answer, model=red_provider_label(),
        ))

    # --- L4 NeMo output rails ---
    verdict_out = await nemo_guard.check_output(text, raw_answer)
    step(
        "output_rails.l4_nemo", "BLOCK" if not verdict_out.allowed else "ALLOW",
        detail=(f"NeMo rail: {verdict_out.rail}" if not verdict_out.allowed else "self-check output: an toàn, không lộ (kể cả mã hoá)")
        + (f" (lỗi: {verdict_out.error})" if verdict_out.error else ""),
    )
    if not verdict_out.allowed:
        return _result(**_finalize(
            "red_advance", text, trace, True, "output_rails.l4_nemo",
            verdict_out.message, t0, raw_answer=raw_answer, model=red_provider_label(),
        ))

    return _result(**_finalize(
        "red_advance", text, trace, False, None, raw_answer, t0,
        raw_answer=raw_answer, model=red_provider_label(),
    ))


def _finalize(target, text, trace, blocked, final_layer, answer, t0, *, raw_answer=None, needs_review=False, model=None) -> dict:
    return {
        "target": target,
        "input": text,
        "blocked": blocked,
        "final_blocking_layer": final_layer,
        "answer": answer,
        "raw_llm_answer": raw_answer,
        "needs_human_review": needs_review,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "model": model,
        "trace": trace,
    }


@app.get("/api/status")
async def status():
    return {
        "has_openrouter_key": bool(get_openrouter_api_key()),
        "blue_model": blue_provider_label(),
        "red_model": red_provider_label(),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = (ROOT / "scripts" / "live_demo_ui.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    if not get_openrouter_api_key():
        print("CẢNH BÁO: chưa có OPENROUTER_API_KEY trong .env — /api/ask sẽ lỗi khi gọi LLM thật.")
    print(f"Blue live demo — Blue: {blue_provider_label()} | Red Advance: {red_provider_label()} (NeMo, khởi tạo lười)")
    port = int(os.environ.get("PORT", 8793))
    print(f"Mở: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
