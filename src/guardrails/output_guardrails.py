"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import json
import re
import textwrap
import unicodedata
from pathlib import Path

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent

# LLM thường in secret với dấu gạch Unicode khác (‑ U+2011, – U+2013, — U+2014...)
# thay vì "-" thường — regex ASCII sẽ không khớp nếu không chuẩn hoá trước.
_DASH_VARIANTS_RE = re.compile(r"[‐‑‒–—―−]")


def _canonicalize_for_filter(text: str) -> str:
    """Chuẩn hoá Unicode + gộp các kiểu gạch ngang về '-' trước khi so regex."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return _DASH_VARIANTS_RE.sub("-", normalized)


# ============================================================
# Nâng cấp ngoài yêu cầu tối thiểu (3 Tầng Phòng Thủ — Output Rails:
# "grounding" + "human review"). LLM-as-Judge trong lab là optional/không
# chấm; đây là một lớp grounding RIÊNG, dựa trên rule + ground truth thật
# (data/pii_hallucination_samples.json) — không cần gọi thêm LLM, chạy tức
# thì, không tốn API.
# ============================================================

_GROUND_TRUTH_CACHE: dict | None = None

_RATE_KEYWORDS = {
    "savings_apy_12m_percent": [r"12[\s-]?(?:month|tháng)", r"\b12m\b"],
    "savings_apy_6m_percent": [r"6[\s-]?(?:month|tháng)", r"\b6m\b"],
    "personal_loan_apr_percent": [r"personal\s+loan", r"vay\s+(?:cá\s+nhân|tiêu\s+dùng)"],
    "home_loan_apr_percent": [r"home\s+loan", r"vay\s+(?:mua\s+)?nhà"],
    "credit_card_cash_advance_apr_percent": [
        r"cash\s+advance", r"rút\s+tiền\s+mặt.{0,20}thẻ",
    ],
}
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _load_ground_truth() -> dict:
    """Nạp data/pii_hallucination_samples.json (ground_truth), cache lại."""
    global _GROUND_TRUTH_CACHE
    if _GROUND_TRUTH_CACHE is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "data" / "pii_hallucination_samples.json"
        )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _GROUND_TRUTH_CACHE = data.get("ground_truth") or {}
        except (FileNotFoundError, json.JSONDecodeError):
            _GROUND_TRUTH_CACHE = {}
    return _GROUND_TRUTH_CACHE


def check_grounding(response: str) -> dict:
    """Output Rails — lớp 'grounding'.

    Đối chiếu số % mà câu trả lời nêu ra (lãi suất tiết kiệm, vay...) với
    ground_truth thật trong ``data/pii_hallucination_samples.json``. Sai số
    liệu không phải lỗ hổng bảo mật nên hàm này KHÔNG chặn/sửa câu trả lời —
    chỉ gắn cờ để audit log / human review, đúng vai trò "grounding" + "human
    review" trong sơ đồ 3 tầng phòng thủ.

    Returns:
        dict với 'grounded' (bool) và 'mismatches' (list các field sai).
    """
    rates = _load_ground_truth().get("rates") or {}
    mismatches = []
    text = response or ""

    for field, keyword_patterns in _RATE_KEYWORDS.items():
        if field not in rates:
            continue
        for kw in keyword_patterns:
            m = re.search(kw, text, re.IGNORECASE)
            if not m:
                continue
            window = text[m.end(): m.end() + 60]
            pm = _PERCENT_RE.search(window)
            if not pm:
                continue
            claimed = float(pm.group(1))
            actual = float(rates[field])
            if abs(claimed - actual) > 0.05:
                mismatches.append({
                    "field": field, "claimed": claimed, "actual": actual,
                })
            break

    return {"grounded": len(mismatches) == 0, "mismatches": mismatches}


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    redacted = _canonicalize_for_filter(response)

    # PII patterns to check
    PII_PATTERNS = {
        "phone_vn": r"\b0\d{9,10}\b",
        "email": r"[\w.\-]+@[\w.\-]+\.[a-zA-Z]{2,}",
        "national_id": r"\b\d{12}\b|\b\d{9}\b",
        "api_key": r"sk-[a-zA-Z0-9-]+",
        "password": r"(?:password|pwd)\s*(?:is|[:=])?\s*[A-Za-z0-9!@#$%^&*_\-]{3,}",
        "internal_host": r"\b[\w-]+(?:\.[\w-]+)*\.internal(?::\d+)?\b",
    }

    for name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted, re.IGNORECASE)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        # Nâng cấp ngoài yêu cầu tối thiểu: Output Rails "grounding" + "human review"
        self.grounding_checked = 0
        self.grounding_mismatches = 0
        self.needs_human_review = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        filter_result = content_filter(response_text)
        if not filter_result["safe"]:
            self.redacted_count += 1
            self.needs_human_review += 1
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=filter_result["redacted"])],
            )

        # Grounding — không chặn/sửa câu trả lời (sai số liệu không phải lỗ
        # hổng bảo mật), chỉ gắn cờ cho audit/human review.
        self.grounding_checked += 1
        grounding_result = check_grounding(self._extract_text(llm_response))
        if not grounding_result["grounded"]:
            self.grounding_mismatches += 1
            self.needs_human_review += 1

        if self.use_llm_judge:
            current_text = self._extract_text(llm_response)
            judge_result = await llm_safety_check(current_text)
            if not judge_result["safe"]:
                self.blocked_count += 1
                llm_response.content = types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text="I cannot share internal system details. "
                        "How else can I help with your banking needs?"
                    )],
                )

        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
