from __future__ import annotations

import os
import re
from io import BytesIO

import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY가 없습니다. .env 파일에 GEMINI_API_KEY=발급받은키 를 입력하세요."
    )

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

USE_DUMMY_RESULT = os.getenv("USE_DUMMY", "false").lower() == "true"

AUTH_ENABLED = False
AUTH_USERNAME = os.getenv("MOODFIT_AUTH_USER", "moodfit")
AUTH_PASSWORD = os.getenv("MOODFIT_AUTH_PASSWORD", "1234")


SYSTEM_PROMPT = """
당신은 한국 20대 패션 트렌드와 데일리룩을 잘 이해하는 전문 AI 패션 스타일리스트입니다.

사용자가 업로드한 OOTD 사진을 분석하여 쉽고 직관적인 패션 피드백을 제공합니다.

중요 규칙:
* 얼굴, 체형, 외모는 절대 평가하지 마세요.
* 의상, 컬러 조합, 핏, 실루엣, 레이어드, 신발, 액세서리만 분석하세요.
* 무조건 칭찬만 하지 말고 장점과 아쉬운 점을 모두 작성하세요.
* 개선 추천은 현재 사진에서 실제로 보이는 요소를 바탕으로 작성하세요.
* 긴 문단 금지.
* 각 항목은 짧고 직관적으로 작성하세요.
* 반드시 아래 형식을 그대로 따르세요.

점수 산정 기준:
* 컬러 조화: 0~25점
* 핏 & 실루엣: 0~25점
* 무드 일관성: 0~25점
* 활용성: 0~25점
* 총점은 위 4개 항목의 합산으로 계산하세요.
* 사진마다 실제 코디 상태에 따라 점수를 다르게 주세요.
* 모든 사진에 같은 점수를 반복하지 마세요.
* 총점은 60~96점 범위 안에서 현실적으로 부여하세요.

출력 형식:

# 👕 MoodFit OOTD 분석 결과

## ⭐ 코디 점수
총점: 숫자점 / 100점
컬러 조화: 숫자점 / 25점
핏 & 실루엣: 숫자점 / 25점
무드 일관성: 숫자점 / 25점
활용성: 숫자점 / 25점

## 🏷 스타일 키워드
#키워드1 #키워드2 #키워드3

## 🎨 컬러 조합
대표 컬러 + 대표 컬러 + 대표 컬러

## 😊 전체 분위기
한 문장으로 요약

## 👍 잘한 포인트
• 장점 1
• 장점 2
• 장점 3

## 💡 개선 추천
• 가장 중요한 개선점 1
• 핏/기장/비율 관련 개선점 2
• 실제 스타일링 수정 제안 3

## 📍 어울리는 장소
• 장소 1
• 장소 2
• 장소 3

## 📝 한 줄 총평
15자~25자 이내의 짧고 센스 있는 한 문장
"""


USER_PROMPT = """
업로드된 OOTD 사진을 분석하세요.

반드시 아래 항목을 모두 작성하세요.

1. 코디 점수
2. 스타일 키워드
3. 컬러 조합
4. 전체 분위기
5. 잘한 포인트
6. 개선 추천
7. 어울리는 장소
8. 한 줄 총평

주의:
* 점수는 이미지의 실제 코디 상태에 따라 다르게 산정하세요.
* 총점은 컬러 조화, 핏 & 실루엣, 무드 일관성, 활용성 점수의 합산이어야 합니다.
* 긴 문단 금지
* 각 항목은 짧고 직관적으로 작성
* 잘한 포인트는 3개의 bullet로 작성
* 개선 추천은 3개의 bullet로 작성
* 얼굴, 체형, 외모 평가 금지
"""


DUMMY_RESULT = """
# 👕 MoodFit OOTD 분석 결과

## ⭐ 코디 점수
총점: 84점 / 100점
컬러 조화: 22점 / 25점
핏 & 실루엣: 19점 / 25점
무드 일관성: 22점 / 25점
활용성: 21점 / 25점

## 🏷 스타일 키워드
#캐주얼 #데일리룩 #미니멀

## 🎨 컬러 조합
블루 + 화이트 + 베이지

## 😊 전체 분위기
편안하고 산뜻한 무드의 데일리 캐주얼룩입니다.

## 👍 잘한 포인트
• 셔츠와 팬츠의 컬러 조합이 부드럽고 안정적이에요
• 레이어드가 자연스러워 데일리룩 느낌이 잘 살아나요
• 운동화 선택으로 편안하고 활동적인 분위기가 더해졌어요

## 💡 개선 추천
• 팬츠 기장이 발등을 많이 덮어 전체 비율이 조금 무거워 보여요
• 셔츠 앞부분을 살짝 넣으면 허리선이 정리돼 실루엣이 좋아져요
• 작은 크로스백이나 시계를 더하면 스타일 포인트가 살아나요

## 📍 어울리는 장소
• 주말 나들이
• 카페 데이트
• 캠퍼스룩

## 📝 한 줄 총평
편안함이 돋보이는 산뜻한 데일리룩
"""


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800;900&display=swap');

/* ── 리셋 & 기본 ── */
* {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    box-sizing: border-box;
}

html, body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    min-height: 100vh !important;
    overflow-x: hidden !important;
}

/* ── 배경: 아이보리 + 세이지 그린 그러데이션 ── */
body {
    background:
        radial-gradient(circle at top left, rgba(168, 213, 162, 0.22), transparent 36%),
        radial-gradient(circle at bottom right, rgba(200, 230, 192, 0.18), transparent 34%),
        linear-gradient(150deg, #f7faf5 0%, #f0f5ee 50%, #e9f0e6 100%) !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 100% !important;
    min-height: 100vh !important;
    margin: 0 auto !important;
    background: transparent !important;
    color: #1e2d1e !important;
}

footer { display: none !important; }

/* ── 레이아웃 쉘 ── */
.main-shell,
.content-shell {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto !important;
    padding-left: 28px;
    padding-right: 28px;
}

.main-shell {
    padding-top: 44px;
    padding-bottom: 28px;
}

.content-shell {
    padding-bottom: 56px;
}

/* ── 히어로 카드 ── */
.hero-card {
    position: relative;
    overflow: hidden;
    border-radius: 36px;
    padding: 40px 44px;
    background:
        linear-gradient(135deg, rgba(27, 42, 27, 0.97) 0%, rgba(52, 82, 52, 0.95) 100%),
        radial-gradient(circle at 85% 15%, rgba(168, 213, 162, 0.35), transparent 30%);
    color: #f0faf0;
    box-shadow: 0 24px 64px rgba(30, 60, 30, 0.28);
}

/* 장식 원 */
.hero-card::before {
    content: "";
    position: absolute;
    right: -60px;
    top: -90px;
    width: 260px;
    height: 260px;
    border-radius: 999px;
    background: rgba(168, 213, 162, 0.14);
    pointer-events: none;
}

.hero-card::after {
    content: "";
    position: absolute;
    right: 80px;
    bottom: -70px;
    width: 180px;
    height: 180px;
    border-radius: 999px;
    background: rgba(120, 180, 120, 0.10);
    pointer-events: none;
}

.hero-kicker {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.18em;
    color: #a8d5a2;
    margin-bottom: 12px;
    text-transform: uppercase;
}

.hero-title {
    font-size: 38px;
    line-height: 1.12;
    font-weight: 900;
    margin-bottom: 14px;
    letter-spacing: -0.04em;
    color: #ffffff;
    text-shadow: 0 2px 16px rgba(0, 0, 0, 0.35);
}

.hero-title em {
    font-style: normal;
    color: #a8d5a2;
}

.hero-desc {
    font-size: 15px;
    line-height: 1.7;
    color: #c8e6c0;
    max-width: 680px;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 24px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 999px;
    background: rgba(168, 213, 162, 0.15);
    border: 1px solid rgba(168, 213, 162, 0.28);
    font-size: 13px;
    font-weight: 700;
    color: #e0f5e0;
    backdrop-filter: blur(6px);
}

/* ── 워크스페이스 ── */
.workspace {
    width: 100% !important;
    margin: 0 auto !important;
    gap: 28px !important;
    align-items: flex-start !important;
}

/* ── 패널 공통 ── */
.upload-panel,
.result-panel {
    border-radius: 32px !important;
    padding: 26px !important;
    background: rgba(247, 252, 246, 0.90) !important;
    border: 1px solid rgba(100, 160, 100, 0.14) !important;
    box-shadow: 0 20px 52px rgba(40, 80, 40, 0.12) !important;
    backdrop-filter: blur(14px);
}

.panel-title {
    font-size: 22px;
    font-weight: 900;
    color: #1e2d1e;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.panel-subtitle {
    font-size: 14px;
    line-height: 1.6;
    color: #5a7a5a;
    margin-bottom: 18px;
}

/* 이미지 업로드 영역 — 커스텀 */
.custom-upload-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 3 / 4;
    border-radius: 22px;
    overflow: hidden;
    cursor: pointer;
    background: #2a422a;
    border: 2px dashed rgba(168, 213, 162, 0.35);
    transition: border-color 0.2s ease, background 0.2s ease;
    margin-bottom: 14px;
}

.custom-upload-wrap:hover,
.custom-upload-wrap.drag-over {
    border-color: rgba(168, 213, 162, 0.7);
    background: #2f4a2f;
}

.upload-placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    pointer-events: none;
}

.upload-icon {
    font-size: 32px;
    color: #a8d5a2;
    margin-bottom: 4px;
}

.upload-text-main {
    font-size: 15px;
    font-weight: 700;
    color: #a8d5a2;
}

.upload-text-sub {
    font-size: 13px;
    color: rgba(168, 213, 162, 0.65);
}

.preview-img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 20px;
    display: block;
}

.preview-note {
    margin-top: 14px;
    border-radius: 18px;
    padding: 14px 16px;
    background: #edf5eb;
    color: #4a6e4a;
    font-size: 13.5px;
    line-height: 1.55;
    border: 1px solid #c8e0c4;
}

/* ── 버튼 ── */
button {
    border-radius: 18px !important;
    min-height: 50px !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    border: none !important;
    background: linear-gradient(135deg, #2d4a2d, #4a7c59) !important;
    color: #f0faf0 !important;
    box-shadow: 0 10px 28px rgba(40, 90, 50, 0.28) !important;
    transition: transform 0.15s ease, filter 0.15s ease !important;
}

button:hover {
    transform: translateY(-2px) !important;
    filter: brightness(1.06) !important;
}

/* ── 결과 래퍼 ── */
.result-wrap { color: #1e2d1e; }

.result-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.result-title {
    font-size: 26px;
    font-weight: 950;
    letter-spacing: -0.04em;
    color: #1e2d1e;
}

.result-subtitle {
    font-size: 13px;
    color: #6a8e6a;
    margin-top: 4px;
}

.report-chip {
    flex: 0 0 auto;
    padding: 9px 14px;
    border-radius: 999px;
    background: #2d4a2d;
    color: #c8e6c0;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 0.08em;
}

/* ── 카드 공통 ── */
.card {
    background: rgba(255, 255, 255, 0.92);
    color: #1e2d1e;
    border-radius: 24px;
    padding: 22px 24px;
    margin-bottom: 14px;
    box-shadow: 0 8px 28px rgba(40, 80, 40, 0.08);
    border: 1px solid rgba(100, 160, 100, 0.12);
}

.card-title {
    font-size: 16px;
    font-weight: 900;
    margin-bottom: 10px;
    color: #1e2d1e;
    letter-spacing: -0.01em;
}

.card-body {
    font-size: 14.5px;
    line-height: 1.75;
    white-space: pre-line;
    color: #3d5c3d;
}

/* ── 점수 카드 ── */
.score-card {
    background: linear-gradient(135deg, #1b2a1b 0%, #2d5237 60%, #3a6645 100%);
    color: #f0faf0;
    padding: 26px 28px;
    border: none;
    box-shadow: 0 16px 48px rgba(30, 80, 40, 0.32);
}

.score-card .card-title,
.score-card .card-body {
    color: #c8e6c0;
}

.score-card-inner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    flex-wrap: nowrap;
}

.score-left { flex: 1; min-width: 0; }

.score-right {
    flex: 0 0 auto;
    display: flex;
    align-items: baseline;
    gap: 4px;
    white-space: nowrap;
}

.score-num {
    font-size: 64px;
    line-height: 1;
    font-weight: 950;
    color: #a8d5a2;
    letter-spacing: -0.06em;
}

.score-total {
    font-size: 20px;
    font-weight: 800;
    color: #c8e6c0;
}

/* 프로그레스 바 */
.progress {
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 999px;
    margin-top: 20px;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #6abf6a, #a8d5a2);
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* 하위 점수 그리드 */
.score-breakdown {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-top: 18px;
}

.score-mini {
    border-radius: 16px;
    padding: 14px 10px;
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(168, 213, 162, 0.20);
    text-align: center;
}

.score-mini-label {
    font-size: 11px;
    color: #a8d5a2;
    margin-bottom: 6px;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.score-mini-value {
    font-size: 17px;
    font-weight: 900;
    color: #e0f5e0;
}

/* ── 태그 ── */
.tag {
    display: inline-block;
    background: #e8f5e9;
    color: #2d5237;
    padding: 7px 14px;
    border-radius: 999px;
    margin: 6px 5px 0 0;
    font-size: 13px;
    font-weight: 800;
    border: 1px solid #c8e0c4;
}

/* ── 컬러 스와치 ── */
.color-swatch-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 6px;
}

.color-swatch {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 13.5px;
    font-weight: 700;
    color: #3d5c3d;
}

.color-dot {
    width: 20px;
    height: 20px;
    border-radius: 999px;
    border: 2px solid rgba(0,0,0,0.08);
    flex-shrink: 0;
}

.color-sep {
    color: #a0c4a0;
    font-size: 16px;
    font-weight: 300;
}

/* ── 카드 그리드 ── */
.card-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 14px;
    align-items: stretch;
}

.card-grid .card {
    margin-bottom: 0;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.card-grid .card .card-body {
    flex: 1;
}

/* 잘한 점 / 개선 추천 */
.good-card {
    background: #f2faf0;
    border-left: 6px solid #6abf6a;
}

.improve-card {
    background: #fffcf0;
    border-left: 6px solid #e8c84a;
}

/* 컬러 팔레트 / 어울리는 장소 */
.mini-card { min-height: 130px; }

.color-card {
    background: #f5faf4;
    border-color: rgba(100, 180, 100, 0.16);
}

.place-card {
    background: #f4f8ff;
    border-color: rgba(100, 130, 200, 0.14);
}

/* ── 한 줄 총평 ── */
.summary-card {
    background: linear-gradient(135deg, #edf7eb, #f5faf4);
    border: 1px solid #c0dcc0;
    padding: 26px 28px;
    position: relative;
    overflow: hidden;
}

.summary-card::before {
    content: '\201C';
    position: absolute;
    top: -8px;
    left: 18px;
    font-size: 80px;
    line-height: 1;
    color: rgba(100, 180, 100, 0.18);
    font-family: Georgia, serif;
    pointer-events: none;
}

.summary-card .card-title {
    font-size: 18px;
    color: #2d4a2d;
}

.summary-card .card-body {
    font-size: 20px;
    font-weight: 900;
    color: #1e3a1e;
    font-style: italic;
    letter-spacing: -0.02em;
    padding-left: 4px;
}

/* ── 빈 상태 ── */
.empty-card {
    min-height: 500px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    border: 2px dashed rgba(80, 160, 80, 0.22);
    background: rgba(240, 250, 240, 0.55);
    border-radius: 24px;
}

.empty-icon {
    font-size: 48px;
    margin-bottom: 16px;
}

.empty-title {
    font-size: 22px;
    font-weight: 950;
    color: #2d4a2d;
    margin-bottom: 10px;
    letter-spacing: -0.02em;
}

.empty-body {
    color: #5a7a5a;
    line-height: 1.75;
    font-size: 14px;
}

/* ── 에러 카드 ── */
.error-card {
    background: #fff5f5;
    border-left: 6px solid #e05555;
}

/* "새 코디 분석하기" 버튼 - 아웃라인 스타일 */
.reset-btn {
    background: transparent !important;
    border: 2px solid #4a7c59 !important;
    color: #2d4a2d !important;
    box-shadow: none !important;
    font-size: 14px !important;
    min-height: 42px !important;
    margin-top: 8px !important;
}

.reset-btn:hover {
    background: #edf7eb !important;
    transform: translateY(-1px) !important;
}

/* ── 반응형 ── */
@media (max-width: 1000px) {
    .main-shell,
    .content-shell {
        padding-left: 16px;
        padding-right: 16px;
    }

    .main-shell { padding-top: 20px; padding-bottom: 20px; }

    .hero-card {
        padding: 30px 26px;
        border-radius: 28px;
    }

    .hero-title { font-size: 30px; }

    .workspace { flex-direction: column; }

    .card-grid { grid-template-columns: 1fr; }

    .score-breakdown { grid-template-columns: repeat(2, 1fr); }

    .result-title-row,
    .score-card-inner { flex-direction: column; }

    .score-num { font-size: 52px; }
}
"""


# 컬러 이름 → HEX 매핑 (주요 한국어/영어 색상)
COLOR_MAP = {
    "블랙": "#1a1a1a",
    "화이트": "#f5f5f5",
    "white": "#f5f5f5",
    "black": "#1a1a1a",
    "베이지": "#d4b896",
    "베이쥬": "#d4b896",
    "크림": "#f2ead8",
    "아이보리": "#f5f0e0",
    "그레이": "#9e9e9e",
    "회색": "#9e9e9e",
    "차콜": "#3c3c3c",
    "네이비": "#1a2a4a",
    "블루": "#4a7ab5",
    "스카이블루": "#87ceeb",
    "카키": "#8a8a5a",
    "올리브": "#6b7c3a",
    "그린": "#4a8a5a",
    "민트": "#7ec8b0",
    "브라운": "#7a5a3a",
    "카멜": "#c49a6c",
    "탄": "#c4a882",
    "레드": "#c0392b",
    "버건디": "#800020",
    "와인": "#722f37",
    "핑크": "#e8a0a0",
    "라벤더": "#b0a0d0",
    "퍼플": "#7a5a9a",
    "옐로우": "#e8c84a",
    "머스타드": "#c8a830",
    "오렌지": "#e8843a",
    "라이트블루": "#a0c4e8",
    "다크네이비": "#0a1428",
    "딥그린": "#1a4a2a",
}


def get_color_hex(color_name: str) -> str:
    name = color_name.strip().lower()
    for key, val in COLOR_MAP.items():
        if key.lower() in name or name in key.lower():
            return val
    return "#cccccc"


def pil_to_bytes(image: Image.Image) -> bytes:
    image = image.copy()
    image.thumbnail((1280, 1280), Image.LANCZOS)
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def extract_section(text: str, title: str) -> str:
    pattern = rf"##\s*[^\n]*?{re.escape(title)}\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, text, re.S)
    if not match:
        return ""
    return match.group(1).strip()


def clean_bullets(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_score(score_text: str) -> int:
    patterns = [
        r"총점\s*[:：]?\s*(\d+)",
        r"(\d+)\s*점\s*/\s*100",
        r"(\d+)\s*/\s*100",
        r"(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, score_text)
        if match:
            return max(0, min(100, int(match.group(1))))
    return 0


def extract_sub_score(score_text: str, label: str) -> int:
    pattern = rf"{re.escape(label)}\s*[:：]?\s*(\d+)"
    match = re.search(pattern, score_text)
    if not match:
        return 0
    return max(0, min(25, int(match.group(1))))


def make_tags(keyword_text: str) -> str:
    words = re.findall(r"#?[가-힣A-Za-z0-9]+", keyword_text)
    if not words:
        return keyword_text
    tags = []
    for word in words[:5]:
        word = word.replace("#", "")
        tags.append(f'<span class="tag">#{word}</span>')
    return "".join(tags)


def make_color_swatches(color_text: str) -> str:
    """컬러 조합 텍스트를 색상 스와치 HTML로 변환"""
    parts = re.split(r"\s*\+\s*", color_text.strip())
    if not parts or len(parts) < 2:
        return f'<div class="card-body">{color_text}</div>'

    items = []
    for i, part in enumerate(parts):
        part = part.strip()
        hex_color = get_color_hex(part)
        items.append(
            f'<div class="color-swatch">'
            f'<div class="color-dot" style="background:{hex_color};"></div>'
            f"{part}"
            f"</div>"
        )
        if i < len(parts) - 1:
            items.append('<span class="color-sep">+</span>')

    return f'<div class="color-swatch-row">{"".join(items)}</div>'


def markdown_to_card_html(text: str) -> str:
    score = extract_section(text, "코디 점수")
    keywords = extract_section(text, "스타일 키워드")
    colors = extract_section(text, "컬러 조합")
    mood = extract_section(text, "전체 분위기")
    good = extract_section(text, "잘한 포인트")
    improve = extract_section(text, "개선 추천")
    places = extract_section(text, "어울리는 장소")
    summary = extract_section(text, "한 줄 총평")

    color_score = extract_sub_score(score, "컬러 조화")
    fit_score = extract_sub_score(score, "핏 & 실루엣")
    mood_score = extract_sub_score(score, "무드 일관성")
    practical_score = extract_sub_score(score, "활용성")

    sub_total = color_score + fit_score + mood_score + practical_score
    parsed_total = extract_score(score)
    score_num = sub_total if sub_total > 0 else parsed_total

    keywords = keywords or "#데일리룩 #캐주얼 #미니멀"
    colors = colors or "화이트 + 베이지 + 그레이"
    mood = mood or "전체 분위기 분석 결과가 부족해요."
    good = good or "• 잘한 점 분석 결과가 부족해요."
    improve = improve or "• 개선 추천 결과가 부족해요."
    places = places or "• 어울리는 장소 분석 결과가 부족해요."
    summary = summary or "분석 결과를 다시 확인해 주세요."

    color_swatches_html = make_color_swatches(colors)

    return f"""
    <div class="result-wrap">
        <div class="result-title-row">
            <div>
                <div class="result-title">MoodFit Style Report</div>
                <div class="result-subtitle">AI가 분석한 오늘의 OOTD 피드백</div>
            </div>
            <div class="report-chip">OOTD ANALYSIS</div>
        </div>

        <!-- 점수 카드 -->
        <div class="card score-card">
            <div class="score-card-inner">
                <div class="score-left">
                    <div class="card-title">⭐ 코디 완성도</div>
                    <div class="card-body">컬러 · 핏 · 무드 · 활용성 종합</div>
                </div>
                <div class="score-right">
                    <span class="score-num">{score_num}</span>
                    <span class="score-total">/ 100</span>
                </div>
            </div>

            <div class="progress">
                <div class="progress-bar" style="width: {score_num}%;"></div>
            </div>

            <div class="score-breakdown">
                <div class="score-mini">
                    <div class="score-mini-label">컬러 조화</div>
                    <div class="score-mini-value">{color_score}/25</div>
                </div>
                <div class="score-mini">
                    <div class="score-mini-label">핏 & 실루엣</div>
                    <div class="score-mini-value">{fit_score}/25</div>
                </div>
                <div class="score-mini">
                    <div class="score-mini-label">무드 일관성</div>
                    <div class="score-mini-value">{mood_score}/25</div>
                </div>
                <div class="score-mini">
                    <div class="score-mini-label">활용성</div>
                    <div class="score-mini-value">{practical_score}/25</div>
                </div>
            </div>
        </div>

        <!-- 스타일 무드 -->
        <div class="card">
            <div class="card-title">😊 스타일 무드</div>
            <div class="card-body">
                {mood}
                <br>
                {make_tags(keywords)}
            </div>
        </div>

        <!-- 잘한 점 / 개선 추천 -->
        <div class="card-grid">
            <div class="card good-card">
                <div class="card-title">👍 잘한 점</div>
                <div class="card-body">{clean_bullets(good)}</div>
            </div>
            <div class="card improve-card">
                <div class="card-title">💡 개선 추천</div>
                <div class="card-body">{clean_bullets(improve)}</div>
            </div>
        </div>

        <!-- 컬러 팔레트 / 어울리는 장소 -->
        <div class="card-grid">
            <div class="card mini-card color-card">
                <div class="card-title">🎨 컬러 팔레트</div>
                {color_swatches_html}
            </div>
            <div class="card mini-card place-card">
                <div class="card-title">📍 어울리는 장소</div>
                <div class="card-body">{clean_bullets(places)}</div>
            </div>
        </div>

        <!-- 한 줄 총평 -->
        <div class="card summary-card">
            <div class="card-title">💬 AI 한 줄 총평</div>
            <div class="card-body">{summary}</div>
        </div>
    </div>
    """


def analyze_outfit(image: Image.Image | None) -> str:
    if image is None:
        return """
        <div class="card error-card">
            <div class="card-title">이미지가 필요해요</div>
            <div class="card-body">OOTD 사진을 먼저 업로드해 주세요.</div>
        </div>
        """

    if USE_DUMMY_RESULT:
        return markdown_to_card_html(DUMMY_RESULT)

    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=SYSTEM_PROMPT,
        )

        image_bytes = pil_to_bytes(image)

        response = model.generate_content(
            [
                {
                    "mime_type": "image/jpeg",
                    "data": image_bytes,
                },
                USER_PROMPT,
            ],
            generation_config={
                "temperature": 0.85,
                "top_p": 0.95,
                "max_output_tokens": 1800,
            },
        )

        if not response.text:
            return """
            <div class="card error-card">
                <div class="card-title">분석 실패</div>
                <div class="card-body">분석 결과를 생성하지 못했습니다. 다른 이미지를 업로드해 주세요.</div>
            </div>
            """

        return markdown_to_card_html(response.text)

    except Exception as e:
        error_text = str(e)

        if "429" in error_text or "ResourceExhausted" in error_text:
            return """
            <div class="card error-card">
                <div class="card-title">요청 한도 초과</div>
                <div class="card-body">요청이 잠시 많아졌어요. 잠시 후 다시 시도해 주세요.</div>
            </div>
            """

        return f"""
        <div class="card error-card">
            <div class="card-title">분석 중 오류 발생</div>
            <div class="card-body">{type(e).__name__}: {str(e)}</div>
        </div>
        """


EMPTY_HTML = """
<div class="empty-card">
    <div>
        <div class="empty-icon">👗</div>
        <div class="empty-title">아직 분석 전이에요</div>
        <div class="empty-body">
            왼쪽에 OOTD 사진을 업로드한 뒤<br>
            <b>AI 스타일 리포트 생성하기</b> 버튼을 눌러주세요.
        </div>
    </div>
</div>
"""


def analyze_from_b64(b64_data: str) -> str:
    """base64 문자열을 PIL Image로 변환 후 분석"""
    import base64

    if not b64_data:
        return """
        <div class="card error-card">
            <div class="card-title">이미지가 필요해요</div>
            <div class="card-body">OOTD 사진을 먼저 업로드해 주세요.</div>
        </div>
        """
    try:
        # data:image/jpeg;base64,... 형식에서 실제 데이터만 추출
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        return analyze_outfit(image)
    except Exception as e:
        return f"""
        <div class="card error-card">
            <div class="card-title">이미지 처리 오류</div>
            <div class="card-body">{str(e)}</div>
        </div>
        """


UPLOADER_JS = """
() => {
    // 전역 변수에 base64 저장 (Textbox 브릿지 대신 사용)
    window._moodfit_b64 = '';

    function initUploader() {
        const wrap = document.getElementById('uploadWrap');
        const input = document.getElementById('fileInput');
        const placeholder = document.getElementById('uploadPlaceholder');
        const preview = document.getElementById('previewImg');
        if (!wrap || !input || wrap._moodfit_init) return;
        wrap._moodfit_init = true;

        function loadFile(file) {
            if (!file || !file.type.startsWith('image/')) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                const b64 = e.target.result;
                window._moodfit_b64 = b64;
                preview.src = b64;
                preview.style.display = 'block';
                placeholder.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        wrap.addEventListener('click', function(e) {
            if (e.target === preview) return;
            input.click();
        });
        input.addEventListener('change', function() {
            if (input.files[0]) loadFile(input.files[0]);
        });
        wrap.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            wrap.classList.add('drag-over');
        });
        wrap.addEventListener('dragleave', function() {
            wrap.classList.remove('drag-over');
        });
        wrap.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            wrap.classList.remove('drag-over');
            if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
        });
    }

    const observer = new MutationObserver(function() {
        if (document.getElementById('uploadWrap')) initUploader();
    });
    observer.observe(document.body, { childList: true, subtree: true });

    initUploader();
    setTimeout(initUploader, 500);
    setTimeout(initUploader, 1500);
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="MoodFit — AI OOTD Feedback",
        theme=gr.themes.Soft(),
        js=UPLOADER_JS,
    ) as demo:

        # 업로드된 이미지 base64를 저장하는 state
        image_state = gr.State("")

        gr.HTML("""
        <div class="main-shell">
            <section class="hero-card">
                <div class="hero-kicker">AI OOTD Feedback Service</div>
                <div class="hero-title">오늘 입은 코디,<br><em>AI</em>가 스타일 리포트로 분석해드려요.</div>
                <div class="hero-desc">
                    MoodFit은 지금 입은 OOTD 사진 한 장으로 컬러 조합, 핏, 분위기, 개선 포인트를 빠르게 피드백하는 AI 패션 분석 서비스입니다.
                </div>
                <div class="hero-badges">
                    <span class="hero-badge">👕 OOTD 분석</span>
                    <span class="hero-badge">🎨 컬러 조합</span>
                    <span class="hero-badge">⭐ 스타일 점수</span>
                    <span class="hero-badge">💡 개선 추천</span>
                </div>
            </section>
        </div>
        """)

        with gr.Column(elem_classes=["content-shell"]):
            with gr.Row(elem_classes=["workspace"]):
                with gr.Column(scale=1, elem_classes=["upload-panel"]):
                    gr.HTML("""
                    <div class="panel-title">Upload Your OOTD</div>
                    <div class="panel-subtitle">
                        전신 또는 상하의가 잘 보이는 사진을 업로드하면 더 정확한 피드백을 받을 수 있어요.
                    </div>

                    <!-- 커스텀 업로드 영역 -->
                    <div class="custom-upload-wrap" id="uploadWrap">
                        <div class="upload-placeholder" id="uploadPlaceholder">
                            <div class="upload-icon">⬆</div>
                            <div class="upload-text-main">사진을 여기에 드롭하거나</div>
                            <div class="upload-text-sub">클릭하여 업로드</div>
                        </div>
                        <img id="previewImg" class="preview-img" style="display:none;" alt="preview"/>
                        <input type="file" id="fileInput" accept="image/*" style="display:none;"/>
                    </div>
                    """)

                    # 숨겨진 Textbox: JS → Python 브릿지
                    image_b64_input = gr.Textbox(
                        value="",
                        visible=False,
                        elem_id="image-state-input",
                    )

                    submit_button = gr.Button("✦ AI 스타일 리포트 생성하기")
                    reset_button = gr.Button(
                        "↺ 새 코디 분석하기", elem_classes=["reset-btn"]
                    )

                    gr.HTML("""
                    <div class="preview-note">
                        <b>Tip.</b> 얼굴이나 외모는 평가하지 않고, 의상·컬러·핏·실루엣 중심으로만 분석합니다.
                    </div>
                    """)

                with gr.Column(scale=1, elem_classes=["result-panel"]):
                    output = gr.HTML(EMPTY_HTML)

        def do_reset():
            return ("", EMPTY_HTML)

        # JS가 window._moodfit_b64를 반환 → image_b64_input에 저장 → analyze_from_b64 호출
        submit_button.click(
            fn=lambda b64: b64,
            inputs=image_b64_input,
            outputs=image_b64_input,
            js="() => [window._moodfit_b64 || '']",
        ).then(
            fn=analyze_from_b64,
            inputs=image_b64_input,
            outputs=output,
        )

        reset_button.click(
            fn=do_reset,
            inputs=[],
            outputs=[image_b64_input, output],
            js="""() => {
                window._moodfit_b64 = '';
                const preview = document.getElementById('previewImg');
                const placeholder = document.getElementById('uploadPlaceholder');
                const fileInput = document.getElementById('fileInput');
                if (preview) { preview.src = ''; preview.style.display = 'none'; }
                if (placeholder) { placeholder.style.display = 'flex'; }
                if (fileInput) fileInput.value = '';
                return [];
            }""",
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    is_space = bool(os.getenv("SPACE_ID"))

    launch_kwargs = {
        "server_name": "0.0.0.0" if is_space else "127.0.0.1",
        "server_port": int(os.getenv("PORT", 7860)),
        "show_api": False,
        "ssr_mode": False,
    }

    if AUTH_ENABLED:
        launch_kwargs["auth"] = (AUTH_USERNAME, AUTH_PASSWORD)

    demo.launch(**launch_kwargs)
