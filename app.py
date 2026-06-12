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

# True  = API 호출 없이 더미 결과로 UI 확인
# False = 실제 Gemini API 호출
USE_DUMMY_RESULT = False


SYSTEM_PROMPT = """
당신은 한국 20대 패션 트렌드와 데일리룩을 잘 이해하는 전문 AI 패션 스타일리스트입니다.

사용자가 업로드한 OOTD 사진을 분석하여 쉽고 직관적인 패션 피드백을 제공합니다.

중요 규칙:
* 얼굴, 체형, 외모는 절대 평가하지 마세요.
* 의상, 컬러 조합, 핏, 실루엣, 레이어드, 신발, 액세서리만 분석하세요.
* 무조건 칭찬만 하지 말고 장점과 아쉬운 점을 모두 작성하세요.
* 개선 추천은 현재 사진에서 실제로 보이는 요소를 바탕으로 작성하세요.
* 개선 추천에는 옷의 핏, 기장, 비율, 실루엣 중 최소 2개 이상을 포함하세요.
* 긴 문단 금지.
* 각 항목은 짧고 직관적으로 작성하세요.
* 반드시 아래 형식을 그대로 따르세요.

코디 점수 기준:
* 컬러 조화 : 25점
* 핏 & 실루엣 : 25점
* 스타일 완성도 : 25점
* 활용성 : 25점

출력 형식:

# 👕 MoodFit OOTD 분석 결과

## ⭐ 코디 점수
88점 / 100점

## 🏷 스타일 키워드
#미니멀 #데일리룩 #캐주얼

## 🎨 컬러 조합
블랙 + 화이트 + 베이지

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
* 긴 문단 금지
* 각 항목은 짧고 직관적으로 작성
* 잘한 포인트는 3개의 bullet로 작성
* 개선 추천은 3개의 bullet로 작성
* 개선 추천 첫 번째 bullet은 가장 중요한 문제를 작성
* 얼굴, 체형, 외모 평가 금지
"""


DUMMY_RESULT = """
# 👕 MoodFit OOTD 분석 결과

## ⭐ 코디 점수
86점 / 100점

## 🏷 스타일 키워드
#미니멀 #데일리룩 #캐주얼

## 🎨 컬러 조합
블랙 + 화이트 + 베이지

## 😊 전체 분위기
깔끔하고 편안한 미니멀 데일리룩입니다.

## 👍 잘한 포인트
• 블랙과 베이지 조합이 안정적이에요
• 화이트 이너 레이어드가 자연스러워요
• 전체적으로 편안한 무드가 잘 살아나요

## 💡 개선 추천
• 상의 기장이 살짝 길어 비율이 내려가 보여요
• 조거 팬츠 밑단이 발목을 잡아 실루엣이 짧아 보여요
• 세미와이드 팬츠를 매치하면 균형이 더 좋아져요

## 📍 어울리는 장소
• 카페 나들이
• 친구와의 약속
• 가벼운 산책

## 📝 한 줄 총평
깔끔하지만 핏 조정이 필요한 룩
"""


CUSTOM_CSS = """
.gradio-container {
    max-width: 1280px !important;
}

.moodfit-wrap {
    padding: 10px;
}

.result-title {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 18px;
    color: #f5f5f5;
}

.card {
    background: #1f2128;
    color: #f3f3f3;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    border: 1px solid #343743;
}

.card-title {
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 8px;
    color: #ffffff;
}

.card-body {
    font-size: 15px;
    line-height: 1.65;
    white-space: pre-line;
    color: #d8dbe5;
}

.card-grid {
    display: grid;
    grid-template-columns: 1fr 1.2fr;
    gap: 14px;
    margin-bottom: 14px;
}

.card-grid .card {
    margin-bottom: 0;
}

.score-card-inner {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.score-num {
    font-size: 40px;
    font-weight: 900;
    color: #9ed27f;
}

.score-total {
    font-size: 20px;
    color: #d8dbe5;
}

.progress {
    width: 100%;
    height: 10px;
    background: #343743;
    border-radius: 999px;
    margin-top: 12px;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #7fbf6a, #b7e48a);
    border-radius: 999px;
}

.tag {
    display: inline-block;
    background: #343743;
    color: #f5f5f5;
    padding: 7px 13px;
    border-radius: 999px;
    margin: 7px 5px 0 0;
    font-size: 14px;
    font-weight: 700;
}

.style-summary-card {
    background: #1f2128;
}

.good-card {
    border-left: 6px solid #7fc86a;
    background: #202820;
    min-height: 185px;
}

.improve-card {
    border-left: 6px solid #f0b43c;
    background: #2b2417;
    min-height: 220px;
}

.mini-card {
    min-height: 130px;
}

.summary-card {
    border-left: 6px solid #8fcf7a;
    background: #1f2820;
    padding: 22px 24px;
}

.summary-card .card-title {
    font-size: 21px;
}

.summary-card .card-body {
    font-size: 17px;
    font-weight: 700;
    color: #f2f7ef;
}

.error-card {
    background: #2b1f1f;
    border-left: 6px solid #e74c3c;
}

button {
    border-radius: 14px !important;
    font-weight: 700 !important;
}

@media (max-width: 900px) {
    .card-grid {
        grid-template-columns: 1fr;
    }
}
"""


def pil_to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def extract_section(text: str, title: str) -> str:
    pattern = rf"##\s*.*?{re.escape(title)}\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, text, re.S)
    if not match:
        return ""
    return match.group(1).strip()


def clean_bullets(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_score(score_text: str) -> int:
    match = re.search(r"(\d+)", score_text)
    if not match:
        return 0
    return max(0, min(100, int(match.group(1))))


def make_tags(keyword_text: str) -> str:
    words = re.findall(r"#?[가-힣A-Za-z0-9]+", keyword_text)
    if not words:
        return keyword_text

    tags = []
    for word in words:
        word = word.replace("#", "")
        tags.append(f'<span class="tag">{word}</span>')
    return "".join(tags)


def markdown_to_card_html(text: str) -> str:
    score = extract_section(text, "코디 점수")
    keywords = extract_section(text, "스타일 키워드")
    colors = extract_section(text, "컬러 조합")
    mood = extract_section(text, "전체 분위기")
    good = extract_section(text, "잘한 포인트")
    improve = extract_section(text, "개선 추천")
    places = extract_section(text, "어울리는 장소")
    summary = extract_section(text, "한 줄 총평")

    score_num = extract_score(score)

    keywords = keywords or "#데일리룩 #캐주얼"
    colors = colors or "컬러 조합 분석 결과가 부족해요."
    mood = mood or "전체 분위기 분석 결과가 부족해요."
    good = good or "• 잘한 점 분석 결과가 부족해요."
    improve = improve or "• 개선 추천 결과가 부족해요."
    places = places or "• 어울리는 장소 분석 결과가 부족해요."
    summary = summary or "분석 결과를 다시 확인해 주세요."

    return f"""
    <div class="moodfit-wrap">
        <div class="result-title">👕 MoodFit OOTD 분석 결과</div>

        <div class="card">
            <div class="score-card-inner">
                <div>
                    <div class="card-title">⭐ 코디 완성도</div>
                    <div class="card-body">오늘의 스타일 점수</div>
                </div>
                <div>
                    <span class="score-num">{score_num}</span>
                    <span class="score-total">/ 100</span>
                </div>
            </div>
            <div class="progress">
                <div class="progress-bar" style="width: {score_num}%;"></div>
            </div>
        </div>

        <div class="card style-summary-card">
            <div class="card-title">😊 스타일 요약</div>
            <div class="card-body">
                {mood}
                <br>
                {make_tags(keywords)}
            </div>
        </div>

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

        <div class="card-grid">
            <div class="card mini-card">
                <div class="card-title">🎨 컬러 조합</div>
                <div class="card-body">{colors}</div>
            </div>

            <div class="card mini-card">
                <div class="card-title">📍 어울리는 장소</div>
                <div class="card-body">{clean_bullets(places)}</div>
            </div>
        </div>

        <div class="card summary-card">
            <div class="card-title">💬 AI 총평</div>
            <div class="card-body">"{summary}"</div>
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
                "temperature": 0.5,
                "top_p": 0.9,
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


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="MoodFit — AI OOTD Feedback",
    ) as demo:
        gr.Markdown("""
            # 👕 MoodFit — AI OOTD Feedback
            OOTD 사진을 업로드하면 AI가 코디 분위기와 개선 포인트를 분석해드려요.
            """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="OOTD 사진 업로드",
                )
                submit_button = gr.Button("AI 코디 분석하기")

            with gr.Column(scale=1):
                output = gr.HTML("""
                    <div class="card">
                        <div class="card-title">분석 결과</div>
                        <div class="card-body">사진을 업로드한 뒤 분석 버튼을 눌러주세요.</div>
                    </div>
                    """)

        submit_button.click(
            fn=analyze_outfit,
            inputs=image_input,
            outputs=output,
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    is_space = bool(os.getenv("SPACE_ID"))

    demo.launch(
        server_name="0.0.0.0" if is_space else "127.0.0.1",
        server_port=int(os.getenv("PORT", 7860)),
        show_api=False,
        ssr_mode=False,
    )
