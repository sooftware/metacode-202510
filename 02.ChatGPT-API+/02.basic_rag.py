import openai
import numpy as np
from typing import List
import os

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text: str) -> List[float]:
    """텍스트를 임베딩 벡터로 변환"""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text.replace("\n", " ")
    )
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """코사인 유사도 계산"""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 샘플 회사 문서들
documents = [
    "우리 회사는 연간 15일의 유급휴가를 제공합니다. 휴가 사용은 3일 전 신청이 필요합니다.",
    "출장비는 국내 1일 10만원, 해외 1일 20만원까지 지원됩니다. 영수증 제출이 필요합니다.",
    "재택근무는 주 2회까지 가능하며 팀장 승인이 필요합니다. 코어타임은 오전 10시-오후 4시입니다.",
    "건강검진은 매년 5월에 실시하며 가족 포함 지원됩니다. 지정 병원 이용시 무료입니다.",
    "회사 식당은 오전 11시 30분부터 오후 2시까지 운영됩니다. 직원 할인가는 5000원입니다."
]


def simple_rag_demo():
    """간단한 RAG 데모"""
    print("=== RAG 기본 개념 체험 ===\n")

    # 1단계: 모든 문서를 임베딩으로 변환
    print("1단계: 문서들을 임베딩으로 변환 중...")
    doc_embeddings = []
    for i, doc in enumerate(documents):
        embedding = get_embedding(doc)
        doc_embeddings.append(embedding)
        print(f"   문서 {i + 1} 임베딩 완료 (차원: {len(embedding)})")

    print("\n2단계: 질문 처리 시작")

    while True:
        # 사용자 질문 입력
        question = input("\n질문을 입력하세요 (종료: quit): ")
        if question.lower() == 'quit':
            break

        # 3단계: 질문을 임베딩으로 변환
        print("   질문 임베딩 생성 중...")
        question_embedding = get_embedding(question)

        # 4단계: 가장 유사한 문서 찾기
        print("   관련 문서 검색 중...")
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = cosine_similarity(question_embedding, doc_embedding)
            similarities.append((i, similarity, documents[i]))

        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 5단계: 가장 관련성 높은 문서 표시
        print("\n   🔍 검색 결과:")
        for i, (doc_idx, sim, doc) in enumerate(similarities[:3]):
            print(f"   {i + 1}. 유사도 {sim:.3f}: {doc}")

        # 6단계: RAG 답변 생성
        best_doc = similarities[0][2]

        rag_prompt = f"""다음 회사 문서를 참조하여 질문에 답변해주세요:

문서: {best_doc}

질문: {question}

답변:"""

        print("\n   🤖 RAG 답변 생성 중...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": rag_prompt}],
            temperature=0.1
        )

        answer = response.choices[0].message.content
        print(f"\n   💬 답변: {answer}")

        # 7단계: 일반 ChatGPT와 비교
        print("\n   📊 비교를 위한 일반 ChatGPT 답변:")
        normal_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
            temperature=0.1
        )

        normal_answer = normal_response.choices[0].message.content
        print(f"   💭 일반 답변: {normal_answer}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("환경변수에 OPENAI_API_KEY를 설정해주세요.")
        print("예: export OPENAI_API_KEY='your-api-key'")
        exit()

    simple_rag_demo()