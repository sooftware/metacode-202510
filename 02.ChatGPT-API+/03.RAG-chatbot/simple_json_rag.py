import json
import openai
import numpy as np
from typing import List, Dict, Any
import os


class PersonalInfoRAG:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text.replace("\n", " ")
        )
        return response.data[0].embedding

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def load_json_file(self, json_file: str):
        """JSON 파일에서 데이터 로드 및 임베딩 생성"""
        print(f"파일 {json_file} 로딩 중...")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 각 카테고리의 아이템들을 문서로 변환
        for category, items in data.items():
            if isinstance(items, list):
                for item in items:
                    # 제목과 내용을 결합
                    text = f"{item.get('title', '')}\n{item.get('content', '')}"

                    # 임베딩 생성
                    embedding = self.get_embedding(text)

                    # 저장
                    self.documents.append(text)
                    self.embeddings.append(embedding)
                    self.metadata.append({
                        'category': category,
                        'title': item.get('title', 'Unknown'),
                        'id': item.get('id', f"{category}_{len(self.documents)}")
                    })

        print(f"총 {len(self.documents)}개 문서 임베딩 완료")

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """유사한 문서 검색"""
        if not self.embeddings:
            return []

        # 질문 임베딩
        query_embedding = self.get_embedding(query)

        # 유사도 계산
        results = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            results.append({
                'document': self.documents[i],
                'similarity': similarity,
                'metadata': self.metadata[i]
            })

        # 정렬 후 상위 결과 반환
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def generate_answer(self, question: str, search_results: List[Dict]) -> str:
        """RAG 기반 답변 생성"""
        if not search_results:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."

        # 검색된 문서들을 컨텍스트로 구성
        context = "\n\n".join([result['document'] for result in search_results])

        prompt = f"""다음 정보를 바탕으로 질문에 친근하게 답변해주세요:

정보:
{context}

질문: {question}

답변:"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    def chat(self):
        """대화형 인터페이스"""
        print("\n개인 정보 챗봇이 준비되었습니다!")
        print("팁: 'quit' 입력시 종료됩니다.\n")

        while True:
            question = input("질문: ")
            if question.lower() in ['quit', 'exit', 'q']:
                print("대화를 종료합니다!")
                break

            # 검색 및 답변
            print("관련 정보 검색 중...")
            search_results = self.search(question)

            print("답변 생성 중...")
            answer = self.generate_answer(question, search_results)

            print(f"\n답변: {answer}\n")

            # 참고 문서 표시
            if search_results:
                print("참고한 정보:")
                for i, result in enumerate(search_results, 1):
                    print(
                        f"   {i}. [{result['metadata']['category']}] {result['metadata']['title']} (유사도: {result['similarity']:.2f})")
                print()


def create_sample_json():
    """샘플 JSON 파일 생성"""
    sample_data = {
        "personal_info": [
            {
                "id": "personal_001",
                "title": "기본 정보",
                "content": "이름: 김개발, 나이: 28세, 직업: 소프트웨어 개발자, 거주지: 서울시 강남구, 취미: 독서, 영화감상, 프로그래밍"
            },
            {
                "id": "personal_002",
                "title": "연락처 정보",
                "content": "이메일: kim.dev@email.com, 전화번호: 010-1234-5678, 회사: ABC테크, 부서: 개발팀"
            }
        ],
        "preferences": [
            {
                "id": "pref_001",
                "title": "음식 취향",
                "content": "좋아하는 음식: 한식, 이탈리안, 일식. 특히 김치찌개와 파스타를 좋아함. 매운 음식을 즐기며, 카페인 중독자로 하루에 커피 3잔은 필수"
            },
            {
                "id": "pref_002",
                "title": "여가 활동",
                "content": "주말에는 주로 독서를 하거나 영화를 본다. 최근에는 SF 소설과 스릴러 영화에 빠져있음. 가끔 친구들과 보드게임 카페도 간다"
            }
        ],
        "work_info": [
            {
                "id": "work_001",
                "title": "직장 정보",
                "content": "ABC테크에서 백엔드 개발자로 근무. 주로 Python과 Django를 사용하며, 최근에는 FastAPI도 배우고 있음. 팀은 총 5명이고 모두 친하다"
            },
            {
                "id": "work_002",
                "title": "업무 스케줄",
                "content": "출근시간은 오전 9시, 퇴근은 보통 6시. 재택근무는 주 2회 가능. 점심시간은 12시부터 1시까지. 회사에서 점심 제공됨"
            }
        ],
        "goals": [
            {
                "id": "goal_001",
                "title": "단기 목표",
                "content": "올해 안에 AWS 자격증 취득하기, FastAPI 마스터하기, 영어 회화 실력 늘리기. 매주 운동 3회 이상 하기"
            },
            {
                "id": "goal_002",
                "title": "장기 목표",
                "content": "3년 내에 시니어 개발자가 되기, 5년 내에 테크리드 역할 맡기, 언젠가는 스타트업 창업해보기"
            }
        ]
    }

    with open('my_info.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print("샘플 데이터 파일 'my_info.json'이 생성되었습니다!")
    print("파일을 수정해서 본인의 정보로 바꿔보세요!\n")


def main():
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("환경변수에 OPENAI_API_KEY를 설정해주세요.")
        return

    # 샘플 데이터가 없으면 생성
    if not os.path.exists('my_info.json'):
        create_sample_json()

    # RAG 시스템 초기화
    rag_system = PersonalInfoRAG()
    rag_system.load_json_file('my_info.json')

    # 대화 시작
    rag_system.chat()


if __name__ == "__main__":
    main()