import streamlit as st
import json
import openai
import numpy as np
from typing import List, Dict, Any
import os

# 페이지 설정
st.set_page_config(
    page_title="개인 정보 RAG 챗봇",
    page_icon="🤖",
    layout="wide"
)

# 세션 상태 초기화
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


class StreamlitRAG:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"임베딩 생성 오류: {e}")
            return []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        if not vec1 or not vec2:
            return 0
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def load_json_data(self, json_data: dict) -> bool:
        """JSON 데이터에서 임베딩 생성"""
        try:
            self.documents.clear()
            self.embeddings.clear()
            self.metadata.clear()

            progress_bar = st.progress(0)
            status_text = st.empty()

            total_items = sum(len(items) for items in json_data.values() if isinstance(items, list))
            current_item = 0

            for category, items in json_data.items():
                if isinstance(items, list):
                    for item in items:
                        status_text.text(f"처리 중: {item.get('title', 'Unknown')}")

                        # 텍스트 결합
                        text = f"{item.get('title', '')}\n{item.get('content', '')}"

                        # 임베딩 생성
                        embedding = self.get_embedding(text)
                        if embedding:
                            self.documents.append(text)
                            self.embeddings.append(embedding)
                            self.metadata.append({
                                'category': category,
                                'title': item.get('title', 'Unknown'),
                                'id': item.get('id', f"{category}_{len(self.documents)}")
                            })

                        current_item += 1
                        progress_bar.progress(current_item / total_items)

            progress_bar.empty()
            status_text.empty()
            return True

        except Exception as e:
            st.error(f"데이터 로딩 오류: {e}")
            return False

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """유사한 문서 검색"""
        if not self.embeddings:
            return []

        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        results = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            results.append({
                'document': self.documents[i],
                'similarity': similarity,
                'metadata': self.metadata[i]
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def generate_answer(self, question: str, search_results: List[Dict]) -> str:
        """RAG 기반 답변 생성"""
        if not search_results:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."

        context = "\n\n".join([result['document'] for result in search_results])

        prompt = f"""다음 정보를 바탕으로 질문에 친근하고 자연스럽게 답변해주세요:

정보:
{context}

질문: {question}

답변:"""

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {e}"


def main():
    st.title("🤖 개인 정보 RAG 챗봇")
    st.markdown("JSON 파일을 업로드하여 개인 맞춤형 AI 챗봇을 만들어보세요!")
    st.markdown("---")

    # 사이드바 - 설정
    with st.sidebar:
        st.header("⚙️ 설정")

        # API 키 입력
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            openai.api_key = api_key

        st.markdown("---")

        # JSON 파일 로드
        st.header("📁 JSON 파일 로드")

        json_data = None
        try:
            with open('info.json', 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            st.success("✅ info.json 파일 로드 성공")

            # 데이터 미리보기
            st.subheader("데이터 미리보기")
            total_items = sum(len(items) for items in json_data.values() if isinstance(items, list))
            st.info(f"총 {len(json_data)}개 카테고리, {total_items}개 항목")

            for category, items in json_data.items():
                if isinstance(items, list):
                    st.write(f"**{category}**: {len(items)}개 항목")

        except FileNotFoundError:
            st.error("❌ info.json 파일을 찾을 수 없습니다. 동일한 폴더에 info.json 파일을 생성해주세요.")
            json_data = None
        except Exception as e:
            st.error(f"❌ 파일 읽기 오류: {e}")
            json_data = None

        # 데이터 로드 버튼
        if st.button("🚀 데이터 로드", type="primary"):
            if not api_key:
                st.error("API 키를 입력해주세요")
            elif json_data:
                with st.spinner("데이터 처리 중..."):
                    if st.session_state.rag_system is None:
                        st.session_state.rag_system = StreamlitRAG()

                    success = st.session_state.rag_system.load_json_data(json_data)
                    if success:
                        st.session_state.data_loaded = True
                        st.success(f"✅ {len(st.session_state.rag_system.documents)}개 문서 로드 완료!")
                        st.rerun()
            else:
                st.error("JSON 파일(info.json)을 생성해주세요")

        # 상태 표시
        if st.session_state.data_loaded:
            st.success("🟢 시스템 준비 완료")
            if st.session_state.rag_system:
                st.info(f"📊 로드된 문서: {len(st.session_state.rag_system.documents)}개")
        else:
            st.warning("🟡 info.json 파일을 생성하고 데이터를 로드해주세요")

        # 채팅 초기화
        if st.button("🗑️ 대화 초기화"):
            st.session_state.chat_history = []
            st.rerun()

    # 메인 영역
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 대화")

        if st.session_state.data_loaded and api_key:
            # 사용자 입력
            user_question = st.text_input(
                "질문을 입력하세요:",
                placeholder="예: 내 취미가 뭐야?, 직장은 어디야?, 목표가 뭐야?",
                key="user_input"
            )

            # 질문 처리
            if user_question:
                with st.spinner("답변 생성 중..."):
                    # 검색
                    search_results = st.session_state.rag_system.search(user_question)

                    # 답변 생성
                    answer = st.session_state.rag_system.generate_answer(user_question, search_results)

                    # 히스토리에 추가
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': answer,
                        'search_results': search_results
                    })

                st.rerun()

            # 대화 히스토리 표시
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"💭 {chat['question']}", expanded=(i == 0)):
                    st.markdown(f"**🤖 답변:**")
                    st.markdown(chat['answer'])

                    if chat['search_results']:
                        st.markdown("**🔍 참조 정보:**")
                        for j, result in enumerate(chat['search_results'], 1):
                            st.markdown(f"""
                            **{j}. {result['metadata']['title']}** 
                            (카테고리: {result['metadata']['category']}, 유사도: {result['similarity']:.2f})

                            {result['document'][:200]}...
                            """)

        else:
            st.info("왼쪽 사이드바에서 API 키를 입력하고 info.json 파일을 생성해주세요.")

    with col2:
        st.header("📊 정보")

        if st.session_state.data_loaded and st.session_state.rag_system:
            # 데이터 통계
            categories = {}
            for meta in st.session_state.rag_system.metadata:
                cat = meta['category']
                categories[cat] = categories.get(cat, 0) + 1

            st.subheader("📈 데이터 통계")
            for category, count in categories.items():
                st.metric(category, f"{count}개")

            # 최근 대화 분석
            if st.session_state.chat_history:
                st.subheader("🔍 최근 검색")
                recent_chat = st.session_state.chat_history[-1]
                if recent_chat['search_results']:
                    for result in recent_chat['search_results']:
                        st.progress(
                            result['similarity'],
                            text=f"{result['metadata']['title']} ({result['similarity']:.2f})"
                        )

        # JSON 구조 가이드
        st.subheader("💡 JSON 구조 가이드")
        st.markdown("""
        ```json
        {
          "카테고리1": [
            {
              "id": "unique_id",
              "title": "제목",
              "content": "내용"
            }
          ]
        }
        ```
        """)

        # 사용법 안내
        st.subheader("📖 사용법")
        st.markdown("""
        1. **API 키 입력**: OpenAI API 키를 입력하세요
        2. **info.json 파일 생성**: 예시 구조를 참고하여 info.json 파일을 생성하세요
        3. **데이터 로드**: '데이터 로드' 버튼을 클릭하세요
        4. **질문하기**: 자연스럽게 질문해보세요

        **질문 예시:**
        - 내 이름이 뭐야?
        - 어떤 회사에서 일해?
        - 취미가 뭐야?
        - 목표가 뭐야?
        """)


if __name__ == "__main__":
    main()