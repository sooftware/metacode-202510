"""
SimpleRAG Streamlit 데모

사용법:
1. 파일을 rag_demo.py로 저장
2. streamlit run rag_demo.py 실행

필수 설치:
pip install streamlit openai python-dotenv numpy
"""

import streamlit as st
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# SimpleRAG 클래스
class SimpleRAG:
    """간단한 RAG 시스템"""

    def __init__(self, client, model="gpt-4o-mini", embedding_model="text-embedding-3-small"):
        self.client = client
        self.model = model
        self.embedding_model = embedding_model
        self.documents = []

    def add_document(self, title, content):
        """문서 추가"""
        embedding = self._get_embedding(content)
        self.documents.append({
            "title": title,
            "content": content,
            "embedding": embedding
        })

    def _get_embedding(self, text):
        """텍스트 임베딩"""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return response.data[0].embedding

    def _cosine_similarity(self, vec1, vec2):
        """코사인 유사도"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search(self, query, top_k=2):
        """문서 검색"""
        if not self.documents:
            return []

        query_embedding = self._get_embedding(query)

        similarities = [
            (doc, self._cosine_similarity(query_embedding, doc["embedding"]))
            for doc in self.documents
        ]

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def answer(self, question, top_k=2):
        """질문에 답변"""
        if not self.documents:
            return "먼저 문서를 추가해주세요.", []

        # 검색
        relevant_docs = self.search(question, top_k=top_k)

        # 컨텍스트 구성
        context = "\n\n".join([
            f"[문서: {doc['title']}]\n{doc['content']}"
            for doc, _ in relevant_docs
        ])

        # 프롬프트
        prompt = f"""
다음 문서들을 참고하여 질문에 답변해주세요.
문서에 없는 내용은 "문서에서 찾을 수 없습니다"라고 답하세요.

참고 문서:
```
{context}
```

질문: {question}
"""

        # 답변 생성
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content, relevant_docs


# Streamlit UI
st.set_page_config(
    page_title="RAG ChatBot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG-based ChatBot")
st.markdown("문서를 기반으로 답변하는 AI 챗봇입니다.")

# 세션 스테이트 초기화
if 'rag' not in st.session_state:
    st.session_state.rag = SimpleRAG(client)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True

# 사이드바: 문서 관리
with st.sidebar:
    st.header("📚 문서 관리")

    # 문서 추가 방법 선택
    doc_input_method = st.radio(
        "문서 입력 방법:",
        ["텍스트 입력", "파일 업로드"]
    )

    if doc_input_method == "텍스트 입력":
        doc_title = st.text_input("문서 제목:", key="doc_title")
        doc_content = st.text_area(
            "문서 내용:",
            height=200,
            placeholder="문서 내용을 입력하세요...",
            key="doc_content"
        )

        if st.button("📄 문서 추가", type="primary"):
            if doc_title and doc_content:
                with st.spinner("임베딩 생성 중..."):
                    st.session_state.rag.add_document(doc_title, doc_content)
                st.success(f"✅ '{doc_title}' 추가 완료!")
                st.rerun()
            else:
                st.error("제목과 내용을 모두 입력해주세요.")

    else:  # 파일 업로드
        uploaded_file = st.file_uploader(
            "문서 파일 업로드 (.txt)",
            type=['txt']
        )

        if uploaded_file is not None:
            doc_title = st.text_input(
                "문서 제목:",
                value=uploaded_file.name.replace('.txt', ''),
                key="file_doc_title"
            )

            if st.button("📄 파일에서 문서 추가", type="primary"):
                if doc_title:
                    content = uploaded_file.read().decode('utf-8')
                    with st.spinner("임베딩 생성 중..."):
                        st.session_state.rag.add_document(doc_title, content)
                    st.success(f"✅ '{doc_title}' 추가 완료!")
                    st.rerun()
                else:
                    st.error("제목을 입력해주세요.")

    st.divider()

    # 등록된 문서 목록
    st.subheader("등록된 문서")
    if st.session_state.rag.documents:
        for i, doc in enumerate(st.session_state.rag.documents, 1):
            with st.expander(f"{i}. {doc['title']}"):
                st.text(doc['content'][:200] + "...")
                if st.button(f"🗑️ 삭제", key=f"del_{i}"):
                    st.session_state.rag.documents.pop(i - 1)
                    st.rerun()
    else:
        st.info("아직 등록된 문서가 없습니다.")

    st.divider()

    # 설정
    st.subheader("⚙️ 설정")
    st.session_state.show_sources = st.checkbox(
        "참고 문서 표시",
        value=True
    )

    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# 메인 영역: 채팅 인터페이스
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 대화")

    # 대화 기록 표시
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.write(message['content'])

                # 참고 문서 표시
                if message['role'] == 'assistant' and 'sources' in message and st.session_state.show_sources:
                    if message['sources']:
                        with st.expander("📚 참고 문서"):
                            for doc, score in message['sources']:
                                st.markdown(f"**{doc['title']}** (유사도: {score:.3f})")
                                st.caption(doc['content'][:150] + "...")

with col2:
    st.subheader("📊 통계")
    st.metric("등록 문서", len(st.session_state.rag.documents))
    st.metric("대화 수", len([m for m in st.session_state.messages if m['role'] == 'user']))

# 사용자 입력
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    if not st.session_state.rag.documents:
        st.error("먼저 사이드바에서 문서를 추가해주세요!")
    else:
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # RAG 답변 생성
        with st.spinner("답변 생성 중..."):
            answer, sources = st.session_state.rag.answer(user_input, top_k=2)

        # AI 메시지 추가
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        st.rerun()

# 샘플 문서 추가 버튼
if not st.session_state.rag.documents:
    st.info("💡 팁: 왼쪽 사이드바에서 문서를 추가하거나, 아래 샘플 문서를 사용해보세요!")

    if st.button("📝 샘플 문서 추가 (회사 규정)", type="secondary"):
        sample_docs = [
            {
                "title": "휴가 정책",
                "content": """회사 휴가 정책:
1. 연차: 입사 1년차 15일, 2년차부터 매년 1일씩 추가 (최대 25일)
2. 병가: 연간 10일, 의사 소견서 필요
3. 경조사: 본인 결혼 5일, 직계가족 사망 3일
4. 휴가 신청: 최소 3일 전 상사 승인 필요
5. 미사용 연차: 다음 해 이월 불가, 금전 보상 가능"""
            },
            {
                "title": "근무 시간",
                "content": """근무 시간 규정:
1. 정규 근무: 평일 09:00 - 18:00 (주 40시간)
2. 점심시간: 12:00 - 13:00 (1시간)
3. 유연근무제: 코어타임 11:00-16:00 준수
4. 재택근무: 주 2회 가능, 사전 신청 필요
5. 초과근무: 사전 승인 필요, 시간당 1.5배 수당"""
            },
            {
                "title": "복리후생",
                "content": """복리후생 제도:
1. 건강검진: 연 1회 종합검진 지원
2. 식대: 월 20만원 식비 지원
3. 교육비: 연 300만원 한도 업무 관련 교육비 지원
4. 경조사비: 직원 경조사 시 10-50만원 지원
5. 자녀 학자금: 중고등학생 자녀 1인당 연 200만원 지원"""
            }
        ]

        with st.spinner("샘플 문서 추가 중..."):
            for doc in sample_docs:
                st.session_state.rag.add_document(doc["title"], doc["content"])

        st.success("✅ 샘플 문서 3개가 추가되었습니다!")
        st.rerun()

# Footer
st.divider()
st.caption("🤖 Powered by OpenAI GPT-4o-mini & RAG")