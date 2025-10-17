import streamlit as st
import json
import os
from typing import TypedDict, Literal, Generator
from dotenv import load_dotenv

# LangSmith 추적 활성화
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Self-Correcting-RAG"

# LangChain 관련
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler

# LangGraph 관련
from langgraph.graph import StateGraph, END

# 페이지 설정
st.set_page_config(
    page_title="🧠 Self-Correcting RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .step-container {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Streamlit Callback Handler for streaming
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# State 정의
class SelfCorrectingRAGState(TypedDict):
    original_question: str
    current_question: str
    question_quality: str
    question_rewrite_count: int
    search_query: str
    retrieved_docs: list
    retrieval_quality: str
    search_retry_count: int
    answer: str
    answer_quality: str
    problem_diagnosis: str
    max_retries: int
    total_iterations: int  # 전체 반복 횟수 추적
    answer_generation_count: int  # 답변 생성 횟수 추적


def init_session_state():
    """세션 상태 초기화"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'steps' not in st.session_state:
        st.session_state.steps = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'app' not in st.session_state:
        st.session_state.app = None


def setup_vectorstore(api_key: str):
    """벡터 스토어 설정"""
    if st.session_state.vectorstore is not None:
        return st.session_state.vectorstore

    # 샘플 데이터 로드
    sample_data = [
        {
            "id": 1,
            "topic": "근무시간",
            "content": "테크노바의 근무시간은 평일 오전 9시부터 오후 6시까지입니다. 점심시간은 12시부터 1시까지 1시간입니다."
        },
        {
            "id": 2,
            "topic": "휴가정책",
            "content": "연차는 입사 후 1년차에 15일이 제공되며, 매년 1일씩 추가되어 최대 25일까지 제공됩니다."
        },
        {
            "id": 3,
            "topic": "복리후생",
            "content": "테크노바는 4대보험, 퇴직연금, 건강검진, 경조사 지원, 자기계발비 지원 등 다양한 복리후생을 제공합니다."
        },
        {
            "id": 4,
            "topic": "재택근무",
            "content": "주 2회 재택근무가 가능하며, 사전에 팀장 승인이 필요합니다."
        },
        {
            "id": 5,
            "topic": "급여",
            "content": "급여는 매월 25일에 지급되며, 성과급은 연 2회(여름, 겨울) 지급됩니다."
        }
    ]

    documents = []
    for item in sample_data:
        doc = Document(
            page_content=f"주제: {item['topic']}\n내용: {item['content']}",
            metadata={"id": item["id"], "topic": item["topic"]}
        )
        documents.append(doc)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="company_info"
    )

    st.session_state.vectorstore = vectorstore
    return vectorstore


def create_rag_graph(api_key: str, retriever, max_retries: int):
    """Self-Correcting RAG 그래프 생성"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

    # 노드 함수들
    def evaluate_question(state: SelfCorrectingRAGState) -> SelfCorrectingRAGState:
        question = state["current_question"]

        with st.status("🔍 질문 평가 중...", expanded=True) as status:
            evaluation_prompt = f"""
다음 질문이 명확하고 구체적인지 평가해주세요.

질문: "{question}"

평가 기준:
1. 질문이 무엇을 묻는지 명확한가?
2. 문맥 없이도 이해 가능한가?
3. 회사 정보에 대한 질문인가?

"good" 또는 "bad" 중 하나로만 답변하세요.
""".strip()

            response = llm.invoke(evaluation_prompt)
            quality = "good" if "good" in response.content.strip().lower() else "bad"

            if quality == "good":
                st.success(f"✅ 질문이 명확합니다: {question}")
                status.update(label="✅ 질문 평가 완료", state="complete")
            else:
                st.warning(f"⚠️ 질문이 불명확합니다: {question}")
                status.update(label="⚠️ 질문 재작성 필요", state="running")

        return {**state, "question_quality": quality}

    def rewrite_question(state: SelfCorrectingRAGState) -> SelfCorrectingRAGState:
        question = state["current_question"]

        with st.status("✏️ 질문 재작성 중...", expanded=True) as status:
            rewrite_prompt = f"""다음 질문을 더 명확하고 구체적으로 재작성해주세요.

원본 질문: "{question}"

재작성 시 고려사항:
1. 회사 정보를 묻는 것이 명확하게 드러나도록
2. 애매한 표현 제거
3. 구체적인 정보 요청으로 변환

재작성된 질문만 답변하세요.
"""
            response = llm.invoke(rewrite_prompt)
            rewritten = response.content.strip()

            st.info(f"📝 원본: {question}")
            st.success(f"✨ 재작성: {rewritten}")
            status.update(label="✅ 질문 재작성 완료", state="complete")

        return {
            **state,
            "current_question": rewritten,
            "question_rewrite_count": state.get("question_rewrite_count", 0) + 1
        }

    def retrieve_documents(state: SelfCorrectingRAGState) -> SelfCorrectingRAGState:
        query = state.get("search_query") or state["current_question"]

        with st.status("📚 문서 검색 중...", expanded=True) as status:
            docs = retriever.invoke(query)

            st.write(f"**검색 쿼리:** {query}")
            st.write(f"**검색된 문서 수:** {len(docs)}")

            for i, doc in enumerate(docs, 1):
                with st.expander(f"📄 문서 {i}: {doc.metadata.get('topic', 'N/A')}"):
                    st.write(doc.page_content)

            status.update(label="✅ 문서 검색 완료", state="complete")

        return {**state, "retrieved_docs": docs, "search_query": query}

    def evaluate_retrieval(state: SelfCorrectingRAGState) -> SelfCorrectingRAGState:
        question = state["current_question"]
        docs = state["retrieved_docs"]

        with st.status("✅ 검색 결과 평가 중...", expanded=True) as status:
            docs_content = "\n\n".join([f"문서 {i + 1}: {doc.page_content[:100]}..."
                                        for i, doc in enumerate(docs)])

            evaluation_prompt = f"""다음 질문에 대해 검색된 문서들이 관련성이 있는지 평가해주세요.

질문: "{question}"

검색된 문서들:
{docs_content}

평가 기준:
1. 문서에 질문에 답할 수 있는 정보가 포함되어 있는가?
2. 문서와 질문의 주제가 일치하는가?

"relevant" 또는 "irrelevant" 중 하나로만 답변하세요.
"""

            response = llm.invoke(evaluation_prompt)
            quality = "relevant" if "relevant" in response.content.strip().lower() else "irrelevant"

            if quality == "relevant":
                st.success("✅ 검색된 문서가 관련이 있습니다!")
                status.update(label="✅ 검색 평가 완료", state="complete")
            else:
                st.warning("⚠️ 검색된 문서가 관련이 없습니다. 재검색이 필요합니다.")
                status.update(label="⚠️ 재검색 필요", state="running")

        return {**state, "retrieval_quality": quality}

    def rewrite_search_query(state: SelfCorrectingRAGState) -> SelfCorrectingRAGState:
        question = state["current_question"]
        previous_query = state.get("search_query", question)

        with st.status("🔄 검색 쿼리 재작성 중...", expanded=True) as status:
            rewrite_prompt = f"""검색 결과가 부적절했습니다. 더 나은 검색 결과를 위해 쿼리를 재작성해주세요.

원본 질문: "{question}"
이전 검색 쿼리: "{previous_query}"

재작성 시 고려사항:
1. 핵심 키워드 강조
2. 동의어나 관련 용어 포함
3. 더 구체적인 표현 사용

재작성된 검색 쿼리만 답변하세요.
"""

            response = llm.invoke(rewrite_prompt)
            new_query = response.content.strip()

            st.info(f"📝 이전: {previous_query}")
            st.success(f"✨ 재작성: {new_query}")
            status.update(label="✅ 쿼리 재작성 완료", state="complete")

        return {
            **state,
            "search_query": new_query,
            "search_retry_count": state.get("search_retry_count", 0) + 1
        }

    def generate_answer(state: SelfCorrectingRAGState) -> SelfCorrectingRAGState:
        question = state["current_question"]
        docs = state["retrieved_docs"]
        context = "\n\n".join([doc.page_content for doc in docs])

        with st.status("💬 답변 생성 중...", expanded=True) as status:
            answer_container = st.empty()

            answer_prompt = f"""당신은 회사 정보를 안내하는 전문 AI 어시스턴트입니다.
아래 컨텍스트를 바탕으로 질문에 정확하고 친절하게 답변해주세요.

컨텍스트:
{context}

질문: {question}

답변:"""

            # 스트리밍 답변 생성
            streaming_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                streaming=True,
                openai_api_key=api_key
            )

            answer = ""
            for chunk in streaming_llm.stream(answer_prompt):
                if chunk.content:
                    answer += chunk.content
                    answer_container.markdown(answer)

            status.update(label="✅ 답변 생성 완료", state="complete")

        return {
            **state,
            "answer": answer,
            "answer_generation_count": state.get("answer_generation_count", 0) + 1
        }

    def evaluate_answer(state: SelfCorrectingRAGState) -> SelfCorrectingRAGState:
        question = state["current_question"]
        answer = state["answer"]
        total_iterations = state.get("total_iterations", 0)

        # 최대 반복 횟수 체크 (전체 시스템 안전장치)
        if total_iterations >= 20:
            st.warning("⚠️ 최대 처리 횟수 도달. 현재 답변으로 종료합니다.")
            return {**state, "answer_quality": "good", "total_iterations": total_iterations + 1}

        with st.status("⭐ 답변 평가 중...", expanded=True) as status:
            evaluation_prompt = f"""다음 답변의 품질을 평가해주세요.

질문: "{question}"
답변: "{answer}"

평가 기준:
1. 질문에 직접적으로 답변하는가?
2. 컨텍스트의 정보를 정확히 사용했는가?
3. 답변이 구체적이고 유용한가?
4. 환각(hallucination)이 없는가?

"good" 또는 "bad" 중 하나로만 답변하세요.
"""

            response = llm.invoke(evaluation_prompt)
            quality = "good" if "good" in response.content.strip().lower() else "bad"

            # 답변 생성 횟수가 2회 이상이면 강제로 good 처리 (무한루프 방지)
            answer_gen_count = state.get("answer_generation_count", 0)
            if quality == "bad" and answer_gen_count >= 2:
                st.warning(f"⚠️ 답변 재생성 {answer_gen_count}회 시도. 현재 답변으로 종료합니다.")
                quality = "good"

            if quality == "good":
                st.success("✅ 답변 품질이 좋습니다!")
                status.update(label="✅ 답변 평가 완료", state="complete")
            else:
                st.warning("⚠️ 답변 품질이 좋지 않습니다. 개선이 필요합니다.")
                status.update(label="⚠️ 답변 개선 필요", state="running")

        return {**state, "answer_quality": quality, "total_iterations": total_iterations + 1}

    def diagnose_problem(state: SelfCorrectingRAGState) -> SelfCorrectingRAGState:
        question = state["current_question"]
        docs = state["retrieved_docs"]
        answer = state["answer"]

        with st.status("🔧 문제 진단 중...", expanded=True) as status:
            docs_summary = "\n".join([f"- {doc.metadata.get('topic', 'N/A')}" for doc in docs])

            diagnosis_prompt = f"""답변에 문제가 있습니다. 어디에서 문제가 발생했는지 진단해주세요.

질문: "{question}"
검색된 문서들:
{docs_summary}
생성된 답변: "{answer}"

다음 중 하나를 선택하세요:
1. "question_issue" - 질문 자체에 문제가 있음 (애매하거나 불명확)
2. "retrieval_issue" - 검색된 문서가 부적절함
3. "generation_issue" - 답변 생성 과정에서 문제 발생

진단 결과만 답변하세요 (question_issue, retrieval_issue, generation_issue 중 하나).
"""

            response = llm.invoke(diagnosis_prompt)
            diagnosis = response.content.strip().lower()

            if "question" in diagnosis:
                diagnosis = "question_issue"
                st.warning("⚠️ 질문에 문제가 있습니다. 질문을 재작성합니다.")
            elif "retrieval" in diagnosis:
                diagnosis = "retrieval_issue"
                st.warning("⚠️ 검색 결과에 문제가 있습니다. 검색을 재시도합니다.")
            else:
                diagnosis = "generation_issue"
                st.warning("⚠️ 답변 생성에 문제가 있습니다. 답변을 재생성합니다.")

            status.update(label="✅ 문제 진단 완료", state="complete")

        return {**state, "problem_diagnosis": diagnosis}

    # 라우팅 함수들
    def route_after_question_eval(state: SelfCorrectingRAGState) -> str:
        if state["question_quality"] == "good":
            return "retrieve"
        elif state.get("question_rewrite_count", 0) >= max_retries:
            return "retrieve"
        else:
            return "rewrite_question"

    def route_after_retrieval_eval(state: SelfCorrectingRAGState) -> str:
        if state["retrieval_quality"] == "relevant":
            return "generate"
        elif state.get("search_retry_count", 0) >= max_retries:
            return "generate"
        else:
            return "rewrite_query"

    def route_after_answer_eval(state: SelfCorrectingRAGState) -> str:
        answer_quality = state["answer_quality"]
        answer_gen_count = state.get("answer_generation_count", 0)

        if answer_quality == "good":
            return "end"
        else:
            # 답변이 bad지만 이미 여러 번 생성했다면 종료
            if answer_gen_count >= 2:
                st.warning("⚠️ 답변 개선 시도 완료. 현재 답변으로 종료합니다.")
                return "end"
            return "diagnose"

    def route_after_diagnosis(state: SelfCorrectingRAGState) -> str:
        diagnosis = state["problem_diagnosis"]
        max_retries_value = state.get("max_retries", 2)
        answer_gen_count = state.get("answer_generation_count", 0)

        # 전체 재시도 횟수 체크
        total_iterations = state.get("total_iterations", 0)
        if total_iterations >= 20:
            st.error("⚠️ 시스템 최대 반복 횟수 도달. 강제 종료합니다.")
            return "end"

        # 재시도 횟수 체크 - 최대 횟수 초과 시 강제 종료
        if diagnosis == "question_issue":
            if state.get("question_rewrite_count", 0) >= max_retries_value:
                st.warning(f"⚠️ 질문 재작성 최대 횟수({max_retries_value}회) 초과. 현재 질문으로 진행합니다.")
                return "retrieve"
            return "rewrite_question"
        elif diagnosis == "retrieval_issue":
            if state.get("search_retry_count", 0) >= max_retries_value:
                st.warning(f"⚠️ 검색 재시도 최대 횟수({max_retries_value}회) 초과. 현재 결과로 진행합니다.")
                return "generate"
            return "rewrite_query"
        else:  # generation_issue
            # 답변 재생성은 2회까지만 허용
            if answer_gen_count >= 2:
                st.warning(f"⚠️ 답변 재생성 최대 횟수(2회) 초과. 현재 답변으로 종료합니다.")
                return "end"
            return "generate"

    # 그래프 구성
    workflow = StateGraph(SelfCorrectingRAGState)

    workflow.add_node("evaluate_question", evaluate_question)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("evaluate_retrieval", evaluate_retrieval)
    workflow.add_node("rewrite_query", rewrite_search_query)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("evaluate_answer", evaluate_answer)
    workflow.add_node("diagnose", diagnose_problem)

    workflow.set_entry_point("evaluate_question")

    workflow.add_conditional_edges(
        "evaluate_question",
        route_after_question_eval,
        {"retrieve": "retrieve", "rewrite_question": "rewrite_question"}
    )

    workflow.add_edge("rewrite_question", "evaluate_question")
    workflow.add_edge("retrieve", "evaluate_retrieval")

    workflow.add_conditional_edges(
        "evaluate_retrieval",
        route_after_retrieval_eval,
        {"generate": "generate", "rewrite_query": "rewrite_query"}
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", "evaluate_answer")

    workflow.add_conditional_edges(
        "evaluate_answer",
        route_after_answer_eval,
        {"end": END, "diagnose": "diagnose"}
    )

    workflow.add_conditional_edges(
        "diagnose",
        route_after_diagnosis,
        {
            "rewrite_question": "rewrite_question",
            "rewrite_query": "rewrite_query",
            "generate": "generate",
            "retrieve": "retrieve",
            "end": END
        }
    )

    return workflow.compile()


def main():
    init_session_state()

    # 헤더
    st.markdown('<h1 class="main-header">🧠 Self-Correcting RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### 스스로 품질을 검증하고 개선하는 지능형 RAG 시스템")

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")

        api_key = st.text_input("OpenAI API Key", type="password", help="OpenAI API 키를 입력하세요")

        with st.expander("🔍 LangSmith 추적 (선택사항)"):
            langsmith_api_key = st.text_input(
                "LangSmith API Key",
                type="password",
                help="LangSmith 추적을 위한 API 키 (https://smith.langchain.com)"
            )
            if langsmith_api_key:
                os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
                st.success("✅ LangSmith 추적이 활성화되었습니다!")
                st.info("프로젝트 이름: Self-Correcting-RAG")
            else:
                os.environ["LANGCHAIN_TRACING_V2"] = "false"

        max_retries = st.slider("최대 재시도 횟수", 1, 5, 2)

        st.divider()

        st.header("📊 시스템 정보")
        st.info("""
        **Self-Correcting RAG**는 각 단계마다 품질을 자동으로 검증하고, 
        문제가 발견되면 스스로 개선하는 고급 RAG 시스템입니다.

        **주요 기능:**
        - ✅ 질문 명확성 자동 평가
        - 🔄 부적절한 질문 자동 재작성
        - 🎯 검색 결과 관련성 평가
        - 🔍 검색 쿼리 최적화
        - 💬 답변 품질 자동 검증
        - 🔧 문제 진단 및 자동 수정
        """)

        if st.button("🗑️ 대화 내역 초기화"):
            st.session_state.messages = []
            st.session_state.steps = []
            st.rerun()

    # 메인 영역
    if not api_key:
        st.warning("⚠️ 사이드바에서 OpenAI API Key를 입력해주세요.")
        st.stop()

    # 벡터 스토어 설정
    try:
        with st.spinner("📚 벡터 스토어 초기화 중..."):
            vectorstore = setup_vectorstore(api_key)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        st.success("✅ 시스템이 준비되었습니다!")
    except Exception as e:
        st.error(f"❌ 초기화 실패: {str(e)}")
        st.stop()

    # 질문 입력
    st.divider()

    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input(
            "💭 질문을 입력하세요",
            placeholder="예: 테크노바의 근무시간은 어떻게 되나요?",
            key="question_input"
        )
    with col2:
        st.write("")  # 정렬을 위한 공백
        st.write("")
        submit_button = st.button("🚀 질문하기", use_container_width=True)

    # 예제 질문
    st.caption("**예제 질문:**")
    example_cols = st.columns(3)
    with example_cols[0]:
        if st.button("💼 근무시간", use_container_width=True):
            st.session_state.question_input = "테크노바의 근무시간은 어떻게 되나요?"
            st.rerun()
    with example_cols[1]:
        if st.button("🏖️ 휴가정책", use_container_width=True):
            st.session_state.question_input = "연차는 몇 일인가요?"
            st.rerun()
    with example_cols[2]:
        if st.button("🏠 재택근무", use_container_width=True):
            st.session_state.question_input = "재택근무가 가능한가요?"
            st.rerun()

    # 질문 처리
    if submit_button and user_question:
        st.divider()
        st.markdown("### 🔄 처리 과정")

        try:
            # RAG 그래프 생성
            app = create_rag_graph(api_key, retriever, max_retries)

            # 초기 상태
            initial_state = {
                "original_question": user_question,
                "current_question": user_question,
                "question_quality": "",
                "question_rewrite_count": 0,
                "search_query": "",
                "retrieved_docs": [],
                "retrieval_quality": "",
                "search_retry_count": 0,
                "answer": "",
                "answer_quality": "",
                "problem_diagnosis": "",
                "max_retries": max_retries,
                "total_iterations": 0,
                "answer_generation_count": 0
            }

            # 실행 (recursion_limit 설정)
            result = app.invoke(
                initial_state,
                config={"recursion_limit": 50}  # 재귀 제한 증가
            )

            # 최종 결과 표시
            st.divider()
            st.markdown("### ✨ 최종 답변")

            st.markdown(f"""
            <div class="success-box">
                <h4>💬 {result['answer']}</h4>
            </div>
            """, unsafe_allow_html=True)

            # 통계 정보
            st.divider()
            st.markdown("### 📊 처리 통계")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("질문 재작성", f"{result['question_rewrite_count']}회")
            with col2:
                st.metric("검색 재시도", f"{result['search_retry_count']}회")
            with col3:
                st.metric("답변 생성", f"{result.get('answer_generation_count', 1)}회")
            with col4:
                st.metric("최종 품질", "✅ Good" if result['answer_quality'] == 'good' else "⚠️ Needs Work")
            with col5:
                st.metric("검색 문서", f"{len(result['retrieved_docs'])}개")

            # 상세 정보
            with st.expander("🔍 상세 정보 보기"):
                st.json({
                    "원본 질문": result['original_question'],
                    "최종 질문": result['current_question'],
                    "검색 쿼리": result['search_query'],
                    "질문 품질": result['question_quality'],
                    "검색 품질": result['retrieval_quality'],
                    "답변 품질": result['answer_quality'],
                    "총 반복 횟수": result.get('total_iterations', 0),
                    "답변 생성 횟수": result.get('answer_generation_count', 1)
                })

        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()