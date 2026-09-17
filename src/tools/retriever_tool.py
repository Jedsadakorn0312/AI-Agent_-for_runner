from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_retriever_tools():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    coach_db = Chroma(
        persist_directory="vector_store/coach_db",
        embedding_function=embeddings
    )

    shoe_db = Chroma(
        persist_directory="vector_store/shoe_db",
        embedding_function=embeddings
    )

    coach_retriever = coach_db.as_retriever(
        search_kwargs={"k": 2}
    )

    shoe_retriever = shoe_db.as_retriever(
        search_kwargs={"k": 2}
    )

    coach_tool = create_retriever_tool(
        retriever=coach_retriever,
        name="search_coach_knowledge",
        description=(
            "ใช้ค้นหาข้อมูลเกี่ยวกับการฝึกซ้อมวิ่ง แผนการวิ่ง "
            "ประเภทการซ้อม pace, easy run, tempo run, interval, race day "
            "และหลักการฝึกจากเอกสาร running coach"
        )
    )

    shoe_tool = create_retriever_tool(
        retriever=shoe_retriever,
        name="search_running_shoes",
        description=(
            "ใช้ค้นหาข้อมูลรองเท้าวิ่ง รุ่นรองเท้า ราคา brand "
            "primary_use, recommended_runs, easy_run, interval, tempo, race_day "
            "และคำแนะนำรองเท้าตามประเภทการซ้อม"
        )
    )

    return [coach_tool, shoe_tool]