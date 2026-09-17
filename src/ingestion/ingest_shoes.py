import json

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def ingest_shoe_data(
    json_path="data/rev_shoes_grouped_classified.json",
    persist_directory="vector_store/shoe_db"
):
    print(f"Loading JSON from {json_path}...")

    with open(json_path, "r", encoding="utf-8") as f:
        shoes_data = json.load(f)

    documents = []
    shoe_list = []

    # รองรับ JSON แบบ list หรือ grouped by brand/category
    if isinstance(shoes_data, list):
        shoe_list = shoes_data

    elif isinstance(shoes_data, dict):
        for key, value in shoes_data.items():

            # ข้าม metadata หรือ summary ที่ไม่ใช่รองเท้าจริง
            if str(key).startswith("_"):
                continue

            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item["category_group"] = key
                        shoe_list.append(item)

            elif isinstance(value, dict):
                value["category_group"] = key
                shoe_list.append(value)

    for shoe in shoe_list:
        if not isinstance(shoe, dict):
            continue

        brand = shoe.get("brand", shoe.get("Brand", "Unknown"))
        model = shoe.get("model", shoe.get("Model", "Unknown"))
        shoe_type = shoe.get("type", "Unknown")

        price_thb = shoe.get("price_thb", shoe.get("Price_thb", None))
        primary_use = shoe.get("primary_use", "Unknown")
        recommended_runs = shoe.get("recommended_runs", [])
        run_suitability_score = shoe.get("run_suitability_score", {})
        not_ideal_for = shoe.get("not_ideal_for", [])
        shoe_tags = shoe.get("shoe_tags", [])
        handle = shoe.get("handle", "")
        category_group = shoe.get("category_group", "")

        try:
            price_float = float(price_thb) if price_thb is not None else -1.0
        except Exception:
            price_float = -1.0

        easy_run_score = run_suitability_score.get("easy_run", 0)
        interval_score = run_suitability_score.get("interval", 0)
        tempo_score = run_suitability_score.get("tempo", 0)
        race_day_score = run_suitability_score.get("race_day", 0)

        # ทำ content ให้ค้น semantic ได้ดีขึ้น
        content = f"""
Brand: {brand}
Model: {model}
Type: {shoe_type}
Price THB: {price_float}
Handle: {handle}

Primary Use: {primary_use}
Recommended Runs: {recommended_runs}
Run Suitability Score: {run_suitability_score}
Easy Run Score: {easy_run_score}
Interval Score: {interval_score}
Tempo Score: {tempo_score}
Race Day Score: {race_day_score}

Shoe Tags: {shoe_tags}
Not Ideal For: {not_ideal_for}

Classification Reason TH: {shoe.get("classification_reason_th", "")}
Caution TH: {shoe.get("caution_th", "")}
Category Group: {category_group}

Search Keywords:
brand {brand}
model {model}
price {price_float}
primary_use {primary_use}
recommended_runs {' '.join(map(str, recommended_runs))}
easy_run {'easy_run' in recommended_runs or primary_use == 'easy_run'}
tempo {'tempo' in recommended_runs or primary_use == 'tempo'}
interval {'interval' in recommended_runs or primary_use == 'interval'}
race_day {'race_day' in recommended_runs or primary_use == 'race_day'}
"""

        # metadata สำคัญสำหรับ filter/rerank ภายหลัง
        metadata = {
            "brand": str(brand),
            "model": str(model),
            "type": str(shoe_type),
            "price_thb": price_float,
            "primary_use": str(primary_use),
            "recommended_runs": ",".join(map(str, recommended_runs)),
            "easy_run_score": int(easy_run_score),
            "interval_score": int(interval_score),
            "tempo_score": int(tempo_score),
            "race_day_score": int(race_day_score),
            "handle": str(handle),
            "category_group": str(category_group),
        }

        documents.append(
            Document(
                page_content=content.strip(),
                metadata=metadata
            )
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print(f"Prepared {len(documents)} shoe documents.")
    print("Creating and persisting Vector Store for Shoe Data...")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print("Shoe data ingestion complete!")
    return vectorstore


if __name__ == "__main__":
    ingest_shoe_data()