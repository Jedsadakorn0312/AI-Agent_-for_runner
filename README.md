# Agentic RAG Running Coach

ผู้ช่วยวางแผนซ้อมวิ่งและแนะนำรองเท้าด้วยระบบ Retrieval-Augmented Generation (RAG)
รองรับคำถามภาษาไทยเกี่ยวกับการเตรียมตัววิ่ง 5K/10K การคำนวณ pace และการเลือกรองเท้า
ตามประเภทการซ้อม งบประมาณ และแบรนด์

## ภาพรวม

ระบบประกอบด้วย backend แบบ FastAPI และ frontend แบบ React/Vite โดย agent สามารถ:

- วางแผนซ้อมสำหรับเป้าหมาย 5K และ 10K
- คำนวณ pace จากระยะทางและเวลา
- ค้นหาและแนะนำรองเท้าสำหรับ easy run, tempo, interval และ race day
- รับเงื่อนไข เช่น งบประมาณ แบรนด์ และประเภทการซ้อม
- แสดงข้อมูล debug เช่น action plan, tool calls, เอกสารที่ค้นพบ และ constraints ผ่านหน้าเว็บ

ข้อมูลความรู้ถูกเก็บใน vector store สองชุด ได้แก่ เอกสารการฝึกวิ่งและข้อมูลรองเท้า

## วิธีการทำงาน

เมื่อได้รับคำถาม ระบบจะ:

1. วิเคราะห์ intent และเงื่อนไขของคำถาม
2. เลือกเครื่องมือที่เหมาะสม ได้แก่ calculator, coach search หรือ shoe search
3. ค้นข้อมูลจาก ChromaDB ด้วย multilingual embeddings
4. กรองและจัดอันดับผลลัพธ์ โดยเฉพาะเงื่อนไขแบรนด์ งบประมาณ และประเภทการซ้อม
5. สร้างคำตอบด้วย Llama 3 พร้อมสรุปแผนหรือรายการรองเท้าที่เกี่ยวข้อง

คำถามที่ไม่เกี่ยวกับการวิ่งหรือรองเท้าวิ่งจะไม่เปิดใช้เครื่องมือค้นข้อมูล

## เทคโนโลยี

- Python
- LangChain และ LangChain Community
- Llama 3 ผ่าน Ollama
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- ChromaDB
- FastAPI และ Uvicorn
- React, Vite และ `lucide-react`

## โครงสร้างข้อมูลและ Vector Store

- [`data/data_runner.pdf`](data/data_runner.pdf): เอกสารความรู้ด้านการฝึกวิ่ง
- [`data/rev_shoes_grouped_classified.json`](data/rev_shoes_grouped_classified.json): ข้อมูลรองเท้าที่จัดกลุ่มและจำแนกประเภทแล้ว
- [`src/ingestion/ingest_pdf.py`](src/ingestion/ingest_pdf.py): แบ่ง PDF เป็น chunks แล้วสร้าง `vector_store/coach_db`
- [`src/ingestion/ingest_shoes.py`](src/ingestion/ingest_shoes.py): แปลงข้อมูลรองเท้าเป็น documents พร้อม metadata แล้วสร้าง `vector_store/shoe_db`

`vector_store/` เป็นไฟล์ฐานข้อมูลที่สร้างในเครื่องและถูกละเว้นจาก Git หากต้องสร้างใหม่
ให้รันคำสั่ง ingestion จากโฟลเดอร์รากของโปรเจกต์:

```bash
python -m src.ingestion.ingest_pdf
python -m src.ingestion.ingest_shoes
```

การรัน ingestion ใหม่จะต้องดาวน์โหลดโมเดล embeddings ในครั้งแรก และควรสร้างฐานข้อมูล
ในโฟลเดอร์ที่ว่างหรือสำรองฐานข้อมูลเดิมก่อน

## การติดตั้งและการรัน

### 1. เตรียม Ollama

ติดตั้ง [Ollama](https://ollama.com/) จากนั้นดาวน์โหลดโมเดลที่ agent ใช้:

```bash
ollama pull llama3
```

ตรวจสอบว่า Ollama ทำงานอยู่ก่อนเริ่ม backend

### 2. ติดตั้ง Python dependencies

```bash
python -m venv .venv
```

เปิดใช้งาน virtual environment ตามระบบปฏิบัติการ แล้วติดตั้ง dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. สร้าง Vector Store

```bash
python -m src.ingestion.ingest_pdf
python -m src.ingestion.ingest_shoes
```

### 4. เปิด backend

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

ตรวจสอบสถานะได้ที่ `http://localhost:8000/health` และดู API documentation ได้ที่
`http://localhost:8000/docs`

### 5. เปิด frontend

เปิด terminal อีกหน้าต่าง:

```bash
cd web
npm install
npm run dev
```

จากนั้นเปิด `http://localhost:5173`

หาก backend ไม่ได้รันที่ `http://localhost:8000` ให้แก้ค่า `API_BASE_URL` ใน
[`web/src/App.jsx`](web/src/App.jsx) ให้ตรงกับ URL ของ backend

นอกจากหน้าเว็บแล้ว ยังสามารถรันโหมดสนทนาใน terminal ได้ด้วย:

```bash
python main.py
```

## ตัวอย่างการใช้งาน

ตัวอย่างคำถามที่ backend มีให้ใน endpoint `/sample-queries`:

```text
อยากทำ 10K sub-55 ตอนนี้วิ่งได้ 60 นาที มีเวลาซ้อม 3 วัน/สัปดาห์ ภายใน 8 สัปดาห์
ช่วยแนะนำรองเท้าวิ่งสำหรับ easy run งบไม่เกิน 4000 บาท ของ nike หน่อย
อยากวิ่ง 5K ให้ได้ 30 นาที ตอนนี้ 40 นาที ซ้อมได้ 4 วัน/สัปดาห์
```

ตัวอย่างผลลัพธ์จาก calculator:

```text
ระยะทาง 5 กิโลเมตร ใช้เวลา 30 นาที จะได้ pace ประมาณ 6:00 นาที/กิโลเมตร
```

หน้าเว็บจะแสดงคำตอบของ coach และส่วน debug ที่เปิดดู action plan, เครื่องมือที่ถูกเรียก
และเอกสารที่ถูกนำมาใช้ประกอบคำตอบได้ ภายใน repository รุ่นนี้ยังไม่มีไฟล์ภาพหน้าจอ
จึงไม่ได้ฝังภาพที่สร้างขึ้นใหม่หรืออ้างภาพที่ไม่มีอยู่จริง

## บทบาทของผู้ช่วยในงานนี้

สำหรับการปรับปรุง repository ครั้งนี้ ผู้ช่วยรับผิดชอบ:

- ตรวจโค้ดและ entry points ที่ใช้จริงเพื่อจัดทำเอกสารการติดตั้งและการใช้งาน
- เติม README นี้ให้ครอบคลุมภาพรวม สถาปัตยกรรม ข้อมูล และข้อจำกัด
- ตรวจสอบกฎละเว้นไฟล์ชั่วคราวและนำไฟล์ cache/vector store ที่ถูกติดตามออกจาก Git index

ผู้ช่วยไม่ได้สร้างหรือรับรองข้อมูลรองเท้าและเอกสารฝึกวิ่งต้นฉบับ และไม่ควรตีความ
คำแนะนำจากระบบแทนคำแนะนำจากโค้ชหรือผู้เชี่ยวชาญทางการแพทย์

## ผลการตรวจสอบ

ตรวจสอบจาก source code แล้วพบว่า:

- endpoint health คือ `GET /health`
- endpoint สนทนาคือ `POST /chat`
- backend ใช้ `llama3` ผ่าน `ChatOllama`
- calculator คำนวณ pace จาก `เวลา / ระยะทาง`
- ทดสอบ calculator ด้วยระยะทาง 5 กิโลเมตรและเวลา 30 นาที ได้ผล `6:00 นาที/กิโลเมตร`
- รัน `npm --prefix web run build` สำเร็จ

ไม่พบ automated test suite หรือผล benchmark ที่เก็บไว้ใน repository จึงไม่ระบุ
ตัวเลขความแม่นยำหรือผลทดสอบอื่นที่ยังไม่ได้รันและบันทึกไว้

## ข้อจำกัด

- คำตอบขึ้นกับคุณภาพและความครอบคลุมของ PDF และ JSON ที่นำเข้า
- ข้อมูลรองเท้าอาจไม่เป็นปัจจุบัน โดยเฉพาะราคา รุ่น และสินค้าที่มีจำหน่าย
- การกรองตามงบประมาณ แบรนด์ และประเภทการซ้อมทำงานกับ metadata ที่มีอยู่เท่านั้น
- หากไม่มีรองเท้าที่ตรงทุกเงื่อนไข ระบบอาจไม่สามารถเสนอผลลัพธ์ที่ตรงทั้งหมดได้
- LLM อาจตีความคำถามหรือเงื่อนไขกำกวมผิดพลาด ผู้ใช้ควรตรวจสอบรายละเอียดก่อนตัดสินใจ
- คำแนะนำการซ้อมไม่ใช่การวินิจฉัยหรือคำแนะนำทางการแพทย์

## License

ยังไม่ได้กำหนด license สำหรับ repository นี้