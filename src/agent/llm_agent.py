import json
import re

from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.tools.calculator_tool import calculate_pace
from src.utils.logger import logger


class RunningCoachAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3",
            temperature=0.0
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.coach_db = Chroma(
            persist_directory="vector_store/coach_db",
            embedding_function=self.embeddings
        )

        self.shoe_db = Chroma(
            persist_directory="vector_store/shoe_db",
            embedding_function=self.embeddings
        )

        self.fetch_k = 6
        self.final_k = 2

        self.shoe_fetch_k = 25
        self.shoe_final_k = 6

        self._last_coach_debug_docs = []
        self._last_shoe_debug_docs = []

    # ---------------------------------------------------------------------
    # 1) LLM ACTION PLANNER
    # ---------------------------------------------------------------------
    def _create_action_plan_with_llm(self, user_input: str, chat_history: str) -> dict:
        planner_prompt = f"""
คุณคือ planning agent สำหรับระบบ Running Coach RAG

วิเคราะห์คำถามผู้ใช้ แล้วตอบเป็น JSON เท่านั้น

ขอบเขตระบบ:
1. แผนซ้อมวิ่ง 5K/10K
2. การคำนวณ pace
3. การแนะนำรองเท้าวิ่ง
4. คำถามเกี่ยวกับ easy run, tempo, interval, long run, race pace

Tools:
1. calculate_pace: ใช้เมื่อมีระยะทาง + เวลา หรือถาม pace
2. search_coach_knowledge: ใช้เมื่อถามแผนซ้อม วิธีซ้อม การพัฒนาการวิ่ง 5K หรือ 10K
3. search_running_shoes: ใช้เมื่อถามรองเท้าวิ่งโดยตรง เช่น รองเท้า / shoe / รุ่น / ยี่ห้อ

กฎ:
- ตอบ JSON เท่านั้น
- ถ้าคำถามไม่เกี่ยวกับการวิ่งหรือรองเท้าวิ่ง ให้ปิด tool ทั้งหมด
- ถ้าถามรองเท้าอย่างเดียว ให้ใช้เฉพาะ search_running_shoes
- ถ้าถามรองเท้าพร้อมแผนซ้อม ให้ใช้ทั้ง coach search และ shoe search
- ถ้าถาม 5K หรือ 10K ให้ใช้ coach_query เจาะ Chapter 13
- ถ้าไม่ได้ถามรองเท้า ห้ามเปิด use_shoe_search

ประวัติสนทนา:
{chat_history}

คำถามผู้ใช้:
{user_input}

ตอบ JSON รูปแบบนี้:
{{
  "reasoning_summary": "เหตุผลสั้น ๆ ภาษาไทย",
  "use_pace_tool": true,
  "use_coach_search": true,
  "use_shoe_search": false,
  "coach_query": "query สำหรับ coach",
  "shoe_query": "query สำหรับรองเท้า"
}}
"""

        logger.info("Agent reasoning: ให้ LLM วิเคราะห์ intent และสร้าง action plan")

        try:
            response = self.llm.invoke(planner_prompt)
            text = response.content if hasattr(response, "content") else str(response)

            plan = self._parse_json_plan(text)

            if not plan:
                logger.info("LLM action plan parse failed. Using fallback rule-based plan.")
                return self._fallback_action_plan(user_input)

            plan = self._normalize_action_plan(plan, user_input)

            logger.info(f"LLM reasoning summary: {plan.get('reasoning_summary', '')}")
            logger.info(
                "LLM action plan: "
                f"use_pace_tool={plan.get('use_pace_tool')}, "
                f"use_coach_search={plan.get('use_coach_search')}, "
                f"use_shoe_search={plan.get('use_shoe_search')}"
            )

            return plan

        except Exception as e:
            logger.info(f"LLM planner error: {str(e)}. Using fallback rule-based plan.")
            return self._fallback_action_plan(user_input)

    def _parse_json_plan(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            pass

        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        return None

    def _normalize_action_plan(self, plan: dict, user_input: str) -> dict:
        fallback = self._fallback_action_plan(user_input)

        normalized = {
            "reasoning_summary": str(plan.get("reasoning_summary", fallback["reasoning_summary"])),
            "use_pace_tool": bool(plan.get("use_pace_tool", fallback["use_pace_tool"])),
            "use_coach_search": bool(plan.get("use_coach_search", fallback["use_coach_search"])),
            "use_shoe_search": bool(plan.get("use_shoe_search", fallback["use_shoe_search"])),
            "coach_query": str(plan.get("coach_query") or fallback["coach_query"]),
            "shoe_query": str(plan.get("shoe_query") or fallback["shoe_query"])
        }

        is_direct_shoe = self._is_direct_shoe_question(user_input)
        is_running = self._is_running_question(user_input)
        has_running_goal = self._has_distance_and_time(user_input)
        has_training_intent = self._has_training_intent(user_input)

        # ---------------------------------------------------------------
        # Out of scope
        # ---------------------------------------------------------------
        if not is_running and not is_direct_shoe:
            normalized["reasoning_summary"] = (
                "คำถามนี้ไม่เกี่ยวกับขอบเขตของระบบโค้ชวิ่งหรือการแนะนำรองเท้าวิ่ง"
            )
            normalized["use_pace_tool"] = False
            normalized["use_coach_search"] = False
            normalized["use_shoe_search"] = False
            normalized["coach_query"] = ""
            normalized["shoe_query"] = ""
            return normalized

        # ---------------------------------------------------------------
        # Shoe only:
        # เช่น “แนะนำรองเท้าวิ่งสำหรับ easy run งบไม่เกิน 4000”
        # แม้มีคำว่า easy run ก็ยังต้องเป็น shoe_only เพราะไม่ได้ขอแผนซ้อม
        # ---------------------------------------------------------------
        if is_direct_shoe and not has_training_intent:
            normalized["reasoning_summary"] = (
                "ผู้ใช้ถามเรื่องรองเท้าวิ่งโดยตรง จึงใช้เฉพาะ search_running_shoes"
            )
            normalized["use_pace_tool"] = False
            normalized["use_coach_search"] = False
            normalized["use_shoe_search"] = True
            normalized["coach_query"] = ""
            normalized["shoe_query"] = user_input
            return normalized

        # ---------------------------------------------------------------
        # Mixed:
        # เช่น “จัดแผนซ้อม 5K และแนะนำรองเท้าสำหรับ tempo run”
        # ---------------------------------------------------------------
        if is_direct_shoe and has_training_intent:
            normalized["reasoning_summary"] = (
                "ผู้ใช้ถามทั้งเรื่องแผนซ้อมและรองเท้าวิ่ง จึงใช้ทั้ง coach search และ shoe search"
            )
            normalized["use_pace_tool"] = has_running_goal
            normalized["use_coach_search"] = True
            normalized["use_shoe_search"] = True
            normalized["coach_query"] = self._build_coach_query(user_input)
            normalized["shoe_query"] = user_input
            return normalized

        # ---------------------------------------------------------------
        # Coach only
        # ---------------------------------------------------------------
        if is_running:
            normalized["use_pace_tool"] = has_running_goal
            normalized["use_coach_search"] = True
            normalized["use_shoe_search"] = False
            normalized["coach_query"] = self._build_coach_query(user_input)
            normalized["shoe_query"] = ""
            return normalized

        return normalized

    def _fallback_action_plan(self, user_input: str) -> dict:
        use_pace_tool = self._has_distance_and_time(user_input)
        use_shoe_search = self._is_direct_shoe_question(user_input)
        use_running_search = self._is_running_question(user_input)
        has_training_intent = self._has_training_intent(user_input)

        # ---------------------------------------------------------------
        # Out of scope
        # ---------------------------------------------------------------
        if not use_running_search and not use_shoe_search:
            return {
                "reasoning_summary": "คำถามนี้อยู่นอกเหนือขอบเขตของระบบ",
                "use_pace_tool": False,
                "use_coach_search": False,
                "use_shoe_search": False,
                "coach_query": "",
                "shoe_query": ""
            }

        # ---------------------------------------------------------------
        # Shoe only
        # ---------------------------------------------------------------
        if use_shoe_search and not has_training_intent:
            return {
                "reasoning_summary": "ผู้ใช้ถามเรื่องรองเท้าวิ่งโดยตรง จึงใช้เฉพาะ shoe search",
                "use_pace_tool": False,
                "use_coach_search": False,
                "use_shoe_search": True,
                "coach_query": "",
                "shoe_query": user_input
            }

        # ---------------------------------------------------------------
        # Mixed
        # ---------------------------------------------------------------
        if use_shoe_search and has_training_intent:
            return {
                "reasoning_summary": "ผู้ใช้ถามทั้งเรื่องแผนซ้อมและรองเท้าวิ่ง",
                "use_pace_tool": use_pace_tool,
                "use_coach_search": True,
                "use_shoe_search": True,
                "coach_query": self._build_coach_query(user_input),
                "shoe_query": user_input
            }

        # ---------------------------------------------------------------
        # Coach only
        # ---------------------------------------------------------------
        return {
            "reasoning_summary": "ระบบใช้ fallback rule เพื่อเลือก tool จาก keyword ในคำถาม",
            "use_pace_tool": use_pace_tool,
            "use_coach_search": use_running_search,
            "use_shoe_search": False,
            "coach_query": self._build_coach_query(user_input) if use_running_search else "",
            "shoe_query": ""
        }

    def _is_direct_shoe_question(self, text: str) -> bool:
        lowered = text.lower()

        shoe_keywords = [
            "รองเท้า", "รองเท้าวิ่ง", "shoe", "shoes", "running shoe",
            "รุ่น", "ยี่ห้อ",
            "adidas", "nike", "asics", "hoka", "brooks",
            "saucony", "new balance", "newbalance", "on running", "xtep"
        ]

        return any(kw in lowered for kw in shoe_keywords)

    def _is_running_question(self, text: str) -> bool:
        lowered = text.lower()

        running_keywords = [
            "วิ่ง", "นักวิ่ง", "ซ้อม", "แผนซ้อม", "pace", "เพซ",
            "5k", "5 km", "5km", "5 กิโล",
            "10k", "10 km", "10km", "10 กิโล",
            "easy run", "tempo", "interval", "long run",
            "recovery run", "race pace", "sub", "ลดเวลา",
            "training", "running", "run"
        ]

        return any(keyword in lowered for keyword in running_keywords)

    def _has_training_intent(self, text: str) -> bool:
        lowered = text.lower()

        training_keywords = [
            "แผนซ้อม",
            "จัดแผน",
            "ตารางซ้อม",
            "โปรแกรมซ้อม",
            "ซ้อมได้",
            "มีเวลาซ้อม",
            "วิ่งให้ได้",
            "ทำเวลา",
            "ลดเวลา",
            "sub",
            "pace เป้าหมาย",
            "เพซเป้าหมาย",
            "ตอนนี้วิ่งได้",
            "ตอนนี้อยู่ที่",
            "ภายใน",
        ]

        has_training_keyword = any(keyword in lowered for keyword in training_keywords)

        has_race_goal = (
            self._has_distance_and_time(text)
            or bool(re.search(r"sub[-\s]?\d+", lowered))
        )

        return has_training_keyword or has_race_goal

    def _get_response_mode(self, action_plan: dict) -> str:
        is_shoe = action_plan.get("use_shoe_search", False)
        is_coach = (
            action_plan.get("use_coach_search", False)
            or action_plan.get("use_pace_tool", False)
        )

        if not is_shoe and not is_coach:
            return "out_of_scope"

        if is_shoe and not is_coach:
            return "shoe_only"

        if is_coach and not is_shoe:
            return "coach_only"

        if is_coach and is_shoe:
            return "mixed"

        return "general"

    # ---------------------------------------------------------------------
    # 2) PACE EXTRACTION
    # ---------------------------------------------------------------------
    def _has_distance_and_time(self, text: str) -> bool:
        lowered = text.lower()

        has_distance = bool(
            re.search(r"(\d+(?:\.\d+)?)\s*(?:km|k|กิโล|กิโลเมตร)", lowered)
        )

        has_time = bool(
            re.search(r"(\d+(?:\.\d+)?)\s*(?:นาที|min|minutes)", lowered)
        )

        has_sub_goal = bool(
            re.search(r"sub[-\s]?(\d+)", lowered)
        )

        return has_distance and (has_time or has_sub_goal)

    def _extract_goal_distance_and_time(self, text: str):
        lowered = text.lower()

        distance_km = None
        target_minutes = None
        current_minutes = None

        distance_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:km|k|กิโล|กิโลเมตร)",
            lowered
        )

        if distance_match:
            distance_km = float(distance_match.group(1))

        sub_match = re.search(r"sub[-\s]?(\d+(?:\.\d+)?)", lowered)
        if sub_match:
            target_minutes = float(sub_match.group(1))

        minute_numbers = re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:นาที|min|minutes)",
            lowered
        )

        if minute_numbers:
            if target_minutes is None:
                target_minutes = float(minute_numbers[0])

                if len(minute_numbers) >= 2:
                    current_minutes = float(minute_numbers[1])
            else:
                current_minutes = float(minute_numbers[0])

        current_match = re.search(
            r"(?:ตอนนี้|ปัจจุบัน|ตอนนี้อยู่ที่)\s*(\d+(?:\.\d+)?)",
            lowered
        )

        if current_match:
            current_minutes = float(current_match.group(1))

        return distance_km, target_minutes, current_minutes

    def _get_pace_info(self, user_input: str) -> str:
        distance_km, target_minutes, current_minutes = self._extract_goal_distance_and_time(user_input)

        pace_info = ""

        if distance_km and target_minutes:
            target_input = f"distance_km={distance_km}, time_minutes={target_minutes}"

            logger.info("Tool call: calculate_pace")
            logger.info(f"Tool input: {target_input}")

            target_pace = calculate_pace.invoke(target_input)

            logger.info(f"Tool result: {target_pace}")
            pace_info += f"\nPace เป้าหมาย: {target_pace}"

        if distance_km and current_minutes:
            current_input = f"distance_km={distance_km}, time_minutes={current_minutes}"

            logger.info("Tool call: calculate_pace")
            logger.info(f"Tool input: {current_input}")

            current_pace = calculate_pace.invoke(current_input)

            logger.info(f"Tool result: {current_pace}")
            pace_info += f"\nPace ปัจจุบัน: {current_pace}"

        return pace_info

    def _build_pace_zone_hint(self, user_input: str) -> str:
        lowered = user_input.lower()

        if self._is_5k_query(lowered) and "30" in lowered:
            return """
Race pace เป้าหมาย: 6:00 นาที/กิโลเมตร
Easy run: 7:30-8:30 นาที/กิโลเมตร
Long easy run: 7:45-8:45 นาที/กิโลเมตร
Tempo run: 6:30-6:50 นาที/กิโลเมตร
Interval: 5:45-6:00 นาที/กิโลเมตร เฉพาะช่วงวิ่งเร็ว
"""

        if self._is_10k_query(lowered) and ("55" in lowered or "sub-55" in lowered or "sub 55" in lowered):
            return """
Race pace เป้าหมาย: 5:30 นาที/กิโลเมตร
Easy run: 6:45-7:30 นาที/กิโลเมตร
Long easy run: 6:50-7:40 นาที/กิโลเมตร
Tempo run: 5:40-5:55 นาที/กิโลเมตร
Interval: 5:05-5:25 นาที/กิโลเมตร เฉพาะช่วงวิ่งเร็ว
"""

        return ""

    # ---------------------------------------------------------------------
    # 3) TRAINING AVAILABILITY EXTRACTION
    # ---------------------------------------------------------------------
    def _extract_training_availability(self, text: str):
        lowered = text.lower()

        days_per_week = None
        total_weeks = None

        day_patterns = [
            r"ซ้อมได้\s*(\d+)\s*วัน",
            r"มีเวลาซ้อม\s*(\d+)\s*วัน",
            r"วิ่งได้\s*(\d+)\s*วัน",
            r"ออกกำลังกายได้\s*(\d+)\s*วัน",
            r"(\d+)\s*วัน\s*/\s*สัปดาห์",
            r"(\d+)\s*วันต่อสัปดาห์",
            r"(\d+)\s*วัน\s*ต่อ\s*สัปดาห์",
            r"(\d+)\s*days?\s*/\s*week",
            r"(\d+)\s*days?\s*per\s*week",
        ]

        for pattern in day_patterns:
            match = re.search(pattern, lowered)
            if match:
                days_per_week = int(match.group(1))
                break

        week_patterns = [
            r"มีเวลา\s*(\d+)\s*สัปดาห์",
            r"ภายใน\s*(\d+)\s*สัปดาห์",
            r"อีก\s*(\d+)\s*สัปดาห์",
            r"ระยะเวลา\s*(\d+)\s*สัปดาห์",
            r"(\d+)\s*สัปดาห์",
            r"within\s*(\d+)\s*weeks?",
            r"in\s*(\d+)\s*weeks?",
            r"(\d+)\s*weeks?",
        ]

        for pattern in week_patterns:
            match = re.search(pattern, lowered)
            if match:
                total_weeks = int(match.group(1))
                break

        if days_per_week is not None:
            days_per_week = max(1, min(days_per_week, 7))

        if total_weeks is not None:
            total_weeks = max(1, min(total_weeks, 52))

        return days_per_week, total_weeks

    def _build_availability_hint(self, user_input: str) -> str:
        training_days_per_week, training_weeks = self._extract_training_availability(user_input)

        availability_hint = ""

        if training_days_per_week:
            availability_hint += f"\nซ้อมได้ {training_days_per_week} วันต่อสัปดาห์"
        else:
            availability_hint += "\nไม่ได้ระบุจำนวนวันซ้อม ให้ใช้ 4 วันต่อสัปดาห์สำหรับเป้าหมายลดเวลา"

        if training_weeks:
            availability_hint += f"\nมีเวลาเตรียมตัว {training_weeks} สัปดาห์"
        else:
            availability_hint += "\nไม่ได้ระบุจำนวนสัปดาห์ ให้แสดงแผน 4 สัปดาห์แรก และแนะนำให้ทำต่อรวม 8-12 สัปดาห์"

        return availability_hint

    # ---------------------------------------------------------------------
    # 4) SHOE CONSTRAINT EXTRACTION
    # ---------------------------------------------------------------------
    def _extract_shoe_constraints(self, user_input: str) -> dict:
        lowered = user_input.lower()

        budget = None
        run_type = None
        brand = None

        budget_patterns = [
            r"งบ(?:ไม่เกิน)?\s*(\d+(?:\.\d+)?)",
            r"ไม่เกิน\s*(\d+(?:\.\d+)?)\s*บาท",
            r"ราคา(?:ไม่เกิน)?\s*(\d+(?:\.\d+)?)",
            r"budget\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in budget_patterns:
            match = re.search(pattern, lowered)
            if match:
                budget = float(match.group(1))
                break

        if "easy" in lowered or "easy run" in lowered or "วิ่งสบาย" in lowered:
            run_type = "easy_run"
        elif "tempo" in lowered:
            run_type = "tempo"
        elif "interval" in lowered or "อินเทอร์วัล" in lowered:
            run_type = "interval"
        elif "race" in lowered or "แข่ง" in lowered or "race day" in lowered:
            run_type = "race_day"

        brand_keywords = {
            "new balance": "NEWBALANCE",
            "newbalance": "NEWBALANCE",
            "nike": "NIKE",
            "adidas": "ADIDAS",
            "asics": "ASICS",
            "hoka": "HOKA",
            "brooks": "BROOKS",
            "saucony": "SAUCONY",
            "on running": "ON",
            "xtep": "XTEP",
        }

        for keyword, brand_name in brand_keywords.items():
            if keyword in lowered:
                brand = brand_name
                break

        if brand is None:
            if re.search(r"\bon\b", lowered):
                brand = "ON"

        return {
            "budget": budget,
            "run_type": run_type,
            "brand": brand
        }

    def _build_shoe_constraint_hint(self, user_input: str) -> str:
        constraints = self._extract_shoe_constraints(user_input)
        budget = constraints.get("budget")
        run_type = constraints.get("run_type")
        brand = constraints.get("brand")

        hint = ""

        if brand:
            hint += f"\nแบรนด์ที่ผู้ใช้ระบุ: {brand}"
        else:
            hint += "\nไม่ได้ระบุแบรนด์"

        if run_type:
            run_type_th = {
                "easy_run": "easy run / วิ่งสบาย",
                "tempo": "tempo run / ซ้อมเร็วต่อเนื่อง",
                "interval": "interval / ซ้อมความเร็วเป็นช่วง",
                "race_day": "race day / วันแข่ง"
            }.get(run_type, run_type)

            hint += f"\nต้องการรองเท้าสำหรับ: {run_type_th}"

        if budget is not None:
            hint += f"\nงบประมาณไม่เกิน {budget:g} บาท"
        else:
            hint += "\nไม่ได้ระบุงบประมาณ ห้ามพูดว่าอยู่ในงบ"

        return hint

    # ---------------------------------------------------------------------
    # 5) QUERY HELPERS
    # ---------------------------------------------------------------------
    def _is_5k_query(self, lowered: str) -> bool:
        return (
            "5k" in lowered
            or "5 km" in lowered
            or "5km" in lowered
            or "5 กิโล" in lowered
        )

    def _is_10k_query(self, lowered: str) -> bool:
        return (
            "10k" in lowered
            or "10 km" in lowered
            or "10km" in lowered
            or "10 กิโล" in lowered
        )

    def _build_coach_query(self, user_input: str) -> str:
        lowered = user_input.lower()

        if self._is_5k_query(lowered):
            return "Chapter 13 5K 10K training R running I training T running easy running phase I"

        if self._is_10k_query(lowered):
            return "Chapter 13 5K 10K training R running I training T running easy running race pace"

        return "Chapter 13 5K 10K training easy running tempo interval race pace"

    # ---------------------------------------------------------------------
    # 6) RETRIEVAL + SCORE + RERANK
    # ---------------------------------------------------------------------
    def _retrieve_coach_docs_with_scores(self, search_query: str, user_input: str):
        logger.info("Tool call: search_coach_knowledge")
        logger.info(f"Tool input: {search_query}")

        raw_results = self.coach_db.similarity_search_with_score(
            search_query,
            k=self.fetch_k
        )

        lowered_user = user_input.lower()

        positive_keywords = []
        negative_keywords = []

        if self._is_5k_query(lowered_user):
            positive_keywords = [
                "5k", "10k", "chapter 13",
                "r running", "i training", "t running",
                "easy running", "phase i"
            ]
            negative_keywords = [
                "marathon", "ultradistance", "triathlon",
                "15k to 30k", "chapter 16", "chapter 17", "chapter 18"
            ]

        elif self._is_10k_query(lowered_user):
            positive_keywords = [
                "5k", "10k", "chapter 13",
                "r running", "i training", "t running",
                "easy running"
            ]
            negative_keywords = [
                "marathon", "ultradistance", "triathlon",
                "chapter 16", "chapter 17", "chapter 18"
            ]

        scored_results = []

        for doc, distance_score in raw_results:
            content_lower = doc.page_content.lower()

            keyword_bonus = sum(1 for kw in positive_keywords if kw in content_lower)
            keyword_penalty = sum(1 for kw in negative_keywords if kw in content_lower)

            adjusted_score = (
                distance_score
                - (0.25 * keyword_bonus)
                + (0.80 * keyword_penalty)
            )

            scored_results.append(
                (doc, distance_score, adjusted_score, keyword_bonus, keyword_penalty)
            )

        scored_results.sort(key=lambda x: x[2])
        selected_results = scored_results[:self.final_k]

        logger.info("Retrieved coach documents & reranked scores:")
        logger.info("Note: lower distance_score / adjusted_score means more relevant.")

        self._last_coach_debug_docs = []

        for i, (doc, distance_score, adjusted_score, bonus, penalty) in enumerate(selected_results, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            preview = doc.page_content[:350].replace("\n", " ")

            logger.info(
                f"[Coach Doc {i}] "
                f"distance_score={distance_score:.4f}, "
                f"adjusted_score={adjusted_score:.4f}, "
                f"keyword_bonus={bonus}, "
                f"keyword_penalty={penalty}, "
                f"source={source}, "
                f"page={page}, "
                f"content={preview[:300]}..."
            )

            self._last_coach_debug_docs.append({
                "rank": i,
                "source": source,
                "page": page,
                "distance_score": distance_score,
                "adjusted_score": adjusted_score,
                "keyword_bonus": bonus,
                "keyword_penalty": penalty,
                "preview": preview
            })

        return [doc for doc, _, _, _, _ in selected_results]

    def _retrieve_shoe_docs_with_scores(self, search_query: str, user_input: str):
        logger.info("Tool call: search_running_shoes")
        logger.info(f"Tool input: {search_query}")

        constraints = self._extract_shoe_constraints(user_input)
        budget = constraints.get("budget")
        run_type = constraints.get("run_type")
        brand_constraint = constraints.get("brand")

        logger.info(
            f"Shoe constraints: brand={brand_constraint}, run_type={run_type}, budget={budget}"
        )

        raw_results = self.shoe_db.similarity_search_with_score(
            search_query,
            k=self.shoe_fetch_k
        )

        scored_results = []

        for doc, distance_score in raw_results:
            metadata = doc.metadata

            brand = metadata.get("brand", "unknown")
            primary_use = metadata.get("primary_use", "unknown")
            recommended_runs = metadata.get("recommended_runs", "")
            price_thb = metadata.get("price_thb", -1)

            if brand_constraint is not None:
                if str(brand).upper() != brand_constraint:
                    continue

            try:
                price_float = float(price_thb)
            except Exception:
                price_float = -1.0

            easy_run_score = int(metadata.get("easy_run_score", 0))
            interval_score = int(metadata.get("interval_score", 0))
            tempo_score = int(metadata.get("tempo_score", 0))
            race_day_score = int(metadata.get("race_day_score", 0))

            if budget is not None:
                if price_float < 0 or price_float > budget:
                    continue

            suitability_score = 0

            if run_type == "easy_run":
                suitability_score = easy_run_score
            elif run_type == "interval":
                suitability_score = interval_score
            elif run_type == "tempo":
                suitability_score = tempo_score
            elif run_type == "race_day":
                suitability_score = race_day_score

            if run_type is not None and suitability_score <= 0:
                continue

            adjusted_score = distance_score - (0.70 * suitability_score)

            if run_type is not None and str(primary_use) == run_type:
                adjusted_score -= 0.80

            if run_type is not None and run_type in str(recommended_runs):
                adjusted_score -= 0.40

            if budget is not None and price_float <= budget:
                adjusted_score -= 0.30

            if brand_constraint is not None and str(brand).upper() == brand_constraint:
                adjusted_score -= 0.50

            scored_results.append(
                (
                    doc,
                    distance_score,
                    adjusted_score,
                    suitability_score,
                    price_float
                )
            )

        scored_results.sort(key=lambda x: x[2])
        selected_results = scored_results[:self.shoe_final_k]

        logger.info("Retrieved shoe documents & filtered/reranked scores:")
        logger.info("Note: lower distance_score / adjusted_score means more relevant.")

        self._last_shoe_debug_docs = []

        if not selected_results:
            logger.info("No shoe documents matched brand/budget/run_type constraints.")
            return []

        for i, (doc, distance_score, adjusted_score, suitability_score, price_float) in enumerate(selected_results, start=1):
            metadata = doc.metadata

            brand = metadata.get("brand", "unknown")
            model = metadata.get("model", "unknown")
            primary_use = metadata.get("primary_use", "unknown")
            recommended_runs = metadata.get("recommended_runs", "unknown")
            preview = doc.page_content[:350].replace("\n", " ")

            logger.info(
                f"[Shoe Doc {i}] "
                f"distance_score={distance_score:.4f}, "
                f"adjusted_score={adjusted_score:.4f}, "
                f"suitability_score={suitability_score}, "
                f"brand={brand}, "
                f"model={model}, "
                f"primary_use={primary_use}, "
                f"recommended_runs={recommended_runs}, "
                f"price_thb={price_float}, "
                f"content={preview[:300]}..."
            )

            self._last_shoe_debug_docs.append({
                "rank": i,
                "brand": brand,
                "model": model,
                "price_thb": price_float,
                "primary_use": primary_use,
                "recommended_runs": recommended_runs,
                "distance_score": distance_score,
                "adjusted_score": adjusted_score,
                "suitability_score": suitability_score,
                "easy_run_score": metadata.get("easy_run_score"),
                "tempo_score": metadata.get("tempo_score"),
                "interval_score": metadata.get("interval_score"),
                "race_day_score": metadata.get("race_day_score"),
                "preview": preview
            })

        return [doc for doc, _, _, _, _ in selected_results]

    # ---------------------------------------------------------------------
    # 7) RESPONSE CONTEXT BUILDERS
    # ---------------------------------------------------------------------
    def _build_clean_shoe_context(self, docs, user_input: str):
        constraints = self._extract_shoe_constraints(user_input)
        brand_constraint = constraints.get("brand")
        budget = constraints.get("budget")
        run_type = constraints.get("run_type")

        if not docs:
            return "ไม่พบข้อมูลรองเท้าที่ตรงเงื่อนไข"

        unique_items = {}

        for doc in docs:
            metadata = doc.metadata

            brand = metadata.get("brand", "ไม่ทราบแบรนด์")
            model = metadata.get("model", "ไม่ทราบรุ่น")
            price = metadata.get("price_thb", "ไม่ทราบราคา")
            primary_use = metadata.get("primary_use", "")
            recommended_runs = metadata.get("recommended_runs", "")

            if brand_constraint is not None:
                if str(brand).upper() != brand_constraint:
                    continue

            try:
                price_float = float(price)
            except Exception:
                price_float = -1.0

            if budget is not None:
                if price_float < 0 or price_float > budget:
                    continue

            if run_type is not None:
                if run_type not in str(primary_use) and run_type not in str(recommended_runs):
                    continue

            model_key = str(model)
            model_key = re.sub(r"\s+", " ", model_key).strip().upper()
            model_key = model_key.replace(" รองเท้าวิ่งถนนผู้ชาย", "")
            model_key = model_key.replace(" รองเท้าวิ่งถนนผู้หญิง", "")
            model_key = model_key.replace("ผู้ชาย", "")
            model_key = model_key.replace("ผู้หญิง", "")
            model_key = model_key.strip()

            content = doc.page_content

            reason_match = re.search(r"Classification Reason TH:\s*(.*)", content)
            caution_match = re.search(r"Caution TH:\s*(.*)", content)

            reason = reason_match.group(1).strip() if reason_match else ""
            caution = caution_match.group(1).strip() if caution_match else ""

            if model_key not in unique_items:
                unique_items[model_key] = {
                    "brand": brand,
                    "model": model,
                    "prices": [],
                    "primary_use": primary_use,
                    "recommended_runs": recommended_runs,
                    "reason": reason,
                    "caution": caution
                }

            if price_float > 0:
                unique_items[model_key]["prices"].append(price_float)

        if not unique_items:
            return "ไม่พบข้อมูลรองเท้าที่ตรงเงื่อนไข"

        lines = []

        for idx, (_, item) in enumerate(list(unique_items.items())[:3], start=1):
            brand = item["brand"]
            model = item["model"]
            primary_use = item["primary_use"]
            recommended_runs = item["recommended_runs"]
            reason = item["reason"]
            caution = item["caution"]
            prices = item["prices"]

            if prices:
                min_price = min(prices)
                max_price = max(prices)
                if min_price == max_price:
                    price_text = f"ประมาณ {min_price:g} บาท"
                else:
                    price_text = f"ประมาณ {min_price:g}-{max_price:g} บาท"
            else:
                price_text = "ไม่ทราบราคา"

            primary_use_th = {
                "easy_run": "เหมาะใช้หลักสำหรับ easy run / วิ่งสบาย",
                "tempo": "เหมาะใช้หลักสำหรับ tempo run / ซ้อมเร็วต่อเนื่อง",
                "interval": "เหมาะใช้หลักสำหรับ interval / ซ้อมความเร็วเป็นช่วง",
                "race_day": "เหมาะใช้หลักสำหรับวันแข่ง"
            }.get(str(primary_use), f"เหมาะใช้หลักสำหรับ {primary_use}")

            clean_recommended = str(recommended_runs).replace(",", ", ")

            line = f"""
{idx}. {brand} {model}
ราคา: {price_text}
การใช้งานหลัก: {primary_use_th}
เหมาะกับการซ้อม: {clean_recommended}
เหตุผล: {reason if reason else "เหมาะกับประเภทการซ้อมที่ผู้ใช้ระบุจากข้อมูลในฐานข้อมูล"}
ข้อควรระวัง: {caution if caution else "จากข้อมูลที่มี ไม่พบข้อควรระวังเฉพาะ แนะนำให้ลองไซซ์และความกระชับก่อนเลือกซื้อ"}
"""
            lines.append(line.strip())

        return "\n\n".join(lines)

    def _build_training_template_hint(self, user_input: str) -> str:
        days_per_week, total_weeks = self._extract_training_availability(user_input)

        if not days_per_week:
            days_per_week = 4

        lowered = user_input.lower()

        is_5k = self._is_5k_query(lowered)
        is_10k = self._is_10k_query(lowered)

        if is_5k and days_per_week == 3:
            return """
ให้ใช้แผนนี้เป็นหลัก ห้ามสรุปรวม และต้องแสดงแยกรายสัปดาห์/รายวันตามนี้:

สัปดาห์ที่ 1:
- วันซ้อมที่ 1: Easy run 25-30 นาที ที่ pace 7:30-8:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Tempo run 10 นาที ที่ pace 6:30-6:50 นาที/กิโลเมตร หลังวอร์มอัป
- วันซ้อมที่ 3: Long easy run 4-5 กิโลเมตร ที่ pace 7:45-8:45 นาที/กิโลเมตร

สัปดาห์ที่ 2:
- วันซ้อมที่ 1: Easy run 30 นาที ที่ pace 7:30-8:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Interval 400 เมตร x 4 รอบ ที่ pace 5:45-6:00 นาที/กิโลเมตร พักเดิน/จ็อก 2 นาที
- วันซ้อมที่ 3: Long easy run 5 กิโลเมตร ที่ pace 7:45-8:45 นาที/กิโลเมตร

สัปดาห์ที่ 3:
- วันซ้อมที่ 1: Easy run 30-35 นาที ที่ pace 7:30-8:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Tempo run 15 นาที ที่ pace 6:30-6:50 นาที/กิโลเมตร
- วันซ้อมที่ 3: Long easy run 5-6 กิโลเมตร ที่ pace 7:45-8:45 นาที/กิโลเมตร

สัปดาห์ที่ 4:
- วันซ้อมที่ 1: Easy run 30 นาที ที่ pace 7:30-8:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Interval 400 เมตร x 5 รอบ ที่ pace 5:45-6:00 นาที/กิโลเมตร พักเดิน/จ็อก 2 นาที
- วันซ้อมที่ 3: ทดสอบ 5K แบบคุมแรง ไม่ต้องกดสุด หรือ Long easy run 5 กิโลเมตร
"""

        if is_5k and days_per_week == 4:
            return """
ให้ใช้แผนนี้เป็นหลัก ห้ามสรุปรวม และต้องแสดงแยกรายสัปดาห์/รายวันตามนี้:

สัปดาห์ที่ 1:
- วันซ้อมที่ 1: Easy run 25-30 นาที ที่ pace 7:30-8:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Tempo run 10-12 นาที ที่ pace 6:30-6:50 นาที/กิโลเมตร หลังวอร์มอัป
- วันซ้อมที่ 3: Recovery easy run 20-25 นาที ที่ pace 8:00-9:00 นาที/กิโลเมตร
- วันซ้อมที่ 4: Long easy run 4-5 กิโลเมตร ที่ pace 7:45-8:45 นาที/กิโลเมตร

สัปดาห์ที่ 2:
- วันซ้อมที่ 1: Easy run 30 นาที ที่ pace 7:30-8:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Interval 400 เมตร x 4 รอบ ที่ pace 5:45-6:00 นาที/กิโลเมตร พักเดิน/จ็อก 2 นาที
- วันซ้อมที่ 3: Recovery easy run 25 นาที ที่ pace 8:00-9:00 นาที/กิโลเมตร
- วันซ้อมที่ 4: Long easy run 5 กิโลเมตร ที่ pace 7:45-8:45 นาที/กิโลเมตร

สัปดาห์ที่ 3:
- วันซ้อมที่ 1: Easy run 30-35 นาที ที่ pace 7:30-8:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Tempo run 15 นาที ที่ pace 6:30-6:50 นาที/กิโลเมตร
- วันซ้อมที่ 3: Recovery easy run 25 นาที ที่ pace 8:00-9:00 นาที/กิโลเมตร
- วันซ้อมที่ 4: Long easy run 5-6 กิโลเมตร ที่ pace 7:45-8:45 นาที/กิโลเมตร

สัปดาห์ที่ 4:
- วันซ้อมที่ 1: Easy run 30 นาที ที่ pace 7:30-8:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Interval 400 เมตร x 5 รอบ ที่ pace 5:45-6:00 นาที/กิโลเมตร พักเดิน/จ็อก 2 นาที
- วันซ้อมที่ 3: Recovery easy run 20-25 นาที ที่ pace 8:00-9:00 นาที/กิโลเมตร
- วันซ้อมที่ 4: ทดสอบ 5K แบบคุมแรง ไม่ต้องกดสุด หรือ Long easy run 5 กิโลเมตร
"""

        if is_10k and days_per_week == 3:
            return """
ให้ใช้แผนนี้เป็นหลัก ห้ามสรุปรวม และต้องแสดงแยกรายสัปดาห์/รายวันตามนี้:

สัปดาห์ที่ 1:
- วันซ้อมที่ 1: Easy run 30 นาที ที่ pace 6:45-7:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Tempo run 12-15 นาที ที่ pace 5:45-5:55 นาที/กิโลเมตร หลังวอร์มอัป
- วันซ้อมที่ 3: Long easy run 6 กิโลเมตร ที่ pace 6:50-7:40 นาที/กิโลเมตร

สัปดาห์ที่ 2:
- วันซ้อมที่ 1: Easy run 35 นาที ที่ pace 6:45-7:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Interval 400 เมตร x 5 รอบ ที่ pace 5:10-5:25 นาที/กิโลเมตร พักเดิน/จ็อก 2 นาที
- วันซ้อมที่ 3: Long easy run 7 กิโลเมตร ที่ pace 6:50-7:40 นาที/กิโลเมตร

สัปดาห์ที่ 3:
- วันซ้อมที่ 1: Easy run 35-40 นาที ที่ pace 6:45-7:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Tempo run 18-20 นาที ที่ pace 5:40-5:55 นาที/กิโลเมตร
- วันซ้อมที่ 3: Long easy run 8 กิโลเมตร ที่ pace 6:50-7:40 นาที/กิโลเมตร

สัปดาห์ที่ 4:
- วันซ้อมที่ 1: Easy run 30-35 นาที ที่ pace 6:45-7:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Interval 800 เมตร x 4 รอบ ที่ pace 5:10-5:25 นาที/กิโลเมตร พักจ็อก 2-3 นาที
- วันซ้อมที่ 3: Long easy run 8-9 กิโลเมตร ที่ pace 6:50-7:40 นาที/กิโลเมตร

หลังสัปดาห์ที่ 4:
- สัปดาห์ที่ 5-6: เพิ่ม tempo เป็น 20-25 นาที หรือเพิ่ม interval ทีละ 1 รอบ
- สัปดาห์ที่ 7: ลดความหนักลงเล็กน้อย แต่ยังแตะ race pace 5:30 นาที/กิโลเมตร เป็นช่วงสั้น ๆ
- สัปดาห์ที่ 8: taper ลดระยะลง และทดสอบ 10K โดยพยายามคุม pace ใกล้ 5:30 นาที/กิโลเมตร
"""

        if is_10k and days_per_week == 4:
            return """
ให้ใช้แผนนี้เป็นหลัก ห้ามสรุปรวม และต้องแสดงแยกรายสัปดาห์/รายวันตามนี้:

สัปดาห์ที่ 1:
- วันซ้อมที่ 1: Easy run 30 นาที ที่ pace 6:45-7:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Tempo run 12-15 นาที ที่ pace 5:45-5:55 นาที/กิโลเมตร หลังวอร์มอัป
- วันซ้อมที่ 3: Recovery easy run 25 นาที ที่ pace 7:00-7:45 นาที/กิโลเมตร
- วันซ้อมที่ 4: Long easy run 6-7 กิโลเมตร ที่ pace 6:50-7:40 นาที/กิโลเมตร

สัปดาห์ที่ 2:
- วันซ้อมที่ 1: Easy run 35 นาที ที่ pace 6:45-7:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Interval 400 เมตร x 5 รอบ ที่ pace 5:10-5:25 นาที/กิโลเมตร พักเดิน/จ็อก 2 นาที
- วันซ้อมที่ 3: Recovery easy run 25-30 นาที ที่ pace 7:00-7:45 นาที/กิโลเมตร
- วันซ้อมที่ 4: Long easy run 7-8 กิโลเมตร ที่ pace 6:50-7:40 นาที/กิโลเมตร

สัปดาห์ที่ 3:
- วันซ้อมที่ 1: Easy run 35-40 นาที ที่ pace 6:45-7:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Tempo run 20 นาที ที่ pace 5:40-5:55 นาที/กิโลเมตร
- วันซ้อมที่ 3: Recovery easy run 25 นาที ที่ pace 7:00-7:45 นาที/กิโลเมตร
- วันซ้อมที่ 4: Long easy run 8-9 กิโลเมตร ที่ pace 6:50-7:40 นาที/กิโลเมตร

สัปดาห์ที่ 4:
- วันซ้อมที่ 1: Easy run 30-35 นาที ที่ pace 6:45-7:30 นาที/กิโลเมตร
- วันซ้อมที่ 2: Interval 800 เมตร x 4 รอบ ที่ pace 5:10-5:25 นาที/กิโลเมตร พักจ็อก 2-3 นาที
- วันซ้อมที่ 3: Recovery easy run 20-25 นาที ที่ pace 7:00-7:45 นาที/กิโลเมตร
- วันซ้อมที่ 4: Long easy run 8 กิโลเมตร ที่ pace 6:50-7:40 นาที/กิโลเมตร
"""

        return ""

    # ---------------------------------------------------------------------
    # 8) DEBUG HELPERS
    # ---------------------------------------------------------------------
    def _build_constraints_debug(self, user_input: str) -> dict:
        distance_km, target_minutes, current_minutes = self._extract_goal_distance_and_time(user_input)
        days_per_week, total_weeks = self._extract_training_availability(user_input)
        shoe_constraints = self._extract_shoe_constraints(user_input)

        return {
            "running": {
                "distance_km": distance_km,
                "target_minutes": target_minutes,
                "current_minutes": current_minutes,
                "training_days_per_week": days_per_week,
                "training_weeks": total_weeks
            },
            "shoe": shoe_constraints
        }

    # ---------------------------------------------------------------------
    # 9) MAIN INVOKE
    # ---------------------------------------------------------------------
    def invoke(self, inputs: dict):
        user_input = inputs.get("input", "")
        chat_history = inputs.get("chat_history", "")

        self._last_coach_debug_docs = []
        self._last_shoe_debug_docs = []

        action_plan = self._create_action_plan_with_llm(
            user_input=user_input,
            chat_history=chat_history
        )

        response_mode = self._get_response_mode(action_plan)
        logger.info(f"Response mode: {response_mode}")

        retrieved_coach_docs = []
        retrieved_shoe_docs = []
        tool_calls = []

        pace_info = ""
        if action_plan.get("use_pace_tool"):
            pace_info = self._get_pace_info(user_input)

            distance_km, target_minutes, current_minutes = self._extract_goal_distance_and_time(user_input)

            if distance_km and target_minutes:
                tool_calls.append({
                    "tool": "calculate_pace",
                    "input": {
                        "distance_km": distance_km,
                        "time_minutes": target_minutes,
                        "type": "target"
                    },
                    "output": f"calculated target pace for {distance_km} km / {target_minutes} min"
                })

            if distance_km and current_minutes:
                tool_calls.append({
                    "tool": "calculate_pace",
                    "input": {
                        "distance_km": distance_km,
                        "time_minutes": current_minutes,
                        "type": "current"
                    },
                    "output": f"calculated current pace for {distance_km} km / {current_minutes} min"
                })

        pace_zone_hint = self._build_pace_zone_hint(user_input)
        availability_hint = self._build_availability_hint(user_input)
        shoe_constraint_hint = self._build_shoe_constraint_hint(user_input)
        training_template_hint = self._build_training_template_hint(user_input)

        if action_plan.get("use_coach_search"):
            coach_query = action_plan.get("coach_query") or self._build_coach_query(user_input)

            retrieved_coach_docs = self._retrieve_coach_docs_with_scores(
                search_query=coach_query,
                user_input=user_input
            )

            tool_calls.append({
                "tool": "search_coach_knowledge",
                "input": coach_query,
                "output": f"retrieved {len(retrieved_coach_docs)} coach documents"
            })

        if action_plan.get("use_shoe_search"):
            shoe_query = action_plan.get("shoe_query") or user_input

            retrieved_shoe_docs = self._retrieve_shoe_docs_with_scores(
                search_query=shoe_query,
                user_input=user_input
            )

            tool_calls.append({
                "tool": "search_running_shoes",
                "input": shoe_query,
                "output": f"retrieved {len(retrieved_shoe_docs)} shoe documents"
            })

        coach_context = "\n\n".join(
            [doc.page_content[:500] for doc in retrieved_coach_docs]
        )

        clean_shoe_context = self._build_clean_shoe_context(retrieved_shoe_docs, user_input)

        if response_mode == "out_of_scope":
            final_prompt = f"""
คุณคือ Running Coach Agent

คำถามผู้ใช้:
{user_input}

คำถามนี้อยู่นอกเหนือขอบเขตของระบบ

ขอบเขตที่ระบบสามารถช่วยได้:
1. ให้คำแนะนำแผนซ้อมวิ่ง 5K/10K
2. คำนวณ pace ปัจจุบันและ pace เป้าหมาย
3. แนะนำรองเท้าวิ่งตามประเภทการซ้อม งบประมาณ และแบรนด์
4. ตอบคำถามที่เกี่ยวข้องกับ easy run, tempo run, interval, long run และการพัฒนาการวิ่ง

ให้ตอบภาษาไทยแบบสุภาพ กระชับ และไม่ต้องเรียกใช้ข้อมูล RAG

คำตอบ:
"""

        elif response_mode == "shoe_only":
            final_prompt = f"""
คุณคือผู้ช่วยแนะนำรองเท้าวิ่ง ตอบภาษาไทยแบบเป็นธรรมชาติ

คำถามนี้เป็นคำถามเรื่องรองเท้าเท่านั้น
ห้ามตอบเรื่อง pace
ห้ามจัดแผนซ้อม
ห้ามเดาเป้าหมายการวิ่งจาก chat history
ตอบเฉพาะรองเท้าที่ตรงแบรนด์ / ประเภทการซ้อม / งบประมาณล่าสุดที่ผู้ใช้ถาม

ห้ามใช้คำเหล่านี้ในคำตอบสุดท้าย:
Primary Use, Recommended Runs, price_thb, run_suitability_score, classification_reason_th, caution_th, RAG

กฎ:
- ตอบภาษาไทยเท่านั้น
- ห้ามแต่งชื่อรุ่นรองเท้าเอง
- ใช้เฉพาะข้อมูลรองเท้าที่ให้มา
- ถ้าไม่พบข้อมูลรองเท้าที่ตรงเงื่อนไข ให้บอกว่าไม่พบจากฐานข้อมูล
- ถ้าผู้ใช้ระบุแบรนด์ ให้แนะนำเฉพาะแบรนด์นั้น
- ถ้าผู้ใช้ระบุงบ ห้ามแนะนำรุ่นที่เกินงบ
- ถ้าไม่ได้ระบุงบ ห้ามพูดว่าอยู่ในงบ
- ห้ามพูดชื่อ field หรือ variable
- ถ้ามีรองเท้าที่ตรงเงื่อนไขหลายรุ่น ให้แนะนำ 2-3 รุ่น ไม่ใช่แค่รุ่นเดียว
- ถ้ารุ่นซ้ำกันแต่ต่างสี/รหัสสินค้า ให้รวมเป็นรุ่นเดียว และบอกช่วงราคา
- ให้เปรียบเทียบสั้น ๆ ว่าแต่ละรุ่นเหมาะกับใคร

คำถามผู้ใช้:
{user_input}

เงื่อนไขรองเท้า:
{shoe_constraint_hint}

ข้อมูลรองเท้าที่ตรงเงื่อนไข:
{clean_shoe_context}

ให้ตอบตามโครงสร้างนี้เท่านั้น:
1. สรุปเงื่อนไขที่ผู้ใช้ต้องการ
2. รุ่นที่แนะนำจากฐานข้อมูล 2-3 รุ่น ถ้ามี
3. เปรียบเทียบสั้น ๆ ว่าแต่ละรุ่นเหมาะกับใคร
4. ข้อควรระวังหรือทางเลือก ถ้าไม่พบรุ่นที่ตรงเงื่อนไข

คำตอบ:
"""

        elif response_mode == "coach_only":
            final_prompt = f"""
คุณคือโค้ชวิ่ง AI ตอบภาษาไทยแบบเป็นธรรมชาติ

ห้ามใช้คำเหล่านี้ในคำตอบสุดท้ายเด็ดขาด:
RAG, Phase, Primary Use, Recommended Runs, price_thb, run_suitability_score, classification_reason_th, caution_th, emphasis

กฎ:
- ตอบภาษาไทยเท่านั้น
- ห้ามตอบเรื่องรองเท้า
- ห้ามแนะนำรองเท้า
- ห้าม copy ตารางจากเอกสาร
- ใช้ pace จาก calculator เท่านั้น
- ถ้ามีแผนซ้อม ให้จัดตามจำนวนวันซ้อมที่ผู้ใช้บอก
- ถ้ามี “โครงสร้างแผนซ้อมที่ควรใช้” ให้ใช้แผนนั้นเป็นหลักและต้องแสดงแยกรายสัปดาห์/รายวัน
- ห้ามสรุปแผนซ้อมแบบรวม ๆ เช่น “วิ่งระยะสั้น 3 ครั้ง”
- ต้องระบุชนิด workout, ระยะเวลา/ระยะทาง, และ pace ของแต่ละวันซ้อม
- ถ้าผู้ใช้บอกซ้อมได้ 3 วัน/สัปดาห์ ต้องมี 3 วันซ้อมต่อสัปดาห์เท่านั้น
- ถ้าผู้ใช้บอกซ้อมได้ 4 วัน/สัปดาห์ ต้องมี 4 วันซ้อมต่อสัปดาห์เท่านั้น

คำถามผู้ใช้:
{user_input}

ข้อมูล pace:
{pace_info}

ข้อมูล pace zone:
{pace_zone_hint}

ข้อมูลวันซ้อม:
{availability_hint}

โครงสร้างแผนซ้อมที่ควรใช้:
{training_template_hint}

ข้อมูลอ้างอิงการฝึกวิ่งแบบย่อ:
{coach_context}

ให้ตอบตามโครงสร้างนี้เท่านั้น:
1. สรุปเป้าหมาย
2. Pace ปัจจุบันและ pace เป้าหมาย
3. แผนซ้อม 4 สัปดาห์แรก
4. แนวทางสัปดาห์ที่เหลือ ถ้าผู้ใช้ระบุเวลามากกว่า 4 สัปดาห์
5. ข้อควรระวัง

คำตอบ:
"""

        elif response_mode == "mixed":
            final_prompt = f"""
คุณคือโค้ชวิ่ง AI และผู้ช่วยแนะนำรองเท้าวิ่ง ตอบภาษาไทยแบบเป็นธรรมชาติ

คำถามนี้มีทั้งเรื่องแผนซ้อมและรองเท้า
ให้ตอบแยกเป็น 2 ส่วนชัดเจน:
ส่วนที่ 1 แผนซ้อม
ส่วนที่ 2 รองเท้าที่แนะนำ

ห้ามใช้คำเหล่านี้ในคำตอบสุดท้ายเด็ดขาด:
RAG, Phase, Primary Use, Recommended Runs, price_thb, run_suitability_score, classification_reason_th, caution_th, emphasis

กฎฝั่งแผนซ้อม:
- ใช้ pace จาก calculator เท่านั้น
- ห้าม copy ตารางจากเอกสาร
- ถ้ามี “โครงสร้างแผนซ้อมที่ควรใช้” ให้ใช้แผนนั้นเป็นหลักและต้องแสดงแยกรายสัปดาห์/รายวัน
- จัดตามจำนวนวันซ้อมที่ผู้ใช้บอก
- ห้ามสรุปแผนซ้อมแบบรวม ๆ

กฎฝั่งรองเท้า:
- ใช้เฉพาะข้อมูลรองเท้าที่ให้มา
- ห้ามแต่งชื่อรุ่นรองเท้าเอง
- ถ้าผู้ใช้ระบุแบรนด์ ให้แนะนำเฉพาะแบรนด์นั้น
- ถ้าผู้ใช้ระบุงบ ห้ามแนะนำรุ่นที่เกินงบ
- ถ้าไม่ได้ระบุงบ ห้ามพูดว่าอยู่ในงบ
- ห้ามพูดชื่อ field หรือ variable
- ถ้ามีรองเท้าที่ตรงเงื่อนไขหลายรุ่น ให้แนะนำ 2-3 รุ่น

คำถามผู้ใช้:
{user_input}

ข้อมูล pace:
{pace_info}

ข้อมูล pace zone:
{pace_zone_hint}

ข้อมูลวันซ้อม:
{availability_hint}

โครงสร้างแผนซ้อมที่ควรใช้:
{training_template_hint}

เงื่อนไขรองเท้า:
{shoe_constraint_hint}

ข้อมูลรองเท้าแบบภาษาคน:
{clean_shoe_context}

ข้อมูลอ้างอิงการฝึกวิ่งแบบย่อ:
{coach_context}

ให้ตอบตามโครงสร้างนี้เท่านั้น:

ส่วนที่ 1: แผนซ้อม
1. สรุปเป้าหมาย
2. Pace ปัจจุบันและ pace เป้าหมาย
3. แผนซ้อม 4 สัปดาห์แรก
4. แนวทางสัปดาห์ที่เหลือ ถ้ามี

ส่วนที่ 2: รองเท้าที่แนะนำ
1. สรุปเงื่อนไขรองเท้า
2. รุ่นที่แนะนำจากฐานข้อมูล 2-3 รุ่น ถ้ามี
3. เปรียบเทียบสั้น ๆ ว่าแต่ละรุ่นเหมาะกับใคร

คำตอบ:
"""

        else:
            final_prompt = f"""
คุณคือผู้ช่วยโค้ชวิ่ง ตอบภาษาไทยแบบกระชับ

คำถามผู้ใช้:
{user_input}

ตอบให้ตรงคำถาม ไม่ต้องเดาข้อมูลที่ผู้ใช้ไม่ได้ให้

คำตอบ:
"""

        logger.info("Agent reasoning: ส่ง context ที่คัดแล้วให้ LLM สร้างคำตอบสุดท้าย")

        response = self.llm.invoke(final_prompt)

        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        return {
            "output": answer,
            "response_mode": response_mode,
            "action_plan": action_plan,
            "tool_calls": tool_calls,
            "retrieved_coach_docs": self._last_coach_debug_docs,
            "retrieved_shoe_docs": self._last_shoe_debug_docs,
            "constraints": self._build_constraints_debug(user_input)
        }


def create_running_coach_agent():
    return RunningCoachAgent()