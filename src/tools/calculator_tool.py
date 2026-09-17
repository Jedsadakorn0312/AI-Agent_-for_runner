from langchain.tools import tool
import re


@tool
def calculate_pace(tool_input: str) -> str:
    """
    ใช้คำนวณ pace นาที/กิโลเมตร

    รูปแบบ input ที่แนะนำ:
    distance_km=5, time_minutes=30

    หรือข้อความที่มีตัวเลข 2 ตัว เช่น:
    5 km 30 minutes
    """

    try:
        numbers = re.findall(r"\d+(?:\.\d+)?", str(tool_input))

        if len(numbers) < 2:
            return (
                "ข้อผิดพลาด: กรุณาระบุระยะทางและเวลา เช่น "
                "distance_km=5, time_minutes=30"
            )

        distance_km = float(numbers[0])
        time_minutes = float(numbers[1])

        if distance_km <= 0 or time_minutes <= 0:
            return "ข้อผิดพลาด: ระยะทางและเวลาต้องเป็นตัวเลขที่มากกว่า 0"

        pace_decimal = time_minutes / distance_km

        minutes = int(pace_decimal)
        seconds = int(round((pace_decimal - minutes) * 60))

        if seconds == 60:
            minutes += 1
            seconds = 0

        return (
            f"ระยะทาง {distance_km:g} กิโลเมตร ใช้เวลา {time_minutes:g} นาที "
            f"จะได้ pace ประมาณ {minutes}:{seconds:02d} นาที/กิโลเมตร"
        )

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการคำนวณ pace: {str(e)}"