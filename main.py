import os

# ปิดระบบ Telemetry ของ ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

import warnings
import traceback

from src.agent.llm_agent import create_running_coach_agent
from src.utils.logger import logger

warnings.filterwarnings("ignore")


def main():
    logger.info("Initializing Agentic RAG - Running Coach & Shoe Recommender...")

    try:
        agent_executor = create_running_coach_agent()
        logger.info("Agent is ready! Type 'exit' to quit.\n")

        chat_history_text = ""

        while True:
            user_input = input("\n🏃 User: ")

            if user_input.lower().strip() in ["exit", "quit"]:
                logger.info("Exit program.")
                break

            logger.info("Agent is processing the request...")

            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history_text
            })

            answer = response.get("output", "ขออภัยค่ะ ระบบไม่สามารถสร้างคำตอบได้")

            print("\n" + "=" * 50)
            print(f"👟 Coach Agent: {answer}")
            print("=" * 50 + "\n")

            chat_history_text += f"User: {user_input}\nCoach: {answer}\n\n"

    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")

    except Exception:
        logger.error("System crashed! Showing full Traceback below:")
        traceback.print_exc()


if __name__ == "__main__":
    main()