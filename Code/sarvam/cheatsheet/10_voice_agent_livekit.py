"""
10_voice_agent_livekit.py
=========================
Sarvam AI - Real-Time Multilingual Voice Agent using LiveKit
Covers:
  - LiveKit AgentSession with Sarvam STT (Saaras v3) and TTS (Bulbul v3)
  - Auto language detection (language="unknown")
  - Customisable agent persona via instructions
  - OpenAI GPT-4o as the LLM brain (replace with your preferred LLM)
  - Three ready-to-use agent personas:
      TutorAgent      - multilingual educational assistant
      BankingAgent    - EMI/account queries in Indian languages
      GovSchemeAgent  - government scheme awareness bot

Prerequisites:
  pip install "livekit-agents[sarvam,openai,silero]" python-dotenv

Environment variables (.env file or exports):
  LIVEKIT_URL         = wss://your-project.livekit.cloud
  LIVEKIT_API_KEY     = APIxxxxxxxx
  LIVEKIT_API_SECRET  = xxxxxxxxxxxxxxxx
  SARVAM_API_KEY      = sk_xxxxxxxxxxxxxxxx
  OPENAI_API_KEY      = sk-proj-xxxxxxxxxxxxxxxx

Usage:
  # Run the agent worker (keep running):
  python 10_voice_agent_livekit.py dev

  # Test via console (type instead of speaking):
  python 10_voice_agent_livekit.py console

  # Change the active agent persona by editing ACTIVE_AGENT below.

Docs:
  Sarvam LiveKit integration:
  https://docs.sarvam.ai/api-reference-docs/integration/build-voice-agent-with-live-kit
  LiveKit Sarvam STT plugin:
  https://docs.livekit.io/agents/models/stt/plugins/sarvam/
"""

import logging
import os

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, sarvam

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

logger = logging.getLogger("sarvam-voice-agent")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Choose which agent to run:
#   "tutor"   -> multilingual educational assistant
#   "banking" -> EMI and account query bot
#   "gov"     -> government scheme awareness bot
# ---------------------------------------------------------------------------
ACTIVE_AGENT = "tutor"   # change this to "banking" or "gov"


# ---------------------------------------------------------------------------
# Agent 1: Multilingual Tutor
# ---------------------------------------------------------------------------
class TutorAgent(Agent):
    """
    Friendly educational assistant that explains concepts in the student's
    own language. Supports Hindi, Gujarati, Marathi, Bengali, Kannada.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a friendly and patient tutor.
                Explain concepts clearly and simply in the student's language.
                Support Hindi, Gujarati, Marathi, Bengali, Kannada, and English.
                Use examples from daily Indian life to illustrate concepts.
                Always encourage the student and never make them feel bad for asking.
                If the student switches language mid-conversation, match their language.
            """,
            stt=sarvam.STT(
                language="unknown",   # auto-detect: student speaks any language
                model="saaras:v3",
                mode="transcribe",
            ),
            llm=openai.LLM(model="gpt-4o"),
            tts=sarvam.TTS(
                target_language_code="hi-IN",  # default; update per student
                model="bulbul:v3",
                speaker="anand",       # professional, warm male voice
            ),
        )

    async def on_enter(self):
        """Greet the student when they join."""
        self.session.generate_reply()


# ---------------------------------------------------------------------------
# Agent 2: Banking / EMI Collection Agent
# ---------------------------------------------------------------------------
class BankingAgent(Agent):
    """
    Professional banking agent that handles EMI reminders, account queries,
    and payment guidance in multiple Indian languages.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a professional and empathetic banking agent for ABC Bank.

                Customer account details:
                - EMI Amount  : Rs. 5,000
                - Due Date    : 15th of each month
                - Loan Type   : Personal Loan

                Your responsibilities:
                - Remind customers about pending EMI payments
                - Explain payment methods: UPI, net banking, bank branch
                - Address queries about due dates, late fees, payment plans
                - Be empathetic if customer has financial difficulties
                - Never be aggressive or threatening

                Payment methods to mention:
                - UPI to ABC Bank
                - ABC Bank net banking portal
                - ABC Bank mobile app
                - Visit the nearest ABC Bank branch

                Start by greeting the customer and politely mentioning their
                pending EMI of Rs. 5,000.

                Support: Hindi, Gujarati, Marathi, Bengali, English.
                Always reply in the language the customer speaks.
            """,
            stt=sarvam.STT(
                language="unknown",   # auto-detect customer's language
                model="saaras:v3",
                mode="transcribe",
            ),
            llm=openai.LLM(model="gpt-4o"),
            tts=sarvam.TTS(
                target_language_code="en-IN",
                model="bulbul:v3",
                speaker="aditya",     # professional male voice
            ),
        )

    async def on_enter(self):
        self.session.generate_reply()


# ---------------------------------------------------------------------------
# Agent 3: Government Scheme Awareness Bot
# ---------------------------------------------------------------------------
class GovSchemeAgent(Agent):
    """
    Explains government schemes to rural and semi-urban users in their
    local language. Covers schemes like PM-JAY, PM Kisan, Jan Dhan.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a helpful government scheme awareness assistant.
                Explain government welfare schemes in simple language.

                Schemes you know about:
                - PM-JAY (Ayushman Bharat): Free health insurance up to Rs. 5 lakh
                - PM Kisan: Rs. 6,000/year for farmers in 3 instalments
                - Jan Dhan Yojana: Free zero-balance bank account
                - PM Ujjwala: Free LPG connection for BPL families
                - Sukanya Samriddhi: Savings scheme for girl child

                Guidelines:
                - Use very simple language, avoid jargon
                - Explain how to apply and what documents are needed
                - Always mention the official helpline numbers where known
                - Be patient and repeat information if asked

                Support all Indian languages. Reply in the language used by the user.
            """,
            stt=sarvam.STT(
                language="unknown",
                model="saaras:v3",
                mode="transcribe",
            ),
            llm=openai.LLM(model="gpt-4o"),
            tts=sarvam.TTS(
                target_language_code="hi-IN",
                model="bulbul:v3",
                speaker="priya",    # warm female voice
            ),
        )

    async def on_enter(self):
        self.session.generate_reply()


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
AGENTS = {
    "tutor":   TutorAgent,
    "banking": BankingAgent,
    "gov":     GovSchemeAgent,
}


# ---------------------------------------------------------------------------
# LiveKit entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    """Called by LiveKit when a user connects to the room."""
    logger.info(f"User connected to room: {ctx.room.name}")
    logger.info(f"Active agent: {ACTIVE_AGENT}")

    agent_class = AGENTS.get(ACTIVE_AGENT, TutorAgent)
    session = AgentSession()
    await session.start(
        agent=agent_class(),
        room=ctx.room,
    )


# ---------------------------------------------------------------------------
# Language customisation helpers (reference, not auto-run)
# ---------------------------------------------------------------------------
def example_hindi_banking_agent():
    """Example: explicitly set Hindi language for a banking agent."""
    return Agent(
        instructions="You are a banking agent. Reply in Hindi.",
        stt=sarvam.STT(
            language="hi-IN",     # fixed Hindi
            model="saaras:v3",
            mode="transcribe",
        ),
        llm=openai.LLM(model="gpt-4o"),
        tts=sarvam.TTS(
            target_language_code="hi-IN",
            model="bulbul:v3",
            speaker="anand",
        ),
    )


def example_gujarati_agent():
    """Example: Gujarati customer service agent."""
    return Agent(
        instructions="You are a customer service agent. Reply in Gujarati.",
        stt=sarvam.STT(language="gu-IN", model="saaras:v3", mode="transcribe"),
        llm=openai.LLM(model="gpt-4o"),
        tts=sarvam.TTS(target_language_code="gu-IN", model="bulbul:v3", speaker="varun"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sarvam AI -- Multilingual Voice Agent (LiveKit)")
    print(f"Active agent : {ACTIVE_AGENT} ({AGENTS[ACTIVE_AGENT].__doc__.strip().splitlines()[0]})")
    print()
    print("Commands:")
    print("  python 10_voice_agent_livekit.py dev      # start agent worker")
    print("  python 10_voice_agent_livekit.py console  # test with text input")
    print()
    print("Docs: https://docs.sarvam.ai/api-reference-docs/integration/build-voice-agent-with-live-kit")
    print()

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
