from datetime import datetime
from pathlib import Path
import uuid
import json
import os
import tempfile
import random
from typing import List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

# Demo counters so /transcribe returns believable answers in sequence
TRANSCRIBE_COUNTER = 0


# ── Helpers ────────────────────────────────────────────────────────────────

def save_session_artifact(session_id: str, payload: dict):
    out = RUNS_DIR / f"{session_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _extract_clean_answer(raw_text: str) -> str:
    if raw_text.startswith("[METRICS:"):
        parts = raw_text.split("\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return raw_text.strip()


def _extract_metrics_line(raw_text: str) -> str:
    first_line = raw_text.split("\n", 1)[0].strip()
    if first_line.startswith("[METRICS:"):
        return first_line
    return ""


def build_interview_plan(job_position: str, company: str, interview_type: str) -> dict:
    role_lower = job_position.lower()

    if "ai" in role_lower or "ml" in role_lower or "machine learning" in role_lower:
        skills = ["python", "ml fundamentals", "model evaluation", "system design", "communication"]
    elif "data" in role_lower:
        skills = ["python", "sql", "data pipelines", "debugging", "communication"]
    elif "cloud" in role_lower or "devops" in role_lower:
        skills = ["linux", "networking", "automation", "cloud concepts", "incident handling"]
    else:
        skills = ["fundamentals", "problem solving", "tools", "architecture", "communication"]

    objectives = [
        "Assess role fundamentals",
        "Assess problem solving and technical depth",
        "Assess hands-on tools and workflow ownership",
        "Assess architecture and tradeoff thinking",
        "Assess communication and real-world readiness",
    ]

    return {
        "workflow_goal": f"Assess candidate readiness for {job_position}" + (f" at {company}" if company else ""),
        "target_skills": skills,
        "question_plan": [
            {"round": i + 1, "objective": objectives[i], "skill": skills[i] if i < len(skills) else "communication"}
            for i in range(5)
        ],
        "scoring_rubric": {
            "technical_depth": "1-10",
            "clarity": "1-10",
            "relevance": "1-10",
            "confidence": "1-10",
            "behavioral_signal": "1-10",
        },
        "guardrails": [
            "Ground feedback in actual answer text",
            "Do not invent missing experience",
            "Use metrics only as supporting evidence",
            "Prefer concise and deterministic feedback",
        ],
    }


def get_mock_questions(job_position: str, company: str, interview_type: str) -> List[str]:
    company_ctx = f" at {company}" if company else ""

    if interview_type == "behavioral":
        return [
            f"Tell me about yourself and why you are interested in the {job_position}{company_ctx} role.",
            "Describe a time you handled a difficult technical problem under pressure. What was your approach?",
            "Tell me about a project where you took ownership beyond your assigned tasks.",
            "Describe a situation where something went wrong in a project. How did you debug and fix it?",
            "Why should we hire you for this role, and what strengths would you bring to the team?",
        ]

    if interview_type == "case_study":
        return [
            f"You are supporting a production platform for a {job_position}{company_ctx} team. What metrics would you track first?",
            "A workflow is producing inconsistent outputs. How would you isolate whether the issue is data, logic, or orchestration?",
            "You need to improve reliability of a multi-step system. What design changes would you prioritize?",
            "How would you evaluate whether an AI-assisted workflow is actually helping users?",
            "If leadership asked you to deploy this system at larger scale, what would you improve first and why?",
        ]

    return [
        f"Can you walk me through your background and how it aligns with a {job_position}{company_ctx} role?",
        "Describe an end-to-end workflow or project you worked on. What components did it include?",
        "How did you measure performance, accuracy, or success in that system?",
        "What technical stack, tools, and debugging methods did you use during development?",
        "What challenges did you face, and how did you improve reliability or add guardrails?",
    ]


def generate_evaluator_feedback(question: str, answer: str, metrics_line: str, round_num: int) -> str:
    answer_lower = answer.lower()
    word_count = len(answer.split())

    positives = []
    improvements = []

    if word_count >= 45:
        positives.append("your answer had enough detail and showed good structure")
    elif word_count >= 25:
        positives.append("your answer was reasonably clear and relevant")
    else:
        improvements.append("your answer was too brief and needed more concrete detail")

    keyword_groups = [
        ["python", "sql", "api", "fastapi", "docker", "aws", "linux", "automation"],
        ["metric", "accuracy", "precision", "recall", "latency", "evaluation"],
        ["debug", "root cause", "guardrail", "validation", "logging", "monitoring"],
        ["agent", "workflow", "planner", "evaluator", "verification"],
    ]

    matched = sum(1 for group in keyword_groups if any(k in answer_lower for k in group))
    if matched >= 2:
        positives.append("you referenced practical engineering concepts instead of staying generic")
    else:
        improvements.append("you could strengthen the answer with more specific tools, metrics, or implementation details")

    if any(x in answer_lower for x in ["i built", "i designed", "i worked on", "i implemented", "i improved"]):
        positives.append("you showed ownership clearly")
    else:
        improvements.append("you should frame your contribution more clearly")

    metrics_comment = ""
    if metrics_line:
        if "Eye Contact:" in metrics_line:
            metrics_comment = " Your behavioral signal looked stable enough to support the answer, but content should remain primary."

    if not improvements:
        sentiment = "POSITIVE:"
        body = (
            f"Your answer addressed the question directly, and {positives[0] if positives else 'it showed relevant understanding'}. "
            f"Your communication was fairly clear and easy to follow.{metrics_comment} "
            f"To make it even stronger, add one concrete metric or implementation detail."
        )
    else:
        sentiment = "NEEDS WORK:"
        body = (
            f"Your answer was relevant, but {improvements[0]}. "
            f"The communication was understandable, though parts felt general or underdeveloped.{metrics_comment} "
            f"To improve it, add a specific example, measurable impact, and clearer ownership."
        )

    return f"{sentiment}\n{body}"


def verify_feedback(question: str, answer: str, feedback: str, metrics_line: str) -> dict:
    answer_lower = answer.lower()
    feedback_lower = feedback.lower()
    issues = []

    if len(answer.split()) < 12 and "detailed" in feedback_lower:
        issues.append("feedback may overstate answer depth")

    if "aws" in feedback_lower and "aws" not in answer_lower:
        issues.append("feedback mentions unstated technology")

    if "excellent" in feedback_lower and len(answer.split()) < 20:
        issues.append("praise may be stronger than evidence")

    grounded = len(issues) == 0
    verdict = "PASS" if grounded else "REVISE"
    reliability = 9 if grounded else 6

    revised_feedback = feedback
    if not grounded:
        revised_feedback = (
            "NEEDS WORK:\n"
            "The answer is directionally relevant but needs more concrete detail and clearer evidence. "
            "Communication is understandable, though it should be more specific. "
            "To improve it, add tools used, measurable outcomes, and your exact contribution."
        )

    return {
        "verdict": verdict,
        "reliability_score": reliability,
        "grounded": grounded,
        "issues": issues,
        "revised_feedback": revised_feedback,
    }


def _extract_filler_words(text: str) -> List[str]:
    fillers = ["um", "uh", "like", "you know", "i mean", "actually", "basically"]
    text_lower = text.lower()
    found = []
    for f in fillers:
        if f in text_lower:
            found.append(f)
    return found


def build_report(conversation: List[Dict], metrics: List[Dict], verifier_data: List[Dict], job_position: str, company: str, interview_type: str) -> dict:
    candidate_answers = [m["content"] for m in conversation if m["role"] == "Candidate"]
    full_text = " ".join(candidate_answers)
    total_words = len(full_text.split())
    avg_words = total_words / max(len(candidate_answers), 1)

    filler_words = _extract_filler_words(full_text)

    avg_eye = None
    if metrics:
        vals = [m.get("eye_contact") for m in metrics if isinstance(m.get("eye_contact"), (int, float))]
        if vals:
            avg_eye = round(sum(vals) / len(vals))

    grounded_passes = 0
    if verifier_data:
        grounded_passes = sum(1 for v in verifier_data if v.get("verification", {}).get("verdict") == "PASS")
        groundedness_score = max(5, min(10, round((grounded_passes / len(verifier_data)) * 10)))
    else:
        groundedness_score = 8

    technical_score = 8 if avg_words >= 45 else 7 if avg_words >= 28 else 5
    communication_score = 8 if avg_words >= 40 else 7 if avg_words >= 22 else 5
    confidence_score = 8 if (avg_eye is not None and avg_eye >= 60) else 7 if avg_eye is not None else 7
    english_fluency_score = 7 if len(filler_words) <= 3 else 6
    overall_score = round((technical_score + communication_score + confidence_score + groundedness_score) / 4)

    recommendation = "Strong Hire" if overall_score >= 8 else "Hire" if overall_score >= 7 else "Maybe" if overall_score >= 6 else "No Hire"

    grammar_observations = [
        "Some responses would be stronger with shorter, cleaner sentences and fewer filler phrases.",
        "A few answers could benefit from more direct opening statements before going into details.",
    ]
    if not filler_words:
        grammar_observations = ["Speech was generally understandable, with only minor opportunities to tighten phrasing."]

    strengths = [
        "You presented an end-to-end workflow clearly and connected your contribution to the broader system.",
        "You referenced practical engineering topics such as metrics, evaluation, tools, and reliability, which made the answers feel implementation-focused.",
        "Your responses showed awareness of real-world delivery concerns like guardrails, repeatability, and structured evaluation.",
    ]

    improvements = [
        "Add more quantified impact such as latency reduction, evaluation pass rate, completion rate, or accuracy improvements.",
        "Make ownership even clearer by separating what you designed, what you implemented, and what you improved.",
        "Use one concrete example per answer so the response feels more grounded and interview-ready.",
    ]

    return {
        "overall_score": overall_score,
        "overall_score_reason": "The interview showed a solid understanding of workflow design, evaluation, and reliability. The strongest parts were the practical framing of the system and the ability to explain implementation choices. The score would improve further with more quantified impact and sharper examples.",
        "technical_score": technical_score,
        "technical_score_reason": "Your answers covered technical workflow components such as planning, evaluation, metrics, and guardrails. You demonstrated enough implementation awareness to discuss system behavior, though more role-specific depth and tooling detail would make the responses stronger.",
        "communication_score": communication_score,
        "communication_score_reason": "Your answers were understandable and mostly structured in a logical way. The communication improves when you start with a direct point, then support it with tools, metrics, and outcomes.",
        "confidence_score": confidence_score,
        "confidence_score_reason": f"Your delivery appeared reasonably stable during the session{f', with average eye contact around {avg_eye}%' if avg_eye is not None else ''}. Confidence would increase further with shorter, more decisive phrasing and stronger opening statements.",
        "english_fluency_score": english_fluency_score,
        "english_fluency_reason": "Your spoken responses were generally understandable and communicated the intended meaning. Fluency can improve further by reducing filler words, tightening sentence structure, and using more direct transitions between points.",
        "filler_words": filler_words,
        "grammar_observations": grammar_observations[:4],
        "fluency_tips": [
            "Start each answer with a one-line summary before giving details.",
            "Reduce filler phrases by pausing briefly instead of speaking while thinking.",
            "Practice using a repeatable structure such as context, action, metric, result.",
        ],
        "groundedness_score": groundedness_score,
        "groundedness_reason": f"The workflow evaluation remained fairly grounded in the candidate answers, with {grounded_passes} verified passes out of {len(verifier_data) if verifier_data else 0} checked feedback items. This indicates the feedback was mostly aligned with the actual content instead of inventing unstated experience.",
        "strengths": strengths,
        "improvements": improvements,
        "summary": "This mock interview demonstrated a solid explanation of an agentic workflow from planning through evaluation and reporting. The strongest dimension was the practical framing of the system, especially around metrics and reliability. The next improvement area is adding sharper ownership language and quantified business or engineering impact. Overall, the performance suggests good interview readiness with room to become more concise and evidence-driven.",
        "recommendation": recommendation,
        "recommendation_reason": "The interview showed relevant technical thinking, workable communication, and awareness of real-world engineering concerns. The profile becomes stronger when supported with more concrete metrics and specific implementation examples.",
    }


def get_demo_transcript() -> str:
    global TRANSCRIBE_COUNTER
    demo_answers = [
        "Sure. I worked on an AI interview evaluation workflow where the user selected a role and interview type, then the system ran a multi step interview loop with planning, questioning, evaluation, verification, and a final report.",
        "The end to end workflow started with a planner stage that defined target skills and question objectives. Then the interviewer asked contextual questions, the candidate response was captured, the evaluator generated structured feedback, and a guardrail verifier checked whether the feedback was grounded in the actual answer.",
        "We used metrics such as workflow completion rate, transcript success rate, grounded feedback pass rate, and average section scores across technical depth, communication, confidence, and behavioral signals.",
        "The stack included Python, FastAPI, WebSocket communication, JavaScript frontend, browser based face analysis, structured JSON logging, and a multi agent style workflow for planning, evaluation, and verification.",
        "One challenge was making the evaluator more deterministic. To improve repeatability, we refined the skill definitions, constrained the output format, and added a verification layer so that misleading or overconfident feedback could be flagged before the final report.",
    ]
    text = demo_answers[min(TRANSCRIBE_COUNTER, len(demo_answers) - 1)]
    TRANSCRIBE_COUNTER += 1
    return text


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Demo transcription endpoint.
    Returns believable mock answers so the full UI flow works without Whisper/API billing.
    """
    try:
        _ = await audio.read()
        return JSONResponse({"text": get_demo_transcript()})
    except Exception as e:
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)


@app.post("/generate-report")
async def generate_report(data: dict):
    try:
        conversation = data.get("conversation", [])
        metrics = data.get("metrics", [])
        verifier_data = data.get("verifier_results", [])
        job_position = data.get("job_position", "the position")
        company = data.get("company", "")
        interview_type = data.get("interview_type", "technical")

        report = build_report(conversation, metrics, verifier_data, job_position, company, interview_type)
        return JSONResponse(report)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── WebSocket Interview Workflow ──────────────────────────────────────────

@app.websocket("/ws/interview")
async def websocket_endpoint(
    websocket: WebSocket,
    pos: str = "AI Engineer",
    company: str = "",
    interview_type: str = "technical",
):
    await websocket.accept()

    session_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    transcript = []
    verifier_results = []

    try:
        plan = build_interview_plan(pos, company, interview_type)
        questions = get_mock_questions(pos, company, interview_type)

        label = f"{pos}" + (f" @ {company}" if company else "")
        type_label = interview_type.replace("_", " ").title()

        await websocket.send_text(f"SYSTEM_INFO:Starting {type_label} interview for {label}...")
        await websocket.send_text(
            "Planner:"
            + f" Workflow goal: {plan.get('workflow_goal', 'Assess candidate readiness')}. "
            + f"Target skills: {', '.join(plan.get('target_skills', []))}"
        )

        for idx, question in enumerate(questions, start=1):
            transcript.append({"role": "Interviewer", "content": question})
            await websocket.send_text(f"Interviewer:{question}")

            await websocket.send_text("SYSTEM_TURN:USER")
            raw_answer = await websocket.receive_text()

            if raw_answer.strip().upper() == "TERMINATE":
                break

            clean_answer = _extract_clean_answer(raw_answer)
            metrics_line = _extract_metrics_line(raw_answer)

            transcript.append({"role": "Candidate", "content": clean_answer})

            feedback = generate_evaluator_feedback(question, clean_answer, metrics_line, idx)
            transcript.append({"role": "Evaluator", "content": feedback})
            await websocket.send_text(f"Evaluator:{feedback}")

            verification = verify_feedback(question, clean_answer, feedback, metrics_line)
            verifier_results.append(
                {
                    "question": question,
                    "answer": clean_answer,
                    "feedback": feedback,
                    "verification": verification,
                }
            )

            guardrail_msg = (
                f"{verification.get('verdict', 'PASS')} | "
                f"reliability={verification.get('reliability_score', 8)}/10 | "
                f"grounded={verification.get('grounded', True)}"
            )
            if verification.get("issues"):
                guardrail_msg += f" | issues: {', '.join(verification['issues'])}"

            await websocket.send_text(f"Guardrail:{guardrail_msg}")

        save_session_artifact(
            session_id,
            {
                "session_id": session_id,
                "job_position": pos,
                "company": company,
                "interview_type": interview_type,
                "plan": plan,
                "transcript": transcript,
                "verifier_results": verifier_results,
                "mode": "mock_demo",
                "stop_reason": "Interview completed",
            },
        )

        await websocket.send_text(f"SYSTEM_INFO:Session saved as {session_id}.json")
        await websocket.send_text("SYSTEM_END:Interview completed")

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_text(f"SYSTEM_ERROR:{str(e)}")
        except Exception:
            pass